// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package agentanalytics

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	bqstorage "cloud.google.com/go/bigquery/storage/apiv1"
	storagepb "cloud.google.com/go/bigquery/storage/apiv1/storagepb"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

// BatchProcessor handles asynchronous batching and writing of events to BigQuery.
type BatchProcessor struct {
	writeClient  *bqstorage.BigQueryWriteClient
	streamWriter *streamWriter
	streamName   string
	config       Config
	queue        chan map[string]any

	appendMu sync.RWMutex
	flushMu  sync.Mutex
	// deferredBatch holds at most one batch whose BigQuery write was cut
	// short by flushRunEnd's ShutdownTimeout; guarded by flushMu and
	// drained before the queue by flush, flushRunEnd, and Close.
	deferredBatch []map[string]any
	arrowSchema   *arrow.Schema
	allocator     memory.Allocator

	// baseCtx carries the values of the context that NewBatchProcessor was
	// given, without its cancellation. Every append, and every upload made
	// while Close drains the queue, runs under a context derived from it, and
	// is bounded by its own timeout.
	baseCtx context.Context
	// uploadCtx bounds the uploads of background and run-end flushes; Close
	// cancels it before waiting on flushMu so an in-flight upload falls back
	// instead of blocking shutdown.
	uploadCtx     context.Context
	cancelUploads context.CancelFunc

	stopCh    chan struct{}
	closeOnce sync.Once
	done      chan struct{}
	started   atomic.Bool // Track if Start() has been called.
	closed    atomic.Bool

	// uploadTimeout overrides Config.GCSUploadTimeout in tests.
	uploadTimeout time.Duration
	// uploadCooldownUntil holds a UnixNano deadline while uploads are paused
	// after a batch-level GCSUploadTimeout expiration (or a fresh caller
	// deadline expiring on the first upload batch of a run-end flush).
	uploadCooldownUntil atomic.Int64

	// pendingUploadBytes tracks total bytes currently held in memory by
	// queued and in-flight pending uploads; maxPendingUploadBytes overrides
	// Config.GCSMaxPendingUploadBytes in tests.
	pendingUploadBytes    atomic.Int64
	maxPendingUploadBytes int64
	uploadBudgetWarned    atomic.Bool
}

// NewBatchProcessor creates a new BatchProcessor instance. If config.GCSOffloader
// is set, the payloads queued for upload with the rows are uploaded through it
// before the rows are written.
//
// Only the values carried by ctx are used: its cancellation and deadline are
// ignored, so canceling ctx neither stops nor drains the processor. Only
// Close stops it: once any background write that is already in flight has
// finished, it uploads the payloads of the queued rows, for at most half of
// Config.ShutdownTimeout in all, and writes the rows within
// Config.ShutdownTimeout, which the uploads do not use up.
func NewBatchProcessor(ctx context.Context, writeClient *bqstorage.BigQueryWriteClient, streamName string, config Config) (*BatchProcessor, error) {
	config = applyDefaults(config)
	denied, err := validateConfig(config)
	if err != nil {
		return nil, err
	}
	arrowSchema := projectArrowSchema(denied)
	schemaBytes, err := serializeArrowSchema(arrowSchema)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize arrow schema: %w", err)
	}
	baseCtx := context.WithoutCancel(ctx)
	uploadCtx, cancelUploads := context.WithCancel(baseCtx)

	return &BatchProcessor{
		writeClient:   writeClient,
		streamWriter:  newStreamWriter(ctx, writeClient, streamName, schemaBytes, config),
		streamName:    streamName,
		config:        config,
		queue:         make(chan map[string]any, config.QueueMaxSize),
		arrowSchema:   arrowSchema,
		allocator:     memory.NewGoAllocator(), // Use Go memory allocator
		baseCtx:       baseCtx,
		uploadCtx:     uploadCtx,
		cancelUploads: cancelUploads,
		stopCh:        make(chan struct{}),
		done:          make(chan struct{}),
	}, nil
}

// Start launches the background goroutine that flushes the queue every
// Config.BatchFlushIntv. The goroutine runs until Close is called; later calls
// to Start have no effect.
func (b *BatchProcessor) Start() {
	if b.started.CompareAndSwap(false, true) {
		go func() {
			defer close(b.done)
			ticker := time.NewTicker(b.config.BatchFlushIntv)
			defer ticker.Stop()

			for {
				select {
				case <-b.stopCh:
					return
				case <-ticker.C:
					b.flush()
				}
			}
		}()
	}
}

func (b *BatchProcessor) shutdownTimeout() time.Duration {
	if b.config.ShutdownTimeout > 0 {
		return b.config.ShutdownTimeout
	}
	return 10 * time.Second
}

// tryReserveUploadBytes reserves n bytes in pendingUploadBytes if within the
// pending upload byte cap.
func (b *BatchProcessor) tryReserveUploadBytes(n int) bool {
	if n <= 0 {
		return true
	}
	limit := cmp.Or(b.maxPendingUploadBytes, b.config.GCSMaxPendingUploadBytes, int64(defaultMaxPendingUploadBytes))
	for {
		cur := b.pendingUploadBytes.Load()
		if cur+int64(n) > limit {
			if b.uploadBudgetWarned.CompareAndSwap(false, true) {
				b.config.logf("agentanalytics: pending GCS upload byte budget (%d bytes) exceeded; keeping payload inline", limit)
			}
			return false
		}
		if b.pendingUploadBytes.CompareAndSwap(cur, cur+int64(n)) {
			b.uploadBudgetWarned.Store(false)
			return true
		}
	}
}

func (b *BatchProcessor) releaseUploadBytes(n int64) {
	if n <= 0 {
		return
	}
	if b.pendingUploadBytes.Add(-n) < 0 {
		b.pendingUploadBytes.Store(0)
	}
	b.uploadBudgetWarned.Store(false)
}

func (b *BatchProcessor) releaseRowUploads(row map[string]any) {
	b.clearDeferredRowUploads(row)
	delete(row, summaryBuilderKey)
	pending, ok := row[rawPartsKey].([]pendingUpload)
	if !ok {
		return
	}
	delete(row, rawPartsKey)
	var total int64
	for _, u := range pending {
		total += u.reserved
	}
	b.releaseUploadBytes(total)
}

func (b *BatchProcessor) hasPendingWorkLocked() bool {
	return len(b.deferredBatch) > 0 || len(b.queue) > 0
}

func (b *BatchProcessor) nextBatchLocked() []map[string]any {
	if len(b.deferredBatch) > 0 {
		batch := b.deferredBatch
		b.deferredBatch = nil
		return batch
	}
	return b.drainQueue()
}

// Append adds a row to the processor queue asynchronously. If the queue is
// full, the row is dropped and a warning is logged.
func (b *BatchProcessor) Append(row map[string]any) {
	dropped, shouldFlush := b.enqueue(row)
	// Log and flush only after enqueue has released appendMu, so that a
	// panicking Logger cannot leave the lock held and block Close.
	if dropped && b.config.Logger != nil {
		b.config.Logger.Println("WARNING: BigQuery event queue is full, dropping event.")
	}
	if shouldFlush {
		go b.flush()
	}
}

// enqueue offers row to the queue while holding appendMu for reading, so that
// Close cannot mark the processor closed mid-send. It reports whether the row
// was dropped because the queue was full and whether the queue has just
// reached BatchSize rows. Rows offered after Close are discarded silently.
// A row that is dropped or discarded has its pending uploads released.
func (b *BatchProcessor) enqueue(row map[string]any) (dropped, shouldFlush bool) {
	b.appendMu.RLock()
	defer b.appendMu.RUnlock()
	if b.closed.Load() {
		b.releaseRowUploads(row)
		return false, false
	}
	select {
	case b.queue <- row:
		return false, len(b.queue) == b.config.BatchSize
	default:
		b.releaseRowUploads(row)
		return true, false
	}
}

// drainQueue acquires up to BatchSize elements from the queue.
func (b *BatchProcessor) drainQueue() []map[string]any {
	queueLen := len(b.queue)
	if queueLen == 0 {
		return nil
	}

	size := queueLen
	if size > b.config.BatchSize {
		size = b.config.BatchSize
	}

	batch := make([]map[string]any, 0, size)
Loop:
	for i := 0; i < size; i++ {
		select {
		case row := <-b.queue:
			batch = append(batch, row)
		default:
			// Channel is empty earlier than size expected
			break Loop
		}
	}
	return batch
}

// writeAndLogBatch writes the batch and logs any errors.
func (b *BatchProcessor) writeAndLogBatch(uploadCtx, writeCtx context.Context, pauseOnCallerDeadline bool, batch []map[string]any) {
	if len(batch) == 0 {
		return
	}
	if err := b.writeBatchWithContexts(uploadCtx, writeCtx, pauseOnCallerDeadline, batch); err != nil && b.config.Logger != nil {
		b.config.Logger.Printf("ERROR: Failed to write batch to BigQuery: %v", err)
	}
}

func (b *BatchProcessor) writeBatchBounded(batch []map[string]any) {
	if len(batch) == 0 {
		return
	}
	b.writeAndLogBatch(b.uploadCtx, b.baseCtx, false, batch)
}

func batchHasPendingUploads(batch []map[string]any) bool {
	for _, row := range batch {
		if pending, ok := row[rawPartsKey].([]pendingUpload); ok && len(pending) > 0 {
			return true
		}
	}
	return false
}

// flush drains any deferredBatch and full batches from the queue and appends
// them. Uses TryLock to avoid blocking.
func (b *BatchProcessor) flush() {
	// Ensure single flush execution at any moment
	if !b.flushMu.TryLock() {
		return
	}
	defer b.flushMu.Unlock()
	if b.closed.Load() {
		return
	}

	batch := b.nextBatchLocked()
	b.writeBatchBounded(batch)

	// If the queue has enough items for another batch, drain and write synchronously
	// unless shutdown has started (in which case Close drains under ShutdownTimeout).
	for !b.closed.Load() && len(b.queue) >= b.config.BatchSize {
		batch = b.nextBatchLocked()
		b.writeBatchBounded(batch)
	}
}

// flushRunEnd drains and writes queued events synchronously under a single
// ShutdownTimeout budget, skipping if a background flush is already running.
// The uploads of each batch get at most half of what is left of the budget, so
// that a Cloud Storage problem leaves time for the BigQuery write. If the first
// batch with uploads times out while the run-end budget is still fresh, later
// batches skip Cloud Storage for a while instead of each waiting for it (see
// resolveUploads). If the run-end budget expires during a batch's BigQuery
// write, the batch (along with any uploads that were cut short by the depleted
// run-end deadline) is retained in deferredBatch for the next flush or Close;
// as with other context expirations after a write is sent, this provides
// at-least-once delivery if the server had already committed the batch.
func (b *BatchProcessor) flushRunEnd() {
	if !b.flushMu.TryLock() {
		return
	}
	defer b.flushMu.Unlock()
	if b.closed.Load() {
		return
	}
	shutdownTimeout := b.shutdownTimeout()
	flushCtx, cancel := context.WithTimeout(b.baseCtx, shutdownTimeout)
	defer cancel()
	// flushCtx always has a deadline: it is made by WithTimeout.
	flushDeadline, _ := flushCtx.Deadline()

	firstUploadBatch := true
	for !b.closed.Load() && b.hasPendingWorkLocked() && flushCtx.Err() == nil {
		retry := len(b.deferredBatch) > 0
		batch := b.nextBatchLocked()
		if len(batch) == 0 {
			break
		}
		hasUploads := batchHasPendingUploads(batch)
		remaining := time.Until(flushDeadline)
		pauseOnCallerDeadline := firstUploadBatch && hasUploads && remaining >= shutdownTimeout*3/4
		uploadCtx, cancelUploads := context.WithDeadline(b.uploadCtx, time.Now().Add(remaining/2))
		b.resolveUploads(uploadCtx, pauseOnCallerDeadline, batch)
		cancelUploads()
		if hasUploads {
			firstUploadBatch = false
		}

		if err := b.appendBatch(flushCtx, batch); err != nil {
			if flushCtx.Err() != nil && !retry {
				b.requeueDeferredUploads(batch)
				b.deferredBatch = batch
				return
			}
			for _, row := range batch {
				b.releaseRowUploads(row)
			}
			b.config.logf("ERROR: Failed to write batch to BigQuery: %v", err)
		}
		b.clearDeferredUploads(batch)
	}
}

// flushAndWaitForInflightBatch waits for any in-flight background flush to
// finish persisting its already-drained batch, then drains and writes all
// remaining queued batches. Cloud Storage and BigQuery are budgeted apart, so
// that a Cloud Storage outage does not cost rows that BigQuery would take:
//
//   - The BigQuery writes of all the batches share one ShutdownTimeout, which
//     is used up only while a write runs. Batches that are still unwritten when
//     it expires are dropped and logged.
//   - The uploads of all the batches share half a ShutdownTimeout. When it is
//     used up, or once the uploads of one batch have timed out and paused the
//     uploads (see resolveUploads), the remaining parts fall back at once.
//
// Both budgets start once the wait is over, so the time spent waiting for the
// in-flight flush does not reduce them. The drain can therefore take up to one
// and a half ShutdownTimeout. Unlike the uploads of the in-flight flush, which
// Close cancels to end that wait, its uploads run under contexts that Close
// leaves alone: the queued rows have not had their chance to upload yet.
func (b *BatchProcessor) flushAndWaitForInflightBatch() {
	b.flushMu.Lock()
	defer b.flushMu.Unlock()

	writeLeft := b.shutdownTimeout()
	uploadLeft := writeLeft / 2
	for b.hasPendingWorkLocked() {
		batch := b.nextBatchLocked()
		if len(batch) == 0 {
			break
		}
		if writeLeft <= 0 {
			for _, row := range batch {
				b.releaseRowUploads(row)
			}
			b.config.logf("WARNING: Failed to write batch of %d events: %v", len(batch), context.DeadlineExceeded)
			continue
		}

		// A context whose time is up ends the uploads at once.
		start := time.Now()
		uploadCtx, cancelUploads := context.WithTimeout(b.baseCtx, uploadLeft)
		b.resolveUploads(uploadCtx, false, batch)
		cancelUploads()
		uploadLeft -= time.Since(start)

		start = time.Now()
		writeCtx, cancelWrite := context.WithTimeout(b.baseCtx, writeLeft)
		if err := b.appendBatch(writeCtx, batch); err != nil {
			b.config.logf("ERROR: Failed to write batch to BigQuery: %v", err)
		}
		cancelWrite()
		writeLeft -= time.Since(start)
		b.clearDeferredUploads(batch)
	}
}

// writeBatch allocates a new Apache Arrow RecordBuilder from the batch and commits the writes to BigQuery.
// Payloads queued for offloading are uploaded to Cloud Storage first, so that
// the rows are final by the time they are serialized.
func (b *BatchProcessor) writeBatch(ctx context.Context, batch []map[string]any) error {
	return b.writeBatchWithContexts(ctx, ctx, false, batch)
}

// writeBatchWithContexts uploads the payloads of batch under uploadCtx and
// then appends the batch under writeCtx.
func (b *BatchProcessor) writeBatchWithContexts(uploadCtx, writeCtx context.Context, pauseOnCallerDeadline bool, batch []map[string]any) error {
	b.resolveUploads(uploadCtx, pauseOnCallerDeadline, batch)
	err := b.appendBatch(writeCtx, batch)
	b.clearDeferredUploads(batch)
	return err
}

// appendBatch appends batch, whose uploads are resolved, to BigQuery under
// writeCtx. A writeCtx without a deadline gets one of ShutdownTimeout.
func (b *BatchProcessor) appendBatch(writeCtx context.Context, batch []map[string]any) error {
	if _, ok := writeCtx.Deadline(); !ok {
		var cancel context.CancelFunc
		writeCtx, cancel = context.WithTimeout(writeCtx, b.shutdownTimeout())
		defer cancel()
	}
	if err := writeCtx.Err(); err != nil {
		return err
	}

	builder := array.NewRecordBuilder(b.allocator, b.arrowSchema)
	defer builder.Release()

	// Need to append rows into the dynamic vectors using Arrow API...
	// For simplicity, we create a JSON block representing the array of maps,
	// but BigQuery requires the specific Arrow vector builders mapping to exact fields.

	// Because mapping arbitrary json tree maps cleanly to Arrow types recursively in Go
	// can be intensely complex inside this writeBatch code, a much simpler equivalent approach
	// is converting the batch directly using apache/arrow_go's JSON reader, or falling back
	// on mapping individually. Let's build vectors for each field manually:

	for i, f := range b.arrowSchema.Fields() {
		fieldVector := builder.Field(i)
		for _, row := range batch {
			val, ok := row[f.Name]
			if !ok || val == nil {
				if f.Name == "content_parts" {
					// Null lists are not allowed for repeated fields in BigQuery.
					listBuilder := fieldVector.(*array.ListBuilder)
					listBuilder.Append(true)
				} else {
					fieldVector.AppendNull()
				}
				continue
			}

			// Based on expected fields defined in ArrowSchema
			switch f.Name {
			case timestampField:
				if v, ok := val.(time.Time); ok {
					fieldVector.(*array.TimestampBuilder).Append(arrow.Timestamp(v.UnixMicro()))
				} else {
					fieldVector.AppendNull()
				}
			case eventIDField, "event_type", "agent", "session_id", "invocation_id", "user_id", "trace_id", "span_id", "parent_span_id", "status", "error_message":
				if v, ok := val.(string); ok {
					fieldVector.(*array.StringBuilder).Append(v)
				} else {
					fieldVector.AppendNull()
				}
			case "content", "attributes", "latency_ms":
				if v, ok := val.(string); ok {
					fieldVector.(*array.StringBuilder).Append(v)
				} else {
					fieldVector.AppendNull()
				}
			case "is_truncated":
				if v, ok := val.(bool); ok {
					fieldVector.(*array.BooleanBuilder).Append(v)
				} else {
					fieldVector.AppendNull()
				}
			case "content_parts":
				listBuilder := fieldVector.(*array.ListBuilder)
				if v, ok := val.([]map[string]any); ok {
					listBuilder.Append(true)
					structBuilder := listBuilder.ValueBuilder().(*array.StructBuilder)
					mimeTypeBuilder := structBuilder.FieldBuilder(0).(*array.StringBuilder)
					uriBuilder := structBuilder.FieldBuilder(1).(*array.StringBuilder)
					objectRefBuilder := structBuilder.FieldBuilder(2).(*array.StructBuilder)
					textContentBuilder := structBuilder.FieldBuilder(3).(*array.StringBuilder)
					partIndexBuilder := structBuilder.FieldBuilder(4).(*array.Int64Builder)
					partAttributesBuilder := structBuilder.FieldBuilder(5).(*array.StringBuilder)
					storageModeBuilder := structBuilder.FieldBuilder(6).(*array.StringBuilder)

					for _, part := range v {
						structBuilder.Append(true)

						if mimeType, ok := part["mime_type"].(string); ok {
							mimeTypeBuilder.Append(mimeType)
						} else {
							mimeTypeBuilder.AppendNull()
						}

						if uri, ok := part["uri"].(string); ok {
							uriBuilder.Append(uri)
						} else {
							uriBuilder.AppendNull()
						}

						if objectRef, ok := part["object_ref"].(map[string]any); ok {
							objectRefBuilder.Append(true)
							uriB := objectRefBuilder.FieldBuilder(0).(*array.StringBuilder)
							versionB := objectRefBuilder.FieldBuilder(1).(*array.StringBuilder)
							authorizerB := objectRefBuilder.FieldBuilder(2).(*array.StringBuilder)
							detailsB := objectRefBuilder.FieldBuilder(3).(*array.StringBuilder)

							if u, ok := objectRef["uri"].(string); ok {
								uriB.Append(u)
							} else {
								uriB.AppendNull()
							}
							if ver, ok := objectRef["version"].(string); ok {
								versionB.Append(ver)
							} else {
								versionB.AppendNull()
							}
							if auth, ok := objectRef["authorizer"].(string); ok {
								authorizerB.Append(auth)
							} else {
								authorizerB.AppendNull()
							}
							if det, ok := objectRef["details"].(string); ok {
								detailsB.Append(det)
							} else {
								detailsB.AppendNull()
							}
						} else {
							objectRefBuilder.AppendNull()
						}

						if text, ok := part["text"].(string); ok {
							textContentBuilder.Append(text)
						} else {
							textContentBuilder.AppendNull()
						}

						if partIdx, ok := part["part_index"].(int64); ok {
							partIndexBuilder.Append(partIdx)
						} else if partIdxFloat, ok := part["part_index"].(float64); ok {
							partIndexBuilder.Append(int64(partIdxFloat))
						} else if partIdxInt, ok := part["part_index"].(int); ok {
							partIndexBuilder.Append(int64(partIdxInt))
						} else {
							partIndexBuilder.AppendNull()
						}

						if partAttributes, ok := part["part_attributes"].(string); ok {
							partAttributesBuilder.Append(partAttributes)
						} else {
							partAttributesBuilder.AppendNull()
						}

						if storageMode, ok := part["storage_mode"].(string); ok {
							storageModeBuilder.Append(storageMode)
						} else {
							storageModeBuilder.AppendNull()
						}
					}
				} else {
					// Null lists are not allowed for repeated fields in BigQuery.
					listBuilder := fieldVector.(*array.ListBuilder)
					listBuilder.Append(true)
				}
			default:
				fieldVector.AppendNull()
			}
		}
	}

	record := builder.NewRecordBatch()
	defer record.Release()

	res, err := b.streamWriter.append(writeCtx, record)
	if err != nil {
		return fmt.Errorf("streamWriter append error: %w", err)
	}

	if res.GetError() != nil || len(res.GetRowErrors()) > 0 {
		if b.config.Logger != nil {
			b.config.Logger.Printf("BigQuery response: %s", res.String())
		}
	}

	if res.GetError() != nil {
		return fmt.Errorf("batch append error from BigQuery: %v", res.GetError().GetMessage())
	}

	if len(res.GetRowErrors()) > 0 {
		return fmt.Errorf("batch append error from BigQuery: %d row errors found", len(res.GetRowErrors()))
	}

	return nil
}

// streamWriter is a wrapper that maintains a persistent BigQuery write stream.
type streamWriter struct {
	writeClient  *bqstorage.BigQueryWriteClient
	streamName   string
	schemaBytes  []byte
	config       Config
	baseCtx      context.Context
	streamCtx    context.Context
	streamCancel context.CancelFunc

	stream interface {
		Send(*storagepb.AppendRowsRequest) error
		Recv() (*storagepb.AppendRowsResponse, error)
		CloseSend() error
	}
	mu sync.Mutex
}

func newStreamWriter(ctx context.Context, client *bqstorage.BigQueryWriteClient, streamName string, schemaBytes []byte, config Config) *streamWriter {
	if ctx == nil {
		ctx = context.Background()
	}
	return &streamWriter{
		writeClient: client,
		streamName:  streamName,
		schemaBytes: schemaBytes,
		config:      config,
		baseCtx:     context.WithoutCancel(ctx),
	}
}

func (s *streamWriter) resetStream() {
	if s.streamCancel != nil {
		s.streamCancel()
		s.streamCancel = nil
	}
	s.streamCtx = nil
	s.stream = nil
}

func (s *streamWriter) close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.resetStream()
}

func (s *streamWriter) append(ctx context.Context, recordBatch arrow.RecordBatch) (*storagepb.AppendRowsResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var lastErr error
	delay := s.config.RetryConfig.InitialDelay
	maxRetries := s.config.RetryConfig.MaxRetries

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
				delay = time.Duration(float64(delay) * s.config.RetryConfig.Multiplier)
				if delay > s.config.RetryConfig.MaxDelay {
					delay = s.config.RetryConfig.MaxDelay
				}
			}
		}

		if err := ctx.Err(); err != nil {
			return nil, err
		}

		if s.stream != nil && s.streamCtx != nil && s.streamCtx.Err() != nil {
			s.resetStream()
		}

		if s.stream == nil {
			baseCtx := s.baseCtx
			if baseCtx == nil || baseCtx.Err() != nil {
				baseCtx = context.WithoutCancel(ctx)
			}
			streamCtx, streamCancel := context.WithCancel(baseCtx)
			stopOpen := context.AfterFunc(ctx, streamCancel)
			stream, err := s.writeClient.AppendRows(streamCtx)
			if !stopOpen() {
				streamCancel()
				if err == nil {
					err = ctx.Err()
				}
				lastErr = fmt.Errorf("failed to open append rows stream: %w", err)
				continue
			}
			if err != nil {
				streamCancel()
				lastErr = fmt.Errorf("failed to open append rows stream: %w", err)
				continue
			}
			s.stream = stream
			s.streamCtx = streamCtx
			s.streamCancel = streamCancel
		}

		batchBytes := serializeRecord(recordBatch, s.config.Logger)
		if batchBytes == nil {
			lastErr = fmt.Errorf("failed to serialize record batch")
			continue
		}

		req := &storagepb.AppendRowsRequest{
			WriteStream: s.streamName,
			Rows: &storagepb.AppendRowsRequest_ArrowRows{
				ArrowRows: &storagepb.AppendRowsRequest_ArrowData{
					WriterSchema: &storagepb.ArrowSchema{
						SerializedSchema: s.schemaBytes,
					},
					Rows: &storagepb.ArrowRecordBatch{
						SerializedRecordBatch: batchBytes,
						RowCount:              recordBatch.NumRows(),
					},
				},
			},
		}

		var stopCall func() bool
		if s.streamCancel != nil {
			stopCall = context.AfterFunc(ctx, s.streamCancel)
		}

		if err := s.stream.Send(req); err != nil {
			// An error not generated by the client is reported as io.EOF here, with the
			// real status left for Recv; drain until it surfaces, releasing the stream.
			// Wrapped errors match too: draining a client-aborted stream is harmless,
			// and an exact match would let the wrapper escape in place of the status.
			if errors.Is(err, io.EOF) {
				for {
					_, rerr := s.stream.Recv()
					if rerr == nil {
						continue
					}
					if !errors.Is(rerr, io.EOF) {
						err = rerr
					}
					break
				}
			}
			if stopCall != nil {
				stopCall()
			}
			s.resetStream() // Reset stream on send error
			lastErr = fmt.Errorf("failed to send row batch: %w", err)
			continue
		}

		res, err := s.stream.Recv()
		if stopCall != nil {
			stopCall()
		}
		if err != nil {
			s.resetStream() // Reset stream on receive error
			lastErr = fmt.Errorf("failed to receive response for row batch: %w", err)
			continue
		}

		return res, nil
	}

	return nil, fmt.Errorf("max retries exceeded: %w", lastErr)
}

func serializeRecord(record arrow.RecordBatch, logger Logger) []byte {
	// GetRecordBatchPayload serializes only the record batch message, excluding the schema.
	payload, err := ipc.GetRecordBatchPayload(record)
	if err != nil {
		if logger != nil {
			logger.Printf("ERROR: Failed to get record batch payload: %v", err)
		}
		return nil
	}
	defer payload.Release()

	var buf bytes.Buffer
	if _, err := payload.WritePayload(&buf); err != nil {
		if logger != nil {
			logger.Printf("ERROR: Failed to write payload: %v", err)
		}
		return nil
	}
	return buf.Bytes()
}

func (b *BatchProcessor) stopAcceptingRows() {
	b.appendMu.Lock()
	b.closed.Store(true)
	close(b.stopCh)
	b.appendMu.Unlock()
	b.cancelUploads()
}

// Close gracefully shuts down the processing loop, flushes, and stops. It
// stops accepting rows, waits for the processing loop and for any background
// flush that is already writing a batch, and then uploads the payloads of the
// rows still queued and writes the rows. The uploads share half a
// ShutdownTimeout and the writes share one, both starting after that wait, so
// a Cloud Storage outage does not use up the time that the writes need. Batches
// that are unwritten when the write budget expires are dropped and logged.
// Close can therefore block for up to about two and a half times
// ShutdownTimeout: once for a write that was already in flight, and then for
// the uploads and the writes of the drain.
func (b *BatchProcessor) Close() {
	b.closeOnce.Do(func() {
		b.stopAcceptingRows()
		if b.started.Load() {
			<-b.done
		}
		b.flushAndWaitForInflightBatch()
		b.streamWriter.close()
	})
}
