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
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/wrapperspb"
)

func createCommittedWriteStream(ctx context.Context, writeClient *bqstorage.BigQueryWriteClient, config Config) (string, error) {
	if writeClient == nil {
		return "", fmt.Errorf("writeClient is nil")
	}
	parent := fmt.Sprintf("projects/%s/datasets/%s/tables/%s", config.ProjectID, config.DatasetID, config.TableName)
	ws, err := writeClient.CreateWriteStream(ctx, &storagepb.CreateWriteStreamRequest{
		Parent: parent,
		WriteStream: &storagepb.WriteStream{
			Type: storagepb.WriteStream_COMMITTED,
		},
	})
	if err != nil {
		return "", fmt.Errorf("failed to create committed write stream: %w", err)
	}
	return ws.GetName(), nil
}

// dropCounters stores atomic event-drop and formatter-failure counters decoupled
// from BatchProcessor so the global pluginRegistry does not retain unclosed
// processors or their underlying gRPC resources.
type dropCounters struct {
	queueFullDrops        atomic.Int64
	appendFailureDrops    atomic.Int64
	formatterFailureDrops atomic.Int64
	shutdownDrops         atomic.Int64
}

func (c *dropCounters) DropStats() DropStats {
	if c == nil {
		return DropStats{}
	}
	q := c.queueFullDrops.Load()
	a := c.appendFailureDrops.Load()
	f := c.formatterFailureDrops.Load()
	s := c.shutdownDrops.Load()
	return DropStats{
		QueueFullDrops:        q,
		AppendFailureDrops:    a,
		FormatterFailureDrops: f,
		ShutdownDrops:         s,
		TotalDrops:            q + a + s,
	}
}

// BatchProcessor handles asynchronous batching and writing of events to BigQuery.
type BatchProcessor struct {
	writeClient            *bqstorage.BigQueryWriteClient
	streamWriter           *streamWriter
	streamName             string
	createdCommittedStream bool
	config                 Config
	queue                  chan map[string]any

	counters *dropCounters

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
// When ExactlyOnceDelivery is enabled, ctx bounds the initial CreateWriteStream
// call during construction. Background writes, stream rotations, and Close use
// only the values carried by ctx: its cancellation and deadline are ignored
// after construction returns, so canceling ctx neither stops nor drains the
// processor. Only Close stops it: once any background write that is already in
// flight has finished, it uploads the payloads of the queued rows, for at most
// half of Config.ShutdownTimeout in all, and writes the rows within
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

	if config.ExactlyOnceDelivery && streamName != "" {
		cancelUploads()
		return nil, fmt.Errorf("agentanalytics: ExactlyOnceDelivery requires an empty streamName (got %q; offsets cannot be used with a _default or caller-supplied stream)", streamName)
	}

	createdCommittedStream := false
	if config.ExactlyOnceDelivery && writeClient != nil {
		createCtx, cancelCreate := context.WithTimeout(ctx, defaultRotateTimeout)
		streamName, err = createCommittedWriteStream(createCtx, writeClient, config)
		cancelCreate()
		if err != nil {
			cancelUploads()
			return nil, err
		}
		createdCommittedStream = true
	}

	return &BatchProcessor{
		writeClient:            writeClient,
		streamWriter:           newStreamWriter(ctx, writeClient, streamName, schemaBytes, config),
		streamName:             streamName,
		createdCommittedStream: createdCommittedStream,
		config:                 config,
		counters:               &dropCounters{},
		queue:                  make(chan map[string]any, config.QueueMaxSize),
		arrowSchema:            arrowSchema,
		allocator:              memory.NewGoAllocator(), // Use Go memory allocator
		baseCtx:                baseCtx,
		uploadCtx:              uploadCtx,
		cancelUploads:          cancelUploads,
		stopCh:                 make(chan struct{}),
		done:                   make(chan struct{}),
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
// A row that is dropped or discarded is counted in DropStats and has its
// pending uploads released.
func (b *BatchProcessor) enqueue(row map[string]any) (dropped, shouldFlush bool) {
	b.appendMu.RLock()
	defer b.appendMu.RUnlock()
	if b.closed.Load() {
		b.recordShutdownDrop(1)
		b.releaseRowUploads(row)
		return false, false
	}
	select {
	case b.queue <- row:
		return false, len(b.queue) == b.config.BatchSize
	default:
		b.recordQueueFullDrop()
		b.releaseRowUploads(row)
		return true, false
	}
}

func (b *BatchProcessor) recordQueueFullDrop() {
	if b != nil && b.counters != nil {
		b.counters.queueFullDrops.Add(1)
	}
}

func (b *BatchProcessor) recordShutdownDrop(n int64) {
	if b != nil && b.counters != nil {
		b.counters.shutdownDrops.Add(n)
	}
}

func (b *BatchProcessor) recordFormatterFailureDrop() {
	if b != nil && b.counters != nil {
		b.counters.formatterFailureDrops.Add(1)
	}
}

func (b *BatchProcessor) recordAppendFailureDrop(n int64) {
	if b != nil && b.counters != nil {
		b.counters.appendFailureDrops.Add(n)
	}
}

// DropStats returns cumulative event-drop and formatter-failure counts.
// TotalDrops sums locally discarded events and writes whose success could not
// be confirmed (QueueFullDrops + AppendFailureDrops + ShutdownDrops); it can
// include rows committed before an acknowledgment was lost, so it is an upper
// bound. FormatterFailureDrops separately counts payload replacement or
// omission, which does not itself discard a row.
func (b *BatchProcessor) DropStats() DropStats {
	if b == nil || b.counters == nil {
		return DropStats{}
	}
	return b.counters.DropStats()
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
		if b.streamWriter.isRotationInCooldown() {
			b.recordAppendFailureDrop(int64(len(batch)))
			for _, row := range batch {
				b.releaseRowUploads(row)
			}
			b.config.logf("ERROR: Failed to write batch to BigQuery: committed write stream rotation in cooldown")
			continue
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
func (b *BatchProcessor) flushAndWaitForInflightBatch() time.Duration {
	b.flushMu.Lock()
	defer b.flushMu.Unlock()

	writeLeft := b.shutdownTimeout()
	uploadLeft := writeLeft / 2
	for b.hasPendingWorkLocked() {
		batch := b.nextBatchLocked()
		if len(batch) == 0 {
			break
		}
		if writeLeft <= 0 || b.streamWriter.isRotationInCooldown() {
			if writeLeft <= 0 {
				b.recordShutdownDrop(int64(len(batch)))
				b.config.logf("WARNING: Failed to write batch of %d events: %v", len(batch), context.DeadlineExceeded)
			} else {
				b.recordAppendFailureDrop(int64(len(batch)))
				b.config.logf("WARNING: Failed to write batch of %d events: committed write stream rotation in cooldown", len(batch))
			}
			for _, row := range batch {
				b.releaseRowUploads(row)
			}
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
		err := b.appendBatch(writeCtx, batch)
		writeExpired := writeCtx.Err() != nil
		cancelWrite()
		writeLeft -= time.Since(start)
		b.clearDeferredUploads(batch)
		if writeExpired && writeLeft > 0 {
			writeLeft = 0
		}
		if err != nil {
			b.config.logf("ERROR: Failed to write batch to BigQuery: %v", err)
		}
	}
	return writeLeft
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
	if b.streamWriter.isRotationInCooldown() {
		b.recordAppendFailureDrop(int64(len(batch)))
		for _, row := range batch {
			b.releaseRowUploads(row)
		}
		return fmt.Errorf("committed write stream rotation in cooldown")
	}
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
		b.recordAppendFailureDrop(int64(len(batch)))
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
		b.recordAppendFailureDrop(int64(len(batch)))
		return fmt.Errorf("streamWriter append error: %w", err)
	}

	if res.GetError() != nil || len(res.GetRowErrors()) > 0 {
		if b.config.Logger != nil {
			b.config.Logger.Printf("BigQuery response: %s", res.String())
		}
	}

	if res.GetError() != nil {
		b.recordAppendFailureDrop(int64(len(batch)))
		return fmt.Errorf("batch append error from BigQuery: %v", res.GetError().GetMessage())
	}

	if len(res.GetRowErrors()) > 0 {
		b.recordAppendFailureDrop(int64(len(batch)))
		return fmt.Errorf("batch append error from BigQuery: %d row errors found", len(res.GetRowErrors()))
	}

	return nil
}

const defaultRotateTimeout = 30 * time.Second

// streamWriter is a wrapper that maintains a persistent BigQuery write stream.
type streamWriter struct {
	writeClient  *bqstorage.BigQueryWriteClient
	streamName   string
	finalizeWg   sync.WaitGroup
	schemaBytes  []byte
	config       Config
	baseCtx      context.Context
	streamCtx    context.Context
	streamCancel context.CancelFunc
	streamOffset atomic.Int64
	// offsetDesynced is set when a batch with an ambiguous send ended without a
	// confirmed outcome, so streamOffset can no longer be trusted.
	offsetDesynced bool
	// freshlyRotated is true after rotateCommittedStream creates a replacement
	// stream until a batch is confirmed committed on that stream.
	freshlyRotated bool
	rotateRetryAt  time.Time
	rotateTimeout  time.Duration
	now            func() time.Time

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
		writeClient:   client,
		streamName:    streamName,
		schemaBytes:   schemaBytes,
		config:        config,
		baseCtx:       context.WithoutCancel(ctx),
		rotateTimeout: defaultRotateTimeout,
		now:           time.Now,
	}
}

func (s *streamWriter) nowOrDefault() time.Time {
	if s.now != nil {
		return s.now()
	}
	return time.Now()
}

func (s *streamWriter) rotateTimeoutOrDefault() time.Duration {
	if s.rotateTimeout > 0 {
		return s.rotateTimeout
	}
	return defaultRotateTimeout
}

func (s *streamWriter) resetStream() {
	if s.stream != nil {
		_ = s.stream.CloseSend()
		s.stream = nil
	}
	if s.streamCancel != nil {
		s.streamCancel()
		s.streamCancel = nil
	}
	s.streamCtx = nil
}

func isAmbiguousSendError(code codes.Code) bool {
	switch code {
	case codes.Unavailable,
		codes.DeadlineExceeded,
		codes.Canceled,
		codes.Aborted,
		codes.Internal,
		codes.Unknown:
		return true
	default:
		return false
	}
}

func (s *streamWriter) isRotationInCooldown() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.config.ExactlyOnceDelivery && s.offsetDesynced && s.nowOrDefault().Before(s.rotateRetryAt)
}

func (s *streamWriter) closeStream() {
	s.mu.Lock()
	s.resetStream()
	s.mu.Unlock()
	s.finalizeWg.Wait()
}

func (s *streamWriter) streamsToFinalize() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.streamName == "" {
		return nil
	}
	name := s.streamName
	s.streamName = ""
	return []string{name}
}

// rotateCommittedStream closes the active bidi stream, creates a fresh
// COMMITTED write stream bounded by rotateTimeout, finalizes the retired
// stream in the background, updates s.streamName, and resets s.streamOffset
// to 0. If a prior rotation failed within the cooldown window, it returns an
// error without calling CreateWriteStream.
// Caller must hold s.mu.
func (s *streamWriter) rotateCommittedStream(ctx context.Context) error {
	s.resetStream()
	if now := s.nowOrDefault(); now.Before(s.rotateRetryAt) {
		return fmt.Errorf("committed write stream rotation in cooldown until %s", s.rotateRetryAt.Format(time.RFC3339Nano))
	}
	timeout := s.rotateTimeoutOrDefault()
	rotateCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	newStream, err := createCommittedWriteStream(rotateCtx, s.writeClient, s.config)
	if err != nil {
		if ctx.Err() == nil {
			s.rotateRetryAt = s.nowOrDefault().Add(timeout)
		}
		return err
	}
	s.rotateRetryAt = time.Time{}
	if s.streamName != "" {
		retired := s.streamName
		s.finalizeWg.Add(1)
		go func(streamToFinalize string, wc *bqstorage.BigQueryWriteClient) {
			defer s.finalizeWg.Done()
			if wc == nil {
				return
			}
			fctx, fcancel := context.WithTimeout(context.Background(), defaultRotateTimeout)
			defer fcancel()
			_, _ = wc.FinalizeWriteStream(fctx, &storagepb.FinalizeWriteStreamRequest{
				Name: streamToFinalize,
			})
		}(retired, s.writeClient)
	}
	s.streamName = newStream
	s.streamOffset.Store(0)
	s.freshlyRotated = true
	return nil
}

// handleExactlyOnceError handles ExactlyOnceDelivery offset errors.
// When AlreadyExists occurs after a prior attempt whose Send succeeded but Recv
// failed (hadAmbiguousSend == true), BigQuery is confirming that the previous
// attempt committed the batch at that offset, so streamOffset is advanced and
// the append succeeds. When AlreadyExists occurs without a prior ambiguous send
// or OutOfRange/NotFound occurs, the stream offset is out of sync with the
// server, so we rotate to a fresh COMMITTED stream (at most once per batch)
// and reset streamOffset to 0. If rotation fails, already occurred in this
// batch, or the current stream was freshly rotated and rejected on its first
// append, rotateErr is returned (and the rotation cooldown is armed) so the
// caller stops retrying or creating streams on every batch.
// Caller must hold s.mu.
func (s *streamWriter) handleExactlyOnceError(
	ctx context.Context,
	code codes.Code,
	numRows int64,
	hadAmbiguousSend *bool,
	rotatedInBatch *bool,
	sendSucceeded bool,
) (bool, *storagepb.AppendRowsResponse, error) {
	switch code {
	case codes.AlreadyExists:
		if *hadAmbiguousSend {
			s.streamOffset.Add(numRows)
			*hadAmbiguousSend = false
			s.offsetDesynced = false
			s.freshlyRotated = false
			return true, &storagepb.AppendRowsResponse{}, nil
		}
		fallthrough
	case codes.OutOfRange, codes.NotFound:
		if *hadAmbiguousSend {
			*hadAmbiguousSend = false
			s.offsetDesynced = true
			return false, nil, fmt.Errorf("batch outcome unknown on stream (%v); dropped to prevent duplicate delivery", code)
		}
		if *rotatedInBatch || s.freshlyRotated {
			s.offsetDesynced = true
			if s.freshlyRotated {
				s.rotateRetryAt = s.nowOrDefault().Add(s.rotateTimeoutOrDefault())
			}
			return false, nil, fmt.Errorf("committed write stream already rotated (%v)", code)
		}
		*rotatedInBatch = true
		if err := s.rotateCommittedStream(ctx); err != nil {
			s.offsetDesynced = true
			return false, nil, fmt.Errorf("failed to rotate committed write stream after %v: %w", code, err)
		}
		s.offsetDesynced = false
		return false, nil, nil
	default:
		if sendSucceeded && isAmbiguousSendError(code) {
			*hadAmbiguousSend = true
		}
		return false, nil, nil
	}
}

func (s *streamWriter) append(ctx context.Context, recordBatch arrow.RecordBatch) (res *storagepb.AppendRowsResponse, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var lastErr error
	delay := s.config.RetryConfig.InitialDelay
	maxRetries := s.config.RetryConfig.MaxRetries
	hadAmbiguousSend := false
	rotatedInBatch := false
	if s.config.ExactlyOnceDelivery {
		if s.offsetDesynced {
			if s.nowOrDefault().Before(s.rotateRetryAt) {
				return nil, fmt.Errorf("committed write stream rotation in cooldown until %s", s.rotateRetryAt.Format(time.RFC3339Nano))
			}
			rotatedInBatch = true
			if rerr := s.rotateCommittedStream(ctx); rerr != nil {
				return nil, fmt.Errorf("committed stream out of sync and not replaced: %w", rerr)
			}
			s.offsetDesynced = false
		}
		defer func() {
			if hadAmbiguousSend && (err != nil || res.GetError() != nil || len(res.GetRowErrors()) > 0) {
				s.offsetDesynced = true
				if s.freshlyRotated {
					s.rotateRetryAt = s.nowOrDefault().Add(s.rotateTimeoutOrDefault())
				}
			}
		}()
	}

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
			if s.streamName != "" {
				baseCtx = metadata.AppendToOutgoingContext(baseCtx, "x-goog-request-params", "write_stream="+s.streamName)
			}
			streamCtx, streamCancel := context.WithCancel(baseCtx)
			stopOpen := context.AfterFunc(ctx, streamCancel)
			stream, streamErr := s.writeClient.AppendRows(streamCtx)
			if !stopOpen() {
				streamCancel()
				if streamErr == nil {
					streamErr = ctx.Err()
				}
				lastErr = fmt.Errorf("failed to open append rows stream: %w", streamErr)
				continue
			}
			if streamErr != nil {
				streamCancel()
				lastErr = fmt.Errorf("failed to open append rows stream: %w", streamErr)
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
		if s.config.ExactlyOnceDelivery {
			req.Offset = wrapperspb.Int64(s.streamOffset.Load())
		}

		var stopCall func() bool
		if s.streamCancel != nil {
			stopCall = context.AfterFunc(ctx, s.streamCancel)
		}

		prevRotated := rotatedInBatch
		if sendErr := s.stream.Send(req); sendErr != nil {
			// An error not generated by the client is reported as io.EOF here, with the
			// real status left for Recv; drain until it surfaces, releasing the stream.
			// Wrapped errors match too: draining a client-aborted stream is harmless,
			// and an exact match would let the wrapper escape in place of the status.
			if errors.Is(sendErr, io.EOF) {
				for {
					_, rerr := s.stream.Recv()
					if rerr == nil {
						continue
					}
					if !errors.Is(rerr, io.EOF) {
						sendErr = rerr
					}
					break
				}
			}
			if stopCall != nil {
				stopCall()
			}
			s.resetStream() // Reset stream on send error
			if s.config.ExactlyOnceDelivery {
				done, okRes, rotateErr := s.handleExactlyOnceError(ctx, status.Code(sendErr), recordBatch.NumRows(), &hadAmbiguousSend, &rotatedInBatch, false)
				if rotateErr != nil {
					return nil, rotateErr
				}
				if done {
					return okRes, nil
				}
				if !prevRotated && rotatedInBatch {
					attempt--
				}
			}
			lastErr = fmt.Errorf("failed to send row batch: %w", sendErr)
			continue
		}

		recvRes, recvErr := s.stream.Recv()
		if stopCall != nil {
			stopCall()
		}
		if recvErr != nil {
			s.resetStream() // Reset stream on receive error
			if s.config.ExactlyOnceDelivery {
				done, okRes, rotateErr := s.handleExactlyOnceError(ctx, status.Code(recvErr), recordBatch.NumRows(), &hadAmbiguousSend, &rotatedInBatch, true)
				if rotateErr != nil {
					return nil, rotateErr
				}
				if done {
					return okRes, nil
				}
				if !prevRotated && rotatedInBatch {
					attempt--
				}
			}
			lastErr = fmt.Errorf("failed to receive response for row batch: %w", recvErr)
			continue
		}

		if s.config.ExactlyOnceDelivery && recvRes.GetError() != nil {
			code := codes.Code(recvRes.GetError().GetCode())
			done, okRes, rotateErr := s.handleExactlyOnceError(ctx, code, recordBatch.NumRows(), &hadAmbiguousSend, &rotatedInBatch, false)
			if rotateErr != nil {
				return nil, rotateErr
			}
			if done {
				return okRes, nil
			}
			if !prevRotated && rotatedInBatch {
				attempt--
			}
			if code == codes.AlreadyExists || code == codes.OutOfRange || code == codes.NotFound {
				lastErr = fmt.Errorf("batch append error from BigQuery: %s", recvRes.GetError().GetMessage())
				continue
			}
		}

		if s.config.ExactlyOnceDelivery && recvRes.GetError() == nil && len(recvRes.GetRowErrors()) == 0 {
			s.streamOffset.Add(recordBatch.NumRows())
			hadAmbiguousSend = false
			s.offsetDesynced = false
			s.freshlyRotated = false
		}

		return recvRes, nil
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
// When ExactlyOnceDelivery is enabled, FinalizeWriteStream runs within whatever
// remains of the ShutdownTimeout write budget, and is skipped with a warning if
// draining already exhausted it. Close can therefore block for up to about two
// and a half times ShutdownTimeout: once for a write that was already in
// flight, and then for the uploads and the writes of the drain.
func (b *BatchProcessor) Close() {
	b.closeOnce.Do(func() {
		b.stopAcceptingRows()
		if b.started.Load() {
			<-b.done
		}
		writeLeft := b.flushAndWaitForInflightBatch()
		b.streamWriter.closeStream()
		if b.createdCommittedStream && b.writeClient != nil {
			streams := b.streamWriter.streamsToFinalize()
			if len(streams) > 0 {
				if writeLeft <= 0 {
					for _, name := range streams {
						b.config.logf("agentanalytics: skipping FinalizeWriteStream for %s: shutdown timeout exhausted while draining; already committed rows remain visible", name)
					}
				} else {
					finalizeCtx, finalizeCancel := context.WithTimeout(b.baseCtx, writeLeft)
					defer finalizeCancel()
					for _, name := range streams {
						if _, err := b.writeClient.FinalizeWriteStream(finalizeCtx, &storagepb.FinalizeWriteStreamRequest{
							Name: name,
						}); err != nil {
							b.config.logf("agentanalytics: failed to finalize write stream %s: %v", name, err)
						}
					}
				}
			}
		}
	})
}
