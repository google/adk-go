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
	"context"
	"fmt"
	"sync"
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

	flushMu     sync.Mutex
	arrowSchema *arrow.Schema
	allocator   memory.Allocator

	ctx     context.Context
	cancel  context.CancelFunc
	done    chan struct{}
	started bool // Track if Start() has been called.
}

// NewBatchProcessor creates a new BatchProcessor instance.
func NewBatchProcessor(ctx context.Context, writeClient *bqstorage.BigQueryWriteClient, streamName string, config Config) (*BatchProcessor, error) {
	ctx, cancel := context.WithCancel(ctx)
	schemaBytes, err := SerializedArrowSchema()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to serialize arrow schema: %w", err)
	}

	return &BatchProcessor{
		writeClient:  writeClient,
		streamWriter: newStreamWriter(writeClient, streamName, schemaBytes, config),
		streamName:   streamName,
		config:       config,
		queue:        make(chan map[string]any, config.QueueMaxSize),
		arrowSchema:  ArrowSchema(),
		allocator:    memory.NewGoAllocator(), // Use Go memory allocator
		ctx:          ctx,
		cancel:       cancel,
		done:         make(chan struct{}),
		started:      false,
	}, nil
}

// Start begins the background goroutine to process and flush the queue periodically.
func (b *BatchProcessor) Start() {
	if !b.started {
		b.started = true
		go func() {
			defer close(b.done)
			ticker := time.NewTicker(b.config.BatchFlushIntv)
			defer ticker.Stop()

			for {
				select {
				case <-b.ctx.Done():
					shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(b.ctx), b.config.ShutdownTimeout)
					defer cancel()
					for len(b.queue) > 0 {
						b.blockingFlush(shutdownCtx) // Force flush remaining on shutdown
					}
					// Wait for any concurrent background flushes to finish
					b.flushMu.Lock()
					b.flushMu.Unlock()
					return
				case <-ticker.C:
					b.flush()
				}
			}
		}()
	}
}

// Append adds a row to the processor queue asynchronously.
func (b *BatchProcessor) Append(row map[string]any) {
	select {
	case b.queue <- row:
		if len(b.queue) == b.config.BatchSize {
			go b.flush()
		}
	default:
		if b.config.Logger != nil {
			b.config.Logger.Println("WARNING: BigQuery event queue is full, dropping event.")
		}
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
func (b *BatchProcessor) writeAndLogBatch(ctx context.Context, batch []map[string]any) {
	if len(batch) == 0 {
		return
	}
	if err := b.writeBatch(ctx, batch); err != nil && b.config.Logger != nil {
		b.config.Logger.Printf("ERROR: Failed to write batch to BigQuery: %v", err)
	}
}

// flush drains the current queue and appends it. Uses TryLock to avoid blocking.
func (b *BatchProcessor) flush() {
	// Ensure single flush execution at any moment
	if !b.flushMu.TryLock() {
		return
	}
	defer b.flushMu.Unlock()

	batch := b.drainQueue()
	b.writeAndLogBatch(b.ctx, batch)

	// If the queue has enough items for another batch, drain and write synchronously.
	for len(b.queue) >= b.config.BatchSize {
		batch = b.drainQueue()
		b.writeAndLogBatch(b.ctx, batch)
	}
}

// blockingFlush drains the current queue and appends it. Uses a blocking Lock.
func (b *BatchProcessor) blockingFlush(ctx context.Context) {
	b.flushMu.Lock()
	defer b.flushMu.Unlock()

	batch := b.drainQueue()
	b.writeAndLogBatch(ctx, batch)
	// No recursive call here; this is for ensuring completion on shutdown.
}

// writeBatch allocates a new Apache Arrow RecordBuilder from the batch and commits the writes to BigQuery.
func (b *BatchProcessor) writeBatch(ctx context.Context, batch []map[string]any) error {
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
			case "timestamp":
				if v, ok := val.(time.Time); ok {
					fieldVector.(*array.TimestampBuilder).Append(arrow.Timestamp(v.UnixMicro()))
				} else {
					fieldVector.AppendNull()
				}
			case "event_type", "agent", "session_id", "invocation_id", "user_id", "trace_id", "span_id", "parent_span_id", "status", "error_message":
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

	res, err := b.streamWriter.append(ctx, record)
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
	writeClient *bqstorage.BigQueryWriteClient
	streamName  string
	schemaBytes []byte
	config      Config

	stream interface {
		Send(*storagepb.AppendRowsRequest) error
		Recv() (*storagepb.AppendRowsResponse, error)
		CloseSend() error
	}
	mu sync.Mutex
}

func newStreamWriter(client *bqstorage.BigQueryWriteClient, streamName string, schemaBytes []byte, config Config) *streamWriter {
	return &streamWriter{
		writeClient: client,
		streamName:  streamName,
		schemaBytes: schemaBytes,
		config:      config,
	}
}

func (s *streamWriter) append(ctx context.Context, recordBatch arrow.RecordBatch) (*storagepb.AppendRowsResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var lastErr error
	delay := s.config.RetryConfig.InitialDelay

	for attempt := 0; attempt <= s.config.RetryConfig.MaxRetries; attempt++ {
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

		var err error
		if s.stream == nil {
			s.stream, err = s.writeClient.AppendRows(ctx)
			if err != nil {
				lastErr = fmt.Errorf("failed to open append rows stream: %w", err)
				continue
			}
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

		if err := s.stream.Send(req); err != nil {
			s.stream = nil // Reset stream on send error
			lastErr = fmt.Errorf("failed to send row batch: %w", err)
			continue
		}

		res, err := s.stream.Recv()
		if err != nil {
			s.stream = nil // Reset stream on receive error
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

// Close gracefully shuts down the processing loop, flushes, and stops.
func (b *BatchProcessor) Close() {
	b.cancel()
	// Only wait on the done channel if the Start() goroutine was actually launched.
	if b.started {
		<-b.done
	}
}
