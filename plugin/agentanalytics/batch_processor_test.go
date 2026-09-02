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
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	storagepb "cloud.google.com/go/bigquery/storage/apiv1/storagepb"
	status "google.golang.org/genproto/googleapis/rpc/status"
)

// mockStream implements the unexported stream interface in streamWriter
type mockStream struct {
	sendErr   error
	recvRes   *storagepb.AppendRowsResponse
	recvErr   error
	sendCount int32
}

func (m *mockStream) Send(req *storagepb.AppendRowsRequest) error {
	atomic.AddInt32(&m.sendCount, 1)
	return m.sendErr
}

func (m *mockStream) Recv() (*storagepb.AppendRowsResponse, error) {
	return m.recvRes, m.recvErr
}

func (m *mockStream) CloseSend() error {
	return nil
}

func TestBatchProcessor_AppendAndFlush(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	config := DefaultConfig()
	config.BatchSize = 2
	config.BatchFlushIntv = 100 * time.Millisecond
	config.RetryConfig.MaxRetries = 0

	bp, err := NewBatchProcessor(ctx, nil, "test_stream", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor error: %v", err)
	}

	mStream := &mockStream{
		recvRes: &storagepb.AppendRowsResponse{},
	}
	bp.streamWriter.stream = mStream

	bp.Start()
	defer bp.Close()

	// Append 3 rows. The first 2 should trigger an immediate flush.
	bp.Append(map[string]any{"event_type": "test_event_1"})
	bp.Append(map[string]any{"event_type": "test_event_2"})
	bp.Append(map[string]any{"event_type": "test_event_3"})

	// Wait for the background goroutine to process the remaining row via periodic flush.
	time.Sleep(300 * time.Millisecond)

	count := atomic.LoadInt32(&mStream.sendCount)
	if count != 2 {
		t.Errorf("Expected 2 batch sends (one for size, one for timer), got %d", count)
	}
}

func TestBatchProcessor_WriteBatchErrors(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name    string
		sendErr error
		recvErr error
		recvRes *storagepb.AppendRowsResponse
	}{
		{
			name:    "Send error",
			sendErr: errors.New("send failed"),
		},
		{
			name:    "Recv error",
			recvErr: errors.New("recv failed"),
		},
		{
			name: "BigQuery append error",
			recvRes: &storagepb.AppendRowsResponse{
				Response: &storagepb.AppendRowsResponse_Error{
					Error: &status.Status{
						Message: "some append error",
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := DefaultConfig()
			config.RetryConfig.MaxRetries = 0 // Prevent generic reconnect on nil client

			bp, err := NewBatchProcessor(ctx, nil, "test_stream", config)
			if err != nil {
				t.Fatalf("NewBatchProcessor error: %v", err)
			}

			mStream := &mockStream{
				sendErr: tt.sendErr,
				recvErr: tt.recvErr,
				recvRes: tt.recvRes,
			}
			bp.streamWriter.stream = mStream

			err = bp.writeBatch(ctx, []map[string]any{
				{"event_type": "error_test_event"},
			})

			if err == nil {
				t.Fatalf("Expected error in writeBatch, got nil")
			}
		})
	}
}

func TestBatchProcessor_DataTypes(t *testing.T) {
	ctx := context.Background()
	config := DefaultConfig()
	config.RetryConfig.MaxRetries = 0

	bp, err := NewBatchProcessor(ctx, nil, "test_stream", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor error: %v", err)
	}

	mStream := &mockStream{
		recvRes: &storagepb.AppendRowsResponse{},
	}
	bp.streamWriter.stream = mStream

	row := map[string]any{
		"timestamp":      time.Now(),
		"event_type":     "data_types_test",
		"agent":          "test_agent",
		"session_id":     "sess-123",
		"invocation_id":  "inv-456",
		"user_id":        "u-789",
		"trace_id":       "trace-001",
		"span_id":        "span-001",
		"parent_span_id": "pspan-001",
		"is_truncated":   true,
		"content":        "{\"hello\":\"world\"}",
		"attributes":     "{\"key\":\"value\"}",
		"latency_ms":     "{\"llm\":100}",
		"status":         "OK",
		"error_message":  "",
	}

	err = bp.writeBatch(ctx, []map[string]any{row})
	if err != nil {
		t.Fatalf("writeBatch failed with data types: %v", err)
	}

	count := atomic.LoadInt32(&mStream.sendCount)
	if count != 1 {
		t.Errorf("Expected 1 send, got %d", count)
	}
}

func TestBatchProcessor_QueueFull(t *testing.T) {
	ctx := context.Background()

	config := DefaultConfig()
	config.BatchSize = 10
	config.QueueMaxSize = 1

	bp, err := NewBatchProcessor(ctx, nil, "test_stream", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor error: %v", err)
	}

	// This should succeed and sit in queue
	bp.Append(map[string]any{"event_type": "1"})

	// This should drop since queue is full (size 1) and we haven't flushed
	bp.Append(map[string]any{"event_type": "2"})

	if len(bp.queue) != 1 {
		t.Errorf("Expected queue size 1, got %d", len(bp.queue))
	}
}

func TestBatchProcessor_ContentPartsEdgeCases(t *testing.T) {
	ctx := context.Background()
	config := DefaultConfig()
	config.RetryConfig.MaxRetries = 0

	bp, err := NewBatchProcessor(ctx, nil, "test_stream", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor error: %v", err)
	}

	mStream := &mockStream{
		recvRes: &storagepb.AppendRowsResponse{},
	}
	bp.streamWriter.stream = mStream

	tests := []struct {
		name string
		row  map[string]any
	}{
		{
			name: "Missing content_parts",
			row: map[string]any{
				"timestamp":  time.Now(),
				"event_type": "missing_parts",
			},
		},
		{
			name: "Nil content_parts",
			row: map[string]any{
				"timestamp":     time.Now(),
				"event_type":    "nil_parts",
				"content_parts": nil,
			},
		},
		{
			name: "Empty content_parts",
			row: map[string]any{
				"timestamp":     time.Now(),
				"event_type":    "empty_parts",
				"content_parts": []map[string]any{},
			},
		},
		{
			name: "Invalid type content_parts",
			row: map[string]any{
				"timestamp":     time.Now(),
				"event_type":    "invalid_parts",
				"content_parts": "not a slice",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err = bp.writeBatch(ctx, []map[string]any{tt.row})
			if err != nil {
				t.Fatalf("writeBatch failed: %v", err)
			}
		})
	}
}
