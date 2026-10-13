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
	"fmt"
	"io"
	"net"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	bqstorage "cloud.google.com/go/bigquery/storage/apiv1"
	storagepb "cloud.google.com/go/bigquery/storage/apiv1/storagepb"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/option"
	statuspb "google.golang.org/genproto/googleapis/rpc/status"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// mockStream implements the unexported stream interface in streamWriter
type mockStream struct {
	sendErr   error
	sendDelay time.Duration
	onSend    func()
	recvRes   *storagepb.AppendRowsResponse
	recvErr   error
	recvSeq   []mockRecv
	sendCount int32
	recvCount int32

	mu   sync.Mutex
	sent []*storagepb.AppendRowsRequest
}

// mockRecv is one scripted Recv result; recvSeq supplies them in order before
// mockStream falls back to recvRes/recvErr. Set one of the two fields: a zero
// value answers (nil, nil), which append and writeBatch report as a success
// when it answers the Recv after a successful Send.
type mockRecv struct {
	res *storagepb.AppendRowsResponse
	err error
}

func (m *mockStream) Send(req *storagepb.AppendRowsRequest) error {
	if m.onSend != nil {
		m.onSend()
	}
	if m.sendDelay > 0 {
		time.Sleep(m.sendDelay)
	}
	atomic.AddInt32(&m.sendCount, 1)
	m.mu.Lock()
	m.sent = append(m.sent, req)
	m.mu.Unlock()
	return m.sendErr
}

// requests returns the requests passed to Send, in order.
func (m *mockStream) requests() []*storagepb.AppendRowsRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.sent)
}

// maxMockRecv bounds how many times Recv answers without a terminal error. The
// drain loop in append stops only on a non-nil error, so a mock that answers
// forever would spin instead of failing the test.
const maxMockRecv = 1000

func (m *mockStream) Recv() (*storagepb.AppendRowsResponse, error) {
	n := atomic.AddInt32(&m.recvCount, 1)
	if i := int(n) - 1; i < len(m.recvSeq) {
		return m.recvSeq[i].res, m.recvSeq[i].err
	}
	if m.recvErr == nil && int(n) > maxMockRecv {
		return nil, fmt.Errorf("mockStream.Recv called %d times with no terminal error set", n)
	}
	if m.recvRes == nil && m.recvErr == nil {
		// No result configured: the stream has already ended.
		return nil, io.EOF
	}
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
					Error: &statuspb.Status{
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

func TestBatchProcessor_WriteBatchEncodesRows(t *testing.T) {
	ctx := context.Background()
	config := DefaultConfig()
	config.RetryConfig.MaxRetries = 0

	bp, err := NewBatchProcessor(ctx, nil, "test_stream", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor error: %v", err)
	}
	mStream := &mockStream{recvRes: &storagepb.AppendRowsResponse{}}
	bp.streamWriter.stream = mStream

	ts := time.Date(2026, 1, 2, 3, 4, 5, 123456789, time.UTC)
	row := map[string]any{
		"timestamp":  ts,
		"event_id":   "0123456789abcdef0123456789abcdef",
		"event_type": "LLM_REQUEST",
		"content":    `{"prompt":"hi"}`,
		"content_parts": []map[string]any{
			{"mime_type": "text/plain", "text": "hi", "part_index": 0, "storage_mode": "INLINE"},
		},
		"status":       "OK",
		"is_truncated": false,
	}
	if err := bp.writeBatch(ctx, []map[string]any{row}); err != nil {
		t.Fatalf("writeBatch() error: %v", err)
	}

	sent := mStream.requests()
	if len(sent) != 1 {
		t.Fatalf("Send received %d requests, want 1", len(sent))
	}
	want := []map[string]any{{
		"timestamp":  ts.Truncate(time.Microsecond),
		"event_id":   "0123456789abcdef0123456789abcdef",
		"event_type": "LLM_REQUEST",
		"content":    `{"prompt":"hi"}`,
		"content_parts": []map[string]any{
			{"mime_type": "text/plain", "text": "hi", "part_index": int64(0), "storage_mode": "INLINE"},
		},
		"status":       "OK",
		"is_truncated": false,
	}}
	if diff := cmp.Diff(want, decodeArrowRows(t, sent[0])); diff != "" {
		t.Errorf("decoded rows mismatch (-want +got):\n%s", diff)
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

// panickingLogger is a Logger whose Println, which reports a dropped event,
// panics.
type panickingLogger struct {
	printlnCalls atomic.Int32
}

func (*panickingLogger) Printf(string, ...any) {}

func (l *panickingLogger) Println(...any) {
	l.printlnCalls.Add(1)
	panic("logger failed")
}

// TestBatchProcessor_QueueFullPanickingLoggerDoesNotDeadlockClose checks that
// a Logger panicking while it reports a dropped event, a panic that the
// plugin recovers, does not leave the processor locked so that Close hangs.
func TestBatchProcessor_QueueFullPanickingLoggerDoesNotDeadlockClose(t *testing.T) {
	logger := &panickingLogger{}
	p, _, mCtx := newTestPluginWithConfig(t, func(c *Config) {
		c.QueueMaxSize = 1
		c.BatchSize = 100
		c.BatchFlushIntv = time.Hour
		c.Logger = logger
	})

	// The first message fills the queue. The second is dropped, and the
	// logger panics while reporting it.
	for range 2 {
		if _, err := p.OnUserMessageCallback()(mCtx, textContent("user", "hello")); err != nil {
			t.Fatalf("OnUserMessageCallback() error = %v", err)
		}
	}
	if got := logger.printlnCalls.Load(); got != 1 {
		t.Fatalf("Logger.Println calls = %d, want 1 for the dropped event", got)
	}

	closed := make(chan error, 1)
	go func() { closed <- p.Close() }()
	select {
	case err := <-closed:
		if err != nil {
			t.Errorf("Close() error = %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Close() did not return after the logger panicked on a dropped event")
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

func TestStreamWriter_SendError(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name          string
		sendErr       error
		recvErr       error
		recvSeq       []mockRecv
		wantHasStatus bool
		wantCode      codes.Code
		wantErrMsg    string
		wantRecvCount int32
	}{
		{
			name:          "server-side termination reports the status from Recv",
			sendErr:       io.EOF,
			recvErr:       status.Error(codes.Unavailable, "the connection is draining"),
			wantHasStatus: true,
			wantCode:      codes.Unavailable,
			wantErrMsg:    "the connection is draining",
			wantRecvCount: 1,
		},
		{
			name:    "buffered responses are drained until the status surfaces",
			sendErr: io.EOF,
			recvSeq: []mockRecv{
				{res: &storagepb.AppendRowsResponse{}},
				{res: &storagepb.AppendRowsResponse{}},
				{res: &storagepb.AppendRowsResponse{}},
			},
			recvErr:       status.Error(codes.ResourceExhausted, "Exceeds 'AppendRows throughput' quota"),
			wantHasStatus: true,
			wantCode:      codes.ResourceExhausted,
			wantErrMsg:    "AppendRows throughput",
			wantRecvCount: 4,
		},
		{
			name:          "a termination without a status is drained exactly once",
			sendErr:       io.EOF,
			recvErr:       io.EOF,
			wantErrMsg:    "EOF",
			wantRecvCount: 1,
		},
		{
			name:          "a wrapped io.EOF still drains and keeps its context",
			sendErr:       fmt.Errorf("send aborted: %w", io.EOF),
			recvErr:       io.EOF,
			wantErrMsg:    "send aborted",
			wantRecvCount: 1,
		},
		{
			name:          "a wrapped io.EOF still yields to the recovered status",
			sendErr:       fmt.Errorf("send aborted: %w", io.EOF),
			recvErr:       status.Error(codes.ResourceExhausted, "Exceeds 'AppendRows throughput' quota"),
			wantHasStatus: true,
			wantCode:      codes.ResourceExhausted,
			wantErrMsg:    "AppendRows throughput",
			wantRecvCount: 1,
		},
		{
			name:          "client-side error is reported as is",
			sendErr:       errors.New("send failed"),
			wantErrMsg:    "send failed",
			wantRecvCount: 0,
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

			mStream := &mockStream{sendErr: tt.sendErr, recvErr: tt.recvErr, recvSeq: tt.recvSeq}
			bp.streamWriter.stream = mStream

			err = bp.writeBatch(ctx, []map[string]any{{"event_type": "send_error_test_event"}})
			if err == nil {
				t.Fatalf("writeBatch: expected an error, got nil")
			}
			if !strings.Contains(err.Error(), "max retries exceeded") {
				t.Errorf("error %q lost the retry-exhaustion wrapper", err.Error())
			}
			if !strings.Contains(err.Error(), "failed to send row batch") {
				t.Errorf("error %q lost the send-error wrapper", err.Error())
			}

			if got := atomic.LoadInt32(&mStream.recvCount); got != tt.wantRecvCount {
				t.Errorf("Recv called %d times, want %d", got, tt.wantRecvCount)
			}
			if got := atomic.LoadInt32(&mStream.sendCount); got != 1 {
				t.Errorf("Send called %d times, want 1", got)
			}
			if bp.streamWriter.stream != nil {
				t.Errorf("streamWriter.stream = %T, want nil", bp.streamWriter.stream)
			}
			st, ok := status.FromError(err)
			if ok != tt.wantHasStatus {
				t.Errorf("gRPC status present = %v, want %v (err: %v)", ok, tt.wantHasStatus, err)
			}
			if tt.wantHasStatus && st.Code() != tt.wantCode {
				t.Errorf("status code = %v, want %v (err: %v)", st.Code(), tt.wantCode, err)
			}
			if !strings.Contains(err.Error(), tt.wantErrMsg) {
				t.Errorf("error %q does not contain %q", err.Error(), tt.wantErrMsg)
			}
		})
	}
}

func TestStreamWriter_RetryOpensAFreshStream(t *testing.T) {
	ctx := context.Background()

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	gSrv := grpc.NewServer()
	storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{})
	go func() { _ = gSrv.Serve(lis) }()
	t.Cleanup(gSrv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("write client: %v", err)
	}

	config := DefaultConfig()
	config.RetryConfig.MaxRetries = 1

	bp, err := NewBatchProcessor(ctx, writeClient, "projects/p/datasets/d/tables/t/_default", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}

	dead := &mockStream{sendErr: io.EOF, recvErr: io.EOF}
	bp.streamWriter.stream = dead

	if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "retry_probe"}}); err != nil {
		t.Errorf("attempt 1 did not open a fresh stream: %v", err)
	}
	if got := atomic.LoadInt32(&dead.sendCount); got != 1 {
		t.Errorf("the aborted stream was reused: Send called %d times on it, want 1", got)
	}
}

func TestBatchProcessor_CloseDrainsAndRejectsNewAppends(t *testing.T) {
	ctx := context.Background()
	config := DefaultConfig()
	config.BatchSize = 100
	config.BatchFlushIntv = time.Hour
	config.RetryConfig.MaxRetries = 0

	bp, err := NewBatchProcessor(ctx, nil, "test_stream", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor error: %v", err)
	}
	mStream := &mockStream{recvRes: &storagepb.AppendRowsResponse{}}
	bp.streamWriter.stream = mStream

	bp.Start()
	bp.Append(map[string]any{"event_type": "before_close"})
	bp.Close()

	// Appends after Close must be ignored.
	bp.Append(map[string]any{"event_type": "after_close"})
	if len(bp.queue) != 0 {
		t.Errorf("queue len after Close = %d, want 0", len(bp.queue))
	}
	if got := atomic.LoadInt32(&mStream.sendCount); got != 1 {
		t.Errorf("Send called %d times, want 1", got)
	}
}

func TestBatchProcessor_ConcurrentAppendAndClose(t *testing.T) {
	for iter := range 20 {
		ctx := context.Background()
		config := DefaultConfig()
		config.BatchSize = 50
		config.QueueMaxSize = 10000
		config.BatchFlushIntv = time.Hour
		config.RetryConfig.MaxRetries = 0

		bp, err := NewBatchProcessor(ctx, nil, "test_stream", config)
		if err != nil {
			t.Fatalf("iter %d: NewBatchProcessor error: %v", iter, err)
		}
		mStream := &mockStream{recvRes: &storagepb.AppendRowsResponse{}}
		bp.streamWriter.stream = mStream
		bp.Start()

		var wg sync.WaitGroup
		start := make(chan struct{})
		for g := range 8 {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				<-start
				for i := range 50 {
					bp.Append(map[string]any{"event_type": fmt.Sprintf("evt_%d_%d", id, i)})
				}
			}(g)
		}
		close(start)
		bp.Close()
		wg.Wait()

		if got := len(bp.queue); got != 0 {
			t.Fatalf("iter %d: queue len after Close and concurrent Appends = %d, want 0 (unflushed items stranded in queue)", iter, got)
		}
	}
}

func TestBatchProcessor_CloseDrainsInFlightAndQueuedBatches(t *testing.T) {
	ctx := context.Background()
	config := DefaultConfig()
	config.BatchSize = 5
	config.QueueMaxSize = 1000
	config.BatchFlushIntv = time.Hour
	config.RetryConfig.MaxRetries = 0

	bp, err := NewBatchProcessor(ctx, nil, "test_stream", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor error: %v", err)
	}

	firstSendStarted := make(chan struct{})
	var firstSendOnce sync.Once
	mStream := &mockStream{
		sendDelay: 15 * time.Millisecond,
		onSend: func() {
			firstSendOnce.Do(func() { close(firstSendStarted) })
		},
		recvRes: &storagepb.AppendRowsResponse{},
	}
	bp.streamWriter.stream = mStream
	bp.Start()

	const totalRows = 35
	for i := range totalRows {
		bp.Append(map[string]any{"event_type": fmt.Sprintf("evt_%d", i)})
	}

	// Wait until the first Append-triggered background flush is in flight, then
	// call Close() while remaining batches are still in the queue.
	<-firstSendStarted
	bp.Close()

	if got := len(bp.queue); got != 0 {
		t.Fatalf("queue len after Close = %d, want 0", got)
	}
	var gotRows int
	for _, req := range mStream.requests() {
		gotRows += len(decodeArrowRows(t, req))
	}
	if gotRows != totalRows {
		t.Errorf("total rows sent = %d, want %d", gotRows, totalRows)
	}
}

func TestBatchProcessor_ReusesCachedStreamAcrossFlushes(t *testing.T) {
	ctx := context.Background()

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	srv := &fakeBigQueryWriteServer{requests: make(chan *storagepb.AppendRowsRequest, 32)}
	gSrv := grpc.NewServer()
	storagepb.RegisterBigQueryWriteServer(gSrv, srv)
	go func() { _ = gSrv.Serve(lis) }()
	t.Cleanup(gSrv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("write client: %v", err)
	}
	t.Cleanup(func() { _ = writeClient.Close() })

	config := DefaultConfig()
	config.BatchSize = 10
	config.BatchFlushIntv = time.Hour
	config.RetryConfig.MaxRetries = 0

	bp, err := NewBatchProcessor(ctx, writeClient, "projects/p/datasets/d/tables/t/_default", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	bp.Start()

	const numFlushes = 20
	for i := range numFlushes {
		bp.Append(map[string]any{"event_type": fmt.Sprintf("flush_%d", i)})
		bp.flush()
	}
	bp.Close()

	if got := srv.streamsOpened.Load(); got != 1 {
		t.Errorf("AppendRows opened %d streams across %d flushes, want 1", got, numFlushes)
	}
	var gotRows int
	for len(srv.requests) > 0 {
		req := <-srv.requests
		gotRows += len(decodeArrowRows(t, req))
	}
	if gotRows != numFlushes {
		t.Errorf("received %d rows across %d flushes with MaxRetries=0, want %d", gotRows, numFlushes, numFlushes)
	}
}

func TestBatchProcessor_ConcurrentCloseWaitsForDrain(t *testing.T) {
	ctx := context.Background()
	config := DefaultConfig()
	config.BatchSize = 5
	config.QueueMaxSize = 100
	config.BatchFlushIntv = time.Hour
	config.RetryConfig.MaxRetries = 0

	bp, err := NewBatchProcessor(ctx, nil, "test_stream", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor error: %v", err)
	}

	firstSendEntered := make(chan struct{})
	releaseFirstSend := make(chan struct{})
	var firstSendOnce sync.Once
	mStream := &mockStream{
		onSend: func() {
			firstSendOnce.Do(func() {
				close(firstSendEntered)
				<-releaseFirstSend
			})
		},
		recvRes: &storagepb.AppendRowsResponse{},
	}
	bp.streamWriter.stream = mStream
	bp.Start()

	const totalRows = 15
	for i := range totalRows {
		bp.Append(map[string]any{"event_type": fmt.Sprintf("evt_%d", i)})
	}

	<-firstSendEntered

	close1Done := make(chan struct{})
	go func() {
		bp.Close()
		close(close1Done)
	}()

	close2Done := make(chan struct{})
	go func() {
		bp.Close()
		close(close2Done)
	}()

	select {
	case <-close2Done:
		t.Fatal("second Close() returned while first Close() was still draining")
	case <-time.After(20 * time.Millisecond):
	}

	close(releaseFirstSend)
	<-close1Done
	<-close2Done

	var gotRows int
	for _, req := range mStream.requests() {
		gotRows += len(decodeArrowRows(t, req))
	}
	if gotRows != totalRows {
		t.Errorf("total rows sent after concurrent Close = %d, want %d", gotRows, totalRows)
	}
}

func TestBatchProcessor_DrainDeliversQueuedRowsAfterConstructorContextCanceled_FreshStream(t *testing.T) {
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	srv := &fakeBigQueryWriteServer{requests: make(chan *storagepb.AppendRowsRequest, 10)}
	gSrv := grpc.NewServer()
	storagepb.RegisterBigQueryWriteServer(gSrv, srv)
	go func() { _ = gSrv.Serve(lis) }()
	t.Cleanup(gSrv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("write client: %v", err)
	}
	t.Cleanup(func() { _ = writeClient.Close() })

	config := DefaultConfig()
	config.BatchSize = 10
	config.BatchFlushIntv = time.Hour
	config.ShutdownTimeout = time.Second
	config.RetryConfig.MaxRetries = 1

	bp, err := NewBatchProcessor(ctx, writeClient, "projects/p/datasets/d/tables/t/_default", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}

	bp.Start()
	bp.Append(map[string]any{"event_type": "queued_fresh_row"})

	// Cancel constructor context before Close()
	cancel()

	bp.Close()

	select {
	case req := <-srv.requests:
		rows := decodeArrowRows(t, req)
		if len(rows) != 1 {
			t.Fatalf("got %d rows, want 1", len(rows))
		}
		if rows[0]["event_type"] != "queued_fresh_row" {
			t.Errorf("event_type = %v, want queued_fresh_row", rows[0]["event_type"])
		}
	case <-time.After(time.Second):
		t.Fatal("queued row was not delivered during shutdown drain with canceled constructor context")
	}

	if bp.streamWriter.stream != nil {
		t.Error("stream was not explicitly closed after draining")
	}
}

func TestBatchProcessor_DrainDeliversQueuedRowsAfterConstructorContextCanceled_CachedStream(t *testing.T) {
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	srv := &fakeBigQueryWriteServer{requests: make(chan *storagepb.AppendRowsRequest, 10)}
	gSrv := grpc.NewServer()
	storagepb.RegisterBigQueryWriteServer(gSrv, srv)
	go func() { _ = gSrv.Serve(lis) }()
	t.Cleanup(gSrv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("write client: %v", err)
	}
	t.Cleanup(func() { _ = writeClient.Close() })

	config := DefaultConfig()
	config.BatchSize = 10
	config.BatchFlushIntv = time.Hour
	config.ShutdownTimeout = time.Second
	config.RetryConfig.MaxRetries = 1

	bp, err := NewBatchProcessor(ctx, writeClient, "projects/p/datasets/d/tables/t/_default", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}

	bp.Start()

	// Send initial batch to establish cached stream
	if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "initial_cached"}}); err != nil {
		t.Fatalf("initial writeBatch: %v", err)
	}
	select {
	case <-srv.requests:
	case <-time.After(time.Second):
		t.Fatal("initial batch not received")
	}

	if bp.streamWriter.stream == nil {
		t.Fatal("expected cached stream to be present")
	}

	bp.Append(map[string]any{"event_type": "queued_cached_row"})

	// Cancel constructor context, rendering cached stream canceled
	cancel()

	bp.Close()

	select {
	case req := <-srv.requests:
		rows := decodeArrowRows(t, req)
		if len(rows) != 1 {
			t.Fatalf("got %d rows, want 1", len(rows))
		}
		if rows[0]["event_type"] != "queued_cached_row" {
			t.Errorf("event_type = %v, want queued_cached_row", rows[0]["event_type"])
		}
	case <-time.After(time.Second):
		t.Fatal("queued row was not delivered during shutdown drain with canceled cached stream")
	}

	if bp.streamWriter.stream != nil {
		t.Error("stream was not explicitly closed after draining")
	}
}

func TestBatchProcessor_RootContextCancelBeforeFirstStreamOpenDrainsQueuedRowOnClose(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	srv := &fakeBigQueryWriteServer{requests: make(chan *storagepb.AppendRowsRequest, 10)}
	gSrv := grpc.NewServer()
	storagepb.RegisterBigQueryWriteServer(gSrv, srv)
	go func() { _ = gSrv.Serve(lis) }()
	t.Cleanup(gSrv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	writeClient, err := bqstorage.NewBigQueryWriteClient(context.Background(), option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("write client: %v", err)
	}
	t.Cleanup(func() { _ = writeClient.Close() })

	config := DefaultConfig()
	config.BatchSize = 10
	config.BatchFlushIntv = time.Hour
	config.ShutdownTimeout = time.Second
	config.RetryConfig.MaxRetries = 1

	bp, err := NewBatchProcessor(ctx, writeClient, "projects/p/datasets/d/tables/t/_default", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}

	bp.Start()

	// Append row WITHOUT prior writeBatch, so streamWriter.stream is nil.
	bp.Append(map[string]any{"event_type": "queued_unopened_stream_row"})

	// Cancel constructor context BEFORE stream is ever opened.
	cancel()

	bp.Close()

	select {
	case req := <-srv.requests:
		rows := decodeArrowRows(t, req)
		if len(rows) != 1 {
			t.Fatalf("got %d rows, want 1", len(rows))
		}
		if rows[0]["event_type"] != "queued_unopened_stream_row" {
			t.Errorf("event_type = %v, want queued_unopened_stream_row", rows[0]["event_type"])
		}
	case <-time.After(time.Second):
		t.Fatal("queued row was not delivered during shutdown drain when root context was cancelled before opening stream")
	}
}

func TestBatchProcessor_InitContextCanceledDoesNotStopProcessing(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	srv := &fakeBigQueryWriteServer{requests: make(chan *storagepb.AppendRowsRequest, 10)}
	gSrv := grpc.NewServer()
	storagepb.RegisterBigQueryWriteServer(gSrv, srv)
	go func() { _ = gSrv.Serve(lis) }()
	t.Cleanup(gSrv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	initCtx, cancelInit := context.WithCancel(context.Background())
	writeClient, err := bqstorage.NewBigQueryWriteClient(context.Background(), option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("write client: %v", err)
	}
	t.Cleanup(func() { _ = writeClient.Close() })

	config := DefaultConfig()
	config.BatchSize = 1
	config.BatchFlushIntv = 10 * time.Millisecond
	config.ShutdownTimeout = time.Second

	bp, err := NewBatchProcessor(initCtx, writeClient, "projects/p/datasets/d/tables/t/_default", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}

	// Cancel the init context immediately after construction.
	cancelInit()

	bp.Start()

	// Append a row after init context is canceled; it should still be flushed and delivered.
	bp.Append(map[string]any{"event_type": "post_cancel_row"})

	select {
	case req := <-srv.requests:
		rows := decodeArrowRows(t, req)
		if len(rows) != 1 || rows[0]["event_type"] != "post_cancel_row" {
			t.Fatalf("got rows %v, want 1 row with post_cancel_row", rows)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("row was not flushed because processor stopped when constructor context was canceled")
	}

	bp.Close()
}

// TestBatchProcessor_ConcurrentStartAndClose verifies that multiple concurrent Start and Close
// calls do not race or cause a close-of-closed-channel panic.
func TestBatchProcessor_ConcurrentStartAndClose(t *testing.T) {
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	t.Cleanup(func() { _ = lis.Close() })
	gSrv := grpc.NewServer()
	t.Cleanup(func() { gSrv.Stop() })
	srv := &fakeBigQueryWriteServer{requests: make(chan *storagepb.AppendRowsRequest, 10)}
	storagepb.RegisterBigQueryWriteServer(gSrv, srv)
	go func() { _ = gSrv.Serve(lis) }()

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	writeClient, err := bqstorage.NewBigQueryWriteClient(context.Background(), option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("write client: %v", err)
	}
	t.Cleanup(func() { _ = writeClient.Close() })

	config := DefaultConfig()
	config.BatchSize = 10
	config.BatchFlushIntv = 10 * time.Millisecond

	bp, err := NewBatchProcessor(context.Background(), writeClient, "projects/p/datasets/d/tables/t/_default", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			bp.Start()
		}()
		wg.Add(1)
		go func() {
			defer wg.Done()
			bp.Close()
		}()
	}
	wg.Wait()
}

// stallingWriteServer reads the first AppendRows request of each stream and
// then never answers, like a stalled BigQuery backend.
type stallingWriteServer struct {
	storagepb.UnimplementedBigQueryWriteServer
	received chan struct{}
}

func (s *stallingWriteServer) AppendRows(stream storagepb.BigQueryWrite_AppendRowsServer) error {
	if _, err := stream.Recv(); err != nil {
		return err
	}
	select {
	case s.received <- struct{}{}:
	case <-stream.Context().Done():
		return stream.Context().Err()
	}
	<-stream.Context().Done()
	return stream.Context().Err()
}

// TestBatchProcessor_StalledAppendRowsEndsAtShutdownTimeout checks that a
// write whose AppendRows response never arrives is abandoned once
// ShutdownTimeout expires, both in a flush (as at the end of a run) and in the
// drain during Close, instead of blocking them indefinitely.
func TestBatchProcessor_StalledAppendRowsEndsAtShutdownTimeout(t *testing.T) {
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	srv := &stallingWriteServer{received: make(chan struct{}, 2)}
	gSrv := grpc.NewServer()
	storagepb.RegisterBigQueryWriteServer(gSrv, srv)
	go func() { _ = gSrv.Serve(lis) }()
	t.Cleanup(gSrv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	writeClient, err := bqstorage.NewBigQueryWriteClient(context.Background(), option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("write client: %v", err)
	}
	t.Cleanup(func() { _ = writeClient.Close() })

	config := DefaultConfig()
	config.BatchSize = 10
	config.BatchFlushIntv = time.Hour
	config.ShutdownTimeout = 200 * time.Millisecond
	config.RetryConfig.MaxRetries = 0

	bp, err := NewBatchProcessor(context.Background(), writeClient, "projects/p/datasets/d/tables/t/_default", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	bp.Start()

	// runWithin runs f and returns how long it took, failing the test if f
	// does not return within 5 seconds.
	runWithin := func(name string, f func()) time.Duration {
		t.Helper()
		start := time.Now()
		done := make(chan struct{})
		go func() {
			defer close(done)
			f()
		}()
		select {
		case <-done:
			return time.Since(start)
		case <-time.After(5 * time.Second):
			t.Fatalf("%s did not return while the AppendRows response was stalled", name)
			return 0
		}
	}
	awaitRequest := func(name string) {
		t.Helper()
		select {
		case <-srv.received:
		case <-time.After(5 * time.Second):
			t.Fatalf("server did not receive the batch written by %s", name)
		}
	}

	bp.Append(map[string]any{"event_type": "stalled_flush_row"})
	if elapsed := runWithin("flush", bp.flush); elapsed < config.ShutdownTimeout {
		t.Errorf("flush returned after %v, before ShutdownTimeout (%v) expired", elapsed, config.ShutdownTimeout)
	}
	awaitRequest("flush")

	bp.Append(map[string]any{"event_type": "stalled_drain_row"})
	runWithin("Close", bp.Close)
	awaitRequest("Close")
}
