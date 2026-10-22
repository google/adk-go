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
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	wrapperspb "google.golang.org/protobuf/types/known/wrapperspb"
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

type scriptedClientStream struct {
	grpc.ClientStream
	openNum int32
}

func (s *scriptedClientStream) SendMsg(m any) error {
	if s.openNum == 1 {
		return nil
	}
	return io.EOF
}

func (s *scriptedClientStream) RecvMsg(m any) error {
	if s.openNum == 1 {
		return status.Error(codes.Unavailable, "ambiguous recv loss")
	}
	return status.Error(codes.AlreadyExists, "offset already committed")
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

	// Appends after Close must be ignored and counted in ShutdownDrops.
	bp.Append(map[string]any{"event_type": "after_close"})
	if len(bp.queue) != 0 {
		t.Errorf("queue len after Close = %d, want 0", len(bp.queue))
	}
	if got := atomic.LoadInt32(&mStream.sendCount); got != 1 {
		t.Errorf("Send called %d times, want 1", got)
	}
	wantStats := DropStats{
		ShutdownDrops: 1,
		TotalDrops:    1,
	}
	if diff := cmp.Diff(wantStats, bp.DropStats()); diff != "" {
		t.Errorf("DropStats() after Close mismatch (-want +got):\n%s", diff)
	}
}

func TestBatchProcessor_DropStats(t *testing.T) {
	ctx := context.Background()

	var nilBP *BatchProcessor
	if got := nilBP.DropStats(); got != (DropStats{}) {
		t.Errorf("nil BatchProcessor.DropStats() = %+v, want zero DropStats", got)
	}

	config := DefaultConfig()
	config.BatchSize = 10
	config.QueueMaxSize = 1
	config.RetryConfig.MaxRetries = 0

	bp, err := NewBatchProcessor(ctx, nil, "test_stream", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor error: %v", err)
	}

	// 1. Trigger 2 queue-full drops (queue capacity is 1).
	bp.Append(map[string]any{"event_type": "ok_1"})
	bp.Append(map[string]any{"event_type": "drop_1"})
	bp.Append(map[string]any{"event_type": "drop_2"})

	// 2. Trigger append failure drops (batch of 3 rows failing on Send).
	bp.streamWriter.stream = &mockStream{sendErr: errors.New("send failure")}
	_ = bp.writeBatch(ctx, []map[string]any{
		{"event_type": "fail_1"},
		{"event_type": "fail_2"},
		{"event_type": "fail_3"},
	})

	// 3. Trigger append failure drops from BigQuery RowErrors (batch of 2 rows).
	bp.streamWriter.stream = &mockStream{
		recvRes: &storagepb.AppendRowsResponse{
			RowErrors: []*storagepb.RowError{{Index: 0, Message: "bad row"}},
		},
	}
	_ = bp.writeBatch(ctx, []map[string]any{
		{"event_type": "row_err_1"},
		{"event_type": "row_err_2"},
	})

	// 4. Trigger append failure drops from in-band AppendRowsResponse.Error (batch of 4 rows).
	bp.streamWriter.stream = &mockStream{
		recvRes: &storagepb.AppendRowsResponse{
			Response: &storagepb.AppendRowsResponse_Error{
				Error: &statuspb.Status{
					Code:    int32(codes.InvalidArgument),
					Message: "in-band batch error",
				},
			},
		},
	}
	_ = bp.writeBatch(ctx, []map[string]any{
		{"event_type": "inband_err_1"},
		{"event_type": "inband_err_2"},
		{"event_type": "inband_err_3"},
		{"event_type": "inband_err_4"},
	})

	// 5. Trigger formatter failure drop (counted separately from TotalDrops because row is still written).
	bp.recordFormatterFailureDrop()

	// 6. Trigger shutdown drop after Close (restore a healthy stream so ok_1 flushes cleanly on Close).
	bp.streamWriter.stream = &mockStream{recvRes: &storagepb.AppendRowsResponse{}}
	bp.Close()
	bp.Append(map[string]any{"event_type": "after_close"})

	want := DropStats{
		QueueFullDrops:        2,
		AppendFailureDrops:    9,
		FormatterFailureDrops: 1,
		ShutdownDrops:         1,
		TotalDrops:            12,
	}
	if diff := cmp.Diff(want, bp.DropStats()); diff != "" {
		t.Errorf("DropStats() mismatch (-want +got):\n%s", diff)
	}
}

type committedStreamTestServer struct {
	storagepb.UnimplementedBigQueryWriteServer

	mu                 sync.Mutex
	createdStreamName  string
	createReqs         []*storagepb.CreateWriteStreamRequest
	createErr          error
	finalizeReqs       []*storagepb.FinalizeWriteStreamRequest
	appendStreamNames  []string
	appendOffsets      []int64
	appendHasOffsets   []bool
	failAppendAttemptN int
	appendAttempts     int
}

func (s *committedStreamTestServer) CreateWriteStream(_ context.Context, req *storagepb.CreateWriteStreamRequest) (*storagepb.WriteStream, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.createReqs = append(s.createReqs, req)
	if s.createErr != nil {
		return nil, s.createErr
	}
	return &storagepb.WriteStream{
		Name: s.createdStreamName,
		Type: req.GetWriteStream().GetType(),
	}, nil
}

func (s *committedStreamTestServer) FinalizeWriteStream(_ context.Context, req *storagepb.FinalizeWriteStreamRequest) (*storagepb.FinalizeWriteStreamResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.finalizeReqs = append(s.finalizeReqs, req)
	return &storagepb.FinalizeWriteStreamResponse{}, nil
}

func (s *committedStreamTestServer) AppendRows(stream storagepb.BigQueryWrite_AppendRowsServer) error {
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		s.mu.Lock()
		s.appendAttempts++
		attempt := s.appendAttempts
		s.appendStreamNames = append(s.appendStreamNames, req.GetWriteStream())
		if req.GetOffset() != nil {
			s.appendHasOffsets = append(s.appendHasOffsets, true)
			s.appendOffsets = append(s.appendOffsets, req.GetOffset().GetValue())
		} else {
			s.appendHasOffsets = append(s.appendHasOffsets, false)
			s.appendOffsets = append(s.appendOffsets, -1)
		}
		shouldFail := s.failAppendAttemptN > 0 && attempt == s.failAppendAttemptN
		s.mu.Unlock()

		if shouldFail {
			return status.Error(codes.Unavailable, "transient stream error")
		}
		if err := stream.Send(&storagepb.AppendRowsResponse{}); err != nil {
			return err
		}
	}
}

func TestBatchProcessor_ExactlyOnceDelivery_CommittedStreamAndOffsets(t *testing.T) {
	ctx := context.Background()

	srv := &committedStreamTestServer{
		createdStreamName:  "projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-1",
		failAppendAttemptN: 2, // Fail the first attempt of batch 2 so retry reuses the same offset!
	}

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
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
	config.ProjectID = "test-proj"
	config.DatasetID = "test-ds"
	config.TableName = "agent_events"
	config.ExactlyOnceDelivery = true
	config.RetryConfig = RetryConfig{
		MaxRetries:   2,
		InitialDelay: time.Millisecond,
		MaxDelay:     5 * time.Millisecond,
		Multiplier:   1.0,
	}

	bp, err := NewBatchProcessor(ctx, writeClient, "", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor() error = %v", err)
	}

	// Verify CreateWriteStream was called with COMMITTED type.
	srv.mu.Lock()
	if len(srv.createReqs) != 1 {
		t.Fatalf("CreateWriteStream calls = %d, want 1", len(srv.createReqs))
	}
	if got := srv.createReqs[0].GetParent(); got != "projects/test-proj/datasets/test-ds/tables/agent_events" {
		t.Errorf("CreateWriteStream parent = %q, want %q", got, "projects/test-proj/datasets/test-ds/tables/agent_events")
	}
	if got := srv.createReqs[0].GetWriteStream().GetType(); got != storagepb.WriteStream_COMMITTED {
		t.Errorf("CreateWriteStream type = %v, want %v", got, storagepb.WriteStream_COMMITTED)
	}
	srv.mu.Unlock()

	// Batch 1 (2 rows) -> offset 0, advances offset to 2.
	if err := bp.writeBatch(ctx, []map[string]any{
		{"event_type": "LLM_REQUEST"},
		{"event_type": "LLM_RESPONSE"},
	}); err != nil {
		t.Fatalf("writeBatch(batch 1) error = %v", err)
	}

	// Batch 2 (3 rows) -> attempt 0 fails at offset 2, retry attempt 1 succeeds at offset 2, advances offset to 5.
	if err := bp.writeBatch(ctx, []map[string]any{
		{"event_type": "TOOL_STARTING"},
		{"event_type": "TOOL_COMPLETED"},
		{"event_type": "AGENT_COMPLETED"},
	}); err != nil {
		t.Fatalf("writeBatch(batch 2) error = %v", err)
	}

	// Batch 3 (1 row) -> offset 5, advances offset to 6.
	if err := bp.writeBatch(ctx, []map[string]any{
		{"event_type": "INVOCATION_COMPLETED"},
	}); err != nil {
		t.Fatalf("writeBatch(batch 3) error = %v", err)
	}

	bp.Close()

	srv.mu.Lock()
	defer srv.mu.Unlock()

	// Expect 4 AppendRows requests: batch1 (offset 0), batch2-attempt0 (offset 2), batch2-attempt1 (offset 2), batch3 (offset 5).
	wantOffsets := []int64{0, 2, 2, 5}
	if diff := cmp.Diff(wantOffsets, srv.appendOffsets); diff != "" {
		t.Errorf("AppendRows offsets mismatch (-want +got):\n%s", diff)
	}
	for i, hasOffset := range srv.appendHasOffsets {
		if !hasOffset {
			t.Errorf("AppendRows request %d had nil Offset", i)
		}
		if srv.appendStreamNames[i] != srv.createdStreamName {
			t.Errorf("AppendRows request %d stream = %q, want %q", i, srv.appendStreamNames[i], srv.createdStreamName)
		}
	}

	if len(srv.finalizeReqs) != 1 {
		t.Fatalf("FinalizeWriteStream calls = %d, want 1", len(srv.finalizeReqs))
	}
	if got := srv.finalizeReqs[0].GetName(); got != srv.createdStreamName {
		t.Errorf("FinalizeWriteStream name = %q, want %q", got, srv.createdStreamName)
	}
}

func TestBatchProcessor_ExactlyOnceDelivery_CreateWriteStreamFailure(t *testing.T) {
	ctx := context.Background()
	srv := &committedStreamTestServer{
		createErr: status.Error(codes.PermissionDenied, "cannot create stream"),
	}

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
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
	config.ProjectID = "test-proj"
	config.DatasetID = "test-ds"
	config.ExactlyOnceDelivery = true

	if _, err := NewBatchProcessor(ctx, writeClient, "", config); err == nil {
		t.Fatal("NewBatchProcessor() error = nil, want CreateWriteStream error")
	}

	t.Run("honors caller ctx cancellation during initial CreateWriteStream", func(t *testing.T) {
		canceledCtx, cancel := context.WithCancel(ctx)
		cancel()
		if _, err := NewBatchProcessor(canceledCtx, writeClient, "", config); err == nil {
			t.Fatal("NewBatchProcessor() with canceled ctx error = nil, want context canceled error")
		}
	})
}

type scriptedCommittedStreamServer struct {
	storagepb.UnimplementedBigQueryWriteServer

	mu                sync.Mutex
	streamNames       []string
	createReqs        []*storagepb.CreateWriteStreamRequest
	finalizeReqs      []*storagepb.FinalizeWriteStreamRequest
	finalizeMetadata  []metadata.MD
	appendStreamNames []string
	appendOffsets     []int64
	onAppend          func(attempt int, req *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error)
	onFinalize        func(ctx context.Context, req *storagepb.FinalizeWriteStreamRequest) (*storagepb.FinalizeWriteStreamResponse, error)
	appendAttempts    int
}

func (s *scriptedCommittedStreamServer) CreateWriteStream(_ context.Context, req *storagepb.CreateWriteStreamRequest) (*storagepb.WriteStream, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.createReqs = append(s.createReqs, req)
	idx := len(s.createReqs) - 1
	name := fmt.Sprintf("projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-%d", idx+1)
	if idx < len(s.streamNames) {
		name = s.streamNames[idx]
	}
	return &storagepb.WriteStream{
		Name: name,
		Type: req.GetWriteStream().GetType(),
	}, nil
}

func (s *scriptedCommittedStreamServer) FinalizeWriteStream(ctx context.Context, req *storagepb.FinalizeWriteStreamRequest) (*storagepb.FinalizeWriteStreamResponse, error) {
	s.mu.Lock()
	s.finalizeReqs = append(s.finalizeReqs, req)
	md, _ := metadata.FromIncomingContext(ctx)
	s.finalizeMetadata = append(s.finalizeMetadata, md)
	fn := s.onFinalize
	s.mu.Unlock()
	if fn != nil {
		return fn(ctx, req)
	}
	return &storagepb.FinalizeWriteStreamResponse{}, nil
}

func (s *scriptedCommittedStreamServer) AppendRows(stream storagepb.BigQueryWrite_AppendRowsServer) error {
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		s.mu.Lock()
		s.appendAttempts++
		attempt := s.appendAttempts
		s.appendStreamNames = append(s.appendStreamNames, req.GetWriteStream())
		if req.GetOffset() != nil {
			s.appendOffsets = append(s.appendOffsets, req.GetOffset().GetValue())
		} else {
			s.appendOffsets = append(s.appendOffsets, -1)
		}
		fn := s.onAppend
		s.mu.Unlock()

		res, appErr := fn(attempt, req)
		if appErr != nil {
			return appErr
		}
		if err := stream.Send(res); err != nil {
			return err
		}
	}
}

func TestBatchProcessor_ExactlyOnceDelivery_AmbiguousSendThenAlreadyExistsAdvancesOffset(t *testing.T) {
	for _, tc := range []struct {
		name          string
		alreadyErrVia string
	}{
		{name: "AlreadyExists via AppendRowsResponse.Error", alreadyErrVia: "response"},
		{name: "AlreadyExists via gRPC status error", alreadyErrVia: "grpc"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			srv := &scriptedCommittedStreamServer{
				onAppend: func(attempt int, _ *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
					switch attempt {
					case 1:
						// Simulate server committing batch 1 (2 rows) at offset 0, then connection dropping before client Recv completes.
						return nil, status.Error(codes.Unavailable, "connection reset after commit")
					case 2:
						// Retry of batch 1 at offset 0 receives AlreadyExists confirming commit.
						if tc.alreadyErrVia == "grpc" {
							return nil, status.Error(codes.AlreadyExists, "offset 0 already exists")
						}
						return &storagepb.AppendRowsResponse{
							Response: &storagepb.AppendRowsResponse_Error{
								Error: &statuspb.Status{
									Code:    int32(codes.AlreadyExists),
									Message: "offset 0 already exists",
								},
							},
						}, nil
					default:
						return &storagepb.AppendRowsResponse{}, nil
					}
				},
			}

			lis, err := net.Listen("tcp", "localhost:0")
			if err != nil {
				t.Fatalf("listen: %v", err)
			}
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
			config.ProjectID = "test-proj"
			config.DatasetID = "test-ds"
			config.TableName = "agent_events"
			config.ExactlyOnceDelivery = true
			config.RetryConfig = RetryConfig{
				MaxRetries:   2,
				InitialDelay: time.Millisecond,
				MaxDelay:     5 * time.Millisecond,
				Multiplier:   1.0,
			}

			bp, err := NewBatchProcessor(ctx, writeClient, "", config)
			if err != nil {
				t.Fatalf("NewBatchProcessor() error = %v", err)
			}
			defer bp.Close()

			// Batch 1 (2 rows): attempt 1 fails after send, attempt 2 gets AlreadyExists -> treated as success, offset advances to 2.
			if err := bp.writeBatch(ctx, []map[string]any{
				{"event_type": "LLM_REQUEST"},
				{"event_type": "LLM_RESPONSE"},
			}); err != nil {
				t.Fatalf("writeBatch(batch 1) error = %v, want nil", err)
			}

			// Batch 2 (1 row): must use offset 2 on the same stream.
			if err := bp.writeBatch(ctx, []map[string]any{
				{"event_type": "INVOCATION_COMPLETED"},
			}); err != nil {
				t.Fatalf("writeBatch(batch 2) error = %v, want nil", err)
			}

			if got := bp.DropStats(); got != (DropStats{}) {
				t.Errorf("DropStats() = %+v, want zero drops", got)
			}

			srv.mu.Lock()
			defer srv.mu.Unlock()
			if len(srv.createReqs) != 1 {
				t.Errorf("CreateWriteStream calls = %d, want 1 (no stream rotation expected)", len(srv.createReqs))
			}
			wantOffsets := []int64{0, 0, 2}
			if diff := cmp.Diff(wantOffsets, srv.appendOffsets); diff != "" {
				t.Errorf("AppendRows offsets mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestBatchProcessor_ExactlyOnceDelivery_UnrecoverableOffsetConflictRotatesStream(t *testing.T) {
	ctx := context.Background()
	srv := &scriptedCommittedStreamServer{
		onAppend: func(attempt int, _ *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
			switch attempt {
			case 1:
				// Batch 1 (2 rows) succeeds at offset 0 on committed-1.
				return &storagepb.AppendRowsResponse{}, nil
			case 2:
				// Batch 2 (3 rows) at offset 2 on committed-1 hits an unexpected AlreadyExists without prior ambiguous send in this call.
				return &storagepb.AppendRowsResponse{
					Response: &storagepb.AppendRowsResponse_Error{
						Error: &statuspb.Status{
							Code:    int32(codes.AlreadyExists),
							Message: "offset 2 already exists from prior timed-out batch",
						},
					},
				}, nil
			default:
				// Retry of Batch 2 on rotated stream committed-2 at offset 0 and subsequent Batch 3 at offset 3 succeed.
				return &storagepb.AppendRowsResponse{}, nil
			}
		},
	}

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
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
	config.ProjectID = "test-proj"
	config.DatasetID = "test-ds"
	config.TableName = "agent_events"
	config.ExactlyOnceDelivery = true
	config.RetryConfig = RetryConfig{
		MaxRetries:   2,
		InitialDelay: time.Millisecond,
		MaxDelay:     5 * time.Millisecond,
		Multiplier:   1.0,
	}

	bp, err := NewBatchProcessor(ctx, writeClient, "", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor() error = %v", err)
	}

	// Batch 1 (2 rows) -> committed-1 @ offset 0.
	if err := bp.writeBatch(ctx, []map[string]any{
		{"event_type": "LLM_REQUEST"},
		{"event_type": "LLM_RESPONSE"},
	}); err != nil {
		t.Fatalf("writeBatch(batch 1) error = %v", err)
	}

	// Batch 2 (3 rows) -> attempt 1 on committed-1 @ offset 2 gets unrecoverable AlreadyExists,
	// rotates to committed-2 @ offset 0, succeeds and advances offset to 3.
	if err := bp.writeBatch(ctx, []map[string]any{
		{"event_type": "TOOL_STARTING"},
		{"event_type": "TOOL_COMPLETED"},
		{"event_type": "AGENT_COMPLETED"},
	}); err != nil {
		t.Fatalf("writeBatch(batch 2) error = %v", err)
	}

	// Batch 3 (1 row) -> committed-2 @ offset 3.
	if err := bp.writeBatch(ctx, []map[string]any{
		{"event_type": "INVOCATION_COMPLETED"},
	}); err != nil {
		t.Fatalf("writeBatch(batch 3) error = %v", err)
	}

	bp.Close()

	srv.mu.Lock()
	defer srv.mu.Unlock()

	if len(srv.createReqs) != 2 {
		t.Fatalf("CreateWriteStream calls = %d, want 2 (initial + rotated)", len(srv.createReqs))
	}
	wantOffsets := []int64{0, 2, 0, 3}
	if diff := cmp.Diff(wantOffsets, srv.appendOffsets); diff != "" {
		t.Errorf("AppendRows offsets mismatch (-want +got):\n%s", diff)
	}
	wantStreams := []string{
		"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-1",
		"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-1",
		"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-2",
		"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-2",
	}
	if diff := cmp.Diff(wantStreams, srv.appendStreamNames); diff != "" {
		t.Errorf("AppendRows stream names mismatch (-want +got):\n%s", diff)
	}
	var gotFinalized []string
	for _, r := range srv.finalizeReqs {
		gotFinalized = append(gotFinalized, r.GetName())
	}
	wantFinalized := []string{
		"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-1",
		"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-2",
	}
	if diff := cmp.Diff(wantFinalized, gotFinalized); diff != "" {
		t.Errorf("FinalizeWriteStream names mismatch (-want +got):\n%s", diff)
	}
}

func TestBatchProcessor_ExactlyOnceDelivery_TerminalAmbiguousSendRotatesStreamOnNextBatch(t *testing.T) {
	t.Run("exhausted ambiguous retries rotate stream before next batch", func(t *testing.T) {
		ctx := context.Background()
		srv := &scriptedCommittedStreamServer{
			onAppend: func(attempt int, _ *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
				switch attempt {
				case 1, 2:
					// Batch A (2 rows) at offset 0 on committed-1 fails with Unavailable on both attempts (MaxRetries=1).
					return nil, status.Error(codes.Unavailable, "ambiguous connection drop")
				default:
					// Batch B (1 row) must rotate to committed-2 at offset 0.
					return &storagepb.AppendRowsResponse{}, nil
				}
			},
		}

		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
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
		config.ProjectID = "test-proj"
		config.DatasetID = "test-ds"
		config.TableName = "agent_events"
		config.ExactlyOnceDelivery = true
		config.RetryConfig = RetryConfig{
			MaxRetries:   1,
			InitialDelay: time.Millisecond,
			MaxDelay:     5 * time.Millisecond,
			Multiplier:   1.0,
		}

		bp, err := NewBatchProcessor(ctx, writeClient, "", config)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}

		// Batch A (2 rows) exhausts retries after ambiguous sends.
		if err := bp.writeBatch(ctx, []map[string]any{
			{"event_type": "LLM_REQUEST"},
			{"event_type": "LLM_RESPONSE"},
		}); err == nil {
			t.Fatal("writeBatch(Batch A) error = nil, want retry exhaustion error")
		}

		// Batch B (1 row) must rotate to committed-2 at offset 0 before sending.
		if err := bp.writeBatch(ctx, []map[string]any{
			{"event_type": "INVOCATION_COMPLETED"},
		}); err != nil {
			t.Fatalf("writeBatch(Batch B) error = %v, want nil", err)
		}

		bp.Close()

		wantStats := DropStats{
			AppendFailureDrops: 2,
			TotalDrops:         2,
		}
		if diff := cmp.Diff(wantStats, bp.DropStats()); diff != "" {
			t.Errorf("DropStats() mismatch (-want +got):\n%s", diff)
		}

		srv.mu.Lock()
		defer srv.mu.Unlock()
		wantStreams := []string{
			"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-1",
			"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-1",
			"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-2",
		}
		if diff := cmp.Diff(wantStreams, srv.appendStreamNames); diff != "" {
			t.Errorf("AppendRows stream names mismatch (-want +got):\n%s", diff)
		}
		wantOffsets := []int64{0, 0, 0}
		if diff := cmp.Diff(wantOffsets, srv.appendOffsets); diff != "" {
			t.Errorf("AppendRows offsets mismatch (-want +got):\n%s", diff)
		}
		var gotFinalized []string
		for _, r := range srv.finalizeReqs {
			gotFinalized = append(gotFinalized, r.GetName())
		}
		wantFinalized := []string{
			"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-1",
			"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-2",
		}
		if diff := cmp.Diff(wantFinalized, gotFinalized); diff != "" {
			t.Errorf("FinalizeWriteStream names mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("ambiguous send followed by rejected response rotates stream before next batch", func(t *testing.T) {
		ctx := context.Background()
		srv := &scriptedCommittedStreamServer{
			onAppend: func(attempt int, _ *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
				switch attempt {
				case 1:
					return nil, status.Error(codes.Unavailable, "ambiguous drop")
				case 2:
					return &storagepb.AppendRowsResponse{
						Response: &storagepb.AppendRowsResponse_Error{
							Error: &statuspb.Status{
								Code:    int32(codes.InvalidArgument),
								Message: "bad batch",
							},
						},
					}, nil
				default:
					return &storagepb.AppendRowsResponse{}, nil
				}
			},
		}

		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
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
		config.ProjectID = "test-proj"
		config.DatasetID = "test-ds"
		config.TableName = "agent_events"
		config.ExactlyOnceDelivery = true
		config.RetryConfig = RetryConfig{
			MaxRetries:   1,
			InitialDelay: time.Millisecond,
			MaxDelay:     5 * time.Millisecond,
			Multiplier:   1.0,
		}

		bp, err := NewBatchProcessor(ctx, writeClient, "", config)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}
		defer bp.Close()

		if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "LLM_REQUEST"}}); err == nil {
			t.Fatal("writeBatch(Batch A) error = nil, want InvalidArgument error")
		}
		if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "LLM_RESPONSE"}}); err != nil {
			t.Fatalf("writeBatch(Batch B) error = %v, want nil", err)
		}

		srv.mu.Lock()
		defer srv.mu.Unlock()
		wantStreams := []string{
			"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-1",
			"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-1",
			"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-2",
		}
		if diff := cmp.Diff(wantStreams, srv.appendStreamNames); diff != "" {
			t.Errorf("AppendRows stream names mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("ambiguous send followed by row errors rotates stream before next batch", func(t *testing.T) {
		ctx := context.Background()
		srv := &scriptedCommittedStreamServer{
			onAppend: func(attempt int, _ *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
				switch attempt {
				case 1:
					return nil, status.Error(codes.Unavailable, "unknown outcome")
				case 2:
					return &storagepb.AppendRowsResponse{RowErrors: []*storagepb.RowError{{Index: 0}}}, nil
				default:
					return &storagepb.AppendRowsResponse{}, nil
				}
			},
		}

		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
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
		config.ProjectID = "test-proj"
		config.DatasetID = "test-ds"
		config.TableName = "agent_events"
		config.ExactlyOnceDelivery = true
		config.RetryConfig = RetryConfig{
			MaxRetries:   1,
			InitialDelay: time.Millisecond,
			MaxDelay:     5 * time.Millisecond,
			Multiplier:   1.0,
		}

		bp, err := NewBatchProcessor(ctx, writeClient, "", config)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}
		defer bp.Close()

		if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "LLM_REQUEST"}}); err == nil {
			t.Fatal("writeBatch(Batch A) error = nil, want row error")
		}
		if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "LLM_RESPONSE"}}); err != nil {
			t.Fatalf("writeBatch(Batch B) error = %v, want nil", err)
		}

		srv.mu.Lock()
		defer srv.mu.Unlock()
		wantStreams := []string{
			"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-1",
			"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-1",
			"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-2",
		}
		if diff := cmp.Diff(wantStreams, srv.appendStreamNames); diff != "" {
			t.Errorf("AppendRows stream names mismatch (-want +got):\n%s", diff)
		}
		wantOffsets := []int64{0, 0, 0}
		if diff := cmp.Diff(wantOffsets, srv.appendOffsets); diff != "" {
			t.Errorf("AppendRows offsets mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestBatchProcessor_ExactlyOnceDelivery_FinalizePreservesCallerContextMetadata(t *testing.T) {
	baseCtx := metadata.AppendToOutgoingContext(context.Background(), "x-test-caller", "preserve-me")
	callerCtx, callerCancel := context.WithCancel(baseCtx)

	srv := &scriptedCommittedStreamServer{
		onAppend: func(_ int, _ *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
			return &storagepb.AppendRowsResponse{}, nil
		},
	}

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	gSrv := grpc.NewServer()
	storagepb.RegisterBigQueryWriteServer(gSrv, srv)
	go func() { _ = gSrv.Serve(lis) }()
	t.Cleanup(gSrv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	writeClient, err := bqstorage.NewBigQueryWriteClient(baseCtx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("write client: %v", err)
	}
	t.Cleanup(func() { _ = writeClient.Close() })

	config := DefaultConfig()
	config.ProjectID = "test-proj"
	config.DatasetID = "test-ds"
	config.TableName = "agent_events"
	config.ExactlyOnceDelivery = true

	bp, err := NewBatchProcessor(callerCtx, writeClient, "", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor() error = %v", err)
	}

	// Cancel the caller context before Close to verify context.WithoutCancel(b.ctx)
	// detaches cancellation while preserving outgoing metadata.
	callerCancel()
	bp.Close()

	srv.mu.Lock()
	if len(srv.finalizeReqs) != 1 {
		srv.mu.Unlock()
		t.Fatalf("FinalizeWriteStream calls = %d, want 1", len(srv.finalizeReqs))
	}
	if got := srv.finalizeMetadata[0].Get("x-test-caller"); !slices.Equal(got, []string{"preserve-me"}) {
		t.Errorf("FinalizeWriteStream incoming metadata x-test-caller = %v, want [preserve-me]", got)
	}
	srv.mu.Unlock()

	t.Run("stalled finalizer honors small ShutdownTimeout and remaining drain budget", func(t *testing.T) {
		srv.mu.Lock()
		srv.onFinalize = func(ctx context.Context, _ *storagepb.FinalizeWriteStreamRequest) (*storagepb.FinalizeWriteStreamResponse, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		}
		srv.mu.Unlock()

		shortCfg := DefaultConfig()
		shortCfg.ProjectID = "test-proj"
		shortCfg.DatasetID = "test-ds"
		shortCfg.TableName = "agent_events"
		shortCfg.ExactlyOnceDelivery = true
		shortCfg.ShutdownTimeout = 5 * time.Millisecond

		bpShort, err := NewBatchProcessor(context.Background(), writeClient, "", shortCfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}

		start := time.Now()
		bpShort.Close()
		elapsed := time.Since(start)
		if elapsed >= 50*time.Millisecond {
			t.Errorf("Close() with stalled finalizer and ShutdownTimeout=5ms took %v, want < 50ms", elapsed)
		}

		// Verify finalize receives only the remaining ShutdownTimeout budget after
		// draining queued rows, not a fresh ShutdownTimeout.
		srv.mu.Lock()
		srv.onAppend = func(_ int, _ *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
			time.Sleep(40 * time.Millisecond)
			return &storagepb.AppendRowsResponse{}, nil
		}
		srv.mu.Unlock()

		remCfg := DefaultConfig()
		remCfg.ProjectID = "test-proj"
		remCfg.DatasetID = "test-ds"
		remCfg.TableName = "agent_events"
		remCfg.ExactlyOnceDelivery = true
		remCfg.BatchFlushIntv = time.Hour
		remCfg.ShutdownTimeout = 60 * time.Millisecond

		bpRem, err := NewBatchProcessor(context.Background(), writeClient, "", remCfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}
		bpRem.Start()
		bpRem.Append(map[string]any{"event_type": "DRAIN_BEFORE_FINALIZE"})

		remStart := time.Now()
		bpRem.Close()
		remElapsed := time.Since(remStart)
		if remElapsed >= 95*time.Millisecond {
			t.Errorf("Close() with 40ms drain + stalled finalizer and ShutdownTimeout=60ms took %v, want < 95ms (remaining budget only)", remElapsed)
		}

		// Verify FinalizeWriteStream is skipped and logged when drain exhausts ShutdownTimeout.
		srv.mu.Lock()
		finalizeCountBefore := len(srv.finalizeReqs)
		srv.onAppend = func(_ int, _ *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
			time.Sleep(80 * time.Millisecond)
			return nil, status.Error(codes.DeadlineExceeded, "drain timed out")
		}
		srv.mu.Unlock()

		exhaustLog := &recordingLogger{}
		exhaustCfg := DefaultConfig()
		exhaustCfg.ProjectID = "test-proj"
		exhaustCfg.DatasetID = "test-ds"
		exhaustCfg.TableName = "agent_events"
		exhaustCfg.ExactlyOnceDelivery = true
		exhaustCfg.BatchFlushIntv = time.Hour
		exhaustCfg.ShutdownTimeout = 20 * time.Millisecond
		exhaustCfg.RetryConfig.MaxRetries = 0
		exhaustCfg.Logger = exhaustLog

		bpExhaust, err := NewBatchProcessor(context.Background(), writeClient, "", exhaustCfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}
		bpExhaust.Start()
		bpExhaust.Append(map[string]any{"event_type": "EXHAUST_DRAIN_BUDGET"})
		bpExhaust.Close()

		srv.mu.Lock()
		finalizeCountAfter := len(srv.finalizeReqs)
		srv.mu.Unlock()
		if finalizeCountAfter != finalizeCountBefore {
			t.Errorf("FinalizeWriteStream calls after exhausted drain = %d, want %d (must skip FinalizeWriteStream when writeLeft <= 0)", finalizeCountAfter, finalizeCountBefore)
		}
		if !exhaustLog.contains("skipping FinalizeWriteStream") {
			t.Errorf("expected skipping FinalizeWriteStream log, got %q", exhaustLog.String())
		}
	})
}

func TestBatchProcessor_ExactlyOnceDelivery_B1AmbiguousExhaustedThenB2AmbiguousDoesNotLoseBatch(t *testing.T) {
	ctx := context.Background()

	// Track per-stream committed row count to model real BigQuery COMMITTED stream offset semantics:
	// - Attempt 1 (B1, 2 rows, committed-1 @ offset 0): server commits B1 (committed-1 len = 2), then drops connection (Unavailable).
	// - Attempt 2 (B1 retry, 2 rows, committed-1 @ offset 0): fails with Unavailable before server processes it -> B1 exhausts MaxRetries=1.
	// - Attempt 3 (B2, 3 rows): because B1 ended with an unconfirmed ambiguous send, streamWriter rotates to committed-2 @ offset 0!
	//   Even if B2's first attempt on committed-2 commits (committed-2 len = 3) and drops connection (Unavailable),
	// - Attempt 4 (B2 retry, 3 rows, committed-2 @ offset 0): server sees offset 0 < committed-2 len (3) and returns AlreadyExists,
	//   which confirms B2's own commit on committed-2 (not B1's commit on committed-1) and advances streamOffset to 3!
	streamCommittedRows := make(map[string]int64)
	srv := &scriptedCommittedStreamServer{
		onAppend: func(attempt int, req *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
			streamName := req.GetWriteStream()
			offset := req.GetOffset().GetValue()
			rows := req.GetArrowRows().GetRows().GetRowCount()

			switch attempt {
			case 1:
				// B1 attempt 1: server commits at offset 0 on committed-1, then connection drops before response arrives.
				streamCommittedRows[streamName] = offset + rows
				return nil, status.Error(codes.Unavailable, "connection lost after server commit of B1")
			case 2:
				// B1 attempt 2 (retry): transient failure before reaching offset check; B1 exhausts retries.
				return nil, status.Error(codes.Unavailable, "transient network failure on B1 retry")
			case 3:
				// B2 attempt 1: must arrive on committed-2 at offset 0; server commits B2, then connection drops.
				if offset < streamCommittedRows[streamName] {
					return &storagepb.AppendRowsResponse{
						Response: &storagepb.AppendRowsResponse_Error{
							Error: &statuspb.Status{
								Code:    int32(codes.AlreadyExists),
								Message: "offset already committed",
							},
						},
					}, nil
				}
				streamCommittedRows[streamName] = offset + rows
				return nil, status.Error(codes.Unavailable, "connection lost after server commit of B2")
			default:
				if offset < streamCommittedRows[streamName] {
					return &storagepb.AppendRowsResponse{
						Response: &storagepb.AppendRowsResponse_Error{
							Error: &statuspb.Status{
								Code:    int32(codes.AlreadyExists),
								Message: "offset already committed",
							},
						},
					}, nil
				}
				streamCommittedRows[streamName] = offset + rows
				return &storagepb.AppendRowsResponse{}, nil
			}
		},
	}

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
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
	config.ProjectID = "test-proj"
	config.DatasetID = "test-ds"
	config.TableName = "agent_events"
	config.ExactlyOnceDelivery = true
	config.RetryConfig = RetryConfig{
		MaxRetries:   1,
		InitialDelay: time.Millisecond,
		MaxDelay:     5 * time.Millisecond,
		Multiplier:   1.0,
	}

	bp, err := NewBatchProcessor(ctx, writeClient, "", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor() error = %v", err)
	}
	defer bp.Close()

	// B1 (2 rows): exhausts retries after an ambiguous send.
	if err := bp.writeBatch(ctx, []map[string]any{
		{"event_type": "B1_ROW_1"},
		{"event_type": "B1_ROW_2"},
	}); err == nil {
		t.Fatal("writeBatch(B1) error = nil, want retry exhaustion")
	}

	// B2 (3 rows): rotates to committed-2 before sending so B2 is committed on committed-2 (not mistaken for B1 on committed-1).
	if err := bp.writeBatch(ctx, []map[string]any{
		{"event_type": "B2_ROW_1"},
		{"event_type": "B2_ROW_2"},
		{"event_type": "B2_ROW_3"},
	}); err != nil {
		t.Fatalf("writeBatch(B2) error = %v, want nil", err)
	}

	// B3 (1 row): must append at offset 3 on committed-2.
	if err := bp.writeBatch(ctx, []map[string]any{
		{"event_type": "B3_ROW_1"},
	}); err != nil {
		t.Fatalf("writeBatch(B3) error = %v, want nil", err)
	}

	srv.mu.Lock()
	defer srv.mu.Unlock()
	if got := streamCommittedRows["projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-2"]; got != 4 {
		t.Errorf("committed-2 total rows = %d, want 4 (3 from B2 + 1 from B3)", got)
	}
	wantOffsets := []int64{0, 0, 0, 0, 3}
	if diff := cmp.Diff(wantOffsets, srv.appendOffsets); diff != "" {
		t.Errorf("AppendRows offsets mismatch (-want +got):\n%s", diff)
	}
}

func TestBatchProcessor_ExactlyOnceDelivery_RotationCooldownAndGuards(t *testing.T) {
	ctx := context.Background()

	t.Run("rejects non-empty or _default streamName when ExactlyOnceDelivery is enabled", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.ExactlyOnceDelivery = true
		if _, err := NewBatchProcessor(ctx, nil, "projects/p/datasets/d/tables/t/streams/_default", cfg); err == nil {
			t.Fatal("NewBatchProcessor() with _default streamName and ExactlyOnceDelivery=true succeeded, want error")
		}
	})

	t.Run("Send failure followed by AlreadyExists rotates stream instead of advancing offset", func(t *testing.T) {
		srv := &scriptedCommittedStreamServer{
			onAppend: func(_ int, _ *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
				return &storagepb.AppendRowsResponse{}, nil
			},
		}
		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
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

		cfg := DefaultConfig()
		cfg.ProjectID = "test-proj"
		cfg.DatasetID = "test-ds"
		cfg.TableName = "agent_events"
		cfg.ExactlyOnceDelivery = true
		cfg.RetryConfig = RetryConfig{
			MaxRetries:   1,
			InitialDelay: time.Millisecond,
			MaxDelay:     5 * time.Millisecond,
			Multiplier:   1.0,
		}

		bp, err := NewBatchProcessor(ctx, writeClient, "", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}
		defer bp.Close()

		// Inject a mockStream whose Send fails with io.EOF and whose Recv surfaces AlreadyExists (sendSucceeded=false).
		bp.streamWriter.stream = &mockStream{
			sendErr: io.EOF,
			recvErr: status.Error(codes.AlreadyExists, "offset 0 already exists on send failure"),
		}

		if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "ROW_1"}, {"event_type": "ROW_2"}}); err != nil {
			t.Fatalf("writeBatch() error = %v, want nil after rotating to committed-2", err)
		}

		srv.mu.Lock()
		defer srv.mu.Unlock()
		if len(srv.createReqs) != 2 {
			t.Fatalf("CreateWriteStream calls = %d, want 2 (initial + rotation after send-path AlreadyExists)", len(srv.createReqs))
		}
		if diff := cmp.Diff([]string{"projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-2"}, srv.appendStreamNames); diff != "" {
			t.Errorf("AppendRows streams mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("Send failure with AlreadyExists after ambiguous send confirms batch and advances offset, and rotation errors on Send or in-band response return immediately", func(t *testing.T) {
		var (
			failCreate        atomic.Bool
			interceptSendMode atomic.Int32
			streamOpenCount   atomic.Int32
		)
		srv := &scriptedCommittedStreamServer{
			onAppend: func(_ int, _ *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
				return &storagepb.AppendRowsResponse{
					Response: &storagepb.AppendRowsResponse_Error{
						Error: &statuspb.Status{
							Code:    int32(codes.OutOfRange),
							Message: "in-band offset out of range",
						},
					},
				}, nil
			},
		}
		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		gSrv := grpc.NewServer(grpc.UnaryInterceptor(func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
			if strings.HasSuffix(info.FullMethod, "/CreateWriteStream") && failCreate.Load() {
				return nil, status.Error(codes.PermissionDenied, "rotation failed")
			}
			return handler(ctx, req)
		}))
		storagepb.RegisterBigQueryWriteServer(gSrv, srv)
		go func() { _ = gSrv.Serve(lis) }()
		t.Cleanup(gSrv.Stop)

		conn, err := grpc.NewClient(
			lis.Addr().String(),
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithStreamInterceptor(func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
				cs, err := streamer(ctx, desc, cc, method, opts...)
				if err != nil || interceptSendMode.Load() == 0 {
					return cs, err
				}
				openNum := streamOpenCount.Add(1)
				return &scriptedClientStream{ClientStream: cs, openNum: openNum}, nil
			}),
		)
		if err != nil {
			t.Fatalf("dial: %v", err)
		}
		t.Cleanup(func() { _ = conn.Close() })

		writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
		if err != nil {
			t.Fatalf("write client: %v", err)
		}
		t.Cleanup(func() { _ = writeClient.Close() })

		cfg := DefaultConfig()
		cfg.ProjectID = "test-proj"
		cfg.DatasetID = "test-ds"
		cfg.TableName = "agent_events"
		cfg.ExactlyOnceDelivery = true
		cfg.RetryConfig = RetryConfig{
			MaxRetries:   2,
			InitialDelay: time.Millisecond,
			MaxDelay:     2 * time.Millisecond,
			Multiplier:   1.0,
		}

		bp, err := NewBatchProcessor(ctx, writeClient, "", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}
		defer bp.Close()

		// 1. Send-path AlreadyExists confirmation after a prior ambiguous send:
		// Stream #1 Send succeeds and Recv returns Unavailable (sets hadAmbiguousSend = true).
		// Stream #2 Send returns io.EOF and drain Recv surfaces AlreadyExists -> confirms batch and advances streamOffset by 2.
		interceptSendMode.Store(1)
		if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "R1"}, {"event_type": "R2"}}); err != nil {
			t.Fatalf("writeBatch() after Send-path AlreadyExists confirmation error = %v, want nil", err)
		}
		if got := bp.streamWriter.streamOffset.Load(); got != 2 {
			t.Errorf("streamOffset after Send-path AlreadyExists confirmation = %d, want 2", got)
		}
		interceptSendMode.Store(0)
		bp.streamWriter.resetStream()

		// 2. Send-path rotation error return: Send fails with io.EOF + OutOfRange while CreateWriteStream fails.
		failCreate.Store(true)
		bp.streamWriter.rotateRetryAt = time.Time{}
		bp.streamWriter.freshlyRotated = false
		bp.streamWriter.stream = &mockStream{
			sendErr: io.EOF,
			recvErr: status.Error(codes.OutOfRange, "send-path out of range"),
		}
		if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "R3"}}); err == nil || !strings.Contains(err.Error(), "failed to rotate committed write stream") {
			t.Fatalf("writeBatch() on Send-path rotation failure = %v, want rotation error", err)
		}

		// 3. In-band response error path rotation error return: Recv returns in-band OutOfRange while CreateWriteStream fails.
		bp.streamWriter.offsetDesynced = false
		bp.streamWriter.rotateRetryAt = time.Time{}
		bp.streamWriter.freshlyRotated = false
		if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "R4"}}); err == nil || !strings.Contains(err.Error(), "failed to rotate committed write stream") {
			t.Fatalf("writeBatch() on in-band rotation failure = %v, want rotation error", err)
		}
	})

	t.Run("rotates at most once per batch, attaches x-goog-request-params header, and enforces cooldown when rotation or freshly rotated stream fails", func(t *testing.T) {
		var createCalls atomic.Int32
		var failCreate atomic.Bool
		var initialCreateHasDeadline atomic.Bool
		var appendReqParamsMu sync.Mutex
		var appendReqParams []string

		srv := &scriptedCommittedStreamServer{
			onAppend: func(_ int, req *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
				switch {
				case strings.HasSuffix(req.GetWriteStream(), "/committed-1") || strings.HasSuffix(req.GetWriteStream(), "/committed-2"):
					// committed-1 and freshly rotated committed-2 both fail with OutOfRange.
					return nil, status.Error(codes.OutOfRange, "offset out of range")
				default:
					return &storagepb.AppendRowsResponse{}, nil
				}
			},
		}

		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		gSrv := grpc.NewServer(
			grpc.UnaryInterceptor(func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
				if strings.HasSuffix(info.FullMethod, "/CreateWriteStream") {
					callNum := createCalls.Add(1)
					if callNum == 1 {
						_, hasDeadline := ctx.Deadline()
						initialCreateHasDeadline.Store(hasDeadline)
					}
					if failCreate.Load() {
						return nil, status.Error(codes.PermissionDenied, "quota or permission failure")
					}
				}
				return handler(ctx, req)
			}),
			grpc.StreamInterceptor(func(srvObj any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
				if strings.HasSuffix(info.FullMethod, "/AppendRows") {
					md, _ := metadata.FromIncomingContext(ss.Context())
					appendReqParamsMu.Lock()
					appendReqParams = append(appendReqParams, md.Get("x-goog-request-params")...)
					appendReqParamsMu.Unlock()
				}
				return handler(srvObj, ss)
			}),
		)
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

		cfg := DefaultConfig()
		cfg.ProjectID = "test-proj"
		cfg.DatasetID = "test-ds"
		cfg.TableName = "agent_events"
		cfg.ExactlyOnceDelivery = true
		cfg.RetryConfig = RetryConfig{
			MaxRetries:   3,
			InitialDelay: time.Millisecond,
			MaxDelay:     5 * time.Millisecond,
			Multiplier:   1.0,
		}

		bp, err := NewBatchProcessor(ctx, writeClient, "", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}
		defer bp.Close()

		if !initialCreateHasDeadline.Load() {
			t.Error("initial CreateWriteStream context had no deadline, want bounded by defaultRotateTimeout")
		}

		now := time.Date(2026, 10, 11, 0, 0, 0, 0, time.UTC)
		bp.streamWriter.now = func() time.Time { return now }
		bp.streamWriter.rotateTimeout = 30 * time.Second

		// Batch 1: attempt 1 on committed-1 gets OutOfRange and rotates once to committed-2 (createCalls: 1 -> 2);
		// attempt 2 on freshly rotated committed-2 gets OutOfRange, aborts without a second rotation, and arms the 30s cooldown.
		if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "B1"}}); err == nil {
			t.Fatal("writeBatch(B1) error = nil, want second-rotation abort error")
		}
		if got := createCalls.Load(); got != 2 {
			t.Fatalf("CreateWriteStream calls after B1 = %d, want 2 (1 initial + at most 1 rotation per batch)", got)
		}

		// Batch 1b within the 30s cooldown armed by freshly rotated committed-2 failing on its first append:
		// must fail fast without calling CreateWriteStream (createCalls stays 2).
		now = now.Add(10 * time.Second)
		if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "B1b"}}); err == nil {
			t.Fatal("writeBatch(B1b) error = nil, want cooldown error after freshly rotated stream failed")
		}
		if got := createCalls.Load(); got != 2 {
			t.Fatalf("CreateWriteStream calls during freshlyRotated cooldown = %d, want 2", got)
		}

		// Advance clock past the first 30s cooldown and configure CreateWriteStream to fail:
		// Batch 2 attempts rotation, fails CreateWriteStream (createCalls: 2 -> 3), and arms a new 30s cooldown.
		now = now.Add(25 * time.Second)
		failCreate.Store(true)
		if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "B2"}}); err == nil {
			t.Fatal("writeBatch(B2) error = nil, want CreateWriteStream error")
		}
		if got := createCalls.Load(); got != 3 {
			t.Fatalf("CreateWriteStream calls after B2 = %d, want 3", got)
		}

		// Batch 3 within cooldown window (10s < 30s): fails fast without calling CreateWriteStream (createCalls stays 3).
		now = now.Add(10 * time.Second)
		if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "B3"}}); err == nil {
			t.Fatal("writeBatch(B3) error = nil, want cooldown error")
		}
		if got := createCalls.Load(); got != 3 {
			t.Fatalf("CreateWriteStream calls during cooldown = %d, want 3 (no extra RPC during cooldown)", got)
		}

		// Advance clock past 30s cooldown and restore CreateWriteStream -> Batch 4 rotates to committed-3 and succeeds.
		now = now.Add(25 * time.Second)
		failCreate.Store(false)
		if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "B4"}}); err != nil {
			t.Fatalf("writeBatch(B4) after cooldown error = %v, want nil", err)
		}
		if got := createCalls.Load(); got != 4 {
			t.Fatalf("CreateWriteStream calls after cooldown expiry = %d, want 4", got)
		}

		appendReqParamsMu.Lock()
		wantParams := []string{
			"write_stream=projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-1",
			"write_stream=projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-2",
			"write_stream=projects/test-proj/datasets/test-ds/tables/agent_events/streams/committed-3",
		}
		if diff := cmp.Diff(wantParams, appendReqParams); diff != "" {
			t.Errorf("AppendRows x-goog-request-params mismatch (-want +got):\n%s", diff)
		}
		appendReqParamsMu.Unlock()
	})

	t.Run("hanging CreateWriteStream during rotation times out at rotateTimeout", func(t *testing.T) {
		var createCalls atomic.Int32
		srv := &scriptedCommittedStreamServer{
			onAppend: func(_ int, _ *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
				return nil, status.Error(codes.NotFound, "stream expired")
			},
		}

		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		gSrv := grpc.NewServer(grpc.UnaryInterceptor(func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
			if strings.HasSuffix(info.FullMethod, "/CreateWriteStream") {
				if createCalls.Add(1) >= 2 {
					<-ctx.Done()
					return nil, status.FromContextError(ctx.Err()).Err()
				}
			}
			return handler(ctx, req)
		}))
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

		cfg := DefaultConfig()
		cfg.ProjectID = "test-proj"
		cfg.DatasetID = "test-ds"
		cfg.TableName = "agent_events"
		cfg.ExactlyOnceDelivery = true
		cfg.RetryConfig.MaxRetries = 0

		bp, err := NewBatchProcessor(ctx, writeClient, "", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}
		defer bp.Close()

		bp.streamWriter.rotateTimeout = 15 * time.Millisecond
		start := time.Now()
		err = bp.writeBatch(ctx, []map[string]any{{"event_type": "ROTATE_TIMEOUT"}})
		elapsed := time.Since(start)
		if err == nil {
			t.Fatal("writeBatch() with hanging rotation CreateWriteStream succeeded, want timeout error")
		}
		if elapsed >= 150*time.Millisecond {
			t.Errorf("writeBatch() with rotateTimeout=15ms took %v, want < 150ms", elapsed)
		}
	})
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

func TestBatchProcessor_ClosePreservesInflightAndQueuedBatchesUnderStalledUploads(t *testing.T) {
	newWriteClient := func(t *testing.T, srv storagepb.BigQueryWriteServer) *bqstorage.BigQueryWriteClient {
		t.Helper()
		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		gSrv := grpc.NewServer()
		storagepb.RegisterBigQueryWriteServer(gSrv, srv)
		go func() { _ = gSrv.Serve(lis) }()
		t.Cleanup(gSrv.Stop)

		conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("dial: %v", err)
		}
		t.Cleanup(func() { _ = conn.Close() })

		wc, err := bqstorage.NewBigQueryWriteClient(context.Background(), option.WithGRPCConn(conn))
		if err != nil {
			t.Fatalf("write client: %v", err)
		}
		t.Cleanup(func() { _ = wc.Close() })
		return wc
	}

	t.Run("inflight size-triggered flush with stalled upload does not lose row on Close", func(t *testing.T) {
		requests := make(chan *storagepb.AppendRowsRequest, 8)
		wc := newWriteClient(t, &fakeBigQueryWriteServer{requests: requests})

		uploadStarted := make(chan struct{})
		off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
			select {
			case <-uploadStarted:
			default:
				close(uploadStarted)
			}
			<-ctx.Done()
			return ctx.Err()
		}}

		cfg := DefaultConfig()
		cfg.BatchSize = 1
		cfg.BatchFlushIntv = time.Hour
		cfg.ShutdownTimeout = 500 * time.Millisecond
		cfg.GCSUploadTimeout = 5 * time.Second
		cfg.GCSOffloader = off
		cfg.Logger = &recordingLogger{}

		bp, err := NewBatchProcessor(context.Background(), wc, "projects/p/datasets/d/tables/t/_default", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor: %v", err)
		}
		bp.Start()

		// Open the stream on a short-lived context that is canceled before Close's
		// final write, verifying streamWriter reopens a canceled stream.
		staleCtx, cancelStale := context.WithCancel(context.Background())
		if err := bp.writeBatch(staleCtx, []map[string]any{{"event_type": "WARMUP"}}); err != nil {
			t.Fatalf("warmup writeBatch: %v", err)
		}
		<-requests
		cancelStale()

		row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10}, imageContent(1, 2, 3))
		row["event_type"] = EventLLMRequest
		bp.Append(row)

		select {
		case <-uploadStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("timed out waiting for in-flight upload to start")
		}

		start := time.Now()
		bp.Close()
		// Close cancels the upload that is in flight. It does not wait for the
		// 5s that the upload could still take.
		if elapsed := time.Since(start); elapsed >= 2*time.Second {
			t.Errorf("Close() took %v, want < 2s: it has to cancel the in-flight upload instead of waiting for GCSUploadTimeout", elapsed)
		}

		select {
		case req := <-requests:
			if got := req.GetArrowRows().GetRows().GetRowCount(); got != 1 {
				t.Errorf("persisted row count = %d, want 1", got)
			}
		case <-time.After(2 * time.Second):
			t.Fatal("in-flight batch was dropped when Close canceled the stalled upload")
		}
	})

	t.Run("shutdown flush caps upload deadline to half of ShutdownTimeout so write succeeds", func(t *testing.T) {
		requests := make(chan *storagepb.AppendRowsRequest, 8)
		wc := newWriteClient(t, &fakeBigQueryWriteServer{requests: requests})

		off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
			<-ctx.Done()
			return ctx.Err()
		}}

		cfg := DefaultConfig()
		cfg.BatchSize = 100
		cfg.BatchFlushIntv = time.Hour
		cfg.ShutdownTimeout = 300 * time.Millisecond
		cfg.GCSUploadTimeout = 10 * time.Second
		cfg.GCSOffloader = off
		cfg.Logger = &recordingLogger{}

		bp, err := NewBatchProcessor(context.Background(), wc, "projects/p/datasets/d/tables/t/_default", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor: %v", err)
		}
		bp.Start()

		row1 := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10}, imageContent(9))
		row1["event_type"] = EventLLMRequest
		row2 := map[string]any{"event_type": EventUserMessageReceived}
		bp.Append(row1)
		bp.Append(row2)

		start := time.Now()
		bp.Close()
		elapsed := time.Since(start)
		if elapsed >= cfg.ShutdownTimeout {
			t.Errorf("Close() took %v, want < %v (uploads should use at most half of ShutdownTimeout)", elapsed, cfg.ShutdownTimeout)
		}

		select {
		case req := <-requests:
			if got := req.GetArrowRows().GetRows().GetRowCount(); got != 2 {
				t.Errorf("persisted row count = %d, want 2", got)
			}
		default:
			t.Fatal("queued rows were lost on Close when upload stalled")
		}
	})

	t.Run("Close shares a single ShutdownTimeout budget across multiple queued batches", func(t *testing.T) {
		wc := newWriteClient(t, &stallingBigQueryWriteServer{})

		cfg := DefaultConfig()
		cfg.BatchSize = 1
		cfg.BatchFlushIntv = time.Hour
		cfg.ShutdownTimeout = 80 * time.Millisecond
		cfg.RetryConfig.MaxRetries = 0
		cfg.GCSOffloader = &fakeOffloader{}
		cfg.Logger = &recordingLogger{}

		bp, err := NewBatchProcessor(context.Background(), wc, "projects/p/datasets/d/tables/t/_default", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor: %v", err)
		}
		bp.Start()

		// Send rows straight into the queue so no background flush goroutine is
		// spawned and Close drains all five batches deterministically.
		for range 5 {
			row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(1, 2, 3, 4))
			row["event_type"] = EventLLMRequest
			bp.queue <- row
		}

		start := time.Now()
		bp.Close()
		elapsed := time.Since(start)
		// Five 1-row batches renewing an 80ms budget per batch would take >= 400ms.
		if elapsed >= 250*time.Millisecond {
			t.Errorf("Close() took %v across 5 stalled batches, want < 250ms (single shared ShutdownTimeout)", elapsed)
		}
		if got := bp.pendingUploadBytes.Load(); got != 0 {
			t.Errorf("pendingUploadBytes after Close = %d, want 0", got)
		}
		wantDrops := DropStats{
			AppendFailureDrops: 1,
			ShutdownDrops:      4,
			TotalDrops:         5,
		}
		if diff := cmp.Diff(wantDrops, bp.DropStats()); diff != "" {
			t.Errorf("DropStats() after stalled Close mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("Close starts its drain budget after the in-flight write finishes", func(t *testing.T) {
		requests := make(chan *storagepb.AppendRowsRequest, 8)
		received := make(chan struct{}, 8)
		// A write takes 300ms. That fits in the 450ms ShutdownTimeout once but
		// not twice, as it would have to if waiting for the in-flight write
		// used up the budget that the queued row is written under.
		wc := newWriteClient(t, &delayedBigQueryWriteServer{
			delay:    300 * time.Millisecond,
			requests: requests,
			received: received,
		})

		logger := &recordingLogger{}
		cfg := DefaultConfig()
		cfg.BatchSize = 1
		cfg.BatchFlushIntv = time.Hour
		cfg.ShutdownTimeout = 450 * time.Millisecond
		cfg.RetryConfig.MaxRetries = 0
		cfg.Logger = logger

		bp, err := NewBatchProcessor(context.Background(), wc, "projects/p/datasets/d/tables/t/_default", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor: %v", err)
		}
		bp.Start()

		bp.Append(map[string]any{"event_type": EventUserMessageReceived}) // starts a background flush
		select {
		case <-received:
		case <-time.After(2 * time.Second):
			t.Fatal("timed out waiting for the background flush to start its write")
		}
		bp.Append(map[string]any{"event_type": EventLLMRequest}) // waits in the queue behind that write

		bp.Close()

		for i := range 2 {
			select {
			case req := <-requests:
				if got := req.GetArrowRows().GetRows().GetRowCount(); got != 1 {
					t.Errorf("request %d has %d rows, want 1", i, got)
				}
			default:
				t.Fatalf("Close persisted %d of 2 batches: the queued row was dropped after waiting for the in-flight write", i)
			}
		}
		if logger.contains("Failed to write batch") {
			t.Errorf("unexpected write failure in the log:\n%s", logger)
		}
	})

	t.Run("Close logs and releases the batches it cannot write within its budget", func(t *testing.T) {
		wc := newWriteClient(t, &stallingBigQueryWriteServer{})

		logger := &recordingLogger{}
		cfg := DefaultConfig()
		cfg.BatchSize = 1
		cfg.BatchFlushIntv = time.Hour
		cfg.ShutdownTimeout = 80 * time.Millisecond
		cfg.RetryConfig.MaxRetries = 0
		cfg.GCSOffloader = &fakeOffloader{}
		cfg.Logger = logger

		bp, err := NewBatchProcessor(context.Background(), wc, "projects/p/datasets/d/tables/t/_default", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor: %v", err)
		}
		bp.Start()

		// The rows go straight into the queue, so that no background flush is
		// running and Close drains all three batches.
		for range 3 {
			row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(1, 2, 3, 4))
			row["event_type"] = EventLLMRequest
			bp.queue <- row
		}
		if got := bp.pendingUploadBytes.Load(); got != 12 {
			t.Fatalf("pendingUploadBytes before Close = %d, want 12", got)
		}

		bp.Close()

		// The first batch uses up the budget on the stalled write, so at least
		// the other two are dropped.
		const dropped = "Failed to write batch of 1 events: context deadline exceeded"
		if got := strings.Count(logger.String(), dropped); got < 2 {
			t.Errorf("log has %d lines with %q, want at least 2:\n%s", got, dropped, logger)
		}
		if got := bp.pendingUploadBytes.Load(); got != 0 {
			t.Errorf("pendingUploadBytes after Close = %d, want 0", got)
		}
	})

	t.Run("background flush starts append timeout after uploads finish", func(t *testing.T) {
		requests := make(chan *storagepb.AppendRowsRequest, 8)
		wc := newWriteClient(t, &delayedBigQueryWriteServer{
			delay:    150 * time.Millisecond,
			requests: requests,
		})

		off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
			select {
			case <-time.After(300 * time.Millisecond):
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}}

		cfg := DefaultConfig()
		cfg.BatchSize = 2
		cfg.BatchFlushIntv = time.Hour
		cfg.ShutdownTimeout = 400 * time.Millisecond
		cfg.GCSUploadTimeout = 400 * time.Millisecond
		cfg.RetryConfig.MaxRetries = 0
		cfg.GCSOffloader = off
		cfg.Logger = &recordingLogger{}

		bp, err := NewBatchProcessor(context.Background(), wc, "projects/p/datasets/d/tables/t/_default", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor: %v", err)
		}
		bp.Start()
		defer bp.Close()

		row1 := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(1, 2, 3))
		row1["event_type"] = EventLLMRequest
		row2 := map[string]any{"event_type": EventUserMessageReceived}
		bp.Append(row1)
		bp.Append(row2) // triggers background flush: 300ms upload + 150ms write > 400ms if shared, but succeeds when write timeout starts after upload

		select {
		case req := <-requests:
			if got := req.GetArrowRows().GetRows().GetRowCount(); got != 2 {
				t.Errorf("persisted row count = %d, want 2", got)
			}
		case <-time.After(2 * time.Second):
			t.Fatal("background flush lost batch when upload used 300ms of 400ms GCSUploadTimeout before 150ms BigQuery append")
		}
	})

	t.Run("Close writes every queued batch when Cloud Storage stalls", func(t *testing.T) {
		const batches = 5
		requests := make(chan *storagepb.AppendRowsRequest, 16)
		// The writes take 5 x 80ms = 400ms. They fit in the 600ms of
		// ShutdownTimeout, but not in the 300ms that would be left of it once
		// the uploads of the first batch had waited for their 300ms, which is
		// half of it.
		wc := newWriteClient(t, &delayedBigQueryWriteServer{delay: 80 * time.Millisecond, requests: requests})

		off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
			<-ctx.Done()
			return ctx.Err()
		}}
		logger := &recordingLogger{}
		cfg := DefaultConfig()
		cfg.BatchSize = 1
		cfg.BatchFlushIntv = time.Hour
		cfg.ShutdownTimeout = 600 * time.Millisecond
		cfg.RetryConfig.MaxRetries = 0
		cfg.GCSOffloader = off
		cfg.Logger = logger

		bp, err := NewBatchProcessor(context.Background(), wc, "projects/p/datasets/d/tables/t/_default", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor: %v", err)
		}
		bp.Start()

		// The rows go straight into the queue, so that no background flush is
		// running and Close drains all of the batches.
		for range batches {
			row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(1, 2, 3, 4))
			row["event_type"] = EventLLMRequest
			bp.queue <- row
		}

		bp.Close()

		rows := collectRows(t, requests, func(rows []map[string]any) bool { return len(rows) >= batches })
		if len(rows) != batches {
			t.Errorf("Close wrote %d rows, want %d", len(rows), batches)
		}
		for i, row := range rows {
			if got := contentPartsOf(t, row)[0]["text"]; got != uploadFailedText {
				t.Errorf("row %d part text = %v, want %q", i, got, uploadFailedText)
			}
		}
		if got := len(off.uploads()); got != 1 {
			t.Errorf("the offloader was called %d times, want once: the batches after the first skip Cloud Storage", got)
		}
		if logger.contains("Failed to write batch") {
			t.Errorf("unexpected write failure in the log:\n%s", logger)
		}
		if got := bp.pendingUploadBytes.Load(); got != 0 {
			t.Errorf("pendingUploadBytes after Close = %d, want 0", got)
		}
	})

	t.Run("Close shares half a ShutdownTimeout between the uploads of all the queued batches", func(t *testing.T) {
		const batches = 16
		requests := make(chan *storagepb.AppendRowsRequest, 2*batches)
		wc := newWriteClient(t, &fakeBigQueryWriteServer{requests: requests})

		// Cloud Storage is slow, not stalled: every upload takes 50ms and
		// succeeds unless its context ends first. The uploads of the drain
		// have 400ms in all, which is half of ShutdownTimeout, so at most 8
		// of them can finish. A drain that gave each batch that long, or all
		// of ShutdownTimeout, would upload every one of the 16.
		off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
			select {
			case <-time.After(50 * time.Millisecond):
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}}
		logger := &recordingLogger{}
		cfg := DefaultConfig()
		cfg.BatchSize = 1
		cfg.BatchFlushIntv = time.Hour
		cfg.ShutdownTimeout = 800 * time.Millisecond
		cfg.RetryConfig.MaxRetries = 0
		cfg.GCSOffloader = off
		cfg.Logger = logger

		bp, err := NewBatchProcessor(context.Background(), wc, "projects/p/datasets/d/tables/t/_default", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor: %v", err)
		}
		bp.Start()

		// The rows go straight into the queue, so that no background flush is
		// running and Close drains all of the batches.
		for range batches {
			row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(1, 2, 3, 4))
			row["event_type"] = EventLLMRequest
			bp.queue <- row
		}

		bp.Close()

		rows := collectRows(t, requests, func(rows []map[string]any) bool { return len(rows) >= batches })
		if len(rows) != batches {
			t.Errorf("Close wrote %d rows, want %d", len(rows), batches)
		}
		var uploaded, failed int
		for _, row := range rows {
			part := contentPartsOf(t, row)[0]
			switch {
			case part["storage_mode"] == "GCS_REFERENCE":
				uploaded++
			case part["text"] == uploadFailedText:
				failed++
			}
		}
		if uploaded+failed != batches {
			t.Errorf("got %d uploaded and %d failed rows, want %d rows in all", uploaded, failed, batches)
		}
		if uploaded < 2 || uploaded > 8 {
			t.Errorf("%d of %d rows were uploaded, want between 2 and 8: the uploads of all the batches share half of ShutdownTimeout", uploaded, batches)
		}
		if logger.contains("Failed to write batch") {
			t.Errorf("unexpected write failure in the log:\n%s", logger)
		}
		if got := bp.pendingUploadBytes.Load(); got != 0 {
			t.Errorf("pendingUploadBytes after Close = %d, want 0", got)
		}
	})

	t.Run("Close uploads the queued rows after canceling the upload that was in flight", func(t *testing.T) {
		requests := make(chan *storagepb.AppendRowsRequest, 8)
		wc := newWriteClient(t, &fakeBigQueryWriteServer{requests: requests})

		// Cloud Storage is healthy: only the first upload is slow enough to
		// be still running when Close cancels it. That says nothing about
		// Cloud Storage, so the upload of the queued row must not be skipped.
		var calls atomic.Int32
		uploadStarted := make(chan struct{})
		off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
			if calls.Add(1) > 1 {
				return nil
			}
			close(uploadStarted)
			<-ctx.Done()
			return ctx.Err()
		}}
		cfg := DefaultConfig()
		cfg.BatchSize = 1
		cfg.BatchFlushIntv = time.Hour
		cfg.GCSOffloader = off
		cfg.Logger = &recordingLogger{}

		bp, err := NewBatchProcessor(context.Background(), wc, "projects/p/datasets/d/tables/t/_default", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor: %v", err)
		}
		bp.Start()

		inflight := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(1))
		inflight["event_type"] = EventLLMRequest
		bp.Append(inflight) // starts a background flush that waits for its upload
		select {
		case <-uploadStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("timed out waiting for the upload of the background flush to start")
		}
		queued := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(2))
		queued["event_type"] = EventLLMRequest
		bp.queue <- queued

		bp.Close()

		rows := collectRows(t, requests, func(rows []map[string]any) bool { return len(rows) >= 2 })
		var canceled, uploaded int
		for _, row := range rows {
			part := contentPartsOf(t, row)[0]
			switch {
			case part["storage_mode"] == "GCS_REFERENCE":
				uploaded++
			case part["text"] == uploadFailedText:
				canceled++
			}
		}
		if canceled != 1 || uploaded != 1 {
			t.Errorf("got %d rows whose upload was canceled and %d uploaded, want 1 and 1", canceled, uploaded)
		}
		if got := bp.pendingUploadBytes.Load(); got != 0 {
			t.Errorf("pendingUploadBytes after Close = %d, want 0", got)
		}
	})

	t.Run("run-end flush stops waiting for Cloud Storage once the uploads of one batch time out", func(t *testing.T) {
		const batches = 4
		requests := make(chan *storagepb.AppendRowsRequest, 16)
		wc := newWriteClient(t, &fakeBigQueryWriteServer{requests: requests})

		off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
			<-ctx.Done()
			return ctx.Err()
		}}
		logger := &recordingLogger{}
		cfg := DefaultConfig()
		cfg.BatchSize = 1
		cfg.BatchFlushIntv = time.Hour
		cfg.ShutdownTimeout = 800 * time.Millisecond
		cfg.GCSOffloader = off
		cfg.Logger = logger

		bp, err := NewBatchProcessor(context.Background(), wc, "projects/p/datasets/d/tables/t/_default", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor: %v", err)
		}
		bp.Start()
		defer bp.Close()

		for range batches {
			row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(1, 2, 3, 4))
			row["event_type"] = EventLLMRequest
			bp.queue <- row
		}

		start := time.Now()
		bp.flushRunEnd()
		elapsed := time.Since(start)

		// The uploads of the first batch wait for half of the ShutdownTimeout,
		// 400ms. If each batch waited for half of what was left, the flush
		// would take 400 + 200 + 100 + 50ms.
		if elapsed >= 600*time.Millisecond {
			t.Errorf("flushRunEnd() took %v, want < 600ms: only the first batch waits for Cloud Storage", elapsed)
		}
		if got := len(off.uploads()); got != 1 {
			t.Errorf("the offloader was called %d times, want once", got)
		}
		rows := collectRows(t, requests, func(rows []map[string]any) bool { return len(rows) >= batches })
		if len(rows) != batches {
			t.Errorf("flushRunEnd() wrote %d rows, want %d", len(rows), batches)
		}
		if logger.contains("Failed to write batch") {
			t.Errorf("unexpected write failure in the log:\n%s", logger)
		}
	})

	t.Run("run-end flush shares one ShutdownTimeout across the batches it writes", func(t *testing.T) {
		wc := newWriteClient(t, &stallingBigQueryWriteServer{})

		cfg := DefaultConfig()
		cfg.BatchSize = 1
		cfg.BatchFlushIntv = time.Hour
		cfg.ShutdownTimeout = 80 * time.Millisecond
		cfg.RetryConfig.MaxRetries = 0
		cfg.Logger = &recordingLogger{}

		bp, err := NewBatchProcessor(context.Background(), wc, "projects/p/datasets/d/tables/t/_default", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor: %v", err)
		}
		bp.Start()
		defer bp.Close()

		const batches = 5
		for range batches {
			bp.queue <- map[string]any{"event_type": EventUserMessageReceived}
		}

		start := time.Now()
		bp.flushRunEnd()
		elapsed := time.Since(start)

		// Five batches that each had an 80ms budget would take 400ms. The
		// stalled write of the first batch uses it up, and the others stay
		// queued for the next flush or for Close.
		if elapsed >= 250*time.Millisecond {
			t.Errorf("flushRunEnd() took %v across %d batches, want < 250ms (one shared ShutdownTimeout)", elapsed, batches)
		}
		if got := len(bp.queue); got != batches-1 {
			t.Errorf("%d rows are still queued, want %d: a flush that runs out of budget leaves the batches it did not write queued", got, batches-1)
		}
	})
}

func TestBatchProcessor_CloseTimeoutCountsAllLostRowsInDropStats(t *testing.T) {
	t.Run("Close timeout with BatchSize=5 and ShutdownTimeout=300ms over 66 events counts all 56 lost rows", func(t *testing.T) {
		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		requests := make(chan *storagepb.AppendRowsRequest, 16)
		srv := &delayedBigQueryWriteServer{
			delay:         40 * time.Millisecond,
			stallAfterReq: 2,
			requests:      requests,
		}
		gSrv := grpc.NewServer()
		storagepb.RegisterBigQueryWriteServer(gSrv, srv)
		go func() { _ = gSrv.Serve(lis) }()
		t.Cleanup(gSrv.Stop)

		conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("dial: %v", err)
		}
		t.Cleanup(func() { _ = conn.Close() })

		wc, err := bqstorage.NewBigQueryWriteClient(context.Background(), option.WithGRPCConn(conn))
		if err != nil {
			t.Fatalf("write client: %v", err)
		}
		t.Cleanup(func() { _ = wc.Close() })

		cfg := DefaultConfig()
		cfg.BatchSize = 5
		cfg.BatchFlushIntv = time.Hour
		cfg.ShutdownTimeout = 300 * time.Millisecond
		cfg.RetryConfig = RetryConfig{
			MaxRetries:   0,
			InitialDelay: time.Millisecond,
			MaxDelay:     time.Millisecond,
			Multiplier:   1.0,
		}
		cfg.Logger = &recordingLogger{}

		bp, err := NewBatchProcessor(context.Background(), wc, "projects/p/datasets/d/tables/t/_default", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor: %v", err)
		}
		bp.Start()

		// Send rows straight into the queue so no background flush goroutine is
		// spawned and Close drains all batches deterministically.
		const totalEvents = 66
		for i := range totalEvents {
			bp.queue <- map[string]any{"event_type": fmt.Sprintf("evt_%d", i)}
		}

		start := time.Now()
		bp.Close()
		elapsed := time.Since(start)
		if elapsed < 150*time.Millisecond {
			t.Errorf("Close() returned in %v, expected it to wait until ShutdownTimeout (%v) expired on stalled server", elapsed, cfg.ShutdownTimeout)
		}

		var persistedRows int64
		for len(requests) > 0 {
			req := <-requests
			persistedRows += req.GetArrowRows().GetRows().GetRowCount()
		}
		if persistedRows != 10 {
			t.Fatalf("persisted rows = %d, want 10 (2 batches of 5)", persistedRows)
		}

		stats := bp.DropStats()
		wantStats := DropStats{
			AppendFailureDrops: 5,
			ShutdownDrops:      51,
			TotalDrops:         56,
		}
		if diff := cmp.Diff(wantStats, stats); diff != "" {
			t.Errorf("DropStats() mismatch (-want +got):\n%s", diff)
		}
		if persistedRows+stats.TotalDrops != totalEvents {
			t.Errorf("persistedRows (%d) + TotalDrops (%d) = %d, want %d", persistedRows, stats.TotalDrops, persistedRows+stats.TotalDrops, totalEvents)
		}
	})

	t.Run("writeBatchWithContexts counts batch in AppendFailureDrops when writeCtx is already canceled", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.RetryConfig.MaxRetries = 0
		bp, err := NewBatchProcessor(context.Background(), nil, "test_stream", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor: %v", err)
		}

		canceledCtx, cancel := context.WithCancel(context.Background())
		cancel()

		batch := []map[string]any{
			{"event_type": "row_1"},
			{"event_type": "row_2"},
			{"event_type": "row_3"},
		}
		err = bp.writeBatchWithContexts(context.Background(), canceledCtx, false, batch)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("writeBatchWithContexts() error = %v, want %v", err, context.Canceled)
		}

		want := DropStats{
			AppendFailureDrops: 3,
			TotalDrops:         3,
		}
		if diff := cmp.Diff(want, bp.DropStats()); diff != "" {
			t.Errorf("DropStats() mismatch (-want +got):\n%s", diff)
		}
	})
}

type stallingBigQueryWriteServer struct {
	storagepb.UnimplementedBigQueryWriteServer
}

func (s *stallingBigQueryWriteServer) AppendRows(stream storagepb.BigQueryWrite_AppendRowsServer) error {
	if _, err := stream.Recv(); err != nil {
		return err
	}
	<-stream.Context().Done()
	return stream.Context().Err()
}

type delayedBigQueryWriteServer struct {
	storagepb.UnimplementedBigQueryWriteServer
	delay         time.Duration
	stallAfterReq int32
	reqCount      atomic.Int32
	requests      chan *storagepb.AppendRowsRequest
	// received, if set, is sent to as each request arrives, before it is
	// delayed; a send that would block is skipped.
	received chan struct{}
}

func (s *delayedBigQueryWriteServer) AppendRows(stream storagepb.BigQueryWrite_AppendRowsServer) error {
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if s.received != nil {
			select {
			case s.received <- struct{}{}:
			default:
			}
		}
		n := s.reqCount.Add(1)
		if s.stallAfterReq > 0 && n > s.stallAfterReq {
			<-stream.Context().Done()
			return stream.Context().Err()
		}
		select {
		case <-time.After(s.delay):
		case <-stream.Context().Done():
			return stream.Context().Err()
		}
		if s.requests != nil {
			s.requests <- req
		}
		if err := stream.Send(&storagepb.AppendRowsResponse{}); err != nil {
			return err
		}
	}
}

// arrivalRecordingBigQueryWriteServer records incoming AppendRows requests on
// arrival, before any delay, so tests can see attempts cut short by caller
// deadline cancellation.
type arrivalRecordingBigQueryWriteServer struct {
	storagepb.UnimplementedBigQueryWriteServer
	delay    time.Duration
	requests chan *storagepb.AppendRowsRequest
}

func (s *arrivalRecordingBigQueryWriteServer) AppendRows(stream storagepb.BigQueryWrite_AppendRowsServer) error {
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if s.requests != nil {
			s.requests <- req
		}
		select {
		case <-time.After(s.delay):
		case <-stream.Context().Done():
			return stream.Context().Err()
		}
		if err := stream.Send(&storagepb.AppendRowsResponse{}); err != nil {
			return err
		}
	}
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

func TestBatchProcessor_FlushRunEndDoesNotDropBatchWhenBudgetExpires(t *testing.T) {
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	requests := make(chan *storagepb.AppendRowsRequest, 16)
	gSrv := grpc.NewServer()
	// Each batch write takes 250ms: Batch 0 succeeds in 250ms, leaving 150ms of
	// the 400ms ShutdownTimeout, so Batch 1's 250ms write times out during the
	// first flushRunEnd and must be deferred to the next flush rather than lost.
	storagepb.RegisterBigQueryWriteServer(gSrv, &delayedBigQueryWriteServer{
		delay:    250 * time.Millisecond,
		requests: requests,
	})
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

	logger := &recordingLogger{}
	cfg := DefaultConfig()
	cfg.BatchSize = 2
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = 400 * time.Millisecond
	cfg.RetryConfig.MaxRetries = 0
	cfg.Logger = logger

	bp, err := NewBatchProcessor(context.Background(), writeClient, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	bp.Start()

	wantOrder := []string{"row-0", "row-1", "row-2", "row-3"}
	for _, ev := range wantOrder {
		bp.queue <- map[string]any{"event_type": ev}
	}

	bp.flushRunEnd()
	if logger.contains("Failed to write batch") {
		t.Errorf("flushRunEnd logged a batch drop error when its budget expired:\n%s", logger)
	}

	// A subsequent flush writes the deferred batch so all 4 rows arrive in order.
	bp.Close()

	rows := collectRows(t, requests, func(rows []map[string]any) bool { return len(rows) >= len(wantOrder) })
	var gotOrder []string
	for _, row := range rows {
		gotOrder = append(gotOrder, fmt.Sprint(row["event_type"]))
	}
	if diff := cmp.Diff(wantOrder, gotOrder); diff != "" {
		t.Errorf("written event_type order mismatch (-want +got):\n%s", diff)
	}
}

func TestBatchProcessor_FlushRunEndBoundedReDeferral(t *testing.T) {
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	requests := make(chan *storagepb.AppendRowsRequest, 16)
	gSrv := grpc.NewServer()
	// Each batch write takes 250ms with ShutdownTimeout of 200ms, so every
	// write attempt during flushRunEnd times out under flushCtx.Err() != nil.
	storagepb.RegisterBigQueryWriteServer(gSrv, &arrivalRecordingBigQueryWriteServer{
		delay:    250 * time.Millisecond,
		requests: requests,
	})
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

	logger := &recordingLogger{}
	cfg := DefaultConfig()
	cfg.BatchSize = 2
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = 200 * time.Millisecond
	cfg.RetryConfig.MaxRetries = 0
	cfg.Logger = logger

	bp, err := NewBatchProcessor(context.Background(), writeClient, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	t.Cleanup(bp.Close)
	bp.Start()

	// Queue 4 rows: batch 1 = {row-0, row-1}, batch 2 = {row-2, row-3}.
	for _, ev := range []string{"row-0", "row-1", "row-2", "row-3"} {
		bp.queue <- map[string]any{"event_type": ev}
	}

	// First flushRunEnd: batch 1 starts writing, times out at 200ms. Because
	// !retry (deferredBatch was empty), it is deferred to deferredBatch.
	bp.flushRunEnd()
	if len(bp.deferredBatch) != 2 {
		t.Fatalf("deferredBatch length after first flushRunEnd = %d, want 2", len(bp.deferredBatch))
	}

	// Second flushRunEnd: batch 1 is popped from deferredBatch (retry == true).
	// Write times out again at 200ms. Since retry is true, batch 1 is dropped
	// rather than re-deferred indefinitely, leaving deferredBatch empty.
	bp.flushRunEnd()
	if len(bp.deferredBatch) != 0 {
		t.Fatalf("deferredBatch length after second flushRunEnd = %d, want 0 (bounded single retry)", len(bp.deferredBatch))
	}
	if !logger.contains("Failed to write batch to BigQuery") {
		t.Errorf("expected drop error logged after retry failure, got:\n%s", logger)
	}

	// Third and fourth flushRunEnd: batch 2 is now reached and attempted (not
	// starved by indefinite re-deferral of batch 1). It is attempted once and
	// retried once before being dropped.
	bp.flushRunEnd()
	bp.flushRunEnd()

	// Drain all requests received by the arrival-recording server.
	var receivedBatches [][]string
	for len(requests) > 0 {
		req := <-requests
		var batchEvents []string
		for _, row := range decodeArrowRows(t, req) {
			batchEvents = append(batchEvents, fmt.Sprint(row["event_type"]))
		}
		receivedBatches = append(receivedBatches, batchEvents)
	}

	// Assert at most two sends per batch, and that batch 2 was written/sent.
	wantBatches := [][]string{
		{"row-0", "row-1"},
		{"row-0", "row-1"},
		{"row-2", "row-3"},
		{"row-2", "row-3"},
	}
	if diff := cmp.Diff(wantBatches, receivedBatches); diff != "" {
		t.Errorf("received batches mismatch (-want +got):\n%s", diff)
	}
}

type dynamicDelayBigQueryWriteServer struct {
	storagepb.UnimplementedBigQueryWriteServer
	delay    *atomic.Int64
	requests chan *storagepb.AppendRowsRequest
}

func (s *dynamicDelayBigQueryWriteServer) AppendRows(stream storagepb.BigQueryWrite_AppendRowsServer) error {
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if s.delay != nil {
			if d := time.Duration(s.delay.Load()); d > 0 {
				select {
				case <-time.After(d):
				case <-stream.Context().Done():
					return stream.Context().Err()
				}
			}
		}
		if s.requests != nil {
			s.requests <- req
		}
		if err := stream.Send(&storagepb.AppendRowsResponse{}); err != nil {
			return err
		}
	}
}

func TestBatchProcessor_BackgroundFlushRetriesDeferredBatch(t *testing.T) {
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	requests := make(chan *storagepb.AppendRowsRequest, 16)
	gSrv := grpc.NewServer()
	var serverDelay atomic.Int64
	serverDelay.Store(int64(50 * time.Millisecond))

	storagepb.RegisterBigQueryWriteServer(gSrv, &dynamicDelayBigQueryWriteServer{
		delay:    &serverDelay,
		requests: requests,
	})
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

	var stalled atomic.Bool
	stalled.Store(true)
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		if stalled.Load() {
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}}
	cfg := DefaultConfig()
	cfg.BatchSize = 1
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = 120 * time.Millisecond
	cfg.RetryConfig.MaxRetries = 0
	cfg.GCSOffloader = off
	cfg.GCSMaxPendingUploadBytes = 64
	cfg.Logger = &recordingLogger{}

	bp, err := NewBatchProcessor(context.Background(), writeClient, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	t.Cleanup(bp.Close)
	bp.Start()

	// Batch 0: no upload; takes 50ms in AppendRows, leaving ~70ms of 120ms ShutdownTimeout (< 3/4).
	bp.queue <- map[string]any{"event_type": EventUserMessageReceived}

	// Batch 1: 64-byte upload; because budget is already depleted, pauseOnCallerDeadline is false.
	// Upload stalls for 35ms (half remaining) and is deferred with 64 bytes retained in pendingUploadBytes.
	// AppendRows takes 50ms, exceeding the remaining ~35ms, so flushRunEnd saves batch 1 in deferredBatch.
	row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(make([]byte, 64)...))
	row["event_type"] = EventLLMRequest
	bp.queue <- row

	bp.flushRunEnd()

	if len(bp.deferredBatch) != 1 {
		t.Fatalf("deferredBatch length after flushRunEnd = %d, want 1", len(bp.deferredBatch))
	}
	if got := bp.pendingUploadBytes.Load(); got != 64 {
		t.Fatalf("pendingUploadBytes after flushRunEnd = %d, want 64", got)
	}

	// Restore healthy storage and BigQuery writes.
	stalled.Store(false)
	serverDelay.Store(0)

	// Background flush must write deferredBatch before draining the queue.
	bp.flush()

	if len(bp.deferredBatch) != 0 {
		t.Fatalf("deferredBatch length after flush = %d, want 0", len(bp.deferredBatch))
	}
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Fatalf("pendingUploadBytes after flush = %d, want 0", got)
	}

	rows := collectRows(t, requests, func(rows []map[string]any) bool { return len(rows) >= 2 })
	if len(rows) != 2 {
		t.Fatalf("persisted rows = %d, want 2", len(rows))
	}
	if got := contentPartsOf(t, rows[1])[0]["storage_mode"]; got != "GCS_REFERENCE" {
		t.Errorf("part storage_mode = %v, want GCS_REFERENCE", got)
	}
}

func TestBatchProcessor_ExactlyOnceDelivery_NotFoundAfterLostResponseIsNotResent(t *testing.T) {
	ctx := context.Background()
	var (
		attempts atomic.Int32
		srv      = &scriptedCommittedStreamServer{}
	)
	srv.onAppend = func(attempt int, req *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
		switch attempts.Add(1) {
		case 1:
			// Send succeeds; Recv fails ambiguously.
			return nil, status.Error(codes.Unavailable, "lost connection")
		case 2:
			// Server reports NotFound on the same stream.
			return nil, status.Error(codes.NotFound, "stream expired or missing")
		default:
			return &storagepb.AppendRowsResponse{}, nil
		}
	}

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
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
	config.ProjectID = "test-proj"
	config.DatasetID = "test-ds"
	config.TableName = "agent_events"
	config.ExactlyOnceDelivery = true
	config.RetryConfig = RetryConfig{MaxRetries: 2, InitialDelay: time.Millisecond, MaxDelay: 5 * time.Millisecond, Multiplier: 1.0}
	bp, err := NewBatchProcessor(ctx, writeClient, "", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor() error = %v", err)
	}
	defer bp.Close()

	_ = bp.writeBatch(ctx, []map[string]any{{"event_type": "LLM_REQUEST"}, {"event_type": "LLM_RESPONSE"}})

	srv.mu.Lock()
	defer srv.mu.Unlock()
	first := srv.appendStreamNames[0]
	for i, name := range srv.appendStreamNames {
		if name != first {
			t.Errorf("attempt %d sent the batch to %s after an attempt on %s whose outcome was unknown; it may now be stored on both streams", i+1, name, first)
		}
	}
	if got := bp.DropStats().AppendFailureDrops; got != 2 {
		t.Errorf("AppendFailureDrops = %d, want 2 (the batch's outcome is unknown, so it is dropped and counted)", got)
	}
}

func TestBatchProcessor_DefaultStream_SendsRoutingHeaderWithoutOffset(t *testing.T) {
	ctx := context.Background()
	var (
		appendReqParamsMu sync.Mutex
		appendReqParams   []string
		receivedOffsets   []*wrapperspb.Int64Value
		srv               = &scriptedCommittedStreamServer{
			onAppend: func(_ int, req *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
				appendReqParamsMu.Lock()
				receivedOffsets = append(receivedOffsets, req.GetOffset())
				appendReqParamsMu.Unlock()
				return &storagepb.AppendRowsResponse{}, nil
			},
		}
	)

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	gSrv := grpc.NewServer(
		grpc.StreamInterceptor(func(srvObj any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
			if strings.HasSuffix(info.FullMethod, "/AppendRows") {
				md, _ := metadata.FromIncomingContext(ss.Context())
				appendReqParamsMu.Lock()
				appendReqParams = append(appendReqParams, md.Get("x-goog-request-params")...)
				appendReqParamsMu.Unlock()
			}
			return handler(srvObj, ss)
		}),
	)
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
	config.ProjectID = "test-proj"
	config.DatasetID = "test-ds"
	config.TableName = "agent_events"
	config.ExactlyOnceDelivery = false
	defaultStream := "projects/test-proj/datasets/test-ds/tables/agent_events/streams/_default"
	bp, err := NewBatchProcessor(ctx, writeClient, defaultStream, config)
	if err != nil {
		t.Fatalf("NewBatchProcessor() error = %v", err)
	}
	defer bp.Close()

	if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "DEFAULT_STREAM"}}); err != nil {
		t.Fatalf("writeBatch() error = %v", err)
	}

	appendReqParamsMu.Lock()
	defer appendReqParamsMu.Unlock()
	wantParams := []string{"write_stream=" + defaultStream}
	if diff := cmp.Diff(wantParams, appendReqParams); diff != "" {
		t.Errorf("default stream x-goog-request-params mismatch (-want +got):\n%s", diff)
	}
	if len(receivedOffsets) != 1 || receivedOffsets[0] != nil {
		t.Errorf("default stream Offset = %v, want [nil]", receivedOffsets)
	}
}

func TestBatchProcessor_ExactlyOnceDelivery_BoundedRotationOnPersistentRejection(t *testing.T) {
	ctx := context.Background()
	var (
		createCalls atomic.Int32
		srv         = &scriptedCommittedStreamServer{
			onAppend: func(_ int, _ *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
				return nil, status.Error(codes.InvalidArgument, "invalid argument rejection")
			},
		}
	)

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	gSrv := grpc.NewServer(
		grpc.UnaryInterceptor(func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
			if strings.HasSuffix(info.FullMethod, "/CreateWriteStream") {
				createCalls.Add(1)
			}
			return handler(ctx, req)
		}),
	)
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
	config.ProjectID = "test-proj"
	config.DatasetID = "test-ds"
	config.TableName = "agent_events"
	config.ExactlyOnceDelivery = true
	config.RetryConfig = RetryConfig{MaxRetries: 1, InitialDelay: time.Millisecond, MaxDelay: 2 * time.Millisecond, Multiplier: 1.0}
	bp, err := NewBatchProcessor(ctx, writeClient, "", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor() error = %v", err)
	}
	defer bp.Close()

	// Write 10 batches. Because InvalidArgument is not an ambiguous send error,
	// none of the batches should trigger stream rotation across batches.
	for i := 0; i < 10; i++ {
		_ = bp.writeBatch(ctx, []map[string]any{{"event_type": fmt.Sprintf("BATCH_%d", i)}})
	}

	if got := createCalls.Load(); got != 1 {
		t.Errorf("CreateWriteStream calls = %d, want 1 (only initial stream creation; persistent rejections must not rotate)", got)
	}
}

func TestBatchProcessor_ExactlyOnceDelivery_RetryAttemptRefundOnRotation(t *testing.T) {
	ctx := context.Background()
	var (
		attempts atomic.Int32
		srv      = &scriptedCommittedStreamServer{}
	)
	srv.onAppend = func(attempt int, req *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
		switch attempts.Add(1) {
		case 1:
			// Stream 1 gets NotFound without prior ambiguous send.
			return nil, status.Error(codes.NotFound, "stream expired")
		case 2:
			// Stream 2 (after rotation) succeeds.
			return &storagepb.AppendRowsResponse{}, nil
		default:
			return &storagepb.AppendRowsResponse{}, nil
		}
	}

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
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
	config.ProjectID = "test-proj"
	config.DatasetID = "test-ds"
	config.TableName = "agent_events"
	config.ExactlyOnceDelivery = true
	// MaxRetries: 0 means no regular retries are allowed.
	// Only the rotation attempt refund allows sending on the newly created stream.
	config.RetryConfig = RetryConfig{MaxRetries: 0}
	bp, err := NewBatchProcessor(ctx, writeClient, "", config)
	if err != nil {
		t.Fatalf("NewBatchProcessor() error = %v", err)
	}
	defer bp.Close()

	if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "EVENT"}}); err != nil {
		t.Fatalf("writeBatch() with MaxRetries: 0 failed unexpectedly after rotation: %v", err)
	}
}
