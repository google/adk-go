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

package agentengine

import (
	"context"
	"errors"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/genproto/googleapis/api/httpbody"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

func TestStreamQueryRequestShape(t *testing.T) {
	srv := &fakeExecServer{chunks: []string{lineA}}
	c := newFakeClient(t, srv)

	// Keys must be genai's camelCase and bytes standard base64: that is what
	// the server decodes into genai.Content.
	msg := &genai.Content{Role: "user", Parts: []*genai.Part{
		{Text: "hello"},
		{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{0xfb, 0xff, 0xfe}}},
		{FunctionResponse: &genai.FunctionResponse{Name: "lookup", Response: map[string]any{"status_code": 200.0}}},
	}}
	events, err := collect(t, c.StreamQuery(t.Context(), StreamQueryRequest{UserID: "u1", SessionID: "s1", Message: msg}))
	if err != nil || len(events) != 1 {
		t.Fatalf("StreamQuery() = %d events, %v; want 1, nil", len(events), err)
	}

	req := srv.request()
	if req.GetName() != "projects/p/locations/us-central1/reasoningEngines/123" {
		t.Errorf("Name = %q", req.GetName())
	}
	if c.Name() != req.GetName() {
		t.Errorf("Client.Name() = %q, want %q", c.Name(), req.GetName())
	}
	if req.GetClassMethod() != "async_stream_query" {
		t.Errorf("ClassMethod = %q, want async_stream_query", req.GetClassMethod())
	}
	in := req.GetInput().AsMap()
	if in["user_id"] != "u1" || in["session_id"] != "s1" {
		t.Errorf("input ids = %v / %v", in["user_id"], in["session_id"])
	}
	wantMsg := map[string]any{"role": "user", "parts": []any{
		map[string]any{"text": "hello"},
		map[string]any{"inlineData": map[string]any{"mimeType": "image/png", "data": "+//+"}},
		map[string]any{"functionResponse": map[string]any{"name": "lookup", "response": map[string]any{"status_code": 200.0}}},
	}}
	if diff := cmp.Diff(wantMsg, in["message"]); diff != "" {
		t.Errorf("input.message mismatch (-want +got):\n%s", diff)
	}
}

func TestStreamQueryRangeIssuesOneTurnEach(t *testing.T) {
	srv := &fakeExecServer{chunks: []string{lineA}}
	c := newFakeClient(t, srv)
	msg := genai.NewContentFromText("first", genai.RoleUser)
	events := c.StreamQuery(t.Context(), StreamQueryRequest{UserID: "u", SessionID: "s", Message: msg})
	if srv.callCount() != 0 {
		t.Fatalf("StreamQuery() issued the RPC before the iterator was ranged")
	}
	msg.Parts[0].Text = "second"
	for range 2 {
		if _, err := collect(t, events); err != nil {
			t.Fatal(err)
		}
	}
	if got := srv.callCount(); got != 2 {
		t.Errorf("ranging twice issued %d RPCs, want 2", got)
	}
	sent := srv.request().GetInput().AsMap()["message"].(map[string]any)["parts"].([]any)[0].(map[string]any)["text"]
	if sent != "second" {
		t.Errorf("message read at range time = %q, want second", sent)
	}
}

func TestStreamQueryCustomClassMethod(t *testing.T) {
	srv := &fakeExecServer{}
	c := newFakeClient(t, srv)
	if _, err := collect(t, c.StreamQuery(t.Context(), StreamQueryRequest{UserID: "u", SessionID: "s", Message: genai.NewContentFromText("x", genai.RoleUser), ClassMethod: "stream_query"})); err != nil {
		t.Fatal(err)
	}
	if got := srv.request().GetClassMethod(); got != "stream_query" {
		t.Errorf("ClassMethod = %q, want stream_query", got)
	}
}

func TestStreamQueryReassemblesAcrossChunks(t *testing.T) {
	srv := &fakeExecServer{chunks: []string{lineA[:15], lineA[15:] + lineB[:3], lineB[3:]}}
	c := newFakeClient(t, srv)
	events, err := collect(t, c.StreamQuery(t.Context(), StreamQueryRequest{UserID: "u", SessionID: "s", Message: genai.NewContentFromText("x", genai.RoleUser)}))
	if err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff([]string{"A", "B"}, texts(events)); diff != "" {
		t.Errorf("(-want +got):\n%s", diff)
	}
}

func TestStreamQueryRemoteErrorLine(t *testing.T) {
	srv := &fakeExecServer{chunks: []string{lineA + `{"error":"model quota"}` + "\n"}}
	c := newFakeClient(t, srv)
	events, err := collect(t, c.StreamQuery(t.Context(), StreamQueryRequest{UserID: "u", SessionID: "s", Message: genai.NewContentFromText("x", genai.RoleUser)}))
	if !errors.Is(err, ErrRemote) || !strings.Contains(err.Error(), "model quota") {
		t.Fatalf("error = %v, want ErrRemote carrying the server message", err)
	}
	if len(events) != 1 {
		t.Errorf("got %d events before the error, want 1", len(events))
	}
}

func TestStreamQueryGRPCStatusError(t *testing.T) {
	srv := &fakeExecServer{err: status.Error(codes.PermissionDenied, "no aiplatform.user")}
	c := newFakeClient(t, srv)
	_, err := collect(t, c.StreamQuery(t.Context(), StreamQueryRequest{UserID: "u", SessionID: "s", Message: genai.NewContentFromText("x", genai.RoleUser)}))
	if status.Code(err) != codes.PermissionDenied {
		t.Fatalf("error = %v, want PermissionDenied status", err)
	}
}

func TestStreamQueryEmptyStream(t *testing.T) {
	c := newFakeClient(t, &fakeExecServer{})
	events, err := collect(t, c.StreamQuery(t.Context(), StreamQueryRequest{UserID: "u", SessionID: "s", Message: genai.NewContentFromText("x", genai.RoleUser)}))
	if err != nil || len(events) != 0 {
		t.Fatalf("got %d events, %v; want 0, nil", len(events), err)
	}
}

func TestStreamQueryCancelledContext(t *testing.T) {
	srv := &fakeExecServer{block: make(chan struct{}), received: make(chan struct{}), chunks: []string{lineA}}
	c := newFakeClient(t, srv)
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() {
		_, err := collect(t, c.StreamQuery(ctx, StreamQueryRequest{UserID: "u", SessionID: "s", Message: genai.NewContentFromText("x", genai.RoleUser)}))
		done <- err
	}()
	// Cancel only once the stream is open, so the Recv path is the one cut.
	select {
	case <-srv.received:
	case <-time.After(5 * time.Second):
		t.Fatal("server never received the request")
	}
	cancel()
	var err error
	select {
	case err = <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("StreamQuery() did not return after cancel")
	}
	if status.Code(err) != codes.Canceled {
		t.Fatalf("error = %v, want status Canceled", err)
	}
}

func TestStreamQueryDeadlineInRecv(t *testing.T) {
	srv := &fakeExecServer{block: make(chan struct{}), received: make(chan struct{}), chunks: []string{lineA}}
	defer close(srv.block)
	c := newFakeClient(t, srv)
	ctx, cancel := context.WithTimeout(t.Context(), 300*time.Millisecond)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := collect(t, c.StreamQuery(ctx, StreamQueryRequest{UserID: "u", SessionID: "s", Message: genai.NewContentFromText("x", genai.RoleUser)}))
		done <- err
	}()
	// The deadline must cut Recv, not the open call.
	select {
	case <-srv.received:
	case <-time.After(5 * time.Second):
		t.Fatal("server never received the request")
	}
	var err error
	select {
	case err = <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("StreamQuery() did not return after the deadline")
	}
	if status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("error = %v, want status DeadlineExceeded", err)
	}
}

func TestStreamQueryRemoteCanceledKeepsCtxLive(t *testing.T) {
	// A server-side Canceled carries the same status code as the caller's
	// own cancellation. Only ctx.Err() tells them apart.
	srv := &fakeExecServer{err: status.Error(codes.Canceled, "server gave up")}
	c := newFakeClient(t, srv)
	ctx := t.Context()
	_, err := collect(t, c.StreamQuery(ctx, StreamQueryRequest{UserID: "u", SessionID: "s", Message: genai.NewContentFromText("x", genai.RoleUser)}))
	if status.Code(err) != codes.Canceled {
		t.Fatalf("error = %v, want status Canceled", err)
	}
	if ctx.Err() != nil {
		t.Fatalf("ctx.Err() = %v, want nil for a remote cancellation", ctx.Err())
	}
}

func TestStreamQueryContextDoneBeforeRange(t *testing.T) {
	srv := &fakeExecServer{chunks: []string{lineA}}
	c := newFakeClient(t, srv)
	ctx, cancel := context.WithCancel(t.Context())
	events := c.StreamQuery(ctx, StreamQueryRequest{UserID: "u", SessionID: "s", Message: genai.NewContentFromText("x", genai.RoleUser)})
	cancel()
	got, err := collect(t, events)
	if status.Code(err) != codes.Canceled || len(got) != 0 {
		t.Fatalf("StreamQuery() = %d events, %v; want 0, status Canceled", len(got), err)
	}
}

func TestStreamQueryDoneBetweenBufferedLinesNonStdlibCtx(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		code codes.Code
	}{
		{"cancel", context.Canceled, codes.Canceled},
		{"deadline", context.DeadlineExceeded, codes.DeadlineExceeded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := &fakeExecServer{chunks: []string{lineA + lineB + lineC}, ctxDone: make(chan struct{})}
			c := newFakeClient(t, srv)
			ctx := newManualCtx(t.Context())
			var got []*Event
			var gotErr error
			for ev, err := range c.StreamQuery(ctx, StreamQueryRequest{UserID: "u", SessionID: "s", Message: genai.NewContentFromText("x", genai.RoleUser)}) {
				if err != nil {
					gotErr = err
					break
				}
				got = append(got, ev)
				ctx.end(tc.err)
			}
			if diff := cmp.Diff([]string{"A"}, texts(got)); diff != "" {
				t.Errorf("events after ctx ended (-want +got):\n%s", diff)
			}
			if status.Code(gotErr) != tc.code {
				t.Fatalf("error = %v, want status %v", gotErr, tc.code)
			}
		})
	}
}

func TestStreamQueryCancelWinsOverBufferedErrorLine(t *testing.T) {
	// Once ctx is done nothing received earlier is yielded, an error line
	// included.
	srv := &fakeExecServer{chunks: []string{lineA + `{"error":"boom"}` + "\n"}}
	c := newFakeClient(t, srv)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	var gotErr error
	for _, err := range c.StreamQuery(ctx, StreamQueryRequest{UserID: "u", SessionID: "s", Message: genai.NewContentFromText("x", genai.RoleUser)}) {
		if err != nil {
			gotErr = err
			break
		}
		cancel()
	}
	if status.Code(gotErr) != codes.Canceled || strings.Contains(gotErr.Error(), "boom") {
		t.Fatalf("error = %v, want status Canceled without the remote error", gotErr)
	}
}

func TestStreamQueryCancelledBetweenBufferedLines(t *testing.T) {
	// Three lines arrive in one chunk and the server finishes at once, so no
	// Recv is left to report the cancellation: the client has to notice it.
	srv := &fakeExecServer{chunks: []string{lineA + lineB + lineC}}
	c := newFakeClient(t, srv)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	var got []*Event
	var gotErr error
	for ev, err := range c.StreamQuery(ctx, StreamQueryRequest{UserID: "u", SessionID: "s", Message: genai.NewContentFromText("x", genai.RoleUser)}) {
		if err != nil {
			gotErr = err
			break
		}
		got = append(got, ev)
		cancel()
	}
	if diff := cmp.Diff([]string{"A"}, texts(got)); diff != "" {
		t.Errorf("events after cancel (-want +got):\n%s", diff)
	}
	if status.Code(gotErr) != codes.Canceled {
		t.Fatalf("error = %v, want status Canceled", gotErr)
	}
}

func TestStreamQueryRejectsNilMessage(t *testing.T) {
	c := newFakeClient(t, &fakeExecServer{})
	_, err := collect(t, c.StreamQuery(t.Context(), StreamQueryRequest{UserID: "u", SessionID: "s"}))
	if err == nil {
		t.Fatal("nil Message accepted, want error")
	}
}

func TestStreamQueryEarlyBreakCancelsStream(t *testing.T) {
	srv := &fakeExecServer{chunks: []string{lineA}, ctxDone: make(chan struct{})}
	c := newFakeClient(t, srv)
	for _, err := range c.StreamQuery(t.Context(), StreamQueryRequest{UserID: "u", SessionID: "s", Message: genai.NewContentFromText("x", genai.RoleUser)}) {
		if err != nil {
			t.Fatal(err)
		}
		break
	}
	select {
	case <-srv.ctxDone:
	case <-time.After(5 * time.Second):
		t.Fatal("server stream context still live after the consumer stopped")
	}
}

func TestNewCallerEndpointOverridesRegional(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen() failed: %v", err)
	}
	server := grpc.NewServer()
	aiplatformpb.RegisterReasoningEngineExecutionServiceServer(server, &fakeExecServer{chunks: []string{lineA}})
	go func() { _ = server.Serve(lis) }()
	t.Cleanup(server.Stop)

	// Bounded so a client still pointed at the regional endpoint fails fast.
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	c, err := New(ctx, Config{Project: "p", Location: "us-central1", ReasoningEngine: "1"},
		option.WithEndpoint(lis.Addr().String()),
		option.WithoutAuthentication(),
		option.WithGRPCDialOption(grpc.WithTransportCredentials(insecure.NewCredentials())))
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })

	events, err := collect(t, c.StreamQuery(ctx, StreamQueryRequest{UserID: "u", SessionID: "s", Message: genai.NewContentFromText("x", genai.RoleUser)}))
	if err != nil {
		t.Fatalf("StreamQuery() error = %v, want the local server to answer", err)
	}
	if diff := cmp.Diff([]string{"A"}, texts(events)); diff != "" {
		t.Errorf("(-want +got):\n%s", diff)
	}
}

// fakeExecServer records the last request and replays scripted chunks.
type fakeExecServer struct {
	aiplatformpb.UnimplementedReasoningEngineExecutionServiceServer

	mu      sync.Mutex
	lastReq *aiplatformpb.StreamQueryReasoningEngineRequest
	calls   int
	chunks  []string
	// received, when non-nil, is closed once the first request has arrived, so
	// a test can act after the stream is open.
	received chan struct{}
	// block, when non-nil, makes the server wait before sending anything, so a
	// test can cancel the client context mid-stream.
	block chan struct{}
	// ctxDone, when non-nil, makes the server hold the stream open after the
	// scripted chunks and close ctxDone once the client cancels it.
	ctxDone chan struct{}
	err     error
}

func (f *fakeExecServer) StreamQueryReasoningEngine(req *aiplatformpb.StreamQueryReasoningEngineRequest, stream aiplatformpb.ReasoningEngineExecutionService_StreamQueryReasoningEngineServer) error {
	f.mu.Lock()
	f.lastReq = req
	f.calls++
	first := f.calls == 1
	f.mu.Unlock()
	if first && f.received != nil {
		close(f.received)
	}
	if f.block != nil {
		select {
		case <-f.block:
		case <-stream.Context().Done():
			return stream.Context().Err()
		}
	}
	for _, c := range f.chunks {
		if err := stream.Send(&httpbody.HttpBody{ContentType: "application/json", Data: []byte(c)}); err != nil {
			return err
		}
	}
	if f.ctxDone != nil {
		<-stream.Context().Done()
		close(f.ctxDone)
	}
	return f.err
}

func (f *fakeExecServer) request() *aiplatformpb.StreamQueryReasoningEngineRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.lastReq
}

func (f *fakeExecServer) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

// newFakeClient serves srv over bufconn and returns a Client dialed to it.
func newFakeClient(t *testing.T, srv *fakeExecServer) *Client {
	t.Helper()
	listener := bufconn.Listen(1024 * 1024)
	server := grpc.NewServer()
	aiplatformpb.RegisterReasoningEngineExecutionServiceServer(server, srv)
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(func() {
		server.Stop()
		_ = listener.Close()
	})
	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("grpc.NewClient() failed: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	c, err := New(t.Context(), Config{Project: "p", Location: "us-central1", ReasoningEngine: "123"},
		option.WithGRPCConn(conn), option.WithoutAuthentication())
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

// manualCtx is a context outside the context package, so a ctx derived from it
// learns that it is done only through context's watcher goroutine. The test
// decides when it ends, which also keeps the deadline case deterministic.
type manualCtx struct {
	context.Context
	done chan struct{}
	mu   sync.Mutex
	err  error
}

func newManualCtx(parent context.Context) *manualCtx {
	return &manualCtx{Context: parent, done: make(chan struct{})}
}

func (m *manualCtx) Done() <-chan struct{} { return m.done }

func (m *manualCtx) Err() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.err
}

func (m *manualCtx) end(err error) {
	m.mu.Lock()
	m.err = err
	m.mu.Unlock()
	close(m.done)
}
