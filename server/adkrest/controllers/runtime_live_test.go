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

package controllers

import (
	"bytes"
	"errors"
	"iter"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/server/adkrest/internal/fakes"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/session"
)

// mockLiveAgent adds live-run behavior to a regular custom agent. Embedding an
// agent created through agent.New keeps the test double within the supported
// agent construction path while allowing the live stream to be controlled.
type mockLiveAgent struct {
	agent.Agent
	runLiveFn func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
}

func (a *mockLiveAgent) RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	return a.runLiveFn(ctx)
}

// recordingLiveSession is the client-to-agent half of a live run. Later tests
// use requests to inspect forwarded WebSocket frames and closed to assert that
// the handler releases the session.
type recordingLiveSession struct {
	requests  chan agent.LiveRequest
	closed    chan struct{}
	closeOnce sync.Once
}

func newRecordingLiveSession() *recordingLiveSession {
	return &recordingLiveSession{
		requests: make(chan agent.LiveRequest, 1),
		closed:   make(chan struct{}),
	}
}

func (s *recordingLiveSession) Send(req agent.LiveRequest) error {
	s.requests <- req
	return nil
}

func (s *recordingLiveSession) Close() error {
	s.closeOnce.Do(func() {
		close(s.closed)
	})
	return nil
}

type synchronizedBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (b *synchronizedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.b.Write(p)
}

func (b *synchronizedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.b.String()
}

const (
	testLiveAppName          = "testApp"
	testLiveUserID           = "testUser"
	testLiveSessionID        = "testSession"
	testWebSocketReadTimeout = time.Second
	// Keep this below RunLiveHandler's one-second close-drain deadline so the
	// test verifies that the handler waits for the client's close reply.
	testCloseDrainGracePeriod = 50 * time.Millisecond
	testCloseReplyExitTimeout = 100 * time.Millisecond
	// testMaxHandlerExits is the capacity of the handler-exit channel, large
	// enough for every connection a test in this file opens.
	testMaxHandlerExits = 8
)

// startRunLiveServer serves RunLiveHandler over cfg, filling in the session
// service and agent loader the live tests share. Every handler return sends on
// exits, which is buffered so a test that never reads it cannot wedge a
// handler.
func startRunLiveServer(
	t *testing.T,
	cfg RuntimeAPIControllerConfig,
	runLiveFn func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error),
) (wsURL string, exits <-chan struct{}) {
	t.Helper()

	baseAgent, err := agent.New(agent.Config{Name: testLiveAppName})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}
	liveAgent := &mockLiveAgent{Agent: baseAgent, runLiveFn: runLiveFn}

	id := fakes.SessionKey{AppName: testLiveAppName, UserID: testLiveUserID, SessionID: testLiveSessionID}
	cfg.SessionService = &fakes.FakeSessionService{
		Sessions: map[fakes.SessionKey]fakes.TestSession{
			id: {
				Id:            id,
				SessionState:  fakes.TestState{},
				SessionEvents: fakes.TestEvents{},
				UpdatedAt:     time.Now(),
			},
		},
	}
	cfg.AgentLoader = agent.NewSingleLoader(liveAgent)

	controller := NewRuntimeAPIControllerWithConfig(cfg)
	handlerExits := make(chan struct{}, testMaxHandlerExits)
	handler := NewErrorHandler(controller.RunLiveHandler)
	server := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		defer func() { handlerExits <- struct{}{} }()
		handler(rw, req)
	}))
	t.Cleanup(server.Close)

	return "ws" + strings.TrimPrefix(server.URL, "http") +
		"/run_live?appName=" + testLiveAppName + "&userId=" + testLiveUserID + "&sessionId=" + testLiveSessionID, handlerExits
}

// dialLive opens one /run_live connection and fails the test unless the
// handshake was accepted.
func dialLive(t *testing.T, wsURL string) *websocket.Conn {
	t.Helper()

	conn, response, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("websocket.Dial() failed: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	if response.StatusCode != http.StatusSwitchingProtocols {
		t.Fatalf("handshake status = %d, want %d", response.StatusCode, http.StatusSwitchingProtocols)
	}
	return conn
}

func dialRunLiveHandler(
	t *testing.T,
	runLiveFn func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error),
) (*websocket.Conn, <-chan struct{}) {
	t.Helper()

	wsURL, exits := startRunLiveServer(t, RuntimeAPIControllerConfig{}, runLiveFn)
	return dialLive(t, wsURL), exits
}

func waitForHandlerExit(t *testing.T, handlerDone <-chan struct{}) {
	t.Helper()

	select {
	case <-handlerDone:
	case <-time.After(time.Second):
		t.Fatal("RunLiveHandler did not exit")
	}
}

func readCloseError(t *testing.T, conn *websocket.Conn) *websocket.CloseError {
	t.Helper()
	if err := conn.SetReadDeadline(time.Now().Add(testWebSocketReadTimeout)); err != nil {
		t.Fatalf("SetReadDeadline() failed: %v", err)
	}
	_, _, err := conn.ReadMessage()
	var closeErr *websocket.CloseError
	if !errors.As(err, &closeErr) {
		t.Fatalf("ReadMessage() error = %v, want *websocket.CloseError", err)
	}
	return closeErr
}

func waitForLiveRequest(t *testing.T, liveSession *recordingLiveSession) agent.LiveRequest {
	t.Helper()

	select {
	case req := <-liveSession.requests:
		return req
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for request to reach the live session")
		return agent.LiveRequest{}
	}
}

func closeLiveClient(t *testing.T, conn *websocket.Conn, liveSession *recordingLiveSession) {
	t.Helper()

	if err := conn.WriteJSON(models.LiveRequest{Close: true}); err != nil {
		t.Fatalf("WriteJSON(close) failed: %v", err)
	}
	select {
	case <-liveSession.closed:
	case <-time.After(time.Second):
		t.Fatal("live session was not closed after the client close request")
	}
}

func TestRunLiveHandler_StreamsEventsOverWebSocket(t *testing.T) {
	liveSession := newRecordingLiveSession()
	wantEvent := makeEvent("invocation-1", testLiveAppName, "Hello from live agent")
	conn, _ := dialRunLiveHandler(t, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return liveSession, func(yield func(*session.Event, error) bool) {
			yield(wantEvent, nil)
		}, nil
	})

	var got models.Event
	if err := conn.SetReadDeadline(time.Now().Add(testWebSocketReadTimeout)); err != nil {
		t.Fatalf("SetReadDeadline() failed: %v", err)
	}
	if err := conn.ReadJSON(&got); err != nil {
		t.Fatalf("ReadJSON() failed: %v", err)
	}
	if got.InvocationID != wantEvent.InvocationID {
		t.Errorf("InvocationID = %q, want %q", got.InvocationID, wantEvent.InvocationID)
	}
	if got.Author != testLiveAppName {
		t.Errorf("Author = %q, want %q", got.Author, testLiveAppName)
	}
	if got.Content == nil || len(got.Content.Parts) != 1 || got.Content.Parts[0].Text != "Hello from live agent" {
		t.Errorf("Content = %#v, want one text part containing live-agent response", got.Content)
	}

	select {
	case <-liveSession.closed:
	case <-time.After(time.Second):
		t.Fatal("live session was not closed after the event stream ended")
	}
}

func TestRunLiveHandler_ForwardsTextMessages(t *testing.T) {
	liveSession := newRecordingLiveSession()
	conn, _ := dialRunLiveHandler(t, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return liveSession, func(yield func(*session.Event, error) bool) {
			<-liveSession.closed
		}, nil
	})

	wantContent := &genai.Content{
		Role:  genai.RoleUser,
		Parts: []*genai.Part{{Text: "Hello over WebSocket"}},
	}
	if err := conn.WriteJSON(models.LiveRequest{Content: wantContent}); err != nil {
		t.Fatalf("WriteJSON(text message) failed: %v", err)
	}

	got := waitForLiveRequest(t, liveSession)
	if got.Content == nil || len(got.Content.Parts) != 1 || got.Content.Parts[0].Text != "Hello over WebSocket" {
		t.Errorf("Content = %#v, want forwarded text content", got.Content)
	}
	if got.RealtimeInput != nil {
		t.Errorf("RealtimeInput = %#v, want nil for a text message", got.RealtimeInput)
	}

	closeLiveClient(t, conn, liveSession)
}

func TestRunLiveHandler_ForwardsBinaryAudio(t *testing.T) {
	liveSession := newRecordingLiveSession()
	conn, _ := dialRunLiveHandler(t, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return liveSession, func(yield func(*session.Event, error) bool) {
			<-liveSession.closed
		}, nil
	})

	wantAudio := []byte{0x01, 0x02, 0x03, 0x04}
	if err := conn.WriteMessage(websocket.BinaryMessage, wantAudio); err != nil {
		t.Fatalf("WriteMessage(binary) failed: %v", err)
	}

	got := waitForLiveRequest(t, liveSession)
	blob, ok := got.RealtimeInput.(*genai.Blob)
	if !ok {
		t.Fatalf("RealtimeInput type = %T, want *genai.Blob", got.RealtimeInput)
	}
	if blob.MIMEType != "audio/pcm;rate=16000" {
		t.Errorf("MIMEType = %q, want %q", blob.MIMEType, "audio/pcm;rate=16000")
	}
	if !bytes.Equal(blob.Data, wantAudio) {
		t.Errorf("Data = %v, want %v", blob.Data, wantAudio)
	}
	if got.Content != nil {
		t.Errorf("Content = %#v, want nil for a binary audio frame", got.Content)
	}

	closeLiveClient(t, conn, liveSession)
}

func TestRunLiveHandler_ForwardsRealtimeInputVariants(t *testing.T) {
	tests := []struct {
		name    string
		message map[string]any
		check   func(*testing.T, agent.LiveRequest)
	}{
		{
			name:    "activity start",
			message: map[string]any{"activityStart": map[string]any{}},
			check: func(t *testing.T, req agent.LiveRequest) {
				if _, ok := req.RealtimeInput.(*genai.ActivityStart); !ok {
					t.Errorf("RealtimeInput type = %T, want *genai.ActivityStart", req.RealtimeInput)
				}
			},
		},
		{
			name:    "activity end",
			message: map[string]any{"activityEnd": map[string]any{}},
			check: func(t *testing.T, req agent.LiveRequest) {
				if _, ok := req.RealtimeInput.(*genai.ActivityEnd); !ok {
					t.Errorf("RealtimeInput type = %T, want *genai.ActivityEnd", req.RealtimeInput)
				}
			},
		},
		{
			name: "blob",
			message: map[string]any{
				"blob": map[string]any{
					"mime_type": "text/plain",
					"data":      []byte("hello"),
				},
			},
			check: func(t *testing.T, req agent.LiveRequest) {
				blob, ok := req.RealtimeInput.(*genai.Blob)
				if !ok {
					t.Fatalf("RealtimeInput type = %T, want *genai.Blob", req.RealtimeInput)
				}
				if blob.MIMEType != "text/plain" {
					t.Errorf("MIMEType = %q, want text/plain", blob.MIMEType)
				}
				if !bytes.Equal(blob.Data, []byte("hello")) {
					t.Errorf("Data = %q, want hello", blob.Data)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			liveSession := newRecordingLiveSession()
			conn, _ := dialRunLiveHandler(t, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
				return liveSession, func(yield func(*session.Event, error) bool) {
					<-liveSession.closed
				}, nil
			})

			if err := conn.WriteJSON(tt.message); err != nil {
				t.Fatalf("WriteJSON() failed: %v", err)
			}
			tt.check(t, waitForLiveRequest(t, liveSession))
			closeLiveClient(t, conn, liveSession)
		})
	}
}

func TestRunLiveHandler_ClientDisconnectStopsLiveRun(t *testing.T) {
	tests := []struct {
		name       string
		disconnect func(*websocket.Conn) error
	}{
		{
			name: "normal close frame",
			disconnect: func(conn *websocket.Conn) error {
				return conn.WriteControl(
					websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, "client finished"),
					time.Now().Add(time.Second),
				)
			},
		},
		{
			name: "abrupt connection close",
			disconnect: func(conn *websocket.Conn) error {
				return conn.Close()
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			liveSession := newRecordingLiveSession()
			iteratorStarted := make(chan struct{})
			iteratorDone := make(chan struct{})
			conn, handlerDone := dialRunLiveHandler(t, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
				return liveSession, func(yield func(*session.Event, error) bool) {
					close(iteratorStarted)
					defer close(iteratorDone)
					<-liveSession.closed
				}, nil
			})

			select {
			case <-iteratorStarted:
			case <-time.After(time.Second):
				t.Fatal("live event iterator did not start")
			}

			if err := tt.disconnect(conn); err != nil {
				t.Fatalf("disconnecting WebSocket client failed: %v", err)
			}

			select {
			case <-liveSession.closed:
			case <-time.After(time.Second):
				t.Fatal("live session was not closed after the WebSocket client disconnected")
			}
			select {
			case <-iteratorDone:
			case <-time.After(time.Second):
				t.Fatal("live event iterator did not exit after the session was closed")
			}
			waitForHandlerExit(t, handlerDone)
		})
	}
}

func TestRunLiveHandler_RunLiveErrorSendsCloseFrameAndDrainsReply(t *testing.T) {
	wantErr := errors.New("live agent failed")
	conn, handlerDone := dialRunLiveHandler(t, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return nil, nil, wantErr
	})

	// Disable gorilla's automatic close reply so the test can verify that the
	// handler waits for the explicit client acknowledgement below.
	conn.SetCloseHandler(func(code int, text string) error {
		return nil
	})
	closeErr := readCloseError(t, conn)
	if closeErr.Code != websocket.CloseInternalServerErr {
		t.Errorf("close code = %d, want %d", closeErr.Code, websocket.CloseInternalServerErr)
	}
	if closeErr.Text != wantErr.Error() {
		t.Errorf("close reason = %q, want %q", closeErr.Text, wantErr.Error())
	}
	select {
	case <-handlerDone:
		t.Fatal("RunLiveHandler returned before receiving the close reply")
	case <-time.After(testCloseDrainGracePeriod):
	}
	if err := conn.WriteControl(
		websocket.CloseMessage,
		websocket.FormatCloseMessage(closeErr.Code, "client acknowledged"),
		time.Now().Add(time.Second),
	); err != nil {
		t.Fatalf("WriteControl(close reply) failed: %v", err)
	}
	// Keep this timeout much shorter than waitForHandlerExit's one-second
	// budget. A longer wait would allow a handler that sleeps instead of
	// draining the close reply to pass this test.
	select {
	case <-handlerDone:
	case <-time.After(testCloseReplyExitTimeout):
		t.Fatalf("RunLiveHandler did not exit promptly after the close reply")
	}
}

func TestRunLiveHandler_PongsDoNotExtendCloseDrain(t *testing.T) {
	conn, handlerDone := dialRunLiveHandler(t, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return nil, nil, errors.New("live agent failed")
	})

	// The client never reads, so the server's close frame goes unanswered, and
	// it keeps sending pongs. The drain after the close frame has to end on its
	// own one-second deadline all the same.
	stop := make(chan struct{})
	t.Cleanup(func() { close(stop) })
	go func() {
		ticker := time.NewTicker(20 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
			}
			if err := conn.WriteControl(websocket.PongMessage, nil, time.Now().Add(time.Second)); err != nil {
				return
			}
		}
	}()

	select {
	case <-handlerDone:
	case <-time.After(3 * time.Second):
		t.Fatal("RunLiveHandler was still draining 3s after its close frame while the client sent pongs")
	}
}

func TestRunLiveHandler_IteratorErrorSendsCloseFrame(t *testing.T) {
	wantErr := errors.New("stream failed")
	liveSession := newRecordingLiveSession()
	consumerPanic := make(chan any, 1)
	continuedAfterError := make(chan struct{}, 1)
	conn, handlerDone := dialRunLiveHandler(t, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return liveSession, func(yield func(*session.Event, error) bool) {
			// Surface a consumer panic to the test goroutine instead of leaving
			// it visible only in net/http's server log.
			defer func() {
				if recovered := recover(); recovered != nil {
					consumerPanic <- recovered
					panic(recovered)
				}
			}()
			if !yield(makeEvent("invocation-1", testLiveAppName, "before failure"), nil) {
				return
			}
			if !yield(nil, wantErr) {
				return
			}
			// Real event producers may continue after a per-item error. The
			// handler must stop consuming before it reaches this event.
			continuedAfterError <- struct{}{}
			yield(makeEvent("invocation-2", testLiveAppName, "after failure"), nil)
		}, nil
	})

	var event models.Event
	if err := conn.SetReadDeadline(time.Now().Add(testWebSocketReadTimeout)); err != nil {
		t.Fatalf("SetReadDeadline() failed: %v", err)
	}
	if err := conn.ReadJSON(&event); err != nil {
		t.Fatalf("ReadJSON() failed: %v", err)
	}
	closeErr := readCloseError(t, conn)
	if closeErr.Code != websocket.CloseInternalServerErr {
		t.Errorf("close code = %d, want %d", closeErr.Code, websocket.CloseInternalServerErr)
	}
	if closeErr.Text != wantErr.Error() {
		t.Errorf("close reason = %q, want %q", closeErr.Text, wantErr.Error())
	}
	waitForHandlerExit(t, handlerDone)
	select {
	case recovered := <-consumerPanic:
		t.Fatalf("RunLiveHandler panicked after the iterator error: %v", recovered)
	default:
	}
	select {
	case <-continuedAfterError:
		t.Fatal("RunLiveHandler continued consuming events after the iterator error")
	default:
	}
}

func TestRunLiveHandler_GracefulClientCloseDoesNotLogWriteError(t *testing.T) {
	var logs synchronizedBuffer
	originalOutput := log.Writer()
	log.SetOutput(&logs)
	t.Cleanup(func() { log.SetOutput(originalOutput) })

	liveSession := newRecordingLiveSession()
	event := makeEvent("invocation-1", testLiveAppName, "tick")
	conn, handlerDone := dialRunLiveHandler(t, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return liveSession, func(yield func(*session.Event, error) bool) {
			for {
				if !yield(event, nil) {
					return
				}
				// Give the handler's reader goroutine time to process the client
				// close frame before the write loop can fill the socket buffer.
				time.Sleep(time.Millisecond)
			}
		}, nil
	})

	if err := conn.WriteControl(
		websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseNormalClosure, "client finished"),
		time.Now().Add(time.Second),
	); err != nil {
		t.Fatalf("WriteControl(close) failed: %v", err)
	}
	waitForHandlerExit(t, handlerDone)
	if got := logs.String(); strings.Contains(got, "WebSocket write error for app "+testLiveAppName) {
		t.Errorf("graceful close produced a write-error log line: %q", got)
	}
}

func TestRunLiveHandler_LogsWriteJSONFailure(t *testing.T) {
	var logs synchronizedBuffer
	originalOutput := log.Writer()
	log.SetOutput(&logs)
	t.Cleanup(func() { log.SetOutput(originalOutput) })

	liveSession := newRecordingLiveSession()
	invalidEvent := makeEvent("invocation-1", testLiveAppName, "invalid output")
	invalidEvent.Output = make(chan struct{}) // json.Marshal cannot encode channels.
	_, handlerDone := dialRunLiveHandler(t, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return liveSession, func(yield func(*session.Event, error) bool) {
			yield(invalidEvent, nil)
		}, nil
	})

	waitForHandlerExit(t, handlerDone)
	if got := logs.String(); !strings.Contains(got, "WebSocket write error for app "+testLiveAppName) {
		t.Errorf("logs = %q, want WebSocket write failure with app name", got)
	}
}

func TestTruncateCloseReason(t *testing.T) {
	longErr := "live session: reconnect budget exhausted: 5 consecutive attempts delivered no content: " +
		"failed to receive message: websocket: close 1006 (abnormal closure): unexpected EOF"

	tests := []struct {
		name   string
		reason string
	}{
		{name: "short reason is untouched", reason: "agent not found"},
		{name: "exactly at the limit", reason: strings.Repeat("a", maxCloseReason)},
		{name: "over the limit", reason: longErr},
		{name: "multi-byte rune straddling the cut", reason: strings.Repeat("é", maxCloseReason)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := truncateCloseReason(tc.reason)
			if len(got) > maxCloseReason {
				t.Errorf("len = %d, want at most %d: gorilla drops the whole frame", len(got), maxCloseReason)
			}
			if !utf8.ValidString(got) {
				t.Error("result is not valid UTF-8, which a close reason must be")
			}
			if !strings.HasPrefix(tc.reason, got) {
				t.Errorf("result %q is not a prefix of the reason", got)
			}
			if len(tc.reason) <= maxCloseReason && got != tc.reason {
				t.Errorf("a reason that already fits was altered: %q", got)
			}
		})
	}

	// The frame gorilla actually builds must be acceptable to it.
	if got := len(websocket.FormatCloseMessage(websocket.CloseInternalServerErr, truncateCloseReason(longErr))); got > 125 {
		t.Errorf("close frame payload = %d bytes, want at most 125", got)
	}
}

// blockingLiveRun returns a run function whose event stream produces nothing
// and ends only when the live session is closed, so a test can watch the
// handler's reaction to the client rather than to the agent.
func blockingLiveRun(liveSession *recordingLiveSession) func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	return func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return liveSession, func(yield func(*session.Event, error) bool) {
			<-liveSession.closed
		}, nil
	}
}

func TestRunLiveHandler_ClosesConnectionOnOversizedMessage(t *testing.T) {
	const readLimit = 64

	liveSession := newRecordingLiveSession()
	wsURL, exits := startRunLiveServer(t, RuntimeAPIControllerConfig{
		MaxLiveMessageBytes: readLimit,
	}, blockingLiveRun(liveSession))
	conn := dialLive(t, wsURL)

	if err := conn.WriteMessage(websocket.BinaryMessage, make([]byte, readLimit*2)); err != nil {
		t.Fatalf("WriteMessage() failed: %v", err)
	}

	if got, want := readCloseError(t, conn).Code, websocket.CloseMessageTooBig; got != want {
		t.Errorf("close code = %d, want %d", got, want)
	}
	select {
	case req := <-liveSession.requests:
		t.Errorf("oversized message reached the live session as %#v, want it dropped", req)
	default:
	}
	waitForHandlerExit(t, exits)
}

func TestRunLiveHandler_AcceptsMessageAtReadLimit(t *testing.T) {
	const readLimit = 1024

	liveSession := newRecordingLiveSession()
	wsURL, _ := startRunLiveServer(t, RuntimeAPIControllerConfig{
		MaxLiveMessageBytes: readLimit,
	}, blockingLiveRun(liveSession))
	conn := dialLive(t, wsURL)

	if err := conn.WriteMessage(websocket.BinaryMessage, make([]byte, readLimit)); err != nil {
		t.Fatalf("WriteMessage() failed: %v", err)
	}

	req := waitForLiveRequest(t, liveSession)
	blob, ok := req.RealtimeInput.(*genai.Blob)
	if !ok {
		t.Fatalf("RealtimeInput type = %T, want *genai.Blob", req.RealtimeInput)
	}
	if len(blob.Data) != readLimit {
		t.Errorf("forwarded blob = %d bytes, want %d", len(blob.Data), readLimit)
	}
}

func TestRunLiveHandler_ClosesUnresponsiveConnection(t *testing.T) {
	const keepaliveTimeout = 100 * time.Millisecond

	liveSession := newRecordingLiveSession()
	// The client never reads, so gorilla never answers the server's pings and
	// the keepalive is the only thing that can end the connection.
	wsURL, exits := startRunLiveServer(t, RuntimeAPIControllerConfig{
		LiveKeepaliveTimeout: keepaliveTimeout,
	}, blockingLiveRun(liveSession))
	dialLive(t, wsURL)

	select {
	case <-liveSession.closed:
	case <-time.After(time.Second):
		t.Fatal("unresponsive connection was not dropped after the keepalive timeout")
	}
	waitForHandlerExit(t, exits)
}

// answerPings starts a goroutine that reads conn until a read fails. Reading is
// what makes gorilla answer a ping, so this is a client that is present but has
// nothing to say.
func answerPings(conn *websocket.Conn) {
	go func() {
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	}()
}

func TestRunLiveHandler_PongKeepsConnectionOpen(t *testing.T) {
	const keepaliveTimeout = 100 * time.Millisecond

	liveSession := newRecordingLiveSession()
	wsURL, _ := startRunLiveServer(t, RuntimeAPIControllerConfig{
		LiveKeepaliveTimeout: keepaliveTimeout,
	}, blockingLiveRun(liveSession))
	answerPings(dialLive(t, wsURL))

	select {
	case <-liveSession.closed:
		t.Error("connection was dropped while the client was answering pings")
	case <-time.After(5 * keepaliveTimeout):
	}
}

func TestRunLiveHandler_KeepsResponsiveClientThroughSlowSetup(t *testing.T) {
	const keepaliveTimeout = 100 * time.Millisecond

	liveSession := newRecordingLiveSession()
	wsURL, _ := startRunLiveServer(t, RuntimeAPIControllerConfig{
		LiveKeepaliveTimeout: keepaliveTimeout,
	}, func(ic agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		time.Sleep(3 * keepaliveTimeout)
		return blockingLiveRun(liveSession)(ic)
	})
	answerPings(dialLive(t, wsURL))

	select {
	case <-liveSession.closed:
		t.Error("client answering pings was dropped after a session setup longer than the keepalive timeout")
	case <-time.After(5 * keepaliveTimeout):
	}
}

// slowSendLiveSession takes delay to accept each request, the way a live flow
// does while its model connection is dialing or reconnecting.
type slowSendLiveSession struct {
	*recordingLiveSession
	delay time.Duration
}

func (s *slowSendLiveSession) Send(req agent.LiveRequest) error {
	time.Sleep(s.delay)
	return s.recordingLiveSession.Send(req)
}

func TestRunLiveHandler_KeepsResponsiveClientThroughSlowSend(t *testing.T) {
	const keepaliveTimeout = 100 * time.Millisecond

	recorder := newRecordingLiveSession()
	liveSession := &slowSendLiveSession{recordingLiveSession: recorder, delay: 3 * keepaliveTimeout}
	wsURL, _ := startRunLiveServer(t, RuntimeAPIControllerConfig{
		LiveKeepaliveTimeout: keepaliveTimeout,
	}, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return liveSession, func(yield func(*session.Event, error) bool) {
			<-recorder.closed
		}, nil
	})
	conn := dialLive(t, wsURL)
	answerPings(conn)

	if err := conn.WriteMessage(websocket.BinaryMessage, []byte("audio")); err != nil {
		t.Fatalf("WriteMessage() failed: %v", err)
	}

	select {
	case <-recorder.closed:
		t.Error("client answering pings was dropped while the agent took longer than the keepalive timeout to accept its message")
	case <-time.After(8 * keepaliveTimeout):
	}
}

func TestRunLiveHandler_IteratorErrorAfterQuietSpellSendsCloseFrame(t *testing.T) {
	const keepaliveTimeout = 100 * time.Millisecond

	wantErr := errors.New("stream failed")
	liveSession := newRecordingLiveSession()
	wsURL, exits := startRunLiveServer(t, RuntimeAPIControllerConfig{
		LiveKeepaliveTimeout: keepaliveTimeout,
	}, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return liveSession, func(yield func(*session.Event, error) bool) {
			if !yield(makeEvent("invocation-1", testLiveAppName, "before failure"), nil) {
				return
			}
			// Quiet for longer than the keepalive timeout, so the deadline set
			// for the write above has expired by the time the error arrives.
			time.Sleep(3 * keepaliveTimeout)
			yield(nil, wantErr)
		}, nil
	})
	conn := dialLive(t, wsURL)

	var event models.Event
	if err := conn.SetReadDeadline(time.Now().Add(testWebSocketReadTimeout)); err != nil {
		t.Fatalf("SetReadDeadline() failed: %v", err)
	}
	if err := conn.ReadJSON(&event); err != nil {
		t.Fatalf("ReadJSON() failed: %v", err)
	}
	// Still reading, so the pings sent while the agent is quiet are answered.
	closeErr := readCloseError(t, conn)
	if closeErr.Code != websocket.CloseInternalServerErr {
		t.Errorf("close code = %d, want %d", closeErr.Code, websocket.CloseInternalServerErr)
	}
	if closeErr.Text != wantErr.Error() {
		t.Errorf("close reason = %q, want %q", closeErr.Text, wantErr.Error())
	}
	waitForHandlerExit(t, exits)
}

func TestRunLiveHandler_DropsPeerThatStopsReading(t *testing.T) {
	const keepaliveTimeout = 200 * time.Millisecond

	liveSession := newRecordingLiveSession()
	// Events large enough to fill the socket buffers within a few writes, so
	// the handler ends up blocked writing to a peer that reads nothing.
	text := strings.Repeat("a", 256<<10)
	wsURL, exits := startRunLiveServer(t, RuntimeAPIControllerConfig{
		LiveKeepaliveTimeout: keepaliveTimeout,
		MaxLiveSessions:      1,
	}, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return liveSession, func(yield func(*session.Event, error) bool) {
			for {
				select {
				case <-liveSession.closed:
					return
				default:
				}
				if !yield(makeEvent("inv", testLiveAppName, text), nil) {
					return
				}
			}
		}, nil
	})
	dialLive(t, wsURL)

	waitForHandlerExit(t, exits)
	// With a cap of one, this handshake succeeds only if the stalled peer's
	// slot was released.
	dialLive(t, wsURL)
}

// stallingConn is the server's end of a connection whose peer can stop
// accepting data. While stalled, a write blocks until the stall ends or its
// write deadline passes, and signals blocked when it starts waiting.
type stallingConn struct {
	net.Conn
	blocked chan struct{}

	mu       sync.Mutex
	deadline time.Time
	// resume is non-nil while writes stall, and is closed to end the stall.
	resume chan struct{}
}

func (c *stallingConn) SetWriteDeadline(t time.Time) error {
	c.mu.Lock()
	c.deadline = t
	c.mu.Unlock()
	return c.Conn.SetWriteDeadline(t)
}

func (c *stallingConn) stall() {
	c.mu.Lock()
	c.resume = make(chan struct{})
	c.mu.Unlock()
}

func (c *stallingConn) unstall() {
	c.mu.Lock()
	close(c.resume)
	c.resume = nil
	c.mu.Unlock()
}

func (c *stallingConn) Write(p []byte) (int, error) {
	c.mu.Lock()
	resume, deadline := c.resume, c.deadline
	c.mu.Unlock()
	if resume != nil {
		var expired <-chan time.Time
		if !deadline.IsZero() {
			timer := time.NewTimer(time.Until(deadline))
			defer timer.Stop()
			expired = timer.C
		}
		select {
		case c.blocked <- struct{}{}:
		default:
		}
		select {
		case <-resume:
		case <-expired:
			return 0, os.ErrDeadlineExceeded
		}
	}
	return c.Conn.Write(p)
}

// stallingListener wraps each accepted connection in a stallingConn and hands
// it to the test on conns.
type stallingListener struct {
	net.Listener
	conns chan *stallingConn
}

func (l *stallingListener) Accept() (net.Conn, error) {
	conn, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	sc := &stallingConn{Conn: conn, blocked: make(chan struct{}, 1)}
	l.conns <- sc
	return sc, nil
}

func TestRunLiveHandler_PingWriteDeadlineIsKeepaliveTimeout(t *testing.T) {
	const keepaliveTimeout = 200 * time.Millisecond

	liveSession := newRecordingLiveSession()
	baseAgent, err := agent.New(agent.Config{Name: testLiveAppName})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}
	// Echoes each binary message back as an event, so the test can tell
	// whether the connection still carries writes.
	liveAgent := &mockLiveAgent{Agent: baseAgent, runLiveFn: func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return liveSession, func(yield func(*session.Event, error) bool) {
			for {
				select {
				case req := <-liveSession.requests:
					blob, ok := req.RealtimeInput.(*genai.Blob)
					if ok && !yield(makeEvent("inv", testLiveAppName, string(blob.Data)), nil) {
						return
					}
				case <-liveSession.closed:
					return
				}
			}
		}, nil
	}}
	id := fakes.SessionKey{AppName: testLiveAppName, UserID: testLiveUserID, SessionID: testLiveSessionID}
	controller := NewRuntimeAPIControllerWithConfig(RuntimeAPIControllerConfig{
		SessionService: &fakes.FakeSessionService{
			Sessions: map[fakes.SessionKey]fakes.TestSession{
				id: {
					Id:            id,
					SessionState:  fakes.TestState{},
					SessionEvents: fakes.TestEvents{},
					UpdatedAt:     time.Now(),
				},
			},
		},
		AgentLoader:          agent.NewSingleLoader(liveAgent),
		LiveKeepaliveTimeout: keepaliveTimeout,
	})
	server := httptest.NewUnstartedServer(NewErrorHandler(controller.RunLiveHandler))
	listener := &stallingListener{Listener: server.Listener, conns: make(chan *stallingConn, 1)}
	server.Listener = listener
	server.Start()
	t.Cleanup(server.Close)

	conn := dialLive(t, "ws"+strings.TrimPrefix(server.URL, "http")+
		"/run_live?appName="+testLiveAppName+"&userId="+testLiveUserID+"&sessionId="+testLiveSessionID)
	serverConn := <-listener.conns

	echoes := make(chan string, 1)
	readErr := make(chan error, 1)
	go func() {
		for {
			var event models.Event
			if err := conn.ReadJSON(&event); err != nil {
				readErr <- err
				return
			}
			if event.Content != nil && len(event.Content.Parts) == 1 {
				echoes <- event.Content.Parts[0].Text
			}
		}
	}()
	// Pongs of the client's own keep the server's read deadline from expiring
	// while its pings are stalled, so only a failed write can end the
	// connection.
	done := make(chan struct{})
	t.Cleanup(func() { close(done) })
	go func() {
		ticker := time.NewTicker(keepaliveTimeout / 4)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				if err := conn.WriteControl(websocket.PongMessage, nil, time.Now().Add(keepaliveTimeout)); err != nil {
					return
				}
			}
		}
	}()

	// The agent is quiet, so the first write to meet the stall is a ping.
	stallPing := func(d time.Duration) {
		t.Helper()
		serverConn.stall()
		select {
		case <-serverConn.blocked:
		case <-time.After(time.Second):
			serverConn.unstall()
			t.Fatal("no ping reached the stalled connection")
		}
		time.Sleep(d)
		serverConn.unstall()
	}
	send := func(text string) {
		t.Helper()
		if err := conn.WriteMessage(websocket.BinaryMessage, []byte(text)); err != nil {
			t.Fatalf("WriteMessage() failed: %v", err)
		}
	}

	stallPing(keepaliveTimeout / 2)
	send("after short stall")
	select {
	case got := <-echoes:
		if got != "after short stall" {
			t.Errorf("echo = %q, want %q", got, "after short stall")
		}
	case err := <-readErr:
		t.Fatalf("connection ended after a ping stall shorter than the keepalive timeout: %v", err)
	case <-time.After(time.Second):
		t.Fatal("echo never arrived after a ping stall shorter than the keepalive timeout")
	}

	stallPing(2 * keepaliveTimeout)
	send("after long stall")
	select {
	case got := <-echoes:
		t.Errorf("echo %q arrived after a ping stalled for twice the keepalive timeout, want the connection closed", got)
	case <-readErr:
	case <-time.After(time.Second):
		t.Fatal("connection neither closed nor echoed after a ping stalled for twice the keepalive timeout")
	}
}

func TestRunLiveHandler_RefusesUpgradePastSessionLimit(t *testing.T) {
	liveSession := newRecordingLiveSession()
	wsURL, exits := startRunLiveServer(t, RuntimeAPIControllerConfig{
		MaxLiveSessions: 1,
	}, blockingLiveRun(liveSession))
	first := dialLive(t, wsURL)

	_, response, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if !errors.Is(err, websocket.ErrBadHandshake) {
		t.Fatalf("second Dial() error = %v, want %v", err, websocket.ErrBadHandshake)
	}
	if got, want := response.StatusCode, http.StatusServiceUnavailable; got != want {
		t.Errorf("second handshake status = %d, want %d", got, want)
	}
	// The refused request ran the handler too, so take its exit before waiting
	// on the one that releases the slot.
	waitForHandlerExit(t, exits)

	// The slot goes back when the first session ends, so the cap bounds
	// concurrency rather than the number of sessions the server ever serves.
	_ = first.Close()
	waitForHandlerExit(t, exits)
	dialLive(t, wsURL)
}

func TestNewRuntimeAPIControllerWithConfig_LiveLimits(t *testing.T) {
	tests := []struct {
		name          string
		cfg           RuntimeAPIControllerConfig
		wantBytes     int64
		wantKeepalive time.Duration
		wantPing      time.Duration
		wantSlotCount int
		wantCapped    bool
	}{
		{
			name:          "unset takes the defaults, with no session cap",
			wantBytes:     defaultMaxLiveMessageBytes,
			wantKeepalive: defaultLiveKeepaliveTimeout,
			wantPing:      defaultLiveKeepaliveTimeout / 2,
		},
		{
			name: "explicit values are kept",
			cfg: RuntimeAPIControllerConfig{
				MaxLiveMessageBytes:  4096,
				LiveKeepaliveTimeout: time.Minute,
				MaxLiveSessions:      3,
			},
			wantBytes:     4096,
			wantKeepalive: time.Minute,
			wantPing:      30 * time.Second,
			wantSlotCount: 3,
			wantCapped:    true,
		},
		{
			name: "negative turns each limit off",
			cfg: RuntimeAPIControllerConfig{
				MaxLiveMessageBytes:  -1,
				LiveKeepaliveTimeout: -1,
				MaxLiveSessions:      -1,
			},
			wantBytes:     -1,
			wantKeepalive: -1,
		},
		{
			// Halving would give zero, which would make time.NewTicker panic
			// on every /run_live request.
			name:          "a one-nanosecond keepalive still pings",
			cfg:           RuntimeAPIControllerConfig{LiveKeepaliveTimeout: time.Nanosecond},
			wantBytes:     defaultMaxLiveMessageBytes,
			wantKeepalive: time.Nanosecond,
			wantPing:      time.Nanosecond,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := NewRuntimeAPIControllerWithConfig(tc.cfg)

			if c.maxLiveMessageBytes != tc.wantBytes {
				t.Errorf("maxLiveMessageBytes = %d, want %d", c.maxLiveMessageBytes, tc.wantBytes)
			}
			if c.liveKeepaliveTimeout != tc.wantKeepalive {
				t.Errorf("liveKeepaliveTimeout = %v, want %v", c.liveKeepaliveTimeout, tc.wantKeepalive)
			}
			if c.livePingInterval != tc.wantPing {
				t.Errorf("livePingInterval = %v, want %v", c.livePingInterval, tc.wantPing)
			}
			if (c.liveSlots != nil) != tc.wantCapped {
				t.Fatalf("session cap in effect = %t, want %t", c.liveSlots != nil, tc.wantCapped)
			}
			if tc.wantCapped && cap(c.liveSlots) != tc.wantSlotCount {
				t.Errorf("session cap = %d, want %d", cap(c.liveSlots), tc.wantSlotCount)
			}
		})
	}
}

func TestAcquireLiveSlot_UncappedAlwaysGrants(t *testing.T) {
	c := NewRuntimeAPIControllerWithConfig(RuntimeAPIControllerConfig{})

	for i := range 3 {
		if _, ok := c.acquireLiveSlot(); !ok {
			t.Fatalf("acquireLiveSlot() #%d refused with the cap disabled", i+1)
		}
	}
}
