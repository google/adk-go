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
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

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
	testCloseDrainMaxLatency  = 500 * time.Millisecond
)

func dialRunLiveHandler(
	t *testing.T,
	runLiveFn func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error),
) (*websocket.Conn, <-chan struct{}) {
	t.Helper()

	baseAgent, err := agent.New(agent.Config{Name: testLiveAppName})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}
	liveAgent := &mockLiveAgent{Agent: baseAgent, runLiveFn: runLiveFn}

	id := fakes.SessionKey{AppName: testLiveAppName, UserID: testLiveUserID, SessionID: testLiveSessionID}
	sessionService := &fakes.FakeSessionService{
		Sessions: map[fakes.SessionKey]fakes.TestSession{
			id: {
				Id:            id,
				SessionState:  fakes.TestState{},
				SessionEvents: fakes.TestEvents{},
				UpdatedAt:     time.Now(),
			},
		},
	}

	controller := NewRuntimeAPIControllerWithConfig(RuntimeAPIControllerConfig{
		SessionService: sessionService,
		AgentLoader:    agent.NewSingleLoader(liveAgent),
	})
	handlerDone := make(chan struct{})
	handler := NewErrorHandler(controller.RunLiveHandler)
	server := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		defer close(handlerDone)
		handler(rw, req)
	}))
	t.Cleanup(server.Close)

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") +
		"/run_live?appName=" + testLiveAppName + "&userId=" + testLiveUserID + "&sessionId=" + testLiveSessionID
	conn, response, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("websocket.Dial() failed: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	if response.StatusCode != http.StatusSwitchingProtocols {
		t.Fatalf("handshake status = %d, want %d", response.StatusCode, http.StatusSwitchingProtocols)
	}
	return conn, handlerDone
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

	closeReceived := make(chan struct{})
	conn.SetCloseHandler(func(code int, text string) error {
		close(closeReceived)
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
	case <-closeReceived:
	case <-time.After(time.Second):
		t.Fatal("client did not receive server close frame")
	}

	select {
	case <-handlerDone:
		t.Fatal("RunLiveHandler returned before receiving the close reply")
	case <-time.After(testCloseDrainGracePeriod):
	}
	replySentAt := time.Now()
	if err := conn.WriteControl(
		websocket.CloseMessage,
		websocket.FormatCloseMessage(closeErr.Code, "client acknowledged"),
		time.Now().Add(time.Second),
	); err != nil {
		t.Fatalf("WriteControl(close reply) failed: %v", err)
	}
	waitForHandlerExit(t, handlerDone)
	if elapsed := time.Since(replySentAt); elapsed > testCloseDrainMaxLatency {
		t.Errorf("RunLiveHandler took %v to exit after the close reply, want less than %v", elapsed, testCloseDrainMaxLatency)
	}
}

func TestRunLiveHandler_IteratorErrorSendsCloseFrame(t *testing.T) {
	wantErr := errors.New("stream failed")
	liveSession := newRecordingLiveSession()
	conn, handlerDone := dialRunLiveHandler(t, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return liveSession, func(yield func(*session.Event, error) bool) {
			if !yield(makeEvent("invocation-1", testLiveAppName, "before failure"), nil) {
				return
			}
			yield(nil, wantErr)
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
