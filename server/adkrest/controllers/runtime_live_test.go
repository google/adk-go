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
	"errors"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

// liveTestAgent adds the RunLive method the runner requires for a live run.
type liveTestAgent struct {
	agent.Agent
	sess *recordingLiveSession
}

func (a *liveTestAgent) RunLive(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	return a.sess, func(func(*session.Event, error) bool) { <-a.sess.closed }, nil
}

// recordingLiveSession forwards every request it is sent to a channel.
type recordingLiveSession struct {
	requests  chan agent.LiveRequest
	closed    chan struct{}
	closeOnce sync.Once
}

func (s *recordingLiveSession) Send(req agent.LiveRequest) error {
	s.requests <- req
	return nil
}

func (s *recordingLiveSession) Close() error {
	s.closeOnce.Do(func() { close(s.closed) })
	return nil
}

func TestRunLiveHandlerEnforcesMessageSizeLimit(t *testing.T) {
	for _, tc := range []struct {
		name    string
		size    int
		forward bool
	}{
		{name: "at limit is forwarded", size: maxLiveMessageBytes, forward: true},
		{name: "one byte over is rejected", size: maxLiveMessageBytes + 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base, err := agent.New(agent.Config{Name: "live"})
			if err != nil {
				t.Fatalf("agent.New() failed: %v", err)
			}
			liveSession := &recordingLiveSession{requests: make(chan agent.LiveRequest, 1), closed: make(chan struct{})}
			c := NewRuntimeAPIController(session.InMemoryService(), nil, agent.NewSingleLoader(&liveTestAgent{Agent: base, sess: liveSession}), nil, 0, runner.PluginConfig{}, true)

			handlerDone := make(chan struct{})
			srv := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
				defer close(handlerDone)
				if err := c.RunLiveHandler(rw, req); err != nil {
					t.Errorf("RunLiveHandler() failed: %v", err)
				}
			}))
			defer srv.Close()

			url := "ws" + strings.TrimPrefix(srv.URL, "http") + "/run_live?app_name=live&user_id=u1&session_id=s1"
			conn, _, err := websocket.DefaultDialer.Dial(url, nil)
			if err != nil {
				t.Fatalf("Dial() failed: %v", err)
			}
			defer func() {
				_ = conn.Close()
				_ = liveSession.Close()
			}()

			if err := conn.WriteMessage(websocket.BinaryMessage, make([]byte, tc.size)); err != nil {
				t.Fatalf("WriteMessage() failed: %v", err)
			}

			if tc.forward {
				select {
				case got := <-liveSession.requests:
					if blob, ok := got.RealtimeInput.(*genai.Blob); !ok || len(blob.Data) != tc.size {
						t.Fatalf("RealtimeInput = %T, want a *genai.Blob of %d bytes", got.RealtimeInput, tc.size)
					}
				case <-time.After(10 * time.Second):
					t.Fatal("message was not forwarded to the live session")
				}
				return
			}

			_ = conn.SetReadDeadline(time.Now().Add(10 * time.Second))
			_, _, err = conn.ReadMessage()
			var closeErr *websocket.CloseError
			if !errors.As(err, &closeErr) || closeErr.Code != websocket.CloseMessageTooBig {
				t.Fatalf("ReadMessage() error = %v, want a close frame with code %d", err, websocket.CloseMessageTooBig)
			}
			select {
			case req := <-liveSession.requests:
				t.Fatalf("oversized message reached the live session: %d bytes", len(req.RealtimeInput.(*genai.Blob).Data))
			default:
			}
			select {
			case <-handlerDone:
			case <-time.After(10 * time.Second):
				t.Fatal("RunLiveHandler did not return after rejecting the message")
			}
		})
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
