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

package adkrest_test

import (
	"errors"
	"iter"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/session"
)

// heldLiveAgent answers a live run with a stream that yields nothing and ends
// when the live session is closed, the way a real live flow ends with its
// connection. Until then a test can hold a /run_live connection open for as
// long as it needs one.
type heldLiveAgent struct {
	agent.Agent
}

func (a *heldLiveAgent) RunLive(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	s := &closableLiveSession{closed: make(chan struct{})}
	return s, func(yield func(*session.Event, error) bool) {
		<-s.closed
	}, nil
}

type closableLiveSession struct {
	once   sync.Once
	closed chan struct{}
}

func (*closableLiveSession) Send(agent.LiveRequest) error { return nil }

func (s *closableLiveSession) Close() error {
	s.once.Do(func() { close(s.closed) })
	return nil
}

// serveHeldRunLive starts an [adkrest.Server] built from cfg around a
// [heldLiveAgent] and returns the /run_live URL of a session that exists.
func serveHeldRunLive(t *testing.T, cfg adkrest.ServerConfig) string {
	t.Helper()

	root, err := agent.New(agent.Config{Name: guardTestApp, Description: "root agent"})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	// The session has to exist, or the run fails and the handler returns at
	// once, which would pass the session-limit case for the wrong reason.
	sessions := session.InMemoryService()
	if _, err := sessions.Create(t.Context(), &session.CreateRequest{
		AppName:   guardTestApp,
		UserID:    "u1",
		SessionID: "s1",
	}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	cfg.SessionService = sessions
	cfg.MemoryService = memory.InMemoryService()
	cfg.ArtifactService = artifact.InMemoryService()
	cfg.AgentLoader = agent.NewSingleLoader(&heldLiveAgent{Agent: root})
	cfg.BindHost = "127.0.0.1"
	srv, err := adkrest.NewServer(cfg)
	if err != nil {
		t.Fatalf("adkrest.NewServer() error = %v", err)
	}
	ts := httptest.NewServer(srv)
	t.Cleanup(ts.Close)

	return "ws" + strings.TrimPrefix(ts.URL, "http") +
		"/run_live?appName=" + guardTestApp + "&userId=u1&sessionId=s1"
}

// dialHeldRunLive opens a /run_live connection and fails the test unless the
// handshake was accepted.
func dialHeldRunLive(t *testing.T, wsURL string) *websocket.Conn {
	t.Helper()

	conn, response, err := websocket.DefaultDialer.DialContext(t.Context(), wsURL, nil)
	if err != nil {
		t.Fatalf("Dial() error = %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	if got, want := response.StatusCode, http.StatusSwitchingProtocols; got != want {
		t.Fatalf("handshake status = %d, want %d", got, want)
	}
	return conn
}

// TestRunLiveLimitsFromServerConfig pins the wiring between
// [adkrest.ServerConfig] and the runtime controller: each limit set on the
// server has to reach the handler that enforces it. Every case sets a limit
// well below its default, so a limit that is not passed through fails it.
func TestRunLiveLimitsFromServerConfig(t *testing.T) {
	tests := []struct {
		name  string
		cfg   adkrest.ServerConfig
		check func(t *testing.T, wsURL string)
	}{
		{
			name: "MaxLiveMessageBytes",
			cfg:  adkrest.ServerConfig{MaxLiveMessageBytes: 64},
			check: func(t *testing.T, wsURL string) {
				conn := dialHeldRunLive(t, wsURL)
				if err := conn.WriteMessage(websocket.BinaryMessage, make([]byte, 128)); err != nil {
					t.Fatalf("WriteMessage() error = %v", err)
				}

				_ = conn.SetReadDeadline(time.Now().Add(5 * time.Second))
				_, _, err := conn.ReadMessage()
				if !websocket.IsCloseError(err, websocket.CloseMessageTooBig) {
					t.Errorf("ReadMessage() after an oversized message: error = %v, want close code %d", err, websocket.CloseMessageTooBig)
				}
			},
		},
		{
			name: "LiveKeepaliveTimeout",
			cfg:  adkrest.ServerConfig{LiveKeepaliveTimeout: 100 * time.Millisecond},
			check: func(t *testing.T, wsURL string) {
				conn := dialHeldRunLive(t, wsURL)
				// Keep reading, so the drop is visible here, but answer no ping.
				conn.SetPingHandler(func(string) error { return nil })

				_ = conn.SetReadDeadline(time.Now().Add(5 * time.Second))
				_, _, err := conn.ReadMessage()
				var netErr net.Error
				if err == nil || errors.As(err, &netErr) && netErr.Timeout() {
					t.Errorf("ReadMessage() on a client that answers no pings: error = %v, want the server to have dropped the connection", err)
				}
			},
		},
		{
			name: "MaxLiveSessions",
			cfg:  adkrest.ServerConfig{MaxLiveSessions: 1},
			check: func(t *testing.T, wsURL string) {
				// The slot is taken before the upgrade, so a completed
				// handshake means the first connection already holds it.
				dialHeldRunLive(t, wsURL)

				_, response, err := websocket.DefaultDialer.DialContext(t.Context(), wsURL, nil)
				if !errors.Is(err, websocket.ErrBadHandshake) {
					t.Fatalf("second Dial() error = %v, want %v", err, websocket.ErrBadHandshake)
				}
				t.Cleanup(func() { _ = response.Body.Close() })
				if got, want := response.StatusCode, http.StatusServiceUnavailable; got != want {
					t.Errorf("second handshake status = %d, want %d", got, want)
				}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.check(t, serveHeldRunLive(t, tc.cfg))
		})
	}
}
