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
	"io"
	"iter"
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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adkrest/internal/fakes"
	"google.golang.org/adk/server/adkrest/internal/models"
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

// mockLiveAgent adds live-run behavior to a regular custom agent, with the live
// stream supplied by the test.
type mockLiveAgent struct {
	agent.Agent
	runLiveFn func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
}

func (a *mockLiveAgent) RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	return a.runLiveFn(ctx)
}

func newRecordingLiveSession() *recordingLiveSession {
	return &recordingLiveSession{
		requests: make(chan agent.LiveRequest, 1),
		closed:   make(chan struct{}),
	}
}

const (
	testLiveAppName          = "testApp"
	testLiveUserID           = "testUser"
	testLiveSessionID        = "testSession"
	testWebSocketReadTimeout = time.Second
	// testMaxHandlerExits is the capacity of the handler-exit channel, large
	// enough for every connection a test in this file opens.
	testMaxHandlerExits = 8
)

// newLiveTestController returns a controller over a session service holding the
// test session. A non-zero keepalive replaces the default keepalive timeout.
func newLiveTestController(loader agent.Loader, keepalive time.Duration) *RuntimeAPIController {
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
	c := NewRuntimeAPIController(sessionService, nil, loader, nil, 0, runner.PluginConfig{}, false)
	if keepalive != 0 {
		c.setLiveKeepalive(keepalive)
	}
	return c
}

// startRunLiveServer serves RunLiveHandler with the given keepalive (zero keeps
// the default). Every handler return sends on exits, which is buffered so a
// test that never reads it cannot wedge a handler.
func startRunLiveServer(
	t *testing.T,
	keepalive time.Duration,
	runLiveFn func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error),
) (wsURL string, exits <-chan struct{}) {
	t.Helper()

	baseAgent, err := agent.New(agent.Config{Name: testLiveAppName})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}
	controller := newLiveTestController(agent.NewSingleLoader(&mockLiveAgent{Agent: baseAgent, runLiveFn: runLiveFn}), keepalive)
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

	wsURL, exits := startRunLiveServer(t, 0, runLiveFn)
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

func TestRunLiveHandler_ClosesUnresponsiveConnection(t *testing.T) {
	const keepaliveTimeout = 100 * time.Millisecond

	liveSession := newRecordingLiveSession()
	// The client never reads, so gorilla never answers the server's pings and
	// the keepalive is the only thing that can end the connection.
	wsURL, exits := startRunLiveServer(t, keepaliveTimeout, blockingLiveRun(liveSession))
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
	wsURL, _ := startRunLiveServer(t, keepaliveTimeout, blockingLiveRun(liveSession))
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
	wsURL, _ := startRunLiveServer(t, keepaliveTimeout, func(ic agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
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
	wsURL, _ := startRunLiveServer(t, keepaliveTimeout, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
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
	wsURL, exits := startRunLiveServer(t, keepaliveTimeout, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
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
	if c.resume != nil {
		close(c.resume)
		c.resume = nil
	}
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

func TestRunLiveHandler_DropsPeerThatStopsReading(t *testing.T) {
	const keepaliveTimeout = 400 * time.Millisecond

	liveSession := newRecordingLiveSession()
	stallReady := make(chan struct{})
	baseAgent, err := agent.New(agent.Config{Name: testLiveAppName})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}
	liveAgent := &mockLiveAgent{Agent: baseAgent, runLiveFn: func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return liveSession, func(yield func(*session.Event, error) bool) {
			<-stallReady
			yield(makeEvent("inv", testLiveAppName, "tick"), nil)
		}, nil
	}}
	controller := newLiveTestController(agent.NewSingleLoader(liveAgent), keepaliveTimeout)
	exits := make(chan struct{}, testMaxHandlerExits)
	handler := NewErrorHandler(controller.RunLiveHandler)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		defer func() { exits <- struct{}{} }()
		handler(rw, req)
	}))
	listener := &stallingListener{Listener: server.Listener, conns: make(chan *stallingConn, 2)}
	server.Listener = listener
	server.Start()
	t.Cleanup(server.Close)

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") +
		"/run_live?appName=" + testLiveAppName + "&userId=" + testLiveUserID + "&sessionId=" + testLiveSessionID
	conn := dialLive(t, wsURL)
	serverConn := <-listener.conns

	// The client keeps sending pongs so the server's read deadline never
	// expires; only the per-event write deadline can end the stalled write
	// after one keepalive timeout.
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

	serverConn.stall()
	t.Cleanup(serverConn.unstall)
	start := time.Now()
	close(stallReady)

	maxExit := 13 * keepaliveTimeout / 10
	select {
	case <-exits:
	case <-time.After(maxExit - time.Since(start)):
		t.Fatalf("RunLiveHandler still running %v after event write stalled, want exit before %v", time.Since(start), maxExit)
	}
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
	controller := newLiveTestController(agent.NewSingleLoader(liveAgent), keepaliveTimeout)
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
	select {
	case <-readErr:
	case <-time.After(time.Second):
		t.Fatal("connection did not close after a ping stalled for twice the keepalive timeout")
	}
}

func TestRunLiveHandler_FailedPingClosesConnection(t *testing.T) {
	const keepalive = 200 * time.Millisecond

	liveSession := newRecordingLiveSession()
	base, err := agent.New(agent.Config{Name: testLiveAppName})
	if err != nil {
		t.Fatal(err)
	}
	controller := newLiveTestController(agent.NewSingleLoader(&mockLiveAgent{Agent: base, runLiveFn: blockingLiveRun(liveSession)}), keepalive)
	exited := make(chan struct{})
	handler := NewErrorHandler(controller.RunLiveHandler)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		defer close(exited)
		handler(rw, req)
	}))
	listener := &stallingListener{Listener: server.Listener, conns: make(chan *stallingConn, 1)}
	server.Listener = listener
	server.Start()
	t.Cleanup(server.Close)

	conn := dialLive(t, "ws"+strings.TrimPrefix(server.URL, "http")+
		"/run_live?appName="+testLiveAppName+"&userId="+testLiveUserID+"&sessionId="+testLiveSessionID)
	serverConn := <-listener.conns

	done := make(chan struct{})
	defer close(done)
	// The agent stays quiet and takes whatever the client sends.
	go func() {
		for {
			select {
			case <-liveSession.requests:
			case <-done:
				return
			}
		}
	}()
	// From here the peer accepts no writes, starting with the next ping, but
	// keeps streaming audio.
	serverConn.stall()
	defer serverConn.unstall()
	go func() {
		ticker := time.NewTicker(keepalive / 4)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				if err := conn.WriteMessage(websocket.BinaryMessage, []byte("pcm")); err != nil {
					return
				}
			}
		}
	}()

	select {
	case <-exited:
	case <-time.After(15 * keepalive):
		t.Errorf("handler still running %v after the peer stopped accepting writes", 15*keepalive)
	}
}

// blockingSendLiveSession's Send blocks until the session closes, as
// internal/llminternal's does while no model connection is taking its input.
type blockingSendLiveSession struct {
	entered   chan struct{}
	enterOnce sync.Once
	closed    chan struct{}
	closeOnce sync.Once
}

func (s *blockingSendLiveSession) Send(agent.LiveRequest) error {
	s.enterOnce.Do(func() { close(s.entered) })
	<-s.closed
	return io.EOF
}

func (s *blockingSendLiveSession) Close() error {
	s.closeOnce.Do(func() { close(s.closed) })
	return nil
}

func TestRunLiveHandler_FailedPingEndsHandlerWhileSendBlocks(t *testing.T) {
	const keepalive = 200 * time.Millisecond

	sess := &blockingSendLiveSession{entered: make(chan struct{}), closed: make(chan struct{})}
	t.Cleanup(func() { _ = sess.Close() })
	base, err := agent.New(agent.Config{Name: testLiveAppName})
	if err != nil {
		t.Fatal(err)
	}
	controller := newLiveTestController(agent.NewSingleLoader(&mockLiveAgent{Agent: base, runLiveFn: func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		// A quiet agent: no events until the session closes.
		return sess, func(func(*session.Event, error) bool) { <-sess.closed }, nil
	}}), keepalive)
	exited := make(chan struct{})
	handler := NewErrorHandler(controller.RunLiveHandler)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		defer close(exited)
		handler(rw, req)
	}))
	listener := &stallingListener{Listener: server.Listener, conns: make(chan *stallingConn, 1)}
	server.Listener = listener
	server.Start()
	t.Cleanup(server.Close)

	conn := dialLive(t, "ws"+strings.TrimPrefix(server.URL, "http")+
		"/run_live?appName="+testLiveAppName+"&userId="+testLiveUserID+"&sessionId="+testLiveSessionID)
	serverConn := <-listener.conns
	if err := conn.WriteMessage(websocket.BinaryMessage, []byte("pcm")); err != nil {
		t.Fatal(err)
	}
	// The reader is now inside Send, where it neither reads nor has a read
	// deadline armed, and the agent emits nothing, so only the failed ping
	// below can end the handler.
	select {
	case <-sess.entered:
	case <-time.After(time.Second):
		t.Fatal("client message never reached the live session's Send")
	}
	serverConn.stall()
	defer serverConn.unstall()

	select {
	case <-exited:
	case <-time.After(5 * keepalive):
		t.Errorf("handler still running %v after the peer stopped accepting writes", 5*keepalive)
	}
}

type smallBufferListener struct {
	net.Listener
	writeBuffer int
}

func (l *smallBufferListener) Accept() (net.Conn, error) {
	conn, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	if tcp, ok := conn.(*net.TCPConn); ok {
		_ = tcp.SetWriteBuffer(l.writeBuffer)
	}
	return conn, nil
}

func TestRunLiveHandler_ReadDeadlineClosesInFlightWrite(t *testing.T) {
	const (
		keepaliveTimeout = 400 * time.Millisecond
		sockBufBytes     = 4096
		eventTextBytes   = 16 << 10
	)

	liveSession := newRecordingLiveSession()
	event := makeEvent("inv", testLiveAppName, strings.Repeat("a", eventTextBytes))
	baseAgent, err := agent.New(agent.Config{Name: testLiveAppName})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}
	liveAgent := &mockLiveAgent{Agent: baseAgent, runLiveFn: func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return liveSession, func(yield func(*session.Event, error) bool) {
			// Start writing just before the read deadline expires, so an
			// in-flight WriteJSON has a fresh write deadline near 1.9x while
			// the read deadline expires at 1.0x.
			time.Sleep(9 * keepaliveTimeout / 10)
			for {
				select {
				case <-liveSession.closed:
					return
				default:
				}
				if !yield(event, nil) {
					return
				}
			}
		}, nil
	}}
	controller := newLiveTestController(agent.NewSingleLoader(liveAgent), keepaliveTimeout)
	exits := make(chan struct{}, testMaxHandlerExits)
	handler := NewErrorHandler(controller.RunLiveHandler)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		defer func() { exits <- struct{}{} }()
		handler(rw, req)
	}))
	server.Listener = &smallBufferListener{Listener: server.Listener, writeBuffer: sockBufBytes}
	server.Start()
	t.Cleanup(server.Close)

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") +
		"/run_live?appName=" + testLiveAppName + "&userId=" + testLiveUserID + "&sessionId=" + testLiveSessionID
	dialer := websocket.Dialer{
		NetDial: func(network, addr string) (net.Conn, error) {
			conn, err := net.Dial(network, addr)
			if err != nil {
				return nil, err
			}
			if tcp, ok := conn.(*net.TCPConn); ok {
				_ = tcp.SetReadBuffer(sockBufBytes)
			}
			return conn, nil
		},
	}

	start := time.Now()
	conn, response, err := dialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("dialer.Dial() failed: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	if response.StatusCode != http.StatusSwitchingProtocols {
		t.Fatalf("handshake status = %d, want %d", response.StatusCode, http.StatusSwitchingProtocols)
	}

	maxExit := 12 * keepaliveTimeout / 10
	select {
	case <-exits:
	case <-time.After(maxExit - time.Since(start)):
		t.Fatalf("RunLiveHandler still running %v after connect, want exit before %v", time.Since(start), maxExit)
	}
}

// failingFlowLiveSession mirrors internal/llminternal's liveSessionImpl when a
// live flow fails while a client Send is pending: closing the session unblocks
// Send with io.EOF.
type failingFlowLiveSession struct {
	sendPending chan struct{}
	pendingOnce sync.Once
	done        chan struct{}
	closeOnce   sync.Once
}

func (s *failingFlowLiveSession) Send(agent.LiveRequest) error {
	s.pendingOnce.Do(func() { close(s.sendPending) })
	<-s.done
	return io.EOF
}

func (s *failingFlowLiveSession) Close() error {
	s.closeOnce.Do(func() { close(s.done) })
	return nil
}

func TestRunLiveHandler_FlowErrorWhileClientSendsDeliversCloseFrame(t *testing.T) {
	wantErr := errors.New("failed to connect live session: websocket: bad handshake")
	sess := &failingFlowLiveSession{
		sendPending: make(chan struct{}),
		done:        make(chan struct{}),
	}
	conn, handlerDone := dialRunLiveHandler(t, func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return sess, func(yield func(*session.Event, error) bool) {
			<-sess.sendPending
			_ = sess.Close()
			time.Sleep(5 * time.Millisecond)
			yield(nil, wantErr)
		}, nil
	})

	if err := conn.WriteMessage(websocket.BinaryMessage, []byte("pcm")); err != nil {
		t.Fatalf("WriteMessage() failed: %v", err)
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

type deadlineRecordingConn struct {
	net.Conn
	mu             sync.Mutex
	recording      bool
	readDeadlines  []time.Time
	writeDeadlines []time.Time
}

func (c *deadlineRecordingConn) startRecording() {
	c.mu.Lock()
	c.recording = true
	c.mu.Unlock()
}

func (c *deadlineRecordingConn) SetDeadline(t time.Time) error {
	c.mu.Lock()
	if c.recording {
		c.readDeadlines = append(c.readDeadlines, t)
		c.writeDeadlines = append(c.writeDeadlines, t)
	}
	c.mu.Unlock()
	return c.Conn.SetDeadline(t)
}

func (c *deadlineRecordingConn) SetReadDeadline(t time.Time) error {
	c.mu.Lock()
	if c.recording {
		c.readDeadlines = append(c.readDeadlines, t)
	}
	c.mu.Unlock()
	return c.Conn.SetReadDeadline(t)
}

func (c *deadlineRecordingConn) SetWriteDeadline(t time.Time) error {
	c.mu.Lock()
	if c.recording {
		c.writeDeadlines = append(c.writeDeadlines, t)
	}
	c.mu.Unlock()
	return c.Conn.SetWriteDeadline(t)
}

type deadlineRecordingListener struct {
	net.Listener
	conns chan *deadlineRecordingConn
}

func (l *deadlineRecordingListener) Accept() (net.Conn, error) {
	conn, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	dc := &deadlineRecordingConn{Conn: conn}
	l.conns <- dc
	return dc, nil
}

func TestRunLiveHandler_NegativeKeepaliveSetsNoDeadlines(t *testing.T) {
	liveSession := newRecordingLiveSession()
	baseAgent, err := agent.New(agent.Config{Name: testLiveAppName})
	if err != nil {
		t.Fatalf("agent.New() failed: %v", err)
	}
	conns := make(chan *deadlineRecordingConn, 1)
	var serverConn *deadlineRecordingConn
	liveAgent := &mockLiveAgent{Agent: baseAgent, runLiveFn: func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		serverConn = <-conns
		serverConn.startRecording()
		return liveSession, func(yield func(*session.Event, error) bool) {
			if !yield(makeEvent("inv", testLiveAppName, "hello"), nil) {
				return
			}
			<-liveSession.closed
		}, nil
	}}
	controller := newLiveTestController(agent.NewSingleLoader(liveAgent), -1)
	server := httptest.NewUnstartedServer(NewErrorHandler(controller.RunLiveHandler))
	server.Listener = &deadlineRecordingListener{Listener: server.Listener, conns: conns}
	server.Start()
	t.Cleanup(server.Close)

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") +
		"/run_live?appName=" + testLiveAppName + "&userId=" + testLiveUserID + "&sessionId=" + testLiveSessionID
	conn := dialLive(t, wsURL)

	var gotEvent models.Event
	if err := conn.SetReadDeadline(time.Now().Add(testWebSocketReadTimeout)); err != nil {
		t.Fatalf("SetReadDeadline() failed: %v", err)
	}
	if err := conn.ReadJSON(&gotEvent); err != nil {
		t.Fatalf("ReadJSON() failed: %v", err)
	}

	select {
	case <-liveSession.closed:
		t.Fatal("live session closed with the keepalive off")
	case <-time.After(50 * time.Millisecond):
	}

	if err := conn.WriteMessage(websocket.BinaryMessage, []byte("audio")); err != nil {
		t.Fatalf("WriteMessage() failed: %v", err)
	}
	got := waitForLiveRequest(t, liveSession)
	if blob, ok := got.RealtimeInput.(*genai.Blob); !ok || string(blob.Data) != "audio" {
		t.Fatalf("RealtimeInput = %#v, want *genai.Blob(\"audio\")", got.RealtimeInput)
	}

	serverConn.mu.Lock()
	readDeadlines := append([]time.Time(nil), serverConn.readDeadlines...)
	writeDeadlines := append([]time.Time(nil), serverConn.writeDeadlines...)
	serverConn.mu.Unlock()

	if len(readDeadlines) == 0 {
		t.Fatal("expected at least one SetReadDeadline call after upgrade")
	}
	for _, d := range readDeadlines {
		if !d.IsZero() {
			t.Errorf("SetReadDeadline(%v) after upgrade, want zero time", d)
		}
	}
	if len(writeDeadlines) == 0 {
		t.Fatal("expected at least one SetWriteDeadline call after upgrade")
	}
	for _, d := range writeDeadlines {
		if !d.IsZero() {
			t.Errorf("SetWriteDeadline(%v) after upgrade, want zero time", d)
		}
	}
}

func TestNewRuntimeAPIController_LiveKeepalive(t *testing.T) {
	tests := []struct {
		name          string
		set           time.Duration // 0 keeps the constructor's default
		wantKeepalive time.Duration
		wantPing      time.Duration
	}{
		// A literal rather than the constant: 1.x has no field to change it, so the
		// value is the contract. See defaultLiveKeepaliveTimeout for its origin.
		{name: "default", wantKeepalive: 40 * time.Second, wantPing: 20 * time.Second},
		{name: "explicit value", set: time.Minute, wantKeepalive: time.Minute, wantPing: 30 * time.Second},
		{name: "negative turns it off", set: -1, wantKeepalive: -1},
		{
			// Halving would give zero, which would make time.NewTicker panic
			// on every /run_live request.
			name:          "a one-nanosecond keepalive keeps a non-zero ping interval",
			set:           time.Nanosecond,
			wantKeepalive: time.Nanosecond,
			wantPing:      time.Nanosecond,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := NewRuntimeAPIController(nil, nil, nil, nil, 0, runner.PluginConfig{}, false)
			if tc.set != 0 {
				c.setLiveKeepalive(tc.set)
			}
			if c.liveKeepaliveTimeout != tc.wantKeepalive {
				t.Errorf("liveKeepaliveTimeout = %v, want %v", c.liveKeepaliveTimeout, tc.wantKeepalive)
			}
			if c.livePingInterval != tc.wantPing {
				t.Errorf("livePingInterval = %v, want %v", c.livePingInterval, tc.wantPing)
			}
		})
	}
}
