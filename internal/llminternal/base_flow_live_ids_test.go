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

package llminternal

import (
	"encoding/json"
	"iter"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// startLiveServerWithSetupComplete runs the Gemini Live wire-protocol fake of
// startLiveServerWithSetup, answering each connection's setup frame with the
// message setupComplete returns for its 1-based connection number, so a test
// can give each connection its own session id.
func startLiveServerWithSetupComplete(t *testing.T, setupComplete func(connNum int) string, serveConn func(connNum int, conn *websocket.Conn)) *genai.Client {
	t.Helper()
	var upgrader websocket.Upgrader
	var mu sync.Mutex
	connCount := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("websocket upgrade failed: %v", err)
			return
		}
		defer func() { _ = conn.Close() }()
		mu.Lock()
		connCount++
		connNum := connCount
		mu.Unlock()
		if _, _, err := conn.ReadMessage(); err != nil {
			t.Errorf("reading setup frame failed: %v", err)
			return
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(setupComplete(connNum))); err != nil {
			t.Errorf("writing setupComplete failed: %v", err)
			return
		}
		serveConn(connNum, conn)
	}))
	t.Cleanup(ts.Close)

	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		Backend:     genai.BackendGeminiAPI,
		APIKey:      "test-api-key",
		HTTPOptions: genai.HTTPOptions{BaseURL: strings.Replace(ts.URL, "http", "ws", 1)},
	})
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}
	return client
}

// pullLiveUntil pulls events from next until last reports true for one, and
// returns every non-nil event pulled, that one included.
func pullLiveUntil(t *testing.T, next func() (*session.Event, error, bool), last func(*session.Event) bool) []*session.Event {
	t.Helper()
	var events []*session.Event
	for {
		ev, err, ok := next()
		if !ok {
			t.Fatalf("iterator ended after %v", describeLiveIDEvents(events))
		}
		if err != nil {
			t.Fatalf("RunLive yielded an error: %v", err)
		}
		if ev == nil {
			continue
		}
		events = append(events, ev)
		if last(ev) {
			return events
		}
	}
}

// cancelAndDrainLive cancels the run and pulls next to exhaustion, so RunLive's
// goroutines end before the test does.
func cancelAndDrainLive(cancel func(), next func() (*session.Event, error, bool)) {
	cancel()
	for {
		if _, _, ok := next(); !ok {
			return
		}
	}
}

func isThanksEvent(ev *session.Event) bool {
	return ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text == "thanks"
}

// liveIDEvent is an event reduced to its kind and its live session id.
type liveIDEvent struct {
	Kind          string
	LiveSessionID string
}

// describeLiveIDEvent names the kind of ev: its calls or responses by id, its
// text, its transcription, its resumption handle, turn complete, or empty.
func describeLiveIDEvent(ev *session.Event) liveIDEvent {
	var ids []string
	for _, fc := range utils.FunctionCalls(ev.Content) {
		ids = append(ids, fc.ID)
	}
	if len(ids) > 0 {
		return liveIDEvent{"call " + strings.Join(ids, ","), ev.LiveSessionID}
	}
	for _, fr := range utils.FunctionResponses(ev.Content) {
		ids = append(ids, fr.ID)
	}
	kind := "empty"
	switch {
	case len(ids) > 0:
		kind = "response " + strings.Join(ids, ",")
	case ev.Content != nil && len(ev.Content.Parts) > 0:
		kind = "text " + ev.Content.Parts[0].Text
	case ev.InputTranscription != nil:
		kind = "input " + ev.InputTranscription.Text
		if ev.InputTranscription.Finished {
			kind += " finished"
		}
	case ev.OutputTranscription != nil:
		kind = "output " + ev.OutputTranscription.Text
		if ev.OutputTranscription.Finished {
			kind += " finished"
		}
	case ev.SessionResumptionHandle != "":
		kind = "resumption " + ev.SessionResumptionHandle
	case ev.TurnComplete:
		kind = "turn complete"
	}
	return liveIDEvent{kind, ev.LiveSessionID}
}

func describeLiveIDEvents(evs []*session.Event) []liveIDEvent {
	var out []liveIDEvent
	for _, ev := range evs {
		out = append(out, describeLiveIDEvent(ev))
	}
	return out
}

// withContent keeps the events that have content.
func withContent(evs []*session.Event) []*session.Event {
	var out []*session.Event
	for _, ev := range evs {
		if ev.Content != nil {
			out = append(out, ev)
		}
	}
	return out
}

func okTool(name string) tool.Tool {
	return &mockFunctionTool{name: name, runFunc: func(agent.Context, map[string]any) (map[string]any, error) {
		return map[string]any{"ok": true}, nil
	}}
}

// A live call without an id gets an adk- id before the model event is
// yielded, after the response processors have seen the response, as in
// adk-python (live/_live_llm_flow.py:399-403, then :468-470). The same id
// reaches the tool, the response event and the toolResponse on the wire, and
// marks the call as long-running; an id the model set is kept.
func TestRunLivePopulatesMissingFunctionCallIDs(t *testing.T) {
	const toolCall = `{"toolCall":{"functionCalls":[{"name":"tool_a","args":{"x":1}},{"id":"explicit-2","name":"tool_b","args":{"x":2}}]}}`
	wireIDs := make(chan []string, 1)
	client := startLiveServerWithSetupComplete(t, func(int) string { return `{"setupComplete":{}}` }, func(_ int, conn *websocket.Conn) {
		if !writeFrames(t, conn, toolCall) {
			return
		}
		frame, ok := awaitFunctionResponse(conn)
		if !ok {
			return
		}
		var msg struct {
			ToolResponse struct {
				FunctionResponses []struct {
					ID string `json:"id"`
				} `json:"functionResponses"`
			} `json:"toolResponse"`
		}
		if err := json.Unmarshal([]byte(frame), &msg); err != nil {
			t.Errorf("decoding toolResponse: %v", err)
		}
		var ids []string
		for _, fr := range msg.ToolResponse.FunctionResponses {
			ids = append(ids, fr.ID)
		}
		wireIDs <- ids
		if !writeFrames(t, conn, serverContentThanks) {
			return
		}
		blockUntilClientCloses(conn)
	})

	var mu sync.Mutex
	var processorIDs []string
	var toolIDs []string
	recordToolID := func(ctx agent.Context) {
		mu.Lock()
		defer mu.Unlock()
		toolIDs = append(toolIDs, ctx.FunctionCallID())
	}
	f := &Flow{
		Model:             &fakeLiveModel{client: client},
		RequestProcessors: liveOnlyProcessors,
		ResponseProcessors: []func(agent.InvocationContext, *model.LLMRequest, *model.LLMResponse) iter.Seq2[*session.Event, error]{
			func(_ agent.InvocationContext, _ *model.LLMRequest, resp *model.LLMResponse) iter.Seq2[*session.Event, error] {
				mu.Lock()
				for _, fc := range utils.FunctionCalls(resp.Content) {
					processorIDs = append(processorIDs, fc.ID)
				}
				mu.Unlock()
				return func(func(*session.Event, error) bool) {}
			},
		},
		Tools: []tool.Tool{
			&longRunningMockTool{mockFunctionTool{name: "tool_a", runFunc: func(ctx agent.Context, _ map[string]any) (map[string]any, error) {
				recordToolID(ctx)
				return map[string]any{"status": "started"}, nil
			}}},
			&mockFunctionTool{name: "tool_b", runFunc: func(ctx agent.Context, _ map[string]any) (map[string]any, error) {
				recordToolID(ctx)
				return map[string]any{"ok": true}, nil
			}},
		},
	}
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()
	// Event ids draw on the same provider, so a constant keeps the call id
	// independent of how many events come first.
	ctx = ctx.WithContext(platform.WithUUIDProvider(ctx, func() string { return "uuid-1" }))

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	defer func() { _ = sess.Close() }()
	next, stop := iter.Pull2(seq)
	defer stop()

	events := withContent(pullLiveUntil(t, next, isThanksEvent))
	cancelAndDrainLive(cancel, next)

	want := []liveIDEvent{
		{"call adk-uuid-1,explicit-2", ""},
		{"response adk-uuid-1,explicit-2", ""},
		{"text thanks", ""},
	}
	if diff := cmp.Diff(want, describeLiveIDEvents(events)); diff != "" {
		t.Fatalf("events mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"adk-uuid-1"}, events[0].LongRunningToolIDs); diff != "" {
		t.Errorf("LongRunningToolIDs mismatch (-want +got):\n%s", diff)
	}
	mu.Lock()
	defer mu.Unlock()
	// The tools may run concurrently, so their order is not compared.
	slices.Sort(toolIDs)
	if diff := cmp.Diff([]string{"adk-uuid-1", "explicit-2"}, toolIDs); diff != "" {
		t.Errorf("tool FunctionCallID()s mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"", "explicit-2"}, processorIDs); diff != "" {
		t.Errorf("ids the response processor saw mismatch (-want +got):\n%s", diff)
	}
	select {
	case got := <-wireIDs:
		if diff := cmp.Diff([]string{"adk-uuid-1", "explicit-2"}, got); diff != "" {
			t.Errorf("toolResponse ids on the wire mismatch (-want +got):\n%s", diff)
		}
	default:
		t.Error("the server never received a toolResponse")
	}
}

// The response event built for the calls of a live model event carries that
// event's live session id, for one call and for parallel calls merged into one
// event, "" when the connection has no id, and the new connection's id after
// a reconnect. Only events with content are compared: whether the empty
// setupComplete event is yielded is not part of this contract.
func TestRunLivePreservesLiveSessionIDOnSingleAndParallelFunctionResponses(t *testing.T) {
	const (
		callC1   = `{"toolCall":{"functionCalls":[{"id":"c1","name":"tool_a","args":{}}]}}`
		callC2C3 = `{"toolCall":{"functionCalls":[{"id":"c2","name":"tool_a","args":{}},{"id":"c3","name":"tool_b","args":{}}]}}`
		callC2   = `{"toolCall":{"functionCalls":[{"id":"c2","name":"tool_b","args":{}}]}}`
		resume   = `{"sessionResumptionUpdate":{"newHandle":"h1","resumable":true}}`
	)
	// answer writes each call in turn, waiting for the function response of
	// each before the next, and reports whether the client stayed connected.
	answer := func(t *testing.T, conn *websocket.Conn, calls ...string) bool {
		for _, c := range calls {
			if !writeFrames(t, conn, c) {
				return false
			}
			if _, ok := awaitFunctionResponse(conn); !ok {
				return false
			}
		}
		return true
	}
	tests := []struct {
		name          string
		setupComplete func(connNum int) string
		serveConn     func(t *testing.T, connNum int, conn *websocket.Conn)
		want          []liveIDEvent
	}{
		{
			name:          "single call and parallel calls carry the connection liveSessionId",
			setupComplete: func(int) string { return `{"setupComplete":{"sessionId":"live-sess-1"}}` },
			serveConn: func(t *testing.T, _ int, conn *websocket.Conn) {
				if answer(t, conn, callC1, callC2C3) && writeFrames(t, conn, serverContentThanks) {
					blockUntilClientCloses(conn)
				}
			},
			want: []liveIDEvent{
				{"call c1", "live-sess-1"},
				{"response c1", "live-sess-1"},
				{"call c2,c3", "live-sess-1"},
				{"response c2,c3", "live-sess-1"},
				{"text thanks", "live-sess-1"},
			},
		},
		{
			name:          "empty setupComplete leaves model and response events with empty LiveSessionID",
			setupComplete: func(int) string { return `{"setupComplete":{}}` },
			serveConn: func(t *testing.T, _ int, conn *websocket.Conn) {
				if answer(t, conn, callC1) && writeFrames(t, conn, serverContentThanks) {
					blockUntilClientCloses(conn)
				}
			},
			want: []liveIDEvent{
				{"call c1", ""},
				{"response c1", ""},
				{"text thanks", ""},
			},
		},
		{
			name: "reconnect switches to the new connection's liveSessionId",
			setupComplete: func(connNum int) string {
				if connNum == 1 {
					return `{"setupComplete":{"sessionId":"live-sess-conn1"}}`
				}
				return `{"setupComplete":{"sessionId":"live-sess-conn2"}}`
			},
			serveConn: func(t *testing.T, connNum int, conn *websocket.Conn) {
				if connNum == 1 {
					// Returning hard-closes the connection, which the flow
					// resumes from.
					if writeFrames(t, conn, resume) {
						answer(t, conn, callC1)
					}
					return
				}
				if answer(t, conn, callC2) && writeFrames(t, conn, serverContentThanks) {
					blockUntilClientCloses(conn)
				}
			},
			want: []liveIDEvent{
				{"call c1", "live-sess-conn1"},
				{"response c1", "live-sess-conn1"},
				{"call c2", "live-sess-conn2"},
				{"response c2", "live-sess-conn2"},
				{"text thanks", "live-sess-conn2"},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := startLiveServerWithSetupComplete(t, tc.setupComplete, func(connNum int, conn *websocket.Conn) {
				tc.serveConn(t, connNum, conn)
			})
			f := &Flow{
				Model:             &fakeLiveModel{client: client},
				RequestProcessors: liveOnlyProcessors,
				Tools:             []tool.Tool{okTool("tool_a"), okTool("tool_b")},
				reconnect:         testLiveReconnectPolicy(3, 3),
			}
			ctx, cancel := newLiveInvocationContext(t)
			defer cancel()

			sess, seq, err := f.RunLive(ctx)
			if err != nil {
				t.Fatalf("RunLive failed: %v", err)
			}
			defer func() { _ = sess.Close() }()
			next, stop := iter.Pull2(seq)
			defer stop()

			events := withContent(pullLiveUntil(t, next, isThanksEvent))
			cancelAndDrainLive(cancel, next)

			if diff := cmp.Diff(tc.want, describeLiveIDEvents(events)); diff != "" {
				t.Errorf("events mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Transcription events, partial and finished, and session resumption events
// carry no live session id, as adk-python builds them without one
// (live/_live_llm_flow.py:269-273, :422-449); the call, response and turn
// complete events of the same connection carry it.
func TestRunLiveTranscriptionAndResumptionEventsCarryNoLiveSessionID(t *testing.T) {
	client := startLiveServerWithSetupComplete(t, func(int) string { return `{"setupComplete":{"sessionId":"s"}}` }, func(_ int, conn *websocket.Conn) {
		if !writeFrames(t, conn,
			`{"serverContent":{"inputTranscription":{"text":"in-txt"}}}`,
			`{"serverContent":{"outputTranscription":{"text":"out-txt"}}}`,
			`{"sessionResumptionUpdate":{"newHandle":"h-1","resumable":true}}`,
			`{"toolCall":{"functionCalls":[{"id":"c1","name":"tool_a","args":{}}]}}`,
		) {
			return
		}
		if _, ok := awaitFunctionResponse(conn); !ok {
			return
		}
		if writeFrames(t, conn, `{"serverContent":{"turnComplete":true}}`) {
			blockUntilClientCloses(conn)
		}
	})
	f := &Flow{
		Model:             &fakeLiveModel{client: client},
		RequestProcessors: liveOnlyProcessors,
		Tools:             []tool.Tool{okTool("tool_a")},
	}
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	defer func() { _ = sess.Close() }()
	next, stop := iter.Pull2(seq)
	defer stop()

	events := pullLiveUntil(t, next, func(ev *session.Event) bool { return ev.TurnComplete })
	cancelAndDrainLive(cancel, next)

	var got []liveIDEvent
	for _, d := range describeLiveIDEvents(events) {
		// The empty setupComplete event is not part of this contract.
		if d.Kind != "empty" {
			got = append(got, d)
		}
	}
	want := []liveIDEvent{
		{"input in-txt", ""},
		{"output out-txt", ""},
		{"resumption h-1", ""},
		{"call c1", "s"},
		{"response c1", "s"},
		{"input in-txt finished", ""},
		{"output out-txt finished", ""},
		{"turn complete", "s"},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("events mismatch (-want +got):\n%s", diff)
	}
}

// A message that carries content, a turn-complete signal or an interrupted
// signal keeps the connection's live session id even when it also carries a
// transcription, as adk-python yields those as responses with the id
// (models/gemini_llm_connection.py:573-595, :616); a transcription alone
// carries none.
func TestRunLiveTranscriptionWithContentOrSignalKeepsLiveSessionID(t *testing.T) {
	tests := []struct {
		name  string
		frame string
		// match picks the event whose id the case checks.
		match func(*session.Event) bool
		want  string
	}{
		{
			name:  "model content with an output transcription",
			frame: `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm","data":"AAAA"}}],"role":"model"},"outputTranscription":{"text":"hi"}}}`,
			match: func(ev *session.Event) bool { return ev.Content != nil && len(ev.Content.Parts) > 0 },
			want:  "s",
		},
		{
			name:  "input transcription with turn complete",
			frame: `{"serverContent":{"inputTranscription":{"text":"hi"},"turnComplete":true}}`,
			match: func(ev *session.Event) bool { return ev.TurnComplete },
			want:  "s",
		},
		{
			name:  "output transcription with interrupted",
			frame: `{"serverContent":{"outputTranscription":{"text":"hi"},"interrupted":true}}`,
			match: func(ev *session.Event) bool { return ev.Interrupted },
			want:  "s",
		},
		{
			name:  "transcription only",
			frame: `{"serverContent":{"inputTranscription":{"text":"hi"}}}`,
			match: func(ev *session.Event) bool { return ev.InputTranscription != nil },
			want:  "",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := startLiveServerWithSetupComplete(t, func(int) string { return `{"setupComplete":{"sessionId":"s"}}` }, func(_ int, conn *websocket.Conn) {
				if writeFrames(t, conn, tc.frame) {
					blockUntilClientCloses(conn)
				}
			})
			f := &Flow{
				Model:             &fakeLiveModel{client: client},
				RequestProcessors: liveOnlyProcessors,
			}
			ctx, cancel := newLiveInvocationContext(t)
			defer cancel()

			sess, seq, err := f.RunLive(ctx)
			if err != nil {
				t.Fatalf("RunLive failed: %v", err)
			}
			defer func() { _ = sess.Close() }()
			next, stop := iter.Pull2(seq)
			defer stop()

			events := pullLiveUntil(t, next, tc.match)
			cancelAndDrainLive(cancel, next)

			if got := events[len(events)-1].LiveSessionID; got != tc.want {
				t.Errorf("LiveSessionID = %q, want %q", got, tc.want)
			}
		})
	}
}
