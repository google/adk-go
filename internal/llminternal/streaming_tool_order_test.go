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
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// A streaming tool fails at once while a second call in the same turn takes
// 50ms. The server should see call_1's pending response before its error.
func TestLiveErrorFollowUpOrder(t *testing.T) {
	const toolCall = `{"toolCall":{"functionCalls":[` +
		`{"id":"call_1","name":"lookup","args":{"value":"x"}},` +
		`{"id":"call_2","name":"slow","args":{}}]}}`
	got := make(chan string, 4)
	client, _ := startFakeLiveServer(t, func(_ int, conn *websocket.Conn) {
		if err := conn.WriteMessage(websocket.TextMessage, []byte(toolCall)); err != nil {
			return
		}
		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var m struct {
				ToolResponse struct {
					FunctionResponses []struct {
						ID       string         `json:"id"`
						Response map[string]any `json:"response"`
					} `json:"functionResponses"`
				} `json:"toolResponse"`
			}
			_ = json.Unmarshal(msg, &m)
			for _, fr := range m.ToolResponse.FunctionResponses {
				if fr.ID == "call_1" && fr.Response["status"] != nil {
					got <- "pending"
				} else if fr.ID == "call_1" && fr.Response["error"] != nil {
					got <- "error"
				}
			}
		}
	})
	type args struct {
		Value string `json:"value"`
	}
	lookup, _ := functiontool.NewStreaming(functiontool.Config{Name: "lookup"},
		func(agent.Context, args) iter.Seq2[string, error] {
			return func(func(string, error) bool) { panic("boom") }
		})
	slow := &mockFunctionTool{name: "slow", runFunc: func(agent.Context, map[string]any) (map[string]any, error) {
		time.Sleep(50 * time.Millisecond)
		return map[string]any{"ok": true}, nil
	}}
	f := &Flow{
		Model:             &fakeLiveModel{client: client},
		RequestProcessors: []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{liveConfigProcessor},
		Tools:             []tool.Tool{lookup, slow},
	}
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()
	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = sess.Close() }()
	go func() {
		for range seq {
		}
	}()
	for _, want := range []string{"pending", "error"} {
		select {
		case response := <-got:
			if response != want {
				t.Fatalf("response kind = %q, want %q", response, want)
			}
		case <-time.After(3 * time.Second):
			t.Fatalf("missing %s response for call_1", want)
		}
	}
}

// If a connection drops after RunLive queues pendingSend for a streaming tool
// call and before inputCh drains it, reconnect discards pendingSend with the
// dead socket. The pending gate must still open so the streaming tool's chunk
// reaches the new connection.
func TestLiveReconnectDropsQueuedPendingResponseReleasesTool(t *testing.T) {
	const toolCall = `{"toolCall":{"functionCalls":[` +
		`{"id":"call_1","name":"lookup","args":{"value":"x"}},` +
		`{"id":"call_2","name":"slow","args":{}}]}}`
	conn1Closed := make(chan struct{})
	chunkReady := make(chan struct{})
	var chunkReadyOnce sync.Once
	sessReady := make(chan struct{})
	gotOnConn2 := make(chan string, 4)

	client, _ := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
		if connNum == 1 {
			<-sessReady
			_ = conn.WriteMessage(websocket.TextMessage, []byte(toolCall))
			_ = conn.Close()
			close(conn1Closed)
			return
		}
		chunkReadyOnce.Do(func() { close(chunkReady) })
		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var m struct {
				ClientContent struct {
					Turns []struct {
						Parts []struct {
							Text string `json:"text"`
						} `json:"parts"`
					} `json:"turns"`
				} `json:"clientContent"`
			}
			_ = json.Unmarshal(msg, &m)
			for _, turn := range m.ClientContent.Turns {
				for _, part := range turn.Parts {
					if part.Text != "" {
						gotOnConn2 <- part.Text
					}
				}
			}
		}
	})

	type args struct {
		Value string `json:"value"`
	}
	var sess agent.LiveSession
	lookup, _ := functiontool.NewStreaming(functiontool.Config{Name: "lookup"},
		func(ctx agent.Context, _ args) iter.Seq2[string, error] {
			return func(yield func(string, error) bool) {
				select {
				case <-chunkReady:
				case <-ctx.Done():
					return
				}
				yield("after-reconnect", nil)
			}
		})
	slow := &mockFunctionTool{name: "slow", runFunc: func(agent.Context, map[string]any) (map[string]any, error) {
		<-conn1Closed
		// Give Connection 1's reader time to observe EOF and park on errChan;
		// also write once through inputCh so Connection 1's sender goroutine
		// fails on the closed socket and exits before pendingSend is queued,
		// ensuring select deterministically takes the reconnect branch.
		time.Sleep(15 * time.Millisecond)
		go func() {
			_ = sess.Send(agent.LiveRequest{RealtimeInput: &genai.Blob{MIMEType: "audio/pcm", Data: []byte{0}}})
		}()
		time.Sleep(15 * time.Millisecond)
		return map[string]any{"ok": true}, nil
	}}

	f := newReconnectFlow(client, testLiveReconnectPolicy(4, 10))
	f.Tools = []tool.Tool{lookup, slow}

	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()
	var seq iter.Seq2[*session.Event, error]
	var err error
	sess, seq, err = f.RunLive(ctx)
	if err != nil {
		t.Fatal(err)
	}
	close(sessReady)
	defer func() { _ = sess.Close() }()
	go func() {
		for range seq {
		}
	}()

	select {
	case text := <-gotOnConn2:
		if text != "Function lookup returned: after-reconnect" {
			t.Fatalf("conn 2 text = %q, want %q", text, "Function lookup returned: after-reconnect")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("streaming tool chunk never reached reconnected session (pending gate stuck)")
	}
}
