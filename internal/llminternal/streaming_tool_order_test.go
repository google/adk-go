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
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
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
	select {
	case first := <-got:
		if first != "pending" {
			t.Errorf("first response for call_1 = %q, want pending", first)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no response for call_1")
	}
}

// The user rejected a confirmation-gated streaming tool. The model should be
// told the call was rejected, as it is for a non-streaming tool.
func TestStreamingRejectionReachesModel(t *testing.T) {
	type args struct {
		Value string `json:"value"`
	}
	lookup, _ := functiontool.NewStreaming(functiontool.Config{Name: "lookup", RequireConfirmation: true},
		func(agent.Context, args) iter.Seq2[string, error] {
			return func(yield func(string, error) bool) { yield("ok", nil) }
		})
	resp := &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{
		FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup", Args: map[string]any{"value": "v"}},
	}}}}
	confs := map[string]*toolconfirmation.ToolConfirmation{"call_1": {Confirmed: false}}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: &mockAgent{name: "a"}})
	ev, err := (&Flow{}).handleFunctionCalls(ctx, map[string]tool.Tool{"lookup": lookup}, resp, confs, nil)
	if err != nil {
		t.Fatal(err)
	}
	msg, _ := ev.Content.Parts[0].FunctionResponse.Response["error"].(string)
	if !strings.Contains(msg, "call is rejected") {
		t.Errorf("model was told %q, want the rejection", msg)
	}
}
