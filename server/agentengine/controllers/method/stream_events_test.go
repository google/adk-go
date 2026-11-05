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

package method

import (
	"encoding/json"
	"io"
	"iter"
	"net/http/httptest"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestAgentEngineStreamsEventsWithoutContent(t *testing.T) {
	// Decode the wire names independently of the session types' camelCase tags.
	type inputRequest struct {
		InterruptID string `json:"interrupt_id"`
		Message     string `json:"message"`
	}
	type wireEvent struct {
		Content      *genai.Content `json:"content"`
		ErrorCode    string         `json:"error_code"`
		ErrorMessage string         `json:"error_message"`
		Actions      struct {
			StateDelta map[string]any `json:"state_delta"`
		} `json:"actions"`
		RequestedInput *inputRequest `json:"requested_input"`
	}
	for _, methodName := range []string{"async_stream_query", "streaming_agent_run_with_events"} {
		t.Run(methodName, func(t *testing.T) {
			for _, tc := range []struct {
				name  string
				event session.Event
			}{
				{
					name: "model_error",
					event: session.Event{LLMResponse: model.LLMResponse{
						ErrorCode: "SAFETY", ErrorMessage: "synthetic block reason",
					}},
				},
				{
					name: "state_delta",
					event: session.Event{Actions: session.EventActions{
						StateDelta: map[string]any{"counter": float64(1)},
					}},
				},
				{
					name: "requested_input",
					event: session.Event{RequestedInput: &session.RequestInput{
						InterruptID: "approval-1", Message: "Review the synthetic result",
					}},
				},
			} {
				t.Run(tc.name, func(t *testing.T) {
					a, err := agent.New(agent.Config{
						Name: "test_agent",
						Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
							return func(yield func(*session.Event, error) bool) {
								for i := range 3 {
									event := session.NewEvent(ctx, ctx.InvocationID())
									event.Author = ctx.Agent().Name()
									switch i {
									case 0:
										event.Content = genai.NewContentFromText("before", genai.RoleModel)
									case 1:
										event.LLMResponse = tc.event.LLMResponse
										event.Actions = tc.event.Actions
										event.RequestedInput = tc.event.RequestedInput
									case 2:
										event.Content = genai.NewContentFromText("after", genai.RoleModel)
									}
									if !yield(event, nil) {
										return
									}
								}
							}
						},
					})
					if err != nil {
						t.Fatal(err)
					}
					svc := session.InMemoryService()
					created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
					if err != nil {
						t.Fatal(err)
					}
					sessionID := created.Session.ID()
					cfg := &launcher.Config{AgentLoader: agent.NewSingleLoader(a), SessionService: svc}
					var handler MethodHandler = NewStreamQueryHandler(cfg, "app", methodName, "async_stream")
					input := map[string]any{
						"user_id": "user", "session_id": sessionID,
						"message": genai.NewContentFromText("synthetic input", genai.RoleUser),
					}
					envelope := methodName == "streaming_agent_run_with_events"
					if envelope {
						handler = NewStreamingAgentRunWithEventsHandler(cfg, "app", methodName, "stream")
						inner, err := json.Marshal(input)
						if err != nil {
							t.Fatal(err)
						}
						input = map[string]any{"request_json": string(inner)}
					}
					payload, err := json.Marshal(map[string]any{"class_method": methodName, "input": input})
					if err != nil {
						t.Fatal(err)
					}
					rw := httptest.NewRecorder()
					if err := handler.Handle(t.Context(), rw, payload); err != nil {
						t.Fatal(err)
					}
					decoder := json.NewDecoder(rw.Body)
					for i := range 3 {
						var got wireEvent
						if envelope {
							var response struct {
								Events    []wireEvent `json:"events"`
								SessionID string      `json:"session_id"`
							}
							if err := decoder.Decode(&response); err != nil {
								t.Fatalf("decode response %d: %v", i, err)
							}
							if len(response.Events) != 1 || response.SessionID != sessionID {
								t.Fatalf("response %d has an unexpected event count or session ID", i)
							}
							got = response.Events[0]
						} else if err := decoder.Decode(&got); err != nil {
							t.Fatalf("decode event %d: %v", i, err)
						}
						if i != 1 {
							text := "before"
							if i == 2 {
								text = "after"
							}
							if !cmp.Equal(got.Content, genai.NewContentFromText(text, genai.RoleModel)) {
								t.Errorf("event %d has unexpected text content", i)
							}
							continue
						}
						if got.Content != nil {
							t.Fatal("event 1 has content; want the contentless event between the text events")
						}
						if got.ErrorCode != tc.event.ErrorCode || got.ErrorMessage != tc.event.ErrorMessage {
							t.Error("event error fields were not preserved")
						}
						if !cmp.Equal(got.Actions.StateDelta, tc.event.Actions.StateDelta, cmpopts.EquateEmpty()) {
							t.Error("event state delta was not preserved")
						}
						var wantInput *inputRequest
						if request := tc.event.RequestedInput; request != nil {
							wantInput = &inputRequest{InterruptID: request.InterruptID, Message: request.Message}
						}
						if !cmp.Equal(got.RequestedInput, wantInput) {
							t.Error("event input request was not preserved")
						}
					}
					var extra json.RawMessage
					if err := decoder.Decode(&extra); err != io.EOF {
						t.Fatal("want end of stream after three events")
					}
				})
			}
		})
	}
}
