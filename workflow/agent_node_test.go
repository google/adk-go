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

package workflow

import (
	"encoding/json"
	"errors"
	"iter"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

type mockSession struct {
	id string
}

func (m *mockSession) ID() string                { return m.id }
func (m *mockSession) AppName() string           { return "test-app" }
func (m *mockSession) UserID() string            { return "test-user" }
func (m *mockSession) State() session.State      { return nil }
func (m *mockSession) Events() session.Events    { return nil }
func (m *mockSession) LastUpdateTime() time.Time { return time.Now() }

func TestAgentNode_New(t *testing.T) {
	type Input struct {
		Value string `json:"value"`
	}
	type Output struct {
		Result string `json:"result"`
	}

	myAgent, err := agent.New(agent.Config{
		Name:        "test_agent",
		Description: "a test agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				event := session.NewEvent(ctx.InvocationID())
				event.Output = map[string]any{"result": "success"}
				yield(event, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}

	ischema, err := jsonschema.For[Input](nil)
	if err != nil {
		t.Fatalf("jsonschema.For[Input] failed: %v", err)
	}
	oschema, err := jsonschema.For[Output](nil)
	if err != nil {
		t.Fatalf("jsonschema.For[Output] failed: %v", err)
	}

	tests := []struct {
		name    string
		creator func() (Node, error)
		want    string
	}{
		{
			name: "NewAgentNodeTyped",
			creator: func() (Node, error) {
				return NewAgentNodeTyped[Input, Output](myAgent, defaultNodeConfig)
			},
		},
		{
			name: "NewAgentNodeWithSchemas",
			creator: func() (Node, error) {
				return NewAgentNodeWithSchemas(myAgent, ischema, oschema, defaultNodeConfig)
			},
		},
		{
			name: "NewAgentNode",
			creator: func() (Node, error) {
				return NewAgentNode(myAgent, defaultNodeConfig)
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node, err := tc.creator()
			if err != nil {
				t.Fatalf("creation failed: %v", err)
			}

			if got, want := node.Name(), "test_agent"; got != want {
				t.Errorf("node.Name() = %q, want %q", got, want)
			}
			if got, want := node.Description(), "a test agent"; got != want {
				t.Errorf("node.Description() = %q, want %q", got, want)
			}

			// Basic internal check via reflection-like cast.
			var inputResolved, outputResolved *jsonschema.Resolved
			switch tn := node.(type) {
			case *AgentNode:
				inputResolved, outputResolved = tn.inputSchema, tn.outputSchema
			default:
				t.Errorf("unknown node type: %T", tn)
			}

			if inputResolved == nil || outputResolved == nil {
				t.Error("expected schemas to be resolved")
			}
		})
	}
}

func TestAgentNode_Run(t *testing.T) {
	type Input struct {
		Val string `json:"val"`
	}
	type Output struct {
		Result string `json:"result"`
	}

	tests := []struct {
		name      string
		agent     func() (agent.Agent, error)
		nodeInput any
		node      func(agent.Agent) (Node, error)
		extract   func(t *testing.T, out any) string
		want      string
		wantErr   string
	}{
		{
			name: "struct_input_output",
			agent: func() (agent.Agent, error) {
				return agent.New(agent.Config{
					Name: "test_agent",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							uc := ctx.UserContent()
							val := "Unknown"
							if uc != nil && len(uc.Parts) > 0 {
								val = uc.Parts[0].Text
							}
							event := session.NewEvent(ctx.InvocationID())
							event.Output = map[string]any{"result": val}
							yield(event, nil)
						}
					},
				})
			},
			nodeInput: Input{Val: "A"},
			node: func(a agent.Agent) (Node, error) {
				return NewAgentNodeTyped[Input, Output](a, defaultNodeConfig)
			},
			extract: func(t *testing.T, out any) string {
				bytes, err := json.Marshal(out)
				if err != nil {
					t.Fatalf("json marshal output: %v", err)
				}
				var output Output
				if err := json.Unmarshal(bytes, &output); err != nil {
					t.Fatalf("json unmarshal output: %v", err)
				}
				return output.Result
			},
			want: `{"val":"A"}`,
		},
		{
			name: "string_output",
			agent: func() (agent.Agent, error) {
				return agent.New(agent.Config{
					Name: "test_agent",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							uc := ctx.UserContent()
							val := "Unknown"
							if uc != nil && len(uc.Parts) > 0 {
								val = uc.Parts[0].Text
							}
							event := session.NewEvent(ctx.InvocationID())
							event.Output = val
							yield(event, nil)
						}
					},
				})
			},
			nodeInput: Input{Val: "B"},
			node: func(a agent.Agent) (Node, error) {
				return NewAgentNodeTyped[Input, string](a, defaultNodeConfig)
			},
			extract: func(t *testing.T, out any) string {
				return out.(string)
			},
			want: `{"val":"B"}`,
		},
		{
			name: "agent_execution_error",
			agent: func() (agent.Agent, error) {
				return agent.New(agent.Config{
					Name: "test_agent",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							yield(nil, errors.New("something went wrong"))
						}
					},
				})
			},
			nodeInput: Input{Val: "C"},
			node: func(a agent.Agent) (Node, error) {
				return NewAgentNodeTyped[Input, Output](a, defaultNodeConfig)
			},
			wantErr: "something went wrong",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			myAgent, err := tc.agent()
			if err != nil {
				t.Fatalf("failed to create agent: %v", err)
			}

			node, err := tc.node(myAgent)
			if err != nil {
				t.Fatalf("node creation failed: %v", err)
			}

			mockCtx := newMockCtx(t)
			mockCtx.sess = &mockSession{id: "test-session-id"} // Fix nil panic
			events := node.Run(mockCtx, tc.nodeInput)

			var got string
			count := 0
			for ev, err := range events {
				if tc.wantErr != "" {
					if err == nil {
						t.Fatal("expected error, got nil")
					}
					if !strings.Contains(err.Error(), tc.wantErr) {
						t.Errorf("expected error containing %q, got: %v", tc.wantErr, err)
					}
					return
				}

				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				count++

				got = tc.extract(t, ev.Output)
			}

			if tc.wantErr != "" {
				t.Error("expected at least one event/error from Run")
				return
			}

			if count != 1 {
				t.Errorf("expected 1 event, got %d", count)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("output mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAgentNode_WorkflowIntegration(t *testing.T) {
	type Input struct {
		Val int `json:"val"`
	}
	type Output struct {
		Result int `json:"result"`
	}

	tests := []struct {
		name  string
		input int
		want  int
	}{
		{
			name:  "chain_agent_and_function",
			input: 5,
			want:  11,
		},
		{
			name:  "chain_agent_and_function_zero",
			input: 0,
			want:  1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			integrationAgent, err := agent.New(agent.Config{
				Name: "integration_agent",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						uc := ctx.UserContent()
						valStr := "0"
						if uc != nil && len(uc.Parts) > 0 {
							valStr = uc.Parts[0].Text
						}
						// valStr will be something like `{"val":5}`
						var val int
						var parsed struct {
							Val int `json:"val"`
						}
						if err := json.Unmarshal([]byte(valStr), &parsed); err == nil {
							val = parsed.Val
						}

						event := session.NewEvent(ctx.InvocationID())
						event.Output = map[string]any{"result": val * 2}
						yield(event, nil)
					}
				},
			})
			if err != nil {
				t.Fatalf("failed to create agent: %v", err)
			}

			agentNode, err := NewAgentNodeTyped[Input, Output](integrationAgent, defaultNodeConfig)
			if err != nil {
				t.Fatalf("NewAgentNodeTyped failed: %v", err)
			}

			// Connect to a function node.
			functionNode := NewFunctionNode[Output, int]("plus_one", func(ctx agent.InvocationContext, in Output) (int, error) {
				return in.Result + 1, nil
			}, NodeConfig{})

			mockCtx := newMockCtx(t)
			mockCtx.sess = &mockSession{id: "test-session-id"} // Ensure session is set

			t.Run("WorkflowExecution", func(t *testing.T) {
				// Use a seed node to pass the struct input to agentNode
				seedNode := NewFunctionNode("seed", func(ctx agent.InvocationContext, input any) (*Input, error) {
					return &Input{Val: tc.input}, nil
				}, NodeConfig{})

				edges := Chain(Start, seedNode, agentNode, functionNode)
				w, err := New("test_workflow", edges)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				events := w.Run(mockCtx)

				var outB any
				for ev, err := range events {
					if err != nil {
						t.Fatalf("workflow failed: %v", err)
					}
					if ev.Output != nil {
						outB = ev.Output
					}
				}

				if diff := cmp.Diff(tc.want, outB); diff != "" {
					t.Errorf("output mismatch (-want +got):\n%s", diff)
				}
			})
		})
	}
}

func TestAgentNode_SynthesizesOutputFromModelText(t *testing.T) {
	wrapped, err := agent.New(agent.Config{
		Name: "talky",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				// Partial must not be promoted to Output.
				partial := session.NewEvent(ctx.InvocationID())
				partial.LLMResponse.Partial = true
				partial.LLMResponse.Content = &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: "Hel"}},
				}
				if !yield(partial, nil) {
					return
				}
				// Thought parts are skipped; text parts concatenate.
				final := session.NewEvent(ctx.InvocationID())
				final.LLMResponse.Content = &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "thinking…", Thought: true},
						{Text: "Hello, "},
						{Text: "world!"},
					},
				}
				yield(final, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	node, err := NewAgentNode(wrapped, NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}

	mockCtx := newMockCtx(t)
	mockCtx.sess = &mockSession{id: "test-session-id"}
	var (
		gotPartial *session.Event
		gotFinal   *session.Event
	)
	for ev, err := range node.Run(mockCtx, "ignored") {
		if err != nil {
			t.Fatalf("node.Run: %v", err)
		}
		if ev.LLMResponse.Partial {
			gotPartial = ev
		} else {
			gotFinal = ev
		}
	}

	if gotPartial == nil || gotFinal == nil {
		t.Fatalf("missing events: partial=%v final=%v", gotPartial, gotFinal)
	}
	if gotPartial.Output != nil {
		t.Errorf("partial event Output = %v, want nil (partials must not be promoted)", gotPartial.Output)
	}
	if got, want := gotFinal.Output, "Hello, world!"; got != want {
		t.Errorf("final event Output = %v, want %q", got, want)
	}
	if gotFinal.NodeInfo == nil || !gotFinal.NodeInfo.MessageAsOutput {
		t.Errorf("final event NodeInfo.MessageAsOutput = %v, want true", gotFinal.NodeInfo)
	}
	if gotPartial.NodeInfo != nil && gotPartial.NodeInfo.MessageAsOutput {
		t.Errorf("partial event MessageAsOutput = true, want false/unset")
	}
}

func TestAgentNode_StampsIsolationScopeOnEvents(t *testing.T) {
	var gotAgentScope string
	wrapped, err := agent.New(agent.Config{
		Name: "scoped",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			gotAgentScope = ctx.IsolationScope()
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx.InvocationID())
				ev.Output = "v"
				yield(ev, nil)
				// An event that already carries a scope is left untouched.
				pre := session.NewEvent(ctx.InvocationID())
				pre.IsolationScope = "preset"
				yield(pre, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	node, err := NewAgentNode(wrapped, NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}

	mockCtx := newMockCtx(t)
	mockCtx.sess = &mockSession{id: "test-session-id"}
	mockCtx.isolationScope = "scope-x"

	var events []*session.Event
	for ev, err := range node.Run(mockCtx, "ignored") {
		if err != nil {
			t.Fatalf("node.Run: %v", err)
		}
		events = append(events, ev)
	}

	if gotAgentScope != "scope-x" {
		t.Errorf("agent ctx IsolationScope = %q, want %q", gotAgentScope, "scope-x")
	}
	if len(events) != 2 {
		t.Fatalf("got %d events, want 2", len(events))
	}
	if events[0].IsolationScope != "scope-x" {
		t.Errorf("event[0] IsolationScope = %q, want %q", events[0].IsolationScope, "scope-x")
	}
	if events[1].IsolationScope != "preset" {
		t.Errorf("event[1] IsolationScope = %q, want %q (preset must be kept)", events[1].IsolationScope, "preset")
	}
}

func TestAgentNode_AutomaticOutputExtraction(t *testing.T) {
	myAgent, err := agent.New(agent.Config{
		Name: "text_only_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				event := session.NewEvent(ctx.InvocationID())
				// Model response with plain text, but no Output set
				event.Content = &genai.Content{
					Parts: []*genai.Part{
						{Text: "This is "},
						{Text: "the output text."},
					},
					Role: "model",
				}
				yield(event, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}

	node, err := NewAgentNode(myAgent, defaultNodeConfig)
	if err != nil {
		t.Fatalf("failed to create AgentNode: %v", err)
	}

	mockCtx := newMockCtx(t)
	mockCtx.sess = &mockSession{id: "test-session"}
	events := node.Run(mockCtx, nil)

	var finalOutput any
	for ev, err := range events {
		if err != nil {
			t.Fatalf("Run returned error: %v", err)
		}
		if ev.Output != nil {
			finalOutput = ev.Output
		}
	}

	if got, want := finalOutput, "This is the output text."; got != want {
		t.Errorf("expected automatically extracted output %q, got %q", want, got)
	}
}
