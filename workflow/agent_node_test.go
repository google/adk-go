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
	"fmt"
	"iter"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"

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
				event.Actions.StateDelta = map[string]any{
					"output": map[string]any{"result": "success"},
				}
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
			name: "NewToolNodeTyped",
			creator: func() (Node, error) {
				return NewAgentNodeTyped[Input, Output](myAgent, defaultNodeConfig)
			},
		},
		{
			name: "NewToolNodeWithSchemas",
			creator: func() (Node, error) {
				return NewAgentNodeWithSchemas(myAgent, ischema, oschema, defaultNodeConfig)
			},
		},
		{
			name: "NewToolNode",
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
			// We use any, any for constructors that don't preserve types in the struct.
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
	type ErrorOutput struct {
		Result int `json:"result"`
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
							event.Actions.StateDelta = map[string]any{
								"output": map[string]any{"result": val},
							}
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
			want: "map[val:A]",
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
							event.Actions.StateDelta = map[string]any{
								"output": val,
							}
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
			want: "map[val:B]",
		},
		{
			name: "schema_validation_error",
			agent: func() (agent.Agent, error) {
				return agent.New(agent.Config{
					Name: "test_agent",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							event := session.NewEvent(ctx.InvocationID())
							event.Actions.StateDelta = map[string]any{
								"output": map[string]any{"result": "not-an-int"},
							}
							yield(event, nil)
						}
					},
				})
			},
			nodeInput: map[string]any{},
			node: func(a agent.Agent) (Node, error) {
				return NewAgentNodeTyped[map[string]any, ErrorOutput](a, defaultNodeConfig)
			},
			wantErr: "converting agent \"test_agent\" output",
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

				output, ok := ev.Actions.StateDelta["output"]
				if !ok {
					t.Fatal("expected output in state delta")
				}

				got = tc.extract(t, output)
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
						// valStr will be something like "map[val:5]"
						var val int
						fmt.Sscanf(valStr, "map[val:%d]", &val)
						
						event := session.NewEvent(ctx.InvocationID())
						event.Actions.StateDelta = map[string]any{
							"output": map[string]any{"result": val * 2},
						}
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
				w, err := New(edges)
				if err != nil {
					t.Fatalf("unexpexted error: %v", err)
				}
				events := w.Run(mockCtx)

				var outB any
				for ev, err := range events {
					if err != nil {
						t.Fatalf("workflow failed: %v", err)
					}
					if ev.Actions.StateDelta != nil {
						if out, ok := ev.Actions.StateDelta["output"]; ok {
							outB = out
						}
					}
				}

				if diff := cmp.Diff(tc.want, outB); diff != "" {
					t.Errorf("output mismatch (-want +got):\n%s", diff)
				}
			})
		})
	}
}
