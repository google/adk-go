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
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
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
				event := session.NewEvent(ctx, ctx.InvocationID())
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
				inputResolved, outputResolved = tn.InputSchema(), tn.OutputSchema()
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
							event := session.NewEvent(ctx, ctx.InvocationID())
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
							event := session.NewEvent(ctx, ctx.InvocationID())
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
			runCtx := agent.NewContext(mockCtx)
			events := node.Run(runCtx, tc.nodeInput)

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

						event := session.NewEvent(ctx, ctx.InvocationID())
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
			functionNode := NewFunctionNode[Output, int]("plus_one", func(ctx agent.Context, in Output) (int, error) {
				return in.Result + 1, nil
			}, NodeConfig{})

			mockCtx := newMockCtx(t)
			mockCtx.sess = &mockSession{id: "test-session-id"} // Ensure session is set

			t.Run("WorkflowExecution", func(t *testing.T) {
				// Use a seed node to pass the struct input to agentNode
				seedNode := NewFunctionNode("seed", func(ctx agent.Context, input any) (*Input, error) {
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
				partial := session.NewEvent(ctx, ctx.InvocationID())
				partial.LLMResponse.Partial = true
				partial.LLMResponse.Content = &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: "Hel"}},
				}
				if !yield(partial, nil) {
					return
				}
				// Thought parts are skipped; text parts concatenate.
				final := session.NewEvent(ctx, ctx.InvocationID())
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
	nc := agent.NewContext(mockCtx)
	var (
		gotPartial *session.Event
		gotFinal   *session.Event
	)
	for ev, err := range node.Run(nc, "ignored") {
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
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Output = "v"
				yield(ev, nil)
				// An event that already carries a scope is left untouched.
				pre := session.NewEvent(ctx, ctx.InvocationID())
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
	exCtx := agent.NewContext(mockCtx)

	var events []*session.Event
	for ev, err := range node.Run(exCtx, "ignored") {
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

func TestAgentNode_ValidationAndContentConversion(t *testing.T) {
	type CustomStruct struct {
		Val string `json:"val"`
	}

	tests := []struct {
		name              string
		nodeInput         any
		parentUserContent *genai.Content
		nodeCreator       func(a agent.Agent) (Node, error)
		wantContent       *genai.Content
	}{
		{
			name:      "struct input into JSON text part",
			nodeInput: CustomStruct{Val: "test-val"},
			nodeCreator: func(a agent.Agent) (Node, error) {
				return NewAgentNodeTyped[CustomStruct, string](a, NodeConfig{})
			},
			wantContent: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: `{"val":"test-val"}`}},
			},
		},
		{
			name:      "string input into text part",
			nodeInput: "direct string input",
			nodeCreator: func(a agent.Agent) (Node, error) {
				return NewAgentNodeTyped[string, string](a, NodeConfig{})
			},
			wantContent: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: "direct string input"}},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			agentInvoked := false
			var gotContent *genai.Content

			myAgent, err := agent.New(agent.Config{
				Name: "test_agent",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						agentInvoked = true
						gotContent = ctx.UserContent()
						event := session.NewEvent(ctx, ctx.InvocationID())
						event.Output = "ok"
						yield(event, nil)
					}
				},
			})
			if err != nil {
				t.Fatalf("failed to create agent: %v", err)
			}

			node, err := tc.nodeCreator(myAgent)
			if err != nil {
				t.Fatalf("failed to create node: %v", err)
			}

			mockCtx := newMockCtx(t)
			mockCtx.sess = &mockSession{id: "test"}
			if tc.parentUserContent != nil {
				mockCtx.userContent = tc.parentUserContent
			}

			exCtx := agent.NewContext(mockCtx)
			events := node.Run(exCtx, tc.nodeInput)
			for _, err := range events {
				if err != nil {
					t.Fatalf("run failed: %v", err)
				}
			}

			if !agentInvoked {
				t.Error("agent was not invoked")
			}

			if tc.wantContent != nil {
				if gotContent == nil {
					t.Fatal("expected user content, got nil")
				}
				if diff := cmp.Diff(tc.wantContent, gotContent); diff != "" {
					t.Errorf("user content mismatch (-want +got):\n%s", diff)
				}
			} else if gotContent != nil {
				t.Errorf("expected no user content, got: %v", gotContent)
			}
		})
	}
}

// TestNodeInputToContent_TypedNilContent guards that a typed-nil
// *genai.Content is treated as nil input instead of panicking on v.Parts.
func TestNodeInputToContent_TypedNilContent(t *testing.T) {
	var typedNil *genai.Content
	var input any = typedNil // non-nil interface wrapping a nil *genai.Content

	got, err := nodeInputToContent(input)
	if err != nil {
		t.Fatalf("nodeInputToContent returned error: %v", err)
	}
	if got != nil {
		t.Errorf("nodeInputToContent(typed-nil) = %#v, want nil", got)
	}
}

func TestAgentNode_InputValidation(t *testing.T) {
	type CustomStruct struct {
		Val string `json:"val"`
	}

	tests := []struct {
		name        string
		nodeCreator func(a agent.Agent) (Node, error)
		nodeInput   any
		wantErr     error
		wantContent *genai.Content
	}{
		{
			name: "validation failure -> scheduler rejects",
			nodeCreator: func(a agent.Agent) (Node, error) {
				return NewAgentNodeTyped[string, string](a, NodeConfig{})
			},
			nodeInput: CustomStruct{Val: "invalid"},
			wantErr:   ErrInputValidation,
		},
		{
			name: "validation success -> agent invoked",
			nodeCreator: func(a agent.Agent) (Node, error) {
				return NewAgentNodeTyped[CustomStruct, string](a, NodeConfig{})
			},
			nodeInput: CustomStruct{Val: "hello"},
			wantContent: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: `{"val":"hello"}`}},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			agentInvoked := false
			var gotContent *genai.Content

			myAgent, err := agent.New(agent.Config{
				Name: "test_agent",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						agentInvoked = true
						gotContent = ctx.UserContent()
						event := session.NewEvent(ctx, ctx.InvocationID())
						event.Output = "ok"
						yield(event, nil)
					}
				},
			})
			if err != nil {
				t.Fatalf("failed to create agent: %v", err)
			}

			node, err := tc.nodeCreator(myAgent)
			if err != nil {
				t.Fatalf("failed to create node: %v", err)
			}

			edges := Chain(Start, node)
			w, err := New("validation_wf", edges)
			if err != nil {
				t.Fatalf("failed to build workflow: %v", err)
			}

			mockCtx := newMockCtx(t)
			mockCtx.sess = &mockSession{id: "test"}

			exCtx := agent.NewContext(mockCtx)
			events := w.RunNode(exCtx, tc.nodeInput)
			var workflowErr error
			for _, err := range events {
				if err != nil {
					workflowErr = err
				}
			}

			if tc.wantErr != nil {
				if workflowErr == nil {
					t.Fatal("expected validation error, got nil")
				}
				if !errors.Is(workflowErr, tc.wantErr) {
					t.Errorf("expected error %v, got: %v", tc.wantErr, workflowErr)
				}
				if agentInvoked {
					t.Error("agent was invoked despite validation failure")
				}
				return
			}

			if workflowErr != nil {
				t.Fatalf("workflow failed: %v", workflowErr)
			}

			if !agentInvoked {
				t.Error("agent was not invoked")
			}

			if tc.wantContent != nil {
				if gotContent == nil {
					t.Fatal("expected user content, got nil")
				}
				if diff := cmp.Diff(tc.wantContent, gotContent); diff != "" {
					t.Errorf("user content mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

// TestAgentNode_StructuredOutputProjectedViaValidation verifies the
// end-to-end path that makes the validation fallback reachable: an
// AgentNode with a structured output schema yields JSON model text,
// and ValidateOutput projects it onto the schema.
func TestAgentNode_StructuredOutputProjectedViaValidation(t *testing.T) {
	wrapped, err := agent.New(agent.Config{
		Name: "json-talky",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				final := session.NewEvent(ctx, ctx.InvocationID())
				final.LLMResponse.Content = &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: `{"value":"hello"}`}},
				}
				yield(final, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	outSchema, err := jsonschema.For[testSchemaInput](nil)
	if err != nil {
		t.Fatalf("jsonschema.For: %v", err)
	}
	node, err := NewAgentNodeWithSchemas(wrapped, nil, outSchema, NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeWithSchemas: %v", err)
	}

	mockCtx := newMockCtx(t)
	mockCtx.sess = &mockSession{id: "test-session-id"}
	exCtx := agent.NewContext(mockCtx)
	var gotFinal *session.Event
	for ev, err := range node.Run(exCtx, "ignored") {
		if err != nil {
			t.Fatalf("node.Run: %v", err)
		}
		if !ev.LLMResponse.Partial {
			gotFinal = ev
		}
	}
	if gotFinal == nil {
		t.Fatal("missing final event")
	}

	// AgentNode itself only synthesizes the raw text; the projection
	// onto the schema happens in ValidateOutput.
	got, err := node.ValidateOutput(gotFinal.Output)
	if err != nil {
		t.Fatalf("ValidateOutput: %v", err)
	}
	gotMap, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("ValidateOutput returned %T, want map[string]any", got)
	}
	if gotMap["value"] != "hello" {
		t.Errorf("got %v, want value=hello", gotMap)
	}
}

func TestAgentNode_AutomaticOutputExtraction(t *testing.T) {
	myAgent, err := agent.New(agent.Config{
		Name: "text_only_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				event := session.NewEvent(ctx, ctx.InvocationID())
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
	exCtx := agent.NewContext(mockCtx)
	events := node.Run(exCtx, nil)

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

// resumeProbeAgent records the user content of every activation and, on its
// first one, raises a long-running interrupt authored by author (a sub-agent
// name exercises attribution beyond the node's own agent).
type resumeProbeAgent struct {
	mu       sync.Mutex
	inputs   []*genai.Content
	author   string
	routes   []string
	interrID string
}

func (p *resumeProbeAgent) new(t *testing.T, name string) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				p.mu.Lock()
				n := len(p.inputs)
				p.inputs = append(p.inputs, ctx.UserContent())
				p.mu.Unlock()

				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = name
				if p.author != "" {
					ev.Author = p.author
				}
				if n == 0 && p.interrID != "" {
					ev.LongRunningToolIDs = []string{p.interrID}
				} else {
					ev.Output = "out"
				}
				if n < len(p.routes) {
					ev.Routes = []string{p.routes[n]}
				}
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	return a
}

func (p *resumeProbeAgent) activation(t *testing.T, i int) *genai.Content {
	t.Helper()
	p.mu.Lock()
	defer p.mu.Unlock()
	if i >= len(p.inputs) {
		t.Fatalf("activation %d missing; node ran %d time(s)", i+1, len(p.inputs))
	}
	return p.inputs[i]
}

// answeredEvent is a user turn answering interrupt id.
func answeredEvent(invocationID, id string) *session.Event {
	ev := &session.Event{InvocationID: invocationID, Author: "user"}
	ev.LLMResponse.Content = &genai.Content{
		Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: id, Response: map[string]any{"response": "ok"}}}},
	}
	return ev
}

// TestAgentNode_LoopBackKeepsInput pins that only the resume activation drops
// the node input: a node re-activated by a loop-back edge later in the same
// invocation starts a fresh lifecycle and must still receive its input.
func TestAgentNode_LoopBackKeepsInput(t *testing.T) {
	const name = "looper"
	probe := &resumeProbeAgent{routes: []string{"loop", "finish"}}
	node, err := NewAgentNode(probe.new(t, name), NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	finish := newRecordingNode("Finish")
	finish.release()

	w := mustNew(t, []Edge{
		{From: Start, To: node},
		{From: node, To: node, Route: StringRoute("loop")},
		{From: node, To: finish, Route: StringRoute("finish")},
	})

	// An interrupt of this node's, answered in the current invocation: a
	// history scan would latch on it and starve every later activation.
	ctx := newSeededMockCtx(t)
	ctx.sess = &eventsSession{events: sliceEvents{
		{InvocationID: "test-invocation-id", Author: name, LongRunningToolIDs: []string{"X"}},
		answeredEvent("test-invocation-id", "X"),
	}}

	for _, err := range w.Run(ctx) {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
	}
	if got := probe.activation(t, 1); got == nil {
		t.Error("loop-back activation got no input, want the routed input")
	}
}

// TestAgentNode_ResumesOnSubAgentInterrupt pins that a pause raised by a
// sub-agent the node's agent delegated to still counts as the node's own
// resume: the engine attributes it to the node and re-enters it, so Run must
// drop the input rather than re-feed it and re-trigger the pending tool.
func TestAgentNode_ResumesOnSubAgentInterrupt(t *testing.T) {
	const (
		name        = "coord"
		invocation  = "test-invocation-id"
		interruptID = "X"
	)
	probe := &resumeProbeAgent{author: "subagent", interrID: interruptID}
	rerun := true
	node, err := NewAgentNode(probe.new(t, name), NodeConfig{RerunOnResume: &rerun})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	producer := newStubNode("producer", "task-input")
	w := mustNew(t, []Edge{{From: Start, To: producer}, {From: producer, To: node}})

	ctx1 := newSeededMockCtx(t)
	ctx1.sess = &eventsSession{events: sliceEvents{}}
	var history sliceEvents
	for ev, err := range w.Run(ctx1) {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if ev.InvocationID == "" {
			ev.InvocationID = invocation
		}
		history = append(history, ev)
	}
	history = append(history, answeredEvent(invocation, interruptID))

	sess := &eventsSession{events: history}
	state, err := w.ReconstructRunState(sess, invocation)
	if err != nil {
		t.Fatalf("ReconstructRunState: %v", err)
	}
	if state == nil {
		t.Fatal("the sub-agent's interrupt was not attributed to the node")
	}
	ctx2 := newSeededMockCtx(t)
	ctx2.sess = sess
	for _, err := range w.Resume(agent.NewContext(ctx2), state, map[string]any{interruptID: "ok"}) {
		if err != nil {
			t.Fatalf("Resume: %v", err)
		}
	}
	if got := probe.activation(t, 1); got != nil {
		t.Errorf("resume activation got input %v, want none (continue from history)", got)
	}
}

// TestAgentNode_FreshDynamicChildKeepsInput pins the other side of resume
// detection: a dynamic child inherits the resume context of the ancestor that
// paused, but a child that never paused is a fresh run and keeps its input.
func TestAgentNode_FreshDynamicChildKeepsInput(t *testing.T) {
	const (
		invocation  = "test-invocation-id"
		interruptID = "ask-1"
	)
	probe := &resumeProbeAgent{}
	childNode, err := NewAgentNode(probe.new(t, "child"), NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	asker := &resumeAskerNode{BaseNode: NewBaseNode("asker", "", NodeConfig{}), id: interruptID}

	orchestrator := NewDynamicNode[any, string]("orch",
		func(nc agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			if _, err := RunNode[any](nc, asker, nil); err != nil {
				return "", err
			}
			out, err := RunNode[any](nc, childNode, "fresh-child-input")
			if err != nil {
				return "", err
			}
			s, _ := out.(string)
			return s, nil
		}, NodeConfig{})

	w := mustNew(t, []Edge{{From: Start, To: orchestrator}})

	ctx1 := newSeededMockCtx(t)
	ctx1.sess = &eventsSession{events: sliceEvents{}}
	var history sliceEvents
	for ev, err := range w.Run(ctx1) {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if ev.InvocationID == "" {
			ev.InvocationID = invocation
		}
		history = append(history, ev)
	}
	history = append(history, answeredEvent(invocation, interruptID))

	sess := &eventsSession{events: history}
	state, err := w.ReconstructRunState(sess, invocation)
	if err != nil {
		t.Fatalf("ReconstructRunState: %v", err)
	}
	if state == nil {
		t.Fatal("no run state rehydrated")
	}
	ctx2 := newSeededMockCtx(t)
	ctx2.sess = sess
	for _, err := range w.Resume(agent.NewContext(ctx2), state, map[string]any{interruptID: "ok"}) {
		if err != nil {
			t.Fatalf("Resume: %v", err)
		}
	}
	if got := probe.activation(t, 0); got == nil {
		t.Error("freshly delegated child got no input, want the delegated input")
	}
}

// resumeAskerNode pauses once on its interrupt, then returns the reply.
type resumeAskerNode struct {
	BaseNode
	id string
}

func (n *resumeAskerNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if resp, ok := ctx.ResumedInput(n.id); ok {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = resp
			yield(ev, nil)
			return
		}
		yield(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: n.id, Message: "?"}), nil)
	}
}

// TestAgentNode_RedelegatedChildKeepsInput pins the other half of
// TestAgentNode_FreshDynamicChildKeepsInput: a dynamic child that paused
// earlier and is delegated AGAIN, with new input, during the resumed
// orchestrator's body. The child inherits the ancestor's resume context, so
// the per-activation flag alone cannot tell the two delegations apart —
// attribution has to be per node PATH (child@1 vs child@2), not per node name.
func TestAgentNode_RedelegatedChildKeepsInput(t *testing.T) {
	const (
		invocation  = "test-invocation-id"
		interruptID = "ask-1"
	)
	probe := &pausingProbeAgent{interrID: interruptID}
	child, err := NewAgentNode(probe.new(t, "child"), NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	orchestrator := NewDynamicNode[any, string]("orch",
		func(nc agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			if _, err := RunNode[any](nc, child, "first"); err != nil {
				return "", err
			}
			out, err := RunNode[any](nc, child, "second")
			s, _ := out.(string)
			return s, err
		}, NodeConfig{})
	w := mustNew(t, []Edge{{From: Start, To: orchestrator}})

	ctx1 := newSeededMockCtx(t)
	ctx1.sess = &eventsSession{events: sliceEvents{}}
	var history sliceEvents
	for ev, err := range w.Run(ctx1) {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if ev.InvocationID == "" {
			ev.InvocationID = invocation
		}
		history = append(history, ev)
	}
	history = append(history, answeredEvent(invocation, interruptID))

	sess := &eventsSession{events: history}
	state, err := w.ReconstructRunState(sess, invocation)
	if err != nil {
		t.Fatalf("ReconstructRunState: %v", err)
	}
	if state == nil {
		t.Fatal("no run state rehydrated")
	}
	ctx2 := newSeededMockCtx(t)
	ctx2.sess = sess
	for _, err := range w.Resume(agent.NewContext(ctx2), state, map[string]any{interruptID: "ok"}) {
		if err != nil {
			t.Fatalf("Resume: %v", err)
		}
	}
	// activation 0: turn 1, "first" (paused). activation 1: the replayed
	// child@1 — input correctly dropped. activation 2: the fresh child@2.
	if got := probe.activation(t, 1); got != nil {
		t.Errorf("replayed child@1 got input %v, want none (it is the paused delegation)", got)
	}
	if got := probe.activation(t, 2); got == nil {
		t.Error("re-delegated child@2 lost its input; a fresh delegation must keep it")
	}
}

// pausingProbeAgent parks its node on the first activation with a real
// RequestInput pause (a bare LongRunningToolIDs event does not halt a
// delegating orchestrator's body), and records every activation's input.
type pausingProbeAgent struct {
	mu       sync.Mutex
	inputs   []*genai.Content
	interrID string
}

func (p *pausingProbeAgent) new(t *testing.T, name string) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				p.mu.Lock()
				n := len(p.inputs)
				p.inputs = append(p.inputs, ctx.UserContent())
				p.mu.Unlock()

				if n == 0 {
					yield(NewRequestInputEvent(ctx, session.RequestInput{
						InterruptID: p.interrID, Message: "?",
					}), nil)
					return
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = name
				ev.Output = "out"
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	return a
}

func (p *pausingProbeAgent) activation(t *testing.T, i int) *genai.Content {
	t.Helper()
	p.mu.Lock()
	defer p.mu.Unlock()
	if i >= len(p.inputs) {
		t.Fatalf("activation %d missing; node ran %d time(s)", i+1, len(p.inputs))
	}
	return p.inputs[i]
}
