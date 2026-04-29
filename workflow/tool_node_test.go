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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

func TestToolNode_New(t *testing.T) {
	type Input struct {
		Value string `json:"value"`
	}
	type Output struct {
		Result string `json:"result"`
	}

	myTool, err := functiontool.New(functiontool.Config{
		Name:        "test_tool",
		Description: "a test tool",
	}, func(ctx tool.Context, in Input) (Output, error) {
		return Output{Result: strings.ToUpper(in.Value)}, nil
	})
	if err != nil {
		t.Fatalf("failed to create tool: %v", err)
	}

	ischema, _ := jsonschema.For[Input](nil)
	oschema, _ := jsonschema.For[Output](nil)

	tests := []struct {
		name    string
		creator func() (Node, error)
		want    string
	}{
		{
			name: "NewToolNodeTyped",
			creator: func() (Node, error) {
				return NewToolNodeTyped[Input, Output](myTool)
			},
		},
		{
			name: "NewToolNodeWithSchemas",
			creator: func() (Node, error) {
				return NewToolNodeWithSchemas(myTool, ischema, oschema)
			},
		},
		{
			name: "NewToolNodeWithSchemasTyped",
			creator: func() (Node, error) {
				return NewToolNodeWithSchemasTyped[Input, Output](myTool, nil, nil)
			},
		},
		{
			name: "NewToolNode",
			creator: func() (Node, error) {
				return NewToolNode(myTool)
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node, err := tc.creator()
			if err != nil {
				t.Fatalf("creation failed: %v", err)
			}

			if got, want := node.Name(), "test_tool"; got != want {
				t.Errorf("node.Name() = %q, want %q", got, want)
			}
			if got, want := node.Description(), "a test tool"; got != want {
				t.Errorf("node.Description() = %q, want %q", got, want)
			}

			// Basic internal check via reflection-like cast.
			// We use any, any for constructors that don't preserve types in the struct.
			var inputResolved, outputResolved *jsonschema.Resolved
			switch tn := node.(type) {
			case *toolNode[Input, Output]:
				inputResolved, outputResolved = tn.inputSchema, tn.outputSchema
			case *toolNode[any, any]:
				inputResolved, outputResolved = tn.inputSchema, tn.outputSchema
			}

			if inputResolved == nil || outputResolved == nil {
				t.Error("expected schemas to be resolved")
			}
		})
	}
}

func TestToolNode_Run(t *testing.T) {
	type Input struct {
		Name string `json:"name"`
	}
	type Output struct {
		Greeting string `json:"greeting"`
	}

	tests := []struct {
		name      string
		tool      func() (tool.Tool, error)
		nodeInput any
		node      func(tool.Tool) (Node, error)
		extract   func(any) string
		want      string
		wantErr   string
	}{
		{
			name: "struct_input_output",
			tool: func() (tool.Tool, error) {
				return functiontool.New(functiontool.Config{
					Name: "greet",
				}, func(ctx tool.Context, in Input) (Output, error) {
					return Output{Greeting: "Hello " + in.Name}, nil
				})
			},
			nodeInput: Input{Name: "World"},
			node: func(t tool.Tool) (Node, error) {
				return NewToolNodeTyped[Input, Output](t)
			},
			extract: func(out any) string {
				return out.(Output).Greeting
			},
			want: "Hello World",
		},
		{
			name: "string_output",
			tool: func() (tool.Tool, error) {
				return functiontool.New(functiontool.Config{
					Name: "greet",
				}, func(ctx tool.Context, in Input) (string, error) {
					return "HELLO " + strings.ToUpper(in.Name), nil
				})
			},
			nodeInput: Input{Name: "world"},
			node: func(t tool.Tool) (Node, error) {
				return NewToolNodeTyped[Input, string](t)
			},
			extract: func(out any) string {
				return out.(string)
			},
			want: "HELLO WORLD",
		},
		{
			name: "schema_validation_error",
			tool: func() (tool.Tool, error) {
				return functiontool.New(functiontool.Config{
					Name: "test_tool",
				}, func(ctx tool.Context, in map[string]any) (map[string]any, error) {
					return map[string]any{"result": "not-an-int"}, nil
				})
			},
			nodeInput: map[string]any{},
			node: func(t tool.Tool) (Node, error) {
				type ErrorOutput struct {
					Result int `json:"result"`
				}
				return NewToolNodeTyped[map[string]any, ErrorOutput](t)
			},
			wantErr: "converting tool \"test_tool\" output",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			myTool, err := tc.tool()
			if err != nil {
				t.Fatalf("failed to create tool: %v", err)
			}

			node, err := tc.node(myTool)
			if err != nil {
				t.Fatalf("node creation failed: %v", err)
			}

			mockCtx := &MockInvocationContext{sess: nil}
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

				got = tc.extract(output)
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

func TestToolNode_WorkflowIntegration(t *testing.T) {
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
			name:  "chain_tool_and_function",
			input: 5,
			want:  11,
		},
		{
			name:  "chain_tool_and_function_zero",
			input: 0,
			want:  1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			doubleTool, err := functiontool.New(functiontool.Config{
				Name: "double",
			}, func(ctx tool.Context, in *Input) (Output, error) {
				return Output{Result: in.Val * 2}, nil
			})
			if err != nil {
				t.Fatalf("failed to create tool: %v", err)
			}

			nodeA, err := NewToolNodeTyped[*Input, Output](doubleTool)
			if err != nil {
				t.Fatalf("NewToolNodeTyped failed: %v", err)
			}

			// Connect to a function node.
			nodeB := NewFunctionNode("plus_one", func(ctx agent.InvocationContext, in *Output) (int, error) {
				return in.Result + 1, nil
			})

			mockCtx := &MockInvocationContext{sess: nil}

			t.Run("DirectChainExecution", func(t *testing.T) {
				// Test direct chain execution.
				eventsA := nodeA.Run(mockCtx, &Input{Val: tc.input})
				var outA any
				for ev, err := range eventsA {
					if err != nil {
						t.Fatalf("nodeA.Run failed: %v", err)
					}
					outA = ev.Actions.StateDelta["output"]
				}

				// outA should be Output struct.
				typedOutA, ok := outA.(Output)
				if !ok {
					t.Fatalf("unexpected output type from nodeA: %T, want Output", outA)
				}

				eventsB := nodeB.Run(mockCtx, &typedOutA)
				var outB any
				for ev, err := range eventsB {
					if err != nil {
						t.Fatalf("nodeB.Run failed: %v", err)
					}
					outB = ev.Actions.StateDelta["output"]
				}

				if diff := cmp.Diff(tc.want, outB); diff != "" {
					t.Errorf("output mismatch (-want +got):\n%s", diff)
				}
			})
		})
	}
}
