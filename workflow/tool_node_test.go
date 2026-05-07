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
			case *toolNode:
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

func TestToolNode_Run(t *testing.T) {
	type Input struct {
		Name string `json:"name"`
	}
	type Output struct {
		Greeting string `json:"greeting"`
	}
	type ErrorOutput struct {
		Result int `json:"result"`
	}

	tests := []struct {
		name      string
		tool      func() (tool.Tool, error)
		nodeInput any
		node      func(tool.Tool) (Node, error)
		extract   func(t *testing.T, out any) string
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
			extract: func(t *testing.T, out any) string {
				bytes, err := json.Marshal(out)
				if err != nil {
					t.Fatalf("json marshal output: %v", err)
				}
				var output Output
				if err := json.Unmarshal(bytes, &output); err != nil {
					t.Fatalf("json unmarsal output: %v", err)
				}
				return output.Greeting
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
			extract: func(t *testing.T, out any) string {
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
				return NewToolNodeTyped[map[string]any, ErrorOutput](t)
			},
			wantErr: "converting tool \"test_tool\" output",
		},
		{
			name: "tool_execution_error",
			tool: func() (tool.Tool, error) {
				return functiontool.New(functiontool.Config{
					Name: "fail_tool",
				}, func(ctx tool.Context, in Input) (*Output, error) {
					return nil, errors.New("something went wrong")
				})
			},
			nodeInput: Input{Name: "World"},
			node: func(t tool.Tool) (Node, error) {
				return NewToolNodeTyped[Input, Output](t)
			},
			wantErr: "tool \"fail_tool\" execution failed: something went wrong",
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

			toolNode, err := NewToolNodeTyped[*Input, *Output](doubleTool)
			if err != nil {
				t.Fatalf("NewToolNodeTyped failed: %v", err)
			}

			// Connect to a function node.
			functionNode := NewFunctionNode[Output, int]("plus_one", func(ctx agent.InvocationContext, in Output) (int, error) {
				return in.Result + 1, nil
			})

			mockCtx := &MockInvocationContext{sess: nil}

			t.Run("WorkflowExecution", func(t *testing.T) {
				// Use a seed node to pass the struct input to toolNode,
				// since Workflow.Run currently only passes strings from UserContent.
				seedNode := NewFunctionNode("seed", func(ctx agent.InvocationContext, input any) (*Input, error) {
					return &Input{Val: tc.input}, nil
				})

				edges := Chain(Start, seedNode, toolNode, functionNode)
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
