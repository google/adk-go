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
	"fmt"
	"iter"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

func TestNewFunctionNodeWithSchema(t *testing.T) {
	type Input struct {
		Value string `json:"value"`
	}
	type Output struct {
		Result string `json:"result"`
	}
	type TargetOutput struct {
		Result int `json:"result"`
	}

	tests := []struct {
		name         string
		nodeName     string
		fn           func(ctx agent.Context, input Input) (map[string]any, error)
		inputSchema  *jsonschema.Schema
		outputSchema *jsonschema.Schema
		input        any
		wantOutput   map[string]any
		wantErr      bool
		errSubstr    string
	}{
		{
			name:     "Success",
			nodeName: "upper",
			fn: func(ctx agent.Context, input Input) (map[string]any, error) {
				return map[string]any{"result": strings.ToUpper(input.Value)}, nil
			},
			inputSchema:  mustSchema[Input](t),
			outputSchema: mustSchema[Output](t),
			input:        Input{Value: "hello"},
			wantOutput:   map[string]any{"result": "HELLO"},
			wantErr:      false,
		},
		{
			name:     "NilInput",
			nodeName: "nil_test",
			fn: func(ctx agent.Context, input Input) (map[string]any, error) {
				if input.Value == "" {
					return map[string]any{"result": "zero"}, nil
				}
				return map[string]any{"result": "not-zero"}, nil
			},
			inputSchema:  mustSchema[Input](t),
			outputSchema: mustSchema[Output](t),
			input:        nil,
			wantOutput:   map[string]any{"result": "zero"},
			wantErr:      false,
		},
		{
			name:     "ValidationError",
			nodeName: "test",
			fn: func(ctx agent.Context, input Input) (map[string]any, error) {
				return map[string]any{"result": "not-an-int"}, nil
			},
			inputSchema:  mustSchema[Input](t),
			outputSchema: mustSchema[TargetOutput](t),
			input:        Input{Value: "hello"},
			wantErr:      true,
			errSubstr:    "validation failed for output",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node, err := NewFunctionNodeWithSchema[Input, map[string]any](tc.nodeName, tc.fn, tc.inputSchema, tc.outputSchema, defaultNodeConfig)
			if err != nil {
				t.Fatalf("NewFunctionNodeWithSchema failed: %v", err)
			}

			mockCtx := &MockInvocationContext{sess: nil}
			exCtx := agent.NewNodeContext(mockCtx, nil)
			events := node.Run(exCtx, tc.input)

			count := 0
			for ev, err := range events {
				if err != nil {
					if !tc.wantErr {
						t.Fatalf("unexpected error: %v", err)
					}
					if tc.errSubstr != "" && !strings.Contains(err.Error(), tc.errSubstr) {
						t.Errorf("expected error containing %q, got %v", tc.errSubstr, err)
					}
					return // Expected error handled
				}
				count++
				if tc.wantErr {
					t.Fatal("expected error, got nil")
				}

				if diff := cmp.Diff(tc.wantOutput, ev.Output); diff != "" {
					t.Errorf("output mismatch (-want +got):\n%s", diff)
				}
			}

			if !tc.wantErr && count != 1 {
				t.Errorf("expected 1 event, got %d", count)
			}
		})
	}
}

func mustSchema[T any](t *testing.T) *jsonschema.Schema {
	t.Helper()
	s, err := jsonschema.For[T](nil)
	if err != nil {
		t.Fatalf("jsonschema.For failed: %v", err)
	}
	return s
}

func TestFunctionNodeDirectEventPropagation(t *testing.T) {
	fn := func(ctx agent.Context, input string) (*session.Event, error) {
		ev := session.NewEvent(ctx.InvocationID())
		ev.Output = input + " processed"
		ev.Routes = []string{"CUSTOM_ROUTE"}
		return ev, nil
	}

	node := NewFunctionNode[string, *session.Event]("event_proc", fn, defaultNodeConfig)
	mockCtx := &MockInvocationContext{sess: nil}
	exCtx := agent.NewNodeContext(mockCtx, nil)

	events := node.Run(exCtx, "hello")

	var yieldedEvents []*session.Event
	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error running node: %v", err)
		}
		yieldedEvents = append(yieldedEvents, ev)
	}

	if len(yieldedEvents) != 1 {
		t.Fatalf("expected 1 event, got %d", len(yieldedEvents))
	}

	ev := yieldedEvents[0]
	if ev.Output != "hello processed" {
		t.Errorf("expected Output 'hello processed', got %v", ev.Output)
	}
	if len(ev.Routes) != 1 || ev.Routes[0] != "CUSTOM_ROUTE" {
		t.Errorf("expected Routes ['CUSTOM_ROUTE'], got %v", ev.Routes)
	}
}

func TestNewFunctionNodeFromState(t *testing.T) {
	type TwoFieldParams struct {
		Foo string `state:"foo_key"`
		Bar int
	}

	type NodeInputParams struct {
		PredecessorOutput string `state:"node_input"`
		OtherVal          int    `state:"other_val"`
	}

	type OutputStruct struct {
		Message string
		Code    int
	}

	tests := []struct {
		name                string
		setupFn             func() (*FunctionNode, error)
		stateData           map[string]any
		input               any
		wantOutput          any
		wantErr             bool
		errSubstr           string
		wantStateFieldNames []string
	}{
		{
			name: "Success",
			setupFn: func() (*FunctionNode, error) {
				return NewFunctionNodeFromState("test1", func(ctx agent.InvocationContext, p TwoFieldParams) (string, error) {
					return fmt.Sprintf("%s-%d", p.Foo, p.Bar), nil
				}, defaultNodeConfig)
			},
			stateData: map[string]any{
				"foo_key": "hello",
				"Bar":     42,
			},
			input:               nil,
			wantOutput:          "hello-42",
			wantStateFieldNames: []string{"foo_key", "Bar"},
		},
		{
			name: "Missing_state_key",
			setupFn: func() (*FunctionNode, error) {
				return NewFunctionNodeFromState("test2", func(ctx agent.InvocationContext, p TwoFieldParams) (string, error) {
					return "", nil
				}, defaultNodeConfig)
			},
			stateData: map[string]any{
				"foo_key": "hello",
				// Bar missing
			},
			wantErr:             true,
			errSubstr:           "missing state value for required field",
			wantStateFieldNames: []string{"foo_key", "Bar"},
		},
		{
			name: "Type_mismatch",
			setupFn: func() (*FunctionNode, error) {
				return NewFunctionNodeFromState("test3", func(ctx agent.InvocationContext, p TwoFieldParams) (string, error) {
					return "", nil
				}, defaultNodeConfig)
			},
			stateData: map[string]any{
				"foo_key": "hello",
				"Bar":     "not-an-int",
			},
			wantErr:             true,
			errSubstr:           "failed to convert state value to field",
			wantStateFieldNames: []string{"foo_key", "Bar"},
		},
		{
			name: "Node_input_success",
			setupFn: func() (*FunctionNode, error) {
				return NewFunctionNodeFromState("test4", func(ctx agent.InvocationContext, p NodeInputParams) (string, error) {
					return fmt.Sprintf("input:%s,other:%d", p.PredecessorOutput, p.OtherVal), nil
				}, defaultNodeConfig)
			},
			stateData: map[string]any{
				"other_val": 100,
			},
			input:               "from_pred",
			wantOutput:          "input:from_pred,other:100",
			wantStateFieldNames: []string{"other_val"},
		},
		{
			name: "Node_input_type_mismatch",
			setupFn: func() (*FunctionNode, error) {
				return NewFunctionNodeFromState("test5", func(ctx agent.InvocationContext, p NodeInputParams) (string, error) {
					return "", nil
				}, defaultNodeConfig)
			},
			stateData: map[string]any{
				"other_val": 100,
			},
			input:               123, // should be string
			wantErr:             true,
			errSubstr:           "invalid input type for node_input",
			wantStateFieldNames: []string{"other_val"},
		},
		{
			name: "Struct_output_success",
			setupFn: func() (*FunctionNode, error) {
				return NewFunctionNodeFromState("test6", func(ctx agent.InvocationContext, p TwoFieldParams) (OutputStruct, error) {
					return OutputStruct{
						Message: p.Foo,
						Code:    p.Bar,
					}, nil
				}, defaultNodeConfig)
			},
			stateData: map[string]any{
				"foo_key": "hello",
				"Bar":     42,
			},
			input:               nil,
			wantOutput:          OutputStruct{Message: "hello", Code: 42},
			wantStateFieldNames: []string{"foo_key", "Bar"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node, err := tc.setupFn()
			if err != nil {
				t.Fatalf("setupFn failed: %v", err)
			}

			if diff := cmp.Diff(tc.wantStateFieldNames, node.StateFieldNames()); diff != "" {
				t.Errorf("StateFieldNames mismatch (-want +got):\n%s", diff)
			}

			mockSess := &mockSessionForTest{
				state: &mockStateForTest{data: tc.stateData},
			}
			mockCtx := &MockInvocationContext{sess: mockSess}
			exCtx := agent.NewNodeContext(mockCtx, nil)

			events := node.Run(exCtx, tc.input)

			var lastErr error
			var output any
			for ev, err := range events {
				if err != nil {
					lastErr = err
					break
				}
				output = ev.Output
			}

			if tc.wantErr {
				if lastErr == nil {
					t.Fatalf("expected error containing %q, got nil", tc.errSubstr)
				}
				if tc.errSubstr != "" && !strings.Contains(lastErr.Error(), tc.errSubstr) {
					t.Errorf("expected error containing %q, got %v", tc.errSubstr, lastErr)
				}
			} else {
				if lastErr != nil {
					t.Fatalf("unexpected error: %v", lastErr)
				}
				if diff := cmp.Diff(tc.wantOutput, output); diff != "" {
					t.Errorf("output mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

type mockStateForTest struct {
	data map[string]any
}

func (m *mockStateForTest) Get(key string) (any, error) {
	if val, ok := m.data[key]; ok {
		return val, nil
	}
	return nil, session.ErrStateKeyNotExist
}

func (m *mockStateForTest) Set(key string, val any) error {
	m.data[key] = val
	return nil
}

func (m *mockStateForTest) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range m.data {
			if !yield(k, v) {
				return
			}
		}
	}
}

type mockSessionForTest struct {
	session.Session
	state session.State
}

func (m *mockSessionForTest) State() session.State {
	return m.state
}
