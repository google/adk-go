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

package workflow_test

import (
	"errors"
	"fmt"
	"iter"
	"maps"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// ss2Schema resolves an object schema with the given title, whose properties
// map a name to its JSON type.
func ss2Schema(t *testing.T, title string, props map[string]string, required ...string) *jsonschema.Resolved {
	t.Helper()
	s := &jsonschema.Schema{Title: title, Type: "object", Properties: map[string]*jsonschema.Schema{}, Required: required}
	for name, typ := range props {
		s.Properties[name] = &jsonschema.Schema{Type: typ}
	}
	r, err := s.Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve(%s) error = %v", title, err)
	}
	return r
}

// ss2SchemaW is the workflow's schema: counter (integer) and name (string),
// both required.
func ss2SchemaW(t *testing.T) *jsonschema.Resolved {
	t.Helper()
	return ss2Schema(t, "WorkflowSchema", map[string]string{"counter": "integer", "name": "string"}, "counter", "name")
}

// ss2SchemaS is a node's schema: x and y (integers), both required.
func ss2SchemaS(t *testing.T) *jsonschema.Resolved {
	t.Helper()
	return ss2Schema(t, "NodeSchema", map[string]string{"x": "integer", "y": "integer"}, "x", "y")
}

// ss2UnusableSchema returns a resolved schema whose property a refers to
// another document, which its author resolved with a loader.
func ss2UnusableSchema(t *testing.T) *jsonschema.Resolved {
	t.Helper()
	s := &jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"a": {Ref: "https://example.invalid/a.json"}},
	}
	r, err := s.Resolve(&jsonschema.ResolveOptions{
		Loader: func(*url.URL) (*jsonschema.Schema, error) {
			return &jsonschema.Schema{Type: "integer"}, nil
		},
	})
	if err != nil {
		t.Fatalf("Resolve(remote $ref schema) error = %v", err)
	}
	return r
}

// ss2Undeclared is the error for a write of key that schema does not declare.
func ss2Undeclared(key, schema string, declared ...string) *workflow.StateSchemaError {
	return &workflow.StateSchemaError{Key: key, Schema: schema, Reason: workflow.StateKeyUndeclared, Declared: declared}
}

// ss2WantRefused reports whether err is, or wraps, a *StateSchemaError equal
// to want.
func ss2WantRefused(t *testing.T, err error, want *workflow.StateSchemaError) {
	t.Helper()
	se, ok := errors.AsType[*workflow.StateSchemaError](err)
	if !ok {
		t.Errorf("error = %v, want a *workflow.StateSchemaError for key %q", err, want.Key)
		return
	}
	if diff := cmp.Diff(want, se, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("StateSchemaError mismatch (-want +got):\n%s", diff)
	}
}

// ss2Writer returns a function node that calls write and returns "ok" with
// write's error.
func ss2Writer(name string, cfg workflow.NodeConfig, write func(ctx agent.Context) error) *workflow.FunctionNode {
	return workflow.NewFunctionNode(name, func(ctx agent.Context, _ any) (string, error) {
		return "ok", write(ctx)
	}, cfg)
}

// ss2Node returns a custom node with configuration cfg whose Run is run.
func ss2Node(name string, cfg workflow.NodeConfig, run func(ctx agent.Context, yield func(*session.Event, error) bool)) *stateNode {
	return &stateNode{BaseNode: workflow.NewBaseNode(name, "", cfg), run: run}
}

// ss2Orchestrator returns a dynamic node "orch" with configuration cfg that
// runs child and returns "done", or the child's error.
func ss2Orchestrator(cfg workflow.NodeConfig, child workflow.Node) workflow.Node {
	return workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		if _, err := workflow.RunNode[any](ctx, child, nil); err != nil {
			return "", err
		}
		return "done", nil
	}, cfg)
}

func ss2NewWorkflow(t *testing.T, edges []workflow.Edge, opts ...workflow.Option) *workflow.Workflow {
	t.Helper()
	wf, err := workflow.New("wf", edges, opts...)
	if err != nil {
		t.Fatalf("workflow.New() error = %v", err)
	}
	return wf
}

func TestNodeStateSchema_ReplacesWorkflowSchema(t *testing.T) {
	var xErr error
	guarded := ss2Writer("guarded", workflow.NodeConfig{StateSchema: ss2SchemaS(t)}, func(ctx agent.Context) error {
		xErr = ctx.State().Set("x", 1)
		return ctx.State().Set("counter", 2)
	})
	wf := newStateWorkflow(t, guarded, workflow.WithStateSchema(ss2SchemaW(t)))

	_, stored, err := runStateWorkflow(t, wf, nil)

	if xErr != nil {
		t.Errorf("Set(x) error = %v, want nil", xErr)
	}
	ss2WantRefused(t, err, ss2Undeclared("counter", "NodeSchema", "x", "y"))
	if _, ok := stored["counter"]; ok {
		t.Errorf("stored counter = %v, want it absent", stored["counter"])
	}
}

func TestNodeStateSchema_DoesNotReachGraphSuccessors(t *testing.T) {
	newA := func(t *testing.T) workflow.Node {
		return ss2Writer("a", workflow.NodeConfig{StateSchema: ss2SchemaS(t)}, func(ctx agent.Context) error {
			return ctx.State().Set("x", 1)
		})
	}

	t.Run("successor writes a key only the workflow declares", func(t *testing.T) {
		b := ss2Writer("b", workflow.NodeConfig{}, func(ctx agent.Context) error {
			return ctx.State().Set("counter", 5)
		})
		wf := ss2NewWorkflow(t, workflow.Chain(workflow.Start, newA(t), b), workflow.WithStateSchema(ss2SchemaW(t)))

		_, stored, err := runStateWorkflow(t, wf, nil)
		if err != nil {
			t.Fatalf("run error = %v, want nil", err)
		}

		if stored["counter"] != 5 || stored["x"] != 1 {
			t.Errorf("stored counter = %v and x = %v, want 5 and 1", stored["counter"], stored["x"])
		}
	})

	t.Run("successor is checked against the workflow schema", func(t *testing.T) {
		var bErr error
		b := ss2Writer("b", workflow.NodeConfig{}, func(ctx agent.Context) error {
			bErr = ctx.State().Set("x", 2)
			return bErr
		})
		wf := ss2NewWorkflow(t, workflow.Chain(workflow.Start, newA(t), b), workflow.WithStateSchema(ss2SchemaW(t)))

		if _, _, err := runStateWorkflow(t, wf, nil); err == nil {
			t.Errorf("run error = nil, want b's refused write")
		}

		ss2WantRefused(t, bErr, ss2Undeclared("x", "WorkflowSchema", "counter", "name"))
	})
}

// Two node schemas with the same title stay apart, and a workflow without a
// schema checks nothing for a node without one.
func TestNodeStateSchema_EachNodeUsesItsOwnSchema(t *testing.T) {
	var n1X, n1Z, n2Z, n2X error
	n1 := ss2Writer("n1", workflow.NodeConfig{StateSchema: ss2SchemaS(t)}, func(ctx agent.Context) error {
		n1X = ctx.State().Set("x", 1)
		n1Z = ctx.State().Set("z", "v")
		return nil
	})
	s2 := ss2Schema(t, "NodeSchema", map[string]string{"z": "string"})
	n2 := ss2Writer("n2", workflow.NodeConfig{StateSchema: s2}, func(ctx agent.Context) error {
		n2Z = ctx.State().Set("z", "v")
		n2X = ctx.State().Set("x", 1)
		return nil
	})
	wf := ss2NewWorkflow(t, workflow.Chain(workflow.Start, n1, n2))

	_, stored, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}

	if n1X != nil {
		t.Errorf("n1 Set(x) error = %v, want nil", n1X)
	}
	ss2WantRefused(t, n1Z, ss2Undeclared("z", "NodeSchema", "x", "y"))
	if n2Z != nil {
		t.Errorf("n2 Set(z) error = %v, want nil", n2Z)
	}
	ss2WantRefused(t, n2X, ss2Undeclared("x", "NodeSchema", "z"))
	want := map[string]any{"x": 1, "z": "v"}
	if diff := cmp.Diff(want, stored); diff != "" {
		t.Errorf("stored state mismatch (-want +got):\n%s", diff)
	}
}

// A node schema that declares no field is a schema, not "inherit".
func TestNodeStateSchema_SchemaWithNoPropertiesIsNotInherit(t *testing.T) {
	tests := []struct {
		name   string
		schema func(t *testing.T) *jsonschema.Resolved
		want   *workflow.StateSchemaError
	}{
		{
			name: "object schema without properties",
			schema: func(t *testing.T) *jsonschema.Resolved {
				return ss2Schema(t, "Empty", nil)
			},
			want: ss2Undeclared("counter", "Empty"),
		},
		{
			name:   "zero Resolved",
			schema: func(*testing.T) *jsonschema.Resolved { return &jsonschema.Resolved{} },
			want:   ss2Undeclared("counter", ""),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var counterErr, flagErr error
			guarded := ss2Writer("guarded", workflow.NodeConfig{StateSchema: tt.schema(t)}, func(ctx agent.Context) error {
				counterErr = ctx.State().Set("counter", 1)
				flagErr = ctx.State().Set("app:flag", true)
				return nil
			})
			wf := newStateWorkflow(t, guarded, workflow.WithStateSchema(ss2SchemaW(t)))

			_, stored, err := runStateWorkflow(t, wf, nil)
			if err != nil {
				t.Fatalf("run error = %v, want nil", err)
			}

			ss2WantRefused(t, counterErr, tt.want)
			if flagErr != nil {
				t.Errorf("Set(app:flag) error = %v, want nil", flagErr)
			}
			want := map[string]any{"app:flag": true}
			if diff := cmp.Diff(want, stored); diff != "" {
				t.Errorf("stored state mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNodeStateSchema_EventDeltaCheckedAgainstNodeSchema(t *testing.T) {
	kinds := []struct {
		name  string
		build func(t *testing.T, cfg workflow.NodeConfig, delta map[string]any) workflow.Node
	}{
		{
			name: "custom node",
			build: func(_ *testing.T, cfg workflow.NodeConfig, delta map[string]any) workflow.Node {
				return ss2Node("guarded", cfg, func(ctx agent.Context, yield func(*session.Event, error) bool) {
					yield(outputEvent(ctx, "ok", delta), nil)
				})
			},
		},
		{
			name: "agent node",
			build: func(t *testing.T, cfg workflow.NodeConfig, delta map[string]any) workflow.Node {
				a, err := agent.New(agent.Config{Name: "guarded", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "guarded"
						maps.Copy(ev.Actions.StateDelta, delta)
						yield(ev, nil)
					}
				}})
				if err != nil {
					t.Fatalf("agent.New() error = %v", err)
				}
				n, err := workflow.NewAgentNode(a, cfg)
				if err != nil {
					t.Fatalf("NewAgentNode() error = %v", err)
				}
				return n
			},
		},
	}
	for _, kind := range kinds {
		t.Run(kind.name+"/declared key accepted", func(t *testing.T) {
			node := kind.build(t, workflow.NodeConfig{StateSchema: ss2SchemaS(t)}, map[string]any{"x": 1})
			wf := newStateWorkflow(t, node, workflow.WithStateSchema(ss2SchemaW(t)))

			events, stored, err := runStateWorkflow(t, wf, nil)
			if err != nil {
				t.Fatalf("run error = %v, want nil", err)
			}

			withX := eventsWithKey(events, "x")
			if len(withX) != 1 || withX[0].Actions.StateDelta["x"] != 1 {
				t.Errorf("caller-visible events with x = %d, want 1 with x = 1", len(withX))
			}
			if stored["x"] != 1 {
				t.Errorf("stored x = %v, want 1", stored["x"])
			}
		})
		t.Run(kind.name+"/undeclared key refused", func(t *testing.T) {
			node := kind.build(t, workflow.NodeConfig{StateSchema: ss2SchemaS(t)}, map[string]any{"counter": 1})
			wf := newStateWorkflow(t, node, workflow.WithStateSchema(ss2SchemaW(t)))

			events, stored, err := runStateWorkflow(t, wf, nil)

			ss2WantRefused(t, err, ss2Undeclared("counter", "NodeSchema", "x", "y"))
			if got := eventsWithKey(events, "counter"); len(got) != 0 {
				t.Errorf("%d caller-visible events carry counter, want 0", len(got))
			}
			if _, ok := stored["counter"]; ok {
				t.Errorf("stored counter = %v, want it absent", stored["counter"])
			}
		})
	}
}

func TestNodeStateSchema_DynamicChildInheritsCallingNodeSchema(t *testing.T) {
	t.Run("key the caller declares is accepted", func(t *testing.T) {
		child := ss2Writer("child", workflow.NodeConfig{}, func(ctx agent.Context) error {
			return ctx.State().Set("y", 3)
		})
		orch := ss2Orchestrator(workflow.NodeConfig{StateSchema: ss2SchemaS(t)}, child)
		wf := newStateWorkflow(t, orch, workflow.WithStateSchema(ss2SchemaW(t)))

		_, stored, err := runStateWorkflow(t, wf, nil)
		if err != nil {
			t.Fatalf("run error = %v, want nil", err)
		}

		if stored["y"] != 3 {
			t.Errorf("stored y = %v, want 3", stored["y"])
		}
	})

	t.Run("key only the workflow declares is refused", func(t *testing.T) {
		child := ss2Writer("child", workflow.NodeConfig{}, func(ctx agent.Context) error {
			return ctx.State().Set("counter", 4)
		})
		orch := ss2Orchestrator(workflow.NodeConfig{StateSchema: ss2SchemaS(t)}, child)
		wf := newStateWorkflow(t, orch, workflow.WithStateSchema(ss2SchemaW(t)))

		_, stored, err := runStateWorkflow(t, wf, nil)

		if !errors.Is(err, workflow.ErrNodeFailed) {
			t.Errorf("run error = %v, want it to wrap ErrNodeFailed", err)
		}
		ss2WantRefused(t, err, ss2Undeclared("counter", "NodeSchema", "x", "y"))
		if _, ok := stored["counter"]; ok {
			t.Errorf("stored counter = %v, want it absent", stored["counter"])
		}
	})
}

func TestNodeStateSchema_DynamicChildOwnSchema(t *testing.T) {
	childSchema := func(t *testing.T) workflow.NodeConfig {
		return workflow.NodeConfig{StateSchema: ss2Schema(t, "ChildSchema", map[string]string{"z": "string"})}
	}
	run := func(t *testing.T, child workflow.Node) (map[string]any, error) {
		orch := ss2Orchestrator(workflow.NodeConfig{StateSchema: ss2SchemaS(t)}, child)
		wf := newStateWorkflow(t, orch, workflow.WithStateSchema(ss2SchemaW(t)))
		_, stored, err := runStateWorkflow(t, wf, nil)
		return stored, err
	}

	t.Run("key only the child declares is accepted", func(t *testing.T) {
		child := ss2Writer("child", childSchema(t), func(ctx agent.Context) error {
			return ctx.State().Set("z", "v")
		})

		stored, err := run(t, child)
		if err != nil {
			t.Fatalf("run error = %v, want nil", err)
		}

		if stored["z"] != "v" {
			t.Errorf("stored z = %v, want v", stored["z"])
		}
	})

	t.Run("key only the caller declares is refused", func(t *testing.T) {
		child := ss2Writer("child", childSchema(t), func(ctx agent.Context) error {
			return ctx.State().Set("x", 1)
		})

		_, err := run(t, child)

		ss2WantRefused(t, err, ss2Undeclared("x", "ChildSchema", "z"))
	})

	t.Run("caller does not check the child's event again", func(t *testing.T) {
		child := ss2Node("child", childSchema(t), func(ctx agent.Context, yield func(*session.Event, error) bool) {
			yield(outputEvent(ctx, "c", map[string]any{"z": "v"}), nil)
		})

		stored, err := run(t, child)
		if err != nil {
			t.Fatalf("run error = %v, want nil", err)
		}

		if stored["z"] != "v" {
			t.Errorf("stored z = %v, want v", stored["z"])
		}
	})
}

// ss2RunItems runs a parallel worker with configuration workerCfg over the
// items 0 and 1 under the workflow schema. Each item, with configuration
// itemCfg, sets first and then second to its index; the errors of the two
// writes are returned per item.
func ss2RunItems(t *testing.T, workerCfg, itemCfg workflow.NodeConfig, first, second string) ([2][2]error, map[string]any) {
	t.Helper()
	var errs [2][2]error
	item := &itemNode{BaseNode: workflow.NewBaseNode("item", "", itemCfg), run: func(ctx agent.Context, in any, yield func(*session.Event, error) bool) {
		i := in.(int)
		errs[i][0] = ctx.State().Set(first, i)
		errs[i][1] = ctx.State().Set(second, i)
		yield(outputEvent(ctx, i, nil), nil)
	}}
	worker, err := workflow.NewParallelWorker("pw", item, 0, workerCfg)
	if err != nil {
		t.Fatalf("NewParallelWorker() error = %v", err)
	}
	seed := workflow.NewFunctionNode("seed", func(agent.Context, any) ([]any, error) {
		return []any{0, 1}, nil
	}, workflow.NodeConfig{})
	wf := ss2NewWorkflow(t, workflow.Chain(workflow.Start, seed, worker), workflow.WithStateSchema(ss2SchemaW(t)))

	_, stored, err := runStateWorkflow(t, wf, nil)
	if err != nil {
		t.Fatalf("run error = %v, want nil", err)
	}
	return errs, stored
}

func TestNodeStateSchema_ParallelWorkerItems(t *testing.T) {
	t.Run("wrapped node's schema", func(t *testing.T) {
		errs, stored := ss2RunItems(t, workflow.NodeConfig{}, workflow.NodeConfig{StateSchema: ss2SchemaS(t)}, "x", "counter")

		for i, e := range errs {
			if e[0] != nil {
				t.Errorf("item %d Set(x) error = %v, want nil", i, e[0])
			}
			ss2WantRefused(t, e[1], ss2Undeclared("counter", "NodeSchema", "x", "y"))
		}
		if x := stored["x"]; x != 0 && x != 1 {
			t.Errorf("stored x = %v, want 0 or 1", x)
		}
	})

	t.Run("wrapped node without a schema uses the workflow's", func(t *testing.T) {
		errs, _ := ss2RunItems(t, workflow.NodeConfig{}, workflow.NodeConfig{}, "counter", "x")

		for i, e := range errs {
			if e[0] != nil {
				t.Errorf("item %d Set(counter) error = %v, want nil", i, e[0])
			}
			ss2WantRefused(t, e[1], ss2Undeclared("x", "WorkflowSchema", "counter", "name"))
		}
	})

	t.Run("worker's schema", func(t *testing.T) {
		workerCfg := workflow.NodeConfig{StateSchema: ss2Schema(t, "WorkerSchema", map[string]string{"k": "integer"})}

		errs, _ := ss2RunItems(t, workerCfg, workflow.NodeConfig{}, "k", "counter")

		for i, e := range errs {
			if e[0] != nil {
				t.Errorf("item %d Set(k) error = %v, want nil", i, e[0])
			}
			ss2WantRefused(t, e[1], ss2Undeclared("counter", "WorkerSchema", "k"))
		}
	})
}

func TestNewWorkflowNodeWithStateSchema(t *testing.T) {
	innerSchema := func(t *testing.T) *jsonschema.Resolved {
		return ss2Schema(t, "InnerSchema", map[string]string{"inner_key": "integer"})
	}
	nested := func(t *testing.T, schema *jsonschema.Resolved, inner workflow.Node) *workflow.WorkflowNode {
		t.Helper()
		n, err := workflow.NewWorkflowNodeWithStateSchema("nested", workflow.Chain(workflow.Start, inner), schema)
		if err != nil {
			t.Fatalf("NewWorkflowNodeWithStateSchema() error = %v", err)
		}
		return n
	}

	t.Run("key the nested schema declares is accepted", func(t *testing.T) {
		inner := ss2Writer("inner", workflow.NodeConfig{}, func(ctx agent.Context) error {
			return ctx.State().Set("inner_key", 1)
		})
		wf := newStateWorkflow(t, nested(t, innerSchema(t), inner), workflow.WithStateSchema(ss2SchemaW(t)))

		_, stored, err := runStateWorkflow(t, wf, nil)
		if err != nil {
			t.Fatalf("run error = %v, want nil", err)
		}

		if stored["inner_key"] != 1 {
			t.Errorf("stored inner_key = %v, want 1", stored["inner_key"])
		}
	})

	t.Run("key only the outer schema declares is refused", func(t *testing.T) {
		inner := ss2Writer("inner", workflow.NodeConfig{}, func(ctx agent.Context) error {
			return ctx.State().Set("counter", 2)
		})
		wf := newStateWorkflow(t, nested(t, innerSchema(t), inner), workflow.WithStateSchema(ss2SchemaW(t)))

		_, _, err := runStateWorkflow(t, wf, nil)

		ss2WantRefused(t, err, ss2Undeclared("counter", "InnerSchema", "inner_key"))
	})

	t.Run("nil schema inherits the outer schema", func(t *testing.T) {
		inner := ss2Writer("inner", workflow.NodeConfig{}, func(ctx agent.Context) error {
			return ctx.State().Set("inner_key", 1)
		})
		wf := newStateWorkflow(t, nested(t, nil, inner), workflow.WithStateSchema(ss2SchemaW(t)))

		_, _, err := runStateWorkflow(t, wf, nil)

		ss2WantRefused(t, err, ss2Undeclared("inner_key", "WorkflowSchema", "counter", "name"))
	})

	t.Run("NewWorkflowNode inherits the outer schema", func(t *testing.T) {
		inner := ss2Writer("inner", workflow.NodeConfig{}, func(ctx agent.Context) error {
			return ctx.State().Set("inner_key", 1)
		})
		n, err := workflow.NewWorkflowNode("nested", workflow.Chain(workflow.Start, inner))
		if err != nil {
			t.Fatalf("NewWorkflowNode() error = %v", err)
		}
		wf := newStateWorkflow(t, n, workflow.WithStateSchema(ss2SchemaW(t)))

		_, _, err = runStateWorkflow(t, wf, nil)

		ss2WantRefused(t, err, ss2Undeclared("inner_key", "WorkflowSchema", "counter", "name"))
	})

	t.Run("successor of the nested node uses the outer schema", func(t *testing.T) {
		inner := ss2Writer("inner", workflow.NodeConfig{}, func(ctx agent.Context) error {
			return ctx.State().Set("inner_key", 1)
		})
		var counterErr, innerKeyErr error
		after := ss2Writer("after", workflow.NodeConfig{}, func(ctx agent.Context) error {
			counterErr = ctx.State().Set("counter", 5)
			innerKeyErr = ctx.State().Set("inner_key", 2)
			return nil
		})
		wf := ss2NewWorkflow(t, workflow.Chain(workflow.Start, nested(t, innerSchema(t), inner), after), workflow.WithStateSchema(ss2SchemaW(t)))

		if _, _, err := runStateWorkflow(t, wf, nil); err != nil {
			t.Fatalf("run error = %v, want nil", err)
		}

		if counterErr != nil {
			t.Errorf("after Set(counter) error = %v, want nil", counterErr)
		}
		ss2WantRefused(t, innerKeyErr, ss2Undeclared("inner_key", "WorkflowSchema", "counter", "name"))
	})

	t.Run("state-bound parameter the nested schema does not declare", func(t *testing.T) {
		reader, err := workflow.NewFunctionNodeFromState("reader", func(agent.InvocationContext, ss2ReaderParams) (string, error) {
			return "ok", nil
		}, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("NewFunctionNodeFromState() error = %v", err)
		}

		n, err := workflow.NewWorkflowNodeWithStateSchema("nested", workflow.Chain(workflow.Start, reader), innerSchema(t))

		if n != nil {
			t.Errorf("NewWorkflowNodeWithStateSchema() node = %v, want nil", n)
		}
		const want = `node "reader" references state field "Foo" which is not declared in StateSchema`
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("NewWorkflowNodeWithStateSchema() error = %v, want it to contain %q", err, want)
		}
	})

	t.Run("input and output schemas set on the returned node", func(t *testing.T) {
		in, err := (&jsonschema.Schema{
			Type:       "object",
			Properties: map[string]*jsonschema.Schema{"q": {Type: "string"}},
			Required:   []string{"q"},
		}).Resolve(nil)
		if err != nil {
			t.Fatalf("Resolve(input schema) error = %v", err)
		}
		out, err := (&jsonschema.Schema{Type: "string"}).Resolve(nil)
		if err != nil {
			t.Fatalf("Resolve(output schema) error = %v", err)
		}
		run := func(t *testing.T, input map[string]any) (innerCalls int, err error) {
			inner := ss2Writer("inner", workflow.NodeConfig{}, func(ctx agent.Context) error {
				innerCalls++
				return ctx.State().Set("counter", 2)
			})
			n := nested(t, innerSchema(t), inner)
			n.BaseNode = workflow.NewBaseNodeWithSchemas("nested", "", workflow.NodeConfig{}, in, out)
			seed := workflow.NewFunctionNode("seed", func(agent.Context, any) (map[string]any, error) {
				return input, nil
			}, workflow.NodeConfig{})
			wf := ss2NewWorkflow(t, workflow.Chain(workflow.Start, seed, n), workflow.WithStateSchema(ss2SchemaW(t)))
			_, _, err = runStateWorkflow(t, wf, nil)
			return innerCalls, err
		}

		_, err = run(t, map[string]any{"q": "hi"})
		ss2WantRefused(t, err, ss2Undeclared("counter", "InnerSchema", "inner_key"))

		innerCalls, err := run(t, map[string]any{"q": 5})
		if !errors.Is(err, workflow.ErrInputValidation) {
			t.Errorf("run error for input {q: 5} = %v, want it to wrap ErrInputValidation", err)
		}
		if innerCalls != 0 {
			t.Errorf("inner ran %d times for input {q: 5}, want 0", innerCalls)
		}
	})
}

// ss2ReaderParams binds the state key Foo.
type ss2ReaderParams struct {
	Foo string
}

func TestNode_UnusableStateSchemaIsRejectedAtConstruction(t *testing.T) {
	guarded := func(schema *jsonschema.Resolved) workflow.Node {
		return ss2Writer("guarded", workflow.NodeConfig{StateSchema: schema}, func(agent.Context) error { return nil })
	}
	tests := []struct {
		name  string
		build func(t *testing.T) (built bool, err error)
		want  string
	}{
		{
			name: "graph node",
			build: func(t *testing.T) (bool, error) {
				wf, err := workflow.New("wf", workflow.Chain(workflow.Start, guarded(ss2UnusableSchema(t))), workflow.WithStateSchema(ss2SchemaW(t)))
				return wf != nil, err
			},
			want: `node "guarded"`,
		},
		{
			name: "wrapped node of a parallel worker",
			build: func(t *testing.T) (bool, error) {
				pw, err := workflow.NewParallelWorker("pw", guarded(ss2UnusableSchema(t)), 2, workflow.NodeConfig{})
				return pw != nil, err
			},
			want: "wrapped node guarded",
		},
		{
			name: "nested workflow",
			build: func(t *testing.T) (bool, error) {
				n, err := workflow.NewWorkflowNodeWithStateSchema("nested", workflow.Chain(workflow.Start, guarded(nil)), ss2UnusableSchema(t))
				return n != nil, err
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			built, err := tt.build(t)

			if built {
				t.Errorf("constructor returned a value, want nil")
			}
			if !errors.Is(err, workflow.ErrStateSchemaUnusable) {
				t.Errorf("error = %v, want it to wrap ErrStateSchemaUnusable", err)
			}
			if err != nil && !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %q, want it to contain %q", err.Error(), tt.want)
			}
		})
	}

	t.Run("reference inside the schema's own document", func(t *testing.T) {
		s, err := (&jsonschema.Schema{
			Type: "object",
			Properties: map[string]*jsonschema.Schema{
				"a": {Type: "integer"},
				"b": {Ref: "#/properties/a"},
			},
		}).Resolve(nil)
		if err != nil {
			t.Fatalf("Resolve(internal $ref schema) error = %v", err)
		}

		if _, err := workflow.New("wf", workflow.Chain(workflow.Start, guarded(s))); err != nil {
			t.Errorf("workflow.New() error = %v, want nil", err)
		}
	})
}

func TestRunNode_UnusableChildSchemaFailsChildAtStart(t *testing.T) {
	newChild := func(t *testing.T, calls *atomic.Int32) workflow.Node {
		return workflow.NewFunctionNode("child", func(agent.Context, any) (string, error) {
			calls.Add(1)
			return "c", nil
		}, workflow.NodeConfig{StateSchema: ss2UnusableSchema(t)})
	}

	t.Run("RunNode returns the error", func(t *testing.T) {
		var calls atomic.Int32
		child := newChild(t, &calls)
		var runNodeErr error
		orch := workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			_, runNodeErr = workflow.RunNode[any](ctx, child, nil)
			return "", runNodeErr
		}, workflow.NodeConfig{})

		_, _, err := runStateWorkflow(t, newStateWorkflow(t, orch), nil)

		nre, ok := errors.AsType[*workflow.NodeRunError](runNodeErr)
		if !ok {
			t.Fatalf("RunNode error = %v, want a *workflow.NodeRunError", runNodeErr)
		}
		if nre.ChildName != "child" {
			t.Errorf("NodeRunError.ChildName = %q, want child", nre.ChildName)
		}
		if !errors.Is(runNodeErr, workflow.ErrStateSchemaUnusable) {
			t.Errorf("RunNode error = %v, want it to wrap ErrStateSchemaUnusable", runNodeErr)
		}
		if errors.Is(runNodeErr, workflow.ErrNodeFailed) {
			t.Errorf("RunNode error = %v, want it not to wrap ErrNodeFailed", runNodeErr)
		}
		if _, ok := errors.AsType[*workflow.NodeRunError](err); !ok {
			t.Errorf("run error = %v, want a *workflow.NodeRunError", err)
		}
		if got := calls.Load(); got != 0 {
			t.Errorf("child ran %d times, want 0", got)
		}
	})

	t.Run("caller that ignores the error", func(t *testing.T) {
		var calls atomic.Int32
		child := newChild(t, &calls)
		orch := workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			_, _ = workflow.RunNode[any](ctx, child, nil)
			return "ok", nil
		}, workflow.NodeConfig{})

		events, _, err := runStateWorkflow(t, newStateWorkflow(t, orch), nil)
		if err != nil {
			t.Fatalf("run error = %v, want nil", err)
		}

		for _, ev := range events {
			if strings.HasPrefix(pathOf(ev), "wf@1/orch@1/child") {
				t.Errorf("caller saw an event of the child at %q", pathOf(ev))
			}
		}
		if got := calls.Load(); got != 0 {
			t.Errorf("child ran %d times, want 0", got)
		}
	})

	t.Run("refused child takes no WithUseAsOutput claim", func(t *testing.T) {
		var calls atomic.Int32
		child := newChild(t, &calls)
		second := workflow.NewFunctionNode("second", func(agent.Context, any) (string, error) {
			return "from second", nil
		}, workflow.NodeConfig{})
		var secondErr error
		orch := workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			_, _ = workflow.RunNode[string](ctx, child, nil, workflow.WithUseAsOutput())
			_, secondErr = workflow.RunNode[string](ctx, second, nil, workflow.WithUseAsOutput())
			return "from orch", secondErr
		}, workflow.NodeConfig{})

		events, _, err := runStateWorkflow(t, newStateWorkflow(t, orch), nil)

		if secondErr != nil || err != nil {
			t.Fatalf("second RunNode error = %v, run error = %v, want nil and nil", secondErr, err)
		}
		var outputs []string
		for _, ev := range events {
			if ev.Output != nil {
				outputs = append(outputs, fmt.Sprintf("%s=%v for %v", pathOf(ev), ev.Output, ev.NodeInfo.OutputFor))
			}
		}
		want := []string{"wf@1/orch@1/second@1=from second for [wf@1/orch@1/second@1 wf@1/orch@1 wf@1]"}
		if diff := cmp.Diff(want, outputs); diff != "" {
			t.Errorf("events with Output mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("parallel worker built at run time", func(t *testing.T) {
		var calls atomic.Int32
		child := newChild(t, &calls)
		var pw *workflow.ParallelWorker
		var pwErr error
		orch := workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			pw, pwErr = workflow.NewParallelWorker("pw", child, 2, workflow.NodeConfig{})
			if pwErr != nil {
				return "", pwErr
			}
			_, err := workflow.RunNode[any](ctx, pw, []any{1, 2})
			return "done", err
		}, workflow.NodeConfig{})

		_, _, err := runStateWorkflow(t, newStateWorkflow(t, orch), nil)

		if pw != nil {
			t.Errorf("NewParallelWorker() worker = %v, want nil", pw)
		}
		if !errors.Is(pwErr, workflow.ErrStateSchemaUnusable) {
			t.Errorf("NewParallelWorker() error = %v, want it to wrap ErrStateSchemaUnusable", pwErr)
		}
		if !errors.Is(err, workflow.ErrStateSchemaUnusable) {
			t.Errorf("run error = %v, want it to wrap ErrStateSchemaUnusable", err)
		}
		if got := calls.Load(); got != 0 {
			t.Errorf("child ran %d times, want 0", got)
		}
	})
}

// ss2FlakyNode is a node whose Config carries schema once unstable is set.
type ss2FlakyNode struct {
	workflow.BaseNode
	schema   *jsonschema.Resolved
	unstable atomic.Bool
	calls    atomic.Int32
}

func (n *ss2FlakyNode) Config() workflow.NodeConfig {
	cfg := n.BaseNode.Config()
	if n.unstable.Load() {
		cfg.StateSchema = n.schema
	}
	return cfg
}

func (n *ss2FlakyNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.calls.Add(1)
		yield(outputEvent(ctx, "ok", nil), nil)
	}
}

// A node whose Config starts returning an unusable schema after the workflow
// was built fails when it is activated, before it runs.
func TestNodeStateSchema_UnstableConfigFailsActivation(t *testing.T) {
	tests := []struct {
		name  string
		edges func(t *testing.T, flaky *ss2FlakyNode) []workflow.Edge
	}{
		{
			name: "graph node",
			edges: func(_ *testing.T, flaky *ss2FlakyNode) []workflow.Edge {
				return workflow.Chain(workflow.Start, flaky)
			},
		},
		{
			name: "item of a parallel worker",
			edges: func(t *testing.T, flaky *ss2FlakyNode) []workflow.Edge {
				seed := workflow.NewFunctionNode("seed", func(agent.Context, any) ([]any, error) {
					return []any{0}, nil
				}, workflow.NodeConfig{})
				return workflow.Chain(workflow.Start, seed, newParallelWorker(t, flaky, 0))
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flaky := &ss2FlakyNode{BaseNode: workflow.NewBaseNode("flaky", "", workflow.NodeConfig{}), schema: ss2UnusableSchema(t)}
			wf := ss2NewWorkflow(t, tt.edges(t, flaky))
			flaky.unstable.Store(true)

			_, _, err := runStateWorkflow(t, wf, nil)

			if !errors.Is(err, workflow.ErrStateSchemaUnusable) {
				t.Errorf("run error = %v, want it to wrap ErrStateSchemaUnusable", err)
			}
			if err != nil && (!strings.Contains(err.Error(), `node "flaky"`) || strings.Contains(err.Error(), "panicked")) {
				t.Errorf("run error = %q, want it to contain %q and not %q", err.Error(), `node "flaky"`, "panicked")
			}
			if got := flaky.calls.Load(); got != 0 {
				t.Errorf("flaky ran %d times, want 0", got)
			}
		})
	}
}

// ss2Yield is one value a run yielded.
type ss2Yield struct {
	ev  *session.Event
	err error
}

// ss2RunRootNode runs node as the root of a runner (runner.Config.Node) in a
// new in-memory session with opts and returns what the run yielded, in order,
// and the stored session state after the run.
func ss2RunRootNode(t *testing.T, node workflow.Node, opts ...runner.RunOption) ([]ss2Yield, map[string]any) {
	t.Helper()
	svc := session.InMemoryService()
	created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Node: node, SessionService: svc})
	if err != nil {
		t.Fatalf("runner.New() error = %v, want nil", err)
	}
	var got []ss2Yield
	for ev, err := range r.Run(t.Context(), "u", created.Session.ID(), genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}, opts...) {
		got = append(got, ss2Yield{ev: ev, err: err})
	}
	return got, storedState(t, svc, created.Session.ID())
}

// ss2Describe names each value of got: the author of an event, or "error".
func ss2Describe(got []ss2Yield) []string {
	var out []string
	for _, y := range got {
		if y.err != nil {
			out = append(out, "error")
			continue
		}
		out = append(out, "event by "+y.ev.Author)
	}
	return out
}

// ss2FirstError returns the first error in got, or nil.
func ss2FirstError(got []ss2Yield) error {
	for _, y := range got {
		if y.err != nil {
			return y.err
		}
	}
	return nil
}

func TestNodeStateSchema_BareRootNode(t *testing.T) {
	t.Run("declared key accepted", func(t *testing.T) {
		n := ss2Writer("n", workflow.NodeConfig{StateSchema: ss2SchemaS(t)}, func(ctx agent.Context) error {
			return ctx.State().Set("x", 1)
		})

		got, stored := ss2RunRootNode(t, n)

		if err := ss2FirstError(got); err != nil {
			t.Fatalf("run error = %v, want nil", err)
		}
		if stored["x"] != 1 {
			t.Errorf("stored x = %v, want 1", stored["x"])
		}
	})

	t.Run("undeclared key refused", func(t *testing.T) {
		n := ss2Writer("n", workflow.NodeConfig{StateSchema: ss2SchemaS(t)}, func(ctx agent.Context) error {
			return ctx.State().Set("counter", 2)
		})

		got, stored := ss2RunRootNode(t, n)

		ss2WantRefused(t, ss2FirstError(got), ss2Undeclared("counter", "NodeSchema", "x", "y"))
		if _, ok := stored["counter"]; ok {
			t.Errorf("stored counter = %v, want it absent", stored["counter"])
		}
	})

	t.Run("unusable schema fails the first run after the user message", func(t *testing.T) {
		var calls atomic.Int32
		n := workflow.NewFunctionNode("n", func(agent.Context, any) (string, error) {
			calls.Add(1)
			return "ok", nil
		}, workflow.NodeConfig{StateSchema: ss2UnusableSchema(t)})

		got, _ := ss2RunRootNode(t, n, runner.WithYieldUserMessage())

		if len(got) != 2 || got[0].ev == nil || got[0].ev.Author != "user" || got[1].err == nil {
			t.Fatalf("run yielded %v, want the user message event and then an error", ss2Describe(got))
		}
		err := got[1].err
		if !errors.Is(err, workflow.ErrStateSchemaUnusable) {
			t.Errorf("run error = %v, want it to wrap ErrStateSchemaUnusable", err)
		}
		for _, want := range []string{"failed to build node workflow", `node "n"`} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("run error = %q, want it to contain %q", err.Error(), want)
			}
		}
		if got := calls.Load(); got != 0 {
			t.Errorf("n ran %d times, want 0", got)
		}
	})
}

// Keys with a colon pass a node's own schema, as they pass a workflow's: the
// node's auth gate keeps its state under such keys.
func TestNodeStateSchema_ColonKeysPassNodeSchema(t *testing.T) {
	const stateKey = "adk_oauth_state:wf_auth:wf@1/guarded@1"
	const credentialKey = "adk_oauth_credential:wf_auth:wf@1/guarded@1"

	t.Run("written through State().Set", func(t *testing.T) {
		var colonErr, xErr error
		guarded := ss2Writer("guarded", workflow.NodeConfig{StateSchema: ss2SchemaS(t)}, func(ctx agent.Context) error {
			colonErr = ctx.State().Set(stateKey, "s")
			xErr = ctx.State().Set("x", 1)
			return nil
		})
		wf := newStateWorkflow(t, guarded, workflow.WithStateSchema(ss2SchemaW(t)))

		_, stored, err := runStateWorkflow(t, wf, nil)
		if err != nil {
			t.Fatalf("run error = %v, want nil", err)
		}

		if colonErr != nil || xErr != nil {
			t.Errorf("Set(%s) error = %v and Set(x) error = %v, want both nil", stateKey, colonErr, xErr)
		}
		want := map[string]any{stateKey: "s", "x": 1}
		if diff := cmp.Diff(want, stored); diff != "" {
			t.Errorf("stored state mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("carried by an event's state delta", func(t *testing.T) {
		guarded := ss2Node("guarded", workflow.NodeConfig{StateSchema: ss2SchemaS(t)}, func(ctx agent.Context, yield func(*session.Event, error) bool) {
			yield(outputEvent(ctx, "ok", map[string]any{credentialKey: "v", "x": 1}), nil)
		})
		wf := newStateWorkflow(t, guarded, workflow.WithStateSchema(ss2SchemaW(t)))

		_, stored, err := runStateWorkflow(t, wf, nil)
		if err != nil {
			t.Fatalf("run error = %v, want nil", err)
		}

		want := map[string]any{credentialKey: "v", "x": 1}
		if diff := cmp.Diff(want, stored); diff != "" {
			t.Errorf("stored state mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestNodeConfig_ConstructorsKeepStateSchema(t *testing.T) {
	fn := func(agent.Context, any) (string, error) { return "ok", nil }
	emit := func(agent.Context, any, func(*session.Event) error) (string, error) { return "ok", nil }
	tests := []struct {
		name  string
		build func(t *testing.T, cfg workflow.NodeConfig) (workflow.Node, error)
	}{
		{"NewFunctionNode", func(_ *testing.T, cfg workflow.NodeConfig) (workflow.Node, error) {
			return workflow.NewFunctionNode("n", fn, cfg), nil
		}},
		{"NewFunctionNodeWithSchema", func(_ *testing.T, cfg workflow.NodeConfig) (workflow.Node, error) {
			return workflow.NewFunctionNodeWithSchema("n", fn, nil, nil, cfg)
		}},
		{"NewEmittingFunctionNode", func(_ *testing.T, cfg workflow.NodeConfig) (workflow.Node, error) {
			return workflow.NewEmittingFunctionNode("n", emit, cfg), nil
		}},
		{"NewEmittingFunctionNodeWithSchema", func(_ *testing.T, cfg workflow.NodeConfig) (workflow.Node, error) {
			return workflow.NewEmittingFunctionNodeWithSchema("n", emit, nil, nil, cfg)
		}},
		{"NewFunctionNodeFromState", func(_ *testing.T, cfg workflow.NodeConfig) (workflow.Node, error) {
			return workflow.NewFunctionNodeFromState("n", func(agent.InvocationContext, ss2ReaderParams) (string, error) {
				return "ok", nil
			}, cfg)
		}},
		{"NewDynamicNode", func(_ *testing.T, cfg workflow.NodeConfig) (workflow.Node, error) {
			return workflow.NewDynamicNode("n", emit, cfg), nil
		}},
		{"NewDynamicNodeWithSchema", func(_ *testing.T, cfg workflow.NodeConfig) (workflow.Node, error) {
			return workflow.NewDynamicNodeWithSchema("n", emit, nil, nil, cfg)
		}},
		{"NewAgentNode", func(t *testing.T, cfg workflow.NodeConfig) (workflow.Node, error) {
			a, err := agent.New(agent.Config{Name: "n", Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(func(*session.Event, error) bool) {}
			}})
			if err != nil {
				t.Fatalf("agent.New() error = %v", err)
			}
			return workflow.NewAgentNode(a, cfg)
		}},
		{"NewToolNode", func(t *testing.T, cfg workflow.NodeConfig) (workflow.Node, error) {
			return workflow.NewToolNode(ss2Tool(t), cfg)
		}},
		{"NewNamedToolNode", func(t *testing.T, cfg workflow.NodeConfig) (workflow.Node, error) {
			return workflow.NewNamedToolNode("named", ss2Tool(t), cfg)
		}},
		{"NewParallelWorker", func(_ *testing.T, cfg workflow.NodeConfig) (workflow.Node, error) {
			return workflow.NewParallelWorker("pw", workflow.NewFunctionNode("item", fn, workflow.NodeConfig{}), 0, cfg)
		}},
		{"NewBaseNode", func(_ *testing.T, cfg workflow.NodeConfig) (workflow.Node, error) {
			return &stateNode{BaseNode: workflow.NewBaseNode("n", "", cfg)}, nil
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := ss2SchemaS(t)

			n, err := tt.build(t, workflow.NodeConfig{StateSchema: s})
			if err != nil {
				t.Fatalf("%s() error = %v", tt.name, err)
			}

			if got := n.Config().StateSchema; got != s {
				t.Errorf("Config().StateSchema = %p, want the schema passed in (%p)", got, s)
			}
		})
	}
}

// ss2ToolArgs is the argument of the tool ss2Tool builds.
type ss2ToolArgs struct {
	Q string `json:"q"`
}

// ss2Tool returns a function tool "echo" that returns its argument.
func ss2Tool(t *testing.T) tool.Tool {
	t.Helper()
	echo, err := functiontool.New(functiontool.Config{Name: "echo", Description: "returns its argument"},
		func(_ agent.Context, in ss2ToolArgs) (map[string]any, error) { return map[string]any{"q": in.Q}, nil })
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return echo
}

// The check that a workflow's schema declares each state-bound parameter reads
// the workflow's schema only, not the node's own.
func TestNew_NodeSchemaDoesNotSatisfyWorkflowSchemaCheck(t *testing.T) {
	newReader := func(t *testing.T) workflow.Node {
		reader, err := workflow.NewFunctionNodeFromState("reader", func(agent.InvocationContext, ss2ReaderParams) (string, error) {
			return "ok", nil
		}, workflow.NodeConfig{StateSchema: ss2Schema(t, "N", map[string]string{"Foo": "string"})})
		if err != nil {
			t.Fatalf("NewFunctionNodeFromState() error = %v", err)
		}
		return reader
	}

	t.Run("workflow schema without the parameter", func(t *testing.T) {
		w2 := ss2Schema(t, "W2", map[string]string{"Bar": "string"})

		_, err := workflow.New("wf", workflow.Chain(workflow.Start, newReader(t)), workflow.WithStateSchema(w2))

		const want = `node "reader" references state field "Foo" which is not declared in StateSchema (declared: [Bar])`
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("workflow.New() error = %v, want it to contain %q", err, want)
		}
	})

	t.Run("no workflow schema", func(t *testing.T) {
		if _, err := workflow.New("wf", workflow.Chain(workflow.Start, newReader(t))); err != nil {
			t.Errorf("workflow.New() error = %v, want nil", err)
		}
	})
}

// One resolved schema is shared by workflows and nodes that run at the same
// time; under -race, a derivation that writes into the shared schema, or a
// memo that is not safe for concurrent use, fails this test.
func TestNodeStateSchema_SharedSchemaConcurrentUse(t *testing.T) {
	t.Run("workflows in parallel", func(t *testing.T) {
		s := ss2SchemaS(t)
		w := ss2SchemaW(t)
		const n = 4
		stored := make([]map[string]any, n)
		errs := make([]error, n)
		var wg sync.WaitGroup
		for i := range n {
			// Even workflows also use s as their own schema, so s is
			// derived as a workflow schema and as a node schema at once.
			wfSchema := w
			if i%2 == 0 {
				wfSchema = s
			}
			wg.Go(func() {
				node := ss2Writer("guarded", workflow.NodeConfig{StateSchema: s}, func(ctx agent.Context) error {
					return ctx.State().Set("x", i)
				})
				wf, err := workflow.New("wf", workflow.Chain(workflow.Start, node), workflow.WithStateSchema(wfSchema))
				if err != nil {
					errs[i] = err
					return
				}
				_, stored[i], errs[i] = runStateWorkflow(t, wf, nil)
			})
		}
		wg.Wait()

		for i := range n {
			if errs[i] != nil {
				t.Errorf("workflow %d error = %v, want nil", i, errs[i])
				continue
			}
			if stored[i]["x"] != i {
				t.Errorf("workflow %d stored x = %v, want %d", i, stored[i]["x"], i)
			}
		}
	})

	t.Run("dynamic children in parallel", func(t *testing.T) {
		const n = 8
		props := map[string]string{}
		for i := range n {
			props[fmt.Sprintf("k%d", i)] = "integer"
		}
		p := ss2Schema(t, "SharedSchema", props)
		undeclared := make([]error, n)
		orch := workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			errs := make([]error, n)
			var wg sync.WaitGroup
			for i := range n {
				child := ss2Writer(fmt.Sprintf("child%d", i), workflow.NodeConfig{StateSchema: p}, func(ctx agent.Context) error {
					undeclared[i] = ctx.State().Set("undeclared", i)
					return ctx.State().Set(fmt.Sprintf("k%d", i), i)
				})
				wg.Go(func() {
					_, errs[i] = workflow.RunNode[any](ctx, child, nil)
				})
			}
			wg.Wait()
			return "done", errors.Join(errs...)
		}, workflow.NodeConfig{})

		_, stored, err := runStateWorkflow(t, newStateWorkflow(t, orch), nil)
		if err != nil {
			t.Fatalf("run error = %v, want nil", err)
		}

		want := map[string]any{"k0": 0, "k1": 1, "k2": 2, "k3": 3, "k4": 4, "k5": 5, "k6": 6, "k7": 7}
		if diff := cmp.Diff(want, stored); diff != "" {
			t.Errorf("stored state mismatch (-want +got):\n%s", diff)
		}
		for i, e := range undeclared {
			if se, ok := errors.AsType[*workflow.StateSchemaError](e); !ok || se.Schema != "SharedSchema" {
				t.Errorf("child%d Set(undeclared) error = %v, want a *workflow.StateSchemaError of SharedSchema", i, e)
			}
		}
	})
}
