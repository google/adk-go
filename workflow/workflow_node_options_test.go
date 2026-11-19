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
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// woGauge records how many of its total branch nodes run at once and the
// highest such count. A branch holds its slot until target branches run at
// once or every branch has started, so a slow scheduler cannot hide the
// concurrency the cap allows, then for 60 ms more.
type woGauge struct {
	target, total          int32
	running, started, peak atomic.Int32
}

func newWoGauge(target, total int32) *woGauge { return &woGauge{target: target, total: total} }

// branch returns a node that holds one slot of the gauge and returns its
// name.
func (g *woGauge) branch(name string) workflow.Node {
	return workflow.NewFunctionNode(name, func(agent.Context, any) (string, error) {
		g.started.Add(1)
		n := g.running.Add(1)
		for {
			p := g.peak.Load()
			if n <= p || g.peak.CompareAndSwap(p, n) {
				break
			}
		}
		deadline := time.Now().Add(5 * time.Second)
		for g.running.Load() < g.target && g.started.Load() < g.total && time.Now().Before(deadline) {
			time.Sleep(time.Millisecond)
		}
		time.Sleep(60 * time.Millisecond)
		g.running.Add(-1)
		return name, nil
	}, workflow.NodeConfig{})
}

// woFan returns the edges Start -> a, b, c -> join, with the node names
// prefixed by prefix and every branch measured by g.
func woFan(g *woGauge, prefix string) []workflow.Edge {
	a, b, c := g.branch(prefix+"a"), g.branch(prefix+"b"), g.branch(prefix+"c")
	return workflow.NewEdgeBuilder().AddFanOut(workflow.Start, a, b, c).AddFanIn(workflow.NewJoinNode(prefix+"join"), a, b, c).Build()
}

// woMust returns v, and panics when err is non-nil; the panic fails the
// test that called it.
func woMust[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func woSchema(t *testing.T, title string, keys ...string) *jsonschema.Resolved {
	t.Helper()
	s := &jsonschema.Schema{Title: title, Type: "object", Properties: map[string]*jsonschema.Schema{}}
	for _, k := range keys {
		s.Properties[k] = &jsonschema.Schema{Type: "integer"}
	}
	return woMust(s.Resolve(nil))
}

// woSession is one session of a resumable runner.
type woSession struct {
	r   *runner.Runner
	svc session.Service
	id  string
}

func woNewSession(t *testing.T, cfg runner.Config) *woSession {
	t.Helper()
	svc := session.InMemoryService()
	cfg.AppName, cfg.SessionService, cfg.Resumable = "app", svc, true
	created := woMust(svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"}))
	return &woSession{r: woMust(runner.New(cfg)), svc: svc, id: created.Session.ID()}
}

// send runs one turn and returns the outputs of its node events as
// "path=output", and the first error.
func (s *woSession) send(t *testing.T, msg *genai.Content) ([]string, error) {
	t.Helper()
	var outs []string
	for ev, err := range s.r.Run(t.Context(), "u", s.id, msg, agent.RunConfig{}) {
		if err != nil {
			return outs, err
		}
		if ev.Output != nil && ev.NodeInfo != nil {
			outs = append(outs, fmt.Sprintf("%s=%v", ev.NodeInfo.Path, ev.Output))
		}
	}
	return outs, nil
}

func (s *woSession) state(t *testing.T, key string) any {
	t.Helper()
	got := woMust(s.svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: s.id}))
	v, err := got.Session.State().Get(key)
	if err != nil {
		return nil
	}
	return v
}

func woUser(text string) *genai.Content { return genai.NewContentFromText(text, genai.RoleUser) }

// woAnswer is the user's answer v to the input request with interrupt id id.
func woAnswer(id, v string) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID: id, Name: workflow.WorkflowInputFunctionCallName, Response: map[string]any{"response": v},
	}}}}
}

// woAsk returns a node that requests input with interrupt id "q" and, once
// resumed with the answer, returns what after returns.
func woAsk(name string, after func(agent.Context) (any, error)) workflow.Node {
	return workflow.NewEmittingFunctionNode(name, func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		if _, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "q", Message: "?"}); err != nil {
			return nil, err
		}
		return after(ctx)
	}, workflow.NodeConfig{RerunOnResume: new(true)})
}

func woWriter(name, key string, v any) workflow.Node {
	return workflow.NewFunctionNode(name, func(ctx agent.Context, _ any) (string, error) {
		return name, ctx.State().Set(key, v)
	}, workflow.NodeConfig{})
}

// woDeltaWriter returns a node that writes key through an event's state
// delta.
func woDeltaWriter(name, key string) workflow.Node {
	return workflow.NewEmittingFunctionNode(name, func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Actions.StateDelta = map[string]any{key: 1}
		return nil, emit(ev)
	}, workflow.NodeConfig{})
}

// woPlainRoot runs workflow.New(name, edges, opts...) as agent.New over its
// Run method.
func woPlainRoot(t *testing.T, edges []workflow.Edge, opts ...workflow.Option) runner.Config {
	t.Helper()
	w := woMust(workflow.New("outer", edges, opts...))
	return runner.Config{Agent: woMust(agent.New(agent.Config{Name: "outer", Run: w.Run}))}
}

func woWantRefused(t *testing.T, err error, key, schema string) {
	t.Helper()
	var se *workflow.StateSchemaError
	if !errors.As(err, &se) {
		t.Fatalf("error = %v, want a *workflow.StateSchemaError", err)
	}
	if se.Key != key || se.Schema != schema {
		t.Errorf("StateSchemaError Key, Schema = %q, %q; want %q, %q", se.Key, se.Schema, key, schema)
	}
}

// TestNewWorkflowNodeWithOptions_SharedOptionSlice builds nodes from one option slice with spare capacity on
// several goroutines. The race detector fails it if the constructor appends to the caller's slice instead of
// to a copy.
func TestNewWorkflowNodeWithOptions_SharedOptionSlice(t *testing.T) {
	const goroutines = 8
	opts := make([]workflow.Option, 0, goroutines)
	opts = append(opts, workflow.WithMaxConcurrency(1))
	errs := make([]error, goroutines)
	var wg sync.WaitGroup
	for i := range goroutines {
		wg.Go(func() {
			_, errs[i] = workflow.NewWorkflowNodeWithOptions(fmt.Sprintf("w%d", i), workflow.Chain(workflow.Start, woWriter("a", "k", 1)), opts...)
		})
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Errorf("NewWorkflowNodeWithOptions(w%d) error = %v, want nil", i, err)
		}
	}
	if len(opts) != 1 || cap(opts) != goroutines {
		t.Errorf("caller's opts len, cap = %d, %d; want 1, %d", len(opts), cap(opts), goroutines)
	}
}

func TestNewWorkflowNodeWithOptions_MaxConcurrencyAsRunnerNode(t *testing.T) {
	order := woSchema(t, "Order", "qty")
	tests := []struct {
		name string
		opts []workflow.Option
		want int32
	}{
		{"cap 1", []workflow.Option{workflow.WithMaxConcurrency(1)}, 1},
		{"cap 2", []workflow.Option{workflow.WithMaxConcurrency(2)}, 2},
		{"no option", nil, 3},
		{"negative cap", []workflow.Option{workflow.WithMaxConcurrency(-1)}, 3},
		{"cap after a state schema", []workflow.Option{workflow.WithStateSchema(order), workflow.WithMaxConcurrency(1)}, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			g := newWoGauge(tc.want, 3)
			n := woMust(workflow.NewWorkflowNodeWithOptions("w", woFan(g, ""), tc.opts...))
			if _, err := woNewSession(t, runner.Config{Node: n}).send(t, woUser("go")); err != nil {
				t.Fatalf("run: %v", err)
			}
			if got := g.peak.Load(); got != tc.want {
				t.Errorf("peak = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestNewWorkflowNodeWithOptions_MaxConcurrencyNested(t *testing.T) {
	nested := func(t *testing.T, g *woGauge, prefix string, opts ...workflow.Option) workflow.Node {
		return woMust(workflow.NewWorkflowNodeWithOptions(prefix+"in", woFan(g, prefix), opts...))
	}
	capOf := func(n int) []workflow.Option {
		if n == 0 {
			return nil
		}
		return []workflow.Option{workflow.WithMaxConcurrency(n)}
	}
	tests := []struct {
		name         string
		siblings     bool
		inner, outer int
		want         int32
	}{
		{"inner cap 1, outer none", false, 1, 0, 1},
		{"inner cap 1, outer cap 2", false, 1, 2, 1},
		{"inner cap 2, outer cap 1", false, 2, 1, 2},
		{"inner none, outer cap 1", false, 0, 1, 3},
		{"two siblings capped at 1, outer none", true, 1, 0, 2},
		{"two uncapped siblings, outer cap 1", true, 0, 1, 3},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			g := newWoGauge(tc.want, 3)
			edges := workflow.Chain(workflow.Start, nested(t, g, "", capOf(tc.inner)...))
			if tc.siblings {
				g.total = 6
				x, y := nested(t, g, "x", capOf(tc.inner)...), nested(t, g, "y", capOf(tc.inner)...)
				edges = workflow.NewEdgeBuilder().AddFanOut(workflow.Start, x, y).AddFanIn(workflow.NewJoinNode("j"), x, y).Build()
			}
			if _, err := woNewSession(t, woPlainRoot(t, edges, capOf(tc.outer)...)).send(t, woUser("go")); err != nil {
				t.Fatalf("run: %v", err)
			}
			if got := g.peak.Load(); got != tc.want {
				t.Errorf("peak = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestNewWorkflowNodeWithOptions_StateSchemaAfterResume(t *testing.T) {
	order := woSchema(t, "Order", "qty")
	asker := func(v any) workflow.Node {
		return woAsk("ask", func(ctx agent.Context) (any, error) { return 3, ctx.State().Set("qty", v) })
	}
	placements := []struct {
		name    string
		build   func(t *testing.T, v any) runner.Config
		wantOut string
	}{
		{"runner node", func(t *testing.T, v any) runner.Config {
			return runner.Config{Node: woMust(workflow.NewWorkflowNodeWithOptions("o", workflow.Chain(workflow.Start, asker(v)), workflow.WithStateSchema(order)))}
		}, "o@1=3"},
	}
	for _, pl := range placements {
		t.Run(pl.name+"/refused answer", func(t *testing.T) {
			s := woNewSession(t, pl.build(t, "not a number"))
			if _, err := s.send(t, woUser("go")); err != nil {
				t.Fatalf("turn 1: %v", err)
			}
			_, err := s.send(t, woAnswer("q", "x"))
			woWantRefused(t, err, "qty", "Order")
			if got := s.state(t, "qty"); got != nil {
				t.Errorf("state qty = %v, want absent", got)
			}
		})
		t.Run(pl.name+"/accepted answer", func(t *testing.T) {
			s := woNewSession(t, pl.build(t, 3))
			if _, err := s.send(t, woUser("go")); err != nil {
				t.Fatalf("turn 1: %v", err)
			}
			outs, err := s.send(t, woAnswer("q", "x"))
			if err != nil {
				t.Fatalf("turn 2: %v", err)
			}
			if !slices.Contains(outs, pl.wantOut) {
				t.Errorf("turn 2 outputs = %v, want one %q", outs, pl.wantOut)
			}
			if got := s.state(t, "qty"); got != 3 {
				t.Errorf("state qty = %v, want 3", got)
			}
		})
	}
}

func TestNewWorkflowNodeWithOptions_NestedSchemaScope(t *testing.T) {
	parent := woSchema(t, "Parent", "p", "shared")
	inner := woSchema(t, "Inner", "i", "shared")
	withSchema := func(s *jsonschema.Resolved) []workflow.Option { return []workflow.Option{workflow.WithStateSchema(s)} }
	roots := []struct {
		name  string
		build func(t *testing.T, edges []workflow.Edge, outer []workflow.Option) runner.Config
	}{
		{"agent.New over Run", func(t *testing.T, edges []workflow.Edge, outer []workflow.Option) runner.Config {
			return woPlainRoot(t, edges, outer...)
		}},
		{"runner node", func(t *testing.T, edges []workflow.Edge, outer []workflow.Option) runner.Config {
			return runner.Config{Node: woMust(workflow.NewWorkflowNodeWithOptions("outer", edges, outer...))}
		}},
	}
	tests := []struct {
		name        string
		outer       []workflow.Option
		inner       []workflow.Option
		node        func() workflow.Node
		wantKey     string
		wantRefusal string
	}{
		{"own schema refuses a parent key", withSchema(parent), withSchema(inner), func() workflow.Node { return woWriter("w", "p", 1) }, "p", "Inner"},
		{"own schema accepts its key", withSchema(parent), withSchema(inner), func() workflow.Node { return woWriter("w", "i", 1) }, "", ""},
		{"own schema refuses an event delta", withSchema(parent), withSchema(inner), func() workflow.Node { return woDeltaWriter("w", "p") }, "p", "Inner"},
		{"no schema inherits the parent's", withSchema(parent), []workflow.Option{workflow.WithMaxConcurrency(1)}, func() workflow.Node { return woWriter("w", "i", 1) }, "i", "Parent"},
		{"no schema accepts a parent key", withSchema(parent), []workflow.Option{workflow.WithMaxConcurrency(1)}, func() workflow.Node { return woWriter("w", "p", 1) }, "", ""},
		{"own schema under a schema-less parent", nil, withSchema(inner), func() workflow.Node { return woWriter("w", "zz", 1) }, "zz", "Inner"},
		{"colon key passes", nil, withSchema(inner), func() workflow.Node { return woWriter("w", "app:x", 1) }, "", ""},
	}
	for _, root := range roots {
		for _, tc := range tests {
			t.Run(root.name+"/"+tc.name, func(t *testing.T) {
				in := woMust(workflow.NewWorkflowNodeWithOptions("in", workflow.Chain(workflow.Start, tc.node()), tc.inner...))
				_, err := woNewSession(t, root.build(t, workflow.Chain(workflow.Start, in), tc.outer)).send(t, woUser("go"))
				if tc.wantRefusal == "" {
					if err != nil {
						t.Errorf("run error = %v, want nil", err)
					}
					return
				}
				woWantRefused(t, err, tc.wantKey, tc.wantRefusal)
			})
		}
		t.Run(root.name+"/parent node after the nested workflow", func(t *testing.T) {
			in := woMust(workflow.NewWorkflowNodeWithOptions("in", workflow.Chain(workflow.Start, woWriter("w1", "i", 1)), workflow.WithStateSchema(inner)))
			s := woNewSession(t, root.build(t, workflow.Chain(workflow.Start, in, woWriter("after", "i", 2)), withSchema(parent)))
			_, err := s.send(t, woUser("go"))
			woWantRefused(t, err, "i", "Parent")
			if got := s.state(t, "i"); got != 1 {
				t.Errorf("state i = %v, want 1", got)
			}
		})
	}
}

func TestNewWorkflowNodeWithOptions_RunnerOnlyOptionsRefused(t *testing.T) {
	tests := []struct {
		name string
		opts []workflow.Option
	}{
		{"WithRootWrapper", []workflow.Option{workflow.WithRootWrapper()}},
		{"WithNodeRootWrapper", []workflow.Option{workflow.WithNodeRootWrapper()}},
		{"WithRootWrapper after a cap", []workflow.Option{workflow.WithMaxConcurrency(1), workflow.WithRootWrapper()}},
		{"WithNodeRootWrapper after a cap", []workflow.Option{workflow.WithMaxConcurrency(1), workflow.WithNodeRootWrapper()}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			n, err := workflow.NewWorkflowNodeWithOptions("nodeX", workflow.Chain(workflow.Start, woWriter("w", "p", 1)), tc.opts...)
			if n != nil || err == nil {
				t.Fatalf("NewWorkflowNodeWithOptions() = %v, %v; want nil node and an error", n, err)
			}
			if !strings.Contains(err.Error(), `"nodeX"`) || !strings.Contains(err.Error(), "runner") {
				t.Errorf("error = %q, want it to name \"nodeX\" and the runner", err)
			}
			if errors.Is(err, workflow.ErrGraphValidation) {
				t.Errorf("error = %v matches ErrGraphValidation, want it not to", err)
			}
		})
	}
}

type woNeedsK struct {
	K int `state:"k"`
}

func TestNewWorkflowNodeWithOptions_ConstructionErrors(t *testing.T) {
	needsK := func(t *testing.T) workflow.Node {
		return woMust(workflow.NewFunctionNodeFromState("needs_k", func(agent.InvocationContext, woNeedsK) (string, error) { return "", nil }, workflow.NodeConfig{}))
	}
	t.Run("unusable state schema", func(t *testing.T) {
		n, err := workflow.NewWorkflowNodeWithOptions("w", workflow.Chain(workflow.Start, woWriter("w1", "a", 1)), workflow.WithStateSchema(ss2UnusableSchema(t)))
		if n != nil || !errors.Is(err, workflow.ErrStateSchemaUnusable) {
			t.Errorf("NewWorkflowNodeWithOptions() = %v, %v; want nil and ErrStateSchemaUnusable", n, err)
		}
	})
	t.Run("state-bound parameter the schema does not declare", func(t *testing.T) {
		n, err := workflow.NewWorkflowNodeWithOptions("w", workflow.Chain(workflow.Start, needsK(t)), workflow.WithStateSchema(woSchema(t, "Inner", "i", "shared")))
		var se *workflow.StateSchemaError
		if n != nil || !errors.As(err, &se) || se.Node != "needs_k" || se.Key != "k" {
			t.Errorf("NewWorkflowNodeWithOptions() = %v, %v; want nil and a StateSchemaError for node needs_k, key k", n, err)
		}
	})
	t.Run("no Start edge", func(t *testing.T) {
		n, err := workflow.NewWorkflowNodeWithOptions("w", workflow.Chain(woWriter("a", "p", 1), woWriter("b", "p", 1)), workflow.WithMaxConcurrency(1))
		if n != nil || !errors.Is(err, workflow.ErrGraphValidation) {
			t.Errorf("NewWorkflowNodeWithOptions() = %v, %v; want nil and ErrGraphValidation", n, err)
		}
	})
	t.Run("nil edges and no options", func(t *testing.T) {
		n, err := workflow.NewWorkflowNodeWithOptions("w", nil)
		if n == nil || err != nil {
			t.Errorf("NewWorkflowNodeWithOptions() = %v, %v; want a node and no error", n, err)
		}
	})
}

func TestNewWorkflowNodeWithOptions_WorkflowLevelSchemas(t *testing.T) {
	in := woMust((&jsonschema.Schema{Type: "object", Required: []string{"user"}, Properties: map[string]*jsonschema.Schema{
		"user": {Type: "string"}, "role": {Type: "string", Default: []byte(`"guest"`)},
	}}).Resolve(nil))
	out := woMust((&jsonschema.Schema{Type: "object", Required: []string{"name"}, Properties: map[string]*jsonschema.Schema{
		"name": {Type: "string"}, "tag": {Type: "string", Default: []byte(`"ok"`)},
	}}).Resolve(nil))
	// recorder returns a node named name that records the inputs it receives
	// and returns result.
	type recorder struct {
		mu   sync.Mutex
		seen []string
	}
	record := func(r *recorder, name string, result any) workflow.Node {
		return workflow.NewFunctionNode(name, func(_ agent.Context, v any) (any, error) {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.seen = append(r.seen, fmt.Sprint(v))
			return result, nil
		}, workflow.NodeConfig{})
	}
	gen := func(v any) workflow.Node {
		return workflow.NewFunctionNode("gen", func(agent.Context, any) (any, error) { return v, nil }, workflow.NodeConfig{})
	}
	underAgent := func(t *testing.T, edges []workflow.Edge) *woSession {
		return woNewSession(t, runner.Config{Agent: woMust(workflowagent.New(workflowagent.Config{Name: "o", Edges: edges}))})
	}

	t.Run("input refused", func(t *testing.T) {
		var body recorder
		n := woMust(workflow.NewWorkflowNodeWithOptions("wf", workflow.Chain(workflow.Start, record(&body, "body", map[string]any{"name": "x"})), workflow.WithInputSchema(in)))
		_, err := woNewSession(t, runner.Config{Node: n}).send(t, woUser(`{"wrong":1}`))
		if !errors.Is(err, workflow.ErrInputValidation) || !strings.Contains(err.Error(), `"wf"`) {
			t.Errorf("run error = %v, want ErrInputValidation naming \"wf\"", err)
		}
		if len(body.seen) != 0 {
			t.Errorf("inner node ran %d times, want 0", len(body.seen))
		}
	})
	t.Run("input defaults filled", func(t *testing.T) {
		var body recorder
		n := woMust(workflow.NewWorkflowNodeWithOptions("wf", workflow.Chain(workflow.Start, record(&body, "body", map[string]any{"name": "x"})), workflow.WithInputSchema(in)))
		if _, err := woNewSession(t, runner.Config{Node: n}).send(t, woUser(`{"user":"u1"}`)); err != nil {
			t.Fatalf("run: %v", err)
		}
		if want := []string{"map[role:guest user:u1]"}; !slices.Equal(body.seen, want) {
			t.Errorf("inner node received %v, want %v", body.seen, want)
		}
	})
	t.Run("output refused", func(t *testing.T) {
		var body recorder
		n := woMust(workflow.NewWorkflowNodeWithOptions("wf", workflow.Chain(workflow.Start, record(&body, "body", map[string]any{"bad": 1})), workflow.WithOutputSchema(out)))
		_, err := woNewSession(t, runner.Config{Node: n}).send(t, woUser("go"))
		if !errors.Is(err, workflow.ErrNodeFailed) {
			t.Errorf("run error = %v, want ErrNodeFailed", err)
		}
	})
	t.Run("next node receives the coerced output", func(t *testing.T) {
		var body, next recorder
		n := woMust(workflow.NewWorkflowNodeWithOptions("wf", workflow.Chain(workflow.Start, record(&body, "term", map[string]any{"name": "x"})), workflow.WithOutputSchema(out)))
		if _, err := underAgent(t, workflow.Chain(workflow.Start, n, record(&next, "next", "done"))).send(t, woUser("go")); err != nil {
			t.Fatalf("run: %v", err)
		}
		if want := []string{"map[name:x tag:ok]"}; !slices.Equal(next.seen, want) {
			t.Errorf("next node received %v, want %v", next.seen, want)
		}
	})
	t.Run("both options in one call", func(t *testing.T) {
		var body, next recorder
		n := woMust(workflow.NewWorkflowNodeWithOptions("wf", workflow.Chain(workflow.Start, record(&body, "body", map[string]any{"name": "x"})),
			workflow.WithInputSchema(in), workflow.WithOutputSchema(out)))
		edges := workflow.Chain(workflow.Start, gen(map[string]any{"user": "u1"}), n, record(&next, "next", "done"))
		if _, err := underAgent(t, edges).send(t, woUser("go")); err != nil {
			t.Fatalf("run: %v", err)
		}
		if want := []string{"map[role:guest user:u1]"}; !slices.Equal(body.seen, want) {
			t.Errorf("inner node received %v, want %v", body.seen, want)
		}
		if want := []string{"map[name:x tag:ok]"}; !slices.Equal(next.seen, want) {
			t.Errorf("next node received %v, want %v", next.seen, want)
		}
	})
	t.Run("both options in one call, input refused", func(t *testing.T) {
		var body recorder
		n := woMust(workflow.NewWorkflowNodeWithOptions("wf", workflow.Chain(workflow.Start, record(&body, "body", map[string]any{"name": "x"})),
			workflow.WithInputSchema(in), workflow.WithOutputSchema(out)))
		_, err := underAgent(t, workflow.Chain(workflow.Start, gen(map[string]any{"wrong": 1}), n)).send(t, woUser("go"))
		if !errors.Is(err, workflow.ErrInputValidation) {
			t.Errorf("run error = %v, want ErrInputValidation", err)
		}
		if len(body.seen) != 0 {
			t.Errorf("inner node ran %d times, want 0", len(body.seen))
		}
	})
}

func TestRunNode_WorkflowNodeWithOptions(t *testing.T) {
	parent := woSchema(t, "Parent", "p", "shared")
	inner := woSchema(t, "Inner", "i", "shared")
	orchestrate := func(child workflow.Node) []workflow.Edge {
		orch := workflow.NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
			if _, err := workflow.RunNode[any](ctx, child, nil); err != nil {
				return "", err
			}
			return "done", nil
		}, workflow.NodeConfig{})
		return workflow.Chain(workflow.Start, orch)
	}
	t.Run("cap", func(t *testing.T) {
		g := newWoGauge(1, 3)
		child := woMust(workflow.NewWorkflowNodeWithOptions("child", woFan(g, ""), workflow.WithMaxConcurrency(1)))
		if _, err := woNewSession(t, woPlainRoot(t, orchestrate(child))).send(t, woUser("go")); err != nil {
			t.Fatalf("run: %v", err)
		}
		if got := g.peak.Load(); got != 1 {
			t.Errorf("peak = %d, want 1", got)
		}
	})
	t.Run("own schema refuses a parent key", func(t *testing.T) {
		child := woMust(workflow.NewWorkflowNodeWithOptions("child", workflow.Chain(workflow.Start, woWriter("w", "p", 1)), workflow.WithStateSchema(inner)))
		s := woNewSession(t, woPlainRoot(t, orchestrate(child), workflow.WithStateSchema(parent)))
		_, err := s.send(t, woUser("go"))
		woWantRefused(t, err, "p", "Inner")
		if got := s.state(t, "p"); got != nil {
			t.Errorf("state p = %v, want absent", got)
		}
	})
	t.Run("own schema accepts its key", func(t *testing.T) {
		child := woMust(workflow.NewWorkflowNodeWithOptions("child", workflow.Chain(workflow.Start, woWriter("w", "i", 1)), workflow.WithStateSchema(inner)))
		s := woNewSession(t, woPlainRoot(t, orchestrate(child), workflow.WithStateSchema(parent)))
		if _, err := s.send(t, woUser("go")); err != nil {
			t.Fatalf("run: %v", err)
		}
		if got := s.state(t, "i"); got != 1 {
			t.Errorf("state i = %v, want 1", got)
		}
	})
}
