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

package workflowagent_test

import (
	"errors"
	"fmt"
	"net/url"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// optGauge records how many of its total branch nodes run at once and the
// highest such count. A branch holds its slot until target branches run at
// once or every branch has started, so a slow scheduler cannot hide the
// concurrency the cap allows, then for 60 ms more.
type optGauge struct {
	target, total          int32
	running, started, peak atomic.Int32
}

func newOptGauge(target, total int32) *optGauge { return &optGauge{target: target, total: total} }

func (g *optGauge) enter() {
	g.started.Add(1)
	n := g.running.Add(1)
	for {
		p := g.peak.Load()
		if n <= p || g.peak.CompareAndSwap(p, n) {
			return
		}
	}
}

// branch returns a node that holds one slot of g and returns its name.
func (g *optGauge) branch(name string) workflow.Node {
	return workflow.NewFunctionNode(name, func(agent.Context, any) (string, error) {
		g.enter()
		deadline := time.Now().Add(5 * time.Second)
		for g.running.Load() < g.target && g.started.Load() < g.total && time.Now().Before(deadline) {
			time.Sleep(time.Millisecond)
		}
		time.Sleep(60 * time.Millisecond)
		g.running.Add(-1)
		return name, nil
	}, workflow.NodeConfig{})
}

// optFan returns the edges from -> a, b, c -> join, every branch measured by
// g.
func optFan(g *optGauge, from workflow.Node) *workflow.EdgeBuilder {
	a, b, c := g.branch("a"), g.branch("b"), g.branch("c")
	return workflow.NewEdgeBuilder().AddFanOut(from, a, b, c).AddFanIn(workflow.NewJoinNode("join"), a, b, c)
}

// optMust returns v, and panics when err is non-nil; the panic fails the
// test that called it.
func optMust[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func optSchema(t *testing.T, title string, keys ...string) *jsonschema.Resolved {
	t.Helper()
	s := &jsonschema.Schema{Title: title, Type: "object", Properties: map[string]*jsonschema.Schema{}}
	for _, k := range keys {
		s.Properties[k] = &jsonschema.Schema{Type: "integer"}
	}
	return optMust(s.Resolve(nil))
}

// optRunner is a resumable runner over one in-memory session service.
type optRunner struct {
	r   *runner.Runner
	svc session.Service
}

func newOptRunner(t *testing.T, a agent.Agent) *optRunner {
	t.Helper()
	svc := session.InMemoryService()
	return &optRunner{r: optMust(runner.New(runner.Config{AppName: "app", Agent: a, SessionService: svc, Resumable: true})), svc: svc}
}

func (r *optRunner) newSession(t *testing.T) string {
	t.Helper()
	return optMust(r.svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})).Session.ID()
}

// send runs one turn of session sid and returns the outputs of its node
// events as "path=output", and the first error.
func (r *optRunner) send(t *testing.T, sid string, msg *genai.Content) ([]string, error) {
	t.Helper()
	var outs []string
	for ev, err := range r.r.Run(t.Context(), "u", sid, msg, agent.RunConfig{}) {
		if err != nil {
			return outs, err
		}
		if ev.Output != nil && ev.NodeInfo != nil {
			outs = append(outs, fmt.Sprintf("%s=%v", ev.NodeInfo.Path, ev.Output))
		}
	}
	return outs, nil
}

func (r *optRunner) state(t *testing.T, sid, key string) any {
	t.Helper()
	got := optMust(r.svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: sid}))
	v, err := got.Session.State().Get(key)
	if err != nil {
		return nil
	}
	return v
}

func optUser(text string) *genai.Content { return genai.NewContentFromText(text, genai.RoleUser) }

// optAnswer is the user's answer v to the input request with interrupt id
// "q".
func optAnswer(v string) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID: "q", Name: workflow.WorkflowInputFunctionCallName, Response: map[string]any{"response": v},
	}}}}
}

// optAsk returns a node that requests input with interrupt id "q" and, once
// resumed with the answer, returns what after returns.
func optAsk(after func(ctx agent.Context, answer any) (any, error)) workflow.Node {
	return workflow.NewEmittingFunctionNode("ask", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		answer, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "q", Message: "?"})
		if err != nil {
			return nil, err
		}
		return after(ctx, answer)
	}, workflow.NodeConfig{RerunOnResume: new(true)})
}

func optWriter(name, key string, v any) workflow.Node {
	return workflow.NewFunctionNode(name, func(ctx agent.Context, _ any) (string, error) {
		return name, ctx.State().Set(key, v)
	}, workflow.NodeConfig{})
}

func optWantRefused(t *testing.T, err error, key, schema string) {
	t.Helper()
	var se *workflow.StateSchemaError
	if !errors.As(err, &se) {
		t.Fatalf("error = %v, want a *workflow.StateSchemaError", err)
	}
	if se.Key != key || se.Schema != schema {
		t.Errorf("StateSchemaError Key, Schema = %q, %q; want %q, %q", se.Key, se.Schema, key, schema)
	}
}

func TestConfig_MaxConcurrency(t *testing.T) {
	for _, tc := range []struct {
		maxConcurrency int
		want           int32
	}{{1, 1}, {2, 2}, {0, 3}, {-1, 3}, {10, 3}} {
		t.Run(fmt.Sprint(tc.maxConcurrency), func(t *testing.T) {
			g := newOptGauge(tc.want, 3)
			a := optMust(workflowagent.New(workflowagent.Config{Name: "w", Edges: optFan(g, workflow.Start).Build(), MaxConcurrency: tc.maxConcurrency}))
			r := newOptRunner(t, a)
			if _, err := r.send(t, r.newSession(t), optUser("go")); err != nil {
				t.Fatalf("run: %v", err)
			}
			if got := g.peak.Load(); got != tc.want {
				t.Errorf("peak = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestConfig_MaxConcurrency_AfterResume(t *testing.T) {
	for _, tc := range []struct {
		maxConcurrency int
		want           int32
	}{{1, 1}, {0, 3}} {
		t.Run(fmt.Sprint(tc.maxConcurrency), func(t *testing.T) {
			g := newOptGauge(tc.want, 3)
			ask := optAsk(func(agent.Context, any) (any, error) { return "go", nil })
			edges := optFan(g, ask).Add(workflow.Start, ask).Build()
			r := newOptRunner(t, optMust(workflowagent.New(workflowagent.Config{Name: "w", Edges: edges, MaxConcurrency: tc.maxConcurrency})))
			sid := r.newSession(t)
			if _, err := r.send(t, sid, optUser("go")); err != nil {
				t.Fatalf("turn 1: %v", err)
			}
			if got := g.peak.Load(); got != 0 {
				t.Fatalf("turn 1 peak = %d, want 0", got)
			}
			if _, err := r.send(t, sid, optAnswer("yes")); err != nil {
				t.Fatalf("turn 2: %v", err)
			}
			if got := g.peak.Load(); got != tc.want {
				t.Errorf("peak = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestConfig_StateSchema_AfterResume(t *testing.T) {
	order := optSchema(t, "Order", "qty")
	build := func(t *testing.T, v any) *optRunner {
		ask := optAsk(func(ctx agent.Context, _ any) (any, error) { return 3, ctx.State().Set("qty", v) })
		return newOptRunner(t, optMust(workflowagent.New(workflowagent.Config{Name: "o", Edges: workflow.Chain(workflow.Start, ask), StateSchema: order})))
	}
	t.Run("refused answer", func(t *testing.T) {
		r := build(t, "not a number")
		sid := r.newSession(t)
		if _, err := r.send(t, sid, optUser("go")); err != nil {
			t.Fatalf("turn 1: %v", err)
		}
		_, err := r.send(t, sid, optAnswer("x"))
		optWantRefused(t, err, "qty", "Order")
		if got := r.state(t, sid, "qty"); got != nil {
			t.Errorf("state qty = %v, want absent", got)
		}
	})
}

type optNeedsK struct {
	K int `state:"k"`
}

func TestNew_StateSchemaErrors(t *testing.T) {
	needsK := func() workflow.Node {
		return optMust(workflow.NewFunctionNodeFromState("needs_k", func(agent.InvocationContext, optNeedsK) (string, error) { return "", nil }, workflow.NodeConfig{}))
	}
	t.Run("unusable schema", func(t *testing.T) {
		s := &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{"a": {Ref: "https://example.invalid/a.json"}}}
		unusable := optMust(s.Resolve(&jsonschema.ResolveOptions{Loader: func(*url.URL) (*jsonschema.Schema, error) {
			return &jsonschema.Schema{Type: "integer"}, nil
		}}))
		a, err := workflowagent.New(workflowagent.Config{Name: "w", Edges: workflow.Chain(workflow.Start, optWriter("w1", "a", 1)), StateSchema: unusable})
		if a != nil || !errors.Is(err, workflow.ErrStateSchemaUnusable) {
			t.Errorf("New() = %v, %v; want nil and ErrStateSchemaUnusable", a, err)
		}
	})
	t.Run("state-bound parameter the schema does not declare", func(t *testing.T) {
		a, err := workflowagent.New(workflowagent.Config{Name: "w", Edges: workflow.Chain(workflow.Start, needsK()), StateSchema: optSchema(t, "Inner", "i", "shared")})
		if a != nil {
			t.Errorf("New() agent = %v, want nil", a)
		}
		var se *workflow.StateSchemaError
		if !errors.As(err, &se) {
			t.Fatalf("New() error = %v, want a *workflow.StateSchemaError", err)
		}
		want := &workflow.StateSchemaError{Node: "needs_k", Key: "k", Schema: "Inner", Reason: workflow.StateBindingUndeclared, Declared: []string{"i", "shared"}}
		if diff := cmp.Diff(want, se); diff != "" {
			t.Errorf("StateSchemaError mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("no state schema", func(t *testing.T) {
		if _, err := workflowagent.New(workflowagent.Config{Name: "w", Edges: workflow.Chain(workflow.Start, needsK())}); err != nil {
			t.Errorf("New() error = %v, want nil", err)
		}
	})
}

func TestConfig_StateSchema_AgentNodeOwnSchemaReplacesParent(t *testing.T) {
	parent := optSchema(t, "Parent", "p", "shared")
	inner := optSchema(t, "Inner", "i", "shared")
	build := func(t *testing.T, key string) (*optRunner, string) {
		wa := optMust(workflowagent.New(workflowagent.Config{Name: "child", Edges: workflow.Chain(workflow.Start, optWriter("w", key, 1)), StateSchema: inner}))
		an := optMust(workflow.NewAgentNode(wa, workflow.NodeConfig{}))
		root := optMust(workflowagent.New(workflowagent.Config{Name: "root", Edges: workflow.Chain(workflow.Start, an), StateSchema: parent}))
		r := newOptRunner(t, root)
		return r, r.newSession(t)
	}
	t.Run("parent key refused", func(t *testing.T) {
		r, sid := build(t, "p")
		_, err := r.send(t, sid, optUser("go"))
		optWantRefused(t, err, "p", "Inner")
		if got := r.state(t, sid, "p"); got != nil {
			t.Errorf("state p = %v, want absent", got)
		}
	})
	t.Run("own key stored", func(t *testing.T) {
		r, sid := build(t, "i")
		if _, err := r.send(t, sid, optUser("go")); err != nil {
			t.Fatalf("run: %v", err)
		}
		if got := r.state(t, sid, "i"); got != 1 {
			t.Errorf("state i = %v, want 1", got)
		}
	})
}

func TestConfig_MaxConcurrency_AsAgentNode(t *testing.T) {
	for _, tc := range []struct {
		name         string
		inner, outer int
		want         int32
	}{
		{"inner cap 1, outer cap 2", 1, 2, 1},
		{"inner none, outer cap 1", 0, 1, 3},
		{"inner cap 2, outer cap 1", 2, 1, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := newOptGauge(tc.want, 3)
			wa := optMust(workflowagent.New(workflowagent.Config{Name: "child", Edges: optFan(g, workflow.Start).Build(), MaxConcurrency: tc.inner}))
			an := optMust(workflow.NewAgentNode(wa, workflow.NodeConfig{}))
			root := optMust(workflowagent.New(workflowagent.Config{Name: "root", Edges: workflow.Chain(workflow.Start, an), MaxConcurrency: tc.outer}))
			r := newOptRunner(t, root)
			if _, err := r.send(t, r.newSession(t), optUser("go")); err != nil {
				t.Fatalf("run: %v", err)
			}
			if got := g.peak.Load(); got != tc.want {
				t.Errorf("peak = %d, want %d", got, tc.want)
			}
		})
	}
}

// One agent with both fields set serves four sessions at once; each session
// keeps its own cap and schema check.
func TestConfig_ConcurrentRunsOfOneAgent(t *testing.T) {
	var mu sync.Mutex
	running, peak := map[string]int{}, map[string]int{}
	branch := func(name string) workflow.Node {
		return workflow.NewFunctionNode(name, func(ctx agent.Context, _ any) (string, error) {
			sid := ctx.Session().ID()
			mu.Lock()
			running[sid]++
			peak[sid] = max(peak[sid], running[sid])
			mu.Unlock()
			time.Sleep(30 * time.Millisecond)
			mu.Lock()
			running[sid]--
			mu.Unlock()
			return name, nil
		}, workflow.NodeConfig{})
	}
	ask := optAsk(func(ctx agent.Context, answer any) (any, error) {
		if answer == "good" {
			return "ok", ctx.State().Set("qty", 3)
		}
		return "ok", ctx.State().Set("qty", "not a number")
	})
	a, b, c := branch("a"), branch("b"), branch("c")
	edges := workflow.NewEdgeBuilder().Add(workflow.Start, ask).AddFanOut(ask, a, b, c).AddFanIn(workflow.NewJoinNode("join"), a, b, c).Build()
	r := newOptRunner(t, optMust(workflowagent.New(workflowagent.Config{Name: "w", Edges: edges, MaxConcurrency: 1, StateSchema: optSchema(t, "Order", "qty")})))

	answers := []string{"good", "bad", "good", "bad"}
	sids := make([]string, len(answers))
	errs := make([]error, len(answers))
	var wg sync.WaitGroup
	for i, answer := range answers {
		sids[i] = r.newSession(t)
		wg.Go(func() {
			if _, err := r.send(t, sids[i], optUser("go")); err != nil {
				errs[i] = fmt.Errorf("turn 1: %w", err)
				return
			}
			_, errs[i] = r.send(t, sids[i], optAnswer(answer))
		})
	}
	wg.Wait()

	for i, answer := range answers {
		if answer == "good" {
			if errs[i] != nil {
				t.Errorf("session %d: error = %v, want nil", i, errs[i])
			}
			if got := r.state(t, sids[i], "qty"); got != 3 {
				t.Errorf("session %d: state qty = %v, want 3", i, got)
			}
			if got := peak[sids[i]]; got != 1 {
				t.Errorf("session %d: peak = %d, want 1", i, got)
			}
			continue
		}
		optWantRefused(t, errs[i], "qty", "Order")
		if got := r.state(t, sids[i], "qty"); got != nil {
			t.Errorf("session %d: state qty = %v, want absent", i, got)
		}
	}
}
