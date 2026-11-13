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

package runner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"runtime/pprof"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/workflow"
)

// closeLog records names in the order their Close ran. Toolsets close in a
// goroutine of their own, so it is guarded by a mutex.
type closeLog struct {
	mu    sync.Mutex
	names []string
}

func (l *closeLog) add(name string) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.names = append(l.names, name)
}

func (l *closeLog) get() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.names...)
}

// countingToolset counts its Close calls and, when log is set, records its
// name there.
type countingToolset struct {
	name   string
	log    *closeLog
	closes atomic.Int32
}

func (c *countingToolset) Name() string                                     { return c.name }
func (c *countingToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return nil, nil }

func (c *countingToolset) Close() error {
	c.closes.Add(1)
	c.log.add(c.name)
	return nil
}

// scriptedCloseToolset runs close as its Close.
type scriptedCloseToolset struct {
	name  string
	close func() error
}

func (s *scriptedCloseToolset) Name() string                                     { return s.name }
func (s *scriptedCloseToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return nil, nil }
func (s *scriptedCloseToolset) Close() error                                     { return s.close() }

// unwrapOnlyToolset wraps inner with Unwrap and no Close.
type unwrapOnlyToolset struct{ inner tool.Toolset }

func (u *unwrapOnlyToolset) Name() string { return "unwrap_only" }
func (u *unwrapOnlyToolset) Tools(c agent.ReadonlyContext) ([]tool.Tool, error) {
	return u.inner.Tools(c)
}
func (u *unwrapOnlyToolset) Unwrap() tool.Toolset { return u.inner }

// agentWrapperTool runs nothing and reports agent as the agent it wraps, as
// agenttool's tool does; this package's tests cannot import agenttool, which
// imports runner. It has a Run method so that workflow.NewToolNode accepts it.
type agentWrapperTool struct{ agent agent.Agent }

func (w *agentWrapperTool) Name() string              { return w.agent.Name() }
func (w *agentWrapperTool) Description() string       { return "" }
func (w *agentWrapperTool) IsLongRunning() bool       { return false }
func (w *agentWrapperTool) WrappedAgent() agent.Agent { return w.agent }

func (w *agentWrapperTool) Run(agent.Context, any) (map[string]any, error) { return nil, nil }

func newCloseTestLLM(t *testing.T, name string, toolsets []tool.Toolset, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{Name: name, Toolsets: toolsets, SubAgents: subAgents})
	if err != nil {
		t.Fatalf("llmagent.New(%q) error = %v", name, err)
	}
	return a
}

func newCloseTestAgentNode(t *testing.T, a agent.Agent) *workflow.AgentNode {
	t.Helper()
	n, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode(%q) error = %v", a.Name(), err)
	}
	return n
}

func newCloseTestWorkflowAgent(t *testing.T, name string, edges []workflow.Edge, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := workflowagent.New(workflowagent.Config{Name: name, Edges: edges, SubAgents: subAgents})
	if err != nil {
		t.Fatalf("workflowagent.New(%q) error = %v", name, err)
	}
	return a
}

func newCloseTestWorkflowNode(t *testing.T, name string, edges []workflow.Edge) *workflow.WorkflowNode {
	t.Helper()
	n, err := workflow.NewWorkflowNode(name, edges)
	if err != nil {
		t.Fatalf("workflow.NewWorkflowNode(%q) error = %v", name, err)
	}
	return n
}

func newCloseTestToolNode(t *testing.T, tl tool.Tool) *workflow.ToolNode {
	t.Helper()
	n, err := workflow.NewToolNode(tl, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewToolNode(%q) error = %v", tl.Name(), err)
	}
	return n
}

func newCloseTestParallelWorker(t *testing.T, name string, wrapped workflow.Node) *workflow.ParallelWorker {
	t.Helper()
	n, err := workflow.NewParallelWorker(name, wrapped, 0, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewParallelWorker(%q) error = %v", name, err)
	}
	return n
}

func newClosePlugin(t *testing.T, name string, closeFn func() error) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{Name: name, CloseFunc: closeFn})
	if err != nil {
		t.Fatalf("plugin.New(%q) error = %v", name, err)
	}
	return p
}

// newCloseTestRunner fills in the app name and an in-memory session service.
func newCloseTestRunner(t *testing.T, cfg Config) *Runner {
	t.Helper()
	cfg.AppName = "close_test"
	cfg.SessionService = session.InMemoryService()
	r, err := New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return r
}

// setToolsetCloseTimeout shortens toolsetCloseTimeout for one test. A test
// that calls it must not call t.Parallel.
func setToolsetCloseTimeout(t *testing.T, d time.Duration) {
	t.Helper()
	old := toolsetCloseTimeout
	toolsetCloseTimeout = d
	t.Cleanup(func() { toolsetCloseTimeout = old })
}

// closeWithin calls r.Close in a goroutine and fails the test when it has not
// returned after limit, so a missing timeout fails instead of hanging.
func closeWithin(t *testing.T, r *Runner, limit time.Duration) (time.Duration, error) {
	t.Helper()
	start := time.Now()
	done := make(chan error, 1)
	go func() { done <- r.Close() }()
	select {
	case err := <-done:
		return time.Since(start), err
	case <-time.After(limit):
		t.Fatalf("Close() has not returned after %v", limit)
		return 0, nil
	}
}

func TestRunner_Close_ClosesToolsetsOfEveryRootShape(t *testing.T) {
	tests := []struct {
		name         string
		config       func(t *testing.T, a, b tool.Toolset) Config
		wantA, wantB int32
	}{
		{
			name: "workflow agent root with an agent node",
			config: func(t *testing.T, a, b tool.Toolset) Config {
				worker := newCloseTestAgentNode(t, newCloseTestLLM(t, "worker", []tool.Toolset{a}))
				return Config{Agent: newCloseTestWorkflowAgent(t, "wf", []workflow.Edge{{From: workflow.Start, To: worker}})}
			},
			wantA: 1, wantB: 0,
		},
		{
			name: "workflow agent root, graph agent with a sub-agent",
			config: func(t *testing.T, a, b tool.Toolset) Config {
				helper := newCloseTestLLM(t, "helper", []tool.Toolset{b})
				worker := newCloseTestAgentNode(t, newCloseTestLLM(t, "worker", []tool.Toolset{a}, helper))
				return Config{Agent: newCloseTestWorkflowAgent(t, "wf", []workflow.Edge{{From: workflow.Start, To: worker}})}
			},
			wantA: 1, wantB: 1,
		},
		{
			name: "llm agent root with a sub-agent",
			config: func(t *testing.T, a, b tool.Toolset) Config {
				helper := newCloseTestLLM(t, "helper", []tool.Toolset{b})
				return Config{Agent: newCloseTestLLM(t, "root", []tool.Toolset{a}, helper)}
			},
			wantA: 1, wantB: 1,
		},
		{
			name: "agent node root",
			config: func(t *testing.T, a, b tool.Toolset) Config {
				return Config{Node: newCloseTestAgentNode(t, newCloseTestLLM(t, "worker", []tool.Toolset{a}))}
			},
			wantA: 1, wantB: 0,
		},
		{
			name: "workflow node root",
			config: func(t *testing.T, a, b tool.Toolset) Config {
				worker := newCloseTestAgentNode(t, newCloseTestLLM(t, "worker", []tool.Toolset{a}))
				return Config{Node: newCloseTestWorkflowNode(t, "inner", []workflow.Edge{{From: workflow.Start, To: worker}})}
			},
			wantA: 1, wantB: 0,
		},
		{
			name: "parallel worker root",
			config: func(t *testing.T, a, b tool.Toolset) Config {
				worker := newCloseTestAgentNode(t, newCloseTestLLM(t, "worker", []tool.Toolset{a}))
				return Config{Node: newCloseTestParallelWorker(t, "pw", worker)}
			},
			wantA: 1, wantB: 0,
		},
		{
			name: "parallel worker inside a workflow agent",
			config: func(t *testing.T, a, b tool.Toolset) Config {
				worker := newCloseTestAgentNode(t, newCloseTestLLM(t, "worker", []tool.Toolset{a}))
				pw := newCloseTestParallelWorker(t, "pw", worker)
				return Config{Agent: newCloseTestWorkflowAgent(t, "wf", []workflow.Edge{{From: workflow.Start, To: pw}})}
			},
			wantA: 1, wantB: 0,
		},
		{
			name: "agent tool in a ToolNode inside a workflow agent",
			config: func(t *testing.T, a, b tool.Toolset) Config {
				tn := newCloseTestToolNode(t, &agentWrapperTool{agent: newCloseTestLLM(t, "child", []tool.Toolset{a})})
				return Config{Agent: newCloseTestWorkflowAgent(t, "wf", []workflow.Edge{{From: workflow.Start, To: tn}})}
			},
			wantA: 1, wantB: 0,
		},
		{
			name: "ToolNode root",
			config: func(t *testing.T, a, b tool.Toolset) Config {
				return Config{Node: newCloseTestToolNode(t, &agentWrapperTool{agent: newCloseTestLLM(t, "child", []tool.Toolset{a})})}
			},
			wantA: 1, wantB: 0,
		},
		{
			name: "function node root",
			config: func(t *testing.T, a, b tool.Toolset) Config {
				fn := workflow.NewFunctionNode("fn", func(agent.Context, any) (string, error) { return "ok", nil }, workflow.NodeConfig{})
				return Config{Node: fn}
			},
			wantA: 0, wantB: 0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a, b := &countingToolset{name: "a"}, &countingToolset{name: "b"}
			r := newCloseTestRunner(t, tc.config(t, a, b))
			if a.closes.Load() != 0 || b.closes.Load() != 0 {
				t.Fatalf("before Close: closes a=%d b=%d, want 0 0", a.closes.Load(), b.closes.Load())
			}

			if err := r.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}

			if got := a.closes.Load(); got != tc.wantA {
				t.Errorf("closes of a = %d, want %d", got, tc.wantA)
			}
			if got := b.closes.Load(); got != tc.wantB {
				t.Errorf("closes of b = %d, want %d", got, tc.wantB)
			}
		})
	}
}

func TestRunner_Close_ClosesSharedToolsetOnce(t *testing.T) {
	tests := []struct {
		name   string
		config func(t *testing.T, a tool.Toolset) Config
	}{
		{
			name: "one toolset on the root and on its sub-agent",
			config: func(t *testing.T, a tool.Toolset) Config {
				sub := newCloseTestLLM(t, "sub", []tool.Toolset{a})
				return Config{Agent: newCloseTestLLM(t, "root", []tool.Toolset{a}, sub)}
			},
		},
		{
			name: "one toolset behind two different wrappers",
			config: func(t *testing.T, a tool.Toolset) Config {
				p := tool.StringPredicate([]string{"x"})
				sub := newCloseTestLLM(t, "sub", []tool.Toolset{tool.WithConfirmation(tool.FilterToolset(a, p), false, nil)})
				return Config{Agent: newCloseTestLLM(t, "root", []tool.Toolset{tool.FilterToolset(a, p)}, sub)}
			},
		},
		{
			name: "one agent in the outer graph and in a nested workflow",
			config: func(t *testing.T, a tool.Toolset) Config {
				worker := newCloseTestLLM(t, "worker", []tool.Toolset{a})
				inner := newCloseTestWorkflowNode(t, "inner", []workflow.Edge{{From: workflow.Start, To: newCloseTestAgentNode(t, worker)}})
				return Config{Agent: newCloseTestWorkflowAgent(t, "wf", []workflow.Edge{
					{From: workflow.Start, To: newCloseTestAgentNode(t, worker)},
					{From: workflow.Start, To: inner},
				})}
			},
		},
		{
			name: "one toolset behind a wrapper without Close and directly on a sub-agent",
			config: func(t *testing.T, a tool.Toolset) Config {
				sub := newCloseTestLLM(t, "sub", []tool.Toolset{a})
				return Config{Agent: newCloseTestLLM(t, "root", []tool.Toolset{&unwrapOnlyToolset{inner: a}}, sub)}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := &countingToolset{name: "a"}
			r := newCloseTestRunner(t, tc.config(t, a))

			if err := r.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}

			if got := a.closes.Load(); got != 1 {
				t.Errorf("closes of a = %d, want 1", got)
			}
		})
	}
}

func TestRunner_Close_ClosesToolsetsThenPluginsInWalkOrder(t *testing.T) {
	log := &closeLog{}
	sub := newCloseTestLLM(t, "sub", []tool.Toolset{&countingToolset{name: "ts-sub", log: log}})
	root := newCloseTestLLM(t, "root", []tool.Toolset{&countingToolset{name: "ts-root", log: log}}, sub)
	plugins := []*plugin.Plugin{
		newClosePlugin(t, "p1", func() error { log.add("p1"); return nil }),
		newClosePlugin(t, "p2", func() error { log.add("p2"); return nil }),
	}
	r := newCloseTestRunner(t, Config{Agent: root, PluginConfig: PluginConfig{Plugins: plugins}})

	if err := r.Close(); err != nil {
		t.Fatalf("first Close() error = %v", err)
	}
	if diff := cmp.Diff([]string{"ts-root", "ts-sub", "p1", "p2"}, log.get()); diff != "" {
		t.Errorf("close order after the first Close (-want +got):\n%s", diff)
	}

	if err := r.Close(); err != nil {
		t.Fatalf("second Close() error = %v", err)
	}
	if diff := cmp.Diff([]string{"ts-root", "ts-sub", "p1", "p2", "ts-root", "ts-sub", "p1", "p2"}, log.get()); diff != "" {
		t.Errorf("close order after the second Close (-want +got):\n%s", diff)
	}
}

func TestRunner_Close_AttemptsEverythingAfterFailures(t *testing.T) {
	setToolsetCloseTimeout(t, 30*time.Millisecond)

	errBoom := errors.New("boom")
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseHanging := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseHanging)
	hangingReturned := make(chan struct{}, 1)

	healthy := &countingToolset{name: "healthy"}
	root := newCloseTestLLM(t, "root", []tool.Toolset{
		&scriptedCloseToolset{name: "failing", close: func() error { return errBoom }},
		&scriptedCloseToolset{name: "panicking", close: func() error { panic("boom user data") }},
		&scriptedCloseToolset{name: "hanging", close: func() error {
			<-release
			hangingReturned <- struct{}{}
			return nil
		}},
		healthy,
	})
	var p1Closes, p2Closes atomic.Int32
	plugins := []*plugin.Plugin{
		newClosePlugin(t, "p1", func() error { p1Closes.Add(1); return errors.New("p1 failed") }),
		newClosePlugin(t, "p2", func() error { p2Closes.Add(1); return nil }),
	}
	r := newCloseTestRunner(t, Config{Agent: root, PluginConfig: PluginConfig{Plugins: plugins}})

	_, err := closeWithin(t, r, 2*time.Second)

	if err == nil {
		t.Fatal("Close() = nil, want the joined errors")
	}
	if !errors.Is(err, errBoom) {
		t.Errorf("errors.Is(Close(), errBoom) = false, want true; err = %v", err)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("errors.Is(Close(), context.DeadlineExceeded) = false, want true; err = %v", err)
	}
	msg := err.Error()
	for _, want := range []string{`close toolset "panicking": panic of type string`, "error closing plugin 'p1'", "p1 failed"} {
		if !strings.Contains(msg, want) {
			t.Errorf("Close() error %q does not contain %q", msg, want)
		}
	}
	if strings.Contains(msg, "boom user data") {
		t.Errorf("Close() error %q contains the panic value", msg)
	}
	if got := healthy.closes.Load(); got != 1 {
		t.Errorf("closes of healthy = %d, want 1", got)
	}
	if got := p1Closes.Load(); got != 1 {
		t.Errorf("closes of p1 = %d, want 1", got)
	}
	if got := p2Closes.Load(); got != 1 {
		t.Errorf("closes of p2 = %d, want 1", got)
	}

	releaseHanging()
	select {
	case <-hangingReturned:
	case <-time.After(time.Second):
		t.Error("the hanging toolset's Close did not return within 1s of its release")
	}
}

func TestRunner_Close_AbandonsAHangingToolsetAfterTheTimeout(t *testing.T) {
	setToolsetCloseTimeout(t, 30*time.Millisecond)

	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	var mu sync.Mutex
	var hangStart, nextStart time.Time

	next := &countingToolset{name: "next"}
	root := newCloseTestLLM(t, "root", []tool.Toolset{
		&scriptedCloseToolset{name: "hanging", close: func() error {
			mu.Lock()
			hangStart = time.Now()
			mu.Unlock()
			<-release
			return nil
		}},
		&scriptedCloseToolset{name: "next", close: func() error {
			mu.Lock()
			nextStart = time.Now()
			mu.Unlock()
			return next.Close()
		}},
	})
	r := newCloseTestRunner(t, Config{Agent: root})

	elapsed, err := closeWithin(t, r, 2*time.Second)

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("errors.Is(Close(), context.DeadlineExceeded) = false, want true; err = %v", err)
	}
	if elapsed < 30*time.Millisecond {
		t.Errorf("Close() returned after %v, want at least 30ms", elapsed)
	}
	if got := next.closes.Load(); got != 1 {
		t.Errorf("closes of next = %d, want 1", got)
	}
	mu.Lock()
	defer mu.Unlock()
	if hangStart.IsZero() || nextStart.IsZero() {
		t.Fatalf("close start times: hanging %v, next %v, want both set", hangStart, nextStart)
	}
	if gap := nextStart.Sub(hangStart); gap < 30*time.Millisecond {
		t.Errorf("next started closing %v after hanging, want at least 30ms (one after another)", gap)
	}
}

// Toolset values whose dynamic type is not comparable, so they cannot be map
// keys. Each counts its closes through a shared pointer.
type mapToolset struct {
	m      map[string]int
	closes *atomic.Int32
}

func (t mapToolset) Name() string                                     { return "map" }
func (t mapToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return nil, nil }
func (t mapToolset) Close() error                                     { t.closes.Add(1); return nil }

type funcToolset struct {
	f      func()
	closes *atomic.Int32
}

func (t funcToolset) Name() string                                     { return "func" }
func (t funcToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return nil, nil }
func (t funcToolset) Close() error                                     { t.closes.Add(1); return nil }

type ifaceToolset struct {
	x      any
	closes *atomic.Int32
}

func (t ifaceToolset) Name() string                                     { return "iface" }
func (t ifaceToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return nil, nil }
func (t ifaceToolset) Close() error                                     { t.closes.Add(1); return nil }

func TestRunner_Close_DoesNotPanicOnToolsetsThatAreNotComparable(t *testing.T) {
	var mapCloses, funcCloses, ifaceCloses atomic.Int32
	toolsets := []tool.Toolset{
		mapToolset{m: map[string]int{"k": 1}, closes: &mapCloses},
		funcToolset{f: func() {}, closes: &funcCloses},
		ifaceToolset{x: []int{1}, closes: &ifaceCloses},
	}
	sub := newCloseTestLLM(t, "sub", toolsets)
	r := newCloseTestRunner(t, Config{Agent: newCloseTestLLM(t, "root", toolsets, sub)})

	if err := r.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	for name, got := range map[string]int32{"map": mapCloses.Load(), "func": funcCloses.Load(), "iface": ifaceCloses.Load()} {
		if got != 2 {
			t.Errorf("closes of %s = %d, want 2 (once per place)", name, got)
		}
	}
}

// A toolset that is not comparable has no identity to de-duplicate on, so
// only walking each agent once keeps the toolset of an agent that two graph
// nodes run, one in the outer graph and one in a nested workflow, from being
// closed twice.
func TestRunner_Close_WalksAnAgentRunByTwoNodesOnce(t *testing.T) {
	var closes atomic.Int32
	worker := newCloseTestLLM(t, "worker", []tool.Toolset{mapToolset{m: map[string]int{"k": 1}, closes: &closes}})
	inner := newCloseTestWorkflowNode(t, "inner", []workflow.Edge{{From: workflow.Start, To: newCloseTestAgentNode(t, worker)}})
	r := newCloseTestRunner(t, Config{Agent: newCloseTestWorkflowAgent(t, "wf", []workflow.Edge{
		{From: workflow.Start, To: newCloseTestAgentNode(t, worker)},
		{From: workflow.Start, To: inner},
	})})

	if err := r.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	if got := closes.Load(); got != 1 {
		t.Errorf("closes of the worker's toolset = %d, want 1", got)
	}
}

func TestRunner_Close_ConcurrentWithCloseAndRun(t *testing.T) {
	ts := &countingToolset{name: "ts"}
	root, err := llmagent.New(llmagent.Config{
		Name:     "root",
		Model:    &scriptedModel{replyFmt: "reply %d"},
		Toolsets: []tool.Toolset{ts},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r := newCloseTestRunner(t, Config{Agent: root, AutoCreateSession: true})

	var wg sync.WaitGroup
	closeErrs := make([]error, 2)
	var runErr error
	wg.Add(3)
	for i := range closeErrs {
		go func() {
			defer wg.Done()
			closeErrs[i] = r.Close()
		}()
	}
	go func() {
		defer wg.Done()
		for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
			if err != nil && runErr == nil {
				runErr = err
			}
		}
	}()
	wg.Wait()

	for i, err := range closeErrs {
		if err != nil {
			t.Errorf("Close() call %d error = %v", i, err)
		}
	}
	if runErr != nil {
		t.Errorf("Run() error = %v", runErr)
	}
	if got := ts.closes.Load(); got != 2 {
		t.Errorf("closes of ts = %d, want 2", got)
	}
}

// plainToolset holds no resource and so does not implement io.Closer.
type plainToolset struct{}

func (plainToolset) Name() string                                     { return "plain" }
func (plainToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return nil, nil }

func TestRunner_Close_SkipsToolsetsThatAreNotClosers(t *testing.T) {
	healthy := &countingToolset{name: "healthy"}
	root := newCloseTestLLM(t, "root", []tool.Toolset{plainToolset{}, healthy})
	r := newCloseTestRunner(t, Config{Agent: root})

	if err := r.Close(); err != nil {
		t.Fatalf("Close() error = %v, want nil", err)
	}

	if got := healthy.closes.Load(); got != 1 {
		t.Errorf("closes of healthy = %d, want 1", got)
	}
}

// labeledGoroutineAlive reports whether a live goroutine carries the profiler
// label key=value. A goroutine inherits the labels of the goroutine that
// starts it, so this finds every goroutine started, directly or not, inside
// pprof.Do with that label, whatever its function is called.
func labeledGoroutineAlive(t *testing.T, key, value string) bool {
	t.Helper()
	var buf bytes.Buffer
	if err := pprof.Lookup("goroutine").WriteTo(&buf, 1); err != nil {
		t.Fatalf("writing the goroutine profile: %v", err)
	}
	return strings.Contains(buf.String(), fmt.Sprintf("%q:%q", key, value))
}

func TestRunner_Close_AbandonedCloseGoroutineEndsWhenCloseReturns(t *testing.T) {
	setToolsetCloseTimeout(t, 30*time.Millisecond)
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseHanging := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseHanging)
	root := newCloseTestLLM(t, "root", []tool.Toolset{
		&scriptedCloseToolset{name: "hanging", close: func() error { <-release; return nil }},
	})
	r := newCloseTestRunner(t, Config{Agent: root})
	const labelKey = "runner-close-test"
	labelValue := t.Name()

	var err error
	pprof.Do(t.Context(), pprof.Labels(labelKey, labelValue), func(context.Context) {
		_, err = closeWithin(t, r, 2*time.Second)
	})

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Close() error = %v, want deadline exceeded", err)
	}
	if !labeledGoroutineAlive(t, labelKey, labelValue) {
		t.Fatal("no goroutine started by Close is alive after the timeout, want the one running the hanging toolset's Close")
	}
	releaseHanging()
	deadline := time.Now().Add(2 * time.Second)
	for labeledGoroutineAlive(t, labelKey, labelValue) {
		if time.Now().After(deadline) {
			t.Fatal("a goroutine started by Close is still alive 2s after the hanging toolset's Close returned")
		}
		time.Sleep(10 * time.Millisecond)
	}
}
