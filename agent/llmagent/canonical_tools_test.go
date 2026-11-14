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

package llmagent_test

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// ctTool returns a function tool named name, which offers the model its
// declaration.
func ctTool(t *testing.T, name string) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: name, Description: name + " tool"},
		func(agent.Context, struct{}) (map[string]any, error) { return map[string]any{}, nil })
	if err != nil {
		t.Fatalf("functiontool.New(%q): %v", name, err)
	}
	return tl
}

// ctToolset lists tools and fails with err when err is set; a failing
// ctToolset with tools returns them alongside its error. onList, when set,
// runs first with the context the toolset received.
type ctToolset struct {
	name   string
	tools  []tool.Tool
	err    error
	onList func(agent.ReadonlyContext)
}

func (s *ctToolset) Name() string { return s.name }

func (s *ctToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	if s.onList != nil {
		s.onList(ctx)
	}
	return s.tools, s.err
}

func ctNames(tools []tool.Tool) []string {
	names := []string{}
	for _, tl := range tools {
		names = append(names, tl.Name())
	}
	return names
}

func mustCanonicalTools(t *testing.T, ctx agent.ReadonlyContext, a agent.Agent) []string {
	t.Helper()
	got, err := llmagent.CanonicalTools(ctx, a)
	if err != nil {
		t.Fatalf("CanonicalTools(%q): %v", a.Name(), err)
	}
	return ctNames(got)
}

// ctBarrier returns two listing hooks that each return only once both have
// started, so two toolsets carrying them finish only when listed at the same
// time. A hook that waits 5 seconds for the other reports a test error and
// returns.
func ctBarrier(t *testing.T) (first, second func(agent.ReadonlyContext)) {
	t.Helper()
	firstStarted, secondStarted := make(chan struct{}), make(chan struct{})
	hook := func(own, other chan struct{}) func(agent.ReadonlyContext) {
		var once sync.Once
		return func(agent.ReadonlyContext) {
			once.Do(func() { close(own) })
			select {
			case <-other:
			case <-time.After(5 * time.Second):
				t.Error("a toolset waited 5s for the other to start listing; toolsets are listed one after another")
			}
		}
	}
	return hook(firstStarted, secondStarted), hook(secondStarted, firstStarted)
}

// ctWait returns once ch is closed. A listing hook calls it to finish after
// a toolset declared later, so declared order and the order in time differ.
// After 5 seconds it reports a test error and returns.
func ctWait(t *testing.T, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.Error("a toolset waited 5s for a later toolset to start listing; toolsets are listed one after another")
	}
}

// declaredOrderToolsets is u3111's shape: plain tools before, between and
// after toolsets, in Config.Toolsets through tool.AsToolset.
func declaredOrderToolsets(t *testing.T) []tool.Toolset {
	t.Helper()
	return []tool.Toolset{
		tool.AsToolset(ctTool(t, "plain_1")),
		&ctToolset{name: "ts1", tools: []tool.Tool{ctTool(t, "ts1_a"), ctTool(t, "ts1_b")}},
		&ctToolset{name: "ts2", tools: []tool.Tool{ctTool(t, "ts2_a")}},
		tool.AsToolset(ctTool(t, "plain_2")),
	}
}

func TestCanonicalTools_ConfigToolsThenToolsets(t *testing.T) {
	a := newLLMAgent(t, llmagent.Config{
		Name:     "a",
		Tools:    []tool.Tool{ctTool(t, "t1")},
		Toolsets: []tool.Toolset{&ctToolset{name: "ts", tools: []tool.Tool{ctTool(t, "a"), ctTool(t, "b")}}},
	})

	got := mustCanonicalTools(t, nil, a)

	if want := []string{"t1", "a", "b"}; !slices.Equal(got, want) {
		t.Errorf("CanonicalTools() = %v, want %v", got, want)
	}
}

func TestCanonicalTools_DeclaredOrderWithAsToolset(t *testing.T) {
	a := newLLMAgent(t, llmagent.Config{Name: "a", Toolsets: declaredOrderToolsets(t)})

	got := mustCanonicalTools(t, nil, a)

	if want := []string{"plain_1", "ts1_a", "ts1_b", "ts2_a", "plain_2"}; !slices.Equal(got, want) {
		t.Errorf("CanonicalTools() = %v, want %v", got, want)
	}
}

func TestCanonicalTools_PassesTheContextToToolsets(t *testing.T) {
	var received []agent.ReadonlyContext
	recorder := &ctToolset{name: "recorder", onList: func(ctx agent.ReadonlyContext) { received = append(received, ctx) }}
	a := newLLMAgent(t, llmagent.Config{Name: "a", Toolsets: []tool.Toolset{recorder}})
	ctx := icontext.NewReadonlyContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a}))

	mustCanonicalTools(t, nil, a)
	mustCanonicalTools(t, ctx, a)

	if len(received) != 2 {
		t.Fatalf("the toolset was listed %d times, want 2", len(received))
	}
	if received[0] != nil {
		t.Errorf("CanonicalTools(nil, a) handed the toolset %T, want a nil context", received[0])
	}
	if received[1] != ctx {
		t.Errorf("CanonicalTools(ctx, a) handed the toolset %T, want the caller's ctx", received[1])
	}
}

func TestCanonicalTools_ListsToolsetsConcurrently(t *testing.T) {
	first, second := ctBarrier(t)
	a := newLLMAgent(t, llmagent.Config{Name: "a", Toolsets: []tool.Toolset{
		&ctToolset{name: "first", tools: []tool.Tool{ctTool(t, "first_tool")}, onList: first},
		&ctToolset{name: "second", tools: []tool.Tool{ctTool(t, "second_tool")}, onList: second},
	}})

	got := mustCanonicalTools(t, nil, a)

	if want := []string{"first_tool", "second_tool"}; !slices.Equal(got, want) {
		t.Errorf("CanonicalTools() = %v, want %v", got, want)
	}
}

func TestCanonicalTools_SkipsFailingToolset(t *testing.T) {
	a := newLLMAgent(t, llmagent.Config{
		Name:  "a",
		Tools: []tool.Tool{ctTool(t, "regular_tool")},
		Toolsets: []tool.Toolset{
			&ctToolset{name: "failing", tools: []tool.Tool{ctTool(t, "partial_tool")}, err: errors.New("mcp down")},
			&ctToolset{name: "working", tools: []tool.Tool{ctTool(t, "working_tool")}},
		},
	})

	got, err := llmagent.CanonicalTools(nil, a)
	if err != nil {
		t.Fatalf("CanonicalTools() error = %v, want the failing toolset skipped", err)
	}

	if want := []string{"regular_tool", "working_tool"}; !slices.Equal(ctNames(got), want) {
		t.Errorf("CanonicalTools() = %v, want %v", ctNames(got), want)
	}
}

func TestCanonicalTools_CancelledContextWithFailingToolsetReturnsError(t *testing.T) {
	ownErr := errors.New("connection reset")
	a := newLLMAgent(t, llmagent.Config{Name: "a", Toolsets: []tool.Toolset{
		&ctToolset{name: "healthy", tools: []tool.Tool{ctTool(t, "healthy_tool")}},
		&ctToolset{name: "broken", err: ownErr},
	}})
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	ctx := icontext.NewReadonlyContext(icontext.NewInvocationContext(cancelled, icontext.InvocationContextParams{Agent: a}))

	got, err := llmagent.CanonicalTools(ctx, a)

	if !errors.Is(err, context.Canceled) {
		t.Errorf("CanonicalTools() error = %v, want one wrapping context.Canceled", err)
	}
	if !errors.Is(err, ownErr) {
		t.Errorf("CanonicalTools() error = %v, want one wrapping the toolset's error", err)
	}
	if err != nil && !strings.Contains(err.Error(), `"broken"`) {
		t.Errorf("CanonicalTools() error = %q, want it to name the toolset \"broken\"", err)
	}
	if got != nil {
		t.Errorf("CanonicalTools() tools = %v, want nil with the error", ctNames(got))
	}
}

func TestCanonicalTools_CancelledContextReportsTheFirstFailingToolsetInDeclaredOrder(t *testing.T) {
	firstErr, secondErr := errors.New("first down"), errors.New("second down")
	secondListed := make(chan struct{})
	a := newLLMAgent(t, llmagent.Config{Name: "a", Toolsets: []tool.Toolset{
		&ctToolset{name: "first_broken", err: firstErr, onList: func(agent.ReadonlyContext) { ctWait(t, secondListed) }},
		&ctToolset{name: "second_broken", err: secondErr, onList: func(agent.ReadonlyContext) { close(secondListed) }},
	}})
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	ctx := icontext.NewReadonlyContext(icontext.NewInvocationContext(cancelled, icontext.InvocationContextParams{Agent: a}))

	_, err := llmagent.CanonicalTools(ctx, a)

	if !errors.Is(err, firstErr) || errors.Is(err, secondErr) {
		t.Errorf("CanonicalTools() error = %v, want the error of first_broken only", err)
	}
	if err != nil && (!strings.Contains(err.Error(), `"first_broken"`) || strings.Contains(err.Error(), "second_broken")) {
		t.Errorf("CanonicalTools() error = %q, want it to name \"first_broken\" and not second_broken", err)
	}
}

func TestCanonicalTools_FirstPanicInDeclaredOrderReachesTheCaller(t *testing.T) {
	secondListed := make(chan struct{})
	a := newLLMAgent(t, llmagent.Config{Name: "a", Toolsets: []tool.Toolset{
		&ctToolset{name: "first", onList: func(agent.ReadonlyContext) { ctWait(t, secondListed); panic("first boom") }},
		&ctToolset{name: "second", onList: func(agent.ReadonlyContext) { close(secondListed); panic("second boom") }},
	}})

	var recovered any
	func() {
		defer func() { recovered = recover() }()
		_, _ = llmagent.CanonicalTools(nil, a)
	}()

	if recovered != "first boom" {
		t.Errorf("recovered %v in the caller's goroutine, want the first toolset's panic value \"first boom\"", recovered)
	}
}

func TestCanonicalTools_CancelledContextWithHealthyToolsetsReturnsTools(t *testing.T) {
	a := newLLMAgent(t, llmagent.Config{Name: "a", Toolsets: []tool.Toolset{
		&ctToolset{name: "one", tools: []tool.Tool{ctTool(t, "one_tool")}},
		&ctToolset{name: "two", tools: []tool.Tool{ctTool(t, "two_tool")}},
	}})
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	ctx := icontext.NewReadonlyContext(icontext.NewInvocationContext(cancelled, icontext.InvocationContextParams{Agent: a}))

	got := mustCanonicalTools(t, ctx, a)

	if want := []string{"one_tool", "two_tool"}; !slices.Equal(got, want) {
		t.Errorf("CanonicalTools() = %v, want %v", got, want)
	}
}

func TestCanonicalTools_ToolsetPanicReachesTheCaller(t *testing.T) {
	a := newLLMAgent(t, llmagent.Config{Name: "a", Toolsets: []tool.Toolset{
		&ctToolset{name: "panicking", onList: func(agent.ReadonlyContext) { panic("boom") }},
		&ctToolset{name: "healthy", tools: []tool.Tool{ctTool(t, "healthy_tool")}},
	}})

	var recovered any
	func() {
		defer func() { recovered = recover() }()
		_, _ = llmagent.CanonicalTools(nil, a)
	}()

	if recovered != "boom" {
		t.Errorf("recovered %v in the caller's goroutine, want the toolset's panic value \"boom\"", recovered)
	}
}

func TestCanonicalTools_ReturnsAFreshSlice(t *testing.T) {
	a := newLLMAgent(t, llmagent.Config{Name: "a", Tools: []tool.Tool{ctTool(t, "own")}})

	first, err := llmagent.CanonicalTools(nil, a)
	if err != nil {
		t.Fatalf("CanonicalTools(): %v", err)
	}
	first[0] = nil
	second, err := llmagent.CanonicalTools(nil, a)
	if err != nil {
		t.Fatalf("CanonicalTools(): %v", err)
	}

	if len(second) != 1 || second[0] == nil || second[0].Name() != "own" {
		t.Errorf("second CanonicalTools() = %v, want [own] after the caller changed the first result", second)
	}
}

func TestCanonicalTools_RejectsNilAndNonLLMAgent(t *testing.T) {
	custom, err := agent.New(agent.Config{Name: "custom"})
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range []agent.Agent{nil, custom} {
		got, err := llmagent.CanonicalTools(nil, a)
		if err == nil {
			t.Errorf("CanonicalTools(nil, %v) error = nil, want an error", a)
		}
		if got != nil {
			t.Errorf("CanonicalTools(nil, %v) = %v, want nil", a, ctNames(got))
		}
	}
}
