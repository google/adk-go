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
	"fmt"
	"iter"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// canonLLM is a model that records each request and answers with the next
// scripted content, or with the text "ok" once the script is used up.
type canonLLM struct {
	name    string
	backend genai.Backend

	mu       sync.Mutex
	requests []*model.LLMRequest
	script   []*genai.Content
}

func (m *canonLLM) Name() string { return m.name }

func (m *canonLLM) GetGoogleLLMVariant() genai.Backend { return m.backend }

func (m *canonLLM) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	m.requests = append(m.requests, req)
	content := genai.NewContentFromText("ok", genai.RoleModel)
	if len(m.script) > 0 {
		content, m.script = m.script[0], m.script[1:]
	}
	m.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop}, nil)
	}
}

func (m *canonLLM) requestModels() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var names []string
	for _, r := range m.requests {
		names = append(names, r.Model)
	}
	return names
}

// Test model patterns are registered once per test binary, because
// model.Register panics for a pattern registered twice (go test -count=2).
var (
	registerCanonPatterns sync.Once

	// memoCalls counts factory calls per name for ^fr-canon-memo-.*$.
	memoCallsMu sync.Mutex
	memoCalls   = map[string]int{}

	// gateOpen decides whether the factory for ^fr-canon-gated-.*$ succeeds.
	gateOpen atomic.Bool

	// memoRuns numbers the runs of TestDefaultModel_MemoizedUntilSetterCalled.
	memoRuns atomic.Int64
)

func registerCanonTestPatterns(t *testing.T) {
	t.Helper()
	registerCanonPatterns.Do(func() {
		model.Register("^fr-canon-memo-.*$", func(_ context.Context, name string) (model.LLM, error) {
			memoCallsMu.Lock()
			memoCalls[name]++
			memoCallsMu.Unlock()
			return &canonLLM{name: name}, nil
		})
		model.Register("^fr-canon-gated-.*$", func(_ context.Context, name string) (model.LLM, error) {
			if !gateOpen.Load() {
				return nil, errors.New("gate closed")
			}
			return &canonLLM{name: name}, nil
		})
		model.Register("^fr-canon-plain-.*$", func(_ context.Context, name string) (model.LLM, error) {
			return &canonLLM{name: name}, nil
		})
	})
}

func memoCallCount(name string) int {
	memoCallsMu.Lock()
	defer memoCallsMu.Unlock()
	return memoCalls[name]
}

// restoreDefaults puts both process-wide defaults back to the built-in names
// when the test ends.
func restoreDefaults(t *testing.T) {
	t.Helper()
	t.Cleanup(func() {
		if err := llmagent.SetDefaultModelByName(llmagent.DefaultModel); err != nil {
			t.Errorf("restore default model: %v", err)
		}
		if err := llmagent.SetDefaultLiveModelByName(llmagent.DefaultLiveModel); err != nil {
			t.Errorf("restore default live model: %v", err)
		}
	})
}

func newLLMAgent(t *testing.T, cfg llmagent.Config) agent.Agent {
	t.Helper()
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New(%q): %v", cfg.Name, err)
	}
	return a
}

func mustCanonicalModel(t *testing.T, a agent.Agent) model.LLM {
	t.Helper()
	m, err := llmagent.CanonicalModel(a)
	if err != nil {
		t.Fatalf("CanonicalModel(%q): %v", a.Name(), err)
	}
	return m
}

func mustCanonicalLiveModel(t *testing.T, a agent.Agent) model.LLM {
	t.Helper()
	m, err := llmagent.CanonicalLiveModel(a)
	if err != nil {
		t.Fatalf("CanonicalLiveModel(%q): %v", a.Name(), err)
	}
	return m
}

func canonName(m model.LLM) string {
	if m == nil {
		return "<nil>"
	}
	return m.Name()
}

func TestCanonicalModel_OwnModelAndLiveModel(t *testing.T) {
	own := &canonLLM{name: "own-model"}
	a := newLLMAgent(t, llmagent.Config{Name: "a", Model: own})

	if got := mustCanonicalModel(t, a); got != own {
		t.Errorf("CanonicalModel() = %s, want the agent's own model own-model", canonName(got))
	}
	if got := mustCanonicalLiveModel(t, a); got != own {
		t.Errorf("CanonicalLiveModel() = %s, want the agent's own model own-model", canonName(got))
	}
}

func TestCanonicalModel_RejectsNilAndNonLLMAgent(t *testing.T) {
	custom, err := agent.New(agent.Config{Name: "custom"})
	if err != nil {
		t.Fatal(err)
	}
	ctx := icontext.NewReadonlyContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: custom}))
	for _, a := range []agent.Agent{nil, custom} {
		if m, err := llmagent.CanonicalModel(a); err == nil {
			t.Errorf("CanonicalModel(%v) = %s, nil; want an error", a, canonName(m))
		}
		if m, err := llmagent.CanonicalLiveModel(a); err == nil {
			t.Errorf("CanonicalLiveModel(%v) = %s, nil; want an error", a, canonName(m))
		}
		if text, _, err := llmagent.CanonicalInstruction(ctx, a); err == nil {
			t.Errorf("CanonicalInstruction(%v) = %q, nil; want an error", a, text)
		}
	}
}

func TestCanonicalModel_UnknownDefaultNameReturnsError(t *testing.T) {
	registerCanonTestPatterns(t)
	restoreDefaults(t)
	a := newLLMAgent(t, llmagent.Config{Name: "a"})

	if err := llmagent.SetDefaultModelByName("fr-unregistered-model-xyz"); err != nil {
		t.Fatal(err)
	}
	if err := llmagent.SetDefaultLiveModelByName("fr-unregistered-live-xyz"); err != nil {
		t.Fatal(err)
	}
	m, err := llmagent.CanonicalModel(a)
	if err == nil || !strings.Contains(err.Error(), "fr-unregistered-model-xyz") {
		t.Errorf("CanonicalModel() = %s, %v; want an error naming fr-unregistered-model-xyz", canonName(m), err)
	}
	m, err = llmagent.CanonicalLiveModel(a)
	if err == nil || !strings.Contains(err.Error(), "fr-unregistered-live-xyz") {
		t.Errorf("CanonicalLiveModel() = %s, %v; want an error naming fr-unregistered-live-xyz", canonName(m), err)
	}

	// A failed resolution is not kept: once the factory succeeds, the same
	// default name resolves without another setter call.
	gateOpen.Store(false)
	t.Cleanup(func() { gateOpen.Store(false) })
	if err := llmagent.SetDefaultModelByName("fr-canon-gated-1"); err != nil {
		t.Fatal(err)
	}
	if err := llmagent.SetDefaultLiveModelByName("fr-canon-gated-live-1"); err != nil {
		t.Fatal(err)
	}
	if m, err := llmagent.CanonicalModel(a); err == nil {
		t.Fatalf("CanonicalModel() with the gate closed = %s, nil; want an error", canonName(m))
	}
	if m, err := llmagent.CanonicalLiveModel(a); err == nil {
		t.Fatalf("CanonicalLiveModel() with the gate closed = %s, nil; want an error", canonName(m))
	}
	gateOpen.Store(true)
	if got := canonName(mustCanonicalModel(t, a)); got != "fr-canon-gated-1" {
		t.Errorf("CanonicalModel() after the gate opened = %s, want fr-canon-gated-1", got)
	}
	if got := canonName(mustCanonicalLiveModel(t, a)); got != "fr-canon-gated-live-1" {
		t.Errorf("CanonicalLiveModel() after the gate opened = %s, want fr-canon-gated-live-1", got)
	}
}

func TestCanonicalModel_InheritsAcrossWorkflowAndCustomAncestors(t *testing.T) {
	// Children are built before their parents, as Go requires.
	leaf := newLLMAgent(t, llmagent.Config{Name: "leaf"})
	custom, err := agent.New(agent.Config{Name: "custom", SubAgents: []agent.Agent{leaf}})
	if err != nil {
		t.Fatal(err)
	}
	loop, err := loopagent.New(loopagent.Config{AgentConfig: agent.Config{Name: "loop", SubAgents: []agent.Agent{custom}}})
	if err != nil {
		t.Fatal(err)
	}
	par, err := parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "par", SubAgents: []agent.Agent{loop}}})
	if err != nil {
		t.Fatal(err)
	}
	seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{par}}})
	if err != nil {
		t.Fatal(err)
	}
	mid := newLLMAgent(t, llmagent.Config{Name: "mid", SubAgents: []agent.Agent{seq}})
	rootLLM := &canonLLM{name: "root-llm"}
	newLLMAgent(t, llmagent.Config{Name: "root", Model: rootLLM, SubAgents: []agent.Agent{mid}})

	if got := mustCanonicalModel(t, leaf); got != rootLLM {
		t.Errorf("CanonicalModel(leaf) = %s, want root-llm", canonName(got))
	}
	if got := mustCanonicalLiveModel(t, leaf); got != rootLLM {
		t.Errorf("CanonicalLiveModel(leaf) = %s, want root-llm", canonName(got))
	}
	if got := mustCanonicalModel(t, mid); got != rootLLM {
		t.Errorf("CanonicalModel(mid) = %s, want root-llm", canonName(got))
	}
}

// findAgentWrapper embeds an agent and overrides one method, as the
// conformance Adapter's live wrapper does.
type findAgentWrapper struct {
	agent.Agent
}

func (w findAgentWrapper) FindAgent(name string) agent.Agent { return w.Agent.FindAgent(name) }

func TestCanonicalModel_InheritsThroughEmbeddingWrapper(t *testing.T) {
	leaf := newLLMAgent(t, llmagent.Config{Name: "leaf"})
	custom, err := agent.New(agent.Config{Name: "custom", SubAgents: []agent.Agent{leaf}})
	if err != nil {
		t.Fatal(err)
	}
	rootLLM := &canonLLM{name: "root-llm"}
	newLLMAgent(t, llmagent.Config{Name: "root", Model: rootLLM, SubAgents: []agent.Agent{findAgentWrapper{custom}}})

	if got := mustCanonicalModel(t, leaf); got != rootLLM {
		t.Errorf("CanonicalModel(leaf) = %s, want root-llm", canonName(got))
	}
}

func TestCanonicalModel_InheritsThroughEmbeddingWrapperInRun(t *testing.T) {
	leaf := newLLMAgent(t, llmagent.Config{Name: "leaf", Description: "the leaf"})
	midLLM := &canonLLM{name: "mid-llm", script: []*genai.Content{transferTo("leaf")}}
	mid := newLLMAgent(t, llmagent.Config{Name: "mid", Description: "the mid", Model: midLLM, SubAgents: []agent.Agent{leaf}})
	rootLLM := &canonLLM{name: "root-llm", script: []*genai.Content{transferTo("mid")}}
	root := newLLMAgent(t, llmagent.Config{Name: "root", Model: rootLLM, SubAgents: []agent.Agent{findAgentWrapper{mid}}})

	if got := mustCanonicalModel(t, leaf); got != midLLM {
		t.Fatalf("CanonicalModel(leaf) outside a run = %s, want mid-llm", canonName(got))
	}

	runAgent(t, root)

	if got, want := strings.Join(midLLM.requestModels(), ","), "mid-llm,mid-llm"; got != want {
		t.Errorf("mid-llm request models = %q, want %q (mid, then leaf inheriting from wrapped mid); root-llm got %q", got, want, strings.Join(rootLLM.requestModels(), ","))
	}
}

func TestCanonicalModel_RunRootWrapperInheritsThroughStaticLink(t *testing.T) {
	restoreDefaults(t)
	fallback := &canonLLM{name: "fallback-model"}
	if err := llmagent.SetDefaultModel(fallback); err != nil {
		t.Fatal(err)
	}
	leaf := newLLMAgent(t, llmagent.Config{Name: "leaf"})
	custom, err := agent.New(agent.Config{
		Name:      "custom",
		SubAgents: []agent.Agent{leaf},
		Run:       func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] { return leaf.Run(ctx) },
	})
	if err != nil {
		t.Fatal(err)
	}
	wrapped := findAgentWrapper{custom}
	rootLLM := &canonLLM{name: "root-llm"}
	newLLMAgent(t, llmagent.Config{Name: "root", Model: rootLLM, SubAgents: []agent.Agent{wrapped}})

	runAgent(t, wrapped)

	if got, want := strings.Join(rootLLM.requestModels(), ","), "root-llm"; got != want {
		t.Errorf("root-llm request models = %q, want %q (leaf under the wrapped run root); fallback got %q", got, want, strings.Join(fallback.requestModels(), ","))
	}
}

func runAgent(t *testing.T, root agent.Agent) []*session.Event {
	t.Helper()
	return runAgentWithContent(t, root, genai.NewContentFromText("hi", genai.RoleUser))
}

func runAgentWithContent(t *testing.T, root agent.Agent, content *genai.Content) []*session.Event {
	t.Helper()
	ctx := t.Context()
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: root, SessionService: svc})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatal(err)
	}
	var events []*session.Event
	for ev, err := range r.Run(ctx, "u", resp.Session.ID(), content, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		events = append(events, ev)
	}
	return events
}

func transferTo(name string) *genai.Content {
	return &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{
		FunctionCall: &genai.FunctionCall{Name: "transfer_to_agent", Args: map[string]any{"agent_name": name}},
	}}}
}

func TestCanonicalModel_TwoParentsStaticVsRunTree(t *testing.T) {
	leaf := newLLMAgent(t, llmagent.Config{Name: "leaf", Description: "the leaf"})
	model1 := &canonLLM{name: "model-1", script: []*genai.Content{transferTo("leaf")}}
	parent1 := newLLMAgent(t, llmagent.Config{Name: "parent1", Model: model1, SubAgents: []agent.Agent{leaf}})
	model2 := &canonLLM{name: "model-2"}
	newLLMAgent(t, llmagent.Config{Name: "parent2", Model: model2, SubAgents: []agent.Agent{leaf}})

	if got := mustCanonicalModel(t, leaf); got != model2 {
		t.Errorf("CanonicalModel(leaf) outside a run = %s, want model-2 (the parent built last)", canonName(got))
	}

	events := runAgent(t, parent1)

	if got, want := strings.Join(model1.requestModels(), ","), "model-1,model-1"; got != want {
		t.Errorf("model-1 request models = %q, want %q (parent1, then leaf)", got, want)
	}
	if got := len(model2.requestModels()); got != 0 {
		t.Errorf("model-2 got %d requests, want 0", got)
	}
	if last := events[len(events)-1]; last.Author != "leaf" {
		t.Errorf("last event author = %q, want leaf", last.Author)
	}
}

func TestCanonicalModel_RunRootInheritsThroughStaticLink(t *testing.T) {
	leaf := newLLMAgent(t, llmagent.Config{Name: "leaf"})
	m := &canonLLM{name: "parent-model"}
	newLLMAgent(t, llmagent.Config{Name: "parent", Model: m, SubAgents: []agent.Agent{leaf}})

	runAgent(t, leaf)

	if got, want := strings.Join(m.requestModels(), ","), "parent-model"; got != want {
		t.Errorf("parent-model request models = %q, want %q", got, want)
	}
}

func TestCanonicalModel_BuiltInDefaultsAndDynamicUpdate(t *testing.T) {
	restoreDefaults(t)
	if llmagent.DefaultModel != "gemini-3.5-flash" {
		t.Errorf("DefaultModel = %q, want gemini-3.5-flash", llmagent.DefaultModel)
	}
	if llmagent.DefaultLiveModel != "gemini-live-2.5-flash-native-audio" {
		t.Errorf("DefaultLiveModel = %q, want gemini-live-2.5-flash-native-audio", llmagent.DefaultLiveModel)
	}
	a := newLLMAgent(t, llmagent.Config{Name: "a"})

	m := mustCanonicalModel(t, a)
	if !gemini.IsGeminiConnector(m) || m.Name() != "gemini-3.5-flash" {
		t.Errorf("CanonicalModel() = %s (Gemini connector: %v), want the Gemini connector gemini-3.5-flash", canonName(m), gemini.IsGeminiConnector(m))
	}
	live := mustCanonicalLiveModel(t, a)
	if !gemini.IsGeminiConnector(live) || live.Name() != "gemini-live-2.5-flash-native-audio" {
		t.Errorf("CanonicalLiveModel() = %s (Gemini connector: %v), want the Gemini connector gemini-live-2.5-flash-native-audio", canonName(live), gemini.IsGeminiConnector(live))
	}

	custom := &canonLLM{name: "custom-default"}
	customLive := &canonLLM{name: "custom-live-default"}
	if err := llmagent.SetDefaultModel(custom); err != nil {
		t.Fatal(err)
	}
	if err := llmagent.SetDefaultLiveModel(customLive); err != nil {
		t.Fatal(err)
	}
	if got := mustCanonicalModel(t, a); got != custom {
		t.Errorf("CanonicalModel() after SetDefaultModel = %s, want custom-default", canonName(got))
	}
	if got := mustCanonicalLiveModel(t, a); got != customLive {
		t.Errorf("CanonicalLiveModel() after SetDefaultLiveModel = %s, want custom-live-default", canonName(got))
	}
}

func TestSetDefault_RejectsEmptyNameAndNil(t *testing.T) {
	restoreDefaults(t)
	preserved := &canonLLM{name: "preserved-model"}
	preservedLive := &canonLLM{name: "preserved-live-model"}
	if err := llmagent.SetDefaultModel(preserved); err != nil {
		t.Fatal(err)
	}
	if err := llmagent.SetDefaultLiveModel(preservedLive); err != nil {
		t.Fatal(err)
	}

	if err := llmagent.SetDefaultModelByName(""); err == nil {
		t.Error(`SetDefaultModelByName("") = nil, want an error`)
	}
	if err := llmagent.SetDefaultLiveModelByName(""); err == nil {
		t.Error(`SetDefaultLiveModelByName("") = nil, want an error`)
	}
	if err := llmagent.SetDefaultModel(nil); err == nil {
		t.Error("SetDefaultModel(nil) = nil, want an error")
	}
	if err := llmagent.SetDefaultLiveModel(nil); err == nil {
		t.Error("SetDefaultLiveModel(nil) = nil, want an error")
	}

	a := newLLMAgent(t, llmagent.Config{Name: "a"})
	if got := mustCanonicalModel(t, a); got != preserved {
		t.Errorf("CanonicalModel() = %s, want preserved-model", canonName(got))
	}
	if got := mustCanonicalLiveModel(t, a); got != preservedLive {
		t.Errorf("CanonicalLiveModel() = %s, want preserved-live-model", canonName(got))
	}
}

func TestSetDefault_ConcurrentWithCanonicalModel(t *testing.T) {
	registerCanonTestPatterns(t)
	restoreDefaults(t)
	a := newLLMAgent(t, llmagent.Config{Name: "a"})
	objA := &canonLLM{name: "fr-obj-a"}
	objB := &canonLLM{name: "fr-obj-b"}

	var wg sync.WaitGroup
	for i := range 8 {
		wg.Go(func() {
			for j := range 50 {
				obj, name := objA, "fr-canon-plain-a"
				if (i+j)%2 == 1 {
					obj, name = objB, "fr-canon-plain-b"
				}
				switch j % 6 {
				case 0:
					_ = llmagent.SetDefaultModel(obj)
				case 1:
					_ = llmagent.SetDefaultModelByName(name)
				case 2:
					_ = llmagent.SetDefaultLiveModel(obj)
				case 3:
					_ = llmagent.SetDefaultLiveModelByName(name)
				case 4:
					if _, err := llmagent.CanonicalModel(a); err != nil {
						t.Errorf("CanonicalModel: %v", err)
					}
				case 5:
					if _, err := llmagent.CanonicalLiveModel(a); err != nil {
						t.Errorf("CanonicalLiveModel: %v", err)
					}
				}
			}
		})
	}
	wg.Wait()

	set := map[string]bool{"fr-obj-a": true, "fr-obj-b": true, "fr-canon-plain-a": true, "fr-canon-plain-b": true}
	if got := canonName(mustCanonicalModel(t, a)); !set[got] {
		t.Errorf("CanonicalModel() after the race = %s, want one of the values set", got)
	}
	if got := canonName(mustCanonicalLiveModel(t, a)); !set[got] {
		t.Errorf("CanonicalLiveModel() after the race = %s, want one of the values set", got)
	}
}

func TestDefaultModel_MemoizedUntilSetterCalled(t *testing.T) {
	registerCanonTestPatterns(t)
	restoreDefaults(t)
	// Names unique to this run, so that go test -count=2 starts from zero.
	run := fmt.Sprint(memoRuns.Add(1))
	name1, name2 := "fr-canon-memo-1-"+run, "fr-canon-memo-2-"+run
	live1, live2 := "fr-canon-memo-live-1-"+run, "fr-canon-memo-live-2-"+run
	a1 := newLLMAgent(t, llmagent.Config{Name: "a1"})
	a2 := newLLMAgent(t, llmagent.Config{Name: "a2"})

	if err := llmagent.SetDefaultModelByName(name1); err != nil {
		t.Fatal(err)
	}
	if err := llmagent.SetDefaultLiveModelByName(live1); err != nil {
		t.Fatal(err)
	}
	m1, m2, m3 := mustCanonicalModel(t, a1), mustCanonicalModel(t, a1), mustCanonicalModel(t, a2)
	if m1 != m2 || m2 != m3 {
		t.Errorf("CanonicalModel returned different objects for one default: %p %p %p", m1, m2, m3)
	}
	l1, l2, l3 := mustCanonicalLiveModel(t, a1), mustCanonicalLiveModel(t, a1), mustCanonicalLiveModel(t, a2)
	if l1 != l2 || l2 != l3 {
		t.Errorf("CanonicalLiveModel returned different objects for one default: %p %p %p", l1, l2, l3)
	}
	if got := memoCallCount(name1); got != 1 {
		t.Errorf("factory calls for the first default = %d, want 1", got)
	}
	if got := memoCallCount(live1); got != 1 {
		t.Errorf("factory calls for the first live default = %d, want 1", got)
	}

	if err := llmagent.SetDefaultModelByName(name2); err != nil {
		t.Fatal(err)
	}
	if err := llmagent.SetDefaultLiveModelByName(live2); err != nil {
		t.Fatal(err)
	}
	n1, n2 := mustCanonicalModel(t, a1), mustCanonicalModel(t, a2)
	if n1 != n2 || n1.Name() != name2 {
		t.Errorf("CanonicalModel after the setter = %s and %s, want one object named %s", canonName(n1), canonName(n2), name2)
	}
	k1, k2 := mustCanonicalLiveModel(t, a1), mustCanonicalLiveModel(t, a2)
	if k1 != k2 || k1.Name() != live2 {
		t.Errorf("CanonicalLiveModel after the setter = %s and %s, want one object named %s", canonName(k1), canonName(k2), live2)
	}
	if got := memoCallCount(name2); got != 1 {
		t.Errorf("factory calls for the second default = %d, want 1", got)
	}
	if got := memoCallCount(live2); got != 1 {
		t.Errorf("factory calls for the second live default = %d, want 1", got)
	}
}

func TestCanonicalInstruction_ProviderPrecedenceAndError(t *testing.T) {
	readonly := func(a agent.Agent) agent.ReadonlyContext {
		return icontext.NewReadonlyContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a}))
	}

	both := newLLMAgent(t, llmagent.Config{
		Name:        "both",
		Instruction: "static {var}",
		InstructionProvider: func(agent.ReadonlyContext) (string, error) {
			return "from provider {raw}", nil
		},
	})
	text, bypass, err := llmagent.CanonicalInstruction(readonly(both), both)
	if err != nil || text != "from provider {raw}" || !bypass {
		t.Errorf("CanonicalInstruction(both) = %q, %v, %v; want %q, true, nil", text, bypass, err, "from provider {raw}")
	}

	wantErr := errors.New("provider failed")
	failing := newLLMAgent(t, llmagent.Config{
		Name: "failing",
		InstructionProvider: func(agent.ReadonlyContext) (string, error) {
			return "ignored", wantErr
		},
	})
	text, bypass, err = llmagent.CanonicalInstruction(readonly(failing), failing)
	if !errors.Is(err, wantErr) || text != "" || bypass {
		t.Errorf("CanonicalInstruction(failing) = %q, %v, %v; want \"\", false, an error wrapping %v", text, bypass, err, wantErr)
	}

	static := newLLMAgent(t, llmagent.Config{Name: "static", Instruction: "static {var}"})
	text, bypass, err = llmagent.CanonicalInstruction(readonly(static), static)
	if err != nil || text != "static {var}" || bypass {
		t.Errorf("CanonicalInstruction(static) = %q, %v, %v; want %q, false, nil", text, bypass, err, "static {var}")
	}
}

func TestCanonicalModel_ConcurrentWithNewCoordinatorOverSubAgent(t *testing.T) {
	restoreDefaults(t)
	fallback := &canonLLM{name: "fallback-model"}
	if err := llmagent.SetDefaultModel(fallback); err != nil {
		t.Fatal(err)
	}
	if err := llmagent.SetDefaultLiveModel(fallback); err != nil {
		t.Fatal(err)
	}

	sub := newLLMAgent(t, llmagent.Config{Name: "worker"})

	var wg sync.WaitGroup
	for i := range 4 {
		wg.Go(func() {
			for j := range 20 {
				if _, err := llmagent.New(llmagent.Config{
					Name:      fmt.Sprintf("coord-%d-%d", i, j),
					Model:     &canonLLM{name: "coord-model"},
					SubAgents: []agent.Agent{sub},
				}); err != nil {
					t.Errorf("llmagent.New: %v", err)
				}
			}
		})
		wg.Go(func() {
			for range 20 {
				if _, err := llmagent.CanonicalModel(sub); err != nil {
					t.Errorf("CanonicalModel: %v", err)
				}
				if _, err := llmagent.CanonicalLiveModel(sub); err != nil {
					t.Errorf("CanonicalLiveModel: %v", err)
				}
			}
		})
		wg.Go(func() {
			for range 5 {
				runAgent(t, sub)
			}
		})
	}
	wg.Wait()
}
