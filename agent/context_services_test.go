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

package agent_test

import (
	"context"
	"errors"
	"io/fs"
	"iter"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
	icontext "google.golang.org/adk/v2/internal/context"
	imemory "google.golang.org/adk/v2/internal/memory"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
)

// foreignInvocation is an agent.InvocationContext written outside package
// agent, whose services come from its own fields, as the conformance Adapter's
// context does.
type foreignInvocation struct {
	agent.InvocationContext
	memory    agent.Memory
	artifacts agent.Artifacts
}

func (f *foreignInvocation) Memory() agent.Memory       { return f.memory }
func (f *foreignInvocation) Artifacts() agent.Artifacts { return f.artifacts }

// WithICDelta and WithContext keep the decorator, which embedding alone would drop.
func (f *foreignInvocation) WithICDelta(d *agent.InvocationContextDelta) agent.InvocationContext {
	return &foreignInvocation{InvocationContext: f.InvocationContext.WithICDelta(d), memory: f.memory, artifacts: f.artifacts}
}

func (f *foreignInvocation) WithContext(ctx context.Context) agent.InvocationContext {
	return &foreignInvocation{InvocationContext: f.InvocationContext.WithContext(ctx), memory: f.memory, artifacts: f.artifacts}
}

// services configures the invocations a test builds.
type services struct {
	goCtx     context.Context // nil means t.Context()
	agent     agent.Agent     // nil means newTestAgent
	memory    agent.Memory
	artifacts agent.Artifacts
	runConfig *agent.RunConfig
}

type ctxKey struct{}

func newTestSession(t *testing.T) session.Session {
	t.Helper()
	resp, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app-1", UserID: "user-1", SessionID: "sess-1"})
	if err != nil {
		t.Fatalf("session.Create() error = %v", err)
	}
	return resp.Session
}

func newTestAgent(t *testing.T) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: "test-agent",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	return a
}

type namedInvocation struct {
	name string
	ic   agent.InvocationContext
}

// newInvocations returns the two kinds of invocation a context can wrap: the
// one the runner and the LLM agent build, and one written outside the module.
func newInvocations(t *testing.T, sess session.Session, s services) []namedInvocation {
	t.Helper()
	goCtx := s.goCtx
	if goCtx == nil {
		goCtx = t.Context()
	}
	a := s.agent
	if a == nil {
		a = newTestAgent(t)
	}
	runnerIC := icontext.NewInvocationContext(goCtx, icontext.InvocationContextParams{
		Agent:        a,
		Session:      sess,
		Memory:       s.memory,
		Artifacts:    s.artifacts,
		RunConfig:    s.runConfig,
		InvocationID: "inv-1",
	})
	return []namedInvocation{
		{"runner invocation", runnerIC},
		{"foreign invocation", &foreignInvocation{InvocationContext: runnerIC, memory: s.memory, artifacts: s.artifacts}},
	}
}

type namedContext struct {
	name    string
	ctx     agent.Context
	actions *session.EventActions // the actions the context records into; nil when it records none
}

// newContexts returns every public way of building a Context over ic.
func newContexts(ic agent.InvocationContext) []namedContext {
	trackingActions := &session.EventActions{}
	toolActions := &session.EventActions{}
	return []namedContext{
		{name: "callback", ctx: agent.NewCallbackContext(ic, nil)},
		{name: "callback with artifact tracking", ctx: agent.NewCallbackContextWithArtifactTracking(ic, trackingActions), actions: trackingActions},
		{name: "tool", ctx: agent.NewToolContext(ic, "fc-1", toolActions, nil), actions: toolActions},
		{name: "promoted", ctx: agent.Promote(ic)},
	}
}

// forEachContext runs fn in a subtest for every context of newContexts over
// every invocation of newInvocations.
func forEachContext(t *testing.T, sess session.Session, s services, fn func(t *testing.T, c namedContext)) {
	t.Helper()
	for _, inv := range newInvocations(t, sess, s) {
		for _, c := range newContexts(inv.ic) {
			t.Run(inv.name+"/"+c.name, func(t *testing.T) { fn(t, c) })
		}
	}
}

// recordingMemory is an agent.Memory written outside the module, with no
// methods beyond the interface's.
type recordingMemory struct {
	sessions []session.Session
	err      error
}

func (m *recordingMemory) AddSessionToMemory(_ context.Context, s session.Session) error {
	m.sessions = append(m.sessions, s)
	return m.err
}

func (m *recordingMemory) SearchMemory(context.Context, string) (*memory.SearchResponse, error) {
	return &memory.SearchResponse{}, m.err
}

// recordingService is a memory.Service that implements memory.EventAdder and
// memory.MemoryAdder and records what reaches it.
type recordingService struct {
	addSessionCalls int
	eventsCalls     int
	eventsCtx       context.Context
	eventsReq       *memory.AddEventsRequest
	memoryCalls     int
	memoryReq       *memory.AddMemoryRequest
	err             error
	returnCtxErr    bool // return the error of the call's context instead of err
}

func (s *recordingService) result(ctx context.Context) error {
	if s.returnCtxErr {
		return ctx.Err()
	}
	return s.err
}

func (s *recordingService) AddSessionToMemory(ctx context.Context, _ session.Session) error {
	s.addSessionCalls++
	return s.result(ctx)
}

func (s *recordingService) SearchMemory(context.Context, *memory.SearchRequest) (*memory.SearchResponse, error) {
	return &memory.SearchResponse{}, nil
}

func (s *recordingService) AddEventsToMemory(ctx context.Context, req *memory.AddEventsRequest) error {
	s.eventsCalls++
	s.eventsCtx = ctx
	s.eventsReq = req
	return s.result(ctx)
}

func (s *recordingService) AddMemory(ctx context.Context, req *memory.AddMemoryRequest) error {
	s.memoryCalls++
	s.memoryReq = req
	return s.result(ctx)
}

// plainService is a memory.Service with neither optional interface.
type plainService struct {
	addSessionCalls int
}

func (s *plainService) AddSessionToMemory(context.Context, session.Session) error {
	s.addSessionCalls++
	return nil
}

func (s *plainService) SearchMemory(context.Context, *memory.SearchRequest) (*memory.SearchResponse, error) {
	return &memory.SearchResponse{}, nil
}

func TestContextArtifactsWithoutService(t *testing.T) {
	const want = "artifact service is not initialized"
	forEachContext(t, newTestSession(t), services{}, func(t *testing.T, c namedContext) {
		arts := c.ctx.Artifacts()
		if arts == nil {
			t.Fatal("Artifacts() = nil, want a value whose methods report the missing service")
		}
		saveResp, err := arts.Save(t.Context(), "a.txt", genai.NewPartFromText("x"))
		if err == nil || err.Error() != want || saveResp != nil {
			t.Errorf("Save() = %v, %v; want nil, %q", saveResp, err, want)
		}
		listResp, err := arts.List(t.Context())
		if err == nil || err.Error() != want || listResp != nil {
			t.Errorf("List() = %v, %v; want nil, %q", listResp, err, want)
		}
		loadResp, err := arts.Load(t.Context(), "a.txt")
		if err == nil || err.Error() != want || loadResp != nil {
			t.Errorf("Load() = %v, %v; want nil, %q", loadResp, err, want)
		}
		versionResp, err := arts.LoadVersion(t.Context(), "a.txt", 1)
		if err == nil || err.Error() != want || versionResp != nil {
			t.Errorf("LoadVersion() = %v, %v; want nil, %q", versionResp, err, want)
		}
		if c.actions != nil && len(c.actions.ArtifactDelta) != 0 {
			t.Errorf("ArtifactDelta = %v, want empty after a failed Save", c.actions.ArtifactDelta)
		}
	})
}

func TestContextMemoryIsTheConfiguredService(t *testing.T) {
	mem := &recordingMemory{}
	forEachContext(t, newTestSession(t), services{memory: mem}, func(t *testing.T, c namedContext) {
		if got := c.ctx.Memory(); got != agent.Memory(mem) {
			t.Errorf("Memory() = %T %p, want the invocation's memory %p", got, got, mem)
		}
	})
}

func TestCallbackContextExposesSession(t *testing.T) {
	sess := newTestSession(t)
	for _, inv := range newInvocations(t, sess, services{}) {
		for _, c := range newContexts(inv.ic) {
			if c.name == "tool" {
				// A tool context reports no session; AddSessionToMemory covers
				// what a tool needs the session for.
				continue
			}
			t.Run(inv.name+"/"+c.name, func(t *testing.T) {
				if got := c.ctx.Session(); got != sess {
					t.Errorf("Session() = %v, want the invocation's session %q", got, sess.ID())
				}
			})
		}
	}
}

func TestAddSessionToMemoryPassesTheInvocationSession(t *testing.T) {
	sess := newTestSession(t)
	mem := &recordingMemory{}
	forEachContext(t, sess, services{memory: mem}, func(t *testing.T, c namedContext) {
		mem.sessions = nil
		if err := agent.AddSessionToMemory(c.ctx); err != nil {
			t.Fatalf("AddSessionToMemory() error = %v", err)
		}
		if len(mem.sessions) != 1 || mem.sessions[0] != sess {
			t.Errorf("service received sessions %v, want exactly the invocation's session %q", mem.sessions, sess.ID())
		}
	})
}

// sessionOverride is an agent.Context written outside the module that wraps an
// ADK context and reports a session of its own.
type sessionOverride struct {
	agent.Context
	session session.Session
}

func (s *sessionOverride) Session() session.Session { return s.session }

// A Context written outside the module is read through its Session method, not
// through the ADK context it wraps. It fails if AddSessionToMemory asks every
// Context, rather than only ADK's own, for the session of the invocation it
// wraps.
func TestAddSessionToMemoryReadsAForeignContextsSession(t *testing.T) {
	sess := newTestSession(t)
	own, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app-1", UserID: "user-1", SessionID: "sess-own"})
	if err != nil {
		t.Fatalf("session.Create() error = %v", err)
	}
	mem := &recordingMemory{}
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: sess, Memory: mem})
	ctx := &sessionOverride{Context: agent.NewToolContext(ic, "fc-1", nil, nil), session: own.Session}

	if err := agent.AddSessionToMemory(ctx); err != nil {
		t.Fatalf("AddSessionToMemory() error = %v", err)
	}

	if len(mem.sessions) != 1 || mem.sessions[0] != own.Session {
		t.Errorf("service received sessions %v, want exactly the context's own session %q", mem.sessions, "sess-own")
	}
}

// decoratedContext is an agent.Context written outside the module that wraps an
// ADK context and overrides none of its methods, as a user's logging decorator
// does.
type decoratedContext struct {
	agent.Context
}

// A tool context reports no session through Session, and a decorator around it
// inherits that. It fails if AddSessionToMemory then passes the Context's
// Session instead of the session of the invocation the decorator wraps.
func TestAddSessionToMemoryThroughADecoratedToolContext(t *testing.T) {
	sess := newTestSession(t)
	mem := &recordingMemory{}
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: sess, Memory: mem})
	ctx := decoratedContext{Context: agent.NewToolContext(ic, "fc-1", nil, nil)}

	if err := agent.AddSessionToMemory(ctx); err != nil {
		t.Fatalf("AddSessionToMemory() error = %v", err)
	}

	if len(mem.sessions) != 1 || mem.sessions[0] != sess {
		t.Errorf("service received sessions %v, want exactly the invocation's session %q", mem.sessions, "sess-1")
	}
}

func TestAddSessionToMemoryWithoutService(t *testing.T) {
	const want = "cannot add session to memory: memory service is not available"
	sess := newTestSession(t)
	forEachContext(t, sess, services{}, func(t *testing.T, c namedContext) {
		if err := agent.AddSessionToMemory(c.ctx); err == nil || err.Error() != want {
			t.Errorf("agent.AddSessionToMemory() error = %v, want %q", err, want)
		}
		if err := c.ctx.Memory().AddSessionToMemory(c.ctx, sess); err == nil || err.Error() != want {
			t.Errorf("Memory().AddSessionToMemory() error = %v, want %q", err, want)
		}
	})
}

func TestAddEventsToMemoryWithoutService(t *testing.T) {
	const want = "cannot add events to memory: memory service is not available"
	events := []*session.Event{session.NewEvent(t.Context(), "inv-1"), session.NewEvent(t.Context(), "inv-1")}
	forEachContext(t, newTestSession(t), services{}, func(t *testing.T, c namedContext) {
		if err := agent.AddEventsToMemory(c.ctx, nil, nil); err == nil || err.Error() != want {
			t.Errorf("AddEventsToMemory(nil events) error = %v, want %q", err, want)
		}
		if err := agent.AddEventsToMemory(c.ctx, events, nil); err == nil || err.Error() != want {
			t.Errorf("AddEventsToMemory(2 events) error = %v, want %q", err, want)
		}
	})
}

func TestAddEventsToMemoryForwardsScope(t *testing.T) {
	e1 := session.NewEvent(t.Context(), "inv-1")
	e2 := session.NewEvent(t.Context(), "inv-1")
	svc := &recordingService{}
	s := services{
		goCtx:  context.WithValue(t.Context(), ctxKey{}, "v"),
		memory: &imemory.Memory{Service: svc, AppName: "wrapper-app", UserID: "wrapper-user", SessionID: "wrapper-sess"},
	}
	forEachContext(t, newTestSession(t), s, func(t *testing.T, c namedContext) {
		*svc = recordingService{}
		if err := agent.AddEventsToMemory(c.ctx, []*session.Event{e1, e2}, map[string]any{"ttl": "1h"}); err != nil {
			t.Fatalf("AddEventsToMemory() error = %v", err)
		}
		if svc.eventsCalls != 1 {
			t.Fatalf("service called %d times, want 1", svc.eventsCalls)
		}
		req := svc.eventsReq
		if req.AppName != "app-1" || req.UserID != "user-1" || req.SessionID != "sess-1" {
			t.Errorf("scope = %q, %q, %q; want \"app-1\", \"user-1\", \"sess-1\"", req.AppName, req.UserID, req.SessionID)
		}
		if len(req.Events) != 2 || req.Events[0] != e1 || req.Events[1] != e2 {
			t.Errorf("Events = %v, want the caller's two events in order", req.Events)
		}
		if diff := cmp.Diff(map[string]any{"ttl": "1h"}, req.CustomMetadata); diff != "" {
			t.Errorf("CustomMetadata mismatch (-want +got):\n%s", diff)
		}
		if got := svc.eventsCtx.Value(ctxKey{}); got != "v" {
			t.Errorf("service context value = %v, want \"v\" from the invocation's context", got)
		}
	})
}

func TestAddEventsToMemoryUnsupportedService(t *testing.T) {
	plain := &plainService{}
	tests := []struct {
		name   string
		memory agent.Memory
	}{
		{"service without memory.EventAdder", &imemory.Memory{Service: plain}},
		{"memory written outside the module", &recordingMemory{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ic := newInvocations(t, newTestSession(t), services{memory: tc.memory})[0].ic
			err := agent.AddEventsToMemory(agent.NewCallbackContext(ic, nil), nil, nil)
			if !errors.Is(err, memory.ErrAddEventsUnsupported) || !errors.Is(err, errors.ErrUnsupported) {
				t.Errorf("AddEventsToMemory() error = %v, want memory.ErrAddEventsUnsupported", err)
			}
			if plain.addSessionCalls != 0 {
				t.Errorf("AddSessionToMemory called %d times, want 0", plain.addSessionCalls)
			}
		})
	}
}

func TestAddMemoryWithoutService(t *testing.T) {
	const want = "cannot add memory: memory service is not available"
	forEachContext(t, newTestSession(t), services{}, func(t *testing.T, c namedContext) {
		if err := agent.AddMemory(c.ctx, nil, nil); err == nil || err.Error() != want {
			t.Errorf("AddMemory(nil memories) error = %v, want %q", err, want)
		}
		if err := agent.AddMemory(c.ctx, []*memory.Entry{{ID: "m1"}}, nil); err == nil || err.Error() != want {
			t.Errorf("AddMemory(1 entry) error = %v, want %q", err, want)
		}
	})
}

func TestAddMemoryForwardsScope(t *testing.T) {
	entry := &memory.Entry{ID: "m1"}
	svc := &recordingService{}
	s := services{memory: &imemory.Memory{Service: svc, AppName: "wrapper-app", UserID: "wrapper-user"}}
	forEachContext(t, newTestSession(t), s, func(t *testing.T, c namedContext) {
		*svc = recordingService{}
		if err := agent.AddMemory(c.ctx, []*memory.Entry{entry}, map[string]any{"k": "v"}); err != nil {
			t.Fatalf("AddMemory() error = %v", err)
		}
		if svc.memoryCalls != 1 {
			t.Fatalf("service called %d times, want 1", svc.memoryCalls)
		}
		req := svc.memoryReq
		if req.AppName != "app-1" || req.UserID != "user-1" {
			t.Errorf("scope = %q, %q; want \"app-1\", \"user-1\"", req.AppName, req.UserID)
		}
		if len(req.Memories) != 1 || req.Memories[0] != entry {
			t.Errorf("Memories = %v, want the caller's entry", req.Memories)
		}
		if diff := cmp.Diff(map[string]any{"k": "v"}, req.CustomMetadata); diff != "" {
			t.Errorf("CustomMetadata mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestAddMemoryUnsupportedService(t *testing.T) {
	tests := []struct {
		name   string
		memory agent.Memory
	}{
		{"in-memory service", &imemory.Memory{Service: memory.InMemoryService()}},
		{"memory written outside the module", &recordingMemory{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ic := newInvocations(t, newTestSession(t), services{memory: tc.memory})[0].ic
			err := agent.AddMemory(agent.NewToolContext(ic, "fc-1", nil, nil), []*memory.Entry{{ID: "m1"}}, nil)
			if !errors.Is(err, memory.ErrAddMemoryUnsupported) || !errors.Is(err, errors.ErrUnsupported) {
				t.Errorf("AddMemory() error = %v, want memory.ErrAddMemoryUnsupported", err)
			}
		})
	}
}

func TestContextMemoryServiceErrorsPassThrough(t *testing.T) {
	errBoom := errors.New("boom")
	calls := []struct {
		name  string
		call  func(agent.Context) error
		count func(*recordingService) int
	}{
		{"AddSessionToMemory", agent.AddSessionToMemory, func(s *recordingService) int { return s.addSessionCalls }},
		{"AddEventsToMemory", func(ctx agent.Context) error { return agent.AddEventsToMemory(ctx, nil, nil) }, func(s *recordingService) int { return s.eventsCalls }},
		{"AddMemory", func(ctx agent.Context) error { return agent.AddMemory(ctx, nil, nil) }, func(s *recordingService) int { return s.memoryCalls }},
	}
	for _, call := range calls {
		t.Run(call.name+"/service error", func(t *testing.T) {
			svc := &recordingService{err: errBoom}
			ic := newInvocations(t, newTestSession(t), services{memory: &imemory.Memory{Service: svc}})[0].ic
			if err := call.call(agent.NewCallbackContext(ic, nil)); !errors.Is(err, errBoom) {
				t.Errorf("error = %v, want the service's error", err)
			}
		})
		t.Run(call.name+"/canceled context", func(t *testing.T) {
			svc := &recordingService{returnCtxErr: true}
			goCtx, cancel := context.WithCancel(t.Context())
			cancel()
			ic := newInvocations(t, newTestSession(t), services{goCtx: goCtx, memory: &imemory.Memory{Service: svc}})[0].ic
			if err := call.call(agent.NewToolContext(ic, "fc-1", nil, nil)); !errors.Is(err, context.Canceled) {
				t.Errorf("error = %v, want context.Canceled from the service", err)
			}
			if got := call.count(svc); got != 1 {
				t.Errorf("service called %d times, want 1: the service decides what a canceled context means", got)
			}
		})
	}
}

func TestSearchMemoryWithoutService(t *testing.T) {
	const want = "memory service is not available"
	forEachContext(t, newTestSession(t), services{}, func(t *testing.T, c namedContext) {
		resp, err := c.ctx.SearchMemory(t.Context(), "test query")
		if err == nil || err.Error() != want || resp != nil {
			t.Errorf("SearchMemory() = %v, %v; want nil, %q", resp, err, want)
		}
	})
}

func TestBeforeAgentCallbackMemoryErrorEndsRun(t *testing.T) {
	const want = "failed to run before agent callback: cannot add session to memory: memory service is not available"
	bodyCalls := 0
	a, err := agent.New(agent.Config{
		Name: "test-agent",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(ctx agent.Context) (*genai.Content, error) {
			return nil, ctx.Memory().AddSessionToMemory(ctx, ctx.Session())
		}},
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				bodyCalls++
				yield(&session.Event{}, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	for _, inv := range newInvocations(t, newTestSession(t), services{agent: a}) {
		t.Run(inv.name, func(t *testing.T) {
			bodyCalls = 0
			// A caller stops at the first error, as the runner's callers do.
			for event, err := range a.Run(inv.ic) {
				if err == nil {
					t.Fatalf("Run() yielded event %v before any error, want the callback's error first", event)
				}
				if event != nil || err.Error() != want {
					t.Errorf("Run() yielded %v, %v; want nil, %q", event, err, want)
				}
				break
			}
			if bodyCalls != 0 {
				t.Errorf("agent body ran %d times, want 0", bodyCalls)
			}
		})
	}
}

func TestCustomMetadataReadsRunnerSnapshot(t *testing.T) {
	s := services{
		goCtx:     runconfig.ToContext(t.Context(), &runconfig.RunConfig{CustomMetadata: map[string]any{"snap": "1"}}),
		runConfig: &agent.RunConfig{CustomMetadata: &map[string]any{"raw": "2", "__adk_internal_x": "y"}},
	}
	forEachContext(t, newTestSession(t), s, func(t *testing.T, c namedContext) {
		if diff := cmp.Diff(map[string]any{"snap": "1"}, agent.CustomMetadata(c.ctx)); diff != "" {
			t.Errorf("CustomMetadata() mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestCustomMetadataFallsBackToFilteredRunConfig(t *testing.T) {
	want := map[string]any{"k": "v"}
	s := services{runConfig: &agent.RunConfig{CustomMetadata: &map[string]any{
		"k":                    "v",
		"__adk_internal_x":     "y",
		"adk:before_run_reply": "z",
		"adk.workflow.n":       "w",
	}}}
	sess := newTestSession(t)
	forEachContext(t, sess, s, func(t *testing.T, c namedContext) {
		if diff := cmp.Diff(want, agent.CustomMetadata(c.ctx)); diff != "" {
			t.Errorf("CustomMetadata() mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("instruction provider context", func(t *testing.T) {
		rc := icontext.NewReadonlyContext(newInvocations(t, sess, s)[0].ic)
		if diff := cmp.Diff(want, agent.CustomMetadata(rc)); diff != "" {
			t.Errorf("CustomMetadata() mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestCustomMetadataIsEmptyMapWhenNothingSet(t *testing.T) {
	tests := []struct {
		name  string
		goCtx context.Context
		rc    *agent.RunConfig
	}{
		{name: "no run config", rc: nil},
		{name: "nil metadata", rc: &agent.RunConfig{}},
		{name: "empty metadata", rc: &agent.RunConfig{CustomMetadata: &map[string]any{}}},
		{name: "only internal keys", rc: &agent.RunConfig{CustomMetadata: &map[string]any{"__adk_internal_x": "y"}}},
		{
			name:  "runner snapshot without metadata",
			goCtx: runconfig.ToContext(t.Context(), &runconfig.RunConfig{}),
			rc:    &agent.RunConfig{CustomMetadata: &map[string]any{"__adk_internal_x": "y"}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ic := newInvocations(t, newTestSession(t), services{goCtx: tc.goCtx, runConfig: tc.rc})[0].ic
			got := agent.CustomMetadata(agent.NewCallbackContext(ic, nil))
			if got == nil || len(got) != 0 {
				t.Errorf("CustomMetadata() = %#v, want an empty, non-nil map", got)
			}
		})
	}
}

func TestCustomMetadataReturnsIndependentCopy(t *testing.T) {
	rc := &runconfig.RunConfig{CustomMetadata: map[string]any{"a": 1}}
	s := services{goCtx: runconfig.ToContext(t.Context(), rc)}
	ctx := agent.NewToolContext(newInvocations(t, newTestSession(t), s)[0].ic, "fc-1", nil, nil)

	got := agent.CustomMetadata(ctx)
	delete(got, "a")
	got["b"] = 2

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			md := agent.CustomMetadata(ctx)
			md["c"] = 3
		})
	}
	wg.Wait()

	if diff := cmp.Diff(map[string]any{"a": 1}, agent.CustomMetadata(ctx)); diff != "" {
		t.Errorf("second CustomMetadata() mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"a": 1}, rc.CustomMetadata); diff != "" {
		t.Errorf("runner snapshot changed (-want +got):\n%s", diff)
	}
}

// bareReadonly is an agent.ReadonlyContext with no RunConfig method and no
// runner snapshot.
type bareReadonly struct {
	agent.ReadonlyContext
}

func (bareReadonly) Value(any) any { return nil }

func TestCustomMetadataOnReadonlyContextWithoutRunConfig(t *testing.T) {
	got := agent.CustomMetadata(bareReadonly{})
	if got == nil || len(got) != 0 {
		t.Errorf("CustomMetadata() = %#v, want an empty, non-nil map", got)
	}
}

// newArtifactStore returns an in-memory artifact service and the agent.Artifacts
// the runner builds over it for the session of newTestSession.
func newArtifactStore() (artifact.Service, agent.Artifacts) {
	store := artifact.InMemoryService()
	return store, &artifactinternal.Artifacts{Service: store, AppName: "app-1", UserID: "user-1", SessionID: "sess-1"}
}

func saveArtifact(t *testing.T, store artifact.Service, part *genai.Part, metadata map[string]any) {
	t.Helper()
	_, err := store.Save(t.Context(), &artifact.SaveRequest{
		AppName: "app-1", UserID: "user-1", SessionID: "sess-1", FileName: "report.txt",
		Part: part, CustomMetadata: metadata,
	})
	if err != nil {
		t.Fatalf("artifact Save() error = %v", err)
	}
}

// It fails if ArtifactVersion does not reach the invocation's artifact service
// through every kind of context, ignores the requested version, or drops fields
// of the record.
func TestArtifactVersion_LookupLatestAndSpecificVersion(t *testing.T) {
	store, arts := newArtifactStore()
	saveArtifact(t, store, genai.NewPartFromText("first"), map[string]any{"stage": "draft"})
	saveArtifact(t, store, genai.NewPartFromBytes([]byte("PDF"), "application/pdf"), map[string]any{"stage": "final"})
	ignoreCreateTime := cmpopts.IgnoreFields(artifact.ArtifactVersion{}, "CreateTime")
	wantLatest := &artifact.ArtifactVersion{
		Version:        2,
		CanonicalURI:   "memory://apps/app-1/users/user-1/sessions/sess-1/artifacts/report.txt/versions/2",
		CustomMetadata: map[string]any{"stage": "final"},
		MimeType:       "application/pdf",
	}
	wantFirst := &artifact.ArtifactVersion{
		Version:        1,
		CanonicalURI:   "memory://apps/app-1/users/user-1/sessions/sess-1/artifacts/report.txt/versions/1",
		CustomMetadata: map[string]any{"stage": "draft"},
		MimeType:       "text/plain",
	}
	v1 := int64(1)

	forEachContext(t, newTestSession(t), services{artifacts: arts}, func(t *testing.T, c namedContext) {
		got, err := agent.ArtifactVersion(c.ctx, "report.txt", nil)
		if err != nil {
			t.Fatalf("ArtifactVersion(nil) error = %v", err)
		}
		if diff := cmp.Diff(wantLatest, got, ignoreCreateTime); diff != "" {
			t.Errorf("ArtifactVersion(nil) mismatch (-want +got):\n%s", diff)
		}
		if got != nil && got.CreateTime.IsZero() {
			t.Error("ArtifactVersion(nil).CreateTime is zero, want the time of the save")
		}
		got, err = agent.ArtifactVersion(c.ctx, "report.txt", &v1)
		if err != nil {
			t.Fatalf("ArtifactVersion(1) error = %v", err)
		}
		if diff := cmp.Diff(wantFirst, got, ignoreCreateTime); diff != "" {
			t.Errorf("ArtifactVersion(1) mismatch (-want +got):\n%s", diff)
		}
		if c.actions != nil && len(c.actions.ArtifactDelta) != 0 {
			t.Errorf("ArtifactDelta = %v, want empty after a lookup", c.actions.ArtifactDelta)
		}
	})

	t.Run("concurrent", func(t *testing.T) {
		ic := newInvocations(t, newTestSession(t), services{artifacts: arts})[0].ic
		ctx := agent.NewToolContext(ic, "fc-1", &session.EventActions{}, nil)
		var wg sync.WaitGroup
		for range 8 {
			wg.Go(func() {
				if got, err := agent.ArtifactVersion(ctx, "report.txt", nil); err != nil || got == nil || got.Version != 2 {
					t.Errorf("ArtifactVersion(nil) = %v, %v; want version 2", got, err)
				}
				if got, err := agent.ArtifactVersion(ctx, "report.txt", &v1); err != nil || got == nil || got.Version != 1 {
					t.Errorf("ArtifactVersion(1) = %v, %v; want version 1", got, err)
				}
			})
		}
		wg.Wait()
	})
}

// countingArtifactService is an artifact.Service that answers every version
// lookup with a record and counts the lookups.
type countingArtifactService struct {
	artifact.Service
	mu      sync.Mutex
	lookups int
}

func (s *countingArtifactService) GetArtifactVersion(_ context.Context, req *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lookups++
	return &artifact.GetArtifactVersionResponse{ArtifactVersion: &artifact.ArtifactVersion{Version: req.Version}}, nil
}

// It fails if ArtifactVersion reports a found record for a missing artifact or
// version, treats a version of 0 or less as the latest or passes it to the
// service, or reports a malformed name as not found.
func TestArtifactVersion_NotFoundAndNonPositiveVersions(t *testing.T) {
	store, arts := newArtifactStore()
	saveArtifact(t, store, genai.NewPartFromText("first"), nil)
	ic := newInvocations(t, newTestSession(t), services{artifacts: arts})[0].ic
	ctx := agent.NewToolContext(ic, "fc-1", &session.EventActions{}, nil)
	v0, v2, vNeg := int64(0), int64(2), int64(-1)

	for _, tc := range []struct {
		name     string
		filename string
		version  *int64
	}{
		{name: "missing artifact", filename: "missing.txt"},
		{name: "missing version", filename: "report.txt", version: &v2},
		{name: "version 0", filename: "report.txt", version: &v0},
		{name: "negative version", filename: "report.txt", version: &vNeg},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := agent.ArtifactVersion(ctx, tc.filename, tc.version)
			if got != nil || !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("ArtifactVersion() = %v, %v; want nil and an error wrapping fs.ErrNotExist", got, err)
			}
		})
	}

	for _, tc := range []struct {
		name     string
		filename string
	}{
		{name: "empty name", filename: ""},
		{name: "name with a path separator", filename: "dir/report.txt"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := agent.ArtifactVersion(ctx, tc.filename, nil)
			if got != nil || err == nil || errors.Is(err, fs.ErrNotExist) {
				t.Errorf("ArtifactVersion() = %v, %v; want nil and an error not wrapping fs.ErrNotExist", got, err)
			}
		})
	}

	t.Run("non-positive version does not call the service", func(t *testing.T) {
		counting := &countingArtifactService{}
		ic := newInvocations(t, newTestSession(t), services{artifacts: &artifactinternal.Artifacts{
			Service: counting, AppName: "app-1", UserID: "user-1", SessionID: "sess-1",
		}})[0].ic
		ctx := agent.NewToolContext(ic, "fc-1", &session.EventActions{}, nil)
		for _, version := range []*int64{&v0, &vNeg} {
			got, err := agent.ArtifactVersion(ctx, "report.txt", version)
			if got != nil || !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("ArtifactVersion(%d) = %v, %v; want nil and an error wrapping fs.ErrNotExist", *version, got, err)
			}
		}
		if counting.lookups != 0 {
			t.Errorf("service received %d lookups, want 0", counting.lookups)
		}
	})
}

// stubArtifacts is an agent.Artifacts written outside the module, with no
// methods beyond the interface's.
type stubArtifacts struct{}

func (stubArtifacts) Save(context.Context, string, *genai.Part) (*artifact.SaveResponse, error) {
	return &artifact.SaveResponse{}, nil
}

func (stubArtifacts) List(context.Context) (*artifact.ListResponse, error) {
	return &artifact.ListResponse{}, nil
}

func (stubArtifacts) Load(context.Context, string) (*artifact.LoadResponse, error) {
	return &artifact.LoadResponse{}, nil
}

func (stubArtifacts) LoadVersion(context.Context, string, int) (*artifact.LoadResponse, error) {
	return &artifact.LoadResponse{}, nil
}

// nilArtifactsContext is an agent.Context written outside the module whose
// invocation has no artifacts at all.
type nilArtifactsContext struct {
	agent.Context
}

func (nilArtifactsContext) Artifacts() agent.Artifacts { return nil }

// It fails if ArtifactVersion checks the version before the missing service,
// or does not report artifacts without version lookup as unsupported.
func TestArtifactVersion_WithoutServiceAndUnsupported(t *testing.T) {
	const want = "artifact service is not initialized"
	v0, vNeg, v1 := int64(0), int64(-1), int64(1)
	versions := []struct {
		name    string
		version *int64
	}{{"latest", nil}, {"0", &v0}, {"-1", &vNeg}, {"1", &v1}}
	checkNoService := func(t *testing.T, ctx agent.Context) {
		t.Helper()
		for _, v := range versions {
			got, err := agent.ArtifactVersion(ctx, "a.txt", v.version)
			if got != nil || err == nil || err.Error() != want || errors.Is(err, errors.ErrUnsupported) {
				t.Errorf("ArtifactVersion(version %s) = %v, %v; want nil, %q", v.name, got, err, want)
			}
		}
	}

	t.Run("no service", func(t *testing.T) {
		forEachContext(t, newTestSession(t), services{}, func(t *testing.T, c namedContext) {
			checkNoService(t, c.ctx)
		})
	})
	t.Run("nil artifacts", func(t *testing.T) {
		ic := newInvocations(t, newTestSession(t), services{})[0].ic
		checkNoService(t, nilArtifactsContext{Context: agent.NewToolContext(ic, "fc-1", nil, nil)})
	})
	t.Run("unsupported", func(t *testing.T) {
		forEachContext(t, newTestSession(t), services{artifacts: stubArtifacts{}}, func(t *testing.T, c namedContext) {
			got, err := agent.ArtifactVersion(c.ctx, "a.txt", nil)
			if got != nil || !errors.Is(err, errors.ErrUnsupported) {
				t.Errorf("ArtifactVersion() = %v, %v; want nil and an error wrapping errors.ErrUnsupported", got, err)
			}
		})
	})
}

// ctxErrArtifactService is an artifact.Service that answers a version lookup
// with the error of the call's context, and with a record when it has none.
type ctxErrArtifactService struct {
	artifact.Service
}

func (ctxErrArtifactService) GetArtifactVersion(ctx context.Context, req *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &artifact.GetArtifactVersionResponse{ArtifactVersion: &artifact.ArtifactVersion{Version: 1}}, nil
}

// It fails if ArtifactVersion calls the service with a context other than the
// caller's, so that a service honoring cancellation answers a canceled lookup
// as if it were live.
func TestArtifactVersion_CanceledContextReachesTheService(t *testing.T) {
	goCtx, cancel := context.WithCancel(t.Context())
	cancel()
	arts := &artifactinternal.Artifacts{Service: ctxErrArtifactService{}, AppName: "app-1", UserID: "user-1", SessionID: "sess-1"}
	forEachContext(t, newTestSession(t), services{goCtx: goCtx, artifacts: arts}, func(t *testing.T, c namedContext) {
		got, err := agent.ArtifactVersion(c.ctx, "report.txt", nil)
		if got != nil || !errors.Is(err, context.Canceled) {
			t.Errorf("ArtifactVersion() = %v, %v; want nil and context.Canceled from the service", got, err)
		}
	})
}
