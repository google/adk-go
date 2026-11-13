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
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

// foreignInvocation is an agent.InvocationContext written outside package
// agent, whose services come from its own fields, as the conformance Adapter's
// context does.
type foreignInvocation struct {
	agent.InvocationContext
	artifacts agent.Artifacts
}

func (f *foreignInvocation) Artifacts() agent.Artifacts { return f.artifacts }

// WithICDelta and WithContext keep the decorator, which embedding alone would drop.
func (f *foreignInvocation) WithICDelta(d *agent.InvocationContextDelta) agent.InvocationContext {
	return &foreignInvocation{InvocationContext: f.InvocationContext.WithICDelta(d), artifacts: f.artifacts}
}

func (f *foreignInvocation) WithContext(ctx context.Context) agent.InvocationContext {
	return &foreignInvocation{InvocationContext: f.InvocationContext.WithContext(ctx), artifacts: f.artifacts}
}

// services configures the invocations a test builds.
type services struct {
	goCtx     context.Context // nil means t.Context()
	artifacts agent.Artifacts
}

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
	runnerIC := icontext.NewInvocationContext(goCtx, icontext.InvocationContextParams{
		Agent:        newTestAgent(t),
		Session:      sess,
		Artifacts:    s.artifacts,
		InvocationID: "inv-1",
	})
	return []namedInvocation{
		{"runner invocation", runnerIC},
		{"foreign invocation", &foreignInvocation{InvocationContext: runnerIC, artifacts: s.artifacts}},
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
