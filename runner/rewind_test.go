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

package runner_test

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"iter"
	"maps"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

const (
	rewindApp     = "app"
	rewindUser    = "u1"
	rewindSession = "s1"
)

// seedEvent is one event appended to the test session before a rewind.
type seedEvent struct {
	invocationID string
	state        map[string]any
	artifacts    map[string]int64
}

// newRewindRunner returns a runner over sessions and artifacts whose root
// agent counts its runs.
func newRewindRunner(t *testing.T, sessions session.Service, artifacts artifact.Service, autoCreate bool) (*runner.Runner, *atomic.Int32) {
	t.Helper()
	runs := &atomic.Int32{}
	root, err := agent.New(agent.Config{
		Name: "root",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			runs.Add(1)
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           rewindApp,
		Agent:             root,
		SessionService:    sessions,
		ArtifactService:   artifacts,
		AutoCreateSession: autoCreate,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	return r, runs
}

// seedSession creates the test session with initialState and appends events,
// one minute apart from 1 January 2026.
func seedSession(t *testing.T, sessions session.Service, initialState map[string]any, events ...seedEvent) {
	t.Helper()
	ctx := t.Context()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: rewindApp, UserID: rewindUser, SessionID: rewindSession, State: initialState})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for i, e := range events {
		ev := session.NewEvent(ctx, e.invocationID)
		ev.Author = "user"
		ev.Timestamp = base.Add(time.Duration(i) * time.Minute)
		if e.state != nil {
			ev.Actions.StateDelta = e.state
		}
		if e.artifacts != nil {
			ev.Actions.ArtifactDelta = e.artifacts
		}
		if err := sessions.AppendEvent(ctx, created.Session, ev); err != nil {
			t.Fatalf("AppendEvent(%d): %v", i, err)
		}
	}
}

func getRewindSession(t *testing.T, sessions session.Service) session.Session {
	t.Helper()
	resp, err := sessions.Get(t.Context(), &session.GetRequest{AppName: rewindApp, UserID: rewindUser, SessionID: rewindSession})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	return resp.Session
}

func lastEvent(t *testing.T, sessions session.Service) *session.Event {
	t.Helper()
	events := getRewindSession(t, sessions).Events()
	if events.Len() == 0 {
		t.Fatal("session has no events")
	}
	return events.At(events.Len() - 1)
}

func TestRewind_AppendsUserEventAndRestoresSessionState(t *testing.T) {
	sessions := session.InMemoryService()
	r, agentRuns := newRewindRunner(t, sessions, nil, false)
	seedSession(t, sessions,
		map[string]any{"initial_only": "set at creation"},
		seedEvent{invocationID: "inv1", state: map[string]any{
			"k1":              "from_inv1",
			"unchanged":       "same",
			"obj":             map[string]any{"nested": []any{"a"}},
			"cleared_in_inv1": "c",
			"gone_before":     "g",
			"app:global":      "app_v1",
			"user:pref":       "user_v1",
			"temp:scratch":    "t",
		}},
		seedEvent{invocationID: "inv1", state: map[string]any{"cleared_in_inv1": nil, "gone_before": nil}},
		seedEvent{invocationID: "inv2", state: map[string]any{"k1": "from_inv2", "k2": "new", "cleared_in_inv1": "back", "user:pref": "user_v2"}, artifacts: map[string]int64{"doc.txt": 1}},
		seedEvent{invocationID: "inv2", state: map[string]any{"k1": "from_inv2b"}},
		seedEvent{invocationID: "inv3", state: map[string]any{"app:global": "app_v3"}},
	)

	var uuids int
	ctx := platform.WithUUIDProvider(t.Context(), func() string { uuids++; return fmt.Sprintf("uuid-%d", uuids) })
	rewindTime := time.Date(2026, 5, 1, 12, 0, 0, 0, time.UTC)
	ctx = platform.WithTimeProvider(ctx, func() time.Time { return rewindTime })

	if err := r.Rewind(ctx, rewindUser, rewindSession, "inv2", nil); err != nil {
		t.Fatalf("Rewind: %v", err)
	}

	got := lastEvent(t, sessions)
	if got.Author != "user" || got.InvocationID != "e-uuid-1" || !got.Timestamp.Equal(rewindTime) || got.Content != nil {
		t.Errorf("rewind event: Author %q, InvocationID %q, Timestamp %v, Content %v; want \"user\", \"e-uuid-1\", %v, nil",
			got.Author, got.InvocationID, got.Timestamp, got.Content, rewindTime)
	}
	if got.Actions.RewindBeforeInvocationID != "inv2" {
		t.Errorf("rewind event RewindBeforeInvocationID = %q, want \"inv2\"", got.Actions.RewindBeforeInvocationID)
	}
	// gone_before was removed before the rewind point but the session still
	// holds it with a nil value, so the delta sets it to nil again.
	wantStateDelta := map[string]any{"k1": "from_inv1", "k2": nil, "cleared_in_inv1": nil, "gone_before": nil, "initial_only": nil}
	if diff := cmp.Diff(wantStateDelta, got.Actions.StateDelta); diff != "" {
		t.Errorf("rewind event StateDelta mismatch (-want +got):\n%s", diff)
	}
	// With no artifact service, a rewind leaves artifacts alone.
	if diff := cmp.Diff(map[string]int64{}, got.Actions.ArtifactDelta, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("rewind event ArtifactDelta mismatch (-want +got):\n%s", diff)
	}

	wantState := map[string]any{
		"k1":              "from_inv1",
		"k2":              nil,
		"unchanged":       "same",
		"obj":             map[string]any{"nested": []any{"a"}},
		"cleared_in_inv1": nil,
		"gone_before":     nil,
		"initial_only":    nil,
		"app:global":      "app_v3",
		"user:pref":       "user_v2",
	}
	if diff := cmp.Diff(wantState, maps.Collect(getRewindSession(t, sessions).State().All())); diff != "" {
		t.Errorf("session state after rewind mismatch (-want +got):\n%s", diff)
	}
	if n := agentRuns.Load(); n != 0 {
		t.Errorf("root agent ran %d times, want 0", n)
	}
}

// tenfoldVersions reports every version of the wrapped service multiplied by
// ten, so a version Save returns is never the latest version plus one. It
// rejects FileData parts, which a rewind must not save, and counts saves.
type tenfoldVersions struct {
	artifact.Service
	saves int
}

func (s *tenfoldVersions) Save(ctx context.Context, req *artifact.SaveRequest) (*artifact.SaveResponse, error) {
	if req.Part != nil && req.Part.FileData != nil {
		return nil, errors.New("tenfoldVersions: FileData part saved")
	}
	s.saves++
	resp, err := s.Service.Save(ctx, req)
	if err != nil {
		return nil, err
	}
	return &artifact.SaveResponse{Version: resp.Version * 10}, nil
}

func (s *tenfoldVersions) Load(ctx context.Context, req *artifact.LoadRequest) (*artifact.LoadResponse, error) {
	inner := *req
	inner.Version /= 10
	return s.Service.Load(ctx, &inner)
}

func (s *tenfoldVersions) save(t *testing.T, fileName, text string) {
	t.Helper()
	if _, err := s.Save(t.Context(), &artifact.SaveRequest{AppName: rewindApp, UserID: rewindUser, SessionID: rewindSession, FileName: fileName, Part: genai.NewPartFromText(text)}); err != nil {
		t.Fatalf("Save(%s): %v", fileName, err)
	}
}

func (s *tenfoldVersions) loadLatest(t *testing.T, fileName string) (*genai.Part, error) {
	t.Helper()
	resp, err := s.Load(t.Context(), &artifact.LoadRequest{AppName: rewindApp, UserID: rewindUser, SessionID: rewindSession, FileName: fileName})
	if err != nil {
		return nil, err
	}
	return resp.Part, nil
}

func TestRewind_RestoresAndTombstonesSessionArtifacts(t *testing.T) {
	sessions := session.InMemoryService()
	artifacts := &tenfoldVersions{Service: artifact.InMemoryService()}
	r, _ := newRewindRunner(t, sessions, artifacts, false)
	artifacts.save(t, "doc.txt", "doc_v1")             // 10
	artifacts.save(t, "doc.txt", "doc_v2")             // 20
	artifacts.save(t, "unchanged.txt", "unchanged_v1") // 10
	artifacts.save(t, "user:avatar.png", "user_v1")    // 10
	artifacts.save(t, "user:avatar.png", "user_v2")    // 20
	artifacts.save(t, "added_in_inv2.txt", "added_v1") // 10
	artifacts.save(t, "gone.txt", "gone_v1")           // 10
	artifacts.save(t, "unrecorded.txt", "unrecorded")  // 10, in no event
	artifacts.save(t, "added_at_zero.txt", "zero_v1")  // 10
	seedSession(t, sessions, nil,
		// gone.txt records version 70, which the service never held.
		seedEvent{invocationID: "inv1", artifacts: map[string]int64{"doc.txt": 10, "unchanged.txt": 10, "user:avatar.png": 10, "gone.txt": 70}},
		// added_at_zero.txt records version 0, the first version of a service
		// that counts from 0 as adk-python's services do.
		seedEvent{invocationID: "inv2", artifacts: map[string]int64{"doc.txt": 20, "added_in_inv2.txt": 10, "user:avatar.png": 20, "gone.txt": 10, "added_at_zero.txt": 0}},
	)

	if err := r.Rewind(t.Context(), rewindUser, rewindSession, "inv2", nil); err != nil {
		t.Fatalf("Rewind: %v", err)
	}

	wantDelta := map[string]int64{"doc.txt": 30, "added_in_inv2.txt": 20, "gone.txt": 20, "added_at_zero.txt": 20}
	if diff := cmp.Diff(wantDelta, lastEvent(t, sessions).Actions.ArtifactDelta); diff != "" {
		t.Errorf("rewind event ArtifactDelta mismatch (-want +got):\n%s", diff)
	}
	got, err := artifacts.loadLatest(t, "doc.txt")
	if err != nil {
		t.Fatalf("Load(doc.txt): %v", err)
	}
	if diff := cmp.Diff(genai.NewPartFromText("doc_v1"), got); diff != "" {
		t.Errorf("Load(doc.txt) after rewind mismatch (-want +got):\n%s", diff)
	}
	for _, name := range []string{"added_in_inv2.txt", "gone.txt", "added_at_zero.txt"} {
		if part, err := artifacts.loadLatest(t, name); !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("Load(%s) after rewind = %v, %v; want an error wrapping fs.ErrNotExist", name, part, err)
		}
	}
}

// loadFails fails every Load with err.
type loadFails struct {
	artifact.Service
	err error
}

func (s loadFails) Load(context.Context, *artifact.LoadRequest) (*artifact.LoadResponse, error) {
	return nil, s.err
}

func TestRewind_ArtifactLoadErrorFailsRewind(t *testing.T) {
	sessions := session.InMemoryService()
	errStorage := errors.New("storage unavailable")
	r, _ := newRewindRunner(t, sessions, loadFails{Service: artifact.InMemoryService(), err: errStorage}, false)
	seedSession(t, sessions, nil,
		seedEvent{invocationID: "inv1", artifacts: map[string]int64{"doc.txt": 1}},
		seedEvent{invocationID: "inv2", artifacts: map[string]int64{"doc.txt": 2}},
	)

	err := r.Rewind(t.Context(), rewindUser, rewindSession, "inv2", nil)

	if !errors.Is(err, errStorage) {
		t.Errorf("Rewind() error = %v, want one wrapping %v", err, errStorage)
	}
	if n := getRewindSession(t, sessions).Events().Len(); n != 2 {
		t.Errorf("session has %d events after a failed rewind, want 2", n)
	}
}

// mapState is a session.State backed by a map.
type mapState map[string]any

func (s mapState) Get(key string) (any, error) {
	v, ok := s[key]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return v, nil
}

func (s mapState) Set(key string, value any) error {
	s[key] = value
	return nil
}

func (s mapState) All() iter.Seq2[string, any] { return maps.All(s) }

// sliceEvents is a session.Events backed by a slice.
type sliceEvents []*session.Event

func (e sliceEvents) All() iter.Seq[*session.Event] { return slices.Values(e) }
func (e sliceEvents) Len() int                      { return len(e) }
func (e sliceEvents) At(i int) *session.Event       { return e[i] }

// heldSession serves only State and Events; the embedded nil interface panics
// on any other method.
type heldSession struct {
	session.Session
	state  mapState
	events sliceEvents
}

func (s heldSession) State() session.State   { return s.state }
func (s heldSession) Events() session.Events { return s.events }

// heldSessionService returns sess from every Get and records appended events
// instead of storing them, so temp: keys, which the in-memory service strips,
// reach Rewind.
type heldSessionService struct {
	session.Service
	sess     heldSession
	appended []*session.Event
}

func (s *heldSessionService) Get(context.Context, *session.GetRequest) (*session.GetResponse, error) {
	return &session.GetResponse{Session: s.sess}, nil
}

func (s *heldSessionService) AppendEvent(_ context.Context, _ session.Session, ev *session.Event) error {
	s.appended = append(s.appended, ev)
	return nil
}

func TestRewind_SkipsTempKeys(t *testing.T) {
	inv1 := session.NewEvent(t.Context(), "inv1")
	inv1.Actions.StateDelta = map[string]any{"k": "v1", "temp:early": "t1"}
	inv2 := session.NewEvent(t.Context(), "inv2")
	inv2.Actions.StateDelta = map[string]any{"k": "v2", "temp:early": "t2", "temp:late": "x"}
	sessions := &heldSessionService{sess: heldSession{
		state:  mapState{"k": "v2", "temp:early": "t2", "temp:late": "x"},
		events: sliceEvents{inv1, inv2},
	}}
	r, _ := newRewindRunner(t, sessions, nil, false)

	if err := r.Rewind(t.Context(), rewindUser, rewindSession, "inv2", nil); err != nil {
		t.Fatalf("Rewind: %v", err)
	}

	if len(sessions.appended) != 1 {
		t.Fatalf("Rewind appended %d events, want 1", len(sessions.appended))
	}
	if diff := cmp.Diff(map[string]any{"k": "v1"}, sessions.appended[0].Actions.StateDelta); diff != "" {
		t.Errorf("rewind event StateDelta mismatch (-want +got):\n%s", diff)
	}
}

func TestRewind_InvocationNotFoundLeavesSessionAndArtifactsUntouched(t *testing.T) {
	sessions := session.InMemoryService()
	artifacts := &tenfoldVersions{Service: artifact.InMemoryService()}
	r, _ := newRewindRunner(t, sessions, artifacts, false)
	artifacts.save(t, "doc.txt", "doc_v1")
	seedSession(t, sessions, nil,
		seedEvent{invocationID: "inv1", state: map[string]any{"k1": "v1"}, artifacts: map[string]int64{"doc.txt": 10}},
		seedEvent{invocationID: "inv2", state: map[string]any{"k1": "v2"}},
	)
	savesBefore := artifacts.saves

	err := r.Rewind(t.Context(), rewindUser, rewindSession, "inv_nonexistent", nil)

	if !errors.Is(err, runner.ErrInvocationNotFound) {
		t.Fatalf("Rewind() error = %v, want one wrapping runner.ErrInvocationNotFound", err)
	}
	if got, want := err.Error(), "invocation ID not found: inv_nonexistent"; got != want {
		t.Errorf("Rewind() error text = %q, want %q", got, want)
	}
	if n := getRewindSession(t, sessions).Events().Len(); n != 2 {
		t.Errorf("session has %d events after a failed rewind, want 2", n)
	}
	if n := artifacts.saves - savesBefore; n != 0 {
		t.Errorf("failed rewind saved %d artifacts, want 0", n)
	}
}

func TestRewind_SessionLookupAndEventWindowConfig(t *testing.T) {
	t.Run("missing session without auto-create", func(t *testing.T) {
		r, _ := newRewindRunner(t, session.InMemoryService(), nil, false)
		err := r.Rewind(t.Context(), rewindUser, "missing", "inv1", nil)
		if !errors.Is(err, session.ErrNotFound) {
			t.Errorf("Rewind() error = %v, want one wrapping session.ErrNotFound", err)
		}
	})

	t.Run("missing session with auto-create", func(t *testing.T) {
		sessions := session.InMemoryService()
		r, _ := newRewindRunner(t, sessions, nil, true)
		err := r.Rewind(t.Context(), rewindUser, rewindSession, "inv_missing", nil)
		if !errors.Is(err, runner.ErrInvocationNotFound) {
			t.Fatalf("Rewind() error = %v, want one wrapping runner.ErrInvocationNotFound", err)
		}
		if n := getRewindSession(t, sessions).Events().Len(); n != 0 {
			t.Errorf("auto-created session has %d events, want 0", n)
		}
	})

	t.Run("event window", func(t *testing.T) {
		sessions := session.InMemoryService()
		r, _ := newRewindRunner(t, sessions, nil, false)
		seedSession(t, sessions, nil,
			seedEvent{invocationID: "inv1", state: map[string]any{"k1": "a"}},
			seedEvent{invocationID: "inv2", state: map[string]any{"k2": "b"}},
		)
		window := &agent.GetSessionConfig{NumRecentEvents: 1}

		if err := r.Rewind(t.Context(), rewindUser, rewindSession, "inv1", window); !errors.Is(err, runner.ErrInvocationNotFound) {
			t.Fatalf("Rewind(inv1, outside the window) error = %v, want one wrapping runner.ErrInvocationNotFound", err)
		}
		if err := r.Rewind(t.Context(), rewindUser, rewindSession, "inv2", window); err != nil {
			t.Fatalf("Rewind(inv2): %v", err)
		}
		// Only inv2's event is loaded, so nothing is known to precede it and k1
		// is removed too.
		want := map[string]any{"k1": nil, "k2": nil}
		if diff := cmp.Diff(want, lastEvent(t, sessions).Actions.StateDelta); diff != "" {
			t.Errorf("rewind event StateDelta mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestRewind_EmptyInvocationIDRejectedBeforeSessionLookup(t *testing.T) {
	sessions := session.InMemoryService()
	r, _ := newRewindRunner(t, sessions, nil, true)
	seedSession(t, sessions, nil, seedEvent{invocationID: "", state: map[string]any{"k": "v"}})

	for _, sessionID := range []string{rewindSession, "missing"} {
		err := r.Rewind(t.Context(), rewindUser, sessionID, "", nil)
		if !errors.Is(err, runner.ErrInvocationNotFound) {
			t.Errorf("Rewind(%q, \"\") error = %v, want one wrapping runner.ErrInvocationNotFound", sessionID, err)
		}
	}
	if n := getRewindSession(t, sessions).Events().Len(); n != 1 {
		t.Errorf("session has %d events, want 1", n)
	}
	_, err := sessions.Get(t.Context(), &session.GetRequest{AppName: rewindApp, UserID: rewindUser, SessionID: "missing"})
	if !errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get(missing) error = %v, want one wrapping session.ErrNotFound: the rejected rewind created the session", err)
	}
}
