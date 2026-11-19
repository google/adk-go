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

package redis_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/google/go-cmp/cmp"
	goredis "github.com/redis/go-redis/v9"

	"google.golang.org/adk/session/redis"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/sessiontestsuite"
)

// newTestService starts a fresh miniredis and returns a service over a
// caller-supplied client with cfg's other fields.
func newTestService(t *testing.T, cfg redis.Config) (*redis.Service, *miniredis.Miniredis, *goredis.Client) {
	t.Helper()
	mr := miniredis.RunT(t)
	client := newClient(t, mr)
	cfg.Client = client
	svc, err := redis.NewService(cfg)
	if err != nil {
		t.Fatalf("NewService() error = %v", err)
	}
	return svc, mr, client
}

func newClient(t *testing.T, mr *miniredis.Miniredis) *goredis.Client {
	t.Helper()
	client := goredis.NewClient(&goredis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = client.Close() })
	return client
}

func mustSet(t *testing.T, mr *miniredis.Miniredis, key, value string) {
	t.Helper()
	if err := mr.Set(key, value); err != nil {
		t.Fatalf("mr.Set(%q) error = %v", key, err)
	}
}

func atTime(ctx context.Context, ts time.Time) context.Context {
	return platform.WithTimeProvider(ctx, func() time.Time { return ts })
}

func mustCreate(t *testing.T, ctx context.Context, svc session.Service, app, user, id string, state map[string]any) session.Session {
	t.Helper()
	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: app, UserID: user, SessionID: id, State: state})
	if err != nil {
		t.Fatalf("Create(%q, %q, %q) error = %v", app, user, id, err)
	}
	return resp.Session
}

func mustGet(t *testing.T, ctx context.Context, svc session.Service, app, user, id string) session.Session {
	t.Helper()
	resp, err := svc.Get(ctx, &session.GetRequest{AppName: app, UserID: user, SessionID: id})
	if err != nil {
		t.Fatalf("Get(%q, %q, %q) error = %v", app, user, id, err)
	}
	return resp.Session
}

func mustAppend(t *testing.T, ctx context.Context, svc session.Service, sess session.Session, ev *session.Event) {
	t.Helper()
	if err := svc.AppendEvent(ctx, sess, ev); err != nil {
		t.Fatalf("AppendEvent(%q) error = %v", sess.ID(), err)
	}
}

func mustUserState(t *testing.T, ctx context.Context, svc *redis.Service, app, user string) map[string]any {
	t.Helper()
	resp, err := svc.GetUserState(ctx, &session.GetUserStateRequest{AppName: app, UserID: user})
	if err != nil {
		t.Fatalf("GetUserState(%q, %q) error = %v", app, user, err)
	}
	if resp.State == nil {
		t.Fatalf("GetUserState(%q, %q) State = nil, want a non-nil map", app, user)
	}
	return resp.State
}

func stateOf(s session.Session) map[string]any {
	return maps.Collect(s.State().All())
}

func deltaEvent(delta map[string]any) *session.Event {
	return &session.Event{Actions: session.EventActions{StateDelta: delta}}
}

// mapState is a session.State over a plain map.
type mapState map[string]any

func (m mapState) Get(k string) (any, error) {
	v, ok := m[k]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return v, nil
}

func (m mapState) Set(k string, v any) error {
	m[k] = v
	return nil
}

func (m mapState) All() iter.Seq2[string, any] { return maps.All(map[string]any(m)) }

// foreignSession is a session.Session the service did not create.
type foreignSession struct {
	appName, userID, id string
	state               session.State
}

func (f *foreignSession) ID() string                { return f.id }
func (f *foreignSession) AppName() string           { return f.appName }
func (f *foreignSession) UserID() string            { return f.userID }
func (f *foreignSession) State() session.State      { return f.state }
func (f *foreignSession) Events() session.Events    { return nil }
func (f *foreignSession) LastUpdateTime() time.Time { return time.Time{} }

// onFirstGet is a go-redis hook that runs fn once, right after the first
// GET of key completes. Armed by setting armed to true.
type onFirstGet struct {
	key   string
	fn    func()
	mu    sync.Mutex
	armed bool
	fired bool
}

func (h *onFirstGet) arm() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.armed = true
}

func (h *onFirstGet) DialHook(next goredis.DialHook) goredis.DialHook { return next }

func (h *onFirstGet) ProcessPipelineHook(next goredis.ProcessPipelineHook) goredis.ProcessPipelineHook {
	return next
}

func (h *onFirstGet) ProcessHook(next goredis.ProcessHook) goredis.ProcessHook {
	return func(ctx context.Context, cmd goredis.Cmder) error {
		err := next(ctx, cmd)
		args := cmd.Args()
		if cmd.Name() != "get" || len(args) < 2 || args[1] != h.key {
			return err
		}
		h.mu.Lock()
		run := h.armed && !h.fired
		h.fired = h.fired || run
		h.mu.Unlock()
		if run {
			h.fn()
		}
		return err
	}
}

func TestService_Conformance(t *testing.T) {
	sessiontestsuite.RunServiceTests(t, sessiontestsuite.SuiteOptions{
		SupportsUserProvidedSessionID: true,
		ChecksRequestedAuthConfigs:    true,
	}, func(t *testing.T) session.Service {
		svc, _, _ := newTestService(t, redis.Config{})
		return svc
	})
}

func TestNewService_LazyConnectionAndDefaults(t *testing.T) {
	ctx := t.Context()

	t.Run("does not dial", func(t *testing.T) {
		svc, err := redis.NewService(redis.Config{Host: "127.0.0.1", Port: 1})
		if err != nil {
			t.Fatalf("NewService() error = %v, want nil: construction must not dial", err)
		}
		t.Cleanup(func() { _ = svc.Close() })
		if _, err := svc.Get(ctx, &session.GetRequest{AppName: "a", UserID: "u", SessionID: "s"}); err == nil {
			t.Error("Get() error = nil, want a connection error from port 1")
		}
	})

	t.Run("default prefix and ttl", func(t *testing.T) {
		svc, mr, _ := newTestService(t, redis.Config{})
		mustCreate(t, ctx, svc, "a", "u", "s", map[string]any{"user:k": "v", "app:ak": "av"})
		for _, k := range []string{"adk:session:a:u:s", "adk:session:user_state:a:u", "adk:session:app_state:a"} {
			if !mr.Exists(k) {
				t.Errorf("key %q does not exist", k)
			}
			if got, want := mr.TTL(k), 604800*time.Second; got != want {
				t.Errorf("TTL(%q) = %v, want %v", k, got, want)
			}
		}
	})
}

func TestNewService_URIPrecedence(t *testing.T) {
	ctx := t.Context()
	mr := miniredis.RunT(t)
	svc, err := redis.NewService(redis.Config{URI: "redis://" + mr.Addr() + "/2", Host: "invalid.invalid", Port: 1})
	if err != nil {
		t.Fatalf("NewService() error = %v", err)
	}
	t.Cleanup(func() { _ = svc.Close() })

	mustCreate(t, ctx, svc, "a", "u", "s", nil)

	if !mr.DB(2).Exists("adk:session:a:u:s") {
		t.Error(`session key not in DB 2: the URI's database index was not used`)
	}
}

// newHostPortService returns a Service that builds its own client from
// cfg with mr's host and port, and closes it at the end of the test.
func newHostPortService(t *testing.T, mr *miniredis.Miniredis, cfg redis.Config) *redis.Service {
	t.Helper()
	port, err := strconv.Atoi(mr.Port())
	if err != nil {
		t.Fatalf("miniredis port %q: %v", mr.Port(), err)
	}
	cfg.Host, cfg.Port = mr.Host(), port
	svc, err := redis.NewService(cfg)
	if err != nil {
		t.Fatalf("NewService() error = %v", err)
	}
	t.Cleanup(func() { _ = svc.Close() })
	return svc
}

func TestNewService_HostPortPasswordDBAndSSL(t *testing.T) {
	t.Run("password", func(t *testing.T) {
		mr := miniredis.RunT(t)
		mr.RequireAuth("pw")
		svc := newHostPortService(t, mr, redis.Config{Password: "pw"})

		mustCreate(t, t.Context(), svc, "a", "u", "s", nil)

		if !mr.Exists("adk:session:a:u:s") {
			t.Errorf("keys after Create with the server's password = %v, want adk:session:a:u:s", mr.Keys())
		}
	})

	t.Run("db", func(t *testing.T) {
		mr := miniredis.RunT(t)
		svc := newHostPortService(t, mr, redis.Config{DB: 3})

		mustCreate(t, t.Context(), svc, "a", "u", "s", nil)

		if !mr.DB(3).Exists("adk:session:a:u:s") {
			t.Errorf("session key is not in database 3; keys in database 0: %v", mr.Keys())
		}
	})

	t.Run("ssl", func(t *testing.T) {
		// miniredis speaks plain text here, so a client that uses TLS fails
		// its handshake and writes nothing.
		mr := miniredis.RunT(t)
		svc := newHostPortService(t, mr, redis.Config{SSL: true})
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		defer cancel()

		_, err := svc.Create(ctx, &session.CreateRequest{AppName: "a", UserID: "u", SessionID: "s"})

		if err == nil {
			t.Error("Create() with SSL against a plain-text server error = nil, want a TLS error")
		}
		if keys := mr.Keys(); len(keys) != 0 {
			t.Errorf("keys after Create with SSL against a plain-text server = %v, want none", keys)
		}
	})
}

func TestService_Create_TempKeysOmittedFromRedis(t *testing.T) {
	ctx := t.Context()
	svc, mr, _ := newTestService(t, redis.Config{})

	got := mustCreate(t, ctx, svc, "a", "u", "s", map[string]any{"temp:scratch": 1})

	if diff := cmp.Diff(map[string]any{"temp:scratch": 1}, stateOf(got)); diff != "" {
		t.Errorf("Create() returned state mismatch (-want +got):\n%s", diff)
	}
	raw, err := mr.Get("adk:session:a:u:s")
	if err != nil {
		t.Fatalf("mr.Get(session key) error = %v", err)
	}
	var stored struct {
		State map[string]any `json:"state"`
	}
	if err := json.Unmarshal([]byte(raw), &stored); err != nil {
		t.Fatalf("stored session is not JSON: %v", err)
	}
	if diff := cmp.Diff(map[string]any{}, stored.State); diff != "" {
		t.Errorf("stored state mismatch (-want +got):\n%s", diff)
	}
}

func TestService_Create_DuplicateDoesNotModifyAppOrUserState(t *testing.T) {
	ctx := t.Context()
	svc, mr, _ := newTestService(t, redis.Config{})
	mustCreate(t, ctx, svc, "a", "u", "s1", map[string]any{"app:k": 1})

	_, err := svc.Create(ctx, &session.CreateRequest{AppName: "a", UserID: "u", SessionID: "s1", State: map[string]any{"app:k": 2}})

	if err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("duplicate Create() error = %v, want an already-exists error", err)
	}
	if got, err := mr.Get("adk:session:app_state:a"); err != nil || got != `{"k":1}` {
		t.Errorf("app state = %q (err %v), want %q", got, err, `{"k":1}`)
	}
}

func TestService_Create_ConcurrentDuplicateWatch(t *testing.T) {
	ctx := t.Context()
	svc, mr, client := newTestService(t, redis.Config{})
	svc2, err := redis.NewService(redis.Config{Client: newClient(t, mr)})
	if err != nil {
		t.Fatalf("NewService() error = %v", err)
	}
	var secondErr error
	hook := &onFirstGet{key: "adk:session:a:u:s1", fn: func() {
		_, secondErr = svc2.Create(ctx, &session.CreateRequest{AppName: "a", UserID: "u", SessionID: "s1", State: map[string]any{"winner": "second"}})
	}}
	client.AddHook(hook)
	hook.arm()

	_, err = svc.Create(ctx, &session.CreateRequest{AppName: "a", UserID: "u", SessionID: "s1", State: map[string]any{"winner": "first"}})

	if secondErr != nil {
		t.Fatalf("the competing Create() error = %v", secondErr)
	}
	if err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Errorf("Create() error = %v, want an already-exists error", err)
	}
	if got := stateOf(mustGet(t, ctx, svc2, "a", "u", "s1")); !cmp.Equal(got, map[string]any{"winner": "second"}) {
		t.Errorf("stored state = %v, want the competing Create's {winner: second}", got)
	}
}

func TestService_Create_KeepsConcurrentScopedStateWrite(t *testing.T) {
	tests := []struct {
		name      string
		scopedKey string
		state     map[string]any
		want      map[string]any
	}{
		{
			name:      "user_state",
			scopedKey: "adk:session:user_state:a:u",
			state:     map[string]any{"user:mine": "y"},
			want:      map[string]any{"user:other": "x", "user:mine": "y"},
		},
		{
			name:      "app_state",
			scopedKey: "adk:session:app_state:a",
			state:     map[string]any{"app:mine": "y"},
			want:      map[string]any{"app:other": "x", "app:mine": "y"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			svc, mr, client := newTestService(t, redis.Config{})
			other := newClient(t, mr)
			// Another client writes the scoped key after Create has read it
			// inside its transaction and before Create's EXEC.
			hook := &onFirstGet{key: tc.scopedKey, fn: func() {
				if err := other.Set(ctx, tc.scopedKey, `{"other":"x"}`, 0).Err(); err != nil {
					t.Errorf("concurrent writer Set() error = %v", err)
				}
			}}
			client.AddHook(hook)
			hook.arm()

			mustCreate(t, ctx, svc, "a", "u", "s1", tc.state)

			if diff := cmp.Diff(tc.want, stateOf(mustGet(t, ctx, svc, "a", "u", "s1"))); diff != "" {
				t.Errorf("Get() state mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDelete_PreservesAppAndUserState(t *testing.T) {
	ctx := t.Context()
	svc, _, _ := newTestService(t, redis.Config{})
	mustCreate(t, ctx, svc, "a", "u", "s1", map[string]any{"sk": "1", "user:uk": "uval", "app:ak": "aval"})

	if err := svc.Delete(ctx, &session.DeleteRequest{AppName: "a", UserID: "u", SessionID: "s1"}); err != nil {
		t.Fatalf("Delete() error = %v", err)
	}

	if diff := cmp.Diff(map[string]any{"uk": "uval"}, mustUserState(t, ctx, svc, "a", "u")); diff != "" {
		t.Errorf("GetUserState() after Delete mismatch (-want +got):\n%s", diff)
	}
	s2 := mustCreate(t, ctx, svc, "a", "u", "s2", nil)
	if diff := cmp.Diff(map[string]any{"user:uk": "uval", "app:ak": "aval"}, stateOf(s2)); diff != "" {
		t.Errorf("Create(s2) state after Delete mismatch (-want +got):\n%s", diff)
	}
}

func TestCreate_EmptyAppNameWritesNothing(t *testing.T) {
	ctx := t.Context()
	svc, mr, _ := newTestService(t, redis.Config{})

	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: "", UserID: "u", SessionID: "s", State: map[string]any{"app:k": "v", "user:k": "v"}})

	if err == nil {
		t.Errorf("Create(AppName: \"\") = %v, nil; want an error", resp)
	}
	if keys := mr.Keys(); len(keys) != 0 {
		t.Errorf("Redis keys after the rejected Create = %q, want none", keys)
	}
}

// adk-python's RedisSessionService validates no id, so the user "" is an
// ordinary user whose session key has an empty user segment.
func TestCreateGetDelete_EmptyUserIDAddressesUserLiterally(t *testing.T) {
	ctx := t.Context()
	svc, mr, _ := newTestService(t, redis.Config{})
	const key = "adk:session:a::s"

	created := mustCreate(t, ctx, svc, "a", "", "s", map[string]any{"k": "v"})
	if created.UserID() != "" {
		t.Errorf("Create() UserID = %q, want \"\"", created.UserID())
	}
	if !mr.Exists(key) {
		t.Fatalf("Redis keys after Create = %q, want %q", mr.Keys(), key)
	}

	got := mustGet(t, ctx, svc, "a", "", "s")
	if got.UserID() != "" || got.ID() != "s" {
		t.Errorf("Get() = user %q, id %q; want user \"\", id \"s\"", got.UserID(), got.ID())
	}
	if diff := cmp.Diff(map[string]any{"k": "v"}, stateOf(got)); diff != "" {
		t.Errorf("Get() state mismatch (-want +got):\n%s", diff)
	}

	if err := svc.Delete(ctx, &session.DeleteRequest{AppName: "a", UserID: "", SessionID: "s"}); err != nil {
		t.Fatalf("Delete() error = %v", err)
	}
	if mr.Exists(key) {
		t.Errorf("key %q still exists after Delete", key)
	}
}

// adk-python's RedisSessionService looks an empty session ID up like any
// other: get_session finds nothing and delete_session deletes nothing.
func TestGetDelete_EmptySessionID(t *testing.T) {
	ctx := t.Context()
	svc, _, _ := newTestService(t, redis.Config{})
	mustCreate(t, ctx, svc, "a", "u1", "s1", nil)

	if _, err := svc.Get(ctx, &session.GetRequest{AppName: "a", UserID: "u1", SessionID: ""}); !errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get(SessionID: \"\") error = %v, want one wrapping session.ErrNotFound", err)
	}
	if err := svc.Delete(ctx, &session.DeleteRequest{AppName: "a", UserID: "u1", SessionID: ""}); err != nil {
		t.Errorf("Delete(SessionID: \"\") error = %v, want nil", err)
	}
	if got := mustGet(t, ctx, svc, "a", "u1", "s1"); got.ID() != "s1" {
		t.Errorf("Get(s1) after Delete(SessionID: \"\") ID = %q, want \"s1\"", got.ID())
	}
	if _, err := svc.Get(ctx, &session.GetRequest{AppName: "", UserID: "u1", SessionID: "s1"}); err == nil || errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get(AppName: \"\") error = %v, want a validation error that does not wrap session.ErrNotFound", err)
	}
	if err := svc.Delete(ctx, &session.DeleteRequest{AppName: "", UserID: "u1", SessionID: "s1"}); err == nil {
		t.Error("Delete(AppName: \"\") error = nil, want a validation error")
	}
}

// The parts of a session key are joined with ":" unescaped, so app "a:b"
// with user "c" and app "a" with user "b:c" share the key of session "s".
// Get reports the identity the stored record holds, as adk-python's
// RedisSessionService does.
func TestGet_ReportsStoredIdentityWhenKeysCollide(t *testing.T) {
	ctx := t.Context()
	svc, _, _ := newTestService(t, redis.Config{})
	mustCreate(t, ctx, svc, "a:b", "c", "s", nil)

	got := mustGet(t, ctx, svc, "a", "b:c", "s")

	if got.AppName() != "a:b" || got.UserID() != "c" || got.ID() != "s" {
		t.Errorf("Get(a, b:c, s) = app %q, user %q, id %q; want app \"a:b\", user \"c\", id \"s\"", got.AppName(), got.UserID(), got.ID())
	}
}

func TestStateScopes_DynamicPropagationAndCumulativeUserState(t *testing.T) {
	ctx := t.Context()
	svc, _, _ := newTestService(t, redis.Config{})
	mustCreate(t, ctx, svc, "a", "u", "s1", map[string]any{"user:theme": "dark", "s1_k": "v1"})
	s2 := mustCreate(t, ctx, svc, "a", "u", "s2", map[string]any{"user:lang": "en", "s2_k": "v2"})
	if diff := cmp.Diff(map[string]any{"user:theme": "dark", "user:lang": "en", "s2_k": "v2"}, stateOf(s2)); diff != "" {
		t.Errorf("Create(s2) state mismatch (-want +got):\n%s", diff)
	}

	mustAppend(t, ctx, svc, s2, deltaEvent(map[string]any{"user:theme": "light", "app:mode": "fast"}))

	want := map[string]any{"s1_k": "v1", "user:theme": "light", "user:lang": "en", "app:mode": "fast"}
	if diff := cmp.Diff(want, stateOf(mustGet(t, ctx, svc, "a", "u", "s1"))); diff != "" {
		t.Errorf("Get(s1) state mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"theme": "light", "lang": "en"}, mustUserState(t, ctx, svc, "a", "u")); diff != "" {
		t.Errorf("GetUserState() mismatch (-want +got):\n%s", diff)
	}
}

func TestStateScopes_NilValuePreservedOnCreateAndAppend(t *testing.T) {
	ctx := t.Context()
	svc, _, _ := newTestService(t, redis.Config{})
	s1 := mustCreate(t, ctx, svc, "a", "u", "s1", map[string]any{"user:pref": "dark", "user:keep": "yes"})

	mustAppend(t, ctx, svc, s1, deltaEvent(map[string]any{"user:pref": nil}))

	want := map[string]any{"user:pref": nil, "user:keep": "yes"}
	listed, err := svc.List(ctx, &session.ListRequest{AppName: "a", UserID: "u"})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(listed.Sessions) != 1 {
		t.Fatalf("List() returned %d sessions, want 1", len(listed.Sessions))
	}
	for name, got := range map[string]session.Session{
		"held session": s1,
		"Get(s1)":      mustGet(t, ctx, svc, "a", "u", "s1"),
		"List()[0]":    listed.Sessions[0],
		"Create(s2)":   mustCreate(t, ctx, svc, "a", "u", "s2", nil),
	} {
		if diff := cmp.Diff(want, stateOf(got)); diff != "" {
			t.Errorf("%s state mismatch (-want +got):\n%s", name, diff)
		}
	}
	if diff := cmp.Diff(map[string]any{"pref": nil, "keep": "yes"}, mustUserState(t, ctx, svc, "a", "u")); diff != "" {
		t.Errorf("GetUserState() mismatch (-want +got):\n%s", diff)
	}
}

func TestGetUserState_IsolationDetachedSnapshotAndEmptyIDs(t *testing.T) {
	ctx := t.Context()
	svc, mr, _ := newTestService(t, redis.Config{})
	mustCreate(t, ctx, svc, "a", "u1", "s1", map[string]any{"user:k": map[string]any{"nested": "v1"}})
	mustSet(t, mr, "adk:session:user_state::u1", `{"k":"empty_app"}`)
	mustSet(t, mr, "adk:session:user_state:a:", `{"k":"empty_user"}`)

	tests := []struct {
		app, user string
		want      map[string]any
	}{
		{app: "a", user: "u2", want: map[string]any{}},
		{app: "other", user: "u1", want: map[string]any{}},
		{app: "", user: "u1", want: map[string]any{"k": "empty_app"}},
		{app: "a", user: "", want: map[string]any{"k": "empty_user"}},
	}
	for _, tc := range tests {
		if diff := cmp.Diff(tc.want, mustUserState(t, ctx, svc, tc.app, tc.user)); diff != "" {
			t.Errorf("GetUserState(%q, %q) mismatch (-want +got):\n%s", tc.app, tc.user, diff)
		}
	}

	first := mustUserState(t, ctx, svc, "a", "u1")
	first["k"].(map[string]any)["nested"] = "mutated"
	if diff := cmp.Diff(map[string]any{"k": map[string]any{"nested": "v1"}}, mustUserState(t, ctx, svc, "a", "u1")); diff != "" {
		t.Errorf("GetUserState() after mutating an earlier result mismatch (-want +got):\n%s", diff)
	}

	if resp, err := svc.GetUserState(ctx, nil); err == nil {
		t.Errorf("GetUserState(nil) = %v, nil; want an error", resp)
	}
}

func TestGet_NegativeNumRecentEventsIsAValidationError(t *testing.T) {
	ctx := t.Context()
	svc, mr, _ := newTestService(t, redis.Config{})
	mustCreate(t, ctx, svc, "a", "u", "s", nil)
	mustAppend(t, ctx, svc, mustGet(t, ctx, svc, "a", "u", "s"), &session.Event{ID: "e0", Timestamp: time.Unix(100, 0)})
	before, err := mr.Get("adk:session:a:u:s")
	if err != nil {
		t.Fatalf("mr.Get() error = %v", err)
	}

	resp, err := svc.Get(ctx, &session.GetRequest{AppName: "a", UserID: "u", SessionID: "s", NumRecentEvents: -1})

	if err == nil {
		t.Fatalf("Get(NumRecentEvents: -1) = %v, nil; want an error", resp)
	}
	if errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get(NumRecentEvents: -1) error = %v, want a validation error that does not wrap session.ErrNotFound", err)
	}
	if diff := cmp.Diff([]string{"adk:session:a:u:s"}, mr.Keys()); diff != "" {
		t.Errorf("Redis keys mismatch (-want +got):\n%s", diff)
	}
	if after, _ := mr.Get("adk:session:a:u:s"); after != before {
		t.Errorf("stored session changed by the rejected Get:\nbefore %s\nafter  %s", before, after)
	}
}

func TestGet_NumRecentEventsZeroAndFilteredAppendPreservesHistory(t *testing.T) {
	ctx := t.Context()
	svc, _, _ := newTestService(t, redis.Config{})
	s1 := mustCreate(t, ctx, svc, "a", "u", "s1", map[string]any{"orig": "v"})
	for i := range 4 {
		mustAppend(t, ctx, svc, s1, &session.Event{ID: "e" + strconv.Itoa(i), Timestamp: time.Unix(int64(100+i), 0)})
	}
	resp, err := svc.Get(ctx, &session.GetRequest{AppName: "a", UserID: "u", SessionID: "s1", NumRecentEvents: 2})
	if err != nil {
		t.Fatalf("Get(NumRecentEvents: 2) error = %v", err)
	}
	view := resp.Session
	if got := view.Events().Len(); got != 2 {
		t.Fatalf("filtered view has %d events, want 2", got)
	}

	mustAppend(t, ctx, svc, view, &session.Event{
		ID:        "e4",
		Timestamp: time.Unix(104, 0),
		Actions:   session.EventActions{StateDelta: map[string]any{"added": float64(1), "temp:scratch": "local"}},
	})

	if got := view.Events().Len(); got != 3 {
		t.Errorf("view has %d events after the append, want 3", got)
	}
	if got, err := view.State().Get("temp:scratch"); err != nil || got != "local" {
		t.Errorf(`view.State().Get("temp:scratch") = %v, %v; want "local"`, got, err)
	}
	full, err := svc.Get(ctx, &session.GetRequest{AppName: "a", UserID: "u", SessionID: "s1", NumRecentEvents: 0})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	var ids []string
	for ev := range full.Session.Events().All() {
		ids = append(ids, ev.ID)
	}
	if diff := cmp.Diff([]string{"e0", "e1", "e2", "e3", "e4"}, ids); diff != "" {
		t.Errorf("stored events mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"orig": "v", "added": float64(1)}, stateOf(full.Session)); diff != "" {
		t.Errorf("stored state mismatch (-want +got):\n%s", diff)
	}
	if e4 := full.Session.Events().At(4); e4 != nil {
		if diff := cmp.Diff(map[string]any{"added": float64(1)}, e4.Actions.StateDelta); diff != "" {
			t.Errorf("stored e4 state delta mismatch (-want +got):\n%s", diff)
		}
	}
}

func eventIDs(evs session.Events) []string {
	var ids []string
	for ev := range evs.All() {
		ids = append(ids, ev.ID)
	}
	return ids
}

func TestGet_NumRecentEventsThenAfter(t *testing.T) {
	ctx := t.Context()
	svc, _, _ := newTestService(t, redis.Config{})
	s1 := mustCreate(t, ctx, svc, "a", "u", "s1", nil)
	// Timestamps do not rise with append order, so trimming first and
	// filtering first keep different events.
	for i, sec := range []int64{200, 100, 160, 170} {
		mustAppend(t, ctx, svc, s1, &session.Event{ID: fmt.Sprintf("e%d", i), Timestamp: time.Unix(sec, 0)})
	}

	resp, err := svc.Get(ctx, &session.GetRequest{AppName: "a", UserID: "u", SessionID: "s1", NumRecentEvents: 3, After: time.Unix(150, 0)})
	if err != nil {
		t.Fatalf("Get(NumRecentEvents: 3, After: 150 s) error = %v", err)
	}

	if diff := cmp.Diff([]string{"e2", "e3"}, eventIDs(resp.Session.Events())); diff != "" {
		t.Errorf("Get(NumRecentEvents: 3, After: 150 s) events mismatch (-want +got):\n%s", diff)
	}
}

func TestGet_OmitEvents(t *testing.T) {
	ctx := t.Context()
	svc, _, _ := newTestService(t, redis.Config{})
	s1 := mustCreate(t, ctx, svc, "a", "u", "s1", map[string]any{"app:a": "av", "user:u": "uv", "sk": "sv"})
	mustAppend(t, ctx, svc, s1, &session.Event{ID: "e0", Timestamp: time.Unix(100, 0)})
	mustAppend(t, ctx, svc, s1, &session.Event{ID: "e1", Timestamp: time.Unix(101, 0)})

	resp, err := svc.Get(ctx, &session.GetRequest{AppName: "a", UserID: "u", SessionID: "s1", OmitEvents: true, NumRecentEvents: 1, After: time.Unix(100, 0)})
	if err != nil {
		t.Fatalf("Get(OmitEvents: true) error = %v", err)
	}
	if n := resp.Session.Events().Len(); n != 0 {
		t.Errorf("Get(OmitEvents: true) returned %d events, want 0", n)
	}
	if diff := cmp.Diff(map[string]any{"app:a": "av", "user:u": "uv", "sk": "sv"}, stateOf(resp.Session)); diff != "" {
		t.Errorf("Get(OmitEvents: true) state mismatch (-want +got):\n%s", diff)
	}

	if diff := cmp.Diff([]string{"e0", "e1"}, eventIDs(mustGet(t, ctx, svc, "a", "u", "s1").Events())); diff != "" {
		t.Errorf("Get() after the omitted read: events mismatch (-want +got):\n%s", diff)
	}

	_, err = svc.Get(ctx, &session.GetRequest{AppName: "a", UserID: "u", SessionID: "s1", OmitEvents: true, NumRecentEvents: -1})
	if err == nil || err.Error() != "num_recent_events must not be negative, got -1" || errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get(OmitEvents: true, NumRecentEvents: -1) error = %v, want \"num_recent_events must not be negative, got -1\" not wrapping session.ErrNotFound", err)
	}

	if _, err := svc.Get(ctx, &session.GetRequest{AppName: "a", UserID: "u", SessionID: "missing", OmitEvents: true}); !errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get(missing, OmitEvents: true) error = %v, want one wrapping session.ErrNotFound", err)
	}
}

func TestAppendEvent_StampsSessionWithEventTimestamp(t *testing.T) {
	tCreate := time.Date(2026, 3, 4, 5, 6, 7, 123456789, time.UTC)
	ctx := atTime(t.Context(), tCreate)
	svc, _, _ := newTestService(t, redis.Config{})

	held := mustCreate(t, ctx, svc, "a", "u", "s1", nil)

	if !held.LastUpdateTime().Equal(tCreate) {
		t.Errorf("Create() LastUpdateTime = %v, want %v", held.LastUpdateTime(), tCreate)
	}
	if got := mustGet(t, ctx, svc, "a", "u", "s1").LastUpdateTime(); !got.Equal(held.LastUpdateTime()) {
		t.Errorf("Get() LastUpdateTime = %v, held session has %v", got, held.LastUpdateTime())
	}
	for _, ts := range []time.Time{
		time.Date(2026, 3, 4, 5, 6, 8, 987654321, time.UTC),
		time.Unix(1700000010, 0).UTC(),
	} {
		mustAppend(t, ctx, svc, held, &session.Event{Timestamp: ts})

		if !held.LastUpdateTime().Equal(ts) {
			t.Errorf("after appending at %v, held LastUpdateTime = %v", ts, held.LastUpdateTime())
		}
		if got := mustGet(t, ctx, svc, "a", "u", "s1").LastUpdateTime(); !got.Equal(ts) {
			t.Errorf("after appending at %v, Get() LastUpdateTime = %v", ts, got)
		}
	}
}

func TestAppendEvent_UnknownDeletedOrExpiredSessionWritesNothing(t *testing.T) {
	never := func(*testing.T, *redis.Service, *miniredis.Miniredis) session.Session {
		return &foreignSession{appName: "a", userID: "u", id: "never_created", state: mapState{}}
	}
	tests := []struct {
		name    string
		cfg     redis.Config
		partial bool
		setup   func(t *testing.T, svc *redis.Service, mr *miniredis.Miniredis) session.Session
	}{
		{
			name:  "never created",
			setup: never,
		},
		{
			name:    "never created, partial event",
			partial: true,
			setup:   never,
		},
		{
			name: "deleted",
			setup: func(t *testing.T, svc *redis.Service, _ *miniredis.Miniredis) session.Session {
				s := mustCreate(t, t.Context(), svc, "a", "u", "s1", nil)
				if err := svc.Delete(t.Context(), &session.DeleteRequest{AppName: "a", UserID: "u", SessionID: "s1"}); err != nil {
					t.Fatalf("Delete() error = %v", err)
				}
				return s
			},
		},
		{
			name: "expired",
			cfg:  redis.Config{TTL: time.Hour},
			setup: func(t *testing.T, svc *redis.Service, mr *miniredis.Miniredis) session.Session {
				s := mustCreate(t, t.Context(), svc, "a", "u", "s1", nil)
				mr.FastForward(2 * time.Hour)
				return s
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			svc, mr, _ := newTestService(t, tc.cfg)
			sess := tc.setup(t, svc, mr)
			ev := deltaEvent(map[string]any{"sk": "1", "user:uk": "1", "app:ak": "1"})
			ev.Partial = tc.partial

			err := svc.AppendEvent(ctx, sess, ev)

			if !errors.Is(err, session.ErrNotFound) {
				t.Errorf("AppendEvent() error = %v, want an error wrapping session.ErrNotFound", err)
			}
			if keys := mr.Keys(); len(keys) != 0 {
				t.Errorf("Redis holds keys %q after a failed append, want none", keys)
			}
			if diff := cmp.Diff(map[string]any{}, mustUserState(t, ctx, svc, "a", "u")); diff != "" {
				t.Errorf("GetUserState() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAppendEvent_ForeignSessionUpdatesStateAndStoresEvent(t *testing.T) {
	ctx := t.Context()

	t.Run("caller-built value", func(t *testing.T) {
		svc, _, _ := newTestService(t, redis.Config{})
		mustCreate(t, ctx, svc, "a", "u", "s1", nil)
		foreign := &foreignSession{appName: "a", userID: "u", id: "s1", state: mapState{}}

		err := svc.AppendEvent(ctx, foreign, &session.Event{
			Author:       "user",
			InvocationID: "inv1",
			Timestamp:    time.Unix(1700000050, 0).UTC(),
			Actions:      session.EventActions{StateDelta: map[string]any{"sk": "v1", "user:uk": "uv1", "temp:tk": "tv1"}},
		})
		if err != nil {
			t.Fatalf("AppendEvent(foreign) error = %v", err)
		}

		if diff := cmp.Diff(map[string]any{"sk": "v1", "user:uk": "uv1", "temp:tk": "tv1"}, stateOf(foreign)); diff != "" {
			t.Errorf("foreign value state mismatch (-want +got):\n%s", diff)
		}
		stored := mustGet(t, ctx, svc, "a", "u", "s1")
		if diff := cmp.Diff(map[string]any{"sk": "v1", "user:uk": "uv1"}, stateOf(stored)); diff != "" {
			t.Errorf("stored state mismatch (-want +got):\n%s", diff)
		}
		if got := stored.Events().Len(); got != 1 {
			t.Fatalf("stored %d events, want 1", got)
		}
		if got := stored.Events().At(0).InvocationID; got != "inv1" {
			t.Errorf("stored event InvocationID = %q, want %q", got, "inv1")
		}
	})

	t.Run("CloneSession value", func(t *testing.T) {
		svc, _, _ := newTestService(t, redis.Config{})
		created := mustCreate(t, ctx, svc, "a", "u", "s1", nil)
		clone := session.CloneSession(created)
		ts := time.Unix(1700000060, 0).UTC()

		mustAppend(t, ctx, svc, clone, &session.Event{Timestamp: ts, Actions: session.EventActions{StateDelta: map[string]any{"sk": "v2"}}})

		if got := clone.Events().Len(); got != 1 {
			t.Errorf("clone has %d events, want 1", got)
		}
		if !clone.LastUpdateTime().Equal(ts) {
			t.Errorf("clone LastUpdateTime = %v, want %v", clone.LastUpdateTime(), ts)
		}
		if got, err := clone.State().Get("sk"); err != nil || got != "v2" {
			t.Errorf(`clone.State().Get("sk") = %v, %v; want "v2"`, got, err)
		}
	})

	t.Run("caller_built_nil_state", func(t *testing.T) {
		svc, _, _ := newTestService(t, redis.Config{})
		mustCreate(t, ctx, svc, "a", "u", "s2", nil)
		foreign := &foreignSession{appName: "a", userID: "u", id: "s2"}

		mustAppend(t, ctx, svc, foreign, &session.Event{ID: "e3", Actions: session.EventActions{StateDelta: map[string]any{"k": "v"}}})

		if diff := cmp.Diff(map[string]any{"k": "v"}, stateOf(mustGet(t, ctx, svc, "a", "u", "s2"))); diff != "" {
			t.Errorf("stored state mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("caller_built_state_set_fails", func(t *testing.T) {
		svc, _, _ := newTestService(t, redis.Config{})
		mustCreate(t, ctx, svc, "a", "u", "s3", nil)
		foreign := &foreignSession{appName: "a", userID: "u", id: "s3", state: refusingState{mapState{}}}

		err := svc.AppendEvent(ctx, foreign, &session.Event{ID: "e4", Actions: session.EventActions{StateDelta: map[string]any{"k": "v"}}})

		if !errors.Is(err, errSetRefused) {
			t.Errorf("AppendEvent() error = %v, want an error wrapping errSetRefused", err)
		}
		if diff := cmp.Diff([]string{"e4"}, eventIDs(mustGet(t, ctx, svc, "a", "u", "s3").Events())); diff != "" {
			t.Errorf("stored events mismatch (-want +got):\n%s", diff)
		}
	})
}

var errSetRefused = errors.New("set refused")

// refusingState is a session.State whose Set always fails.
type refusingState struct{ mapState }

func (refusingState) Set(string, any) error { return errSetRefused }

func TestAppendEvent_ConcurrentAppendsPreserveEventsAndScopedState(t *testing.T) {
	ctx := t.Context()
	svc, _, _ := newTestService(t, redis.Config{})
	s1 := mustCreate(t, ctx, svc, "a", "u", "s1", nil)
	s2 := mustCreate(t, ctx, svc, "a", "u", "s2", nil)
	s3 := mustCreate(t, ctx, svc, "a", "u", "s3", nil)

	// Appends to s1 share only the user state key with appends to s2, and
	// only the app state key with appends to s3, so each scoped key alone
	// must keep concurrent writers from losing each other's keys.
	var wg sync.WaitGroup
	errs := make(chan error, 30)
	for i := range 10 {
		wg.Go(func() {
			errs <- svc.AppendEvent(ctx, s1, deltaEvent(map[string]any{
				fmt.Sprintf("s_k_%d", i): i, fmt.Sprintf("user:u_k_%d", i): i, fmt.Sprintf("app:a_k_%d", i): i,
			}))
		})
		wg.Go(func() {
			errs <- svc.AppendEvent(ctx, s2, deltaEvent(map[string]any{fmt.Sprintf("user:u2_k_%d", i): i}))
		})
		wg.Go(func() {
			errs <- svc.AppendEvent(ctx, s3, deltaEvent(map[string]any{fmt.Sprintf("app:a3_k_%d", i): i}))
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent AppendEvent() error = %v", err)
		}
	}

	got := mustGet(t, ctx, svc, "a", "u", "s1")
	if n := got.Events().Len(); n != 10 {
		t.Errorf("s1 has %d events, want 10", n)
	}
	want := map[string]any{}
	for i := range 10 {
		for _, k := range []string{"s_k_%d", "user:u_k_%d", "user:u2_k_%d", "app:a_k_%d", "app:a3_k_%d"} {
			want[fmt.Sprintf(k, i)] = float64(i)
		}
	}
	if diff := cmp.Diff(want, stateOf(got)); diff != "" {
		t.Errorf("s1 state mismatch (-want +got):\n%s", diff)
	}
}

// afterExec is a go-redis hook that, once armed, runs fns[i] right after the
// i-th MULTI/EXEC pipeline completes, before the caller sees its result.
type afterExec struct {
	mu    sync.Mutex
	fns   []func()
	armed bool
}

func (h *afterExec) arm() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.armed = true
}

func (h *afterExec) DialHook(next goredis.DialHook) goredis.DialHook { return next }

func (h *afterExec) ProcessHook(next goredis.ProcessHook) goredis.ProcessHook { return next }

func (h *afterExec) ProcessPipelineHook(next goredis.ProcessPipelineHook) goredis.ProcessPipelineHook {
	return func(ctx context.Context, cmds []goredis.Cmder) error {
		err := next(ctx, cmds)
		if len(cmds) == 0 || cmds[len(cmds)-1].Name() != "exec" {
			return err
		}
		var fn func()
		h.mu.Lock()
		if h.armed && len(h.fns) > 0 {
			fn, h.fns = h.fns[0], h.fns[1:]
		}
		h.mu.Unlock()
		if fn != nil {
			fn()
		}
		return err
	}
}

func TestAppendEvent_ConcurrentAppendsReachHeldSessionInStoredOrder(t *testing.T) {
	ctx := t.Context()
	svc, _, client := newTestService(t, redis.Config{})
	held := mustCreate(t, ctx, svc, "app", "user", "s1", nil)
	evs := []*session.Event{
		{ID: "ev-1", Timestamp: time.Unix(100, 0).UTC(), Actions: session.EventActions{
			StateDelta: map[string]any{"k": "v1", "user:uk": "u1"},
		}},
		{ID: "ev-2", Timestamp: time.Unix(200, 0).UTC(), Actions: session.EventActions{
			StateDelta: map[string]any{"k": "v2", "user:uk": "u2"},
		}},
		{ID: "ev-3", Timestamp: time.Unix(300, 0).UTC(), Actions: session.EventActions{
			StateDelta: map[string]any{"k": "v3", "user:uk": "u3"},
		}},
	}

	// Each append but the last stops right after its transaction commits and
	// starts the next append through the same held session. It stays stopped
	// until the next append returns or, when the next append waits for it to
	// finish, for at most a second. Three appends make the third arrive while
	// the second holds the session after waiting for the first.
	errs := make(chan error, len(evs))
	done := []chan struct{}{make(chan struct{}), make(chan struct{}), make(chan struct{})}
	start := func(i int) {
		go func() {
			defer close(done[i])
			errs <- svc.AppendEvent(ctx, held, evs[i])
		}()
	}
	startAndWait := func(i int) func() {
		return func() {
			start(i)
			select {
			case <-done[i]:
			case <-time.After(time.Second):
			}
		}
	}
	hook := &afterExec{fns: []func(){startAndWait(1), startAndWait(2)}}
	client.AddHook(hook)
	hook.arm()

	start(0)
	for _, d := range done {
		<-d
	}
	for range evs {
		if err := <-errs; err != nil {
			t.Fatalf("AppendEvent() error = %v", err)
		}
	}

	stored := mustGet(t, ctx, svc, "app", "user", "s1")
	wantIDs := []string{"ev-1", "ev-2", "ev-3"}
	wantState := map[string]any{"k": "v3", "user:uk": "u3"}
	wantTime := time.Unix(300, 0).UTC()
	for _, tc := range []struct {
		name string
		sess session.Session
	}{{"stored", stored}, {"held", held}} {
		var ids []string
		for ev := range tc.sess.Events().All() {
			ids = append(ids, ev.ID)
		}
		if diff := cmp.Diff(wantIDs, ids); diff != "" {
			t.Errorf("%s event IDs mismatch (-want +got):\n%s", tc.name, diff)
		}
		if diff := cmp.Diff(wantState, stateOf(tc.sess)); diff != "" {
			t.Errorf("%s state mismatch (-want +got):\n%s", tc.name, diff)
		}
		if got := tc.sess.LastUpdateTime(); !got.Equal(wantTime) {
			t.Errorf("%s LastUpdateTime = %v, want %v", tc.name, got, wantTime)
		}
	}
}

// pausingState is a session.State over a map whose first Set runs pause
// before it stores the value.
type pausingState struct {
	pause  func()
	mu     sync.Mutex
	m      map[string]any
	paused bool
}

func (s *pausingState) Get(k string) (any, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	v, ok := s.m[k]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return v, nil
}

func (s *pausingState) Set(k string, v any) error {
	s.mu.Lock()
	first := !s.paused
	s.paused = true
	s.mu.Unlock()
	if first {
		s.pause()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m[k] = v
	return nil
}

func (s *pausingState) All() iter.Seq2[string, any] {
	s.mu.Lock()
	defer s.mu.Unlock()
	return maps.All(maps.Clone(s.m))
}

func TestAppendEvent_ConcurrentAppendsReachForeignSessionStateInStoredOrder(t *testing.T) {
	ctx := t.Context()
	svc, _, _ := newTestService(t, redis.Config{})
	mustCreate(t, ctx, svc, "app", "user", "s1", nil)
	st := &pausingState{m: map[string]any{}}
	held := &foreignSession{appName: "app", userID: "user", id: "s1", state: st}
	ev1 := &session.Event{ID: "ev-1", Timestamp: time.Unix(100, 0).UTC(), Actions: session.EventActions{
		StateDelta: map[string]any{"k": "v1"},
	}}
	ev2 := &session.Event{ID: "ev-2", Timestamp: time.Unix(200, 0).UTC(), Actions: session.EventActions{
		StateDelta: map[string]any{"k": "v2"},
	}}

	// ev-1's append stops after Redis has its event and before the held
	// state takes its delta, and ev-2's append starts then. It stays stopped
	// until ev-2's append returns or, when ev-2's append waits for it to
	// finish, for at most a second.
	errs := make(chan error, 2)
	secondDone := make(chan struct{})
	st.pause = func() {
		go func() {
			defer close(secondDone)
			errs <- svc.AppendEvent(ctx, held, ev2)
		}()
		select {
		case <-secondDone:
		case <-time.After(time.Second):
		}
	}

	errs <- svc.AppendEvent(ctx, held, ev1)
	<-secondDone
	for range 2 {
		if err := <-errs; err != nil {
			t.Fatalf("AppendEvent() error = %v", err)
		}
	}

	want := map[string]any{"k": "v2"}
	if diff := cmp.Diff(want, stateOf(mustGet(t, ctx, svc, "app", "user", "s1"))); diff != "" {
		t.Errorf("stored state mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(want, stateOf(held)); diff != "" {
		t.Errorf("held state mismatch (-want +got):\n%s", diff)
	}
}

func TestService_AppendEvent_DeletedDuringRetryReturnsNotFound(t *testing.T) {
	ctx := t.Context()
	svc, mr, client := newTestService(t, redis.Config{})
	s1 := mustCreate(t, ctx, svc, "a", "u", "s1", nil)
	other := newClient(t, mr)
	hook := &onFirstGet{key: "adk:session:a:u:s1", fn: func() { other.Del(ctx, "adk:session:a:u:s1") }}
	client.AddHook(hook)
	hook.arm()

	err := svc.AppendEvent(ctx, s1, deltaEvent(map[string]any{"user:uk": "1"}))

	if !errors.Is(err, session.ErrNotFound) {
		t.Errorf("AppendEvent() error = %v, want an error wrapping session.ErrNotFound", err)
	}
	for _, k := range []string{"adk:session:a:u:s1", "adk:session:user_state:a:u"} {
		if mr.Exists(k) {
			t.Errorf("key %q exists after the session was deleted mid-append", k)
		}
	}
}

func TestService_AppendEvent_PartialEventDoesNotWriteStateOrTimestamp(t *testing.T) {
	t0 := time.Unix(1700000000, 0).UTC()
	ctx := atTime(t.Context(), t0)
	svc, _, _ := newTestService(t, redis.Config{})
	s1 := mustCreate(t, ctx, svc, "a", "u", "s1", nil)

	err := svc.AppendEvent(ctx, s1, &session.Event{
		Timestamp:   t0.Add(time.Hour),
		LLMResponse: model.LLMResponse{Partial: true},
		Actions:     session.EventActions{StateDelta: map[string]any{"user:k": 1}},
	})
	if err != nil {
		t.Fatalf("AppendEvent(partial) error = %v", err)
	}

	if !s1.LastUpdateTime().Equal(t0) {
		t.Errorf("held LastUpdateTime = %v, want %v", s1.LastUpdateTime(), t0)
	}
	if diff := cmp.Diff(map[string]any{}, stateOf(s1)); diff != "" {
		t.Errorf("held state mismatch (-want +got):\n%s", diff)
	}
	if got := mustGet(t, ctx, svc, "a", "u", "s1").LastUpdateTime(); !got.Equal(t0) {
		t.Errorf("stored LastUpdateTime = %v, want %v", got, t0)
	}
	if diff := cmp.Diff(map[string]any{}, mustUserState(t, ctx, svc, "a", "u")); diff != "" {
		t.Errorf("GetUserState() mismatch (-want +got):\n%s", diff)
	}
}

func TestList_OrderingGlobEscapingPrefixGuardAndCorruptKeySkip(t *testing.T) {
	ctx := t.Context()
	svc, mr, _ := newTestService(t, redis.Config{KeyPrefix: "test:session:"})
	at := func(sec int64) context.Context { return atTime(ctx, time.Unix(sec, 0)) }
	mustCreate(t, at(20), svc, "app1", "u2", "s1", nil)
	mustCreate(t, at(20), svc, "app1", "u1", "s2", nil)
	s := mustCreate(t, at(20), svc, "app1", "u1", "s1", nil)
	mustAppend(t, ctx, svc, s, &session.Event{Timestamp: time.Unix(20, 0), Actions: session.EventActions{StateDelta: map[string]any{"user:lang": "en", "app:mode": "fast"}}})
	mustCreate(t, at(10), svc, "app1", "u2", "s0", nil)
	mustCreate(t, at(5), svc, "app1", "u1:sub", "s9", nil)
	mustSet(t, mr, "test:session:app1:u1:corrupt", "{bad")

	type row struct{ User, ID string }
	list := func(t *testing.T, app, user string) ([]row, []session.Session) {
		t.Helper()
		resp, err := svc.List(ctx, &session.ListRequest{AppName: app, UserID: user})
		if err != nil {
			t.Fatalf("List(%q, %q) error = %v", app, user, err)
		}
		var rows []row
		for _, s := range resp.Sessions {
			rows = append(rows, row{s.UserID(), s.ID()})
		}
		return rows, resp.Sessions
	}

	t.Run("all users", func(t *testing.T) {
		rows, sessions := list(t, "app1", "")
		want := []row{{"u1:sub", "s9"}, {"u2", "s0"}, {"u1", "s1"}, {"u1", "s2"}, {"u2", "s1"}}
		if diff := cmp.Diff(want, rows); diff != "" {
			t.Fatalf("List() order mismatch (-want +got):\n%s", diff)
		}
		for _, s := range sessions {
			if n := s.Events().Len(); n != 0 {
				t.Errorf("listed session %s/%s has %d events, want 0", s.UserID(), s.ID(), n)
			}
			if got, err := s.State().Get("app:mode"); err != nil || got != "fast" {
				t.Errorf(`listed session %s/%s state["app:mode"] = %v, %v; want "fast"`, s.UserID(), s.ID(), got, err)
			}
		}
	})

	t.Run("one user", func(t *testing.T) {
		rows, sessions := list(t, "app1", "u1")
		if diff := cmp.Diff([]row{{"u1", "s1"}, {"u1", "s2"}}, rows); diff != "" {
			t.Fatalf("List() mismatch (-want +got):\n%s", diff)
		}
		for _, s := range sessions {
			if got, err := s.State().Get("user:lang"); err != nil || got != "en" {
				t.Errorf(`listed session %s state["user:lang"] = %v, %v; want "en"`, s.ID(), got, err)
			}
		}
	})

	t.Run("glob characters match literally", func(t *testing.T) {
		for _, tc := range []struct{ app, user string }{
			{"app1", "*"}, {"app1", "u?"}, {"app1", "[u]1"}, {"app1", `u\1`}, {"*", ""},
		} {
			if rows, _ := list(t, tc.app, tc.user); len(rows) != 0 {
				t.Errorf("List(%q, %q) = %v, want no sessions", tc.app, tc.user, rows)
			}
		}
		mustCreate(t, ctx, svc, "app[1]?*", "u", "s", nil)
		mustCreate(t, ctx, svc, "app1", "u[1]", "sx", nil)
		if rows, _ := list(t, "app[1]?*", ""); !cmp.Equal(rows, []row{{"u", "s"}}) {
			t.Errorf(`List("app[1]?*", "") = %v, want [{u s}]`, rows)
		}
		if rows, _ := list(t, "app1", "u[1]"); !cmp.Equal(rows, []row{{"u[1]", "sx"}}) {
			t.Errorf(`List("app1", "u[1]") = %v, want [{u[1] sx}]`, rows)
		}
	})
}

// listed is the (UserID, ID) pair of a listed session.
type listed struct{ UserID, ID string }

func mustList(t *testing.T, ctx context.Context, svc session.Service, app, user string) []listed {
	t.Helper()
	resp, err := svc.List(ctx, &session.ListRequest{AppName: app, UserID: user})
	if err != nil {
		t.Fatalf("List(%q, %q) error = %v", app, user, err)
	}
	ids := []listed{}
	for _, s := range resp.Sessions {
		ids = append(ids, listed{s.UserID(), s.ID()})
	}
	return ids
}

// scanHook is a go-redis hook that passes every SCAN reply through edit.
type scanHook struct {
	edit func(*goredis.ScanCmd)
}

func (h *scanHook) DialHook(next goredis.DialHook) goredis.DialHook { return next }

func (h *scanHook) ProcessHook(next goredis.ProcessHook) goredis.ProcessHook {
	return func(ctx context.Context, cmd goredis.Cmder) error {
		err := next(ctx, cmd)
		if sc, ok := cmd.(*goredis.ScanCmd); ok && err == nil {
			h.edit(sc)
		}
		return err
	}
}

func (h *scanHook) ProcessPipelineHook(next goredis.ProcessPipelineHook) goredis.ProcessPipelineHook {
	return next
}

// SCAN can return a key more than once; adk-python then lists the session
// once per occurrence (_redis_session_service.py:294-295).
func TestList_ReturnsEachSessionOnceWhenScanRepeatsKeys(t *testing.T) {
	ctx := t.Context()
	svc, _, client := newTestService(t, redis.Config{})
	mustCreate(t, ctx, svc, "a", "u", "s1", nil)
	mustCreate(t, ctx, svc, "a", "u", "s2", nil)
	client.AddHook(&scanHook{edit: func(sc *goredis.ScanCmd) {
		page, cursor := sc.Val()
		sc.SetVal(append(page, page...), cursor)
	}})

	got := mustList(t, ctx, svc, "a", "u")

	if diff := cmp.Diff([]listed{{"u", "s1"}, {"u", "s2"}}, got); diff != "" {
		t.Errorf("List(a, u) mismatch (-want +got):\n%s", diff)
	}
}

func TestList_ExcludesKeysOfOtherAppsUnderTheSamePrefix(t *testing.T) {
	ctx := t.Context()
	svc, _, _ := newTestService(t, redis.Config{})
	// adk:session:a:b:u:s matches the pattern adk:session:a:*.
	mustCreate(t, ctx, svc, "a:b", "u", "s", nil)
	mustCreate(t, ctx, svc, "a", "u2", "s2", nil)
	// adk:session:app_state:x matches the pattern adk:session:app_state:*.
	mustCreate(t, ctx, svc, "x", "u", "s", map[string]any{"app:k": 1})

	if diff := cmp.Diff([]listed{{"u2", "s2"}}, mustList(t, ctx, svc, "a", "")); diff != "" {
		t.Errorf("List(a) mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]listed{}, mustList(t, ctx, svc, "app_state", "")); diff != "" {
		t.Errorf("List(app_state) mismatch (-want +got):\n%s", diff)
	}
}

// adk-python lists only the user "" when user_id is "" and not None
// (_redis_session_service.py:279-282, :303).
func TestList_FilterEmptyUserID(t *testing.T) {
	ctx := atTime(t.Context(), time.Unix(10, 0))
	svc, _, _ := newTestService(t, redis.Config{})
	mustCreate(t, ctx, svc, "a", "", "s_empty", map[string]any{"app:g": "gv", "user:role": "anon", "sk": "v_empty"})
	mustCreate(t, ctx, svc, "a", "u1", "s_u1", map[string]any{"user:role": "member", "sk": "v_u1"})
	// adk:session:a:x::s_collide starts with adk:session:a: and not with
	// adk:session:a::.
	mustCreate(t, ctx, svc, "a:x", "", "s_collide", map[string]any{"user:role": "other_app"})
	// adk:session:a:::u:s_colon starts with adk:session:a::, but the stored
	// user is ":u".
	mustCreate(t, ctx, svc, "a", ":u", "s_colon", nil)
	list := func(t *testing.T, req *session.ListRequest) ([]listed, []session.Session) {
		t.Helper()
		resp, err := svc.List(ctx, req)
		if err != nil {
			t.Fatalf("List(%+v) error = %v", req, err)
		}
		got := []listed{}
		for _, s := range resp.Sessions {
			got = append(got, listed{s.UserID(), s.ID()})
		}
		return got, resp.Sessions
	}

	got, sessions := list(t, &session.ListRequest{AppName: "a", UserID: "", FilterEmptyUserID: true})
	if diff := cmp.Diff([]listed{{"", "s_empty"}}, got); diff != "" {
		t.Fatalf("List(UserID: \"\", FilterEmptyUserID: true) mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"app:g": "gv", "user:role": "anon", "sk": "v_empty"}, stateOf(sessions[0])); diff != "" {
		t.Errorf("List(UserID: \"\", FilterEmptyUserID: true) state mismatch (-want +got):\n%s", diff)
	}

	got, _ = list(t, &session.ListRequest{AppName: "a", UserID: ""})
	if diff := cmp.Diff([]listed{{"", "s_empty"}, {":u", "s_colon"}, {"u1", "s_u1"}}, got); diff != "" {
		t.Errorf("List(UserID: \"\") mismatch (-want +got):\n%s", diff)
	}

	got, sessions = list(t, &session.ListRequest{AppName: "a", UserID: "u1", FilterEmptyUserID: true})
	if diff := cmp.Diff([]listed{{"u1", "s_u1"}}, got); diff != "" {
		t.Fatalf("List(UserID: \"u1\", FilterEmptyUserID: true) mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"app:g": "gv", "user:role": "member", "sk": "v_u1"}, stateOf(sessions[0])); diff != "" {
		t.Errorf("List(UserID: \"u1\", FilterEmptyUserID: true) state mismatch (-want +got):\n%s", diff)
	}
}

func TestList_EqualTimesOrderedBySessionIDWhateverTheScanOrder(t *testing.T) {
	ctx := t.Context()
	svc, _, client := newTestService(t, redis.Config{})
	// Sixteen sessions whose update times alternate against their IDs: an
	// order by time alone is free to interleave the IDs of equal times.
	for i := range 16 {
		at := atTime(ctx, time.Unix(int64(20-10*(i%2)), 0))
		mustCreate(t, at, svc, "a", "u", fmt.Sprintf("s%02d", i), nil)
	}
	// miniredis returns SCAN keys sorted; a real server returns them in hash
	// order, which this reversal stands in for.
	client.AddHook(&scanHook{edit: func(sc *goredis.ScanCmd) {
		page, cursor := sc.Val()
		slices.Reverse(page)
		sc.SetVal(page, cursor)
	}})

	got := mustList(t, ctx, svc, "a", "u")

	want := []listed{
		{"u", "s01"},
		{"u", "s03"},
		{"u", "s05"},
		{"u", "s07"},
		{"u", "s09"},
		{"u", "s11"},
		{"u", "s13"},
		{"u", "s15"},
		{"u", "s00"},
		{"u", "s02"},
		{"u", "s04"},
		{"u", "s06"},
		{"u", "s08"},
		{"u", "s10"},
		{"u", "s12"},
		{"u", "s14"},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("List(a, u) mismatch (-want +got):\n%s", diff)
	}
}

func TestService_Close_ClosesOwnedClientAndLeavesCallerClientOpen(t *testing.T) {
	ctx := t.Context()

	t.Run("owned client", func(t *testing.T) {
		mr := miniredis.RunT(t)
		port, err := strconv.Atoi(mr.Port())
		if err != nil {
			t.Fatalf("miniredis port %q: %v", mr.Port(), err)
		}
		svc, err := redis.NewService(redis.Config{Host: mr.Host(), Port: port})
		if err != nil {
			t.Fatalf("NewService() error = %v", err)
		}
		mustCreate(t, ctx, svc, "a", "u", "s1", nil)

		if err := svc.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}

		if _, err := svc.Get(ctx, &session.GetRequest{AppName: "a", UserID: "u", SessionID: "s1"}); !errors.Is(err, goredis.ErrClosed) {
			t.Errorf("Get() after Close error = %v, want goredis.ErrClosed", err)
		}
		if err := svc.Close(); err != nil {
			t.Errorf("second Close() error = %v, want nil", err)
		}
	})

	t.Run("caller client", func(t *testing.T) {
		svc, _, client := newTestService(t, redis.Config{})
		mustCreate(t, ctx, svc, "a", "u", "s1", nil)

		if err := svc.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}

		if got := client.Ping(ctx).Val(); got != "PONG" {
			t.Errorf("caller client Ping() = %q, want PONG", got)
		}
	})
}

func TestTTL_ExpiryAndSelectiveRefresh(t *testing.T) {
	ctx := t.Context()
	const (
		sessionKey = "adk:session:a:u:s1"
		userKey    = "adk:session:user_state:a:u"
		appKey     = "adk:session:app_state:a"
	)
	svc, mr, _ := newTestService(t, redis.Config{TTL: 60 * time.Second})
	ttls := func() map[string]time.Duration {
		return map[string]time.Duration{sessionKey: mr.TTL(sessionKey), userKey: mr.TTL(userKey), appKey: mr.TTL(appKey)}
	}

	s1 := mustCreate(t, ctx, svc, "a", "u", "s1", map[string]any{"sk": "1", "user:uk": "1", "app:ak": "1"})
	if diff := cmp.Diff(map[string]time.Duration{sessionKey: 60 * time.Second, userKey: 60 * time.Second, appKey: 60 * time.Second}, ttls()); diff != "" {
		t.Fatalf("TTLs after Create mismatch (-want +got):\n%s", diff)
	}

	mr.FastForward(40 * time.Second)
	mustGet(t, ctx, svc, "a", "u", "s1")
	mustUserState(t, ctx, svc, "a", "u")
	if diff := cmp.Diff(map[string]time.Duration{sessionKey: 20 * time.Second, userKey: 20 * time.Second, appKey: 20 * time.Second}, ttls()); diff != "" {
		t.Fatalf("TTLs after reads mismatch (-want +got):\n%s", diff)
	}

	mustAppend(t, ctx, svc, s1, deltaEvent(map[string]any{"sk": "2"}))
	if diff := cmp.Diff(map[string]time.Duration{sessionKey: 60 * time.Second, userKey: 20 * time.Second, appKey: 20 * time.Second}, ttls()); diff != "" {
		t.Fatalf("TTLs after a session-only append mismatch (-want +got):\n%s", diff)
	}

	mr.FastForward(25 * time.Second)
	if diff := cmp.Diff(map[string]any{}, mustUserState(t, ctx, svc, "a", "u")); diff != "" {
		t.Errorf("GetUserState() after user state expired mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"sk": "2"}, stateOf(mustGet(t, ctx, svc, "a", "u", "s1"))); diff != "" {
		t.Errorf("Get() state after scoped state expired mismatch (-want +got):\n%s", diff)
	}

	mr.FastForward(40 * time.Second)
	if _, err := svc.Get(ctx, &session.GetRequest{AppName: "a", UserID: "u", SessionID: "s1"}); !errors.Is(err, session.ErrNotFound) {
		t.Errorf("Get() after the session expired error = %v, want session.ErrNotFound", err)
	}
	if resp, err := svc.List(ctx, &session.ListRequest{AppName: "a", UserID: "u"}); err != nil || len(resp.Sessions) != 0 {
		t.Errorf("List() after the session expired = %v, %v; want no sessions", resp, err)
	}

	t.Run("negative ttl writes no expiry", func(t *testing.T) {
		svc, mr, _ := newTestService(t, redis.Config{TTL: -1})
		mustCreate(t, ctx, svc, "a", "u", "s1", map[string]any{"user:uk": "1"})
		for _, k := range []string{sessionKey, userKey} {
			if !mr.Exists(k) || mr.TTL(k) != 0 {
				t.Errorf("key %q: exists %v, TTL %v; want it to exist with no TTL", k, mr.Exists(k), mr.TTL(k))
			}
		}
	})
}
