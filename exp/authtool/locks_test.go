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

package authtool_test

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/v2/exp/authtool"
	"google.golang.org/adk/v2/tool/toolauth"
)

// getConcurrently runs GetAuthCredential once per context, all at once, and
// returns the access tokens and errors in the order of ctxs.
func getConcurrently(ms []*authtool.CredentialManager, ctxs []*fakeToolContext) ([]string, []error) {
	tokens := make([]string, len(ctxs))
	errs := make([]error, len(ctxs))
	var wg sync.WaitGroup
	for i := range ctxs {
		wg.Go(func() {
			cred, err := ms[i].GetAuthCredential(ctxs[i])
			tokens[i], errs[i] = accessToken(cred), err
		})
	}
	wg.Wait()
	return tokens, errs
}

// Test 35. adk-python serializes calls per app, user and credential key, and
// the second caller reuses the saved credential (credential_manager.py:107-131,
// :259-279). Fails if the lock is per call, or released before the save.
func TestGetAuthCredential_SameKeySerialized(t *testing.T) {
	ts := newTokenServer(t)
	ts.set(func(s *tokenServer) { s.delay = 100 * time.Millisecond })
	m := newManager(t, ccConfig(ts.URL))
	svc := newFakeService()
	var ms []*authtool.CredentialManager
	var ctxs []*fakeToolContext
	for range 8 {
		ms = append(ms, m)
		ctxs = append(ctxs, newToolContext(t, svc))
	}

	tokens, errs := getConcurrently(ms, ctxs)
	for i := range tokens {
		if errs[i] != nil || tokens[i] != "cc-access" {
			t.Errorf("call %d = %s, %v, want cc-access, nil", i, tokens[i], errs[i])
		}
	}
	if n := len(ts.seen()); n != 1 {
		t.Errorf("token requests = %d, want 1", n)
	}
}

// Test 36. Without a credential service every call exchanges, one at a time.
// Fails if calls for one key overlap.
func TestGetAuthCredential_SameKeyWithoutServiceNeverOverlaps(t *testing.T) {
	ts := newTokenServer(t)
	ts.set(func(s *tokenServer) { s.delay = 50 * time.Millisecond })
	m := newManager(t, ccConfig(ts.URL))
	ms := []*authtool.CredentialManager{m, m, m}
	ctxs := []*fakeToolContext{newToolContext(t, nil), newToolContext(t, nil), newToolContext(t, nil)}

	tokens, errs := getConcurrently(ms, ctxs)
	for i := range tokens {
		if errs[i] != nil || tokens[i] != "cc-access" {
			t.Errorf("call %d = %s, %v, want cc-access, nil", i, tokens[i], errs[i])
		}
	}
	if n := len(ts.seen()); n != 3 {
		t.Errorf("token requests = %d, want 3", n)
	}
	if p := ts.peak(); p != 1 {
		t.Errorf("token requests in flight at once = %d, want 1", p)
	}
}

// Test 37. Fails if one lock covers different keys, users or apps.
func TestGetAuthCredential_DifferentKeysAndUsersRunConcurrently(t *testing.T) {
	tests := []struct {
		name        string
		secondKey   string
		secondUser  string
		secondApp   string
		sameManager bool
	}{
		{name: "two keys", secondKey: "key-2", secondUser: "user-1", secondApp: "app-1"},
		{name: "two users", secondKey: "key-1", secondUser: "user-2", secondApp: "app-1", sameManager: true},
		{name: "two apps", secondKey: "key-1", secondUser: "user-1", secondApp: "app-2", sameManager: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := newTokenServer(t)
			ts.set(func(s *tokenServer) { s.holdUntil = 2 })
			first := newManager(t, ccConfig(ts.URL))
			second := first
			if !tt.sameManager {
				cfg := ccConfig(ts.URL)
				cfg.CredentialKey = tt.secondKey
				second = newManager(t, cfg)
			}
			ctx2 := newToolContext(t, nil)
			ctx2.userID, ctx2.appName = tt.secondUser, tt.secondApp

			tokens, errs := getConcurrently([]*authtool.CredentialManager{first, second}, []*fakeToolContext{newToolContext(t, nil), ctx2})
			for i := range tokens {
				if errs[i] != nil || tokens[i] != "cc-access" {
					t.Errorf("call %d = %s, %v, want cc-access, nil", i, tokens[i], errs[i])
				}
			}
			if p := ts.peak(); p != 2 {
				t.Errorf("token requests in flight at once = %d, want 2", p)
			}
		})
	}
}

// Test 38. Fails if waiting for the lock ignores the caller's context, as a
// plain sync.Mutex would.
func TestGetAuthCredential_CancelledWhileWaiting(t *testing.T) {
	ts := newTokenServer(t)
	release := make(chan struct{})
	ts.set(func(s *tokenServer) { s.release = release })
	released := sync.OnceFunc(func() { close(release) })
	t.Cleanup(released)
	m := newManager(t, ccConfig(ts.URL))

	type result struct {
		cred *toolauth.Credential
		err  error
	}
	first := make(chan result, 1)
	go func() {
		cred, err := m.GetAuthCredential(newToolContext(t, nil))
		first <- result{cred, err}
	}()
	select {
	case <-ts.arrived:
	case <-time.After(5 * time.Second):
		t.Fatal("first call never reached the token endpoint")
	}

	ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()
	second := make(chan result, 1)
	go func() {
		cred, err := m.GetAuthCredential(newToolContextFrom(t, ctx, nil))
		second <- result{cred, err}
	}()
	select {
	case r := <-second:
		if r.cred != nil || !errors.Is(r.err, context.DeadlineExceeded) {
			t.Errorf("second call = %s, %v, want nil and context.DeadlineExceeded", accessToken(r.cred), r.err)
		}
	case <-time.After(time.Second):
		t.Error("second call still waits a second after its context ended")
	}

	released()
	r := <-first
	if r.err != nil || accessToken(r.cred) != "cc-access" {
		t.Errorf("first call = %s, %v, want cc-access, nil", accessToken(r.cred), r.err)
	}
}

// Test 39. Fails if the lock table keeps an entry per app, user and key after
// the call ends.
func TestGetAuthCredential_LockEntriesAreReleased(t *testing.T) {
	m := newManager(t, apiKeyConfig())
	ctx := newToolContext(t, nil)
	var before, after runtime.MemStats
	runtime.GC()
	runtime.GC()
	runtime.ReadMemStats(&before)

	for i := range 20_000 {
		ctx.userID = fmt.Sprintf("user-%d", i)
		if _, err := m.GetAuthCredential(ctx); err != nil {
			t.Fatalf("GetAuthCredential() error = %v", err)
		}
	}
	runtime.GC()
	runtime.GC()
	runtime.ReadMemStats(&after)
	if grew := int64(after.HeapObjects) - int64(before.HeapObjects); grew > 5_000 {
		t.Errorf("heap objects grew by %d after 20000 calls, want at most 5000", grew)
	}
}

// Test 40. Fails under -race if the exchanger registry has no guard, or if a
// registration is not seen by a call that starts after it returns.
func TestCredentialManager_RegisterDuringUse(t *testing.T) {
	ts := newTokenServer(t)
	m := newManager(t, ccConfig(ts.URL))
	fake := &countingExchanger{result: oauthCred(toolauth.OAuth2Auth{AccessToken: "fake-access"}), didExchange: true}

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			if _, err := m.GetAuthCredential(newToolContext(t, nil)); err != nil {
				t.Errorf("GetAuthCredential() error = %v", err)
			}
		})
	}
	wg.Go(func() { m.RegisterCredentialExchanger(toolauth.CredentialTypeOAuth2, fake) })
	wg.Wait()

	before := fake.count()
	cred, err := m.GetAuthCredential(newToolContext(t, nil))
	if err != nil || accessToken(cred) != "fake-access" {
		t.Fatalf("GetAuthCredential() after registering = %s, %v, want fake-access, nil", accessToken(cred), err)
	}
	if n := fake.count() - before; n != 1 {
		t.Errorf("fake exchanger calls by the last call = %d, want 1", n)
	}
}
