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

package agent

import (
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/session"
)

var errRefused = errors.New("refused by the test validator")

// refuseKey is a StateValidator that refuses every write of one key.
type refuseKey struct{ key string }

func (r *refuseKey) ValidateStateEntry(key string, _ any) error {
	if key == r.key {
		return errRefused
	}
	return nil
}

// newValidatorTestInvocation returns an invocation over an empty in-memory
// session, and that session.
func newValidatorTestInvocation(t *testing.T) (InvocationContext, session.Session) {
	t.Helper()
	created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{
		AppName: "app",
		UserID:  "user",
	})
	if err != nil {
		t.Fatalf("session.Create() error = %v", err)
	}
	ic := &invocationContext{
		Context: t.Context(),
		agent:   &agent{name: "agent"},
		session: created.Session,
	}
	return ic, created.Session
}

func stateHas(t *testing.T, s session.Session, key string) bool {
	t.Helper()
	_, err := s.State().Get(key)
	if errors.Is(err, session.ErrStateKeyNotExist) {
		return false
	}
	if err != nil {
		t.Fatalf("session State().Get(%q) error = %v", key, err)
	}
	return true
}

func TestCallbackContextState_SetRefusedByValidator(t *testing.T) {
	ic, sess := newValidatorTestInvocation(t)
	actions := &session.EventActions{StateDelta: map[string]any{}, ArtifactDelta: map[string]int64{}}
	var v StateValidator = &refuseKey{key: "bad"}
	ctx := PromoteWithDelta(ic, &CommonContextDelta{EventActions: actions, StateValidator: &v})

	if err := ctx.State().Set("bad", 1); err != errRefused {
		t.Fatalf("State().Set(%q) error = %v, want errRefused unwrapped", "bad", err)
	}
	if _, ok := actions.StateDelta["bad"]; ok {
		t.Errorf("StateDelta has %q after a refused Set", "bad")
	}
	if stateHas(t, sess, "bad") {
		t.Errorf("session state has %q after a refused Set", "bad")
	}

	if err := ctx.State().Set("good", 2); err != nil {
		t.Fatalf("State().Set(%q) error = %v", "good", err)
	}
	if diff := cmp.Diff(map[string]any{"good": 2}, actions.StateDelta); diff != "" {
		t.Errorf("StateDelta mismatch (-want +got):\n%s", diff)
	}
	if got, err := sess.State().Get("good"); err != nil || got != 2 {
		t.Errorf("session State().Get(%q) = %v, %v; want 2, nil", "good", got, err)
	}
}

func TestCallbackContextState_SetCreatesDeltaMap(t *testing.T) {
	ic, _ := newValidatorTestInvocation(t)
	actions := &session.EventActions{}
	ctx := PromoteWithDelta(ic, &CommonContextDelta{EventActions: actions})

	if err := ctx.State().Set("k", 1); err != nil {
		t.Fatalf("State().Set(%q) error = %v", "k", err)
	}
	if diff := cmp.Diff(map[string]any{"k": 1}, actions.StateDelta); diff != "" {
		t.Errorf("StateDelta mismatch (-want +got):\n%s", diff)
	}
	if actions.ArtifactDelta == nil {
		t.Errorf("ArtifactDelta is nil, want the map WithDelta creates")
	}
}

func TestToolAndCallbackContexts_DropStateValidator(t *testing.T) {
	tests := []struct {
		name  string
		build func(validated Context) Context
	}{
		{
			name:  "tool context",
			build: func(validated Context) Context { return NewToolContext(validated, "fc-1", nil, nil) },
		},
		{
			name:  "callback context",
			build: func(validated Context) Context { return NewCallbackContext(validated, nil) },
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ic, _ := newValidatorTestInvocation(t)
			var v StateValidator = &refuseKey{key: "bad"}
			validated := PromoteWithDelta(ic, &CommonContextDelta{
				EventActions:   &session.EventActions{},
				StateValidator: &v,
			})
			ctx := tc.build(validated)

			if err := ctx.State().Set("bad", 1); err != nil {
				t.Fatalf("State().Set(%q) error = %v, want nil", "bad", err)
			}
			if got, err := ctx.State().Get("bad"); err != nil || got != 1 {
				t.Errorf("State().Get(%q) = %v, %v; want 1, nil", "bad", got, err)
			}
		})
	}
}

func TestWithDelta_PassesActionsAndValidatorDown(t *testing.T) {
	ic, _ := newValidatorTestInvocation(t)
	actions := &session.EventActions{}
	var v StateValidator = &refuseKey{key: "bad"}
	a := PromoteWithDelta(ic, &CommonContextDelta{EventActions: actions, StateValidator: &v})
	path := "wf@1/node@1"
	b := a.WithDelta(&CommonContextDelta{Path: &path})

	if b.Actions() != actions {
		t.Errorf("derived Actions() = %p, want the given EventActions %p", b.Actions(), actions)
	}
	if err := b.State().Set("bad", 1); err != errRefused {
		t.Errorf("derived State().Set(%q) error = %v, want errRefused", "bad", err)
	}
	got, ok := b.(interface{ StateValidator() StateValidator })
	if !ok {
		t.Fatalf("derived context %T has no StateValidator method", b)
	}
	if got.StateValidator() != v {
		t.Errorf("derived StateValidator() = %v, want the given validator", got.StateValidator())
	}
}

func TestWithDelta_NilStateValidatorRemovesIt(t *testing.T) {
	ic, _ := newValidatorTestInvocation(t)
	var v StateValidator = &refuseKey{key: "bad"}
	validated := PromoteWithDelta(ic, &CommonContextDelta{EventActions: &session.EventActions{}, StateValidator: &v})

	ctx := validated.WithDelta(&CommonContextDelta{StateValidator: new(StateValidator)})

	if err := ctx.State().Set("bad", 1); err != nil {
		t.Errorf("State().Set(%q) error = %v, want nil once the validator is removed", "bad", err)
	}
}
