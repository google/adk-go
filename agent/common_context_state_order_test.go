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
	"context"
	"errors"
	"maps"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/internal/adkcontext"
	"google.golang.org/adk/v2/session"
)

const (
	// stillBlocked is how long an operation must stay pending to count as
	// waiting for its turn.
	stillBlocked = 50 * time.Millisecond
	// deadlockGuard bounds every wait that must end.
	deadlockGuard = 5 * time.Second
)

// orderedCall is the tool context of call 1 of a two-call batch whose call 0
// has not returned, over an in-memory session that holds {"seed": 1}.
type orderedCall struct {
	order   *adkcontext.CallOrder
	cancel  context.CancelFunc
	session session.Session
	actions *session.EventActions
	tool    Context
}

func newOrderedCall(t *testing.T) orderedCall {
	t.Helper()
	created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{
		AppName: "app",
		UserID:  "user",
		State:   map[string]any{"seed": 1},
	})
	if err != nil {
		t.Fatalf("session.Create() error = %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	order := adkcontext.NewCallOrder(2)
	ic := &invocationContext{
		Context: adkcontext.WithCallPlace(ctx, order, 1),
		agent:   &agent{name: "agent"},
		session: created.Session,
	}
	actions := &session.EventActions{StateDelta: map[string]any{}}
	return orderedCall{
		order:   order,
		cancel:  cancel,
		session: created.Session,
		actions: actions,
		tool:    NewToolContext(ic, "fc-1", actions, nil),
	}
}

// start runs op on its own goroutine and returns a channel that is closed when
// op has returned.
func start(op func()) <-chan struct{} {
	done := make(chan struct{})
	go func() {
		defer close(done)
		op()
	}()
	return done
}

func TestCallbackContextState_GetSetAllWaitForTurn(t *testing.T) {
	tests := []struct {
		name string
		// op uses the call's State; check runs after op has returned.
		op    func(c orderedCall) func(t *testing.T)
		check func(t *testing.T, c orderedCall)
	}{
		{
			name: "Get",
			op: func(c orderedCall) func(t *testing.T) {
				got, err := c.tool.State().Get("seed")
				return func(t *testing.T) {
					if err != nil || got != 1 {
						t.Errorf("State().Get(seed) = %v, %v, want 1, nil", got, err)
					}
				}
			},
		},
		{
			name: "Set",
			op: func(c orderedCall) func(t *testing.T) {
				err := c.tool.State().Set("k", "v")
				return func(t *testing.T) {
					if err != nil {
						t.Errorf("State().Set(k, v) = %v, want nil", err)
					}
					if got, err := c.session.State().Get("k"); err != nil || got != "v" {
						t.Errorf("session state k = %v, %v, want v, nil", got, err)
					}
					if got := c.actions.StateDelta["k"]; got != "v" {
						t.Errorf("Actions().StateDelta[k] = %v, want v", got)
					}
				}
			},
		},
		{
			name: "All",
			op: func(c orderedCall) func(t *testing.T) {
				got := maps.Collect(c.tool.State().All())
				return func(t *testing.T) {
					if diff := cmp.Diff(map[string]any{"seed": 1}, got); diff != "" {
						t.Errorf("State().All() mismatch (-want +got):\n%s", diff)
					}
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := newOrderedCall(t)
			var check func(t *testing.T)
			done := start(func() { check = tt.op(c) })

			select {
			case <-done:
				t.Fatalf("State().%s returned while call 0 was running, want it pending", tt.name)
			case <-time.After(stillBlocked):
			}
			c.order.Finish(0)
			select {
			case <-done:
			case <-time.After(deadlockGuard):
				t.Fatalf("State().%s still pending %v after call 0 finished", tt.name, deadlockGuard)
			}
			check(t)
		})
	}
}

func TestCallbackContextState_CanceledWaitFailsWithoutWriting(t *testing.T) {
	c := newOrderedCall(t)
	c.cancel()

	_, getErr := c.tool.State().Get("seed")
	setErr := c.tool.State().Set("k", "v")
	all := maps.Collect(c.tool.State().All())

	for _, tc := range []struct {
		method string
		err    error
		want   string
	}{
		{"Get(seed)", getErr, "cannot get state: wait for earlier function calls: context canceled"},
		{"Set(k, v)", setErr, "cannot set state: wait for earlier function calls: context canceled"},
	} {
		if tc.err == nil {
			t.Errorf("State().%s error = nil, want %q", tc.method, tc.want)
			continue
		}
		if tc.err.Error() != tc.want {
			t.Errorf("State().%s error = %q, want %q", tc.method, tc.err, tc.want)
		}
		if !errors.Is(tc.err, context.Canceled) {
			t.Errorf("State().%s error = %v, want errors.Is(err, context.Canceled)", tc.method, tc.err)
		}
		if strings.Contains(tc.err.Error(), "seed") || strings.Contains(tc.err.Error(), `"k"`) {
			t.Errorf("State().%s error %q holds the state key", tc.method, tc.err)
		}
	}
	if got, err := c.session.State().Get("k"); !errors.Is(err, session.ErrStateKeyNotExist) {
		t.Errorf("session state k = %v, %v after a failed Set, want ErrStateKeyNotExist", got, err)
	}
	if got, ok := c.actions.StateDelta["k"]; ok {
		t.Errorf("Actions().StateDelta[k] = %v after a failed Set, want no entry", got)
	}
	if len(all) != 0 {
		t.Errorf("State().All() after a failed wait yielded %v, want no entries", all)
	}
}

func TestReadonlyState_DoesNotWaitForTurn(t *testing.T) {
	c := newOrderedCall(t)
	var (
		got    any
		getErr error
		all    map[string]any
	)
	done := start(func() {
		got, getErr = c.tool.ReadonlyState().Get("seed")
		all = maps.Collect(c.tool.ReadonlyState().All())
	})

	select {
	case <-done:
	case <-time.After(deadlockGuard):
		t.Fatalf("ReadonlyState() still pending %v while call 0 was running, want no wait", deadlockGuard)
	}
	if getErr != nil || got != 1 {
		t.Errorf("ReadonlyState().Get(seed) = %v, %v, want 1, nil", got, getErr)
	}
	if diff := cmp.Diff(map[string]any{"seed": 1}, all); diff != "" {
		t.Errorf("ReadonlyState().All() mismatch (-want +got):\n%s", diff)
	}
	c.order.Finish(0)
}
