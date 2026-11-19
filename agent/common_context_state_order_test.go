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
	"maps"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/internal/adkcontext"
	"google.golang.org/adk/v2/session"
)

// deadlockGuard bounds every wait that must end.
const deadlockGuard = 5 * time.Second

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
