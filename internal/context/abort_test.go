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

package context

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func closedSignal() chan struct{} {
	ch := make(chan struct{})
	close(ch)
	return ch
}

func TestAbortState_Aborted(t *testing.T) {
	sent := make(chan struct{}, 1)
	sent <- struct{}{}
	tests := []struct {
		name   string
		signal chan struct{}
		want   bool
	}{
		{name: "open", signal: make(chan struct{}), want: false},
		{name: "closed", signal: closedSignal(), want: true},
		{name: "value_sent", signal: sent, want: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := WithAbortState(t.Context(), NewAbortState(t.Context(), tc.signal))
			if got := Aborted(ctx); got != tc.want {
				t.Fatalf("Aborted() = %v, want %v", got, tc.want)
			}
			// A received value is consumed; the state must remember it.
			if got := Aborted(ctx); got != tc.want {
				t.Errorf("second Aborted() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestAbortState_NilStateMeansNoAbort(t *testing.T) {
	if s := NewAbortState(t.Context(), nil); s != nil {
		t.Fatalf("NewAbortState(t.Context(), nil) = %v, want nil", s)
	}
	outer := WithAbortState(t.Context(), NewAbortState(t.Context(), closedSignal()))
	inner := WithAbortState(outer, nil)
	if !Aborted(outer) {
		t.Fatal("Aborted(outer) = false, want true")
	}
	if Aborted(inner) {
		t.Error("Aborted(inner) = true, want false: a nil state must shadow the outer one")
	}
	if Aborted(t.Context()) {
		t.Error("Aborted on a context with no state = true, want false")
	}
	var s *AbortState
	if s.ClaimSeal() {
		t.Error("nil ClaimSeal() = true, want false")
	}
	if s.Signal() != nil {
		t.Error("nil Signal() != nil")
	}
}

func TestAbortState_SignalIsTheCallersChannel(t *testing.T) {
	ch := make(chan struct{})
	s := AbortStateFrom(WithAbortState(t.Context(), NewAbortState(t.Context(), ch)))
	if got := s.Signal(); got != (<-chan struct{})(ch) {
		t.Errorf("Signal() = %v, want the channel given to NewAbortState", got)
	}
}

// TestNewAbortState_NestedRunOfTheSameSignalSharesFiring checks that the
// state of a run nested in a run with the same signal fires with the
// enclosing state, whichever of them receives the signal, while a nested run
// with another signal does not, and that each state seals once on its own.
func TestNewAbortState_NestedRunOfTheSameSignalSharesFiring(t *testing.T) {
	sig := make(chan struct{}, 1)
	outer := NewAbortState(t.Context(), sig)
	toolCtx := WithAbortState(t.Context(), outer)
	nested := NewAbortState(toolCtx, sig)
	otherSignal := NewAbortState(toolCtx, make(chan struct{}))

	sig <- struct{}{}
	if !nested.Aborted() {
		t.Fatal("nested Aborted() after the send = false, want true")
	}
	if !outer.Aborted() {
		t.Error("outer Aborted() after the nested state received the signal = false, want true")
	}
	if otherSignal.Aborted() {
		t.Error("Aborted() of a nested state with another signal = true, want false")
	}
	if !nested.ClaimSeal() {
		t.Error("nested ClaimSeal() = false, want true")
	}
	if !outer.ClaimSeal() {
		t.Error("outer ClaimSeal() after the nested state sealed = false, want true")
	}
}

func TestAbortState_ClaimSealOnce(t *testing.T) {
	s := NewAbortState(t.Context(), make(chan struct{}))
	var wins atomic.Int32
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			if s.ClaimSeal() {
				wins.Add(1)
			}
		})
	}
	wg.Wait()
	if got := wins.Load(); got != 1 {
		t.Errorf("ClaimSeal returned true %d times, want 1", got)
	}
}

func TestAbortState_Watch(t *testing.T) {
	t.Run("signal_cancels_child_not_parent", func(t *testing.T) {
		sig := make(chan struct{}, 1)
		s := NewAbortState(t.Context(), sig)
		parent := t.Context()
		child, stop := s.Watch(parent)
		defer stop()

		sig <- struct{}{}
		select {
		case <-child.Done():
		case <-time.After(5 * time.Second):
			t.Fatal("child context not canceled 5s after the signal")
		}
		if got := context.Cause(child); got != context.Canceled {
			t.Errorf("context.Cause(child) = %v, want context.Canceled", got)
		}
		if parent.Err() != nil {
			t.Errorf("parent.Err() = %v, want nil", parent.Err())
		}
		// The watcher consumed the sent value; the state must still report it.
		if !s.Aborted() {
			t.Error("Aborted() = false after the watcher saw the signal, want true")
		}
	})
	t.Run("stop_cancels_child_and_ends_watch", func(t *testing.T) {
		sig := make(chan struct{}, 1)
		s := NewAbortState(t.Context(), sig)
		child, stop := s.Watch(t.Context())
		stop()
		stop()
		if child.Err() == nil {
			t.Error("child.Err() = nil after stop, want non-nil")
		}
		sig <- struct{}{}
		// No watcher is left to consume the value, so the first poll does.
		if !s.Aborted() {
			t.Error("Aborted() = false, want true")
		}
	})
	t.Run("fired_before_watch_is_seen_when_watch_returns", func(t *testing.T) {
		// One P: the watcher goroutine cannot run before this goroutine
		// blocks, so only Watch itself can have taken the signal by the time
		// it returns. No test in this package runs in parallel.
		defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
		sent := make(chan struct{}, 1)
		sent <- struct{}{}
		for name, sig := range map[string]chan struct{}{"value_sent": sent, "closed": closedSignal()} {
			t.Run(name, func(t *testing.T) {
				s := NewAbortState(t.Context(), sig)
				child, stop := s.Watch(t.Context())
				defer stop()

				// Checked at once: the runner polls right after Watch returns,
				// and a run whose signal fired before it started must not start.
				if !s.Aborted() {
					t.Error("Aborted() right after Watch = false, want true")
				}
				if child.Err() == nil {
					t.Error("child.Err() right after Watch = nil, want context.Canceled")
				}
			})
		}
	})
	t.Run("value_received_by_any_reader_cancels_every_watch", func(t *testing.T) {
		// One P: the watcher goroutines usually have not run when the value is
		// sent, so the poll below receives it and the watchers can only learn
		// of it from the state. When a watcher has run and receives the value
		// itself, the other watch can still only learn of it from the state.
		defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
		sig := make(chan struct{}, 1)
		s := NewAbortState(t.Context(), sig)
		first, stopFirst := s.Watch(t.Context())
		defer stopFirst()
		second, stopSecond := s.Watch(t.Context())
		defer stopSecond()

		sig <- struct{}{}
		s.Aborted()
		for name, child := range map[string]context.Context{"first": first, "second": second} {
			select {
			case <-child.Done():
			case <-time.After(5 * time.Second):
				t.Errorf("%s watch: child context not canceled 5s after the signal was received", name)
			}
		}
		if !s.Aborted() {
			t.Error("Aborted() after the watches were canceled = false, want true")
		}
	})
	t.Run("check_after_a_send_sees_a_value_the_receiver_took", func(t *testing.T) {
		// One P, and two yields of it after Watch: the receiver goroutine has
		// run and is parked on the channel, so the send hands the value to it
		// and leaves the channel empty, and the receiver cannot trip the state
		// before this goroutine blocks.
		defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
		sig := make(chan struct{}, 1)
		s := NewAbortState(t.Context(), sig)
		child, stop := s.Watch(t.Context())
		defer stop()
		runtime.Gosched()
		runtime.Gosched()

		sig <- struct{}{}
		if !s.Aborted() {
			t.Fatal("Aborted() right after the send = false, want true")
		}
		select {
		case <-child.Done():
		case <-time.After(5 * time.Second):
			t.Error("child context not canceled 5s after the signal")
		}
	})
	t.Run("checks_watches_and_a_send_at_once", func(t *testing.T) {
		// Checks run beside watches that start and stop, and one value is
		// sent: every check that starts after the send has returned reports
		// true, and no check waits on a receiver that has ended.
		sig := make(chan struct{}, 1)
		outer := NewAbortState(t.Context(), sig)
		nested := NewAbortState(WithAbortState(t.Context(), outer), sig)
		var sent atomic.Bool
		var missed atomic.Int32
		var wg sync.WaitGroup
		for i := range 8 {
			s := outer
			if i%2 == 1 {
				s = nested
			}
			wg.Go(func() {
				for range 200 {
					_, stop := s.Watch(t.Context())
					after := sent.Load()
					if !s.Aborted() && after {
						missed.Add(1)
					}
					stop()
				}
			})
		}
		wg.Go(func() {
			runtime.Gosched()
			sig <- struct{}{}
			sent.Store(true)
		})
		wg.Wait()
		if got := missed.Load(); got != 0 {
			t.Errorf("%d checks that started after the send reported false, want 0", got)
		}
		if !outer.Aborted() || !nested.Aborted() {
			t.Error("Aborted() after the send = false, want true on both states")
		}
	})
	t.Run("nil_state_returns_parent", func(t *testing.T) {
		var s *AbortState
		parent := t.Context()
		child, stop := s.Watch(parent)
		stop()
		if child != parent {
			t.Error("nil Watch returned a new context, want parent")
		}
	})
}

func TestIsAbortEcho(t *testing.T) {
	aborted := WithAbortState(t.Context(), NewAbortState(t.Context(), closedSignal()))
	notAborted := WithAbortState(t.Context(), NewAbortState(t.Context(), make(chan struct{})))
	callerCanceled, cancel := context.WithCancel(aborted)
	cancel()
	tests := []struct {
		name   string
		runCtx context.Context
		err    error
		want   bool
	}{
		{name: "wrapped_cancel_after_abort", runCtx: aborted, err: fmt.Errorf("failed to run sub-agent %q: %w", "a", context.Canceled), want: true},
		{name: "nil_error", runCtx: aborted, err: nil, want: false},
		{name: "not_aborted", runCtx: notAborted, err: context.Canceled, want: false},
		{name: "no_state", runCtx: t.Context(), err: context.Canceled, want: false},
		{name: "caller_canceled_run_context", runCtx: callerCanceled, err: context.Canceled, want: false},
		{name: "deadline", runCtx: aborted, err: context.DeadlineExceeded, want: false},
		{name: "other_error", runCtx: aborted, err: errors.New("quota exceeded"), want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsAbortEcho(tc.runCtx, tc.err); got != tc.want {
				t.Errorf("IsAbortEcho() = %v, want %v", got, tc.want)
			}
		})
	}
}
