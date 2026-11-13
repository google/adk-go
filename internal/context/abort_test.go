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
			ctx := WithAbortState(t.Context(), NewAbortState(tc.signal))
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
	if s := NewAbortState(nil); s != nil {
		t.Fatalf("NewAbortState(nil) = %v, want nil", s)
	}
	outer := WithAbortState(t.Context(), NewAbortState(closedSignal()))
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
	s := AbortStateFrom(WithAbortState(t.Context(), NewAbortState(ch)))
	if got := s.Signal(); got != (<-chan struct{})(ch) {
		t.Errorf("Signal() = %v, want the channel given to NewAbortState", got)
	}
}

func TestAbortState_ClaimSealOnce(t *testing.T) {
	s := NewAbortState(make(chan struct{}))
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
		s := NewAbortState(sig)
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
		s := NewAbortState(sig)
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
	aborted := WithAbortState(t.Context(), NewAbortState(closedSignal()))
	notAborted := WithAbortState(t.Context(), NewAbortState(make(chan struct{})))
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
