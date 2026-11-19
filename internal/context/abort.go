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
	"sync"
	"sync/atomic"
)

// AbortState is the abort state of one run that was given an abort signal:
// the caller's channel, whether the signal has fired, and the guard that lets
// the run seal its invocation once. Every context derived from the run's
// context reaches it through [AbortStateFrom].
//
// The signal fires when the caller closes the channel or sends one value on
// it. A check ([AbortState.Aborted]) that happens after the close or the send
// sees it, whichever reader of the state received the value: a check itself,
// or the goroutine that receives for the watches ([AbortState.Watch]). From
// then on the state stays fired: every later check reports true and every
// watch of the state cancels its context. The state of a run nested in another
// run with the same signal shares this with the state of the enclosing run
// (see [NewAbortState]), so the signal stops both runs whichever of them
// receives it.
//
// A nil *AbortState means the run has no abort signal; every method is safe
// on nil. Use the pointer; the state must not be copied. All methods are safe
// for concurrent use.
type AbortState struct {
	fire *abortFire
	// sealed is per run: each nested run seals its own invocation.
	sealed atomic.Bool
}

// abortFire is the part of an abort state that the states of nested runs
// with the same signal share.
//
// Two kinds of reader receive from signal, and each trips the state before it
// acts on the value: a check, which receives without blocking while it holds
// mu, and the receiver goroutine, the only reader that blocks on signal. A
// check that finds no value asks the receiver and waits for its answer; the
// receiver answers only from its select, where it holds no value, so a value
// it has received is visible to the check as a closed fired before the check
// returns.
type abortFire struct {
	signal <-chan struct{}
	// fired is closed, once, by trip.
	fired    chan struct{}
	tripOnce sync.Once

	// mu serializes the checks with each other and with starting and ending
	// the receiver. The receiver never takes it.
	mu sync.Mutex
	// receiver is the running receiver goroutine, or nil while no watch is
	// active. watches counts the active watches.
	receiver *abortReceiver
	watches  int
}

// abortReceiver is the goroutine that receives from the signal for the
// watches of one abortFire.
type abortReceiver struct {
	// asked takes a check's question; the receiver's answer is taking it.
	asked chan struct{}
	done  chan struct{}
	wg    sync.WaitGroup
}

type abortStateKey struct{}

// NewAbortState returns the abort state of a run for signal, or nil when
// signal is nil. ctx is the context the run is called with. When ctx carries
// the abort state of an enclosing run with the same signal, as the context of
// a tool does, the new state has fired exactly when that state has: it
// shares whether the signal has fired, and has a seal guard of its own.
func NewAbortState(ctx context.Context, signal <-chan struct{}) *AbortState {
	if signal == nil {
		return nil
	}
	if enclosing := AbortStateFrom(ctx); enclosing != nil && enclosing.fire.signal == signal {
		return &AbortState{fire: enclosing.fire}
	}
	return &AbortState{fire: &abortFire{signal: signal, fired: make(chan struct{})}}
}

// WithAbortState returns a copy of ctx that carries s. It stores s even when
// s is nil, so a nested run without a signal does not see the state of the
// run that contains it.
func WithAbortState(ctx context.Context, s *AbortState) context.Context {
	return context.WithValue(ctx, abortStateKey{}, s)
}

// AbortStateFrom returns the newest abort state ctx carries, or nil.
func AbortStateFrom(ctx context.Context) *AbortState {
	s, _ := ctx.Value(abortStateKey{}).(*AbortState)
	return s
}

// Aborted reports whether the signal has fired. It does not wait for the
// signal: it waits only for other checks of the state and, while a watch is
// active, for the watch's receiver goroutine to answer. Once it reports true
// it always does. It reports false on a nil state.
func (s *AbortState) Aborted() bool {
	if s == nil {
		return false
	}
	f := s.fire
	select {
	case <-f.fired:
		return true
	default:
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	select {
	case <-f.fired:
		return true
	case <-f.signal:
		s.trip()
		return true
	default:
	}
	if f.receiver == nil {
		return false
	}
	// The receiver ends only after it has tripped the state, or when the last
	// watch ends, which needs mu; so it answers, or fired is closed.
	select {
	case <-f.fired:
		return true
	case f.receiver.asked <- struct{}{}:
		return false
	}
}

// trip records that the signal has fired. Every reader that receives from the
// signal calls it before it acts on the signal. It may be called more than
// once.
func (s *AbortState) trip() {
	s.fire.trip()
}

func (f *abortFire) trip() {
	f.tripOnce.Do(func() { close(f.fired) })
}

// Signal returns the caller's channel, or nil for a nil state.
func (s *AbortState) Signal() <-chan struct{} {
	if s == nil {
		return nil
	}
	return s.fire.signal
}

// ClaimSeal reports true to its first caller only. It reports false on a nil
// state.
func (s *AbortState) ClaimSeal() bool {
	return s != nil && s.sealed.CompareAndSwap(false, true)
}

// Watch returns a child of parent that is canceled, with the cause
// context.Canceled, when the signal fires, and stop, which ends the watch and
// cancels the child. A signal that fired before Watch is called, or whose
// value Watch receives, leaves the child already canceled when Watch returns,
// and Watch starts no goroutine. Otherwise the watch runs one goroutine, which
// ends when the signal fires, through any reader of the state, or when stop is
// called, whichever is first. The watches of a state, and of the states that
// share it, also share one receiver goroutine, which runs while any of them is
// active. stop returns only after the watch's goroutine has ended, and after
// the receiver has ended when the watch was the last one. stop may be called
// more than once. On a nil state Watch returns parent and a no-op stop.
func (s *AbortState) Watch(parent context.Context) (ctx context.Context, stop context.CancelFunc) {
	if s == nil {
		return parent, func() {}
	}
	// A plain cancel, not a custom cause: plugin/agentanalytics treats a
	// context whose cause is context.Canceled as a consumer stop, not a
	// failure.
	ctx, cancel := context.WithCancel(parent)
	// Checked on the caller's goroutine, so that a signal that has already
	// fired is seen by the caller's next Aborted. A goroutine that received a
	// sent value would make it visible only some time later.
	if s.Aborted() {
		cancel()
		return ctx, cancel
	}
	s.fire.addWatch()
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() {
		select {
		case <-s.fire.fired:
			cancel()
		case <-done:
		}
	})
	var once sync.Once
	return ctx, func() {
		once.Do(func() {
			close(done)
			wg.Wait()
			s.fire.removeWatch()
		})
		cancel()
	}
}

// addWatch counts a new watch and starts the receiver for the first one.
func (f *abortFire) addWatch() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.watches++
	if f.receiver != nil {
		return
	}
	r := &abortReceiver{asked: make(chan struct{}), done: make(chan struct{})}
	f.receiver = r
	r.wg.Go(func() {
		for {
			select {
			case <-f.signal:
				f.trip()
				return
			case <-f.fired:
				return
			case <-r.asked:
			case <-r.done:
				return
			}
		}
	})
}

// removeWatch counts a watch out and ends the receiver after the last one.
// It waits for the receiver while it holds mu, so that no check runs between
// the receiver's receiving a value and its tripping the state.
func (f *abortFire) removeWatch() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.watches--
	if f.watches > 0 {
		return
	}
	close(f.receiver.done)
	f.receiver.wg.Wait()
	f.receiver = nil
}

// Aborted reports whether the abort state ctx carries has fired. It reports
// false when ctx carries none.
func Aborted(ctx context.Context) bool {
	return AbortStateFrom(ctx).Aborted()
}

// IsAbortEcho reports whether err is the run's own abort cancellation coming
// back up the call chain: err wraps context.Canceled, the abort state of
// runCtx has fired, and runCtx itself is not done. runCtx is the run's own
// context, which an abort never cancels, so after the caller cancels runCtx
// no error is an echo. It reports false for a nil err.
func IsAbortEcho(runCtx context.Context, err error) bool {
	return err != nil && errors.Is(err, context.Canceled) && Aborted(runCtx) && runCtx.Err() == nil
}
