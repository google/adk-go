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
// A nil *AbortState means the run has no abort signal; every method is safe
// on nil. Use the pointer; the state must not be copied. All methods are safe
// for concurrent use.
type AbortState struct {
	signal  <-chan struct{}
	tripped atomic.Bool
	sealed  atomic.Bool
}

type abortStateKey struct{}

// NewAbortState returns the abort state for signal, or nil when signal is
// nil.
func NewAbortState(signal <-chan struct{}) *AbortState {
	if signal == nil {
		return nil
	}
	return &AbortState{signal: signal}
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

// Aborted reports whether the signal has fired. It never blocks, and once it
// reports true it always does. It reports false on a nil state.
func (s *AbortState) Aborted() bool {
	if s == nil {
		return false
	}
	if s.tripped.Load() {
		return true
	}
	select {
	case <-s.signal:
		s.tripped.Store(true)
		return true
	default:
		return false
	}
}

// Signal returns the caller's channel, or nil for a nil state.
func (s *AbortState) Signal() <-chan struct{} {
	if s == nil {
		return nil
	}
	return s.signal
}

// ClaimSeal reports true to its first caller only. It reports false on a nil
// state.
func (s *AbortState) ClaimSeal() bool {
	return s != nil && s.sealed.CompareAndSwap(false, true)
}

// Watch returns a child of parent that is canceled, with the cause
// context.Canceled, when the signal fires, and stop, which ends the watch and
// cancels the child. The watch runs one goroutine, which ends when the signal
// fires or stop is called, whichever is first; stop returns only after it has
// ended. stop may be called more than once. On a nil state Watch returns
// parent and a no-op stop.
func (s *AbortState) Watch(parent context.Context) (ctx context.Context, stop context.CancelFunc) {
	if s == nil {
		return parent, func() {}
	}
	// A plain cancel, not a custom cause: plugin/agentanalytics treats a
	// context whose cause is context.Canceled as a consumer stop, not a
	// failure.
	ctx, cancel := context.WithCancel(parent)
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() {
		select {
		case <-s.signal:
			s.tripped.Store(true)
			cancel()
		case <-done:
		}
	})
	var once sync.Once
	return ctx, func() {
		once.Do(func() { close(done) })
		wg.Wait()
		cancel()
	}
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
