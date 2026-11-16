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

package adkcontext

import (
	"context"
	"sync"
	"sync/atomic"
)

// CallOrder orders the use of shared state by the function calls of one batch
// (the calls of one model response) by their index in that response. Call i's
// turn begins when calls 0 to i-1 have all returned.
//
// Create one with [NewCallOrder]. A nil *CallOrder orders nothing: [CallOrder.Finish]
// does nothing and [CallOrder.Await] returns nil at once. All methods are safe
// for concurrent use.
type CallOrder struct {
	finished []chan struct{} // finished[i] is closed when call i has returned
	once     []sync.Once
}

// NewCallOrder returns an order for calls calls, none of them finished. calls
// must be >= 0; a negative value panics.
func NewCallOrder(calls int) *CallOrder {
	finished := make([]chan struct{}, calls)
	for i := range finished {
		finished[i] = make(chan struct{})
	}
	return &CallOrder{finished: finished, once: make([]sync.Once, calls)}
}

// Finish marks call as returned and releases the calls that wait for it. call
// must be in [0, calls) of [NewCallOrder]; otherwise Finish panics. Finishing a
// call again does nothing, so a deferred Finish may follow an explicit one.
func (o *CallOrder) Finish(call int) {
	if o == nil {
		return
	}
	o.once[call].Do(func() { close(o.finished[call]) })
}

// Await blocks until calls 0 to call-1 have all finished, and returns nil. If
// ctx is done first, it returns ctx.Err() unwrapped. A done ctx never turns a
// call that is already released into an error: the earlier calls are checked
// before ctx. call must be in [0, calls] of [NewCallOrder]; otherwise Await
// panics.
func (o *CallOrder) Await(ctx context.Context, call int) error {
	if o == nil {
		return nil
	}
	for _, done := range o.finished[:call] {
		select {
		case <-done:
			continue
		default:
		}
		select {
		case <-done:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

// callPlaceKey is the context value key of a *callPlace.
const callPlaceKey ctxKey = 2

// callPlace is a call's place in a batch, as stored in the call's context.
// Contexts derived from the call's context share it, so turnBegun is set once
// for the call and its callbacks.
type callPlace struct {
	order     *CallOrder // nil: the context belongs to no batch, and a nil order never waits
	call      int
	turnBegun atomic.Bool
}

// placeOf returns the place ctx carries, or nil.
func placeOf(ctx context.Context) *callPlace {
	p, _ := ctx.Value(callPlaceKey).(*callPlace)
	return p
}

// WithCallPlace returns a copy of ctx that carries call's place in o, so that
// [AwaitEarlierCalls] on it and on every context derived from it waits for the
// calls before call. The place replaces any place ctx already carries. A nil o
// is stored too: it removes an outer place, as [WithoutCallPlace] does.
func WithCallPlace(ctx context.Context, o *CallOrder, call int) context.Context {
	return context.WithValue(ctx, callPlaceKey, &callPlace{order: o, call: call})
}

// WithoutCallPlace returns a copy of ctx that carries no place, so a run
// started from ctx is not ordered against the batch that ctx belongs to.
// [AwaitEarlierCalls] on the result, and on every context derived from it,
// returns nil at once, and [SkipCallOrder] on it does nothing.
func WithoutCallPlace(ctx context.Context) context.Context {
	return WithCallPlace(ctx, nil, 0)
}

// AwaitEarlierCalls waits for the turn of the call whose place ctx carries:
// until every earlier call of its batch has returned. It returns nil at once
// when ctx carries no place, when the place was removed with [WithoutCallPlace],
// or when the call's turn has already begun. If ctx is done before the turn
// begins, it returns ctx.Err() unwrapped and the turn has not begun.
func AwaitEarlierCalls(ctx context.Context) error {
	p := placeOf(ctx)
	if p == nil || p.turnBegun.Load() {
		return nil
	}
	if err := p.order.Await(ctx, p.call); err != nil {
		return err
	}
	p.turnBegun.Store(true)
	return nil
}

// SkipCallOrder begins the turn of the call whose place ctx carries without
// waiting: later [AwaitEarlierCalls] by the same call, through ctx or any
// context derived from it, return nil at once. It does not release the calls
// after this one; they still wait for this call to return. It does nothing when
// ctx carries no place or a place removed with [WithoutCallPlace].
//
// Only code that keeps its own lock around its whole read-modify-write of
// state may call it: after it, the call does not see the writes of earlier
// calls that have not returned.
func SkipCallOrder(ctx context.Context) {
	if p := placeOf(ctx); p != nil {
		p.turnBegun.Store(true)
	}
}
