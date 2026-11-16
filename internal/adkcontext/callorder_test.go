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

package adkcontext_test

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/v2/internal/adkcontext"
)

const (
	// stillBlocked is how long a wait must stay pending to count as blocked.
	stillBlocked = 50 * time.Millisecond
	// deadlockGuard bounds every wait that is expected to return, so a broken
	// order fails the test instead of hanging it.
	deadlockGuard = 5 * time.Second
)

// async runs wait on its own goroutine and returns a channel that receives its
// result.
func async(wait func() error) <-chan error {
	result := make(chan error, 1)
	go func() { result <- wait() }()
	return result
}

func requireBlocked(t *testing.T, result <-chan error, what string) {
	t.Helper()
	select {
	case err := <-result:
		t.Fatalf("%s returned %v, want it still blocked after %v", what, err, stillBlocked)
	case <-time.After(stillBlocked):
	}
}

func receive(t *testing.T, result <-chan error, what string) error {
	t.Helper()
	select {
	case err := <-result:
		return err
	case <-time.After(deadlockGuard):
		t.Fatalf("%s still blocked after %v", what, deadlockGuard)
		return nil
	}
}

func canceledContext(t *testing.T) context.Context {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	return ctx
}

func TestCallOrder_AwaitWaitsForAllEarlierCalls(t *testing.T) {
	ctx := t.Context()
	o := adkcontext.NewCallOrder(3)

	if err := receive(t, async(func() error { return o.Await(ctx, 0) }), "Await(ctx, 0)"); err != nil {
		t.Fatalf("Await(ctx, 0) on a fresh order = %v, want nil", err)
	}

	got := async(func() error { return o.Await(ctx, 2) })
	o.Finish(1)
	requireBlocked(t, got, "Await(ctx, 2) with call 0 unfinished")
	o.Finish(0)
	if err := receive(t, got, "Await(ctx, 2) after Finish(0) and Finish(1)"); err != nil {
		t.Errorf("Await(ctx, 2) after Finish(0) and Finish(1) = %v, want nil", err)
	}
}

func TestCallOrder_AwaitReturnsContextErrorWhileEarlierCallRuns(t *testing.T) {
	o := adkcontext.NewCallOrder(2)

	canceled := canceledContext(t)
	if err := receive(t, async(func() error { return o.Await(canceled, 1) }), "Await(canceled, 1)"); err != context.Canceled {
		t.Errorf("Await(canceled ctx, 1) = %v, want context.Canceled", err)
	}

	timeout, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	if err := receive(t, async(func() error { return o.Await(timeout, 1) }), "Await(timeout, 1)"); err != context.DeadlineExceeded {
		t.Errorf("Await(ctx with a 20ms timeout, 1) = %v, want context.DeadlineExceeded", err)
	}
}

func TestCallOrder_FinishedEarlierCallsIgnoreDoneContext(t *testing.T) {
	o := adkcontext.NewCallOrder(3)
	o.Finish(0)
	o.Finish(1)
	canceled := canceledContext(t)

	if err := receive(t, async(func() error { return o.Await(canceled, 2) }), "Await(canceled, 2)"); err != nil {
		t.Errorf("Await(canceled ctx, 2) after Finish(0) and Finish(1) = %v, want nil", err)
	}
	if err := receive(t, async(func() error { return o.Await(canceled, 0) }), "Await(canceled, 0)"); err != nil {
		t.Errorf("Await(canceled ctx, 0) = %v, want nil", err)
	}
}

func TestCallOrder_FinishTwiceAndNilOrder(t *testing.T) {
	t.Run("finish twice", func(t *testing.T) {
		o := adkcontext.NewCallOrder(2)
		o.Finish(0)
		o.Finish(0)
		if err := receive(t, async(func() error { return o.Await(t.Context(), 1) }), "Await(ctx, 1)"); err != nil {
			t.Errorf("Await(ctx, 1) after Finish(0) twice = %v, want nil", err)
		}
	})
	t.Run("nil order", func(t *testing.T) {
		var o *adkcontext.CallOrder
		o.Finish(3)
		if err := receive(t, async(func() error { return o.Await(t.Context(), 5) }), "nil order Await(ctx, 5)"); err != nil {
			t.Errorf("nil order Await(ctx, 5) = %v, want nil", err)
		}
	})
}

func TestAwaitEarlierCalls_UsesThePlaceInTheContext(t *testing.T) {
	parent := t.Context()
	o := adkcontext.NewCallOrder(2)
	placed := adkcontext.WithCallPlace(parent, o, 1)

	if err := receive(t, async(func() error { return adkcontext.AwaitEarlierCalls(parent) }), "no place"); err != nil {
		t.Errorf("AwaitEarlierCalls(ctx without a place) = %v, want nil", err)
	}
	without := adkcontext.WithoutCallPlace(placed)
	if err := receive(t, async(func() error { return adkcontext.AwaitEarlierCalls(without) }), "WithoutCallPlace"); err != nil {
		t.Errorf("AwaitEarlierCalls(WithoutCallPlace(placed)) = %v, want nil", err)
	}
	nilOrder := adkcontext.WithCallPlace(placed, nil, 0)
	if err := receive(t, async(func() error { return adkcontext.AwaitEarlierCalls(nilOrder) }), "nil order place"); err != nil {
		t.Errorf("AwaitEarlierCalls(WithCallPlace(placed, nil, 0)) = %v, want nil", err)
	}

	got := async(func() error { return adkcontext.AwaitEarlierCalls(placed) })
	requireBlocked(t, got, "AwaitEarlierCalls(place 1) with call 0 unfinished")
	o.Finish(0)
	if err := receive(t, got, "AwaitEarlierCalls(place 1) after Finish(0)"); err != nil {
		t.Errorf("AwaitEarlierCalls(place 1) after Finish(0) = %v, want nil", err)
	}
}

func TestCallOrder_ConcurrentAwaitAndFinishUnderRace(t *testing.T) {
	ctx := t.Context()
	const calls = 32
	o := adkcontext.NewCallOrder(calls)
	var (
		mu  sync.Mutex
		got []int
		wg  sync.WaitGroup
	)
	for i := calls - 1; i >= 0; i-- {
		wg.Go(func() {
			if err := o.Await(ctx, i); err != nil {
				t.Errorf("Await(ctx, %d) = %v, want nil", i, err)
			}
			mu.Lock()
			got = append(got, i)
			mu.Unlock()
			o.Finish(i)
		})
	}
	allDone := make(chan struct{})
	go func() { wg.Wait(); close(allDone) }()
	select {
	case <-allDone:
	case <-time.After(deadlockGuard):
		t.Fatalf("%d calls still running after %v", calls, deadlockGuard)
	}

	want := []int{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
	}
	mu.Lock()
	defer mu.Unlock()
	if !slices.Equal(got, want) {
		t.Errorf("calls ran in order %v, want %v", got, want)
	}
}

func TestSkipCallOrder_StartsTheTurnWithoutWaiting(t *testing.T) {
	parent := t.Context()
	o := adkcontext.NewCallOrder(4)

	ctx2 := adkcontext.WithCallPlace(parent, o, 2)
	adkcontext.SkipCallOrder(ctx2)
	if err := receive(t, async(func() error { return adkcontext.AwaitEarlierCalls(ctx2) }), "AwaitEarlierCalls(skipped place 2)"); err != nil {
		t.Errorf("AwaitEarlierCalls(place 2) after SkipCallOrder = %v, want nil", err)
	}

	ctx3 := adkcontext.WithCallPlace(parent, o, 3)
	got := async(func() error { return adkcontext.AwaitEarlierCalls(ctx3) })
	requireBlocked(t, got, "AwaitEarlierCalls(place 3) with calls 0 to 2 unfinished")
	o.Finish(0)
	o.Finish(1)
	requireBlocked(t, got, "AwaitEarlierCalls(place 3) with call 2 skipped but unfinished")
	o.Finish(2)
	if err := receive(t, got, "AwaitEarlierCalls(place 3) after Finish(0), Finish(1), Finish(2)"); err != nil {
		t.Errorf("AwaitEarlierCalls(place 3) after calls 0 to 2 finished = %v, want nil", err)
	}

	adkcontext.SkipCallOrder(parent)
	adkcontext.SkipCallOrder(adkcontext.WithoutCallPlace(ctx2))
}
