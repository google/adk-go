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

package platform_test

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/platform"
)

func TestRunTasksDefaultRunsAllTasks(t *testing.T) {
	const n = 16
	var count atomic.Int64
	seen := make([]bool, n)
	var mu sync.Mutex

	tasks := make([]func(context.Context), n)
	for i := range tasks {
		tasks[i] = func(context.Context) {
			count.Add(1)
			mu.Lock()
			seen[i] = true
			mu.Unlock()
		}
	}
	platform.RunTasks(context.Background(), tasks)

	if got := count.Load(); got != n {
		t.Fatalf("ran %d tasks, want %d", got, n)
	}
	for i, ok := range seen {
		if !ok {
			t.Errorf("task %d did not run", i)
		}
	}
}

func TestRunTasksEmptyIsNoOp(t *testing.T) {
	// Neither a nil nor an empty slice should block or panic.
	platform.RunTasks(context.Background(), nil)
	platform.RunTasks(context.Background(), []func(context.Context){})
}

func TestWithTaskRunnerIsUsed(t *testing.T) {
	const n = 5
	var order []int
	used := false

	// A sequential runner: runs every task inline, in slice order, on the
	// caller's goroutine. This is the shape a caller installs to keep ADK's
	// tool fan-out off goroutines, e.g. to bound concurrency or run on its own
	// executor.
	seq := func(ctx context.Context, tasks []func(context.Context)) {
		used = true
		for _, task := range tasks {
			task(ctx)
		}
	}

	ctx := platform.WithTaskRunner(context.Background(), seq)
	tasks := make([]func(context.Context), n)
	for i := range tasks {
		tasks[i] = func(context.Context) {
			order = append(order, i) // safe: no concurrency under the sequential runner
		}
	}
	platform.RunTasks(ctx, tasks)

	if !used {
		t.Fatal("installed TaskRunner was not used")
	}
	for i := 0; i < n; i++ {
		if order[i] != i {
			t.Fatalf("sequential runner ran out of order: got %v", order)
		}
	}
}

func TestWithTaskRunnerNilFallsBack(t *testing.T) {
	var count atomic.Int64
	ctx := platform.WithTaskRunner(context.Background(), nil)
	tasks := []func(context.Context){
		func(context.Context) { count.Add(1) },
		func(context.Context) { count.Add(1) },
		func(context.Context) { count.Add(1) },
	}
	platform.RunTasks(ctx, tasks)
	if got := count.Load(); got != 3 {
		t.Fatalf("ran %d tasks, want 3", got)
	}
}

// sentinelKey is a context key used by the per-task context tests.
type sentinelKey struct{}

func TestRunTasksDefaultPassesContext(t *testing.T) {
	// With no installed runner, the default runner must invoke every task with
	// a context derived from (carrying the values of) the parent ctx.
	const n = 8
	const want = "sentinel-value"
	parent := context.WithValue(context.Background(), sentinelKey{}, want)

	var mu sync.Mutex
	got := make([]any, n)
	tasks := make([]func(context.Context), n)
	for i := range tasks {
		tasks[i] = func(taskCtx context.Context) {
			v := taskCtx.Value(sentinelKey{})
			mu.Lock()
			got[i] = v
			mu.Unlock()
		}
	}
	platform.RunTasks(parent, tasks)

	for i, v := range got {
		if v != want {
			t.Errorf("task %d received sentinel %v, want %q", i, v, want)
		}
	}
}

func TestWithTaskRunnerReceivesPerTaskContext(t *testing.T) {
	// An installed runner can hand each task its own distinct context, e.g. to
	// scope per-task cancellation, deadlines, or values to each fanned-out task
	// instead of sharing one.
	const n = 6

	// perTask gives task i a context carrying the value i.
	perTask := func(ctx context.Context, tasks []func(context.Context)) {
		for i, task := range tasks {
			task(context.WithValue(ctx, sentinelKey{}, i))
		}
	}

	ctx := platform.WithTaskRunner(context.Background(), perTask)
	got := make([]any, n)
	tasks := make([]func(context.Context), n)
	for i := range tasks {
		tasks[i] = func(taskCtx context.Context) {
			got[i] = taskCtx.Value(sentinelKey{}) // safe: sequential, no concurrency
		}
	}
	platform.RunTasks(ctx, tasks)

	for i, v := range got {
		if v != i {
			t.Errorf("task %d observed per-task value %v, want %d", i, v, i)
		}
	}
}
