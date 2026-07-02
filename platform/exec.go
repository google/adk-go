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

package platform

import (
	"context"
	"sync"
)

// TaskRunner runs a batch of independent tasks, invoking each one exactly
// once and passing it a context.Context. An implementation must block until
// every task has completed.
//
// Each task receives a context so the runner can give it its own per-task
// context rather than sharing one. The default runner passes the same ctx it
// was given to every task; a custom runner may instead derive a distinct
// context per task.
//
// The default runner (used when none is installed on the context) runs the
// tasks concurrently, one goroutine per task. A caller that wants to control
// how the fan-out executes — for example to bound concurrency, dispatch onto
// its own executor, or run the tasks sequentially in a single goroutine — can
// install its own runner with WithTaskRunner.
type TaskRunner func(ctx context.Context, tasks []func(context.Context))

// taskRunnerKey is the context key under which a TaskRunner is stored.
type taskRunnerKey struct{}

// WithTaskRunner returns a copy of ctx that carries runner. RunTasks called
// with the returned context, or any context derived from it, uses runner
// instead of the default goroutine-based runner. A nil runner is ignored by
// RunTasks, which then falls back to the default.
//
// WithTaskRunner is the concurrency analog of WithTimeProvider and
// WithUUIDProvider: it lets a host environment substitute its own execution
// strategy for ADK's internal fan-out without the ADK runtime taking a
// dependency on that environment.
func WithTaskRunner(ctx context.Context, runner TaskRunner) context.Context {
	return context.WithValue(ctx, taskRunnerKey{}, runner)
}

// RunTasks runs every task in tasks and blocks until all of them complete. If
// ctx carries a TaskRunner installed with WithTaskRunner, that runner is used;
// otherwise RunTasks runs the tasks concurrently on one goroutine each.
//
// Each task is invoked with a context.Context. Under the default runner every
// task receives ctx; an installed runner may instead pass each task its own
// per-task context. Under the default runner, the tasks run concurrently and
// must be safe to do so. An empty or nil slice returns without doing anything.
func RunTasks(ctx context.Context, tasks []func(context.Context)) {
	if ctx != nil {
		if r, ok := ctx.Value(taskRunnerKey{}).(TaskRunner); ok && r != nil {
			r(ctx, tasks)
			return
		}
	}
	if len(tasks) == 0 {
		return
	}
	var wg sync.WaitGroup
	wg.Add(len(tasks))
	for _, task := range tasks {
		go func() {
			defer wg.Done()
			task(ctx)
		}()
	}
	wg.Wait()
}
