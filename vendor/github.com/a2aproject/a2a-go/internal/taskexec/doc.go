// Copyright 2025 The A2A Authors
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

// Package taskexec provides a concurrent agent invocation manager.
// The manager enforces concurrency control on task level and guarantees that at any given moment
// there's only one goroutine which is mutating the state of an [a2a.Task].
//
// For every [Execution] the [Manager] starts two goroutines in an [errgroup.Group]:
//   - One calls [Executor] and starts producing events writing them to an [eventqueue.Queue].
//   - The second one reads events in a loop and passes them through [Processor] responsible for deciding when to stop.
//
// For cancelations the handling is different depending on whether there exists a concurrent execution:
//   - When a concurrent execution exists, only one goroutine is started which calls [Canceler]. It writes an event to the same queue
//     the execution is using and expects the concurrently-running event consumer to handle it.
//   - When there's no concurrent execution, the mechanism is the same as for the execution: one goroutine which calls cancel and the second one
//     which processes events.
//
// Event consumer continues to run waiting for a terminal event to be produced, even if execution goroutine finished.
//
// [Sequence diagram].
//
// [Sequence diagram]: https://github.com/a2aproject/a2a-go/docs/diagrams/task-execution/task_execution.png
package taskexec
