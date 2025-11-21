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

package taskexec

import (
	"context"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
)

// Processor implementation handles events produced during AgentExecution.
type Processor interface {
	// Process is called for each event produced by the started Execution.
	// Execution finishes when either a non-nil result or a non-nil error is returned.
	// the terminal value becomes the result of the execution.
	// Called in a separate goroutine.
	Process(context.Context, a2a.Event) (*a2a.SendMessageResult, error)
}

// Executor implementation starts an agent execution.
type Executor interface {
	Processor
	// Start starts publishing events to the queue. Called in a separate goroutine.
	Execute(context.Context, eventqueue.Queue) error
}

// Canceler implementation sends a Task cancelation signal.
type Canceler interface {
	Processor
	// Cancel attempts to cancel a Task.
	// Expected to produce a Task update event with canceled state.
	Cancel(context.Context, eventqueue.Queue) error
}
