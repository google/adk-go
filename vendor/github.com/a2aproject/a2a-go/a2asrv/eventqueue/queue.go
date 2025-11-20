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

package eventqueue

import (
	"context"
	"errors"

	"github.com/a2aproject/a2a-go/a2a"
)

var (
	// ErrQueueClosed indicates that the event queue has been closed.
	ErrQueueClosed = errors.New("queue is closed")
)

// Reader defines the interface for reading events from a queue.
// A2A server stack reads events written by [a2asrv.AgentExecutor].
type Reader interface {
	// Read dequeues an event or blocks if the queue is empty.
	Read(ctx context.Context) (a2a.Event, error)
}

// Writer defines the interface for writing events to a queue.
// [a2asrv.AgentExecutor] translates agent responses to Messages, Tasks or Task update events.
type Writer interface {
	// Write enqueues an event or blocks if a bounded queue is full.
	Write(ctx context.Context, event a2a.Event) error
}

// Queue defines the interface for publishing and consuming
// events generated during agent execution.
type Queue interface {
	Reader
	Writer

	// Close shuts down a connection to the queue.
	Close() error
}
