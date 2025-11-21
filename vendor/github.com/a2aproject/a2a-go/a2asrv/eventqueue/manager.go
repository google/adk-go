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

	"github.com/a2aproject/a2a-go/a2a"
)

// Manager manages event queues on a per-task basis.
// It provides lifecycle management for task-specific event queues,
// enabling multiple clients to attach to the same task's event stream.
type Manager interface {
	// GetOrCreate returns an existing queue if one exists, or creates a new one.
	GetOrCreate(ctx context.Context, taskID a2a.TaskID) (Queue, error)

	// Get returns an existing queue if one exists.
	Get(ctx context.Context, taskId a2a.TaskID) (Queue, bool)

	// Destroy closes the queue for the specified task and frees all associates resources.
	Destroy(ctx context.Context, taskID a2a.TaskID) error
}
