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

package a2asrv

import (
	"context"

	"github.com/a2aproject/a2a-go/a2a"
)

// PushSender defines the interface for sending push notifications
// about task state changes to external endpoints.
type PushSender interface {
	// SendPush sends a push notification containing the latest task state. If an error is returned execution is stopped.
	SendPush(ctx context.Context, config *a2a.PushConfig, task *a2a.Task) error
}

// PushConfigStore manages push notification configurations for tasks.
type PushConfigStore interface {
	// Save creates or updates a push notification configuration for a task. If no ID is set
	// on the provided config, it will have a store-generated ID for the returned config.
	// PushConfig has an ID and a Task can have multiple associated configurations.
	Save(ctx context.Context, taskID a2a.TaskID, config *a2a.PushConfig) (*a2a.PushConfig, error)

	// Get retrieves a push configuration registered for a Task with the given configID.
	Get(ctx context.Context, taskID a2a.TaskID, configID string) (*a2a.PushConfig, error)

	// List retrieves all registered push configurations for a Task. Returning an error stops the execution.
	List(ctx context.Context, taskID a2a.TaskID) ([]*a2a.PushConfig, error)

	// Delete removes a push configuration registered for a Task with the given configID.
	Delete(ctx context.Context, taskID a2a.TaskID, configID string) error

	// DeleteAll removes all registered push configurations of a Task.
	DeleteAll(ctx context.Context, taskID a2a.TaskID) error
}

// TaskStore provides storage for [a2a.Task]-s.
type TaskStore interface {
	// Save stores a task.
	Save(ctx context.Context, task *a2a.Task) error

	// Get retrieves a task by ID. If a Task doesn't exist the method should return [a2a.ErrTaskNotFound].
	Get(ctx context.Context, taskID a2a.TaskID) (*a2a.Task, error)
}
