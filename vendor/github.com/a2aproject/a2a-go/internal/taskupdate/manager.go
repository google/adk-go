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

package taskupdate

import (
	"context"
	"fmt"
	"slices"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/internal/utils"
	"github.com/a2aproject/a2a-go/log"
)

// Saver is used for saving the [a2a.Task] after updating its state.
type Saver interface {
	Save(ctx context.Context, task *a2a.Task) error
}

// Manager is used for processing [a2a.Event] related to an [a2a.Task]. It updates
// the Task accordingly and uses [Saver] to store the new state.
type Manager struct {
	lastSavedTask *a2a.Task
	saver         Saver
}

// NewManager is a [Manager] constructor function.
func NewManager(saver Saver, task *a2a.Task) *Manager {
	return &Manager{lastSavedTask: task, saver: saver}
}

func (mgr *Manager) SetTaskFailed(ctx context.Context, cause error) {
	task := mgr.lastSavedTask
	task.Status = a2a.TaskStatus{State: a2a.TaskStateFailed}

	if _, err := mgr.saveTask(ctx, task); err != nil {
		log.Error(ctx, "failed to store failed task state", err)
	} else {
		// TODO(yarolegovich): consider storing cause.Error() as part of failed task
		log.Info(ctx, "task moved to failed state", "cause", cause)
	}
}

// Process validates the event associated with the managed [a2a.Task] and integrates the new state into it.
func (mgr *Manager) Process(ctx context.Context, event a2a.Event) (*a2a.Task, error) {
	if mgr.lastSavedTask == nil {
		return nil, fmt.Errorf("event processor Task not set")
	}

	switch v := event.(type) {
	case *a2a.Message:
		return mgr.lastSavedTask, nil

	case *a2a.Task:
		if err := mgr.validate(v.ID, v.ContextID); err != nil {
			return nil, err
		}
		return mgr.saveTask(ctx, v)

	case *a2a.TaskArtifactUpdateEvent:
		if err := mgr.validate(v.TaskID, v.ContextID); err != nil {
			return nil, err
		}
		return mgr.updateArtifact(ctx, v)

	case *a2a.TaskStatusUpdateEvent:
		if err := mgr.validate(v.TaskID, v.ContextID); err != nil {
			return nil, err
		}
		return mgr.updateStatus(ctx, v)

	default:
		return nil, fmt.Errorf("unexpected event type %T", v)
	}
}

func (mgr *Manager) updateArtifact(ctx context.Context, event *a2a.TaskArtifactUpdateEvent) (*a2a.Task, error) {
	task := mgr.lastSavedTask

	// The copy is required because the event will be passed to subscriber goroutines, while
	// the artifact might be modified in our goroutine by other TaskArtifactUpdateEvent-s.
	artifact, err := utils.DeepCopy(event.Artifact)
	if err != nil {
		return nil, fmt.Errorf("failed to copy artifact: %w", err)
	}

	updateIdx := slices.IndexFunc(task.Artifacts, func(a *a2a.Artifact) bool {
		return a.ID == artifact.ID
	})

	if updateIdx < 0 {
		if event.Append {
			return nil, fmt.Errorf("no artifact found for update")
		}
		task.Artifacts = append(task.Artifacts, artifact)
		return mgr.saveTask(ctx, task)
	}

	if !event.Append {
		task.Artifacts[updateIdx] = artifact
		return mgr.saveTask(ctx, task)
	}

	toUpdate := task.Artifacts[updateIdx]
	toUpdate.Parts = append(toUpdate.Parts, artifact.Parts...)
	if toUpdate.Metadata == nil && artifact.Metadata != nil {
		toUpdate.Metadata = make(map[string]any, len(artifact.Description))
	}
	for k, v := range artifact.Metadata {
		toUpdate.Metadata[k] = v
	}
	return mgr.saveTask(ctx, task)
}

func (mgr *Manager) updateStatus(ctx context.Context, event *a2a.TaskStatusUpdateEvent) (*a2a.Task, error) {
	task, err := utils.DeepCopy(mgr.lastSavedTask)
	if err != nil {
		return nil, err
	}

	if task.Status.Message != nil {
		task.History = append(task.History, task.Status.Message)
	}

	if event.Metadata != nil {
		if task.Metadata == nil {
			task.Metadata = make(map[string]any)
		}
		for k, v := range event.Metadata {
			task.Metadata[k] = v
		}
	}

	task.Status = event.Status

	return mgr.saveTask(ctx, task)
}

func (mgr *Manager) saveTask(ctx context.Context, task *a2a.Task) (*a2a.Task, error) {
	if err := mgr.saver.Save(ctx, task); err != nil {
		return nil, fmt.Errorf("failed to save task state: %w", err)
	}
	mgr.lastSavedTask = task
	return task, nil
}

func (mgr *Manager) validate(taskID a2a.TaskID, contextID string) error {
	if mgr.lastSavedTask.ID != taskID {
		return fmt.Errorf("task IDs don't match: %s != %s", mgr.lastSavedTask.ID, taskID)
	}
	if mgr.lastSavedTask.ContextID != contextID {
		return fmt.Errorf("context IDs don't match: %s != %s", mgr.lastSavedTask.ContextID, contextID)
	}
	return nil
}
