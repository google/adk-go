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

package taskstore

import (
	"context"
	"encoding/gob"
	"sync"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/internal/utils"
)

// Mem stores deep-copied [a2a.Task]-s in memory.
type Mem struct {
	mu    sync.RWMutex
	tasks map[a2a.TaskID]*a2a.Task
}

func init() {
	gob.Register(map[string]any{})
	gob.Register([]any{})
}

// NewMem creates an empty [Mem] store.
func NewMem() *Mem {
	return &Mem{
		tasks: make(map[a2a.TaskID]*a2a.Task),
	}
}

func (s *Mem) Save(ctx context.Context, task *a2a.Task) error {
	if err := validateTask(task); err != nil {
		return err
	}

	copy, err := utils.DeepCopy(task)
	if err != nil {
		return err
	}

	s.mu.Lock()
	s.tasks[task.ID] = copy
	s.mu.Unlock()

	return nil
}

func (s *Mem) Get(ctx context.Context, taskID a2a.TaskID) (*a2a.Task, error) {
	s.mu.RLock()
	task, ok := s.tasks[taskID]
	s.mu.RUnlock()

	if !ok {
		return nil, a2a.ErrTaskNotFound
	}

	return utils.DeepCopy(task)
}
