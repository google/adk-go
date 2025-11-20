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

package push

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"sync"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/internal/utils"
	"github.com/google/uuid"
)

// ErrPushConfigNotFound indicates that a push config with the provided ID was not found.
var ErrPushConfigNotFound = errors.New("push config not found")

// InMemoryPushConfigStore implements a2asrv.PushConfigStore.
type InMemoryPushConfigStore struct {
	mu      sync.RWMutex
	configs map[a2a.TaskID]map[string]*a2a.PushConfig
}

// NewInMemoryStore creates an empty store.
func NewInMemoryStore() *InMemoryPushConfigStore {
	return &InMemoryPushConfigStore{
		configs: make(map[a2a.TaskID]map[string]*a2a.PushConfig),
	}
}

// NewID creates a new random identifier.
func newID() string {
	return uuid.NewString()
}

func validateConfig(config *a2a.PushConfig) error {
	if config == nil {
		return errors.New("push config cannot be nil")
	}
	if config.URL == "" {
		return errors.New("push config endpoint cannot be empty")
	}
	if _, err := url.ParseRequestURI(config.URL); err != nil {
		return fmt.Errorf("invalid push config endpoint URL: %w", err)
	}
	return nil
}

// Save adds a copy of push config to the store.
func (s *InMemoryPushConfigStore) Save(ctx context.Context, taskID a2a.TaskID, config *a2a.PushConfig) (*a2a.PushConfig, error) {
	if err := validateConfig(config); err != nil {
		return nil, fmt.Errorf("%w: %w", a2a.ErrInvalidParams, err)
	}

	toSave, err := utils.DeepCopy(config)
	if err != nil {
		return nil, err
	}

	if toSave.ID == "" {
		toSave.ID = newID()
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.configs[taskID]; !ok {
		s.configs[taskID] = make(map[string]*a2a.PushConfig)
	}
	s.configs[taskID][toSave.ID] = toSave

	savedCopy, err := utils.DeepCopy(toSave)
	if err != nil {
		return nil, err
	}

	return savedCopy, nil
}

// Get returns a copy of stored config for a task and with given ID.
func (s *InMemoryPushConfigStore) Get(ctx context.Context, taskID a2a.TaskID, configID string) (*a2a.PushConfig, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if configs, ok := s.configs[taskID]; ok {
		if config, ok := configs[configID]; ok {
			return utils.DeepCopy(config)
		}
	}

	return nil, ErrPushConfigNotFound
}

// List returns a copy of stored configs for a task.
func (s *InMemoryPushConfigStore) List(ctx context.Context, taskID a2a.TaskID) ([]*a2a.PushConfig, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	configs, ok := s.configs[taskID]
	if !ok {
		return []*a2a.PushConfig{}, nil
	}

	result := make([]*a2a.PushConfig, 0, len(configs))
	for _, config := range configs {
		copy, err := utils.DeepCopy(config)
		if err != nil {
			return nil, err
		}
		result = append(result, copy)
	}
	return result, nil
}

// Delete removes a single config from a store.
func (s *InMemoryPushConfigStore) Delete(ctx context.Context, taskID a2a.TaskID, configID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if configs, ok := s.configs[taskID]; ok {
		delete(configs, configID)
	}
	return nil
}

// DeleteAll removes all stored configs for a task.
func (s *InMemoryPushConfigStore) DeleteAll(ctx context.Context, taskID a2a.TaskID) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.configs, taskID)
	return nil
}
