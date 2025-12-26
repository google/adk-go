// Copyright 2025 Google LLC
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

package storage

import (
	"context"
	"sync"

	"google.golang.org/adk/evaluation"
)

// MemoryStorage provides in-memory storage for eval sets and results.
// This implementation is suitable for testing and development.
type MemoryStorage struct {
	mu sync.RWMutex

	// evalSets maps appName -> evalSetID -> EvalSet
	evalSets map[string]map[string]*evaluation.EvalSet

	// results maps resultID -> EvalSetResult
	results map[string]*evaluation.EvalSetResult

	// resultsByApp maps appName -> []resultID
	resultsByApp map[string][]string

	// resultsByEvalSet maps evalSetID -> []resultID
	resultsByEvalSet map[string][]string
}

// NewMemoryStorage creates a new in-memory storage instance.
func NewMemoryStorage() *MemoryStorage {
	return &MemoryStorage{
		evalSets:         make(map[string]map[string]*evaluation.EvalSet),
		results:          make(map[string]*evaluation.EvalSetResult),
		resultsByApp:     make(map[string][]string),
		resultsByEvalSet: make(map[string][]string),
	}
}

// SaveEvalSet stores an evaluation set.
func (m *MemoryStorage) SaveEvalSet(ctx context.Context, appName string, evalSet *evaluation.EvalSet) error {
	if evalSet == nil {
		return evaluation.ErrInvalidInput
	}

	if evalSet.ID == "" {
		return evaluation.ErrInvalidInput
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.evalSets[appName]; !exists {
		m.evalSets[appName] = make(map[string]*evaluation.EvalSet)
	}

	// Deep copy to prevent external modifications
	copied := *evalSet
	m.evalSets[appName][evalSet.ID] = &copied

	return nil
}

// GetEvalSet retrieves an evaluation set by ID.
func (m *MemoryStorage) GetEvalSet(ctx context.Context, appName, evalSetID string) (*evaluation.EvalSet, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	appEvalSets, exists := m.evalSets[appName]
	if !exists {
		return nil, evaluation.ErrNotFound
	}

	evalSet, exists := appEvalSets[evalSetID]
	if !exists {
		return nil, evaluation.ErrNotFound
	}

	// Deep copy to prevent external modifications
	copied := *evalSet
	return &copied, nil
}

// ListEvalSets returns all evaluation sets for an application.
func (m *MemoryStorage) ListEvalSets(ctx context.Context, appName string) ([]evaluation.EvalSet, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	appEvalSets, exists := m.evalSets[appName]
	if !exists {
		return []evaluation.EvalSet{}, nil
	}

	sets := make([]evaluation.EvalSet, 0, len(appEvalSets))
	for _, evalSet := range appEvalSets {
		copied := *evalSet
		sets = append(sets, copied)
	}

	return sets, nil
}

// DeleteEvalSet removes an evaluation set.
func (m *MemoryStorage) DeleteEvalSet(ctx context.Context, appName, evalSetID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	appEvalSets, exists := m.evalSets[appName]
	if !exists {
		return evaluation.ErrNotFound
	}

	if _, exists := appEvalSets[evalSetID]; !exists {
		return evaluation.ErrNotFound
	}

	delete(appEvalSets, evalSetID)

	return nil
}

// SaveEvalSetResult stores evaluation results.
func (m *MemoryStorage) SaveEvalSetResult(ctx context.Context, result *evaluation.EvalSetResult) error {
	if result == nil {
		return evaluation.ErrInvalidInput
	}

	if result.EvalSetResultID == "" {
		return evaluation.ErrInvalidInput
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Deep copy to prevent external modifications
	copied := *result
	m.results[result.EvalSetResultID] = &copied

	// Index by eval set ID
	if result.EvalSetID != "" {
		m.resultsByEvalSet[result.EvalSetID] = append(
			m.resultsByEvalSet[result.EvalSetID],
			result.EvalSetResultID,
		)
	}

	// Determine app name from eval set
	for appName, appEvalSets := range m.evalSets {
		if _, exists := appEvalSets[result.EvalSetID]; exists {
			m.resultsByApp[appName] = append(
				m.resultsByApp[appName],
				result.EvalSetResultID,
			)
			break
		}
	}

	return nil
}

// GetEvalSetResult retrieves evaluation results by ID.
func (m *MemoryStorage) GetEvalSetResult(ctx context.Context, resultID string) (*evaluation.EvalSetResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result, exists := m.results[resultID]
	if !exists {
		return nil, evaluation.ErrNotFound
	}

	// Deep copy to prevent external modifications
	copied := *result
	return &copied, nil
}

// ListEvalSetResults returns all evaluation results for an application.
func (m *MemoryStorage) ListEvalSetResults(ctx context.Context, appName string) ([]evaluation.EvalSetResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	resultIDs, exists := m.resultsByApp[appName]
	if !exists {
		return []evaluation.EvalSetResult{}, nil
	}

	results := make([]evaluation.EvalSetResult, 0, len(resultIDs))
	for _, resultID := range resultIDs {
		if result, exists := m.results[resultID]; exists {
			copied := *result
			results = append(results, copied)
		}
	}

	return results, nil
}

// GetEvalSetResultsByEvalSetID returns all results for a specific eval set.
func (m *MemoryStorage) GetEvalSetResultsByEvalSetID(ctx context.Context, evalSetID string) ([]evaluation.EvalSetResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	resultIDs, exists := m.resultsByEvalSet[evalSetID]
	if !exists {
		return []evaluation.EvalSetResult{}, nil
	}

	results := make([]evaluation.EvalSetResult, 0, len(resultIDs))
	for _, resultID := range resultIDs {
		if result, exists := m.results[resultID]; exists {
			copied := *result
			results = append(results, copied)
		}
	}

	return results, nil
}
