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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"google.golang.org/adk/evaluation"
)

// FileStorage provides file-based storage for eval sets and results.
// Files are stored as JSON in the specified directory structure:
//
//	<basePath>/
//	  eval_sets/
//	    <appName>/
//	      <evalSetID>.json
//	  results/
//	    <resultID>.json
type FileStorage struct {
	mu       sync.RWMutex
	basePath string
}

// NewFileStorage creates a new file-based storage instance.
func NewFileStorage(basePath string) (*FileStorage, error) {
	// Create directory structure
	if err := os.MkdirAll(filepath.Join(basePath, "eval_sets"), 0755); err != nil {
		return nil, fmt.Errorf("failed to create eval_sets directory: %w", err)
	}

	if err := os.MkdirAll(filepath.Join(basePath, "results"), 0755); err != nil {
		return nil, fmt.Errorf("failed to create results directory: %w", err)
	}

	return &FileStorage{
		basePath: basePath,
	}, nil
}

// SaveEvalSet stores an evaluation set.
func (f *FileStorage) SaveEvalSet(ctx context.Context, appName string, evalSet *evaluation.EvalSet) error {
	if evalSet == nil || evalSet.ID == "" {
		return evaluation.ErrInvalidInput
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	// Create app directory
	appDir := filepath.Join(f.basePath, "eval_sets", appName)
	if err := os.MkdirAll(appDir, 0755); err != nil {
		return fmt.Errorf("failed to create app directory: %w", err)
	}

	// Write eval set to file
	filePath := filepath.Join(appDir, fmt.Sprintf("%s.json", evalSet.ID))
	data, err := json.MarshalIndent(evalSet, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal eval set: %w", err)
	}

	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write eval set file: %w", err)
	}

	return nil
}

// GetEvalSet retrieves an evaluation set by ID.
func (f *FileStorage) GetEvalSet(ctx context.Context, appName, evalSetID string) (*evaluation.EvalSet, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	filePath := filepath.Join(f.basePath, "eval_sets", appName, fmt.Sprintf("%s.json", evalSetID))

	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, evaluation.ErrNotFound
		}
		return nil, fmt.Errorf("failed to read eval set file: %w", err)
	}

	var evalSet evaluation.EvalSet
	if err := json.Unmarshal(data, &evalSet); err != nil {
		return nil, fmt.Errorf("failed to unmarshal eval set: %w", err)
	}

	return &evalSet, nil
}

// ListEvalSets returns all evaluation sets for an application.
func (f *FileStorage) ListEvalSets(ctx context.Context, appName string) ([]evaluation.EvalSet, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	appDir := filepath.Join(f.basePath, "eval_sets", appName)

	// Check if directory exists
	if _, err := os.Stat(appDir); os.IsNotExist(err) {
		return []evaluation.EvalSet{}, nil
	}

	entries, err := os.ReadDir(appDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read app directory: %w", err)
	}

	var evalSets []evaluation.EvalSet
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		data, err := os.ReadFile(filepath.Join(appDir, entry.Name()))
		if err != nil {
			continue
		}

		var evalSet evaluation.EvalSet
		if err := json.Unmarshal(data, &evalSet); err != nil {
			continue
		}

		evalSets = append(evalSets, evalSet)
	}

	return evalSets, nil
}

// DeleteEvalSet removes an evaluation set.
func (f *FileStorage) DeleteEvalSet(ctx context.Context, appName, evalSetID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	filePath := filepath.Join(f.basePath, "eval_sets", appName, fmt.Sprintf("%s.json", evalSetID))

	if err := os.Remove(filePath); err != nil {
		if os.IsNotExist(err) {
			return evaluation.ErrNotFound
		}
		return fmt.Errorf("failed to delete eval set file: %w", err)
	}

	return nil
}

// SaveEvalSetResult stores evaluation results.
func (f *FileStorage) SaveEvalSetResult(ctx context.Context, result *evaluation.EvalSetResult) error {
	if result == nil || result.EvalSetResultID == "" {
		return evaluation.ErrInvalidInput
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	filePath := filepath.Join(f.basePath, "results", fmt.Sprintf("%s.json", result.EvalSetResultID))

	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal result: %w", err)
	}

	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write result file: %w", err)
	}

	return nil
}

// GetEvalSetResult retrieves evaluation results by ID.
func (f *FileStorage) GetEvalSetResult(ctx context.Context, resultID string) (*evaluation.EvalSetResult, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	filePath := filepath.Join(f.basePath, "results", fmt.Sprintf("%s.json", resultID))

	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, evaluation.ErrNotFound
		}
		return nil, fmt.Errorf("failed to read result file: %w", err)
	}

	var result evaluation.EvalSetResult
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal result: %w", err)
	}

	return &result, nil
}

// ListEvalSetResults returns all evaluation results for an application.
func (f *FileStorage) ListEvalSetResults(ctx context.Context, appName string) ([]evaluation.EvalSetResult, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	resultsDir := filepath.Join(f.basePath, "results")

	entries, err := os.ReadDir(resultsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return []evaluation.EvalSetResult{}, nil
		}
		return nil, fmt.Errorf("failed to read results directory: %w", err)
	}

	var results []evaluation.EvalSetResult
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		data, err := os.ReadFile(filepath.Join(resultsDir, entry.Name()))
		if err != nil {
			continue
		}

		var result evaluation.EvalSetResult
		if err := json.Unmarshal(data, &result); err != nil {
			continue
		}

		// Filter by app name by checking if eval set belongs to this app
		evalSet, err := f.GetEvalSet(ctx, appName, result.EvalSetID)
		if err == nil && evalSet != nil {
			results = append(results, result)
		}
	}

	return results, nil
}

// GetEvalSetResultsByEvalSetID returns all results for a specific eval set.
func (f *FileStorage) GetEvalSetResultsByEvalSetID(ctx context.Context, evalSetID string) ([]evaluation.EvalSetResult, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	resultsDir := filepath.Join(f.basePath, "results")

	entries, err := os.ReadDir(resultsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return []evaluation.EvalSetResult{}, nil
		}
		return nil, fmt.Errorf("failed to read results directory: %w", err)
	}

	var results []evaluation.EvalSetResult
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		data, err := os.ReadFile(filepath.Join(resultsDir, entry.Name()))
		if err != nil {
			continue
		}

		var result evaluation.EvalSetResult
		if err := json.Unmarshal(data, &result); err != nil {
			continue
		}

		if result.EvalSetID == evalSetID {
			results = append(results, result)
		}
	}

	return results, nil
}
