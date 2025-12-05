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

package evaluation

import (
	"context"
	"errors"
)

var (
	// ErrNotFound indicates the requested resource was not found.
	ErrNotFound = errors.New("evaluation: not found")

	// ErrAlreadyExists indicates the resource already exists.
	ErrAlreadyExists = errors.New("evaluation: already exists")

	// ErrInvalidInput indicates invalid input parameters.
	ErrInvalidInput = errors.New("evaluation: invalid input")
)

// Storage defines persistence for eval sets and results.
type Storage interface {
	// EvalSet operations

	// SaveEvalSet stores an evaluation set.
	SaveEvalSet(ctx context.Context, appName string, evalSet *EvalSet) error

	// GetEvalSet retrieves an evaluation set by ID.
	GetEvalSet(ctx context.Context, appName, evalSetID string) (*EvalSet, error)

	// ListEvalSets returns all evaluation sets for an application.
	ListEvalSets(ctx context.Context, appName string) ([]EvalSet, error)

	// DeleteEvalSet removes an evaluation set.
	DeleteEvalSet(ctx context.Context, appName, evalSetID string) error

	// EvalSetResult operations

	// SaveEvalSetResult stores evaluation results.
	SaveEvalSetResult(ctx context.Context, result *EvalSetResult) error

	// GetEvalSetResult retrieves evaluation results by ID.
	GetEvalSetResult(ctx context.Context, resultID string) (*EvalSetResult, error)

	// ListEvalSetResults returns all evaluation results for an application.
	ListEvalSetResults(ctx context.Context, appName string) ([]EvalSetResult, error)

	// GetEvalSetResultsByEvalSetID returns all results for a specific eval set.
	GetEvalSetResultsByEvalSetID(ctx context.Context, evalSetID string) ([]EvalSetResult, error)
}
