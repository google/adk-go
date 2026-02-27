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

package openai

import (
	"context"
	"fmt"
)

const (
	// DefaultMaxIterations is the default maximum number of multi-turn iterations
	DefaultMaxIterations = 10

	// ContextKeyMaxIterations is the context key for max iterations override
	ContextKeyMaxIterations = "openai_max_iterations"

	// ContextKeyCurrentIteration is the context key for tracking current iteration
	ContextKeyCurrentIteration = "openai_current_iteration"
)

// IterationGuard protects against infinite loops in multi-turn conversations.
type IterationGuard struct {
	maxIterations int
}

// NewIterationGuard creates a new iteration guard with the specified max iterations.
func NewIterationGuard(maxIterations int) *IterationGuard {
	if maxIterations <= 0 {
		maxIterations = DefaultMaxIterations
	}

	return &IterationGuard{
		maxIterations: maxIterations,
	}
}

// CheckIteration checks if we've exceeded the maximum number of iterations.
// Returns an error if the limit has been reached.
func (ig *IterationGuard) CheckIteration(ctx context.Context) error {
	currentIteration := ig.GetCurrentIteration(ctx)

	maxIterations := ig.maxIterations

	// Check for context override
	if ctxMax, ok := ctx.Value(ContextKeyMaxIterations).(int); ok && ctxMax > 0 {
		maxIterations = ctxMax
	}

	if currentIteration >= maxIterations {
		return fmt.Errorf("maximum iterations exceeded: %d/%d (possible infinite loop)", currentIteration, maxIterations)
	}

	return nil
}

// IncrementIteration increments the iteration counter in the context and returns a new context.
func (ig *IterationGuard) IncrementIteration(ctx context.Context) context.Context {
	current := ig.GetCurrentIteration(ctx)
	return context.WithValue(ctx, ContextKeyCurrentIteration, current+1)
}

// GetCurrentIteration retrieves the current iteration count from the context.
func (ig *IterationGuard) GetCurrentIteration(ctx context.Context) int {
	if iteration, ok := ctx.Value(ContextKeyCurrentIteration).(int); ok {
		return iteration
	}
	return 0
}

// ResetIteration resets the iteration counter in the context.
func (ig *IterationGuard) ResetIteration(ctx context.Context) context.Context {
	return context.WithValue(ctx, ContextKeyCurrentIteration, 0)
}

// WithMaxIterations creates a new context with a custom max iterations value.
func WithMaxIterations(ctx context.Context, maxIterations int) context.Context {
	return context.WithValue(ctx, ContextKeyMaxIterations, maxIterations)
}

// GetMaxIterations retrieves the max iterations from context or returns default.
func GetMaxIterations(ctx context.Context) int {
	if max, ok := ctx.Value(ContextKeyMaxIterations).(int); ok && max > 0 {
		return max
	}
	return DefaultMaxIterations
}
