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

package evaluators

import (
	"context"
	"fmt"
	"reflect"
	"time"

	"google.golang.org/adk/evaluation"
)

// ToolTrajectoryEvaluator implements TOOL_TRAJECTORY_AVG_SCORE.
// It validates that tool call sequences match expected trajectories exactly.
type ToolTrajectoryEvaluator struct{}

// NewToolTrajectoryEvaluator creates a new tool trajectory evaluator.
func NewToolTrajectoryEvaluator(config evaluation.EvaluatorConfig) (evaluation.Evaluator, error) {
	return &ToolTrajectoryEvaluator{}, nil
}

// Evaluate validates tool call trajectories.
func (e *ToolTrajectoryEvaluator) Evaluate(ctx context.Context, params evaluation.EvaluateParams) (*evaluation.MetricResult, error) {
	if len(params.Actual) == 0 {
		return nil, fmt.Errorf("no actual invocations provided")
	}

	if len(params.Expected) == 0 {
		return nil, fmt.Errorf("no expected invocations provided")
	}

	// Compare tool trajectories for each invocation
	var totalScore float64
	numComparisons := 0

	for i := 0; i < len(params.Actual) && i < len(params.Expected); i++ {
		actualCalls := params.Actual[i].ToolCalls
		expectedCalls := params.Expected[i].ToolCalls

		if e.trajectoryMatches(actualCalls, expectedCalls) {
			totalScore += 1.0
		}
		numComparisons++
	}

	if numComparisons == 0 {
		return nil, fmt.Errorf("no tool trajectories to compare")
	}

	avgScore := totalScore / float64(numComparisons)

	// Determine status based on threshold
	status := evaluation.EvalStatusPassed
	if params.Criterion != nil && params.Criterion.GetThreshold() != nil {
		threshold := params.Criterion.GetThreshold()
		if avgScore < threshold.MinScore {
			status = evaluation.EvalStatusFailed
		}
	}

	return &evaluation.MetricResult{
		MetricType:  evaluation.MetricToolTrajectoryAvgScore,
		Score:       avgScore,
		Status:      status,
		EvaluatedAt: time.Now(),
	}, nil
}

// MetricType returns the metric type.
func (e *ToolTrajectoryEvaluator) MetricType() evaluation.MetricType {
	return evaluation.MetricToolTrajectoryAvgScore
}

// RequiresExpected indicates that this evaluator needs expected tool calls.
func (e *ToolTrajectoryEvaluator) RequiresExpected() bool {
	return true
}

// trajectoryMatches checks if two tool call sequences match exactly.
func (e *ToolTrajectoryEvaluator) trajectoryMatches(actual, expected []evaluation.ToolCall) bool {
	// Sequences must have the same length
	if len(actual) != len(expected) {
		return false
	}

	// Compare each tool call in the sequence
	for i := range actual {
		if !e.toolCallMatches(actual[i], expected[i]) {
			return false
		}
	}

	return true
}

// toolCallMatches checks if two tool calls match.
func (e *ToolTrajectoryEvaluator) toolCallMatches(actual, expected evaluation.ToolCall) bool {
	// Tool names must match exactly
	if actual.ToolName != expected.ToolName {
		return false
	}

	// Arguments must match (deep equality)
	if !e.argumentsMatch(actual.Arguments, expected.Arguments) {
		return false
	}

	return true
}

// argumentsMatch checks if two argument maps match.
func (e *ToolTrajectoryEvaluator) argumentsMatch(actual, expected map[string]any) bool {
	// Both nil or empty
	if len(actual) == 0 && len(expected) == 0 {
		return true
	}

	// Different number of arguments
	if len(actual) != len(expected) {
		return false
	}

	// Deep equality check
	return reflect.DeepEqual(actual, expected)
}
