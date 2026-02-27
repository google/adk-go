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

import "context"

// Evaluator defines the core evaluation interface.
// All metric evaluators must implement this interface.
type Evaluator interface {
	// Evaluate runs evaluation on agent invocations.
	// It returns detailed metric results including scores, status, and rubric breakdowns.
	Evaluate(ctx context.Context, params EvaluateParams) (*MetricResult, error)

	// MetricType returns the metric this evaluator produces.
	MetricType() MetricType

	// RequiresExpected indicates if expected invocations are needed for evaluation.
	RequiresExpected() bool
}

// EvaluateParams encapsulates all parameters needed for evaluation.
type EvaluateParams struct {
	// Actual invocations from the agent being tested
	Actual []Invocation

	// Expected/reference invocations (optional for some metrics)
	Expected []Invocation

	// Rubrics for this evaluation (optional, used by rubric-based evaluators)
	Rubrics map[string]Rubric

	// Criterion defines the evaluation threshold and configuration
	Criterion Criterion

	// Context provides conversation and agent context
	Context EvaluationContext
}

// EvaluatorFactory creates evaluators for specific metrics.
type EvaluatorFactory func(config EvaluatorConfig) (Evaluator, error)

// EvaluatorConfig provides configuration for evaluator creation.
type EvaluatorConfig struct {
	// LLM is the LLM instance to use for LLM-as-Judge evaluators
	LLM interface{} // model.LLM interface to avoid circular dependency

	// JudgeModel is the LLM model name to use for LLM-as-Judge evaluators
	JudgeModel string

	// NumSamples is the number of evaluation samples for LLM-based metrics
	NumSamples int

	// ModelConfig contains model-specific configuration (temperature, top_p, etc.)
	ModelConfig map[string]any

	// CustomPrompts allows overriding default evaluation prompts
	CustomPrompts map[string]string
}
