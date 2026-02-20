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

	"google.golang.org/adk/evaluation"
	"google.golang.org/adk/evaluation/llmjudge"
	"google.golang.org/adk/model"
)

// ResponseQualityEvaluator assesses response quality using rubrics.
// It evaluates response quality using custom rubrics and LLM-as-Judge.
type ResponseQualityEvaluator struct {
	judge         *llmjudge.Judge
	promptBuilder *llmjudge.PromptBuilder
}

// NewResponseQualityEvaluator creates a new response quality evaluator.
func NewResponseQualityEvaluator(config evaluation.EvaluatorConfig) (evaluation.Evaluator, error) {
	if config.LLM == nil {
		return nil, fmt.Errorf("LLM is required for response quality evaluation")
	}

	llm, ok := config.LLM.(model.LLM)
	if !ok {
		return nil, fmt.Errorf("LLM must implement model.LLM interface")
	}

	if config.JudgeModel == "" {
		return nil, fmt.Errorf("judge model name is required for response quality evaluation")
	}

	numSamples := config.NumSamples
	if numSamples <= 0 {
		numSamples = 3 // Default to 3 samples for rubric-based evaluation
	}

	judge := llmjudge.NewJudge(llmjudge.Config{
		LLM:        llm,
		ModelName:  config.JudgeModel,
		NumSamples: numSamples,
	})

	return &ResponseQualityEvaluator{
		judge:         judge,
		promptBuilder: llmjudge.NewPromptBuilder(),
	}, nil
}

// Evaluate assesses response quality using rubrics.
func (e *ResponseQualityEvaluator) Evaluate(ctx context.Context, params evaluation.EvaluateParams) (*evaluation.MetricResult, error) {
	if len(params.Actual) == 0 {
		return nil, fmt.Errorf("no actual invocations provided")
	}

	if len(params.Rubrics) == 0 {
		return nil, fmt.Errorf("no rubrics provided for response quality evaluation")
	}

	// Evaluate the final (last) invocation
	actualIdx := len(params.Actual) - 1
	finalInvocation := params.Actual[actualIdx]

	// Build response quality prompt
	prompt := e.promptBuilder.BuildResponseQualityPrompt(
		finalInvocation.UserQuery,
		finalInvocation.AgentResponse,
		params.Rubrics,
	)

	// Evaluate using LLM judge with rubrics
	result, err := e.judge.EvaluateRubrics(
		ctx,
		prompt,
		params.Rubrics,
		evaluation.MetricResponseQuality,
	)
	if err != nil {
		return nil, fmt.Errorf("LLM judge evaluation failed: %w", err)
	}

	// Apply threshold
	if params.Criterion != nil && params.Criterion.GetThreshold() != nil {
		threshold := params.Criterion.GetThreshold()
		if result.Score < threshold.MinScore {
			result.Status = evaluation.EvalStatusFailed
		} else {
			result.Status = evaluation.EvalStatusPassed
		}
	}

	return result, nil
}

// MetricType returns the metric type.
func (e *ResponseQualityEvaluator) MetricType() evaluation.MetricType {
	return evaluation.MetricResponseQuality
}

// RequiresExpected indicates that this evaluator does not need expected responses.
func (e *ResponseQualityEvaluator) RequiresExpected() bool {
	return false
}
