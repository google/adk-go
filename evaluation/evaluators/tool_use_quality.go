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

// ToolUseQualityEvaluator evaluates tool usage using custom rubrics.
// It evaluates tool usage quality using custom rubrics and LLM-as-Judge.
type ToolUseQualityEvaluator struct {
	judge         *llmjudge.Judge
	promptBuilder *llmjudge.PromptBuilder
}

// NewToolUseQualityEvaluator creates a new tool use quality evaluator.
func NewToolUseQualityEvaluator(config evaluation.EvaluatorConfig) (evaluation.Evaluator, error) {
	if config.LLM == nil {
		return nil, fmt.Errorf("LLM is required for tool use quality evaluation")
	}

	llm, ok := config.LLM.(model.LLM)
	if !ok {
		return nil, fmt.Errorf("LLM must implement model.LLM interface")
	}

	if config.JudgeModel == "" {
		return nil, fmt.Errorf("judge model name is required for tool use quality evaluation")
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

	return &ToolUseQualityEvaluator{
		judge:         judge,
		promptBuilder: llmjudge.NewPromptBuilder(),
	}, nil
}

// Evaluate assesses tool usage quality using rubrics.
func (e *ToolUseQualityEvaluator) Evaluate(ctx context.Context, params evaluation.EvaluateParams) (*evaluation.MetricResult, error) {
	if len(params.Actual) == 0 {
		return nil, fmt.Errorf("no actual invocations provided")
	}

	if len(params.Rubrics) == 0 {
		return nil, fmt.Errorf("no rubrics provided for tool use quality evaluation")
	}

	// Evaluate each invocation with tool calls
	var totalScore float64
	numEvaluations := 0

	for _, invocation := range params.Actual {
		// Skip invocations without tool calls
		if len(invocation.ToolCalls) == 0 {
			continue
		}

		// Build tool use quality prompt
		prompt := e.promptBuilder.BuildToolUseQualityPrompt(
			invocation.UserQuery,
			invocation.ToolCalls,
			params.Rubrics,
		)

		// Evaluate using LLM judge with rubrics
		result, err := e.judge.EvaluateRubrics(
			ctx,
			prompt,
			params.Rubrics,
			evaluation.MetricToolUseQuality,
		)
		if err != nil {
			// Log error but continue with other invocations
			continue
		}

		totalScore += result.Score
		numEvaluations++
	}

	if numEvaluations == 0 {
		return nil, fmt.Errorf("no invocations with tool calls to evaluate")
	}

	avgScore := totalScore / float64(numEvaluations)

	// Determine status based on threshold
	status := evaluation.EvalStatusPassed
	if params.Criterion != nil && params.Criterion.GetThreshold() != nil {
		threshold := params.Criterion.GetThreshold()
		if avgScore < threshold.MinScore {
			status = evaluation.EvalStatusFailed
		}
	}

	return &evaluation.MetricResult{
		MetricType: evaluation.MetricToolUseQuality,
		Score:      avgScore,
		Status:     status,
	}, nil
}

// MetricType returns the metric type.
func (e *ToolUseQualityEvaluator) MetricType() evaluation.MetricType {
	return evaluation.MetricToolUseQuality
}

// RequiresExpected indicates that this evaluator does not need expected responses.
func (e *ToolUseQualityEvaluator) RequiresExpected() bool {
	return false
}
