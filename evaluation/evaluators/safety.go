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

// SafetyEvaluator evaluates response safety and harmlessness.
// It evaluates response safety and harmlessness using LLM-as-Judge.
type SafetyEvaluator struct {
	judge         *llmjudge.Judge
	promptBuilder *llmjudge.PromptBuilder
}

// NewSafetyEvaluator creates a new safety evaluator.
func NewSafetyEvaluator(config evaluation.EvaluatorConfig) (evaluation.Evaluator, error) {
	if config.LLM == nil {
		return nil, fmt.Errorf("LLM is required for safety evaluation")
	}

	llm, ok := config.LLM.(model.LLM)
	if !ok {
		return nil, fmt.Errorf("LLM must implement model.LLM interface")
	}

	if config.JudgeModel == "" {
		return nil, fmt.Errorf("judge model name is required for safety evaluation")
	}

	numSamples := config.NumSamples
	if numSamples <= 0 {
		numSamples = 3 // Default to 3 samples for safety evaluation
	}

	judge := llmjudge.NewJudge(llmjudge.Config{
		LLM:        llm,
		ModelName:  config.JudgeModel,
		NumSamples: numSamples,
	})

	return &SafetyEvaluator{
		judge:         judge,
		promptBuilder: llmjudge.NewPromptBuilder(),
	}, nil
}

// Evaluate assesses response safety.
func (e *SafetyEvaluator) Evaluate(ctx context.Context, params evaluation.EvaluateParams) (*evaluation.MetricResult, error) {
	if len(params.Actual) == 0 {
		return nil, fmt.Errorf("no actual invocations provided")
	}

	// Evaluate each invocation for safety
	var totalScore float64
	numEvaluations := 0

	for _, invocation := range params.Actual {
		// Build safety evaluation prompt
		prompt := e.promptBuilder.BuildSafetyPrompt(invocation.AgentResponse)

		// Evaluate using LLM judge with verdict (yes = safe, no = unsafe)
		result, err := e.judge.EvaluateVerdict(ctx, prompt, evaluation.MetricSafety)
		if err != nil {
			// Log error but continue with other invocations
			continue
		}

		totalScore += result.Score
		numEvaluations++
	}

	if numEvaluations == 0 {
		return nil, fmt.Errorf("failed to evaluate any invocations for safety")
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
		MetricType: evaluation.MetricSafety,
		Score:      avgScore,
		Status:     status,
	}, nil
}

// MetricType returns the metric type.
func (e *SafetyEvaluator) MetricType() evaluation.MetricType {
	return evaluation.MetricSafety
}

// RequiresExpected indicates that this evaluator does not need expected responses.
func (e *SafetyEvaluator) RequiresExpected() bool {
	return false
}
