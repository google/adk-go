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

// ResponseEvaluationScoreEvaluator implements RESPONSE_EVALUATION_SCORE.
// It assesses response coherence and quality on a 1-5 scale using LLM-as-Judge.
type ResponseEvaluationScoreEvaluator struct {
	judge         *llmjudge.Judge
	promptBuilder *llmjudge.PromptBuilder
}

// NewResponseEvaluationScoreEvaluator creates a new response evaluation score evaluator.
func NewResponseEvaluationScoreEvaluator(config evaluation.EvaluatorConfig) (evaluation.Evaluator, error) {
	if config.LLM == nil {
		return nil, fmt.Errorf("LLM is required for RESPONSE_EVALUATION_SCORE")
	}

	llm, ok := config.LLM.(model.LLM)
	if !ok {
		return nil, fmt.Errorf("LLM must implement model.LLM interface")
	}

	if config.JudgeModel == "" {
		return nil, fmt.Errorf("judge model name is required for RESPONSE_EVALUATION_SCORE")
	}

	numSamples := config.NumSamples
	if numSamples <= 0 {
		numSamples = 1 // Default to 1 sample for coherence
	}

	judge := llmjudge.NewJudge(llmjudge.Config{
		LLM:        llm,
		ModelName:  config.JudgeModel,
		NumSamples: numSamples,
	})

	return &ResponseEvaluationScoreEvaluator{
		judge:         judge,
		promptBuilder: llmjudge.NewPromptBuilder(),
	}, nil
}

// Evaluate assesses response coherence and quality.
func (e *ResponseEvaluationScoreEvaluator) Evaluate(ctx context.Context, params evaluation.EvaluateParams) (*evaluation.MetricResult, error) {
	if len(params.Actual) == 0 {
		return nil, fmt.Errorf("no actual invocations provided")
	}

	// Evaluate each invocation and average the scores
	var totalScore float64
	numEvaluations := 0

	for _, invocation := range params.Actual {
		// Build coherence evaluation prompt
		prompt := e.promptBuilder.BuildCoherencePrompt(
			invocation.UserQuery,
			invocation.AgentResponse,
		)

		// Evaluate using LLM judge
		result, err := e.judge.EvaluateScore(ctx, prompt, evaluation.MetricResponseEvaluationScore)
		if err != nil {
			// Log error but continue with other invocations
			continue
		}

		// Normalize score from 1-5 scale to 0-1 scale
		normalizedScore := (result.Score - 1.0) / 4.0
		totalScore += normalizedScore
		numEvaluations++
	}

	if numEvaluations == 0 {
		return nil, fmt.Errorf("failed to evaluate any invocations")
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
		MetricType: evaluation.MetricResponseEvaluationScore,
		Score:      avgScore,
		Status:     status,
	}, nil
}

// MetricType returns the metric type.
func (e *ResponseEvaluationScoreEvaluator) MetricType() evaluation.MetricType {
	return evaluation.MetricResponseEvaluationScore
}

// RequiresExpected indicates that this evaluator does not need expected responses.
func (e *ResponseEvaluationScoreEvaluator) RequiresExpected() bool {
	return false
}
