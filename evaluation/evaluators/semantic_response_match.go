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

// SemanticResponseMatchEvaluator implements semantic response matching using LLM-as-Judge.
// It validates response correctness with semantic understanding and format flexibility.
type SemanticResponseMatchEvaluator struct {
	judge         *llmjudge.Judge
	promptBuilder *llmjudge.PromptBuilder
}

// NewSemanticResponseMatchEvaluator creates a new semantic response match evaluator.
func NewSemanticResponseMatchEvaluator(config evaluation.EvaluatorConfig) (evaluation.Evaluator, error) {
	if config.LLM == nil {
		return nil, fmt.Errorf("LLM is required for semantic response match")
	}

	llm, ok := config.LLM.(model.LLM)
	if !ok {
		return nil, fmt.Errorf("LLM must implement model.LLM interface")
	}

	if config.JudgeModel == "" {
		return nil, fmt.Errorf("judge model name is required for semantic response match")
	}

	numSamples := config.NumSamples
	if numSamples <= 0 {
		numSamples = 3 // Default to 3 samples for better reliability
	}

	judge := llmjudge.NewJudge(llmjudge.Config{
		LLM:        llm,
		ModelName:  config.JudgeModel,
		NumSamples: numSamples,
	})

	return &SemanticResponseMatchEvaluator{
		judge:         judge,
		promptBuilder: llmjudge.NewPromptBuilder(),
	}, nil
}

// Evaluate compares final responses using LLM-as-Judge with semantic understanding.
func (e *SemanticResponseMatchEvaluator) Evaluate(ctx context.Context, params evaluation.EvaluateParams) (*evaluation.MetricResult, error) {
	if len(params.Actual) == 0 {
		return nil, fmt.Errorf("no actual invocations provided")
	}

	if len(params.Expected) == 0 {
		return nil, fmt.Errorf("no expected invocations provided")
	}

	// Evaluate the final (last) invocation
	actualIdx := len(params.Actual) - 1
	expectedIdx := len(params.Expected) - 1

	actualInvocation := params.Actual[actualIdx]
	expectedInvocation := params.Expected[expectedIdx]

	// Build evaluation prompt
	prompt := e.promptBuilder.BuildFinalResponseMatchPrompt(
		actualInvocation.UserQuery,
		actualInvocation.AgentResponse,
		expectedInvocation.AgentResponse,
	)

	// Evaluate using LLM judge with verdict (yes/no)
	result, err := e.judge.EvaluateVerdict(ctx, prompt, evaluation.MetricSemanticResponseMatch)
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
func (e *SemanticResponseMatchEvaluator) MetricType() evaluation.MetricType {
	return evaluation.MetricSemanticResponseMatch
}

// RequiresExpected indicates that this evaluator needs expected responses.
func (e *SemanticResponseMatchEvaluator) RequiresExpected() bool {
	return true
}
