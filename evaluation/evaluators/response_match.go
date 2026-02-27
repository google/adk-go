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
	"strings"
	"time"
	"unicode"

	"google.golang.org/adk/evaluation"
)

// ResponseMatchEvaluator implements RESPONSE_MATCH_SCORE using ROUGE-1 algorithm.
// It compares agent responses against reference responses using unigram overlap.
type ResponseMatchEvaluator struct{}

// NewResponseMatchEvaluator creates a new response match evaluator.
func NewResponseMatchEvaluator(config evaluation.EvaluatorConfig) (evaluation.Evaluator, error) {
	return &ResponseMatchEvaluator{}, nil
}

// Evaluate compares actual and expected responses using ROUGE-1.
func (e *ResponseMatchEvaluator) Evaluate(ctx context.Context, params evaluation.EvaluateParams) (*evaluation.MetricResult, error) {
	if len(params.Actual) == 0 {
		return nil, fmt.Errorf("no actual invocations provided")
	}

	if len(params.Expected) == 0 {
		return nil, fmt.Errorf("no expected invocations provided")
	}

	// Calculate ROUGE-1 score for each invocation pair
	var totalScore float64
	numComparisons := 0

	for i := 0; i < len(params.Actual) && i < len(params.Expected); i++ {
		actual := params.Actual[i].AgentResponse
		expected := params.Expected[i].AgentResponse

		score := e.calculateROUGE1(actual, expected)
		totalScore += score
		numComparisons++
	}

	if numComparisons == 0 {
		return nil, fmt.Errorf("no responses to compare")
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
		MetricType:  evaluation.MetricResponseMatch,
		Score:       avgScore,
		Status:      status,
		EvaluatedAt: time.Now(),
	}, nil
}

// MetricType returns the metric type.
func (e *ResponseMatchEvaluator) MetricType() evaluation.MetricType {
	return evaluation.MetricResponseMatch
}

// RequiresExpected indicates that this evaluator needs expected responses.
func (e *ResponseMatchEvaluator) RequiresExpected() bool {
	return true
}

// calculateROUGE1 computes ROUGE-1 F1 score between two texts.
// ROUGE-1 measures unigram (single word) overlap.
func (e *ResponseMatchEvaluator) calculateROUGE1(actual, expected string) float64 {
	actualTokens := e.tokenize(actual)
	expectedTokens := e.tokenize(expected)

	if len(actualTokens) == 0 || len(expectedTokens) == 0 {
		if len(actualTokens) == 0 && len(expectedTokens) == 0 {
			return 1.0 // Both empty, perfect match
		}
		return 0.0
	}

	// Count overlapping tokens
	actualSet := make(map[string]int)
	for _, token := range actualTokens {
		actualSet[token]++
	}

	expectedSet := make(map[string]int)
	for _, token := range expectedTokens {
		expectedSet[token]++
	}

	// Calculate overlap
	overlap := 0
	for token, count := range actualSet {
		if expectedCount, exists := expectedSet[token]; exists {
			overlap += min(count, expectedCount)
		}
	}

	// Calculate precision, recall, and F1
	precision := float64(overlap) / float64(len(actualTokens))
	recall := float64(overlap) / float64(len(expectedTokens))

	if precision+recall == 0 {
		return 0.0
	}

	f1 := 2 * (precision * recall) / (precision + recall)
	return f1
}

// tokenize splits text into lowercase tokens.
func (e *ResponseMatchEvaluator) tokenize(text string) []string {
	// Convert to lowercase
	text = strings.ToLower(text)

	// Split on whitespace and punctuation
	var tokens []string
	var current strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			current.WriteRune(r)
		} else {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
		}
	}

	// Add last token
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

	return tokens
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
