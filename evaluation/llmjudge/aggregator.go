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

package llmjudge

import (
	"fmt"

	"google.golang.org/adk/evaluation"
)

// ResultAggregator combines multiple evaluation samples using various strategies.
type ResultAggregator struct{}

// NewResultAggregator creates a new result aggregator.
func NewResultAggregator() *ResultAggregator {
	return &ResultAggregator{}
}

// AggregateSamples combines multiple evaluation samples into a single result.
// For scores: uses mean
// For rubrics: uses majority voting per rubric
func (a *ResultAggregator) AggregateSamples(samples []*evaluation.MetricResult) (*evaluation.MetricResult, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("no samples to aggregate")
	}

	if len(samples) == 1 {
		return samples[0], nil
	}

	// Aggregate scores using mean
	totalScore := 0.0
	for _, sample := range samples {
		totalScore += sample.Score
	}
	meanScore := totalScore / float64(len(samples))

	// Aggregate rubric scores using majority voting
	aggregatedRubrics := a.aggregateRubricScores(samples)

	// Collect all judge responses
	allResponses := make([]string, 0, len(samples))
	for _, sample := range samples {
		allResponses = append(allResponses, sample.JudgeResponses...)
	}

	return &evaluation.MetricResult{
		Score:          meanScore,
		Status:         samples[0].Status,
		RubricScores:   aggregatedRubrics,
		JudgeResponses: allResponses,
	}, nil
}

// aggregateRubricScores aggregates rubric scores using majority voting.
func (a *ResultAggregator) aggregateRubricScores(samples []*evaluation.MetricResult) map[string]evaluation.RubricScore {
	if len(samples) == 0 {
		return nil
	}

	// Collect all rubric IDs
	rubricIDs := make(map[string]bool)
	for _, sample := range samples {
		for rubricID := range sample.RubricScores {
			rubricIDs[rubricID] = true
		}
	}

	// Aggregate each rubric using majority voting
	aggregated := make(map[string]evaluation.RubricScore)
	for rubricID := range rubricIDs {
		aggregated[rubricID] = a.aggregateRubric(rubricID, samples)
	}

	return aggregated
}

// aggregateRubric aggregates a single rubric across samples using majority voting.
func (a *ResultAggregator) aggregateRubric(rubricID string, samples []*evaluation.MetricResult) evaluation.RubricScore {
	yesCount := 0
	noCount := 0
	totalScore := 0.0

	for _, sample := range samples {
		if rs, exists := sample.RubricScores[rubricID]; exists {
			totalScore += rs.Score
			if rs.Verdict == "yes" {
				yesCount++
			} else {
				noCount++
			}
		}
	}

	// Majority voting
	verdict := "no"
	if yesCount > noCount {
		verdict = "yes"
	}

	// Average score
	avgScore := totalScore / float64(len(samples))

	return evaluation.RubricScore{
		RubricID: rubricID,
		Score:    avgScore,
		Verdict:  verdict,
	}
}

// AggregateAcrossInvocations combines results from multiple invocations.
// This calculates the mean score across all invocations.
func (a *ResultAggregator) AggregateAcrossInvocations(results []*evaluation.MetricResult) (*evaluation.MetricResult, error) {
	if len(results) == 0 {
		return nil, fmt.Errorf("no results to aggregate")
	}

	if len(results) == 1 {
		return results[0], nil
	}

	// Calculate mean score
	totalScore := 0.0
	for _, result := range results {
		totalScore += result.Score
	}
	meanScore := totalScore / float64(len(results))

	// Aggregate rubric scores
	aggregatedRubrics := a.aggregateRubricAcrossInvocations(results)

	// Collect all responses
	allResponses := make([]string, 0)
	for _, result := range results {
		allResponses = append(allResponses, result.JudgeResponses...)
	}

	// Determine overall status
	status := evaluation.EvalStatusPassed
	for _, result := range results {
		if result.Status == evaluation.EvalStatusFailed {
			status = evaluation.EvalStatusFailed
			break
		}
	}

	return &evaluation.MetricResult{
		Score:          meanScore,
		Status:         status,
		RubricScores:   aggregatedRubrics,
		JudgeResponses: allResponses,
	}, nil
}

// aggregateRubricAcrossInvocations aggregates rubric scores across invocations.
func (a *ResultAggregator) aggregateRubricAcrossInvocations(results []*evaluation.MetricResult) map[string]evaluation.RubricScore {
	if len(results) == 0 {
		return nil
	}

	// Collect all rubric IDs
	rubricIDs := make(map[string]bool)
	for _, result := range results {
		for rubricID := range result.RubricScores {
			rubricIDs[rubricID] = true
		}
	}

	// Calculate mean score for each rubric
	aggregated := make(map[string]evaluation.RubricScore)
	for rubricID := range rubricIDs {
		totalScore := 0.0
		count := 0

		for _, result := range results {
			if rs, exists := result.RubricScores[rubricID]; exists {
				totalScore += rs.Score
				count++
			}
		}

		if count > 0 {
			aggregated[rubricID] = evaluation.RubricScore{
				RubricID: rubricID,
				Score:    totalScore / float64(count),
			}
		}
	}

	return aggregated
}
