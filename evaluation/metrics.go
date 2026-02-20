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

// MetricType identifies a specific evaluation metric.
type MetricType string

const (
	// Response quality metrics

	// MetricResponseMatch compares agent response against reference using ROUGE-1.
	// Score: 0.0 - 1.0 (higher is better)
	// Uses algorithmic comparison, no LLM required.
	MetricResponseMatch MetricType = "RESPONSE_MATCH_SCORE"

	// MetricSemanticResponseMatch uses LLM-as-Judge to validate response correctness.
	// Allows format variations while focusing on semantic accuracy.
	// Score: 0.0 (invalid) or 1.0 (valid)
	// Supports multiple samples with majority voting.
	MetricSemanticResponseMatch MetricType = "SEMANTIC_RESPONSE_MATCH"

	// MetricResponseEvaluationScore assesses response coherence and quality.
	// Score: 1-5 scale (higher is better)
	// Measures logical flow, clarity, and completeness.
	MetricResponseEvaluationScore MetricType = "RESPONSE_EVALUATION_SCORE"

	// Tool usage metrics

	// MetricToolTrajectoryAvgScore validates tool call sequences.
	// Compares actual vs expected tool trajectories.
	// Score: 0.0 or 1.0 per invocation, averaged across all invocations.
	// Validates both tool names and arguments.
	MetricToolTrajectoryAvgScore MetricType = "TOOL_TRAJECTORY_AVG_SCORE"

	// MetricToolUseQuality evaluates tool usage using custom rubrics.
	// Uses LLM-as-Judge with rubric-based assessment.
	// Score: 0.0 - 1.0 based on rubric criteria.
	MetricToolUseQuality MetricType = "RUBRIC_BASED_TOOL_USE_QUALITY"

	// Safety & quality metrics

	// MetricSafety evaluates response safety and harmlessness.
	// Score: 0.0 - 1.0 (higher = safer)
	// Checks for toxic language, harmful instructions, bias, etc.
	MetricSafety MetricType = "SAFETY"

	// MetricHallucinations detects unsupported or contradictory claims.
	// Two-step process: segmentation + classification.
	// Score: Percentage of supported + not_applicable sentences.
	// Range: 0.0 - 1.0 (higher is better)
	MetricHallucinations MetricType = "HALLUCINATIONS"

	// MetricResponseQuality assesses response quality using rubrics.
	// Uses LLM-as-Judge with custom quality criteria.
	// Score: 0.0 - 1.0 based on rubric verdicts.
	MetricResponseQuality MetricType = "RUBRIC_BASED_RESPONSE_QUALITY"
)

// AllMetrics returns a list of all available metric types.
func AllMetrics() []MetricType {
	return []MetricType{
		MetricResponseMatch,
		MetricSemanticResponseMatch,
		MetricResponseEvaluationScore,
		MetricToolTrajectoryAvgScore,
		MetricToolUseQuality,
		MetricSafety,
		MetricHallucinations,
		MetricResponseQuality,
	}
}

// String returns the string representation of the metric type.
func (m MetricType) String() string {
	return string(m)
}

// IsResponseMetric returns true if the metric evaluates response quality.
func (m MetricType) IsResponseMetric() bool {
	switch m {
	case MetricResponseMatch,
		MetricSemanticResponseMatch,
		MetricResponseEvaluationScore,
		MetricResponseQuality:
		return true
	default:
		return false
	}
}

// IsToolMetric returns true if the metric evaluates tool usage.
func (m MetricType) IsToolMetric() bool {
	switch m {
	case MetricToolTrajectoryAvgScore,
		MetricToolUseQuality:
		return true
	default:
		return false
	}
}

// IsSafetyMetric returns true if the metric evaluates safety aspects.
func (m MetricType) IsSafetyMetric() bool {
	switch m {
	case MetricSafety,
		MetricHallucinations:
		return true
	default:
		return false
	}
}

// RequiresLLM returns true if the metric requires an LLM for evaluation.
func (m MetricType) RequiresLLM() bool {
	switch m {
	case MetricResponseMatch,
		MetricToolTrajectoryAvgScore:
		return false
	default:
		return true
	}
}

// RequiresExpectedResponse returns true if the metric needs an expected response.
func (m MetricType) RequiresExpectedResponse() bool {
	switch m {
	case MetricResponseMatch,
		MetricSemanticResponseMatch,
		MetricToolTrajectoryAvgScore:
		return true
	default:
		return false
	}
}
