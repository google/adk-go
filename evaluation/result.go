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

import "time"

// EvalStatus represents the evaluation outcome.
type EvalStatus string

const (
	EvalStatusPassed       EvalStatus = "PASSED"
	EvalStatusFailed       EvalStatus = "FAILED"
	EvalStatusNotEvaluated EvalStatus = "NOT_EVALUATED"
	EvalStatusError        EvalStatus = "ERROR"
)

// EvalSetResult aggregates all evaluation outcomes for a complete eval set.
type EvalSetResult struct {
	// Identification
	EvalSetResultID string `json:"eval_set_result_id"`
	EvalSetID       string `json:"eval_set_id"`
	Name            string `json:"name,omitempty"`

	// Aggregate metrics
	OverallScore float64    `json:"overall_score"`
	Status       EvalStatus `json:"overall_eval_status"`

	// Detailed per-case results
	EvalCaseResults []EvalCaseResult `json:"eval_case_results"`

	// Timestamps
	CreatedAt   time.Time `json:"creation_timestamp"`
	CompletedAt time.Time `json:"completed_timestamp"`
}

// EvalCaseResult contains detailed results for a single eval case.
type EvalCaseResult struct {
	// Identification
	EvalSetID string `json:"eval_set_id"`
	EvalID    string `json:"eval_id"`
	SessionID string `json:"session_id"`
	UserID    string `json:"user_id,omitempty"`

	// Overall status
	FinalEvalStatus EvalStatus `json:"final_eval_status"`

	// Overall metric results (aggregated across invocations)
	OverallMetricResults map[string]MetricResult `json:"overall_eval_metric_results"`

	// Per-invocation breakdown
	MetricResultsPerInvocation []InvocationMetricResults `json:"eval_metric_result_per_invocation"`

	// Session details (optional, for reproducibility)
	SessionDetails *SessionDetails `json:"session_details,omitempty"`
}

// InvocationMetricResults tracks metrics for a single invocation.
type InvocationMetricResults struct {
	InvocationIndex int    `json:"invocation_index"`
	UserQuery       string `json:"user_query"`
	AgentResponse   string `json:"agent_response"`

	// Metric scores for this invocation
	MetricResults map[string]MetricResult `json:"metric_results"`

	// Tool trajectory details
	ActualToolCalls   []ToolCall `json:"actual_tool_calls,omitempty"`
	ExpectedToolCalls []ToolCall `json:"expected_tool_calls,omitempty"`
	TrajectoryMatch   bool       `json:"trajectory_match,omitempty"`

	// Timing
	ProcessingTimeMs int64 `json:"processing_time_ms,omitempty"`
}

// MetricResult contains detailed metric evaluation results.
type MetricResult struct {
	MetricType MetricType `json:"metric_type"`
	Score      float64    `json:"score"`
	Status     EvalStatus `json:"status"`

	// Rubric-based scoring details
	RubricScores map[string]RubricScore `json:"rubric_scores,omitempty"`

	// LLM-as-Judge details
	JudgeResponses []string `json:"judge_responses,omitempty"`

	// Error information
	ErrorMessage string `json:"error_message,omitempty"`

	// Metadata
	EvaluatedAt time.Time `json:"evaluated_at"`
}

// RubricScore tracks individual rubric evaluation.
type RubricScore struct {
	RubricID    string  `json:"rubric_id"`
	Score       float64 `json:"score"`
	Verdict     string  `json:"verdict,omitempty"`     // "yes", "no", or multi-category
	Explanation string  `json:"explanation,omitempty"` // LLM's reasoning
}

// SessionDetails captures complete execution context for reproducibility.
type SessionDetails struct {
	SessionID   string         `json:"session_id"`
	Events      []SessionEvent `json:"events,omitempty"`
	FinalState  map[string]any `json:"final_state,omitempty"`
	TotalTokens int            `json:"total_tokens,omitempty"`
	TotalTimeMs int64          `json:"total_time_ms,omitempty"`
}

// SessionEvent tracks individual agent events.
type SessionEvent struct {
	EventType string         `json:"event_type"`
	Timestamp time.Time      `json:"timestamp"`
	Data      map[string]any `json:"data,omitempty"`
}
