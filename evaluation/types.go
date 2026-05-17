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

// EvalSet represents a collection of evaluation test cases.
// An EvalSet groups related test scenarios for systematic agent evaluation.
type EvalSet struct {
	ID          string     `json:"eval_set_id"`
	Name        string     `json:"name,omitempty"`
	Description string     `json:"description,omitempty"`
	EvalCases   []EvalCase `json:"eval_cases"`
	CreatedAt   time.Time  `json:"creation_timestamp"`
}

// EvalCase represents a single evaluation scenario.
// Each case defines a conversation flow and expected outcomes for agent testing.
type EvalCase struct {
	ID string `json:"eval_id"`

	// Conversation defines the interaction flow between user and agent
	Conversation []ConversationTurn `json:"conversation,omitempty"`

	// SessionInput provides initialization context for the agent
	SessionInput *SessionInput `json:"session_input,omitempty"`

	// Expected outcomes for validation
	ExpectedResponse  string             `json:"expected_response,omitempty"`
	ExpectedToolCalls []ExpectedToolCall `json:"expected_tool_calls,omitempty"`
	FinalSessionState map[string]any     `json:"final_session_state,omitempty"`

	// Rubrics define custom evaluation criteria
	Rubrics map[string]Rubric `json:"rubrics,omitempty"`

	CreatedAt time.Time `json:"creation_timestamp"`
}

// ConversationTurn represents a single user/agent interaction in a conversation.
type ConversationTurn struct {
	Role    string `json:"role"`    // "user" or "agent"
	Content string `json:"content"` // The message content

	// ExpectedInvocation defines the expected agent response and tool calls for this turn.
	// This is used for multi-turn evaluation where expectations can vary per turn.
	ExpectedInvocation *Invocation `json:"expected_invocation,omitempty"`
}

// ExpectedToolCall defines an expected tool invocation.
type ExpectedToolCall struct {
	ToolName  string         `json:"tool_name"`
	Arguments map[string]any `json:"arguments"`
}

// SessionInput provides agent initialization context.
type SessionInput struct {
	AppName      string         `json:"app_name"`
	UserID       string         `json:"user_id,omitempty"`
	SessionState map[string]any `json:"session_state,omitempty"`
}

// EvalConfig defines evaluation criteria and thresholds.
type EvalConfig struct {
	// Criteria defines the list of criteria for the evaluation.
	Criteria []Criterion `json:"criteria"`

	// JudgeLLM is the LLM instance to use for LLM-as-Judge evaluators.
	// This field is not serialized to JSON.
	JudgeLLM interface{} `json:"-"`

	// JudgeModel is the default model name for LLM-as-Judge evaluators.
	JudgeModel string `json:"judge_model,omitempty"`
}

// Criterion defines evaluation requirements for a specific metric.
type Criterion interface {
	GetThreshold() *Threshold
	GetMetricType() MetricType
}

// Threshold defines pass/fail scoring bounds.
type Threshold struct {
	MinScore   float64    `json:"min_score"`
	MaxScore   float64    `json:"max_score,omitempty"`
	MetricType MetricType `json:"metric_type"`
}

// GetThreshold returns the threshold (implements Criterion interface).
func (t *Threshold) GetThreshold() *Threshold {
	return t
}

// GetMetricType returns the metric type for the threshold.
func (t *Threshold) GetMetricType() MetricType {
	return t.MetricType
}

// LLMAsJudgeCriterion defines criteria for LLM-based evaluation.
type LLMAsJudgeCriterion struct {
	Threshold   *Threshold     `json:"threshold"`
	JudgeModel  string         `json:"judge_model"`
	NumSamples  int            `json:"num_samples,omitempty"`  // Default: 1
	ModelConfig map[string]any `json:"model_config,omitempty"` // Temperature, top_p, etc.
	MetricType  MetricType     `json:"metric_type"`
}

// GetThreshold returns the threshold.
func (c *LLMAsJudgeCriterion) GetThreshold() *Threshold {
	return c.Threshold
}

// GetMetricType returns the metric type.
func (c *LLMAsJudgeCriterion) GetMetricType() MetricType {
	return c.MetricType
}

// RubricCriterion defines criteria for rubric-based evaluation.
type RubricCriterion struct {
	Threshold  *Threshold        `json:"threshold"`
	Rubrics    map[string]Rubric `json:"rubrics"`
	MetricType MetricType        `json:"metric_type"`
}

// GetThreshold returns the threshold.
func (c *RubricCriterion) GetThreshold() *Threshold {
	return c.Threshold
}

// GetMetricType returns the metric type.
func (c *RubricCriterion) GetMetricType() MetricType {
	return c.MetricType
}

// Rubric defines a specific evaluation criterion.
type Rubric struct {
	RubricID      string   `json:"rubric_id"`
	RubricContent string   `json:"rubric_content"`
	MaxScore      float64  `json:"max_score,omitempty"`  // Default: 1.0
	Categories    []string `json:"categories,omitempty"` // For multi-category rubrics
}

// Invocation represents one agent interaction.
type Invocation struct {
	UserQuery     string
	AgentResponse string
	ToolCalls     []ToolCall
	SessionID     string
	Timestamp     time.Time
}

// ToolCall represents a tool invocation.
type ToolCall struct {
	ToolName  string         `json:"tool_name"`
	Arguments map[string]any `json:"arguments"`
	Result    string         `json:"result,omitempty"`
}

// EvaluationContext provides conversation context for evaluation.
type EvaluationContext struct {
	SystemInstructions string
	ToolDefinitions    []ToolDefinition
	PreviousMessages   []ConversationTurn
}

// ToolDefinition describes a tool available to the agent.
type ToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}
