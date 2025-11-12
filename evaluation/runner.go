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

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"golang.org/x/sync/semaphore"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// Runner orchestrates evaluation execution.
type Runner struct {
	agentRunner     *runner.Runner
	registry        *Registry
	storage         Storage
	sessionService  session.Service
	appName         string
	rateLimitDelay  time.Duration
	evalSemaphore   *semaphore.Weighted
	lastLLMEvalTime time.Time
}

// RunnerConfig provides configuration for creating a runner.
type RunnerConfig struct {
	AgentRunner    *runner.Runner
	Storage        Storage
	SessionService session.Service
	AppName        string

	// RateLimitDelay is the delay between LLM-based evaluations to avoid rate limits
	// Default: 6 seconds (for free tier with 10 RPM limit)
	RateLimitDelay time.Duration

	// MaxConcurrentEvals limits concurrent LLM evaluations
	// Default: 1 (sequential)
	MaxConcurrentEvals int64
}

// NewRunner creates a new evaluation runner.
func NewRunner(config RunnerConfig) *Runner {
	maxConcurrent := config.MaxConcurrentEvals
	if maxConcurrent <= 0 {
		maxConcurrent = 1
	}

	return &Runner{
		agentRunner:    config.AgentRunner,
		registry:       DefaultRegistry,
		storage:        config.Storage,
		sessionService: config.SessionService,
		appName:        config.AppName,
		rateLimitDelay: config.RateLimitDelay,
		evalSemaphore:  semaphore.NewWeighted(maxConcurrent),
	}
}

// RunEvalSet executes a complete evaluation suite.
func (r *Runner) RunEvalSet(ctx context.Context, evalSet *EvalSet, config *EvalConfig) (*EvalSetResult, error) {
	if evalSet == nil {
		return nil, fmt.Errorf("eval set is nil")
	}

	if config == nil {
		return nil, fmt.Errorf("eval config is nil")
	}

	// Create result container
	result := &EvalSetResult{
		EvalSetResultID: uuid.New().String(),
		EvalSetID:       evalSet.ID,
		Name:            evalSet.Name,
		EvalCaseResults: make([]EvalCaseResult, 0, len(evalSet.EvalCases)),
		CreatedAt:       time.Now(),
	}

	// Run each eval case
	for _, evalCase := range evalSet.EvalCases {
		caseResult, err := r.runEvalCase(ctx, &evalCase, config)
		if err != nil {
			// Create error result for this case
			caseResult = &EvalCaseResult{
				EvalSetID:       evalSet.ID,
				EvalID:          evalCase.ID,
				FinalEvalStatus: EvalStatusError,
				OverallMetricResults: map[string]MetricResult{
					"error": {
						Status:       EvalStatusError,
						ErrorMessage: err.Error(),
					},
				},
			}
		}
		result.EvalCaseResults = append(result.EvalCaseResults, *caseResult)
	}

	// Calculate overall score and status
	result.OverallScore = r.calculateOverallScore(result.EvalCaseResults)
	result.Status = r.determineOverallStatus(result.EvalCaseResults, config)
	result.CompletedAt = time.Now()

	// Store results
	if err := r.storage.SaveEvalSetResult(ctx, result); err != nil {
		return nil, fmt.Errorf("failed to save eval set result: %w", err)
	}

	return result, nil
}

// runEvalCase executes evaluation for a single case.
func (r *Runner) runEvalCase(ctx context.Context, evalCase *EvalCase, config *EvalConfig) (*EvalCaseResult, error) {
	// Create session for this eval case
	sessionID := uuid.New().String()

	// TODO: Initialize session with SessionInput state if evalCase.SessionInput != nil && r.agentRunner != nil

	// Run conversation through agent
	actualInvocations, err := r.runConversation(ctx, sessionID, evalCase.Conversation)
	if err != nil {
		return nil, fmt.Errorf("failed to run conversation: %w", err)
	}

	// Build expected invocations from eval case
	expectedInvocations := r.buildExpectedInvocations(evalCase)

	// Build evaluation context
	evalContext := r.buildEvaluationContext(evalCase)

	// Run all configured evaluators
	metricResults := make(map[string]MetricResult)
	perInvocationResults := make([]InvocationMetricResults, len(actualInvocations))

	for _, criterion := range config.Criteria {
		metricType := criterion.GetMetricType()
		metricName := string(metricType)

		// Create evaluator for this metric
		evaluator, err := r.createEvaluator(metricType, criterion, config)
		if err != nil {
			metricResults[metricName] = MetricResult{
				MetricType:   metricType,
				Status:       EvalStatusError,
				ErrorMessage: fmt.Sprintf("failed to create evaluator: %v", err),
			}
			continue
		}

		// Apply rate limiting for LLM-based evaluators
		isLLMBased := r.isLLMBasedMetric(metricType)
		if isLLMBased {
			if err := r.applyRateLimit(ctx); err != nil {
				metricResults[metricName] = MetricResult{
					MetricType:   metricType,
					Status:       EvalStatusError,
					ErrorMessage: fmt.Sprintf("rate limit wait cancelled: %v", err),
				}
				continue
			}
		}

		// Build evaluation params
		params := EvaluateParams{
			Actual:    actualInvocations,
			Expected:  expectedInvocations,
			Rubrics:   evalCase.Rubrics,
			Criterion: criterion,
			Context:   evalContext,
		}

		// Run evaluation
		result, err := evaluator.Evaluate(ctx, params)
		if err != nil {
			result = &MetricResult{
				MetricType:   metricType,
				Status:       EvalStatusError,
				ErrorMessage: err.Error(),
			}
		}

		metricResults[metricName] = *result
	}

	// Build per-invocation results
	for i, invocation := range actualInvocations {
		invocationResult := InvocationMetricResults{
			InvocationIndex: i,
			UserQuery:       invocation.UserQuery,
			AgentResponse:   invocation.AgentResponse,
			ActualToolCalls: invocation.ToolCalls,
			MetricResults:   make(map[string]MetricResult),
		}

		if i < len(expectedInvocations) {
			invocationResult.ExpectedToolCalls = expectedInvocations[i].ToolCalls
		}

		perInvocationResults[i] = invocationResult
	}

	// Determine overall status for this case
	finalStatus := r.determineCaseStatus(metricResults, config)

	return &EvalCaseResult{
		EvalSetID:                  evalCase.ID,
		EvalID:                     evalCase.ID,
		SessionID:                  sessionID,
		FinalEvalStatus:            finalStatus,
		OverallMetricResults:       metricResults,
		MetricResultsPerInvocation: perInvocationResults,
	}, nil
}

// runConversation runs the conversation through the agent.
func (r *Runner) runConversation(ctx context.Context, sessionID string, conversation []ConversationTurn) ([]Invocation, error) {
	if r.agentRunner == nil {
		return nil, nil
	}

	// Create session if needed
	if err := r.createSession(ctx, sessionID); err != nil {
		return nil, err
	}

	var invocations []Invocation
	for _, turn := range conversation {
		invocation, err := r.processTurn(ctx, sessionID, turn)
		if err != nil {
			return nil, err
		}
		if invocation != nil {
			invocations = append(invocations, *invocation)
		}
	}

	return invocations, nil
}

// createSession creates a new session for the evaluation.
func (r *Runner) createSession(ctx context.Context, sessionID string) error {
	if r.sessionService == nil {
		return nil
	}

	_, err := r.sessionService.Create(ctx, &session.CreateRequest{
		AppName:   r.appName,
		UserID:    "eval-user",
		SessionID: sessionID,
	})
	// Ignore error if session already exists
	if err != nil && !strings.Contains(err.Error(), "already exists") {
		return fmt.Errorf("failed to create session: %w", err)
	}
	return nil
}

// processTurn processes a single turn of a conversation.
func (r *Runner) processTurn(ctx context.Context, sessionID string, turn ConversationTurn) (*Invocation, error) {
	if turn.Role != "user" {
		return nil, nil
	}

	// Create user message content
	userMsg := &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{genai.NewPartFromText(turn.Content)},
	}

	// Run agent and collect events
	response, toolCalls, err := r.runAgentAndCollectEvents(ctx, sessionID, userMsg)
	if err != nil {
		return nil, err
	}

	return &Invocation{
		UserQuery:     turn.Content,
		AgentResponse: response,
		ToolCalls:     toolCalls,
		SessionID:     sessionID,
		Timestamp:     time.Now(),
	}, nil
}

// runAgentAndCollectEvents runs the agent and collects the response and tool calls.
func (r *Runner) runAgentAndCollectEvents(ctx context.Context, sessionID string, userMsg *genai.Content) (string, []ToolCall, error) {
	var fullResponse strings.Builder
	var toolCalls []ToolCall

	for event, err := range r.agentRunner.Run(ctx, "eval-user", sessionID, userMsg, agent.RunConfig{}) {
		if err != nil {
			return "", nil, fmt.Errorf("agent run error: %w", err)
		}

		if event != nil && event.Content != nil {
			// Collect response text and tool calls from content parts
			for _, part := range event.Content.Parts {
				if part == nil {
					continue
				}
				// Handle text responses
				if part.Text != "" {
					fullResponse.WriteString(part.Text)
				}
				// Handle function calls
				if part.FunctionCall != nil {
					args := make(map[string]any)
					if part.FunctionCall.Args != nil {
						for k, v := range part.FunctionCall.Args {
							args[k] = v
						}
					}
					toolCalls = append(toolCalls, ToolCall{
						ToolName:  part.FunctionCall.Name,
						Arguments: args,
					})
				}
			}
		}
	}

	return fullResponse.String(), toolCalls, nil
}

// buildExpectedInvocations creates expected invocations from eval case.
func (r *Runner) buildExpectedInvocations(evalCase *EvalCase) []Invocation {
	var invocations []Invocation

	for _, turn := range evalCase.Conversation {
		if turn.ExpectedInvocation != nil {
			invocations = append(invocations, *turn.ExpectedInvocation)
		}
	}

	// If there's a top-level ExpectedResponse, and no turn-specific expected invocations,
	// create a single expected invocation from the top-level fields for backward compatibility.
	if len(invocations) == 0 && evalCase.ExpectedResponse != "" {
		invocations = append(invocations, Invocation{
			AgentResponse: evalCase.ExpectedResponse,
			ToolCalls:     r.convertExpectedToolCalls(evalCase.ExpectedToolCalls),
		})
	}

	return invocations
}

// convertExpectedToolCalls converts ExpectedToolCall to ToolCall.
func (r *Runner) convertExpectedToolCalls(expected []ExpectedToolCall) []ToolCall {
	var toolCalls []ToolCall

	for _, etc := range expected {
		toolCalls = append(toolCalls, ToolCall{
			ToolName:  etc.ToolName,
			Arguments: etc.Arguments,
		})
	}

	return toolCalls
}

// buildEvaluationContext creates evaluation context from eval case.
func (r *Runner) buildEvaluationContext(evalCase *EvalCase) EvaluationContext {
	return EvaluationContext{
		// TODO: Populate from session or agent configuration
		SystemInstructions: "",
		ToolDefinitions:    []ToolDefinition{},
		PreviousMessages:   evalCase.Conversation,
	}
}

// createEvaluator creates an evaluator for a metric type.
func (r *Runner) createEvaluator(metricType MetricType, criterion Criterion, config *EvalConfig) (Evaluator, error) {
	evaluatorConfig := EvaluatorConfig{
		LLM:        config.JudgeLLM,
		JudgeModel: config.JudgeModel,
	}

	// Extract additional config from criterion
	if llmCriterion, ok := criterion.(*LLMAsJudgeCriterion); ok {
		// Override defaults if provided
		if llmCriterion.JudgeModel != "" {
			evaluatorConfig.JudgeModel = llmCriterion.JudgeModel
		}
		evaluatorConfig.NumSamples = llmCriterion.NumSamples
		evaluatorConfig.ModelConfig = llmCriterion.ModelConfig
	}

	return r.registry.CreateEvaluator(metricType, evaluatorConfig)
}

// calculateOverallScore calculates the overall score across all cases.
func (r *Runner) calculateOverallScore(cases []EvalCaseResult) float64 {
	if len(cases) == 0 {
		return 0.0
	}

	totalScore := 0.0
	numScores := 0

	for _, caseResult := range cases {
		for _, metricResult := range caseResult.OverallMetricResults {
			if metricResult.Status != EvalStatusError {
				totalScore += metricResult.Score
				numScores++
			}
		}
	}

	if numScores == 0 {
		return 0.0
	}

	return totalScore / float64(numScores)
}

// determineOverallStatus determines the overall evaluation status.
func (r *Runner) determineOverallStatus(cases []EvalCaseResult, config *EvalConfig) EvalStatus {
	if len(cases) == 0 {
		return EvalStatusNotEvaluated
	}

	allPassed := true
	anyFailed := false

	for _, caseResult := range cases {
		if caseResult.FinalEvalStatus == EvalStatusFailed {
			anyFailed = true
			allPassed = false
		} else if caseResult.FinalEvalStatus != EvalStatusPassed {
			allPassed = false
		}
	}

	if anyFailed {
		return EvalStatusFailed
	}

	if allPassed {
		return EvalStatusPassed
	}

	return EvalStatusNotEvaluated
}

// determineCaseStatus determines the status for a single eval case.
func (r *Runner) determineCaseStatus(metricResults map[string]MetricResult, config *EvalConfig) EvalStatus {
	if len(metricResults) == 0 {
		return EvalStatusNotEvaluated
	}

	allPassed := true
	anyFailed := false
	hasEvaluated := false

	for _, result := range metricResults {
		if result.Status == EvalStatusFailed {
			anyFailed = true
			allPassed = false
			hasEvaluated = true
		} else if result.Status == EvalStatusPassed {
			hasEvaluated = true
		} else if result.Status != EvalStatusError {
			// Non-error, non-passed, non-failed statuses affect pass/fail
			allPassed = false
			hasEvaluated = true
		}
		// Ignore ERROR status - it means the metric couldn't be evaluated (e.g., no tool calls)
	}

	if anyFailed {
		return EvalStatusFailed
	}

	if allPassed && hasEvaluated {
		return EvalStatusPassed
	}

	return EvalStatusNotEvaluated
}

// SetRegistry sets a custom evaluator registry.
func (r *Runner) SetRegistry(registry *Registry) {
	r.registry = registry
}

// applyRateLimit enforces rate limiting between LLM calls.
func (r *Runner) applyRateLimit(ctx context.Context) error {
	if err := r.evalSemaphore.Acquire(ctx, 1); err != nil {
		return err
	}
	defer r.evalSemaphore.Release(1)

	if !r.lastLLMEvalTime.IsZero() {
		elapsed := time.Since(r.lastLLMEvalTime)
		if elapsed < r.rateLimitDelay {
			sleepDuration := r.rateLimitDelay - elapsed
			select {
			case <-time.After(sleepDuration):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}

	r.lastLLMEvalTime = time.Now()
	return nil
}

// isLLMBasedMetric returns true if the metric requires LLM calls.
func (r *Runner) isLLMBasedMetric(metricType MetricType) bool {
	switch metricType {
	case MetricResponseMatch, MetricToolTrajectoryAvgScore:
		return false
	default:
		return true
	}
}

// RegisterAgent registers an agent for evaluation with default configuration.
// Deprecated: Use NewRunner with RunnerConfig instead.
func RegisterAgent(agentRunner *runner.Runner) *Runner {
	return NewRunner(RunnerConfig{
		AgentRunner: agentRunner,
		Storage:     nil,
	})
}
