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

	"google.golang.org/adk/evaluation"
	"google.golang.org/adk/evaluation/llmjudge"
	"google.golang.org/adk/model"
)

// HallucinationsEvaluator detects unsupported or contradictory claims.
// It detects unsupported or contradictory claims using a two-step process:
// 1. Segment response into individual factual claims
// 2. Classify each claim against the context
type HallucinationsEvaluator struct {
	judge         *llmjudge.Judge
	promptBuilder *llmjudge.PromptBuilder
	parser        *llmjudge.ResponseParser
}

// Classification labels for hallucination detection
const (
	LabelSupported     = "Supported"
	LabelUnsupported   = "Unsupported"
	LabelContradictory = "Contradictory"
	LabelDisputed      = "Disputed"
	LabelNotApplicable = "Not applicable"
)

var validLabels = []string{
	LabelSupported,
	LabelUnsupported,
	LabelContradictory,
	LabelDisputed,
	LabelNotApplicable,
}

// NewHallucinationsEvaluator creates a new hallucinations evaluator.
func NewHallucinationsEvaluator(config evaluation.EvaluatorConfig) (evaluation.Evaluator, error) {
	if config.LLM == nil {
		return nil, fmt.Errorf("LLM is required for hallucination detection")
	}

	llm, ok := config.LLM.(model.LLM)
	if !ok {
		return nil, fmt.Errorf("LLM must implement model.LLM interface")
	}

	if config.JudgeModel == "" {
		return nil, fmt.Errorf("judge model name is required for hallucination detection")
	}

	// Use single sample for hallucination detection to save costs
	// (this is a deterministic classification task)
	numSamples := 1

	judge := llmjudge.NewJudge(llmjudge.Config{
		LLM:        llm,
		ModelName:  config.JudgeModel,
		NumSamples: numSamples,
	})

	return &HallucinationsEvaluator{
		judge:         judge,
		promptBuilder: llmjudge.NewPromptBuilder(),
		parser:        llmjudge.NewResponseParser(),
	}, nil
}

// Evaluate detects hallucinations in responses.
func (e *HallucinationsEvaluator) Evaluate(ctx context.Context, params evaluation.EvaluateParams) (*evaluation.MetricResult, error) {
	if len(params.Actual) == 0 {
		return nil, fmt.Errorf("no actual invocations provided")
	}

	// Evaluate each invocation
	var totalScore float64
	numEvaluations := 0

	for _, invocation := range params.Actual {
		score, err := e.evaluateInvocation(ctx, invocation, params.Context)
		if err != nil {
			// Log error but continue with other invocations
			continue
		}

		totalScore += score
		numEvaluations++
	}

	if numEvaluations == 0 {
		return nil, fmt.Errorf("failed to evaluate any invocations for hallucinations")
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
		MetricType: evaluation.MetricHallucinations,
		Score:      avgScore,
		Status:     status,
	}, nil
}

// MetricType returns the metric type.
func (e *HallucinationsEvaluator) MetricType() evaluation.MetricType {
	return evaluation.MetricHallucinations
}

// RequiresExpected indicates that this evaluator does not need expected responses.
func (e *HallucinationsEvaluator) RequiresExpected() bool {
	return false
}

// evaluateInvocation evaluates a single invocation for hallucinations.
func (e *HallucinationsEvaluator) evaluateInvocation(
	ctx context.Context,
	invocation evaluation.Invocation,
	evalContext evaluation.EvaluationContext,
) (float64, error) {
	// Step 1: Segment response into sentences
	sentences, err := e.segmentResponse(ctx, invocation.AgentResponse)
	if err != nil {
		return 0, fmt.Errorf("failed to segment response: %w", err)
	}

	if len(sentences) == 0 {
		return 1.0, nil // Empty response, no hallucinations
	}

	// Build context string for classification
	contextStr := e.buildContext(invocation, evalContext)

	// Step 2: Classify each sentence
	supportedCount := 0
	notApplicableCount := 0

	for _, sentence := range sentences {
		classification, err := e.classifySentence(ctx, sentence, contextStr)
		if err != nil {
			// Skip sentences we can't classify
			continue
		}

		if classification == LabelSupported || classification == LabelNotApplicable {
			if classification == LabelSupported {
				supportedCount++
			} else {
				notApplicableCount++
			}
		}
	}

	// Score = percentage of supported + not_applicable sentences
	score := float64(supportedCount+notApplicableCount) / float64(len(sentences))
	return score, nil
}

// segmentResponse breaks down a response into individual factual claims.
func (e *HallucinationsEvaluator) segmentResponse(ctx context.Context, response string) ([]string, error) {
	prompt := e.promptBuilder.BuildHallucinationSegmentPrompt(response)

	// Use judge to segment the response
	result, err := e.judge.EvaluateWithPrompt(ctx, prompt, evaluation.MetricHallucinations)
	if err != nil {
		return nil, err
	}

	if len(result.JudgeResponses) == 0 {
		return nil, fmt.Errorf("no segmentation response from judge")
	}

	// Parse segmented sentences (one per line)
	segmentedText := result.JudgeResponses[0]
	sentences := strings.Split(segmentedText, "\n")

	// Filter out empty lines
	var filtered []string
	for _, sentence := range sentences {
		trimmed := strings.TrimSpace(sentence)
		if trimmed != "" {
			filtered = append(filtered, trimmed)
		}
	}

	return filtered, nil
}

// classifySentence classifies a single sentence against the context.
func (e *HallucinationsEvaluator) classifySentence(ctx context.Context, sentence, context string) (string, error) {
	prompt := e.promptBuilder.BuildHallucinationClassifyPrompt(sentence, context)

	// Use judge to classify the sentence
	result, err := e.judge.EvaluateWithPrompt(ctx, prompt, evaluation.MetricHallucinations)
	if err != nil {
		return "", err
	}

	if len(result.JudgeResponses) == 0 {
		return "", fmt.Errorf("no classification response from judge")
	}

	// Parse classification label
	classificationText := result.JudgeResponses[0]
	classification, err := e.parser.ParseClassification(classificationText, validLabels)
	if err != nil {
		return "", fmt.Errorf("failed to parse classification: %w", err)
	}

	return classification, nil
}

// buildContext constructs context string from invocation and evaluation context.
func (e *HallucinationsEvaluator) buildContext(
	invocation evaluation.Invocation,
	evalContext evaluation.EvaluationContext,
) string {
	var parts []string

	// Add system instructions
	if evalContext.SystemInstructions != "" {
		parts = append(parts, fmt.Sprintf("System Instructions:\n%s", evalContext.SystemInstructions))
	}

	// Add user query
	if invocation.UserQuery != "" {
		parts = append(parts, fmt.Sprintf("User Query:\n%s", invocation.UserQuery))
	}

	// Add tool definitions
	if len(evalContext.ToolDefinitions) > 0 {
		var toolDefs strings.Builder
		toolDefs.WriteString("Available Tools:\n")
		for _, tool := range evalContext.ToolDefinitions {
			toolDefs.WriteString(fmt.Sprintf("- %s: %s\n", tool.Name, tool.Description))
		}
		parts = append(parts, toolDefs.String())
	}

	// Add tool call results
	if len(invocation.ToolCalls) > 0 {
		var toolResults strings.Builder
		toolResults.WriteString("Tool Call Results:\n")
		for _, tc := range invocation.ToolCalls {
			toolResults.WriteString(fmt.Sprintf("- %s: %s\n", tc.ToolName, tc.Result))
		}
		parts = append(parts, toolResults.String())
	}

	// Add previous conversation
	if len(evalContext.PreviousMessages) > 0 {
		var prev strings.Builder
		prev.WriteString("Previous Conversation:\n")
		for _, msg := range evalContext.PreviousMessages {
			prev.WriteString(fmt.Sprintf("%s: %s\n", msg.Role, msg.Content))
		}
		parts = append(parts, prev.String())
	}

	return strings.Join(parts, "\n\n")
}
