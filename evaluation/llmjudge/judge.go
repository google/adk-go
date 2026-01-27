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
	"context"
	"fmt"
	"time"

	"google.golang.org/adk/evaluation"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// Judge implements LLM-as-Judge evaluation pattern.
// It uses an LLM to evaluate agent responses based on custom prompts and criteria.
type Judge struct {
	llm        model.LLM
	modelName  string
	numSamples int
	config     *genai.GenerateContentConfig
	parser     *ResponseParser
	aggregator *ResultAggregator
}

// Config contains configuration for the LLM judge.
type Config struct {
	LLM         model.LLM
	ModelName   string
	NumSamples  int
	Temperature *float32
	TopP        *float32
	TopK        *int32
}

// NewJudge creates a new LLM-as-Judge instance.
func NewJudge(cfg Config) *Judge {
	if cfg.NumSamples <= 0 {
		cfg.NumSamples = 1
	}

	genCfg := &genai.GenerateContentConfig{}
	if cfg.Temperature != nil {
		genCfg.Temperature = cfg.Temperature
	}
	if cfg.TopP != nil {
		genCfg.TopP = cfg.TopP
	}
	if cfg.TopK != nil {
		topK := float32(*cfg.TopK)
		genCfg.TopK = &topK
	}

	return &Judge{
		llm:        cfg.LLM,
		modelName:  cfg.ModelName,
		numSamples: cfg.NumSamples,
		config:     genCfg,
		parser:     NewResponseParser(),
		aggregator: NewResultAggregator(),
	}
}

// EvaluateWithPrompt evaluates using a custom prompt.
// This is the core evaluation method that runs multiple samples and aggregates results.
func (j *Judge) EvaluateWithPrompt(ctx context.Context, prompt string, metricType evaluation.MetricType) (*evaluation.MetricResult, error) {
	samples := make([]*evaluation.MetricResult, 0, j.numSamples)

	// Run multiple evaluation samples
	for i := 0; i < j.numSamples; i++ {
		sample, err := j.evaluateSingle(ctx, prompt)
		if err != nil {
			return nil, fmt.Errorf("sample %d failed: %w", i+1, err)
		}
		sample.MetricType = metricType
		samples = append(samples, sample)
	}

	// Aggregate samples
	aggregated, err := j.aggregator.AggregateSamples(samples)
	if err != nil {
		return nil, fmt.Errorf("failed to aggregate samples: %w", err)
	}

	aggregated.MetricType = metricType
	aggregated.EvaluatedAt = time.Now()

	return aggregated, nil
}

// evaluateSingle runs a single evaluation sample.
func (j *Judge) evaluateSingle(ctx context.Context, prompt string) (*evaluation.MetricResult, error) {
	// Create LLM request
	req := &model.LLMRequest{
		Model: j.modelName,
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					genai.NewPartFromText(prompt),
				},
			},
		},
		Config: j.config,
	}

	// Generate response (non-streaming)
	var response string
	for resp, err := range j.llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return nil, fmt.Errorf("LLM generation failed: %w", err)
		}

		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			if resp.Content.Parts[0].Text != "" {
				response = resp.Content.Parts[0].Text
			}
		}
	}

	if response == "" {
		return nil, fmt.Errorf("LLM returned empty response")
	}

	// Parse response
	return &evaluation.MetricResult{
		Score:          0,
		Status:         evaluation.EvalStatusNotEvaluated,
		JudgeResponses: []string{response},
	}, nil
}

// EvaluateScore evaluates and returns a numeric score from the prompt.
func (j *Judge) EvaluateScore(ctx context.Context, prompt string, metricType evaluation.MetricType) (*evaluation.MetricResult, error) {
	result, err := j.EvaluateWithPrompt(ctx, prompt, metricType)
	if err != nil {
		return nil, err
	}

	// Parse score from first judge response
	if len(result.JudgeResponses) > 0 {
		score, err := j.parser.ParseScore(result.JudgeResponses[0])
		if err == nil {
			result.Score = score
		}
	}

	return result, nil
}

// EvaluateVerdict evaluates and returns a binary yes/no verdict.
func (j *Judge) EvaluateVerdict(ctx context.Context, prompt string, metricType evaluation.MetricType) (*evaluation.MetricResult, error) {
	result, err := j.EvaluateWithPrompt(ctx, prompt, metricType)
	if err != nil {
		return nil, err
	}

	// Parse verdict from judge responses
	if len(result.JudgeResponses) > 0 {
		verdict, err := j.parser.ParseVerdict(result.JudgeResponses[0])
		if err == nil {
			if verdict == "yes" {
				result.Score = 1.0
			} else {
				result.Score = 0.0
			}
		}
	}

	return result, nil
}

// EvaluateRubrics evaluates multiple rubrics and returns rubric scores.
func (j *Judge) EvaluateRubrics(ctx context.Context, prompt string, rubrics map[string]evaluation.Rubric, metricType evaluation.MetricType) (*evaluation.MetricResult, error) {
	result, err := j.EvaluateWithPrompt(ctx, prompt, metricType)
	if err != nil {
		return nil, err
	}

	// Parse rubric scores from judge responses
	if len(result.JudgeResponses) > 0 {
		rubricScores, err := j.parser.ParseRubricScores(result.JudgeResponses[0], rubrics)
		if err == nil {
			result.RubricScores = rubricScores

			// Calculate overall score as average of rubric scores
			if len(rubricScores) > 0 {
				totalScore := 0.0
				for _, rs := range rubricScores {
					totalScore += rs.Score
				}
				result.Score = totalScore / float64(len(rubricScores))
			}
		}
	}

	return result, nil
}
