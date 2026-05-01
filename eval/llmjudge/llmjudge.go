// Copyright 2026 Google LLC
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

// Package llmjudge provides an eval.Scorer that asks an LLM to grade an
// agent's output against the case's expected output. Mirrors the
// LLM-as-judge pattern from adk-python's evaluation suite.
//
// Usage:
//
//	judge := llmjudge.New(llmjudge.Config{LLM: gemini})
//	report, _ := (&eval.Runner{Agent: a, Scorer: judge, Threshold: 0.7}).Run(ctx, set)
//
// The default judge prompt asks the model to score on a 0–1 scale and
// to provide a one-line rationale. Concrete prompt templates are
// configurable via Config.PromptTemplate.
package llmjudge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/eval"
	"google.golang.org/adk/model"
)

// Config configures a Scorer.
type Config struct {
	// LLM is the model used to judge each case. Required.
	LLM model.LLM

	// PromptTemplate overrides the default judge prompt. The template
	// receives three placeholders: {expected}, {actual}, {input}. When
	// empty, DefaultPromptTemplate is used.
	PromptTemplate string

	// Name overrides the scorer name reported in eval reports.
	Name string
}

// DefaultPromptTemplate is the prompt the judge sees when no override
// is supplied. The format requests a strict JSON response so parsing
// is deterministic.
const DefaultPromptTemplate = `You are an evaluation judge. Score how well the actual output answers the question relative to the expected output.

Question:
{input}

Expected output:
{expected}

Actual output:
{actual}

Respond ONLY in JSON of this exact shape, no prose, no code fences:
{"score": <number between 0 and 1>, "reason": "<one short sentence>"}`

// Scorer implements eval.Scorer using an LLM as the judge.
type Scorer struct {
	cfg Config
}

// New constructs a Scorer.
func New(cfg Config) (*Scorer, error) {
	if cfg.LLM == nil {
		return nil, errors.New("llmjudge: Config.LLM is required")
	}
	if cfg.PromptTemplate == "" {
		cfg.PromptTemplate = DefaultPromptTemplate
	}
	if cfg.Name == "" {
		cfg.Name = "llm_judge"
	}
	return &Scorer{cfg: cfg}, nil
}

// Name implements eval.Scorer.
func (s *Scorer) Name() string { return s.cfg.Name }

// Score implements eval.Scorer. The case's Input and ExpectedOutput
// are substituted into the prompt template; the LLM's response is
// parsed as JSON to extract the score and reason.
func (s *Scorer) Score(ctx context.Context, c eval.Case, output string) (float64, string, error) {
	prompt := s.cfg.PromptTemplate
	prompt = strings.ReplaceAll(prompt, "{input}", c.Input)
	prompt = strings.ReplaceAll(prompt, "{expected}", c.ExpectedOutput)
	prompt = strings.ReplaceAll(prompt, "{actual}", output)

	req := &model.LLMRequest{
		Model: s.cfg.LLM.Name(),
		Contents: []*genai.Content{{
			Role:  genai.RoleUser,
			Parts: []*genai.Part{{Text: prompt}},
		}},
	}
	var combined strings.Builder
	for resp, err := range s.cfg.LLM.GenerateContent(ctx, req, false) {
		if err != nil {
			return 0, "", fmt.Errorf("llmjudge: model error: %w", err)
		}
		if resp == nil || resp.Content == nil {
			continue
		}
		for _, p := range resp.Content.Parts {
			if p != nil && p.Text != "" {
				combined.WriteString(p.Text)
			}
		}
	}
	body := strings.TrimSpace(combined.String())
	if body == "" {
		return 0, "", errors.New("llmjudge: empty model response")
	}
	score, reason, err := parseJudgeResponse(body)
	if err != nil {
		return 0, "", fmt.Errorf("llmjudge: parse response %q: %w", body, err)
	}
	return clamp01(score), reason, nil
}

// parseJudgeResponse extracts {score, reason} from the model's body.
// The strict JSON shape from DefaultPromptTemplate is preferred; a
// fallback regex pattern handles models that wrap their JSON in code
// fences or add explanatory prose despite the instruction.
func parseJudgeResponse(body string) (float64, string, error) {
	type judge struct {
		Score  float64 `json:"score"`
		Reason string  `json:"reason"`
	}
	// Direct JSON parse.
	var j judge
	if err := json.Unmarshal([]byte(body), &j); err == nil {
		return j.Score, j.Reason, nil
	}
	// Strip code fences.
	if stripped := stripCodeFences(body); stripped != body {
		if err := json.Unmarshal([]byte(stripped), &j); err == nil {
			return j.Score, j.Reason, nil
		}
	}
	// Last-ditch regex.
	if score, ok := extractScoreRegex(body); ok {
		reason := extractReasonRegex(body)
		return score, reason, nil
	}
	return 0, "", errors.New("could not parse JSON or extract score")
}

func stripCodeFences(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```json")
		s = strings.TrimPrefix(s, "```JSON")
		s = strings.TrimPrefix(s, "```")
		s = strings.TrimSuffix(s, "```")
	}
	return strings.TrimSpace(s)
}

var (
	scoreRe  = regexp.MustCompile(`(?i)"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)`)
	reasonRe = regexp.MustCompile(`(?i)"?reason"?\s*[:=]\s*"?([^"\n]+)"?`)
)

func extractScoreRegex(s string) (float64, bool) {
	m := scoreRe.FindStringSubmatch(s)
	if len(m) < 2 {
		return 0, false
	}
	v, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

func extractReasonRegex(s string) string {
	m := reasonRe.FindStringSubmatch(s)
	if len(m) < 2 {
		return ""
	}
	return strings.TrimSpace(m[1])
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
