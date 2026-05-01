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

// prompt_optimizer ranks prompt templates by accuracy on a small QA set.
package main

import (
	"context"
	"fmt"
	"strings"

	"google.golang.org/adk/eval"
	"google.golang.org/adk/optimize"
)

// stubAgent simulates an LLM that follows the prompt template loosely:
// when the template includes the words "step-by-step", responses
// include the reference answer's keywords more often.
type stubAgent struct{ template string }

func (s *stubAgent) Run(_ context.Context, _, _, q string) (string, error) {
	tpl := strings.ToLower(s.template)
	verbose := strings.Contains(tpl, "step-by-step") || strings.Contains(tpl, "explain your reasoning")
	switch q {
	case "How long is the warranty?":
		if verbose {
			return "Step 1: check policy. Step 2: confirm. The warranty is 2 years.", nil
		}
		return "Maybe a year.", nil
	case "What is the return window?":
		if verbose {
			return "You may return items within 30 days of delivery.", nil
		}
		return "I think a couple weeks.", nil
	case "Do you ship internationally?":
		if verbose {
			return "Yes, we ship to most countries via DHL.", nil
		}
		return "Sometimes.", nil
	case "What payment methods are accepted?":
		if verbose {
			return "We accept Visa, Mastercard, AmEx, and PayPal.", nil
		}
		return "Card.", nil
	}
	return "I don't know.", nil
}

func main() {
	templates := []string{
		"Answer the question.",
		"Answer concisely.",
		"Answer in one sentence.",
		"Step-by-step, answer the question.",
		"Explain your reasoning, then answer.",
		"Be helpful and explain.",
	}

	variants := make([]*optimize.Variant, 0, len(templates))
	for i, t := range templates {
		variants = append(variants, &optimize.Variant{
			ID:          fmt.Sprintf("v%d", i+1),
			Spec:        t,
			Description: t,
		})
	}

	evalSet := &eval.Set{
		Name: "product_faq",
		Cases: []eval.Case{
			{ID: "q1", Input: "How long is the warranty?", ExpectedOutput: "2 years"},
			{ID: "q2", Input: "What is the return window?", ExpectedOutput: "30 days"},
			{ID: "q3", Input: "Do you ship internationally?", ExpectedOutput: "ship to"},
			{ID: "q4", Input: "What payment methods are accepted?", ExpectedOutput: "Visa\nMastercard\nPayPal"},
		},
	}

	scoreFn := func(ctx context.Context, v *optimize.Variant) (float64, map[string]any, error) {
		template := v.Spec.(string)
		runner := &eval.Runner{
			Agent:     &stubAgent{template: template},
			Scorer:    eval.ContainsScorer{},
			Threshold: 1.0,
		}
		report, err := runner.Run(ctx, evalSet)
		if err != nil {
			return 0, nil, err
		}
		return report.PassRate(), map[string]any{
			"pass":  report.PassCount,
			"fail":  report.FailCount,
			"error": report.ErrorCount,
		}, nil
	}

	search := &optimize.Search{
		Sampler: optimize.NewGridSampler("templates", variants),
		Score:   scoreFn,
	}
	results, err := search.Run(context.Background())
	if err != nil {
		fmt.Println("Search:", err)
		return
	}
	fmt.Println("=== prompt template ranking (best first) ===")
	for _, r := range results {
		v := r.Variant
		fmt.Printf("  pass_rate=%.2f  meta=%v  template=%q\n",
			r.Score, r.Meta, v.Spec.(string))
	}
}
