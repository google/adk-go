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

package llminternal

import (
	"bytes"
	"fmt"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
)

// benchGenerateContentConfig returns an agent GenerateContentConfig of the
// kind basicRequestProcessor copies on every model call: sampling parameters,
// four safety settings, a thinking budget and labels, plus fds function
// declarations whose parameters are nested object schemas (8 schema nodes
// each) and, when inlineBytes > 0, a system instruction that carries that many
// bytes of inline data.
func benchGenerateContentConfig(fds, inlineBytes int) *genai.GenerateContentConfig {
	cfg := &genai.GenerateContentConfig{
		Temperature:     genai.Ptr[float32](0.2),
		TopP:            genai.Ptr[float32](0.95),
		TopK:            genai.Ptr[float32](40),
		MaxOutputTokens: 2048,
		Seed:            genai.Ptr[int32](7),
		StopSequences:   []string{"END", "STOP"},
		SafetySettings: []*genai.SafetySetting{
			{Category: genai.HarmCategoryHarassment, Threshold: genai.HarmBlockThresholdBlockOnlyHigh},
			{Category: genai.HarmCategoryHateSpeech, Threshold: genai.HarmBlockThresholdBlockOnlyHigh},
			{Category: genai.HarmCategorySexuallyExplicit, Threshold: genai.HarmBlockThresholdBlockOnlyHigh},
			{Category: genai.HarmCategoryDangerousContent, Threshold: genai.HarmBlockThresholdBlockOnlyHigh},
		},
		ThinkingConfig: &genai.ThinkingConfig{IncludeThoughts: true, ThinkingBudget: genai.Ptr[int32](1024)},
		Labels:         map[string]string{"team": "adk", "env": "bench"},
	}
	if fds > 0 {
		decls := make([]*genai.FunctionDeclaration, fds)
		for i := range decls {
			decls[i] = &genai.FunctionDeclaration{
				Name:        fmt.Sprintf("lookup_order_%d", i),
				Description: "Looks up an order and returns its line items.",
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"order_id": {Type: genai.TypeString, Description: "Order identifier."},
						"limit":    {Type: genai.TypeInteger, Minimum: genai.Ptr(1.0), Maximum: genai.Ptr(100.0)},
						"status":   {Type: genai.TypeString, Enum: []string{"open", "shipped", "closed"}},
						"items": {
							Type: genai.TypeArray,
							Items: &genai.Schema{
								Type: genai.TypeObject,
								Properties: map[string]*genai.Schema{
									"sku":      {Type: genai.TypeString},
									"quantity": {Type: genai.TypeInteger},
								},
								Required: []string{"sku"},
							},
						},
					},
					Required: []string{"order_id"},
				},
			}
		}
		cfg.Tools = []*genai.Tool{{FunctionDeclarations: decls}}
	}
	if inlineBytes > 0 {
		cfg.SystemInstruction = &genai.Content{
			Role: genai.RoleUser,
			Parts: []*genai.Part{
				{Text: "You are a support agent. Follow the attached policy."},
				{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: bytes.Repeat([]byte{0xA5}, inlineBytes)}},
			},
		}
	}
	return cfg
}

// BenchmarkBasicRequestProcessor measures the per-model-call copy of the
// agent's GenerateContentConfig, which basicRequestProcessor makes so request
// processors and callbacks can edit req.Config without touching the agent.
func BenchmarkBasicRequestProcessor(b *testing.B) {
	for _, bc := range []struct {
		name        string
		fds, inline int
	}{
		{"fds=0", 0, 0},
		{"fds=5", 5, 0},
		{"fds=20", 20, 0},
		{"inline=100KB", 0, 100 << 10},
	} {
		b.Run(bc.name, func(b *testing.B) {
			base, err := agent.New(agent.Config{Name: "bench_agent"})
			if err != nil {
				b.Fatal(err)
			}
			a := &mockLLMAgent{
				Agent: base,
				s: &State{
					Model:                 &mockLLM{name: "m"},
					GenerateContentConfig: benchGenerateContentConfig(bc.fds, bc.inline),
				},
			}
			ctx := icontext.NewInvocationContext(b.Context(), icontext.InvocationContextParams{Agent: a})
			f := &Flow{}
			req := &model.LLMRequest{}
			b.ReportAllocs()
			for b.Loop() {
				req.Config = nil
				for _, err := range basicRequestProcessor(ctx, req, f) {
					if err != nil {
						b.Fatal(err)
					}
				}
			}
			if req.Config == a.s.GenerateContentConfig {
				b.Fatal("req.Config aliases the agent's GenerateContentConfig")
			}
		})
	}
}
