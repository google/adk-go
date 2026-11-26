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
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

var benchText = strings.Repeat("The quick brown fox jumps over the lazy dog. ", 8)

// benchEvents returns n alternating user and agent events: text turns,
// except that every fourth agent turn is a function call and the agent turn
// after it the call's response.
func benchEvents(n int, agentName string) []*session.Event {
	start := time.Unix(1_700_000_000, 0)
	events := make([]*session.Event, 0, n)
	for i := range n {
		ev := &session.Event{
			ID:           fmt.Sprintf("ev-%06d", i),
			InvocationID: fmt.Sprintf("inv-%06d", i/2),
			Timestamp:    start.Add(time.Duration(i) * time.Second),
			Author:       agentName,
		}
		switch {
		case i%2 == 0:
			ev.Author = "user"
			ev.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText(benchText, genai.RoleUser)}
		case i%8 == 1:
			ev.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{ID: fmt.Sprintf("call-%06d", i), Name: "lookup", Args: map[string]any{"q": "weather"}},
			}}}}
		case i%8 == 3:
			ev.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{ID: fmt.Sprintf("call-%06d", i-2), Name: "lookup", Response: map[string]any{"result": benchText}},
			}}}}
		default:
			ev.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText(benchText, genai.RoleModel)}
		}
		events = append(events, ev)
	}
	return events
}

// BenchmarkBuildContents measures prompt assembly, the per-model-call pass
// that copies the whole history into the request.
func BenchmarkBuildContents(b *testing.B) {
	for _, bc := range []struct {
		name    string
		events  int
		imageKB int
	}{
		{"events=100", 100, 0},
		{"events=1000", 1000, 0},
		{"events=10/image=1MB", 10, 1024},
	} {
		b.Run(bc.name, func(b *testing.B) {
			events := benchEvents(bc.events, "agent")
			if bc.imageKB > 0 {
				events[0].LLMResponse.Content = &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
					{Text: "what is this?"},
					{InlineData: &genai.Blob{MIMEType: "image/png", Data: make([]byte, bc.imageKB<<10)}},
				}}
			}
			user := genai.NewContentFromText("hello", genai.RoleUser)
			b.ReportAllocs()
			for b.Loop() {
				if _, err := buildContentsDefault("agent", "", "", events, false, user); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// benchGenerateContentConfig returns an agent config with sampling
// parameters, four safety settings, a thinking budget and labels, plus fds
// function declarations with nested object schemas and, when inlineBytes > 0,
// a system instruction carrying that many bytes of inline data.
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
						"limit":    {Type: genai.TypeInteger, Minimum: new(1.0), Maximum: new(100.0)},
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
// agent's GenerateContentConfig.
func BenchmarkBasicRequestProcessor(b *testing.B) {
	for _, bc := range []struct {
		name        string
		fds, inline int
	}{
		{"fds=20", 20, 0},
		{"inline=100KB", 0, 100 << 10},
	} {
		b.Run(bc.name, func(b *testing.B) {
			base, err := agent.New(agent.Config{Name: "bench_agent"})
			if err != nil {
				b.Fatal(err)
			}
			a := &mockLLMAgent{Agent: base, s: &State{
				Model:                 &mockLLM{name: "m"},
				GenerateContentConfig: benchGenerateContentConfig(bc.fds, bc.inline),
			}}
			ctx := icontext.NewInvocationContext(b.Context(), icontext.InvocationContextParams{Agent: a})
			req := &model.LLMRequest{}
			b.ReportAllocs()
			for b.Loop() {
				req.Config = nil
				for _, err := range basicRequestProcessor(ctx, req, &Flow{}) {
					if err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}
