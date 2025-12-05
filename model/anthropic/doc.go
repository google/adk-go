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

// Package anthropic implements the [model.LLM] interface for Anthropic Claude models.
//
// This package provides support for both the direct Anthropic API and
// Anthropic models via Google Cloud Vertex AI.
//
// # Basic Usage
//
// To use the direct Anthropic API:
//
//	model, err := anthropic.NewModel(ctx, "claude-sonnet-4-20250514", &anthropic.Config{
//		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
//	})
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	agent, err := llmagent.New(llmagent.Config{
//		Name:  "my_agent",
//		Model: model,
//		// ... other config
//	})
//
// # Vertex AI Usage
//
// To use Anthropic models via Vertex AI:
//
//	model, err := anthropic.NewModel(ctx, "claude-sonnet-4@20250514", &anthropic.Config{
//		Variant:         anthropic.VariantVertexAI,
//		VertexProjectID: os.Getenv("GOOGLE_CLOUD_PROJECT"),
//		VertexRegion:    os.Getenv("GOOGLE_CLOUD_REGION"),
//	})
//
// Alternatively, set the ANTHROPIC_USE_VERTEX environment variable to "1" or "true"
// to automatically use Vertex AI without specifying the variant in code.
//
// # Supported Models
//
// The package supports all Anthropic Claude models, including:
//   - claude-sonnet-4-5 / claude-sonnet-4-5-20250929 (Claude Sonnet 4.5)
//   - claude-opus-4-5 / claude-opus-4-5-20251101 (Claude Opus 4.5)
//   - claude-sonnet-4-0 / claude-sonnet-4-20250514 (Claude Sonnet 4)
//   - claude-opus-4-0 / claude-opus-4-20250514 (Claude Opus 4)
//   - claude-opus-4-1-20250805 (Claude Opus 4.1)
//   - claude-haiku-4-5 / claude-haiku-4-5-20251001 (Claude Haiku 4.5)
//   - claude-3-5-haiku-latest / claude-3-5-haiku-20241022 (Claude 3.5 Haiku)
//
// Deprecated models (reaching end-of-life):
//   - claude-3-7-sonnet-latest / claude-3-7-sonnet-20250219 (EOL: Feb 19, 2026)
//   - claude-3-opus-latest / claude-3-opus-20240229 (EOL: Jan 5, 2026)
//
// For Vertex AI, model names follow the format: claude-{variant}-{version}@{date}
//
// # Features
//
// The package supports:
//   - Streaming and non-streaming responses
//   - Tool/function calling
//   - Extended thinking (mapped to genai.Part with Thought=true)
//   - Multimodal inputs (text, images)
//   - PDF document processing (beta)
//   - System instructions
package anthropic
