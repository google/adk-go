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

package anthropic

import (
	"context"
	"fmt"
	"iter"
	"log/slog"
	"os"
	"regexp"
	"strings"

	anthropicapi "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/vertex"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/model/anthropic/internal/converters"
)

const defaultMaxTokens = 4096

type anthropicModel struct {
	client           anthropicapi.Client
	name             anthropicapi.Model
	variant          string
	defaultMaxTokens int
}

// NewModel returns [model.LLM], backed by Anthropic Claude.
//
// It creates an Anthropic client based on the provided configuration.
// If Variant is not specified, it checks the ANTHROPIC_USE_VERTEX environment variable.
//
// For direct Anthropic API, set APIKey in the config or the ANTHROPIC_API_KEY
// environment variable.
//
// For Vertex AI, set VertexProjectID and VertexLocation in the config or use
// GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.
func NewModel(ctx context.Context, modelName anthropicapi.Model, cfg *Config) (model.LLM, error) {
	if cfg == nil {
		cfg = &Config{}
	}

	variant := cfg.Variant
	if variant == "" {
		variant = GetVariant()
	}

	var client anthropicapi.Client

	switch variant {
	case VariantVertexAI:
		projectID := cfg.VertexProjectID
		if projectID == "" {
			projectID = os.Getenv("GOOGLE_CLOUD_PROJECT")
		}
		if projectID == "" {
			return nil, fmt.Errorf("VertexProjectID is required for Vertex AI (set GOOGLE_CLOUD_PROJECT)")
		}

		location := cfg.VertexLocation
		if location == "" {
			location = os.Getenv("GOOGLE_CLOUD_LOCATION")
		}
		if location == "" {
			return nil, fmt.Errorf("VertexLocation is required for Vertex AI (set GOOGLE_CLOUD_LOCATION)")
		}

		client = newVertexClient(ctx, cfg)
	default:
		client = newAPIClient(cfg)
	}

	maxTokens := cfg.DefaultMaxTokens
	if maxTokens == 0 {
		maxTokens = defaultMaxTokens
	}

	return &anthropicModel{
		client:           client,
		name:             modelName,
		variant:          variant,
		defaultMaxTokens: maxTokens,
	}, nil
}

// newAPIClient creates a client for the direct Anthropic API.
func newAPIClient(cfg *Config) anthropicapi.Client {
	opts := []option.RequestOption{}

	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}

	return anthropicapi.NewClient(opts...)
}

// newVertexClient creates a client for Anthropic via Vertex AI.
// Note: The caller must validate that projectID and region are set before calling this.
func newVertexClient(ctx context.Context, cfg *Config) anthropicapi.Client {
	projectID := cfg.VertexProjectID
	if projectID == "" {
		projectID = os.Getenv("GOOGLE_CLOUD_PROJECT")
	}

	location := cfg.VertexLocation
	if location == "" {
		location = os.Getenv("GOOGLE_CLOUD_LOCATION")
	}

	return anthropicapi.NewClient(
		vertex.WithGoogleAuth(ctx, location, projectID),
	)
}

// Name returns the model name.
func (m *anthropicModel) Name() string {
	return string(m.name)
}

// GenerateContent calls the Anthropic model.
func (m *anthropicModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	m.maybeAppendUserContent(req)

	if stream {
		return m.generateStream(ctx, req)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req)
		yield(resp, err)
	}
}

// generate calls the model synchronously.
func (m *anthropicModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	// Vertex AI doesn't support OutputConfig — fall back to prompt-based JSON.
	if m.variant == VariantVertexAI && req.Config != nil && req.Config.ResponseSchema != nil {
		return m.generateWithPromptBasedJSON(ctx, req)
	}

	params, err := m.convertRequest(req)
	if err != nil {
		return nil, fmt.Errorf("failed to convert request: %w", err)
	}

	msg, err := m.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("failed to call model: %w", err)
	}

	resp, err := converters.MessageToLLMResponse(msg)
	if err != nil {
		return nil, fmt.Errorf("failed to convert response: %w", err)
	}

	return resp, nil
}

// generateWithPromptBasedJSON falls back to prompt-based JSON for Vertex AI.
// It embeds the JSON schema in the system instruction and asks the model to respond with valid JSON.
func (m *anthropicModel) generateWithPromptBasedJSON(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	req = embedSchemaAsSystemPrompt(req)

	params, err := m.convertRequest(req)
	if err != nil {
		return nil, fmt.Errorf("failed to convert request: %w", err)
	}

	msg, err := m.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("failed to call model: %w", err)
	}

	resp, err := converters.MessageToLLMResponse(msg)
	if err != nil {
		return nil, fmt.Errorf("failed to convert response: %w", err)
	}

	stripMarkdownFromResponse(ctx, resp)

	return resp, nil
}

// generateStream returns a stream of responses from the model.
func (m *anthropicModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	// Vertex AI doesn't support OutputConfig — embed schema as a system prompt instead.
	promptBasedJSON := m.variant == VariantVertexAI && req.Config != nil && req.Config.ResponseSchema != nil
	if promptBasedJSON {
		req = embedSchemaAsSystemPrompt(req)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		params, err := m.convertRequest(req)
		if err != nil {
			yield(nil, fmt.Errorf("failed to convert request: %w", err))
			return
		}

		stream := m.client.Messages.NewStreaming(ctx, params)
		message := anthropicapi.Message{}

		for stream.Next() {
			event := stream.Current()

			// Accumulate the message
			if err := message.Accumulate(event); err != nil {
				yield(nil, fmt.Errorf("failed to accumulate message: %w", err))
				return
			}

			// Handle different event types for streaming
			switch ev := event.AsAny().(type) {
			case anthropicapi.ContentBlockDeltaEvent:
				// Handle text deltas
				switch delta := ev.Delta.AsAny().(type) {
				case anthropicapi.TextDelta:
					resp := converters.StreamDeltaToPartialResponse(delta.Text)
					if !yield(resp, nil) {
						return
					}
				case anthropicapi.ThinkingDelta:
					resp := converters.StreamThinkingDeltaToPartialResponse(delta.Thinking)
					if !yield(resp, nil) {
						return
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("stream error: %w", err))
			return
		}

		// Yield the final complete response
		finalResp, err := converters.MessageToLLMResponse(&message)
		if err != nil {
			yield(nil, fmt.Errorf("failed to convert stream response: %w", err))
			return
		}
		if promptBasedJSON {
			stripMarkdownFromResponse(ctx, finalResp)
		}
		finalResp.TurnComplete = true
		yield(finalResp, nil)
	}
}

// convertRequest converts an LLMRequest to Anthropic MessageNewParams.
func (m *anthropicModel) convertRequest(req *model.LLMRequest) (anthropicapi.MessageNewParams, error) {
	messages, err := converters.ContentsToMessages(req.Contents)
	if err != nil {
		return anthropicapi.MessageNewParams{}, fmt.Errorf("failed to convert contents: %w", err)
	}

	params := anthropicapi.MessageNewParams{
		Model:     anthropicapi.Model(m.name),
		Messages:  messages,
		MaxTokens: int64(m.defaultMaxTokens),
	}

	if req.Config != nil {
		// System instruction
		if req.Config.SystemInstruction != nil {
			params.System = converters.SystemInstructionToSystem(req.Config.SystemInstruction)
		}

		// Generation parameters
		if req.Config.Temperature != nil {
			params.Temperature = anthropicapi.Float(float64(*req.Config.Temperature))
		}
		if req.Config.TopP != nil {
			params.TopP = anthropicapi.Float(float64(*req.Config.TopP))
		}
		if req.Config.TopK != nil {
			params.TopK = anthropicapi.Int(int64(*req.Config.TopK))
		}
		if len(req.Config.StopSequences) > 0 {
			params.StopSequences = req.Config.StopSequences
		}
		if req.Config.MaxOutputTokens > 0 {
			params.MaxTokens = int64(req.Config.MaxOutputTokens)
		}

		// Tools
		if len(req.Config.Tools) > 0 {
			params.Tools = converters.ToolsToAnthropicTools(req.Config.Tools)
		}

		// Tool choice from ToolConfig
		if req.Config.ToolConfig != nil {
			toolChoice, err := converters.ToolConfigToToolChoice(req.Config.ToolConfig)
			if err != nil {
				return anthropicapi.MessageNewParams{}, err
			}
			params.ToolChoice = toolChoice
		}

		// Structured output format (not supported on Vertex AI — handled via prompt-based fallback)
		if req.Config.ResponseSchema != nil && m.variant != VariantVertexAI {
			schemaMap := converters.SchemaToMap(req.Config.ResponseSchema)
			params.OutputConfig = anthropicapi.OutputConfigParam{
				Format: anthropicapi.JSONOutputFormatParam{
					Schema: schemaMap,
				},
			}
		}

		// Thinking config
		if req.Config.ThinkingConfig != nil {
			params.Thinking = converters.ThinkingConfigToAnthropicThinking(req.Config.ThinkingConfig)
		}
	}

	return params, nil
}

// maybeAppendUserContent ensures the conversation ends with a user message.
// Anthropic requires strictly alternating user/assistant turns.
func (m *anthropicModel) maybeAppendUserContent(req *model.LLMRequest) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents,
			genai.NewContentFromText("Handle the requests as specified in the System Instruction.", "user"))
		return
	}

	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != "user" {
		req.Contents = append(req.Contents,
			genai.NewContentFromText("Continue processing previous requests as instructed.", "user"))
	}
}

// embedSchemaAsSystemPrompt returns a cloned request with the JSON schema embedded
// in the system instruction and ResponseSchema cleared. This is the prompt-based
// fallback for Vertex AI, which doesn't support OutputConfig.
func embedSchemaAsSystemPrompt(req *model.LLMRequest) *model.LLMRequest {
	schemaJSON := converters.SchemaToJSONString(req.Config.ResponseSchema)

	jsonInstruction := fmt.Sprintf(`You must respond with valid JSON that conforms to the following JSON schema:

%s

Respond ONLY with the JSON object, no markdown code fences, no explanations.`, schemaJSON)

	// Clone the config to avoid mutating the original request.
	modifiedConfig := *req.Config
	if modifiedConfig.SystemInstruction == nil {
		modifiedConfig.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: jsonInstruction}},
		}
	} else {
		existingText := ""
		for _, part := range modifiedConfig.SystemInstruction.Parts {
			if part.Text != "" {
				existingText += part.Text + "\n\n"
			}
		}
		modifiedConfig.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: existingText + jsonInstruction}},
			Role:  modifiedConfig.SystemInstruction.Role,
		}
	}
	modifiedConfig.ResponseSchema = nil

	return &model.LLMRequest{
		Model:    req.Model,
		Contents: req.Contents,
		Config:   &modifiedConfig,
		Tools:    req.Tools,
	}
}

// markdownFenceRegex matches ```json ... ``` or ``` ... ``` code blocks.
// Uses non-greedy matching to handle the first complete fence block.
// The (?s) flag enables dotall mode so . matches newlines.
var markdownFenceRegex = regexp.MustCompile("(?s)^\\s*```(?:json)?\\s*\n?(.*?)\\s*```\\s*$")

// markdownFenceExtractRegex is a more permissive regex that extracts JSON from
// anywhere in the response, handling cases where the model adds preamble text.
var markdownFenceExtractRegex = regexp.MustCompile("(?s)```(?:json)?\\s*\n?(\\{.*?\\}|\\[.*?\\])\\s*```")

// stripMarkdownFromResponse removes markdown code fences from text parts in the response.
// This handles cases where the model wraps JSON in ```json ... ``` despite instructions.
// It tries two approaches:
// 1. First, check if the entire text is wrapped in fences (strict match)
// 2. If not, try to extract a JSON object/array from within fences (permissive match)
func stripMarkdownFromResponse(ctx context.Context, resp *model.LLMResponse) {
	if resp == nil || resp.Content == nil || len(resp.Content.Parts) == 0 {
		return
	}

	for _, part := range resp.Content.Parts {
		if part == nil || part.Text == "" {
			continue
		}

		original := part.Text

		// First try strict match: entire text is wrapped in fences
		if matches := markdownFenceRegex.FindStringSubmatch(part.Text); len(matches) > 1 {
			part.Text = strings.TrimSpace(matches[1])
			slog.DebugContext(ctx, "stripped markdown fences from JSON response (strict)",
				"original_length", len(original),
				"stripped_length", len(part.Text))
			continue
		}

		// Fall back to permissive match: extract JSON from within fences
		if matches := markdownFenceExtractRegex.FindStringSubmatch(part.Text); len(matches) > 1 {
			part.Text = strings.TrimSpace(matches[1])
			slog.DebugContext(ctx, "extracted JSON from markdown fences (permissive)",
				"original_length", len(original),
				"stripped_length", len(part.Text))
		}
	}
}
