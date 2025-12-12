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

// Package anthropic implements the model.LLM interface backed by Claude models
// served via Vertex AI.
package anthropic

import (
	"context"
	"fmt"
	"iter"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/anthropics/anthropic-sdk-go/vertex"

	"google.golang.org/adk/model"
)

const (
	envProjectID = "GOOGLE_CLOUD_PROJECT"
	envLocation  = "GOOGLE_CLOUD_LOCATION"

	defaultMaxTokens  = 8192
	defaultOAuthScope = "https://www.googleapis.com/auth/cloud-platform"
)

const (
	ProviderVertexAI   = "vertex_ai"
	ProviderAnthropic  = "anthropic"
	ProviderAWSBedrock = "aws_bedrock"
)

// Config controls how the Anthropic-backed model is initialized.
type Config struct {
	// Provider indicates which service is used to access the Anthropic models.
	// Supported values are "vertex_ai", "aws_bedrock", and "anthropic". Default is "vertex_ai".
	Provider string
	// APIKey is the API key used to authenticate with the Anthropic API.
	// Only required when Provider is "anthropic".
	APIKey string
	// MaxTokens sets the maximum number of tokens the model can generate.
	MaxTokens int64
	// ClientOptions are forwarded to the underlying Anthropics SDK client.
	ClientOptions []option.RequestOption
}

func (c *Config) applyDefaults() {
	if c.ClientOptions == nil {
		c.ClientOptions = []option.RequestOption{}
	}
	if c.MaxTokens == 0 {
		c.MaxTokens = defaultMaxTokens
	}
	if c.Provider == "" {
		c.Provider = ProviderVertexAI
	}
}

type AnthropicModel struct {
	client anthropic.Client

	name      string
	maxTokens int64
}

// NewModel returns [model.LLM] backed by the Anthropic API.
func NewModel(ctx context.Context, modelName string, cfg *Config) (model.LLM, error) {
	if modelName == "" {
		return nil, fmt.Errorf("model name must be provided")
	}

	if cfg == nil {
		cfg = &Config{}
	}
	cfg.applyDefaults()

	opts := append([]option.RequestOption{}, cfg.ClientOptions...)

	switch cfg.Provider {
	case ProviderAnthropic:
		if cfg.APIKey == "" {
			return nil, fmt.Errorf("API key must be provided to use Anthropic provider")
		}
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	case ProviderAWSBedrock:
		// Do nothing special for AWS Bedrock for now. User need to provide the client option
		// via `bedrock.WithConfig()` or `bedrock.WithLoadDefaultConfig()`.
	default:
		projectID := os.Getenv(envProjectID)
		location := os.Getenv(envLocation)
		if projectID == "" || location == "" {
			return nil, fmt.Errorf("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set to use Anthropic on Vertex")
		}
		opts = append(opts, vertex.WithGoogleAuth(ctx, location, projectID, defaultOAuthScope))
	}

	return &AnthropicModel{
		name:      modelName,
		maxTokens: cfg.MaxTokens,
		client:    anthropic.NewClient(opts...),
	}, nil
}

func (m *AnthropicModel) Name() string {
	return m.name
}

// GenerateContent issues a Messages.New call. When stream is true, the Anthropic
// streaming API is used to emit partial responses as they arrive.
func (m *AnthropicModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if stream {
		return m.generateStream(ctx, req)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req)
		if !yield(resp, err) {
			return
		}
	}
}

func (m *AnthropicModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("llm request must not be empty")
	}

	requestBuilder := RequestBuilder{modelName: m.name, maxTokens: m.maxTokens}
	params, err := requestBuilder.FromLLMRequest(req)
	if err != nil {
		return nil, err
	}

	msg, err := m.client.Messages.New(ctx, *params)
	if err != nil {
		return nil, fmt.Errorf("failed to send llm request to anthropic: %w", err)
	}

	responseBuilder := ResponseBuilder{}
	return responseBuilder.FromMessage(msg)
}

func (m *AnthropicModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		builder := RequestBuilder{modelName: m.name, maxTokens: m.maxTokens}
		params, err := builder.FromLLMRequest(req)
		if err != nil {
			yield(nil, err)
			return
		}

		stream := m.client.Messages.NewStreaming(ctx, *params)
		for resp, err := range readStreamEvents(stream) {
			if !yield(resp, err) {
				return
			}
		}
	}
}

func readStreamEvents(stream *ssestream.Stream[anthropic.MessageStreamEventUnion]) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if stream == nil {
			yield(nil, fmt.Errorf("the stream is empty"))
			return
		}
		defer func() {
			_ = stream.Close()
		}()

		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("got the stream error: %w", err))
			return
		}

		var message anthropic.Message
		for stream.Next() {
			event := stream.Current()
			if err := message.Accumulate(event); err != nil {
				yield(nil, fmt.Errorf("accumulate stream event error: %w", err))
				return
			}

			partialResponse := parsePartialStreamEvent(event)
			if partialResponse != nil {
				if !yield(partialResponse, nil) {
					return
				}
			}

			if _, ok := event.AsAny().(anthropic.MessageStopEvent); ok {
				responseBuilder := ResponseBuilder{}
				finalResponse, err := responseBuilder.FromMessage(&message)
				if err != nil {
					yield(nil, err)
					return
				}
				finalResponse.TurnComplete = true
				if !yield(finalResponse, nil) {
					return
				}
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("got the stream error: %w", err))
		}
	}
}
