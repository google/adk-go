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

package openai

import (
	"context"
	"fmt"
	"iter"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/llminternal/converters"
	"google.golang.org/adk/model"
)

type openAIModel struct {
	client *openai.Client
	name   string
}

func NewModel(_ context.Context, modelName string, client openai.Client) (model.LLM, error) {
	// We drop the context because OpenAI doesn't take a context when creating a new Client
	if modelName == "" {
		return nil, ErrModelNameRequired
	}
	if len(client.Options) == 0 {
		return nil, ErrClientRequired
	}
	return &openAIModel{
		client: &client,
		name:   modelName,
	}, nil
}

func (m *openAIModel) Name() string { return m.name }

// GenerateContent converts a generic LLMRequest into an OpenAI-specific request,
// then calls the OpenAI API. It handles both streaming and non-streaming responses.
func (m *openAIModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if req == nil {
		return singleErrorSequence(ErrRequestNil)
	}
	params, err := buildOpenAIParams(m.name, req)
	if err != nil {
		return singleErrorSequence(err)
	}
	if stream {
		return m.generateStream(ctx, params)
	}
	return m.generate(ctx, params)
}

func (m *openAIModel) generate(ctx context.Context, params responses.ResponseNewParams) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.client.Responses.New(ctx, params)
		if err != nil {
			yield(nil, fmt.Errorf("openai: call failed: %w", err))
			return
		}
		genaiResp, err := convertResponse(resp)
		if err != nil {
			yield(nil, err)
			return
		}
		llmResp := converters.Genai2LLMResponse(genaiResp)
		attachMetadata(llmResp, resp)
		yield(llmResp, nil)
	}
}

func (m *openAIModel) generateStream(ctx context.Context, params responses.ResponseNewParams) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		stream := m.client.Responses.NewStreaming(ctx, params)
		if stream == nil {
			yield(nil, ErrStreamingUnavailable)
			return
		}
		if err := stream.Err(); err != nil {
			yield(nil, err)
			return
		}

		aggregator := llminternal.NewStreamingResponseAggregator()
		translator := newStreamTranslator()

		for stream.Next() {
			event := stream.Current()
			// First, we convert the OpenAI streaming event format to our generic genai.GenerateContentResponse format.
			genaiResp, err := translator.process(event)
			if err != nil {
				yield(nil, err)
				return
			}
			if genaiResp == nil {
				continue
			}
			// Then, we accumulate the streaming responses and yield them as discrete LLMResponses.
			for resp, err := range aggregator.ProcessResponse(ctx, genaiResp) {
				if !yield(resp, err) {
					return
				}
			}
		}
		if err := stream.Err(); err != nil {
			yield(nil, err)
			return
		}
		if err := stream.Close(); err != nil {
			yield(nil, err)
			return
		}
		if final := aggregator.Close(); final != nil {
			yield(final, nil)
		}
	}
}

func attachMetadata(resp *model.LLMResponse, openaiResp *responses.Response) {
	if resp == nil || openaiResp == nil {
		return
	}
	if resp.CustomMetadata == nil {
		resp.CustomMetadata = map[string]any{}
	}
	resp.CustomMetadata["openai_response_id"] = openaiResp.ID
	resp.CustomMetadata["openai_model"] = openaiResp.Model
}

func singleErrorSequence(err error) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(nil, err)
	}
}
