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

package openaimodel

import (
	"context"
	"fmt"
	"iter"
	"net/http"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/llminternal/converters"
	"google.golang.org/adk/v2/model"
)

// ClientConfig configures the OpenAI client. Mirrors model/gemini, which takes
// *genai.ClientConfig. Empty APIKey/BaseURL fall back to the OPENAI_API_KEY /
// OPENAI_BASE_URL env vars (handled by openai-go's default options).
type ClientConfig struct {
	APIKey     string
	BaseURL    string       // for OpenAI-compatible endpoints
	HTTPClient *http.Client // optional; e.g. for tests

	// Options is an escape hatch for advanced openai-go request options,
	// appended after the options derived from the fields above.
	Options []option.RequestOption
}

type openAIModel struct {
	client *openai.Client
	name   string
}

// NewModel constructs a new openAIModel.
// The context is unused but kept for signature parity with other model constructors (e.g., gemini.NewModel).
func NewModel(_ context.Context, modelName string, cfg *ClientConfig) (model.LLM, error) {
	if modelName == "" {
		return nil, ErrModelNameRequired
	}
	if cfg == nil {
		cfg = &ClientConfig{}
	}
	var opts []option.RequestOption
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	if cfg.HTTPClient != nil {
		opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
	}
	opts = append(opts, cfg.Options...)
	client := openai.NewClient(opts...)
	return &openAIModel{client: &client, name: modelName}, nil
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
		defer func() { _ = stream.Close() }()

		aggregator := llminternal.NewStreamingResponseAggregator()
		translator := newStreamTranslator()

		var openaiResp *responses.Response
		// Set alongside openaiResp by a terminal event, the only kind that says
		// why the turn ended.
		var sawFinalResponse bool

		for stream.Next() {
			event := stream.Current()
			// First terminal object wins: a later one, or a stray
			// "response.created", would relabel a truncated turn a clean stop.
			if !sawFinalResponse {
				switch event.Type {
				case responseCreated:
					created := event.AsResponseCreated()
					openaiResp = &created.Response
				case responseCompleted:
					completed := event.AsResponseCompleted()
					openaiResp = &completed.Response
					sawFinalResponse = true
				case responseIncomplete:
					incomplete := event.AsResponseIncomplete()
					openaiResp = &incomplete.Response
					sawFinalResponse = true
				}
			}

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
				if err == nil && openaiResp != nil {
					attachMetadata(resp, openaiResp)
				}
				if !yield(resp, err) {
					return
				}
			}
		}
		if err := stream.Err(); err != nil {
			yield(nil, err)
			return
		}

		final := aggregator.Close()
		if final == nil {
			// No delta reached the aggregator, but the turn can still be
			// complete: a provider that batches its output puts the whole
			// message on the terminal event. Rebuild it the way the blocking
			// path would, so the two agree on such a stream.
			if !sawFinalResponse {
				return
			}
			genaiResp, err := convertResponse(openaiResp)
			if err != nil {
				// Blocking fails the call on unusable output; match it rather
				// than pass an empty turn off as a successful one.
				yield(nil, err)
				return
			}
			final = converters.Genai2LLMResponse(genaiResp)
		}
		finalizeStreamResponse(final, openaiResp, sawFinalResponse)
		yield(final, nil)
	}
}

// finalizeStreamResponse closes out a streamed turn on the aggregated response.
//
// Deltas carry no finish reason (see singlePartResponse), so this is the one
// response that marks the turn complete, and the last point where the terminal
// OpenAI response is in reach — hence the fields copied here, which are what
// let a streamed turn report what the same turn reports unstreamed. An erroring
// stream never arrives: the error ends the turn in place of TurnComplete.
func finalizeStreamResponse(final *model.LLMResponse, openaiResp *responses.Response, sawFinalResponse bool) {
	final.TurnComplete = true
	if openaiResp != nil {
		attachMetadata(final, openaiResp)
		final.ModelVersion = string(openaiResp.Model)
	}
	if !sawFinalResponse {
		// The model never said why it stopped, and finishReason would read that
		// silence as a clean stop. Usage is left alone for the same reason: only
		// "response.created" is in hand and its counts are zero, which would
		// report a turn that did real work as having cost nothing.
		final.FinishReason = genai.FinishReasonUnspecified
		return
	}
	// sawFinalResponse implies openaiResp != nil.
	final.UsageMetadata = convertUsage(openaiResp.Usage)
	final.FinishReason = finishReason(openaiResp)
	final.LogprobsResult = convertLogprobs(openaiResp.Output)
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
