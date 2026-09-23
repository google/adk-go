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
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/llminternal/converters"
	"google.golang.org/adk/v2/model"
)

// chatModel talks to the Chat Completions API, the surface OpenAI-compatible
// third-party providers implement. It is what [NewModel] returns for a
// [ClientConfig] whose API is [APIChatCompletions].
type chatModel struct {
	client *openai.Client
	name   string
}

func (m *chatModel) Name() string { return m.name }

// GenerateContent converts a generic LLMRequest into a Chat Completions request
// and calls the API, streaming or not.
func (m *chatModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if req == nil {
		return singleErrorSequence(ErrRequestNil)
	}
	params, err := buildChatParams(m.name, req)
	if err != nil {
		return singleErrorSequence(err)
	}
	timeout := requestTimeout(req.Config)
	if stream {
		return m.generateStream(ctx, params, timeout)
	}
	return m.generate(ctx, params, timeout)
}

// generate runs one blocking Chat Completions call.
func (m *chatModel) generate(ctx context.Context, params openai.ChatCompletionNewParams, timeout time.Duration) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		// Shadowed, not reassigned: the closure captures ctx by reference, so
		// assigning to it would leave a second range over this sequence
		// starting from the deadline the first one already cancelled.
		ctx := ctx
		if timeout > 0 {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, timeout)
			defer cancel()
		}
		resp, err := m.client.Chat.Completions.New(ctx, params)
		if err != nil {
			yield(nil, fmt.Errorf("openai: call failed: %w", err))
			return
		}
		genaiResp, err := convertChatCompletion(resp)
		if err != nil {
			yield(nil, err)
			return
		}
		llmResp := converters.Genai2LLMResponse(genaiResp)
		attachChatMetadata(llmResp, resp)
		yield(llmResp, nil)
	}
}

// generateStream runs one streamed Chat Completions call, yielding a partial
// per delta and one final response built from the accumulated turn.
func (m *chatModel) generateStream(ctx context.Context, params openai.ChatCompletionNewParams, timeout time.Duration) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		ctx := ctx
		if timeout > 0 {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, timeout)
			defer cancel()
		}
		// Chat Completions reports usage on a streamed turn only when asked, so
		// a streamed turn would otherwise account for nothing.
		params.StreamOptions.IncludeUsage = param.NewOpt(true)

		stream := m.client.Chat.Completions.NewStreaming(ctx, params)
		defer func() { _ = stream.Close() }()

		aggregator := llminternal.NewStreamingResponseAggregator()
		translator := newChatStreamTranslator()

		for stream.Next() {
			genaiResp := translator.process(stream.Current())
			if genaiResp == nil {
				continue
			}
			for resp, err := range aggregator.ProcessResponse(ctx, genaiResp) {
				if err == nil {
					attachChatMetadata(resp, translator.completion())
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
		completion := translator.completion()
		genaiResp, err := convertChatCompletion(completion)
		switch {
		case err == nil && !carriesContent(final):
			// The deltas contributed nothing that survived aggregation, which is
			// what a turn of nothing but tool calls looks like. The snapshot
			// holds the whole turn.
			final = converters.Genai2LLMResponse(genaiResp)
		case err == nil:
			// Only the snapshot states the turn's tool calls, so it replaces the
			// aggregate whenever it retains everything already streamed.
			if content := genaiResp.Candidates[0].Content; completedContentSupersedes(final.Content, content) {
				final.Content = content
			}
		case carriesContent(final) && isEmptyOutput(err):
			// The snapshot holds nothing but the deltas produced a turn, so
			// report it rather than fail a call the model answered. An unusable
			// snapshot, such as a tool call whose arguments do not parse, fails
			// as blocking does: only it states the calls, so dropping one would
			// pass the turn off as whole.
		default:
			yield(nil, err)
			return
		}
		if final == nil {
			return
		}
		finalizeChatStreamResponse(final, completion)
		yield(final, nil)
	}
}

// finalizeChatStreamResponse closes out a streamed turn. Deltas carry no finish
// reason or usage, so this is the one response that reports them, and it takes
// both from the accumulated snapshot rather than from anything streamed.
func finalizeChatStreamResponse(final *model.LLMResponse, completion *openai.ChatCompletion) {
	final.TurnComplete = true
	if completion == nil {
		return
	}
	attachChatMetadata(final, completion)
	if completion.Model != "" {
		final.ModelVersion = completion.Model
	}
	final.UsageMetadata = convertChatUsage(completion.Usage)
	if len(completion.Choices) == 0 {
		// Nothing said why the turn ended, and reading that as a clean stop
		// would have a caller that retries on anything but STOP accept a
		// partial answer as final.
		final.FinishReason = genai.FinishReasonUnspecified
		return
	}
	choice := completion.Choices[0]
	final.FinishReason = chatFinishReason(choice.FinishReason)
	final.LogprobsResult = convertChatLogprobs(choice.Logprobs)
}

// attachChatMetadata records the response ID and model, under the same keys the
// Responses path uses. The model is a plain string here, which is what the key
// has always been documented to carry.
func attachChatMetadata(resp *model.LLMResponse, completion *openai.ChatCompletion) {
	if resp == nil || completion == nil || completion.ID == "" {
		return
	}
	if resp.CustomMetadata == nil {
		resp.CustomMetadata = map[string]any{}
	}
	resp.CustomMetadata["openai_response_id"] = completion.ID
	resp.CustomMetadata["openai_model"] = completion.Model
}
