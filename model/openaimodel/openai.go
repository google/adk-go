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
	"errors"
	"fmt"
	"iter"
	"net/http"
	"strings"

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
		attachFinishSignal(llmResp, resp, false)
		yield(llmResp, nil)
	}
}

func (m *openAIModel) generateStream(ctx context.Context, params responses.ResponseNewParams) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		stream := m.client.Responses.NewStreaming(ctx, params)
		defer func() { _ = stream.Close() }()

		aggregator := llminternal.NewStreamingResponseAggregator()
		translator := newStreamTranslator()

		var term terminalEvent

		for stream.Next() {
			event := stream.Current()
			// First terminal object wins: a later one, or a stray
			// "response.created", would relabel a truncated turn a clean stop.
			// An event whose response never decoded is not one of those, hence
			// carriesResponse.
			if !term.seen {
				switch event.Type {
				case responseCreated:
					created := event.AsResponseCreated()
					if carriesResponse(&created.Response) {
						term.resp = &created.Response
					}
				case responseCompleted:
					completed := event.AsResponseCompleted()
					if carriesResponse(&completed.Response) {
						term.resp, term.seen = &completed.Response, true
					}
				case responseIncomplete:
					incomplete := event.AsResponseIncomplete()
					if carriesResponse(&incomplete.Response) {
						term.resp, term.seen, term.incomplete = &incomplete.Response, true, true
					}
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
				if err == nil && term.resp != nil {
					attachMetadata(resp, term.resp)
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
		if !carriesContent(final) && term.seen {
			// The deltas contributed nothing that survived aggregation, but the
			// terminal event can still hold the whole turn: a batched message,
			// or a tool call the aggregator dropped. Rebuild it the way the
			// blocking path would, so the two agree on such a stream.
			genaiResp, err := convertResponse(term.resp)
			switch {
			case err == nil:
				final = converters.Genai2LLMResponse(genaiResp)
			case final != nil && isEmptyOutput(err):
				// Nothing to rebuild from, but the aggregator did produce a
				// turn: report it with the reason the event carries rather than
				// failing a call the model answered. A truncated turn is
				// exactly the shape that arrives with no output.
			default:
				// Blocking fails the call on unusable output; match it rather
				// than pass an empty turn off as a successful one.
				yield(nil, err)
				return
			}
		}
		if final == nil {
			// No aggregated turn and nothing to rebuild one from.
			return
		}
		if err := adoptTerminalCalls(final, term); err != nil {
			yield(nil, err)
			return
		}
		finalizeStreamResponse(final, term)
		yield(final, nil)
	}
}

// terminalEvent is what the stream's terminal event said, as distinct from what
// the response it carried repeated: a "response.incomplete" declares a turn cut
// short even when its payload reports no status and no reason.
type terminalEvent struct {
	resp *responses.Response
	// seen is set by a terminal event, the only kind that says why the turn
	// ended. It implies resp != nil.
	seen       bool
	incomplete bool
}

// carriesResponse reports whether an event delivered the response object the
// schema marks required. AsResponse* discards its unmarshal error and Response
// is a value field, so an omitted or empty object hands back a zero value that
// would otherwise outrank a well-formed event later in the turn. ID is required
// of a response, so its presence stands for the object's.
func carriesResponse(resp *responses.Response) bool {
	return resp.JSON.ID.Valid()
}

// isEmptyOutput reports whether a conversion failed for want of anything to
// convert, as against something unusable.
func isEmptyOutput(err error) bool {
	return errors.Is(err, ErrNoOutputItems) || errors.Is(err, ErrNoTextOrToolContent)
}

// carriesContent reports whether a response holds anything a caller can read.
// An aggregated turn can arrive empty: a streamed function call with no name is
// dropped, and deltas may contribute no part at all.
func carriesContent(resp *model.LLMResponse) bool {
	return resp != nil && resp.Content != nil && len(resp.Content.Parts) > 0
}

// adoptTerminalCalls appends the tool calls the terminal event carries that the
// aggregated turn does not already hold. Text is the deltas' to report, but a
// tool call has no partial form to prefer, and one lost in aggregation makes a
// turn that called a tool read as a plain answer. Blocking returns those calls
// for the same body.
func adoptTerminalCalls(final *model.LLMResponse, term terminalEvent) error {
	if !term.seen {
		return nil
	}
	held := map[string]bool{}
	if final.Content != nil {
		for _, part := range final.Content.Parts {
			if call := part.FunctionCall; call != nil {
				held[call.ID+"\x00"+call.Name] = true
			}
		}
	}
	var missing []*genai.Part
	for _, item := range term.resp.Output {
		if item.Type != "function_call" || held[item.CallID+"\x00"+item.Name] {
			continue
		}
		part, err := convertFunctionCall(item)
		if err != nil {
			// Blocking rejects the same body; fail rather than drop the call.
			return err
		}
		missing = append(missing, part)
	}
	if len(missing) == 0 {
		return nil
	}
	if final.Content == nil {
		final.Content = &genai.Content{Role: string(genai.RoleModel)}
	}
	final.Content.Parts = append(final.Content.Parts, missing...)
	return nil
}

// finalizeStreamResponse closes out a streamed turn on the aggregated response.
//
// Deltas carry no finish reason (see singlePartResponse), so this is the one
// response that marks the turn complete, and the last point where the terminal
// OpenAI response is in reach — hence the fields copied here, which are what
// let a streamed turn report what the same turn reports unstreamed. An erroring
// stream never arrives: the error ends the turn in place of TurnComplete.
func finalizeStreamResponse(final *model.LLMResponse, term terminalEvent) {
	final.TurnComplete = true
	if term.resp != nil {
		attachMetadata(final, term.resp)
		final.ModelVersion = string(term.resp.Model)
	}
	if !term.seen {
		// The model never said why it stopped, and finishReason would read that
		// silence as a clean stop. Usage is left alone for the same reason: only
		// "response.created" is in hand and its counts are zero, which would
		// report a turn that did real work as having cost nothing.
		final.FinishReason = genai.FinishReasonUnspecified
		return
	}
	// term.seen implies term.resp != nil.
	final.UsageMetadata = convertUsage(term.resp.Usage)
	final.FinishReason = finishReason(term.resp, term.incomplete)
	final.LogprobsResult = logprobsFor(term.resp, answerText(final.Content))
	attachFinishSignal(final, term.resp, term.incomplete)
}

// FinishMessageKey is the [model.LLMResponse.CustomMetadata] key under which a
// turn that ended badly but still carries an answer reports the provider's own
// account of why — a content filter's "content_filter", an incomplete reason
// this package does not map, or a server error message. Its FinishReason says
// the turn was cut short; this says what the provider called it.
//
// A turn left with nothing to read reports the same wording in ErrorCode and
// ErrorMessage instead, which callers treat as a failed turn.
//
// It reaches in-process callers, session storage and A2A metadata, but not
// REST: server/adkrest maps events field by field and omits CustomMetadata, so
// an ADK Web consumer sees the FinishReason alone.
const FinishMessageKey = "openai_finish_message"

// attachFinishSignal surfaces why a turn did not end cleanly, in the place that
// suits what the turn produced.
//
// ErrorCode is not advisory: tool/agenttool fails the tool call on a non-empty
// one and discards the content, server/adka2a marks the A2A task failed, and
// model/gemini leaves it empty for any candidate carrying content. A turn with
// content therefore reports the provider's wording as metadata beside a
// FinishReason that already says it was cut short; only a turn with nothing to
// read uses the error fields. genai's PromptFeedback suits neither branch: the
// framework converter reads it only for a response with no candidates, and
// convertResponse always emits one.
func attachFinishSignal(resp *model.LLMResponse, openaiResp *responses.Response, incompleteEvent bool) {
	if resp == nil || openaiResp == nil {
		return
	}
	switch resp.FinishReason {
	case genai.FinishReasonSafety, genai.FinishReasonOther:
	default:
		// MAX_TOKENS and STOP say all there is to say by themselves.
		return
	}
	msg := finishMessage(openaiResp, incompleteEvent)
	if carriesContent(resp) {
		if msg != "" {
			if resp.CustomMetadata == nil {
				resp.CustomMetadata = map[string]any{}
			}
			resp.CustomMetadata[FinishMessageKey] = msg
		}
		return
	}
	if resp.FinishReason == genai.FinishReasonSafety {
		// The same code model/gemini reports for a blocked prompt, so a caller
		// gating on it works across both.
		resp.ErrorCode = string(genai.BlockedReasonSafety)
	} else {
		resp.ErrorCode = string(genai.FinishReasonOther)
	}
	resp.ErrorMessage = msg
}

// answerText is the response's text as a caller reads it, thoughts excluded:
// what logprobs have to describe.
func answerText(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var text strings.Builder
	for _, part := range content.Parts {
		if part.Thought {
			continue
		}
		text.WriteString(part.Text)
	}
	return text.String()
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
