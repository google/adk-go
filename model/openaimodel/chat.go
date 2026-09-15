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
	"iter"
	"time"

	"github.com/openai/openai-go/v3"

	"google.golang.org/adk/v2/model"
)

// errNotImplemented marks a scaffolded Chat Completions stub, and disappears
// with the conversion code that replaces it.
var errNotImplemented = errors.New("openai: chat completions support is not implemented yet")

// chatModel talks to the Chat Completions API, the surface OpenAI-compatible
// third-party providers implement. It is the [model.LLM] a [ClientConfig] with
// API set to [APIChatCompletions] produces.
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

// generate runs one blocking Chat Completions call, bounded by timeout when the
// caller asked for one.
func (m *chatModel) generate(ctx context.Context, params openai.ChatCompletionNewParams, timeout time.Duration) iter.Seq2[*model.LLMResponse, error] {
	return singleErrorSequence(errNotImplemented)
}

// generateStream runs one streamed Chat Completions call, yielding a partial
// per delta and one final response the blocking converter builds.
func (m *chatModel) generateStream(ctx context.Context, params openai.ChatCompletionNewParams, timeout time.Duration) iter.Seq2[*model.LLMResponse, error] {
	return singleErrorSequence(errNotImplemented)
}

// attachChatMetadata records the response ID and model on an LLMResponse, the
// way the Responses path records its own.
func attachChatMetadata(resp *model.LLMResponse, completion *openai.ChatCompletion) { //nolint:unused // scaffold; wired when the conversion code lands
}
