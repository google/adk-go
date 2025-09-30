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

package gemini

import (
	"context"
	"fmt"
	"iter"

	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/llm"
	"google.golang.org/genai"
)

// TODO: test coverage
type model struct {
	client *genai.Client
	name   string
}

func NewModel(ctx context.Context, modelName string, cfg *genai.ClientConfig) (llm.Model, error) {
	client, err := genai.NewClient(ctx, cfg)
	if err != nil {
		return nil, err
	}
	return &model{name: modelName, client: client}, nil
}

func (m *model) Name() string {
	return m.name
}

// Generate calls the model synchronously returning result from the first candidate.
func (m *model) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	m.maybeAppendUserContent(req)

	resp, err := m.client.Models.GenerateContent(ctx, m.name, req.Contents, req.GenerateConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to call model: %w", err)
	}
	if len(resp.Candidates) == 0 {
		// shouldn't happen?
		return nil, fmt.Errorf("empty response")
	}
	return llm.CreateResponse(resp), nil
}

// GenerateStream calls the model synchronously.
func (m *model) GenerateStream(ctx context.Context, req *llm.Request) iter.Seq2[*llm.Response, error] {
	m.maybeAppendUserContent(req)
	aggregator := llminternal.NewStreamingResponseAggregator()
	return func(yield func(*llm.Response, error) bool) {
		for resp, err := range m.client.Models.GenerateContentStream(ctx, m.name, req.Contents, req.GenerateConfig) {
			if err != nil {
				yield(nil, err)
				return
			}
			for llmResponse, err := range aggregator.ProcessResponse(ctx, resp) {
				if !yield(llmResponse, err) {
					return // Consumer stopped
				}
			}
		}
		if closeResult := aggregator.Close(); closeResult != nil {
			yield(closeResult, nil)
		}
	}
}

// maybeAppendUserContent appends a user content, so that model can continue to output.
func (m *model) maybeAppendUserContent(req *llm.Request) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents, genai.NewContentFromText("Handle the requests as specified in the System Instruction.", "user"))
	}

	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != "user" {
		req.Contents = append(req.Contents, genai.NewContentFromText("Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.", "user"))
	}
}
