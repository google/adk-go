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

package model

import (
	"context"
	"fmt"

	"github.com/google/adk-go"
	"google.golang.org/genai"
)

var _ adk.Model = (*GeminiModel)(nil)

type GeminiModel struct {
	client *genai.Client
	name   string
}

func NewGeminiModel(ctx context.Context, model string, cfg *genai.ClientConfig) (*GeminiModel, error) {
	client, err := genai.NewClient(ctx, cfg)
	if err != nil {
		return nil, err
	}
	return &GeminiModel{name: model, client: client}, nil
}

func (m *GeminiModel) Name() string {
	return m.name
}

func (m *GeminiModel) GenerateContent(ctx context.Context, req *adk.LLMRequest, stream bool) adk.LLMResponseStream {
	if m.client == nil {
		return func(yield func(*adk.LLMResponse, error) bool) {
			yield(nil, fmt.Errorf("model uninitialized"))
		}
	}
	if stream {
		return func(yield func(*adk.LLMResponse, error) bool) {
			for resp, err := range m.client.Models.GenerateContentStream(ctx, m.name, req.Contents, req.GenerateConfig) {
				if err != nil {
					yield(nil, err)
					return
				}
				if len(resp.Candidates) == 0 {
					// shouldn't happen?
					yield(nil, fmt.Errorf("empty response"))
					return
				}
				candidate := resp.Candidates[0]
				complete := candidate.FinishReason != ""
				if !yield(&adk.LLMResponse{
					Content:           candidate.Content,
					GroundingMetadata: candidate.GroundingMetadata,
					Partial:           !complete,
					TurnComplete:      complete,
					Interrupted:       false, // no interruptions in unary
				}, nil) {
					return
				}
			}
		}
	} else {
		return func(yield func(*adk.LLMResponse, error) bool) {
			resp, err := m.client.Models.GenerateContent(ctx, m.name, req.Contents, req.GenerateConfig)
			if err != nil {
				yield(nil, err)
				return
			}
			if len(resp.Candidates) == 0 {
				// shouldn't happen?
				yield(nil, fmt.Errorf("empty response"))
				return
			}
			candidate := resp.Candidates[0]
			if !yield(&adk.LLMResponse{
				Content:           candidate.Content,
				GroundingMetadata: candidate.GroundingMetadata,
			}, nil) {
				return
			}
		}
	}

	// TODO(hakim): write test (deterministic)
}
