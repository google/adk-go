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
