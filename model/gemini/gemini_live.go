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
	"strings"
	"sync"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

type geminiLiveConnection struct {
	session   *genai.Session
	sendMu    sync.Mutex
	closeOnce sync.Once
	closeErr  error
}

func (c *geminiLiveConnection) Send(_ context.Context, req *model.LiveRequest) error {
	c.sendMu.Lock()
	defer c.sendMu.Unlock()

	switch {
	case req.Close:
		return c.Close()
	case req.ToolResponse != nil:
		return c.session.SendToolResponse(genai.LiveToolResponseInput{
			FunctionResponses: req.ToolResponse,
		})
	case req.RealtimeInput != nil:
		return c.session.SendRealtimeInput(genai.LiveRealtimeInput{
			Media: req.RealtimeInput.Media,
			Audio: req.RealtimeInput.Audio,
			Video: req.RealtimeInput.Video,
			Text:  req.RealtimeInput.Text,
		})
	case req.Content != nil:
		tc := true // default: model responds after this content
		if req.TurnComplete != nil {
			tc = *req.TurnComplete
		}
		return c.session.SendClientContent(genai.LiveClientContentInput{
			Turns:        []*genai.Content{req.Content},
			TurnComplete: &tc,
		})
	default:
		return fmt.Errorf("empty LiveRequest: at least one field must be set")
	}
}

func (c *geminiLiveConnection) Receive(_ context.Context) (*model.LLMResponse, error) {
	msg, err := c.session.Receive()
	if err != nil {
		return nil, err
	}
	return mapServerMessage(msg), nil
}

func (c *geminiLiveConnection) Close() error {
	c.closeOnce.Do(func() {
		c.closeErr = c.session.Close()
	})
	return c.closeErr
}

func mapServerMessage(msg *genai.LiveServerMessage) *model.LLMResponse {
	resp := &model.LLMResponse{
		CustomMetadata: make(map[string]any),
	}

	if msg.ServerContent != nil {
		sc := msg.ServerContent
		resp.TurnComplete = sc.TurnComplete
		resp.Interrupted = sc.Interrupted

		if sc.ModelTurn != nil {
			resp.Content = sc.ModelTurn
			// Detect audio content
			for _, part := range sc.ModelTurn.Parts {
				if part.InlineData != nil && strings.HasPrefix(part.InlineData.MIMEType, "audio/") {
					resp.CustomMetadata["is_audio"] = true
					break
				}
			}
		}

		if sc.InputTranscription != nil && sc.InputTranscription.Text != "" {
			resp.InputTranscription = &genai.Transcription{
				Text:     sc.InputTranscription.Text,
				Finished: sc.InputTranscription.Finished,
			}
			if resp.Content == nil {
				resp.Content = genai.NewContentFromText(sc.InputTranscription.Text, "user")
			}
		}
		if sc.OutputTranscription != nil && sc.OutputTranscription.Text != "" {
			resp.OutputTranscription = &genai.Transcription{
				Text:     sc.OutputTranscription.Text,
				Finished: sc.OutputTranscription.Finished,
			}
			if resp.Content == nil {
				resp.Content = genai.NewContentFromText(sc.OutputTranscription.Text, "model")
			}
		}

		if sc.GroundingMetadata != nil {
			resp.GroundingMetadata = sc.GroundingMetadata
		}
	}

	if msg.ToolCall != nil {
		parts := make([]*genai.Part, 0, len(msg.ToolCall.FunctionCalls))
		for _, fc := range msg.ToolCall.FunctionCalls {
			parts = append(parts, &genai.Part{FunctionCall: fc})
		}
		resp.Content = &genai.Content{Role: "model", Parts: parts}
	}

	if msg.ToolCallCancellation != nil {
		resp.CustomMetadata["tool_cancellation_ids"] = msg.ToolCallCancellation.IDs
	}

	if msg.GoAway != nil {
		resp.CustomMetadata["go_away"] = true
	}

	if msg.UsageMetadata != nil {
		resp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     msg.UsageMetadata.PromptTokenCount,
			CandidatesTokenCount: msg.UsageMetadata.ResponseTokenCount,
			TotalTokenCount:      msg.UsageMetadata.TotalTokenCount,
		}
	}

	return resp
}

// ConnectLive establishes a live bidirectional connection.
func (m *geminiModel) ConnectLive(ctx context.Context, req *model.LLMRequest) (model.LiveConnection, error) {
	cfg := req.LiveConfig
	if cfg == nil {
		cfg = &genai.LiveConnectConfig{}
	}
	sess, err := m.client.Live.Connect(ctx, m.modelName(req), cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to connect live: %w", err)
	}
	return &geminiLiveConnection{session: sess}, nil
}

var _ model.LiveCapableLLM = (*geminiModel)(nil)
