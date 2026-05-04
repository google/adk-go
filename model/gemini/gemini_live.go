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
	"time"

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
			Media:          req.RealtimeInput.Media,
			Audio:          req.RealtimeInput.Audio,
			Video:          req.RealtimeInput.Video,
			Text:           req.RealtimeInput.Text,
			ActivityStart:  req.RealtimeInput.ActivityStart,
			ActivityEnd:    req.RealtimeInput.ActivityEnd,
			AudioStreamEnd: req.RealtimeInput.AudioStreamEnd,
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
	beforeRecv := time.Now()
	msg, err := c.session.Receive()
	if err != nil {
		return nil, err
	}
	now := time.Now()
	resp := mapServerMessage(msg)
	resp.ReceivedAt = now
	resp.ReceiveLatency = now.Sub(beforeRecv)
	return resp, nil
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
		mapServerContent(msg.ServerContent, resp)
	}
	if msg.ToolCall != nil {
		mapToolCall(msg.ToolCall, resp)
	}
	if msg.ToolCallCancellation != nil {
		resp.CustomMetadata["tool_cancellation_ids"] = msg.ToolCallCancellation.IDs
	}
	mapGoAway(msg, resp)
	mapSessionResumptionUpdate(msg, resp)
	if msg.VoiceActivityDetectionSignal != nil {
		resp.CustomMetadata["vad_signal_type"] = string(msg.VoiceActivityDetectionSignal.VADSignalType)
	}
	if msg.UsageMetadata != nil {
		mapUsageMetadata(msg.UsageMetadata, resp)
	}

	return resp
}

func mapServerContent(sc *genai.LiveServerContent, resp *model.LLMResponse) {
	resp.TurnComplete = sc.TurnComplete
	resp.Interrupted = sc.Interrupted

	if sc.ModelTurn != nil {
		resp.Content = sc.ModelTurn
		for _, part := range sc.ModelTurn.Parts {
			if part.InlineData != nil && strings.HasPrefix(part.InlineData.MIMEType, "audio/") {
				resp.CustomMetadata["is_audio"] = true
				break
			}
		}
	}

	mapTranscriptions(sc, resp)

	if sc.GroundingMetadata != nil {
		resp.GroundingMetadata = sc.GroundingMetadata
	}
	if sc.TurnComplete && sc.TurnCompleteReason != "" {
		resp.CustomMetadata["turn_complete_reason"] = string(sc.TurnCompleteReason)
	}
	if sc.GenerationComplete {
		resp.CustomMetadata["generation_complete"] = true
	}
	if sc.WaitingForInput {
		resp.CustomMetadata["waiting_for_input"] = true
	}
}

func mapTranscriptions(sc *genai.LiveServerContent, resp *model.LLMResponse) {
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
}

func mapToolCall(tc *genai.LiveServerToolCall, resp *model.LLMResponse) {
	parts := make([]*genai.Part, 0, len(tc.FunctionCalls))
	for _, fc := range tc.FunctionCalls {
		parts = append(parts, &genai.Part{FunctionCall: fc})
	}
	resp.Content = &genai.Content{Role: "model", Parts: parts}
}

func mapGoAway(msg *genai.LiveServerMessage, resp *model.LLMResponse) {
	if msg.GoAway == nil {
		return
	}
	resp.GoAway = msg.GoAway
	resp.CustomMetadata["go_away"] = true
	if msg.GoAway.TimeLeft > 0 {
		resp.CustomMetadata["go_away_time_left_ms"] = float64(msg.GoAway.TimeLeft.Milliseconds())
	}
}

func mapSessionResumptionUpdate(msg *genai.LiveServerMessage, resp *model.LLMResponse) {
	if msg.SessionResumptionUpdate == nil {
		return
	}
	resp.SessionResumptionUpdate = msg.SessionResumptionUpdate
	resp.CustomMetadata["session_resumption"] = true
	if msg.SessionResumptionUpdate.NewHandle != "" {
		resp.CustomMetadata["session_resumption_handle"] = msg.SessionResumptionUpdate.NewHandle
	}
	resp.CustomMetadata["session_resumption_resumable"] = msg.SessionResumptionUpdate.Resumable
	if msg.SessionResumptionUpdate.LastConsumedClientMessageIndex > 0 {
		resp.CustomMetadata["last_consumed_client_message_index"] = float64(msg.SessionResumptionUpdate.LastConsumedClientMessageIndex)
	}
}

func mapUsageMetadata(um *genai.UsageMetadata, resp *model.LLMResponse) {
	resp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     um.PromptTokenCount,
		CandidatesTokenCount: um.ResponseTokenCount,
		TotalTokenCount:      um.TotalTokenCount,
	}
	if um.CachedContentTokenCount > 0 {
		resp.CustomMetadata["usage_cached_token_count"] = float64(um.CachedContentTokenCount)
	}
	if um.ToolUsePromptTokenCount > 0 {
		resp.CustomMetadata["usage_tool_use_prompt_tokens"] = float64(um.ToolUsePromptTokenCount)
	}
	if um.ThoughtsTokenCount > 0 {
		resp.CustomMetadata["usage_thoughts_token_count"] = float64(um.ThoughtsTokenCount)
	}
	mapTokenDetails(um, resp)
}

func mapTokenDetails(um *genai.UsageMetadata, resp *model.LLMResponse) {
	flattenTokenDetails := func(details []*genai.ModalityTokenCount) []any {
		out := make([]any, 0, len(details))
		for _, d := range details {
			out = append(out, map[string]any{
				"modality":    string(d.Modality),
				"token_count": float64(d.TokenCount),
			})
		}
		return out
	}
	if len(um.PromptTokensDetails) > 0 {
		resp.CustomMetadata["usage_prompt_tokens_details"] = flattenTokenDetails(um.PromptTokensDetails)
	}
	if len(um.CacheTokensDetails) > 0 {
		resp.CustomMetadata["usage_cache_tokens_details"] = flattenTokenDetails(um.CacheTokensDetails)
	}
	if len(um.ResponseTokensDetails) > 0 {
		resp.CustomMetadata["usage_response_tokens_details"] = flattenTokenDetails(um.ResponseTokensDetails)
	}
	if len(um.ToolUsePromptTokensDetails) > 0 {
		resp.CustomMetadata["usage_tool_use_prompt_tokens_details"] = flattenTokenDetails(um.ToolUsePromptTokensDetails)
	}
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
