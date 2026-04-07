// Copyright 2026 Google LLC
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

package googlellm

import (
	"context"
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// LiveConnection wraps the underlying GenAI SDK live session.
type LiveConnection struct {
	// Using the correct Session type from the GenAI SDK.
	sdkSession *genai.Session

	inputTranscriptionText  string
	outputTranscriptionText string
}

// NewLiveConnection creates a new LiveConnection.
func NewLiveConnection(session *genai.Session) *LiveConnection {
	return &LiveConnection{sdkSession: session}
}

// SendHistory sends the conversation history to prime the session.
func (c *LiveConnection) SendHistory(ctx context.Context, history []*genai.Content) error {
	var filteredHistory []*genai.Content
	for _, content := range history {
		if content == nil {
			continue
		}
		var filteredParts []*genai.Part
		for _, part := range content.Parts {
			if part == nil {
				continue
			}
			// In Python, we filter audio parts because Live API doesn't support them well via SendContent.
			// Let's assume content.Parts contains standard text or function calls.
			filteredParts = append(filteredParts, part)
		}
		if len(filteredParts) > 0 {
			filteredHistory = append(filteredHistory, &genai.Content{
				Parts: filteredParts,
				Role:  content.Role,
			})
		}
	}

	if len(filteredHistory) > 0 {
		err := c.sdkSession.SendClientContent(genai.LiveClientContentInput{
			Turns: filteredHistory,
		})
		if err != nil {
			return fmt.Errorf("failed to send history: %w", err)
		}
	}

	return nil
}

// SendContent sends unary content or function responses to the model.
func (c *LiveConnection) SendContent(ctx context.Context, content *genai.Content) error {
	if content == nil || len(content.Parts) == 0 {
		return fmt.Errorf("empty content")
	}

	if content.Parts[0].FunctionResponse != nil {
		var functionResponses []*genai.FunctionResponse
		for _, part := range content.Parts {
			if part.FunctionResponse != nil {
				functionResponses = append(functionResponses, part.FunctionResponse)
			}
		}
		err := c.sdkSession.SendToolResponse(genai.LiveToolResponseInput{
			FunctionResponses: functionResponses,
		})
		if err != nil {
			return fmt.Errorf("failed to send tool response: %w", err)
		}
	} else {
		err := c.sdkSession.SendClientContent(genai.LiveClientContentInput{
			Turns: []*genai.Content{content},
		})
		if err != nil {
			return fmt.Errorf("failed to send content: %w", err)
		}
	}

	return nil
}

// SendRealtime sends real-time input (audio/video).
func (c *LiveConnection) SendRealtime(ctx context.Context, input any) error {
	switch v := input.(type) {
	case *genai.Blob:
		v.MIMEType = "audio/pcm"
		return c.sdkSession.SendRealtimeInput(genai.LiveRealtimeInput{
			Audio: v,
		})
	case *genai.ActivityStart:
		return c.sdkSession.SendRealtimeInput(genai.LiveRealtimeInput{
			ActivityStart: v,
		})
	case *genai.ActivityEnd:
		return c.sdkSession.SendRealtimeInput(genai.LiveRealtimeInput{
			ActivityEnd: v,
		})
	default:
		return fmt.Errorf("unsupported real-time input type: %T", input)
	}
}

// Recv receives a response from the live server connection.
func (c *LiveConnection) Recv(ctx context.Context) (*model.LLMResponse, error) {
	msg, err := c.sdkSession.Receive()
	if err != nil {
		return nil, fmt.Errorf("failed to receive message: %w", err)
	}

	if msg == nil {
		return nil, nil
	}

	resp := &model.LLMResponse{}

	if msg.ServerContent != nil {
		content := msg.ServerContent
		if content.ModelTurn != nil {
			resp.Content = content.ModelTurn
		}
		resp.TurnComplete = content.TurnComplete
		resp.Interrupted = content.Interrupted

		if content.InputTranscription != nil {
			if resp.Content == nil {
				resp.Content = &genai.Content{Role: "user"}
			}
			resp.Content.Parts = append(resp.Content.Parts, &genai.Part{
				Text: content.InputTranscription.Text,
			})
			c.inputTranscriptionText += content.InputTranscription.Text
		}
		if content.OutputTranscription != nil {
			if resp.Content == nil {
				resp.Content = &genai.Content{Role: "model"}
			}
			c.outputTranscriptionText += content.OutputTranscription.Text
			resp.Content.Parts = append(resp.Content.Parts, &genai.Part{
				Text: content.OutputTranscription.Text,
			})
		}
	}

	if msg.ToolCall != nil {
		if resp.Content == nil {
			resp.Content = &genai.Content{Role: "model"}
		}
		for _, call := range msg.ToolCall.FunctionCalls {
			if call != nil {
				resp.Content.Parts = append(resp.Content.Parts, &genai.Part{
					FunctionCall: call,
				})
			}
		}
	}

	return resp, nil
}

// Close closes the live server connection.
func (c *LiveConnection) Close() error {
	if c.sdkSession != nil {
		return c.sdkSession.Close()
	}
	return nil
}
