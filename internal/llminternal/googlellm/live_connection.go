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
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// LiveConnection wraps the underlying GenAI SDK live session.
type LiveConnection struct {
	// Using the correct Session type from the GenAI SDK.
	sdkSession *genai.Session

	inputTranscriptionText  string
	outputTranscriptionText string
	bufferedResponses       []*model.LLMResponse
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
			if part.InlineData != nil && strings.HasPrefix(part.InlineData.MIMEType, "audio/") {
				continue
			}
			filteredParts = append(filteredParts, part)
			fmt.Printf("filtered part: %v\n", part.Text)
		}
		if len(filteredParts) > 0 {
			filteredHistory = append(filteredHistory, &genai.Content{
				Parts: filteredParts,
				Role:  content.Role,
			})
		}
	}
	fmt.Printf("sending history: of size %d\n", len(filteredHistory))
	turnComplete := false
	if len(filteredHistory) > 0 {
		err := c.sdkSession.SendClientContent(genai.LiveClientContentInput{
			Turns:        filteredHistory,
			TurnComplete: &turnComplete,
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
		if len(content.Parts) == 1 && content.Parts[0].Text != "" {
			fmt.Printf("Attempting to send text via SendRealtimeInput\n")
			err := c.sdkSession.SendRealtimeInput(genai.LiveRealtimeInput{
				Text: content.Parts[0].Text,
			})
			if err != nil {
				return fmt.Errorf("failed to send realtime text: %w", err)
			}
			return nil
		}

		turnComplete := true
		err := c.sdkSession.SendClientContent(genai.LiveClientContentInput{
			Turns:        []*genai.Content{content},
			TurnComplete: &turnComplete,
		})
		if err != nil {
			return fmt.Errorf("failed to send content: %w", err)
		}
	}

	fmt.Printf("sending message\n")

	return nil
}

// SendRealtime sends real-time input (audio/video).
func (c *LiveConnection) SendRealtime(ctx context.Context, input any) error {
	switch v := input.(type) {
	case *genai.Blob:
		if v.MIMEType == "" {
			// Detect PNG by signature: \x89PNG\r\n\x1a\n
			isPNG := len(v.Data) >= 8 &&
				v.Data[0] == 0x89 && v.Data[1] == 0x50 && v.Data[2] == 0x4E && v.Data[3] == 0x47 &&
				v.Data[4] == 0x0D && v.Data[5] == 0x0A && v.Data[6] == 0x1A && v.Data[7] == 0x0A

			if isPNG {
				v.MIMEType = "image/png"
			} else {
				v.MIMEType = "audio/pcm"
			}
		}

		if strings.HasPrefix(v.MIMEType, "image/") {
			fmt.Printf("sending image (%s)\n", v.MIMEType)
			return c.sdkSession.SendRealtimeInput(genai.LiveRealtimeInput{
				Video: v,
			})
		}

		return c.sdkSession.SendRealtimeInput(genai.LiveRealtimeInput{
			Audio: v,
		})
	case *genai.ActivityStart:
		fmt.Printf("sending activity start\n")
		return c.sdkSession.SendRealtimeInput(genai.LiveRealtimeInput{
			ActivityStart: v,
		})
	case *genai.ActivityEnd:
		fmt.Printf("sending activity end\n")
		return c.sdkSession.SendRealtimeInput(genai.LiveRealtimeInput{
			ActivityEnd: v,
		})
	default:
		return fmt.Errorf("unsupported real-time input type: %T", input)
	}
}

// Recv receives a response from the live server connection.
func (c *LiveConnection) Recv(ctx context.Context) (*model.LLMResponse, error) {
	if len(c.bufferedResponses) > 0 {
		resp := c.bufferedResponses[0]
		c.bufferedResponses = c.bufferedResponses[1:]
		return resp, nil
	}

	msg, err := c.sdkSession.Receive()
	if err != nil {
		return nil, fmt.Errorf("failed to receive message: %w", err)
	}

	if msg == nil {
		return nil, nil
	}

	if msg.ServerContent != nil && msg.ServerContent.OutputTranscription != nil {
		fmt.Printf("recieved message: %s\n", msg.ServerContent.OutputTranscription.Text)
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
			resp.InputTranscription = content.InputTranscription
			c.inputTranscriptionText += content.InputTranscription.Text
			resp.Partial = true // Mark chunks as partial so they are not saved to session
		}
		if content.OutputTranscription != nil {
			resp.OutputTranscription = content.OutputTranscription
			c.outputTranscriptionText += content.OutputTranscription.Text
			resp.Partial = true // Mark chunks as partial so they are not saved to session
		}

		// Handle transcription finalization on completion signals
		if content.TurnComplete || content.Interrupted {
			if c.inputTranscriptionText != "" || c.outputTranscriptionText != "" {
				if c.inputTranscriptionText != "" {
					inputResp := &model.LLMResponse{
						Partial: false,
						InputTranscription: &genai.Transcription{
							Text:     c.inputTranscriptionText,
							Finished: true,
						},
					}
					c.inputTranscriptionText = ""
					c.bufferedResponses = append(c.bufferedResponses, inputResp)
				}
				if c.outputTranscriptionText != "" {
					outputResp := &model.LLMResponse{
						Partial: false,
						OutputTranscription: &genai.Transcription{
							Text:     c.outputTranscriptionText,
							Finished: true,
						},
					}
					c.outputTranscriptionText = ""
					c.bufferedResponses = append(c.bufferedResponses, outputResp)
				}

				// Append the current response (which has TurnComplete or Interrupted) to the buffer
				// so it is delivered AFTER the transcriptions
				c.bufferedResponses = append(c.bufferedResponses, resp)

				// Return the first one from buffer
				first := c.bufferedResponses[0]
				c.bufferedResponses = c.bufferedResponses[1:]
				return first, nil
			}
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
