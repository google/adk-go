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
package gemini

import (
	"context"
	"fmt"

	"github.com/rs/zerolog/log"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// Connect establishes a bidirectional streaming connection to the model.
func (m *geminiModel) Connect(ctx context.Context, req *model.LLMRequest) (model.LiveConnection, error) {
	session, err := m.client.Live.Connect(ctx, m.name, req.LiveConnectConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to live model: %w", err)
	}

	return &liveConnection{
		session: session,
	}, nil
}

type liveConnection struct {
	session *genai.Session

	inputTranscriptionText  string
	outputTranscriptionText string
}

func (c *liveConnection) Send(req *model.LiveRequest) error {
	if req.Close {
		return c.Close()
	}
	if req.ActivityStart != nil {
		return c.session.SendRealtimeInput(genai.LiveSendRealtimeInputParameters{
			ActivityStart: req.ActivityStart,
		})
	}
	if req.ActivityEnd != nil {
		return c.session.SendRealtimeInput(genai.LiveSendRealtimeInputParameters{
			ActivityEnd: req.ActivityEnd,
		})
	}
	if req.Content != nil {
		content := req.Content
		if content.Parts != nil && content.Parts[0].FunctionResponse != nil {
			var functionResponses []*genai.FunctionResponse
			for _, part := range content.Parts {
				if part.FunctionResponse != nil {
					functionResponses = append(functionResponses, part.FunctionResponse)
				}
			}
			return c.session.SendToolResponse(genai.LiveToolResponseInput{
				FunctionResponses: functionResponses,
			})
		} else {
			log.Info().Interface("content", req.Content).Msg("Sending content to live connection")
			turnComplete := true
			return c.session.SendClientContent(genai.LiveClientContentInput{
				Turns:        []*genai.Content{req.Content},
				TurnComplete: &turnComplete,
			})
		}
	}
	if req.RealtimeInput != nil {
		return c.session.SendRealtimeInput(*req.RealtimeInput)
	}
	if req.ToolResponse != nil {
		return c.session.SendToolResponse(*req.ToolResponse)
	}
	return nil
}

func (c *liveConnection) receive(ctx context.Context) (<-chan *genai.LiveServerMessage, <-chan error) {
	out := make(chan *genai.LiveServerMessage, 100)
	errChan := make(chan error, 1)
	go func() {
		defer close(out)
		defer close(errChan)
		for {
			msg, err := c.session.Receive()
			if err != nil {
				// We don't use the helper for errChan since it's buffered(1) and we return immediately
				select {
				case errChan <- err:
				case <-ctx.Done():
				}
				return
			}
			select {
			case out <- msg:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out, errChan
}

func (c *liveConnection) process(ctx context.Context, in <-chan *genai.LiveServerMessage) <-chan *model.LLMResponse {
	out := make(chan *model.LLMResponse, 100)
	// errChan := make(chan error, 1)

	go func() {
		defer close(out)
		// defer close(errChan)

		send := func(resp *model.LLMResponse) bool {
			select {
			case out <- resp:
				return true
			case <-ctx.Done():
				return false
			}
		}

		var text string
		for {
			select {
			case msg, ok := <-in:
				if !ok {
					return
				}

				log.Info().Interface("msg", msg).Msg("Received message in gemini_live")

				if msg.UsageMetadata != nil {
					if !send(&model.LLMResponse{
						UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
							CacheTokensDetails:         msg.UsageMetadata.CacheTokensDetails,
							CachedContentTokenCount:    msg.UsageMetadata.CachedContentTokenCount,
							PromptTokenCount:           msg.UsageMetadata.PromptTokenCount,
							PromptTokensDetails:        msg.UsageMetadata.PromptTokensDetails,
							ThoughtsTokenCount:         msg.UsageMetadata.ThoughtsTokenCount,
							ToolUsePromptTokenCount:    msg.UsageMetadata.ToolUsePromptTokenCount,
							ToolUsePromptTokensDetails: msg.UsageMetadata.ToolUsePromptTokensDetails,
							TotalTokenCount:            msg.UsageMetadata.TotalTokenCount,
							TrafficType:                msg.UsageMetadata.TrafficType,
						},
					}) {
						return
					}
				}

				if msg.ServerContent != nil {
					content := msg.ServerContent.ModelTurn
					if content != nil && len(content.Parts) > 0 {
						resp := &model.LLMResponse{
							Content:     content,
							Interrupted: msg.ServerContent.Interrupted,
						}
						if content.Parts[0].Text != "" {
							text += content.Parts[0].Text
							resp.Partial = true
						} else if text != "" && content.Parts[0].InlineData == nil {
							if !send(c.buildFullTextResponse(text)) {
								return
							}
							text = ""
						}
						if !send(resp) {
							return
						}
					}

					// Note: in some cases, tool_call may arrive before
					// generation_complete, causing transcription to appear after
					// tool_call in the session log.
					if msg.ServerContent.InputTranscription != nil {
						if msg.ServerContent.InputTranscription.Text != "" {
							c.inputTranscriptionText += msg.ServerContent.InputTranscription.Text
							if !send(&model.LLMResponse{
								InputTranscription: &genai.Transcription{
									Text:     msg.ServerContent.InputTranscription.Text,
									Finished: false,
								},
								Partial: true,
							}) {
								return
							}
						}

						// finished=True and partial transcription may happen in the same
						// message.
						if msg.ServerContent.InputTranscription.Finished {
							if !send(&model.LLMResponse{
								InputTranscription: &genai.Transcription{
									Text:     c.inputTranscriptionText,
									Finished: true,
								},
								Partial: false,
							}) {
								return
							}
							c.inputTranscriptionText = ""
						}
					}
					if msg.ServerContent.OutputTranscription != nil {
						if msg.ServerContent.OutputTranscription.Text != "" {
							c.outputTranscriptionText += msg.ServerContent.OutputTranscription.Text
							if !send(&model.LLMResponse{
								OutputTranscription: &genai.Transcription{
									Text:     msg.ServerContent.OutputTranscription.Text,
									Finished: false,
								},
								Partial: true,
							}) {
								return
							}
						}

						// finished=True and partial transcription may happen in the same
						// message.
						if msg.ServerContent.OutputTranscription.Finished {
							if !send(&model.LLMResponse{
								OutputTranscription: &genai.Transcription{
									Text:     c.outputTranscriptionText,
									Finished: true,
								},
								Partial: false,
							}) {
								return
							}
							c.outputTranscriptionText = ""
						}
					}

					// The Gemini API might not send a transcription finished signal.
					// Instead, we rely on generation_complete, turn_complete or
					// interrupted signals to flush any pending transcriptions.
					if msg.ServerContent.Interrupted ||
						msg.ServerContent.TurnComplete ||
						msg.ServerContent.GenerationComplete {

						if c.inputTranscriptionText != "" {
							if !send(&model.LLMResponse{
								InputTranscription: &genai.Transcription{
									Text:     c.inputTranscriptionText,
									Finished: true,
								},
								Partial: false,
							}) {
								return
							}
							c.inputTranscriptionText = ""
						}

						if c.outputTranscriptionText != "" {
							if !send(&model.LLMResponse{
								OutputTranscription: &genai.Transcription{
									Text:     c.outputTranscriptionText,
									Finished: true,
								},
								Partial: false,
							}) {
								return
							}
							c.outputTranscriptionText = ""
						}

					}
					if msg.ServerContent.TurnComplete {
						if text != "" {
							if !send(c.buildFullTextResponse(text)) {
								return
							}
							text = ""
						}
						if !send(&model.LLMResponse{
							TurnComplete: true,
							Interrupted:  msg.ServerContent.Interrupted,
						}) {
							return
						}
						continue
					}
					// in case of empty content or parts, we sill surface it
					// in case it's an interrupted message, we merge the previous partial
					// text. Other we don't merge. because content can be none when model
					// safety threshold is triggered
					if msg.ServerContent.Interrupted {
						if text != "" {
							if !send(c.buildFullTextResponse(text)) {
								return
							}
							text = ""
						} else {
							if !send(&model.LLMResponse{
								Interrupted: msg.ServerContent.Interrupted,
							}) {
								return
							}
						}
					}
				}

				if msg.ToolCall != nil {
					resp := &model.LLMResponse{}
					// Map ToolCall to model.LLMResponse content parts
					parts := make([]*genai.Part, 0)
					for _, fc := range msg.ToolCall.FunctionCalls {
						parts = append(parts, &genai.Part{
							FunctionCall: fc,
						})
					}
					if resp.Content == nil {
						resp.Content = &genai.Content{Role: "model"}
					}
					resp.Content.Parts = append(resp.Content.Parts, parts...)
					if !send(resp) {
						return
					}
				}

				if msg.SessionResumptionUpdate != nil {
					log.Debug().Interface("session_resumption_update", msg.SessionResumptionUpdate).Msg("Received session resumption update")
					if !send(&model.LLMResponse{
						LiveSessionResumptionUpdate: msg.SessionResumptionUpdate,
					}) {
						return
					}
				}
			case <-ctx.Done():
				// errChan <- ctx.Err()
				return
			}
		}
	}()
	return out
}

func (c *liveConnection) buildFullTextResponse(text string) *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				genai.NewPartFromText(text),
			},
		},
	}
}

func (c *liveConnection) Receive(ctx context.Context) (<-chan *model.LLMResponse, <-chan error) {
	msgs, errs1 := c.receive(ctx)
	resps := c.process(ctx, msgs)

	errChan := make(chan error, 1)
	go func() {
		defer close(errChan)
		for {
			select {
			case err, ok := <-errs1:
				if ok && err != nil {
					select {
					case errChan <- err:
					case <-ctx.Done():
					}
					return
				}
				if !ok {
					errs1 = nil
				}
			case <-ctx.Done():
				return
			}

			if errs1 == nil {
				return
			}
		}
	}()

	return resps, errChan
}

func (c *liveConnection) Close() error {
	return c.session.Close()
}
