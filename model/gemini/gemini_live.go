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

func (c *liveConnection) Receive(ctx context.Context) (<-chan *model.LLMResponse, <-chan error) {
	out := make(chan *model.LLMResponse, 100)
	errChan := make(chan error)
	go func() {
		defer close(out)
		defer close(errChan)
		for {
			msg, err := c.session.Receive()
			if err != nil {
				errChan <- err
				return
			}

			if msg.UsageMetadata != nil {
				resp := &model.LLMResponse{}
				// we use generate content response usage metadata for now
				resp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
					CacheTokensDetails:         msg.UsageMetadata.CacheTokensDetails,
					CachedContentTokenCount:    msg.UsageMetadata.CachedContentTokenCount,
					PromptTokenCount:           msg.UsageMetadata.PromptTokenCount,
					PromptTokensDetails:        msg.UsageMetadata.PromptTokensDetails,
					ThoughtsTokenCount:         msg.UsageMetadata.ThoughtsTokenCount,
					ToolUsePromptTokenCount:    msg.UsageMetadata.ToolUsePromptTokenCount,
					ToolUsePromptTokensDetails: msg.UsageMetadata.ToolUsePromptTokensDetails,
					TotalTokenCount:            msg.UsageMetadata.TotalTokenCount,
					TrafficType:                msg.UsageMetadata.TrafficType,
				}
				out <- resp
			}

			if msg.ServerContent != nil {
				resp := &model.LLMResponse{}
				// Map ServerContent to model.LLMResponse
				if msg.ServerContent.ModelTurn != nil {
					resp.Content = msg.ServerContent.ModelTurn
				}
				resp.TurnComplete = msg.ServerContent.TurnComplete
				resp.Interrupted = msg.ServerContent.Interrupted

				if msg.ServerContent.GroundingMetadata != nil {
					resp.GroundingMetadata = msg.ServerContent.GroundingMetadata
				}
				out <- resp
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
				out <- resp
			}

			if msg.SessionResumptionUpdate != nil {
				log.Debug().Interface("session_resumption_update", msg.SessionResumptionUpdate).Msg("Received session resumption update")
				resp := &model.LLMResponse{
					LiveSessionResumptionUpdate: msg.SessionResumptionUpdate,
				}
				out <- resp
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

func (c *liveConnection) Close() error {
	return c.session.Close()
}
