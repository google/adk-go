package gemini

import (
	"context"
	"fmt"

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
	if req.Content != nil {
		// Wrap single content in slice as LiveClientContentInput expects turns
		turnComplete := true
		return c.session.SendClientContent(genai.LiveClientContentInput{
			Turns:        []*genai.Content{req.Content},
			TurnComplete: &turnComplete,
		})
	}
	if req.RealtimeInput != nil {
		return c.session.SendRealtimeInput(*req.RealtimeInput)
	}
	if req.ToolResponse != nil {
		return c.session.SendToolResponse(*req.ToolResponse)
	}
	return nil
}

func (c *liveConnection) Receive() (*model.LLMResponse, error) {
	msg, err := c.session.Receive()
	if err != nil {
		return nil, err
	}

	resp := &model.LLMResponse{}

	if msg.ServerContent != nil {
		// Map ServerContent to model.LLMResponse
		if msg.ServerContent.ModelTurn != nil {
			resp.Content = msg.ServerContent.ModelTurn
		}
		resp.TurnComplete = msg.ServerContent.TurnComplete
		resp.Interrupted = msg.ServerContent.Interrupted

		if msg.ServerContent.GroundingMetadata != nil {
			resp.GroundingMetadata = msg.ServerContent.GroundingMetadata
		}
	}

	if msg.UsageMetadata != nil {
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
	}

	if msg.ToolCall != nil {
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
	}

	if msg.ToolCallCancellation != nil {
		// TODO: Handle cancellation? Maybe just return empty response or specific error?
	}

	return resp, nil
}

func (c *liveConnection) Close() error {
	return c.session.Close()
}
