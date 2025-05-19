package adk

import (
	"context"
	"iter"

	"github.com/google/adk-go/agent"
	"github.com/google/adk-go/event"
	"google.golang.org/genai"
)

type Runner interface {
	AppName() string
	Run(ctx context.Context, userID, sessionID string, newMessage *genai.Content, runConfig *agent.RunConfig) iter.Seq2[*event.Event, error]
	// TODO(jbd): Add RunLive.
}
