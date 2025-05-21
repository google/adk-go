// Package agent provides ADK agents.
package agent

import (
	"context"
	"iter"

	"github.com/google/adk-go/event"
	"github.com/google/uuid"
	"google.golang.org/genai"
)

// Agent is implemented by ADK agents.
type Agent interface {
	Name() string
	Description() string
	ParentAgent() Agent
	RootAgent() Agent
	SubAgents() []Agent
	Run(ctx context.Context, parentCtx *InvocationContext) iter.Seq2[*event.Event, error]
	RunLive(ctx context.Context, parentCtxs iter.Seq[*InvocationContext]) iter.Seq2[*event.Event, error]
}

// InvocationContext is the invocation context.
type InvocationContext struct {
	// TODO(jbd): ArtifactService artifact.Service
	// TODO(jbd): SessionService session.Service
	// TODO(jbd): Session        *session.Session
	InvocationID  string
	Branch        string
	Agent         Agent
	EndInvocation bool
	UserContent   *genai.Content
	// TODO(jbd): TranscriptionCache
	RunConfig *RunConfig
}

type StreamingMode string

const (
	StreamingModeNone StreamingMode = "none"
	StreamingModeSSE  StreamingMode = "sse"
	StreamingModeBidi StreamingMode = "bidi"
)

// RunConfig represents the runtime related configuration.
type RunConfig struct {
	SpeechConfig                   *genai.SpeechConfig
	OutputAudioTranscriptionConfig *genai.AudioTranscriptionConfig
	ResponseModalities             []string
	StreamingMode                  StreamingMode
	SaveInputBlobsAsArtifacts      bool
	SupportCFC                     bool
	MaxLLMCalls                    int
}

// NewInvocationID creates a new flow invocation ID.
func NewInvocationID() string {
	return uuid.NewString()
}
