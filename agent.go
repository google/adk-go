package adk

import (
	"context"

	"github.com/google/uuid"
	"google.golang.org/genai"
)

// Agent is the agent type.
type Agent interface {
	Name() string
	Description() string

	// Run runs the agent with the invocation context.
	Run(ctx context.Context, parentCtx *InvocationContext) (EventStream, error)
	// TODO: finalize the interface.
}

// InvocationContext is the agent invocation context.
type InvocationContext struct {
	InvocationID  string
	Branch        string
	Agent         Agent
	EndInvocation bool
	UserContent   *genai.Content
	RunConfig     *AgentRunConfig

	SessionService SessionService
	Session        *Session

	// TODO(jbd): ArtifactService
	// TODO(jbd): TranscriptionCache
}

// Cancel cancels the invocation.
func (ic *InvocationContext) Cancel(error) {
	// TODO(hakim): this implements adk-python InvocationContext.end_invocation.
	panic("unimplemented")
}

type StreamingMode string

const (
	StreamingModeNone StreamingMode = "none"
	StreamingModeSSE  StreamingMode = "sse"
	StreamingModeBidi StreamingMode = "bidi"
)

// AgentRunConfig represents the runtime related configuration.
type AgentRunConfig struct {
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
