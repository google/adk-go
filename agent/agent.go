package agent

import (
	"context"
	"iter"

	"github.com/google/adk-go/event"
	"github.com/google/uuid"
	"google.golang.org/genai"
)

type Agent interface {
	Name() string
	Description() string
	ParentAgent() Agent
	RootAgent() Agent
	SubAgents() []Agent
	Run(ctx context.Context, parentCtx *InvocationContext) iter.Seq2[*event.Event, error]
	RunLive(ctx context.Context, parentCtxs iter.Seq[*InvocationContext]) iter.Seq2[*event.Event, error]
}
type InvocationContext struct {
	// ArtifactService artifact.Service
	// SessionService session.Service
	// Session        *session.Session
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

type RunConfig struct {
	SpeechConfig                   *genai.SpeechConfig
	OutputAudioTranscriptionConfig *genai.AudioTranscriptionConfig
	ResponseModalities             []string
	StreamingMode                  StreamingMode
	SaveInputBlobsAsArtifacts      bool
	SupportCFC                     bool
	MaxLLMCalls                    int
}

func NewInvocationID() string {
	return uuid.NewString()
}
