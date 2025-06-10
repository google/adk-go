package agent

import (
	"context"

	"github.com/google/adk-go"
	"google.golang.org/genai"
)

// LLMAgent is an LLM-based Agent.
type LLMAgent struct {
	AgentName        string
	AgentDescription string

	Model adk.Model

	Instruction           string
	GlobalInstruction     string
	Tools                 []adk.Tool
	GenerateContentConfig *genai.GenerateContentConfig

	// LLM-based agent transfer configs.
	DisallowTransferToParent bool
	DisallowTransferToPeers  bool

	// BeforeModelCallback
	// AfterModelCallback
	// BeforeToolCallback
	// AfterToolCallback
}

func (a *LLMAgent) Name() string        { return a.AgentName }
func (a *LLMAgent) Description() string { return a.AgentDescription }
func (a *LLMAgent) Run(ctx context.Context, parentCtx *adk.InvocationContext) (adk.EventStream, error) {
	// TODO: Select model (LlmAgent.canonical_model)
	// TODO: Singleflow, Autoflow Run.
	panic("unimplemented")
}

var _ adk.Agent = (*LLMAgent)(nil)

// TODO: Do we want to abstract "Flow" too?
