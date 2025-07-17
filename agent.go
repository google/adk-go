// Copyright 2025 Google LLC
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

package adk

import (
	"context"
	"fmt"
	"iter"

	"github.com/google/adk-go/internal/itype"
	"github.com/google/uuid"
	"google.golang.org/genai"
)

// Agent is the Agent type ADK framework interacts with.
// Use the constructor functions provided in the agent package.
// For example, the following creates a trivial LLM agent.
//
/*
	import "github.com/google/adk-go/agent"

	a, err := agent.NewLLMAgent(agent.LLMAgentConfig{
		Name: "my agent",
		Model: model,
		Instruction: "answer to user's question nicely",
	})
*/
// To develope a custom agent, see the documentation in
// github.com/google/adk-go/agent/base.
type Agent struct {
	name        string
	description string
	parentAgent *Agent
	subAgents   []*Agent

	impl AgentImpl
}

// AgentImpl implements custom agent's logic.
type AgentImpl interface {
	Run(context.Context, *InvocationContext) iter.Seq2[*Event, error]
	// TODO: RunLive
}

// An InvocationContext represents the data of a single invocation of an agent.
//
// An invocation:
//  1. Starts with a user message and ends with a final response.
//  2. Can contain one or multiple agent calls.
//  3. Is handled by runner.Run().
//
// An invocation runs an agent until it does not request to transfer to another
// agent.
//
// An agent call:
//  1. Is handled by [Agent.Run].
//  2. Ends when [Agent.Run] ends.
//
// An LLM agent call is an agent with a BaseLLMFlow.
// An LLM agent call can contain one or multiple steps.
//
// An LLM agent runs steps in a loop until:
//  1. A final response is generated.
//  2. The agent transfers to another agent.
//  3. The [InvocationContext.End] is called by any callbacks or tools.
//
// A step:
//  1. Calls the LLM only once and yields its response.
//  2. Calls the tools and yields their responses if requested.
//
// The summarization of the function response is considered another step, since
// it is another llm call.
// A step ends when it's done calling llm and tools, or if the end_invocation
// is set to true at any time.
//
//	┌─────────────────────── invocation ──────────────────────────┐
//	┌──────────── llm_agent_call_1 ────────────┐ ┌─ agent_call_2 ─┐
//	┌──── step_1 ────────┐ ┌───── step_2 ──────┐
//	[call_llm] [call_tool] [call_llm] [transfer]
type InvocationContext struct {
	// The id of this invocation context set by runner. Readonly.
	InvocationID string

	// The branch of the invocation context.
	// The format is like agent_1.agent_2.agent_3, where agent_1 is the parent of
	//  agent_2, and agent_2 is the parent of agent_3.
	// Branch is used when multiple sub-agents shouldn't see their peer agents'
	// conversation history.
	Branch string
	// The current agent of this invocation context. Readonly.
	Agent *Agent
	// The user content that started this invocation. Readonly.
	UserContent *genai.Content
	// Configurations for live agents under this invocation.
	RunConfig *AgentRunConfig

	// The current session of this invocation context. Readonly.
	Session *Session

	SessionService SessionService
	// TODO(jbd): ArtifactService
	// TODO(jbd): TranscriptionCache

	cancel context.CancelCauseFunc
}

// NewInvocationContext creates a new invocation context for the given agent
// and returns context.Context that is bound to the invocation context.
func NewInvocationContext(ctx context.Context, agent *Agent) (context.Context, *InvocationContext) {
	ctx, cancel := context.WithCancelCause(ctx)
	return ctx, &InvocationContext{
		InvocationID: "e-" + uuid.NewString(),
		Agent:        agent,
		cancel:       cancel,
	}
}

// End ends the invocation and cancels the context.Context bound to it.
func (ic *InvocationContext) End(err error) {
	ic.cancel(err)
}

type StreamingMode string

const (
	StreamingModeNone StreamingMode = "none"
	StreamingModeSSE  StreamingMode = "sse"
	StreamingModeBidi StreamingMode = "bidi"
)

// AgentRunConfig represents the runtime related configuration.
type AgentRunConfig struct {
	// Speech configuration for the live agent.
	SpeechConfig *genai.SpeechConfig
	// Output transcription for live agents with audio response.
	OutputAudioTranscriptionConfig *genai.AudioTranscriptionConfig
	// The output modalities. If not set, it's default to AUDIO.
	ResponseModalities []string
	// Streaming mode, None or StreamingMode.SSE or StreamingMode.BIDI.
	StreamingMode StreamingMode
	// Whether or not to save the input blobs as artifacts
	SaveInputBlobsAsArtifacts bool

	// Whether to support CFC (Compositional Function Calling). Only applicable for
	// StreamingModeSSE. If it's true. the LIVE API will be invoked since only LIVE
	// API supports CFC.
	//
	// .. warning::
	//      This feature is **experimental** and its API or behavior may change
	//     in future releases.
	SupportCFC bool

	// A limit on the total number of llm calls for a given run.
	//
	// Valid Values:
	//  - More than 0 and less than sys.maxsize: The bound on the number of llm
	//    calls is enforced, if the value is set in this range.
	//  - Less than or equal to 0: This allows for unbounded number of llm calls.
	MaxLLMCalls int
}

// Agent's constructor function is defined in a different package.
// That complicates initializing unexported fields.
// The following registers the initialization function that
// can be called from the agent/base package.
// Note: this package imports itype, which means itype cannot
// rely on the types defined in this package. RegisterConfigureAgent
// thus returns the constructed *Agent object as 'any'.

func init() {
	itype.RegisterNewAgent(func(cfg itype.AgentConfig) (any, error) { return newAgent(cfg) })
}

func newAgent(cfg itype.AgentConfig) (*Agent, error) {
	a := &Agent{name: cfg.Name, description: cfg.Description}
	subAgents := make([]*Agent, 0, len(cfg.SubAgents))
	for _, s := range cfg.SubAgents {
		subagent, ok := s.(*Agent)
		if !ok || subagent == nil {
			return a, fmt.Errorf("subagent %v is not *adk.Agent", s)
		}
		subAgents = append(subAgents, subagent)
	}
	return a, a.addSubAgents(subAgents...)
}

func (a *Agent) Name() string           { return a.name }
func (a *Agent) Description() string    { return a.description }
func (a *Agent) Parent() *Agent         { return a.parentAgent }
func (a *Agent) SubAgents() []*Agent    { return a.subAgents }
func (a *Agent) SetImpl(impl AgentImpl) { a.impl = impl }
func (a *Agent) Impl() AgentImpl        { return a.impl }
func (a *Agent) Run(ctx context.Context, parentCtx *InvocationContext) iter.Seq2[*Event, error] {
	ctx, parentCtx = newInvocationContext(ctx, a, parentCtx)
	// TODO: telemetry
	return a.impl.Run(ctx, parentCtx)
}

func newInvocationContext(ctx context.Context, a *Agent, p *InvocationContext) (context.Context, *InvocationContext) {
	ctx, c := NewInvocationContext(ctx, a)
	if p != nil {
		// copy everything but Agent and internal state.
		c.InvocationID = p.InvocationID
		c.Branch = p.Branch // TODO: why don't we update branch?
		c.UserContent = p.UserContent
		c.RunConfig = p.RunConfig
		c.Session = p.Session
	}
	return ctx, c
}

// addSubAgents adds the agents to the subagent list.
func (a *Agent) addSubAgents(agents ...*Agent) error {
	names := map[string]bool{}
	for _, subagent := range a.subAgents {
		names[subagent.Name()] = true
	}
	// run sanity check (no duplicate name, no multiple parents)
	for _, subagent := range agents {
		name := subagent.Name()
		if names[name] {
			return fmt.Errorf("multiple subagents with the same name (%q) are not allowed", name)
		}
		if parent := subagent.Parent(); parent != nil {
			return fmt.Errorf("agent %q already has parent %q", name, parent.Name())
		}
		names[name] = true
	}

	// mutate.
	for _, subagent := range agents {
		a.subAgents = append(a.subAgents, subagent)
		subagent.parentAgent = a
	}
	return nil
}
