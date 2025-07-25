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
	"iter"

	"github.com/google/uuid"
	"google.golang.org/genai"
)

// Agent is the agent type.
type Agent interface {
	// Spec is the Agent's spec.
	Spec() *AgentSpec

	// Run runs the agent with the invocation context.
	Run(ctx context.Context, invocationCtx *InvocationContext) iter.Seq2[*Event, error]

	// TODO: RunLive.
}

type AgentSpec struct {
	// Agent name.
	// It must be unique within the agent tree.
	// Agent name cannot be "user" since it is reserved for user input.
	Name string

	// Description about the agent's capability.
	// The model uses this to determine whether to delegate control to the agent.
	// One-line description is enough and preferred.
	Description string

	// The parent agent of this agent.
	// Note that an agent can ONLY be added as sub-agent once.
	// If you want to add one agent twice as sub-agent, consider to create two agent
	// instances with identical config, but with different name and add them to the
	// agent tree.
	Parent Agent

	// The sub-agents of this agent.
	SubAgents []Agent

	// Configuration for the agent backed by LLM.
	LLMAgent *LLMAgentSpec

	// TODO:
	//  BeforeAgentCallback
	//  AfterAgentCallback
}

type LLMAgentSpec struct {
	Model Model

	Instruction       string
	GlobalInstruction string

	Tools []Tool

	GenerateContentConfig *genai.GenerateContentConfig

	// LLM-based agent transfer configs.
	DisallowTransferToParent bool
	DisallowTransferToPeers  bool

	// Whether to include contents in the model request.
	// When set to 'none', the model request will not include any contents, such as
	// user messages, tool requests, etc.
	IncludeContents string

	// The input schema when agent is used as a tool.
	InputSchema *genai.Schema

	// The output schema when agent replies.
	//
	// NOTE: when this is set, agent can only reply and cannot use any tools,
	// such as function tools, RAGs, agent transfer, etc.
	OutputSchema *genai.Schema

	// OutputKey
	// Planner
	// CodeExecutor
	// Examples

	// BeforeModelCallback
	// AfterModelCallback
	// BeforeToolCallback
	// AfterToolCallback
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
//  1. Is handled by [Agent.Run] through [Agent.Spec]'s Run.
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
	Agent Agent
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
func NewInvocationContext(ctx context.Context, agent Agent, sessionService SessionService, session *Session) (context.Context, *InvocationContext) {
	ctx, cancel := context.WithCancelCause(ctx)
	return ctx, &InvocationContext{
		InvocationID:   "e-" + uuid.NewString(),
		Agent:          agent,
		cancel:         cancel,
		SessionService: sessionService,
		Session:        session,
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
