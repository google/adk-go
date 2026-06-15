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

package workflow

import (
	"encoding/json"
	"fmt"
	"iter"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	internalcontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/session"
)

// AgentNode wraps a standard agent.Agent. Wrapped agents should emit their final output via
// Event.Output to be propagated to successor nodes
type AgentNode struct {
	BaseNode
	agent agent.Agent
}

// newAgentNodeWithSchemasTyped creates a new node wrapping an agent with explicitly provided schemas.
// If a schema is nil, it will be inferred from the corresponding generic type Input or Output.
func newAgentNodeWithSchemasTyped[Input, Output any](a agent.Agent, inputSchema, outputSchema *jsonschema.Schema, cfg NodeConfig) (*AgentNode, error) {
	if a == nil {
		return nil, fmt.Errorf("agent cannot be nil")
	}
	ischema, err := resolvedSchema[Input](inputSchema)
	if err != nil {
		return nil, fmt.Errorf("resolving input schema for agent %q: %w", a.Name(), err)
	}
	oschema, err := resolvedSchema[Output](outputSchema)
	if err != nil {
		return nil, fmt.Errorf("resolving output schema for agent %q: %w", a.Name(), err)
	}

	return &AgentNode{
		BaseNode: NewBaseNodeWithSchemas(a.Name(), a.Description(), cfg, ischema, oschema),
		agent:    a,
	}, nil
}

// NewAgentNodeWithSchemas is a convenience wrapper for NewAgentNodeWithSchemasTyped[any, any].
// It uses explicitly provided schemas for both input and output.
func NewAgentNodeWithSchemas(a agent.Agent, inputSchema, outputSchema *jsonschema.Schema, cfg NodeConfig) (*AgentNode, error) {
	return newAgentNodeWithSchemasTyped[any, any](a, inputSchema, outputSchema, cfg)
}

// NewAgentNodeTyped creates a new node wrapping an agent using generics to
// automatically infer input and output schemas from the provided types.
func NewAgentNodeTyped[Input, Output any](a agent.Agent, cfg NodeConfig) (*AgentNode, error) {
	return newAgentNodeWithSchemasTyped[Input, Output](a, nil, nil, cfg)
}

// NewAgentNode creates a new node wrapping an agent. Input and output schemas
// are inferred as `any`.
func NewAgentNode(a agent.Agent, cfg NodeConfig) (*AgentNode, error) {
	return NewAgentNodeTyped[any, any](a, cfg)
}

// Run implements the Node interface.
func (n *AgentNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		userContent, err := nodeInputToContent(input)
		if err != nil {
			yield(nil, err)
			return
		}

		// Use existing agent context instead of implementing a new one.
		// Branch is inherited from ctx so the agent runs under the
		// activation's branch; the scheduler assigns sub-branches at
		// fan-out, and the LLM flow's history filter scopes events
		// by branch prefix.
		params := internalcontext.InvocationContextParams{
			Artifacts:      ctx.Artifacts(),
			Memory:         ctx.Memory(),
			Session:        ctx.Session(),
			Branch:         ctx.Branch(),
			IsolationScope: ctx.IsolationScope(),
			Agent:          n.agent,
			UserContent:    userContent,
			RunConfig:      ctx.RunConfig(),
			EndInvocation:  ctx.Ended(),
			InvocationID:   ctx.InvocationID(),
		}
		agentCtx := internalcontext.NewInvocationContext(ctx, params)
		exCtx := agent.NewNodeContext(agentCtx, nil)

		type NodeRunner interface {
			RunNode(ctx agent.Context, nodeInput any) iter.Seq2[*session.Event, error]
		}

		var events iter.Seq2[*session.Event, error]
		if runner, ok := n.agent.(NodeRunner); ok {
			events = runner.RunNode(exCtx, input)
		} else {
			events = n.agent.Run(exCtx)
		}

		// Task-mode LlmAgents set their output exclusively via
		// finish_task (handled by llmagent.runTask, which stamps
		// ev.Output from the FC's args on the successful FR). Plain
		// model text emitted by a task agent is user-facing chatter
		// — typically a question to the user mid-conversation — and
		// must NOT be promoted to the node output. Promoting it
		// would make the chat coordinator's dispatchTaskFC see a
		// non-nil output, synthesise a delegation-closing FR, and
		// route the next user reply to the coordinator instead of
		// back into the still-open task scope. See
		// llm_agent_wrapper.go:dispatchAndYield's out==nil branch.
		skipSynthesize := false
		if llmA, ok := n.agent.(llminternal.Agent); ok && llmA != nil {
			if llminternal.Reveal(llmA).Mode == llminternal.ModeTask {
				skipSynthesize = true
			}
		}

		for event, err := range events {
			if err != nil {
				yield(nil, err)
				return
			}

			if !skipSynthesize {
				synthesizeAgentOutput(event)
			}

			// Tag the event for scope filtering (mirrors adk-python
			// NodeRunner._enrich_event). The scheduler stamps delegated
			// child events; this covers the direct agent-wrapper path.
			if sc := ctx.IsolationScope(); sc != "" && event.IsolationScope == "" {
				event.IsolationScope = sc
			}

			// The output schema (if any) is applied by the scheduler via
			// ValidateOutput; synthesizeAgentOutput leaves the raw model
			// text for defaultValidateOutput to project onto the schema.
			if !yield(event, nil) {
				return
			}
		}
	}
}

// synthesizeAgentOutput sets Event.Output from concatenated model
// text on final model responses so RunNode returns the agent's
// reply instead of the zero value. Empty model text yields an empty
// "" output (a value, not "no output"), matching adk-python and
// messageAsOutput; non-model events are left untouched.
//
// It also stamps NodeInfo.MessageAsOutput so readers (live and
// resume) know this event's output was derived from the model
// message, mirroring adk-python's process_llm_agent_output which
// sets event.output and node_info.message_as_output together.
//
// Long-running-tool events (e.g. a tool that called
// ctx.RequestConfirmation and is awaiting the user's reply) are
// excluded: IsFinalResponse() returns true for them so the flow
// loop terminates the round, but they represent a pause, not a
// completion. Treating them as MessageAsOutput would cache an
// empty "" as the agent's "output" and, on resume, short-circuit
// the re-run via collectNodeOutputs / WithRunID-replay — making
// the chat wrapper synthesise a bogus completion FR for what is
// still an open delegation.
func synthesizeAgentOutput(event *session.Event) {
	if event == nil || event.Output != nil {
		return
	}
	if !event.IsFinalResponse() {
		return
	}
	if len(event.LongRunningToolIDs) > 0 {
		return
	}
	if text, ok := messageText(event); ok {
		event.Output = text
		if event.NodeInfo == nil {
			event.NodeInfo = &session.NodeInfo{}
		}
		event.NodeInfo.MessageAsOutput = true
	}
}

// messageText concatenates the non-thought model text of an event. ok
// is false when the event carries no model content, distinguishing it
// from a model message with empty text.
func messageText(event *session.Event) (text string, ok bool) {
	if event == nil {
		return "", false
	}
	content := event.LLMResponse.Content
	if content == nil || content.Role != "model" {
		return "", false
	}
	var b []byte
	for _, p := range content.Parts {
		if p == nil || p.Text == "" || p.Thought {
			continue
		}
		b = append(b, p.Text...)
	}
	return string(b), true
}

// childEventOutput returns the output an event carries: its Output, or
// the model text when MessageAsOutput is set.
func childEventOutput(event *session.Event) (any, bool) {
	if event.Output != nil {
		return event.Output, true
	}
	if event.NodeInfo != nil && event.NodeInfo.MessageAsOutput {
		if text, ok := messageText(event); ok {
			return text, true
		}
	}
	return nil, false
}

func nodeInputToContent(input any) (*genai.Content, error) {
	switch v := input.(type) {
	case nil:
		return nil, nil
	case *genai.Content:
		if v == nil {
			return nil, nil
		}
		return &genai.Content{Role: "user", Parts: v.Parts}, nil
	case string:
		return &genai.Content{Role: "user", Parts: []*genai.Part{{Text: v}}}, nil
	case json.Marshaler:
		b, err := v.MarshalJSON()
		if err != nil {
			return nil, fmt.Errorf("marshaling input: %w", err)
		}
		return &genai.Content{Role: "user", Parts: []*genai.Part{{Text: string(b)}}}, nil
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("marshaling input to JSON: %w", err)
		}
		return &genai.Content{Role: "user", Parts: []*genai.Part{{Text: string(b)}}}, nil
	}
}
