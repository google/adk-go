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
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"slices"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	internalcontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/session"
)

// AgentNode wraps a standard agent.Agent. Wrapped agents should emit their final output via
// Event.Output to be propagated to successor nodes
type AgentNode struct {
	BaseNode
	agent agent.Agent
	// modelOutputSchema is the output schema in the form the model is asked
	// to answer with, or nil. Run binds it per invocation; see
	// llminternal.WithBoundOutputSchema for why it is not set on the agent.
	modelOutputSchema *genai.Schema
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

	// The wrapped agent's Run already emits an invoke_agent span, so
	// the scheduler must not add a redundant invoke_node wrapper —
	// whether this node is activated by a static edge or delegated to
	// via RunNode. Mirrors runner.newAgentNode.
	cfg.EmitsOwnSpan = true

	return &AgentNode{
		BaseNode:          NewBaseNodeWithSchemas(a.Name(), a.Description(), cfg, ischema, oschema),
		agent:             a,
		modelOutputSchema: modelOutputSchema(oschema),
	}, nil
}

// NewAgentNodeWithSchemas is a convenience wrapper for NewAgentNodeWithSchemasTyped[any, any].
// It uses explicitly provided schemas for both input and output.
//
// The output schema is passed to the model as described for [NewAgentNodeTyped].
func NewAgentNodeWithSchemas(a agent.Agent, inputSchema, outputSchema *jsonschema.Schema, cfg NodeConfig) (*AgentNode, error) {
	return newAgentNodeWithSchemasTyped[any, any](a, inputSchema, outputSchema, cfg)
}

// NewAgentNodeTyped creates a new node wrapping an agent using generics to
// automatically infer input and output schemas from the provided types.
//
// The output schema is passed to the model when it is a closed object with
// properties (a struct, not a pointer to one; a hand-written schema must set
// additionalProperties to false on every object) and the wrapped agent is an
// LLM agent that sets no response format of its own (OutputSchema, or
// ResponseSchema, ResponseJsonSchema or ResponseMIMEType in
// GenerateContentConfig), runs single_turn (the default at a node; chat and
// task agents talk to the user between replies), and cannot transfer to other
// agents.
//
// It goes out as the request's response schema or, on Gemini models that
// cannot combine tools with one, as a set_model_response tool the model is
// told to call. That tool is a function declaration, so, as with a declared
// OutputSchema, an agent with a Gemini built-in tool such as Google Search
// fails on models that cannot mix built-in tools with function declarations
// (on Gemini 1.x, before the model is called); declare an untyped node for it.
//
// Value constraints such as bounds and patterns are not sent, a pointer or
// slice field is sent as not nullable, and nothing is sent for a schema the
// model API cannot express, such as one with an any or map field or a
// non-string enum. Either way the node validates the reply against the full
// output schema, and the agent's OutputKey, if set, still receives the reply
// text rather than a parsed value.
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
		// Resolving the mode reads ctx, so a nil one is rejected before that
		// rather than dereferenced further down. Exported, and RunLLMAgentAsNode
		// — the other exported entry point that resolves a mode — rejects it the
		// same way. Placed after nodeInputToContent, which never touches ctx, so
		// an unmarshalable input still reports the marshaling error the merge
		// base reported for it. Untyped nil only: a typed-nil agent.Context
		// still panics, as it does everywhere else in these packages.
		if ctx == nil {
			yield(nil, fmt.Errorf("AgentNode.Run: nil context for agent %q", n.agent.Name()))
			return
		}

		// A graph node is a one-shot placement: an agent that declares no
		// mode runs single_turn here. Bound under this agent's own key — its
		// name and its identity — so it cannot govern a peer this agent later
		// transfers to, nor a same-named agent nested beneath it.
		//
		// The binding goes into the context this function passes DOWN, not back
		// into ctx. Routing it through ctx.WithAgentContext would be the obvious
		// shape and is a crash: that method returns nil for a tool context and
		// for a callback context, both of which log and carry on, and the nil is
		// dereferenced a few lines below. Neither wrapper reaches this function:
		// SingleTurnTool does drive a node from inside a tool, but workflow.RunNode
		// uses the caller's context only to fetch its SubScheduler, and the child
		// context comes from the scheduler's own parent. Every symbol on that path
		// is exported, so an out-of-tree caller can still arrive here with either.
		// Guarding the nil and keeping the old ctx is not the fix either — the
		// binding only reaches the request processors through the context handed
		// downstream, so dropping it would run an undeclared agent as chat at a
		// single_turn node.
		bound := context.Context(ctx)
		if llmA, ok := n.agent.(llminternal.Agent); ok && llmA != nil {
			state := llminternal.Reveal(llmA)
			mode := llminternal.ResolveMode(state.Mode, llminternal.ModeSingleTurn)
			bound = llminternal.WithBoundMode(ctx, n.agent.Name(), state, mode)
			// Offer the node's output shape to the model. Each request decides
			// whether to use it, against the mode bound for it and the agent's
			// own setup. A nil schema is still bound, so this placement clears
			// one an outer placement bound.
			bound = llminternal.WithBoundOutputSchema(bound, n.agent.Name(), state, n.modelOutputSchema)
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
		agentCtx := internalcontext.NewInvocationContext(bound, params)
		exCtx := agent.NewContext(agentCtx)

		type NodeRunner interface {
			RunNode(ctx agent.Context, nodeInput any) iter.Seq2[*session.Event, error]
		}

		var events iter.Seq2[*session.Event, error]
		if runner, ok := n.agent.(NodeRunner); ok {
			events = runner.RunNode(exCtx, input)
		} else {
			events = n.agent.Run(exCtx)
		}

		// Task-mode LlmAgents set their output via runTask, not model text.
		synthesizeMode := true
		if llmA, ok := n.agent.(llminternal.Agent); ok && llmA != nil {
			synthesizeMode = llminternal.Reveal(llmA).Mode != llminternal.ModeTask
		}

		for event, err := range events {
			if err != nil {
				yield(nil, err)
				return
			}

			// A composite agent yields one final response per sub-agent
			// (each a distinct author); synthesizing all of them would
			// trip the one-output-per-node rule.
			if synthesizeMode && isOwnAgentEvent(event, n.agent.Name()) {
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
// the chat wrapper synthesize a bogus completion FR for what is
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

// isOwnAgentEvent reports whether ev came from the node's own agent.
// Un-authored events count as own (agent.Run stamps the node agent's
// name); composite sub-agents keep their own author.
func isOwnAgentEvent(ev *session.Event, nodeAgentName string) bool {
	if ev == nil {
		return false
	}
	return ev.Author == "" || ev.Author == nodeAgentName
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

// outputCountsFor reports whether event's output can count for the node at
// path. An event whose OutputFor is set was already attributed by the
// scheduler that ran its emitter and counts only for the paths listed there:
// a task agent that transfers to a workflow forwards the events of the nodes
// that workflow runs, and a nested task agent's finish_task result arrives
// with OutputFor naming the nested node, not the task agent. An event with no
// OutputFor has not been attributed yet and counts for the node emitting it.
func outputCountsFor(event *session.Event, path string) bool {
	if event.NodeInfo == nil || len(event.NodeInfo.OutputFor) == 0 {
		return true
	}
	return slices.Contains(event.NodeInfo.OutputFor, path)
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
