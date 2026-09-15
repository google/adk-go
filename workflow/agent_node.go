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
	"strings"

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

	// derivedOutputSchema is the model-facing form of oschema, computed once
	// here because oschema is fixed for the node's lifetime. It is bound into
	// the invocation context on each Run (see llminternal.WithBoundOutputSchema)
	// rather than written onto the wrapped agent's State: the same agent.Agent
	// can back more than one node, so State is shared and a construction-time
	// write there either races (TestNewAgentNode_ConcurrentWrappingIsRaceFree
	// wraps one agent from 8 goroutines) or, without a race, lets whichever
	// node was built last win for every placement of that agent.
	derivedOutputSchema *genai.Schema
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

	// oschema only reached BaseNode.ValidateOutput before this fix, so a typed
	// node validated the model's reply against Output's shape on the way out
	// without ever asking the model to produce that shape on the way in. Derive
	// the model-facing form here; Run binds it per invocation (see
	// llminternal.WithBoundOutputSchema) rather than writing it onto the
	// agent's shared State, and an explicit llmagent.Config.OutputSchema is
	// left untouched (llminternal.OutputSchemaFor prefers the declaration).
	derivedOutputSchema, err := genaiSchemaFromResolved(oschema)
	if err != nil {
		return nil, fmt.Errorf("deriving model output schema for agent %q: %w", a.Name(), err)
	}

	return &AgentNode{
		BaseNode:            NewBaseNodeWithSchemas(a.Name(), a.Description(), cfg, ischema, oschema),
		agent:               a,
		derivedOutputSchema: derivedOutputSchema,
	}, nil
}

// genaiSchemaFromResolved converts a resolved JSON Schema to the *genai.Schema
// the model API expects, or nil for no real constraint (Output `any` marshals
// to the bare `true`). Mirrors openaimodel.schemaToMap in reverse.
func genaiSchemaFromResolved(resolved *jsonschema.Resolved) (*genai.Schema, error) {
	if resolved == nil {
		return nil, nil
	}
	raw, err := json.Marshal(resolved.Schema())
	if err != nil {
		return nil, fmt.Errorf("marshaling resolved output schema: %w", err)
	}
	if string(raw) == "true" || string(raw) == "false" {
		return nil, nil
	}
	var generic any
	if err := json.Unmarshal(raw, &generic); err != nil {
		return nil, fmt.Errorf("decoding resolved output schema: %w", err)
	}
	uppercaseSchemaTypes(generic)
	raw, err = json.Marshal(generic)
	if err != nil {
		return nil, fmt.Errorf("re-encoding resolved output schema: %w", err)
	}
	var modelSchema genai.Schema
	if err := json.Unmarshal(raw, &modelSchema); err != nil {
		return nil, fmt.Errorf("decoding output schema as genai.Schema: %w", err)
	}
	return &modelSchema, nil
}

// uppercaseSchemaTypes rewrites every "type" keyword value to the
// upper-case spelling genai.Schema.Type expects (STRING, OBJECT, ARRAY,
// ...); jsonschema-go emits the standard lower-case JSON Schema spelling.
func uppercaseSchemaTypes(val any) {
	switch v := val.(type) {
	case map[string]any:
		if t, ok := v["type"]; ok {
			switch tVal := t.(type) {
			case string:
				v["type"] = strings.ToUpper(tVal)
			case []any:
				applyNullableTypeArray(v, tVal)
			}
		}
		for _, child := range v {
			uppercaseSchemaTypes(child)
		}
	case []any:
		for _, child := range v {
			uppercaseSchemaTypes(child)
		}
	}
}

// applyNullableTypeArray collapses a JSON Schema type array (jsonschema-go's
// ["null", X] for a Go pointer or omitempty field) into genai.Schema's
// single-value Type plus Nullable, uppercasing what remains.
func applyNullableTypeArray(v map[string]any, types []any) {
	var rest []string
	for _, t := range types {
		s, ok := t.(string)
		if !ok {
			continue
		}
		if s == "null" {
			v["nullable"] = true
			continue
		}
		rest = append(rest, strings.ToUpper(s))
	}
	if len(rest) == 1 {
		v["type"] = rest[0]
	} else {
		delete(v, "type")
	}
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
			bound = llminternal.WithBoundOutputSchema(bound, n.agent.Name(), state, n.derivedOutputSchema)
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
