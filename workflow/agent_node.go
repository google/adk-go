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

	cfg = applyAgentNodeDefaults(a, cfg)

	return &AgentNode{
		BaseNode: NewBaseNodeWithSchemas(a.Name(), a.Description(), cfg, ischema, oschema),
		agent:    a,
	}, nil
}

// applyAgentNodeDefaults fills in the AgentNode config defaults, mirroring
// applyDynamicDefaults. EmitsOwnSpan is always set: the agent's Run already
// emits invoke_agent, so the scheduler must not add a redundant invoke_node
// wrapper. LlmAgent nodes additionally default to single_turn mode and to
// re-entry on resume (so a paused agent finishes instead of handing off),
// matching runner.newAgentNode and the mode/rerun_on_resume defaults of
// adk-python's build_node; other kinds keep the engine default and an explicit
// RerunOnResume always wins.
//
// The mode default is a write to the wrapped agent's own state, not to cfg.
func applyAgentNodeDefaults(a agent.Agent, cfg NodeConfig) NodeConfig {
	cfg.EmitsOwnSpan = true

	llmA, ok := a.(llminternal.Agent)
	if !ok {
		return cfg
	}
	if state := llminternal.Reveal(llmA); state.Mode == llminternal.ModeUnset {
		state.Mode = llminternal.ModeSingleTurn
	}
	if cfg.RerunOnResume == nil {
		rerun := true
		cfg.RerunOnResume = &rerun
	}
	return cfg
}

// NewAgentNodeWithSchemas is a convenience wrapper for NewAgentNodeWithSchemasTyped[any, any].
// It uses explicitly provided schemas for both input and output, and applies
// the same LlmAgent defaults as [NewAgentNode].
func NewAgentNodeWithSchemas(a agent.Agent, inputSchema, outputSchema *jsonschema.Schema, cfg NodeConfig) (*AgentNode, error) {
	return newAgentNodeWithSchemasTyped[any, any](a, inputSchema, outputSchema, cfg)
}

// NewAgentNodeTyped creates a new node wrapping an agent using generics to
// automatically infer input and output schemas from the provided types.
// It applies the same LlmAgent defaults as [NewAgentNode].
func NewAgentNodeTyped[Input, Output any](a agent.Agent, cfg NodeConfig) (*AgentNode, error) {
	return newAgentNodeWithSchemasTyped[Input, Output](a, nil, nil, cfg)
}

// NewAgentNode creates a new node wrapping an agent. Input and output schemas
// are inferred as `any`.
//
// When a is an LlmAgent, its mode defaults to single_turn and
// cfg.RerunOnResume defaults to &true, so a node paused on a long-running tool
// request re-enters and finishes instead of handing the raw reply to its
// successor. An explicit RerunOnResume is respected, and other agent kinds
// keep the engine default (handoff). The mode default is applied to the agent
// itself, so wrapping one agent in several nodes settles its mode once.
func NewAgentNode(a agent.Agent, cfg NodeConfig) (*AgentNode, error) {
	return NewAgentNodeTyped[any, any](a, cfg)
}

// Run implements the Node interface.
func (n *AgentNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		// On resume, re-feeding the input would make a single_turn/task
		// LlmAgent re-call the still-pending tool and pause again; drop it
		// so the agent continues from history. Mirrors runner.runAgentNodeBody.
		if n.isResuming(ctx) {
			input = nil
		}
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

// isResuming reports whether this activation resumes a pause of THIS node, in
// which case Run drops the node input so the agent continues from history.
//
// Both signals are needed. The scheduler's per-activation resume flag alone
// over-fires: a dynamic child inherits the resume context of the ancestor that
// paused, so a freshly delegated child would lose its input. A history scan
// alone latches: history never un-answers an interrupt, so a later loop-back,
// retry or per-item activation of the same node would lose its input too.
//
// Interrupts are attributed by the activation's node PATH ("orch@1/child@2"),
// which the scheduler stamps on every event leaving a node. The path, not the
// node name, is the identity that matters: an orchestrator delegating the same
// child twice produces child@1 and child@2, and only the delegation that
// paused may drop its input. A pause raised by a sub-agent the node's agent
// delegated to carries the node's own path, so it still counts as this node's
// — matching the engine, which parks and re-enters the node on exactly those
// interrupts.
func (n *AgentNode) isResuming(ctx agent.Context) bool {
	ra, ok := ctx.(interface{ IsResumeActivation() bool })
	if !ok || !ra.IsResumeActivation() {
		return false
	}
	return raisedInterrupt(ctx.Session(), ctx.InvocationID(), ctx.Path(), n.Name())
}

// raisedInterrupt reports whether the activation at nodePath raised a
// long-running interrupt in invocationID. Runs only on a resume activation,
// keeping the scan off the normal path.
//
// nodePath is empty for a root-wrapper activation, where the scheduler stamps
// the bare node name instead. An event carrying no path at all is attributed by
// Author, the same fallback rehydration's eventNodeName uses.
func raisedInterrupt(sess session.Session, invocationID, nodePath, nodeName string) bool {
	if sess == nil || nodeName == "" {
		return false
	}
	events := sess.Events()
	if events == nil {
		return false
	}
	want := nodePath
	if want == "" {
		want = nodeName
	}
	for i := 0; i < events.Len(); i++ {
		ev := events.At(i)
		if ev == nil || len(ev.LongRunningToolIDs) == 0 {
			continue
		}
		if invocationID != "" && ev.InvocationID != invocationID {
			continue
		}
		if ev.NodeInfo != nil && ev.NodeInfo.Path != "" {
			if samePathActivation(ev.NodeInfo.Path, want) {
				return true
			}
			continue
		}
		if ev.Author == nodeName {
			return true
		}
	}
	return false
}

// samePathActivation reports whether an event's node path names the same
// activation as nodePath. It compares trailing segments rather than the whole
// string: a workflow agent records its events under its own prefix
// ("wf@1/worker@1") while a later Resume rebuilds the node path without it
// ("worker@1"), so exact equality would miss a node's own pause across the
// turn boundary.
func samePathActivation(evPath, nodePath string) bool {
	if evPath == nodePath {
		return true
	}
	if evPath == "" || nodePath == "" {
		return false
	}
	return strings.HasSuffix(evPath, "/"+nodePath) || strings.HasSuffix(nodePath, "/"+evPath)
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
