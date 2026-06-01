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
func (n *AgentNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		validatedInput, err := n.ValidateInput(input)
		if err != nil {
			yield(nil, err)
			return
		}

		var userContent *genai.Content
		if validatedInput != nil {
			switch v := validatedInput.(type) {
			case string:
				userContent = &genai.Content{
					Parts: []*genai.Part{{Text: v}},
				}
			case *genai.Content:
				userContent = v
			default:
				b, err := json.Marshal(v)
				if err != nil {
					yield(nil, fmt.Errorf("marshaling input for agent %q to JSON: %w", n.agent.Name(), err))
					return
				}
				userContent = &genai.Content{
					Parts: []*genai.Part{{Text: string(b)}},
				}
			}
		}

		// Use existing agent context instead of implementing a new one.
		// Branch is inherited from ctx so the agent runs under the
		// activation's branch; the scheduler assigns sub-branches at
		// fan-out, and the LLM flow's history filter scopes events
		// by branch prefix.
		params := internalcontext.InvocationContextParams{
			Artifacts:     ctx.Artifacts(),
			Memory:        ctx.Memory(),
			Session:       ctx.Session(),
			Branch:        ctx.Branch(),
			Agent:         n.agent,
			UserContent:   userContent,
			RunConfig:     ctx.RunConfig(),
			EndInvocation: ctx.Ended(),
			InvocationID:  ctx.InvocationID(),
		}
		agentCtx := internalcontext.NewInvocationContext(ctx, params)

		for event, err := range n.agent.Run(agentCtx) {
			if err != nil {
				yield(nil, err)
				return
			}

			synthesizeAgentOutput(event)

			// TODO: add output validation
			if !yield(event, nil) {
				return
			}
		}
	}
}

// synthesizeAgentOutput sets Event.Output from concatenated model
// text on final model responses so RunNode returns the agent's
// reply instead of the zero value.
func synthesizeAgentOutput(event *session.Event) {
	if event == nil || event.Output != nil {
		return
	}
	if !event.IsFinalResponse() {
		return
	}
	content := event.LLMResponse.Content
	if content == nil || content.Role != "model" {
		return
	}
	var b []byte
	for _, p := range content.Parts {
		if p == nil || p.Text == "" || p.Thought {
			continue
		}
		b = append(b, p.Text...)
	}
	if len(b) == 0 {
		return
	}
	event.Output = string(b)
}
