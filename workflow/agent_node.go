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
	"fmt"
	"iter"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	internalcontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/typeutil"
	"google.golang.org/adk/session"
)

// AgentNode wraps a standard agent.Agent.
type AgentNode struct {
	BaseNode
	agent        agent.Agent
	inputSchema  *jsonschema.Resolved
	outputSchema *jsonschema.Resolved
}

// newAgentNodeWithSchemasTyped creates a new node wrapping an agent with explicitly provided schemas.
// If a schema is nil, it will be inferred from the corresponding generic type Input or Output.
func newAgentNodeWithSchemasTyped[Input, Output any](a agent.Agent, inputSchema, outputSchema *jsonschema.Schema, cfg NodeConfig) (Node, error) {
	if a == nil {
		return nil, fmt.Errorf("agent cannot be nil")
	}
	ischema, err := resolvedSchema[Input](inputSchema)
	if err != nil {
		return nil, fmt.Errorf("resolving input schema for agent %q: %w", a.Name(), err)
	}
	if ischema == nil {
		return nil, fmt.Errorf("resolved input schema for agent %q is nil", a.Name())
	}
	oschema, err := resolvedSchema[Output](outputSchema)
	if err != nil {
		return nil, fmt.Errorf("resolving output schema for agent %q: %w", a.Name(), err)
	}
	if oschema == nil {
		return nil, fmt.Errorf("resolved output schema for agent %q is nil", a.Name())
	}

	return &AgentNode{
		BaseNode:     NewBaseNode(a.Name(), a.Description(), cfg),
		agent:        a,
		inputSchema:  ischema,
		outputSchema: oschema,
	}, nil
}

// NewAgentNodeWithSchemas is a convenience wrapper for NewAgentNodeWithSchemasTyped[any, any].
// It uses explicitly provided schemas for both input and output.
func NewAgentNodeWithSchemas(a agent.Agent, inputSchema, outputSchema *jsonschema.Schema, cfg NodeConfig) (Node, error) {
	return newAgentNodeWithSchemasTyped[any, any](a, inputSchema, outputSchema, cfg)
}

// NewAgentNodeTyped creates a new node wrapping an agent using generics to
// automatically infer input and output schemas from the provided types.
func NewAgentNodeTyped[Input, Output any](a agent.Agent, cfg NodeConfig) (Node, error) {
	return newAgentNodeWithSchemasTyped[Input, Output](a, nil, nil, cfg)
}

// NewAgentNode creates a new node wrapping an agent. Input and output schemas
// are inferred as `any`.
func NewAgentNode(a agent.Agent, cfg NodeConfig) (Node, error) {
	return NewAgentNodeTyped[any, any](a, cfg)
}

// Run implements the Node interface.
func (n *AgentNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		// Validate input if schema is available
		var err error
		if n.inputSchema != nil {
			input, err = typeutil.ConvertToWithJSONSchema[any, any](input, n.inputSchema)
			if err != nil {
				yield(nil, fmt.Errorf("converting input for agent %q: %w", n.agent.Name(), err))
				return
			}
		}

		var userContent *genai.Content
		if input != nil {
			switch v := input.(type) {
			case string:
				userContent = &genai.Content{
					Parts: []*genai.Part{{Text: v}},
				}
			case *genai.Content:
				userContent = v
			default:
				userContent = &genai.Content{
					Parts: []*genai.Part{{Text: fmt.Sprint(v)}},
				}
			}
		}

		// Use existing agent context instead of implementing a new one
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

			// Validate output if schema is available and event has output
			if n.outputSchema != nil && event.Actions.StateDelta != nil {
				if output, ok := event.Actions.StateDelta["output"]; ok {
					if err := n.outputSchema.Validate(output); err != nil {
						yield(nil, fmt.Errorf("converting agent %q output: validation failed: %w", n.agent.Name(), err))
						return
					}
				}
			}

			if !yield(event, nil) {
				return
			}
		}
	}
}
