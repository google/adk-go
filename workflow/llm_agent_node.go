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
	"errors"
	"fmt"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
)

// LlmAgentNode wraps an agent.Agent so a workflow can run it as a
// graph vertex. The node's input is rendered into the agent's
// UserContent; the agent runs as a sub-invocation; emitted events
// forward through the workflow's emitter; the final text response is
// captured as the node's Output.
//
// Mirrors adk-python's _llm_agent_wrapper.
type LlmAgentNode struct {
	Base
	agent agent.Agent
}

// FromAgent wraps a (typically Gemini-backed) agent.Agent as a workflow
// Node. Use this to compose multi-agent pipelines on the workflow graph
// — each LlmAgent becomes a vertex; their outputs feed downstream nodes
// like any other.
//
//	classifier := llmagent.New(llmagent.Config{Name: "classifier", ...})
//	writer     := llmagent.New(llmagent.Config{Name: "writer", ...})
//	wf, _ := workflow.New(workflow.Config{
//	    Edges: []workflow.Edge{
//	        workflow.Connect(workflow.START, workflow.FromAgent(classifier)),
//	        workflow.Connect(workflow.FromAgent(classifier), workflow.FromAgent(writer)),
//	    },
//	})
//
// The node name defaults to the agent's name. Override via WithDescription
// and the other NodeOpt helpers as needed.
func FromAgent(a agent.Agent, opts ...NodeOpt) *LlmAgentNode {
	if a == nil {
		panic("workflow: FromAgent requires a non-nil agent")
	}
	o := applyOpts(opts)
	n := &LlmAgentNode{agent: a}
	desc := o.description
	if desc == "" {
		desc = a.Description()
	}
	if err := n.SetMetadata(a.Name(), desc, o.toSpec()); err != nil {
		panic(err)
	}
	return n
}

// RunImpl invokes the wrapped agent as a sub-call. The input is
// rendered as the user content; events forward to the parent emitter;
// the agent's last text response becomes the node's Output.
func (n *LlmAgentNode) RunImpl(ctx *NodeContext, input any, em EventEmitter) error {
	uc := renderUserContent(input)
	child := icontext.NewInvocationContext(ctx.InvocationContext, icontext.InvocationContextParams{
		Artifacts:    ctx.InvocationContext.Artifacts(),
		Memory:       ctx.InvocationContext.Memory(),
		Session:      ctx.InvocationContext.Session(),
		Agent:        n.agent,
		UserContent:  uc,
		RunConfig:    ctx.InvocationContext.RunConfig(),
		InvocationID: ctx.InvocationContext.InvocationID(),
		Branch:       ctx.InvocationContext.Branch(),
	})

	var lastText string
	for ev, err := range n.agent.Run(child) {
		if err != nil {
			return fmt.Errorf("agent_node %q: %w", n.Name(), err)
		}
		if ev != nil {
			if t := extractText(ev.LLMResponse.Content); t != "" && !ev.LLMResponse.Partial {
				lastText = t
			}
			if err := em.Event(ev); err != nil {
				return err
			}
		}
	}
	if lastText == "" {
		return em.Output("") // ensure downstream sees a deterministic value
	}
	return em.Output(lastText)
}

// renderUserContent turns whatever the upstream node produced into a
// *genai.Content. Strings, *genai.Content, and JoinNode-style maps are
// recognized; other shapes fall back to fmt.Sprintf.
func renderUserContent(input any) *genai.Content {
	if input == nil {
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: ""}}}
	}
	switch v := input.(type) {
	case *genai.Content:
		return v
	case string:
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: v}}}
	case map[string]any:
		// Output of JoinNode: render each predecessor's contribution as a
		// labeled paragraph so the next agent has clean context.
		var b strings.Builder
		for k, val := range v {
			fmt.Fprintf(&b, "## %s\n%v\n\n", k, val)
		}
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: b.String()}}}
	default:
		return &genai.Content{
			Role:  genai.RoleUser,
			Parts: []*genai.Part{{Text: fmt.Sprintf("%v", input)}},
		}
	}
}

func extractText(c *genai.Content) string {
	if c == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range c.Parts {
		if p != nil && p.Text != "" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

var _ = errors.New
