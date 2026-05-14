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
	"iter"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// agentNode wraps a standard agent.Agent so it can be used as a
// node in a workflow graph. The wrapped agent runs with the
// engine-supplied per-node InvocationContext; every event it
// yields is forwarded to the workflow's event stream unchanged.
//
// This is the minimum viable AgentNode: it does not yet add
// schema validation, prompt injection, or structured-output
// enforcement at the workflow boundary (the api_design_final_v2
// document describes those as follow-ups). The wrapped agent's
// own input handling — including its existing OutputSchema if
// any — is what governs what the agent actually does.
//
// The node inherits its name and description from the wrapped
// agent.
type agentNode struct {
	BaseNode
	agent agent.Agent
}

// NewAgentNode wraps an agent.Agent as a workflow Node. The
// returned Node forwards every event the agent yields to the
// workflow's event stream and treats the agent's run as the
// node's activation; successor scheduling follows the normal
// rules (see scheduler.handleCompletion).
//
// Use this for any agent.Agent — LLMAgent (the common case),
// SequentialAgent, custom agents built via agent.New, or other
// workflow-wrapped agents (workflowagent.New) for nesting.
//
// # Output forwarding to successor nodes
//
// The workflow scheduler reads each activation's "output" value
// from Event.Actions.StateDelta["output"] (the magic key
// scheduler.handleEvent recognises). To forward an LLM agent's
// reply to the next node in the graph, set its OutputKey to
// "output" when constructing the agent — the LLMAgent code path
// at agent/llmagent/llmagent.go:425-429 writes the final text
// to that StateDelta key automatically. Without OutputKey set,
// the successor will receive nil as its input.
//
// (Once the design's typed-output bridge lands, the engine will
// translate AgentNode's OUT type parameter into structured-
// output enforcement and a typed StateDelta entry. Until then,
// OutputKey="output" is the explicit knob.)
func NewAgentNode(a agent.Agent, cfg NodeConfig) Node {
	return &agentNode{
		BaseNode: NewBaseNode(a.Name(), a.Description(), cfg),
		agent:    a,
	}
}

// Run is the Node interface implementation: it delegates to the
// wrapped agent and forwards events. Honouring the caller's
// break (the !yield short-circuit) lets the scheduler cancel
// the node mid-flight without leaving the wrapped agent's
// goroutine spinning.
func (n *agentNode) Run(ctx agent.InvocationContext, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		for ev, err := range n.agent.Run(ctx) {
			if !yield(ev, err) {
				return
			}
		}
	}
}
