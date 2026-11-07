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

package utils

import (
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/session"
)

// FunctionCallEvent finds the event with the function call that the first
// function response in msg answers. It returns nil when msg has no function
// response or no event has the call.
//
// This assumes that even if the user provides multiple function responses, all
// the function calls were made by the same agent. Otherwise it would be
// impossible to rearrange session events such that every function response has
// a corresponding call filtering by author.
func FunctionCallEvent(events session.Events, msg *genai.Content) *session.Event {
	functionResponses := FunctionResponses(msg)
	if len(functionResponses) == 0 {
		return nil
	}
	callID := functionResponses[0].ID
	for i := events.Len() - 1; i >= 0; i-- {
		event := events.At(i)
		for _, part := range FunctionCalls(event.Content) {
			if part.ID == callID {
				return event
			}
		}
	}
	return nil
}

// GraphResumeIndex returns the index, among the sub-agents of ctx.Agent(), of
// the one holding the graph workflow that made the call ctx.UserContent()
// answers, or -1 if there is none. A sequential, parallel or loop agent starts
// from that sub-agent: the ones ahead of or beside it never saw the call, and
// an LLM agent among them fails on the unmatched response.
func GraphResumeIndex(ctx agent.InvocationContext) int {
	if len(FunctionResponses(ctx.UserContent())) == 0 {
		return -1
	}
	ev := FunctionCallEvent(ctx.Session().Events(), ctx.UserContent())
	if ev == nil {
		return -1
	}
	for i, sub := range ctx.Agent().SubAgents() {
		if calledFromGraph(sub, ev.Author, false) {
			return i
		}
	}
	return -1
}

// calledFromGraph reports whether the agent named author is in the tree of a,
// at or below a graph workflow built by workflowagent.New.
func calledFromGraph(a agent.Agent, author string, inGraph bool) bool {
	if ia, ok := a.(agentinternal.Agent); ok && agentinternal.Reveal(ia).AgentType == agentinternal.TypeWorkflowAgent {
		inGraph = true
	}
	if a.Name() == author {
		return inGraph
	}
	for _, sub := range a.SubAgents() {
		if calledFromGraph(sub, author, inGraph) {
			return true
		}
	}
	return false
}
