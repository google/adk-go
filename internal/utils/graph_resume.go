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
	"context"
	"slices"

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
	return findCallEvent(events, functionResponses[0].ID)
}

func findCallEvent(events session.Events, callID string) *session.Event {
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

type graphResumeDoneKey struct{}

// WithGraphResumeDone returns ctx with resuming marked as done, so
// GraphResumeIndexes finds nothing for it or any agent run under it. A loop
// agent runs its later passes with this context: the user content stays the
// same for the whole turn, but only the first pass narrows. This only stops the
// narrowing; a workflow on a later pass still sees the same message.
func WithGraphResumeDone(ctx agent.InvocationContext) agent.InvocationContext {
	return ctx.WithContext(context.WithValue(ctx, graphResumeDoneKey{}, true))
}

// GraphResumeIndexes returns, in ascending order, the indexes among the
// sub-agents of ctx.Agent() of the ones holding a caller of the function
// responses in ctx.UserContent(). It returns nil, meaning all sub-agents run,
// when none of the callers is in a graph workflow under ctx.Agent(), or when
// ctx is marked by WithGraphResumeDone.
//
// A parallel agent runs the selected sub-agents; a sequential or loop agent
// starts from the lowest index. The ones skipped never saw the calls, and an
// LLM agent among them fails on the unmatched responses.
//
// A node agent is found through SubAgents, so a workflow only matches if its
// node agents are listed in workflowagent.Config.SubAgents. Otherwise all
// sub-agents run.
func GraphResumeIndexes(ctx agent.InvocationContext) []int {
	if ctx.Value(graphResumeDoneKey{}) != nil {
		return nil
	}
	selected := map[int]bool{}
	inGraph := false
	for _, response := range FunctionResponses(ctx.UserContent()) {
		ev := findCallEvent(ctx.Session().Events(), response.ID)
		if ev == nil {
			continue
		}
		for i, sub := range ctx.Agent().SubAgents() {
			if sub.FindAgent(ev.Author) == nil {
				continue
			}
			selected[i] = true
			inGraph = inGraph || calledFromGraph(sub, ev.Author, false)
		}
	}
	if !inGraph {
		return nil
	}
	indexes := make([]int, 0, len(selected))
	for i := range selected {
		indexes = append(indexes, i)
	}
	slices.Sort(indexes)
	return indexes
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

// ResumedInputs returns the answers ctx was resumed with, keyed by interrupt
// ID: every function response the user sent in ctx's invocation that
// ctx.ResumedInput knows. It returns nil when there are none.
//
// A composite agent hands its sub-agents a new context, which would otherwise
// lose the answers a workflow node was resumed with, and a sub-agent that asks
// with workflow.ResumeOrRequestInput would ask again.
func ResumedInputs(ctx agent.Context) map[string]any {
	var resumed map[string]any
	for ev := range ctx.Session().Events().All() {
		if ev.Author != "user" || ev.InvocationID != ctx.InvocationID() {
			continue
		}
		for _, response := range FunctionResponses(ev.Content) {
			if v, ok := ctx.ResumedInput(response.ID); ok {
				if resumed == nil {
					resumed = map[string]any{}
				}
				resumed[response.ID] = v
			}
		}
	}
	return resumed
}
