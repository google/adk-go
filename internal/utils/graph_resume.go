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
	"strings"

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
	return FindFunctionCallEvent(events, functionResponses[0].ID)
}

// FindFunctionCallEvent returns the most recent event carrying a function call
// with callID, or nil when none exists.
func FindFunctionCallEvent(events session.Events, callID string) *session.Event {
	if events == nil || callID == "" {
		return nil
	}
	for i := events.Len() - 1; i >= 0; i-- {
		event := events.At(i)
		if event == nil {
			continue
		}
		for _, part := range FunctionCalls(event.Content) {
			if part.ID == callID {
				return event
			}
		}
	}
	return nil
}

// ShouldPauseInvocation reports whether event pauses the invocation to wait for
// a long-running tool response (such as adk_request_confirmation,
// adk_request_credential, or adk_request_input), matching adk-python's
// InvocationContext.should_pause_invocation.
func ShouldPauseInvocation(event *session.Event) bool {
	if event == nil || len(event.LongRunningToolIDs) == 0 {
		return false
	}
	for _, call := range FunctionCalls(event.Content) {
		if slices.Contains(event.LongRunningToolIDs, call.ID) {
			return true
		}
	}
	return false
}

// HasUnresolvedLongRunningCalls reports whether the current invocation still
// has an unanswered long-running function call emitted by an agent under
// ctx.Agent() on ctx.Branch() (or a sub-branch of it). For example, in
// Sequential{Parallel{a, b}, after}, answering only a leaves b's call
// unresolved so after waits until b is answered too.
func HasUnresolvedLongRunningCalls(ctx agent.InvocationContext) bool {
	if ctx.Session() == nil || ctx.InvocationID() == "" {
		return false
	}
	invID := ctx.InvocationID()
	branch := ctx.Branch()
	curAgent := ctx.Agent()
	events := ctx.Session().Events()

	pending := map[string]bool{}
	for i := range events.Len() {
		ev := events.At(i)
		if ev == nil || ev.Partial {
			continue
		}
		for _, resp := range FunctionResponses(ev.Content) {
			delete(pending, resp.ID)
		}
		if ev.InvocationID != invID || len(ev.LongRunningToolIDs) == 0 {
			continue
		}
		if !inBranchSubtree(branch, ev.Branch) || curAgent.FindAgent(ev.Author) == nil {
			continue
		}
		for _, call := range FunctionCalls(ev.Content) {
			if slices.Contains(ev.LongRunningToolIDs, call.ID) {
				pending[call.ID] = true
			}
		}
	}
	return len(pending) > 0
}

type (
	graphResumeDoneKey  struct{}
	workflowResumingKey struct{}
)

// WithWorkflowResuming marks ctx as resuming an enclosing sequential or loop
// workflow so a nested ParallelAgent narrows to the branches holding the
// answered callers even when those branches are leaf agents.
func WithWorkflowResuming(ctx agent.InvocationContext) agent.InvocationContext {
	return ctx.WithContext(context.WithValue(ctx, workflowResumingKey{}, true))
}

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
// when none of the callers is in a graph, sequential, or loop workflow under
// (or at) ctx.Agent(), or when ctx is marked by WithGraphResumeDone.
//
// A parallel agent runs the selected sub-agents; a sequential or loop agent
// starts from the lowest index. The ones skipped either already ran before the
// pause or never saw the calls.
//
// A graph workflow node agent is found through SubAgents, so a graph workflow
// only matches if its node agents are listed in workflowagent.Config.SubAgents.
func GraphResumeIndexes(ctx agent.InvocationContext) []int {
	if ctx.Value(graphResumeDoneKey{}) != nil {
		return nil
	}
	responses := FunctionResponses(ctx.UserContent())
	if len(responses) == 0 || ctx.Session() == nil {
		return nil
	}
	selected := map[int]bool{}
	inWorkflow := false
	parentResumable := isResumableWorkflowAgent(ctx.Agent()) || ctx.Value(workflowResumingKey{}) != nil
	events := ctx.Session().Events()
	for _, response := range responses {
		ev := FindFunctionCallEvent(events, response.ID)
		if ev == nil {
			continue
		}
		for i, sub := range ctx.Agent().SubAgents() {
			if sub.FindAgent(ev.Author) == nil {
				continue
			}
			selected[i] = true
			inWorkflow = inWorkflow || calledFromWorkflow(sub, ev.Author, parentResumable)
		}
	}
	if !inWorkflow {
		return nil
	}
	indexes := make([]int, 0, len(selected))
	for i := range selected {
		indexes = append(indexes, i)
	}
	slices.Sort(indexes)
	return indexes
}

// ResumeSubAgent returns the agent to invoke when resuming sub for a function
// response in ctx.UserContent(). Composite and graph workflow sub-agents handle
// their own narrowing; when a non-composite sub-agent (such as an LlmAgent)
// transferred to a descendant before pausing, this returns the first composite
// descendant along the path or the calling descendant itself.
func ResumeSubAgent(ctx agent.InvocationContext, sub agent.Agent) agent.Agent {
	if ctx.Value(graphResumeDoneKey{}) != nil || ctx.Session() == nil || isCompositeAgent(sub) {
		return sub
	}
	events := ctx.Session().Events()
	for _, response := range FunctionResponses(ctx.UserContent()) {
		ev := FindFunctionCallEvent(events, response.ID)
		if ev == nil {
			continue
		}
		if ev.Author == sub.Name() {
			return sub
		}
		if target, ok := findResumeTarget(sub, ev.Author); ok {
			return target
		}
	}
	return sub
}

// CompletedLoopPasses infers how many full passes of ctx.Agent() completed in
// the current invocation before the pause being resumed, used as a stateless
// fallback when no in-memory loop checkpoint is recorded.
func CompletedLoopPasses(ctx agent.InvocationContext) uint {
	if ctx.Value(graphResumeDoneKey{}) != nil || ctx.Session() == nil || ctx.InvocationID() == "" {
		return 0
	}
	effectiveSubs := effectiveSequentialSubAgents(ctx.Agent())
	if len(effectiveSubs) == 0 {
		return 0
	}
	invID := ctx.InvocationID()
	branch := ctx.Branch()
	curAgent := ctx.Agent()
	events := ctx.Session().Events()

	var passes uint
	prevIdx := -1
	for i := range events.Len() {
		ev := events.At(i)
		if ev == nil || ev.Partial || ev.InvocationID != invID || ev.Author == "user" {
			continue
		}
		if !inBranchSubtree(branch, ev.Branch) {
			continue
		}
		if curAgent.FindAgent(ev.Author) == nil {
			passes = 0
			prevIdx = -1
			continue
		}
		if len(effectiveSubs) == 1 {
			if !isCompositeAgent(effectiveSubs[0]) && isCompletedTurnEvent(ev) {
				passes++
			}
			continue
		}
		idx := slices.IndexFunc(effectiveSubs, func(sub agent.Agent) bool {
			return sub.FindAgent(ev.Author) != nil
		})
		if idx < 0 {
			continue
		}
		if prevIdx >= 0 && idx < prevIdx {
			passes++
		}
		prevIdx = idx
	}
	return passes
}

func effectiveSequentialSubAgents(a agent.Agent) []agent.Agent {
	subs := a.SubAgents()
	for len(subs) == 1 {
		ia, ok := subs[0].(agentinternal.Agent)
		if !ok || agentinternal.Reveal(ia).AgentType != agentinternal.TypeSequentialAgent {
			break
		}
		inner := subs[0].SubAgents()
		if len(inner) == 0 {
			break
		}
		subs = inner
	}
	return subs
}

func isCompletedTurnEvent(ev *session.Event) bool {
	if ev.Content == nil || len(FunctionCalls(ev.Content)) > 0 || len(FunctionResponses(ev.Content)) > 0 {
		return false
	}
	return slices.ContainsFunc(ev.Content.Parts, func(p *genai.Part) bool {
		return p != nil && !p.Thought
	})
}

func inBranchSubtree(parentBranch, eventBranch string) bool {
	if parentBranch == "" {
		return true
	}
	return eventBranch == parentBranch || strings.HasPrefix(eventBranch, parentBranch+".")
}

func isResumableWorkflowAgent(a agent.Agent) bool {
	ia, ok := a.(agentinternal.Agent)
	if !ok {
		return false
	}
	switch agentinternal.Reveal(ia).AgentType {
	case agentinternal.TypeWorkflowAgent, agentinternal.TypeSequentialAgent, agentinternal.TypeLoopAgent:
		return true
	default:
		return false
	}
}

func isCompositeAgent(a agent.Agent) bool {
	ia, ok := a.(agentinternal.Agent)
	if !ok {
		return false
	}
	switch agentinternal.Reveal(ia).AgentType {
	case agentinternal.TypeWorkflowAgent, agentinternal.TypeSequentialAgent, agentinternal.TypeLoopAgent, agentinternal.TypeParallelAgent:
		return true
	default:
		return false
	}
}

func calledFromWorkflow(a agent.Agent, author string, inWorkflow bool) bool {
	if isResumableWorkflowAgent(a) {
		inWorkflow = true
	}
	if a.Name() == author {
		return inWorkflow
	}
	for _, sub := range a.SubAgents() {
		if calledFromWorkflow(sub, author, inWorkflow) {
			return true
		}
	}
	return false
}

func findResumeTarget(cur agent.Agent, author string) (agent.Agent, bool) {
	for _, child := range cur.SubAgents() {
		if child.Name() == author {
			return child, true
		}
		if child.FindAgent(author) != nil {
			if isCompositeAgent(child) {
				return child, true
			}
			return findResumeTarget(child, author)
		}
	}
	return nil, false
}
