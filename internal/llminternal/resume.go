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

package llminternal

import (
	"iter"
	"slices"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

// ResumeRoute is how the router or the flow handles a function call and its
// answer.
type ResumeRoute int

const (
	// ResumeContinue means: the flow takes a fresh model step; the router does
	// its normal history scan.
	ResumeContinue ResumeRoute = iota
	// ResumeRouteToAuthor means: the router runs the agent that made the call.
	ResumeRouteToAuthor
	// ResumeReplayCalls means: the flow dispatches the call event again.
	ResumeReplayCalls
)

// DecideResumeAction returns how to handle callEvent given answerEvent, the
// event that answers it (nil for none), on a runner that is resumable or not.
//
// No answer gives ResumeContinue. An answer authored "user" whose branch
// carries the ID of one of callEvent's calls (see [BranchCarriesCall]) gives
// ResumeReplayCalls. An answer authored "user", or any answer when resumable
// is true, gives ResumeRouteToAuthor. Anything else gives ResumeContinue.
// callEvent must not be nil.
//
// It ports adk-python's decide_resume_action
// (flows/llm_flows/core/_resume.py).
func DecideResumeAction(callEvent, answerEvent *session.Event, resumable bool) ResumeRoute {
	if answerEvent == nil {
		return ResumeContinue
	}
	if answerEvent.Author == "user" && BranchCarriesCall(answerEvent.Branch, utils.FunctionCalls(callEvent.Content)) {
		return ResumeReplayCalls
	}
	if answerEvent.Author == "user" || resumable {
		return ResumeRouteToAuthor
	}
	return ResumeContinue
}

// ShouldPauseInvocation reports whether the invocation pauses right after ev:
// ev has a function call whose ID is in ev.LongRunningToolIDs, and no event
// after ev in ctx's session is authored "user" on a branch that carries that
// call's ID (see [BranchCarriesCall]). Such a user event means the call is
// resuming to process nested input, so it does not pause the invocation.
//
// A paused invocation can be resumed later; an ended one cannot. It ports
// adk-python's InvocationContext.should_pause_invocation
// (agents/invocation_context.py).
func ShouldPauseInvocation(ctx agent.InvocationContext, ev *session.Event) bool {
	calls := utils.FunctionCalls(ev.Content)
	if len(ev.LongRunningToolIDs) == 0 || len(calls) == 0 {
		return false
	}
	later := eventsAfter(ctx.Session(), ev.ID)
	for _, fc := range calls {
		if !slices.Contains(ev.LongRunningToolIDs, fc.ID) {
			continue
		}
		resolving := slices.ContainsFunc(later, func(e *session.Event) bool {
			return e.Author == "user" && BranchCarriesCall(e.Branch, []*genai.FunctionCall{fc})
		})
		if !resolving {
			return true
		}
	}
	return false
}

// eventsAfter returns the events of sess stored after the event with the
// given ID, or nil when sess is nil or does not hold that event.
func eventsAfter(sess session.Session, id string) []*session.Event {
	if sess == nil {
		return nil
	}
	events := sess.Events()
	for i := events.Len() - 1; i >= 0; i-- {
		if events.At(i).ID != id {
			continue
		}
		var out []*session.Event
		for j := i + 1; j < events.Len(); j++ {
			out = append(out, events.At(j))
		}
		return out
	}
	return nil
}

// BranchCarriesCall reports whether branch was opened by one of calls. A
// branch is a dot-joined path of "name@run_id" segments; it carries a call
// when the run ID of one of its segments, compared whole, equals the ID of
// one of calls. An empty branch, and a call with an empty ID, carry nothing.
//
// It ports adk-python's _branch_carries_call
// (flows/llm_flows/core/_resume.py).
func BranchCarriesCall(branch string, calls []*genai.FunctionCall) bool {
	if branch == "" {
		return false
	}
	runIDs := map[string]bool{}
	for segment := range strings.SplitSeq(branch, ".") {
		if i := strings.LastIndexByte(segment, '@'); i >= 0 && i < len(segment)-1 {
			runIDs[segment[i+1:]] = true
		}
	}
	return slices.ContainsFunc(calls, func(fc *genai.FunctionCall) bool {
		return fc != nil && fc.ID != "" && runIDs[fc.ID]
	})
}

// InvocationEvents returns the events of ctx's session, in session order,
// optionally limited to the current invocation and to the current branch.
//
// With currentInvocation, only events whose InvocationID is ctx's are kept.
// With currentBranch, an event not authored "user" is kept only when its
// branch equals ctx's branch exactly. An event authored "user" is kept when
// ctx's branch is empty, or its own branch is empty, equals ctx's branch or
// descends from it; and when it carries function responses on a non-empty
// ctx branch, at least one of their IDs must also match a function call
// issued on ctx's branch or a branch below it.
//
// The returned slice is new; the events are shared. It ports adk-python's
// InvocationContext._get_events (agents/invocation_context.py).
func InvocationEvents(ctx agent.InvocationContext, currentInvocation, currentBranch bool) []*session.Event {
	sess := ctx.Session()
	if sess == nil {
		return nil
	}
	var branchCallIDs map[string]bool
	var out []*session.Event
	for ev := range sess.Events().All() {
		if currentInvocation && ev.InvocationID != ctx.InvocationID() {
			continue
		}
		if currentBranch && !onBranch(ctx, ev, &branchCallIDs) {
			continue
		}
		out = append(out, ev)
	}
	return out
}

// onBranch reports whether InvocationEvents keeps ev for ctx's branch.
// branchCallIDs caches the IDs of the function calls issued on ctx's branch
// or below it; it is filled on first use.
func onBranch(ctx agent.InvocationContext, ev *session.Event, branchCallIDs *map[string]bool) bool {
	branch := ctx.Branch()
	if ev.Author != "user" {
		return ev.Branch == branch
	}
	if responses := utils.FunctionResponses(ev.Content); len(responses) > 0 && branch != "" {
		if *branchCallIDs == nil {
			*branchCallIDs = callIDsOnBranch(ctx.Session(), branch)
		}
		answered, matched := false, false
		for _, fr := range responses {
			if fr.ID == "" {
				continue
			}
			answered = true
			matched = matched || (*branchCallIDs)[fr.ID]
		}
		if answered && !matched {
			return false
		}
	}
	return ev.Branch == "" || branch == "" || ev.Branch == branch || strings.HasPrefix(ev.Branch, branch+".")
}

// callIDsOnBranch returns the IDs of the function calls in sess issued on
// branch or a branch below it.
func callIDsOnBranch(sess session.Session, branch string) map[string]bool {
	ids := map[string]bool{}
	for ev := range sess.Events().All() {
		if ev.Branch != branch && !strings.HasPrefix(ev.Branch, branch+".") {
			continue
		}
		for _, fc := range utils.FunctionCalls(ev.Content) {
			if fc.ID != "" {
				ids[fc.ID] = true
			}
		}
	}
	return ids
}

// NewEndOfAgentEvent returns the event that marks the end of author's run in
// ctx's invocation: ctx's invocation ID and branch, no content, and
// Actions.EndOfAgent true. It ports adk-python's
// BaseAgent._create_agent_state_event for an agent that ended
// (agents/base_agent.py).
func NewEndOfAgentEvent(ctx agent.InvocationContext, author string) *session.Event {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Author = author
	ev.Branch = ctx.Branch()
	ended := true
	ev.Actions.EndOfAgent = &ended
	return ev
}

// TransferToAgentToolName is the name of the transfer_to_agent tool, which
// a transfer response carries.
const TransferToAgentToolName = transferAgentName

// TransferTarget returns the agent named name among the agents that ctx's
// agent may transfer to (its sub-agents, and its parent and peers unless
// they are disallowed), or nil when there is none.
func TransferTarget(ctx agent.InvocationContext, name string) agent.Agent {
	parents := parentmap.FromContext(ctx)
	for _, target := range transferTargets(ctx.Agent(), parents[ctx.Agent().Name()]) {
		if target.Name() == name {
			return target
		}
	}
	return nil
}

// RunTransferTarget runs target, the agent a transfer hands the invocation
// to, on ctx: through its RunNode method when it has one, as an LlmAgent
// does, so that it binds its own mode, and through its Run method otherwise.
func RunTransferTarget(ctx agent.InvocationContext, target agent.Agent) iter.Seq2[*session.Event, error] {
	type nodeRunner interface {
		RunNode(ctx agent.Context, nodeInput any) iter.Seq2[*session.Event, error]
	}
	if nr, ok := target.(nodeRunner); ok {
		return nr.RunNode(agent.Promote(ctx), nil)
	}
	return target.Run(ctx)
}

// FindCallEvent returns the last of events that holds a function call with
// the ID callID, or nil when callID is empty or no event holds one.
func FindCallEvent(events []*session.Event, callID string) *session.Event {
	if callID == "" {
		return nil
	}
	for _, event := range slices.Backward(events) {
		for _, fc := range utils.FunctionCalls(event.Content) {
			if fc.ID == callID {
				return event
			}
		}
	}
	return nil
}
