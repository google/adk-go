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
	"google.golang.org/adk/v2/internal/agent/runconfig"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolauth"
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
//
// It also records author's end in ctx's run config (RecordAgentEnd, which
// does nothing outside a resumable run), as adk-python sets the end flag
// before it builds the event (agents/llm_agent.py, _run_async_impl), so that a
// resumable ParallelAgent above author learns that author ended.
func NewEndOfAgentEvent(ctx agent.InvocationContext, author string) *session.Event {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Author = author
	ev.Branch = ctx.Branch()
	ended := true
	ev.Actions.EndOfAgent = &ended
	if rc := runconfig.FromContext(ctx); rc != nil {
		rc.RecordAgentEnd(author)
	}
	return ev
}

// NewAgentStateEvent returns the event that records state as author's
// checkpoint in ctx's invocation: ctx's invocation ID and branch, no content,
// and Actions.AgentState set to state as given. A nil state becomes an empty,
// non-nil map, because a nil Actions.AgentState means the event carries no
// state. It ports adk-python's BaseAgent._create_agent_state_event for an
// agent with a state (agents/base_agent.py). It records nothing in ctx's run
// config.
func NewAgentStateEvent(ctx agent.InvocationContext, author string, state map[string]any) *session.Event {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Author = author
	ev.Branch = ctx.Branch()
	if state == nil {
		state = map[string]any{}
	}
	ev.Actions.AgentState = state
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

// requestInputFunctionCallName is the HITL input request name
// (workflow.WorkflowInputFunctionCallName, workflow/request_input.go),
// declared here because workflow imports internal/llminternal.
const requestInputFunctionCallName = "adk_request_input"

// resumeAction is what the LLM flow does before calling the model. The zero
// value is resumeActionContinue.
type resumeAction int

const (
	resumeActionContinue    resumeAction = iota // proceed to callLLM
	resumeActionPause                           // return from runOneStep without events
	resumeActionReplayCalls                     // run function calls on resumeDecision.event, then return
)

// resumeDecision pairs a resumeAction with the call event to replay when
// action is resumeActionReplayCalls; event is nil for every other action.
type resumeDecision struct {
	action resumeAction
	event  *session.Event
}

// isInvocationResumable reports whether the per-run RunConfig installed by the
// runner marks this run resumable; false when no RunConfig is installed.
func isInvocationResumable(ctx agent.InvocationContext) bool {
	cfg := runconfig.FromContext(ctx)
	return cfg != nil && cfg.Resumable
}

// decideStepResume decides whether the next LLM step of ctx's agent calls the
// model, pauses, or first runs function calls that never ran, from the events
// of ctx's invocation on ctx's branch. tools are the tools the step can run,
// by name. It never changes an event of ctx's session; a replayed event may
// be a copy holding only the calls that never ran.
//
// It ports adk-python's decide_step_resume (flows/llm_flows/core/_resume.py).
func decideStepResume(ctx agent.InvocationContext, tools map[string]tool.Tool) resumeDecision {
	events := InvocationEvents(ctx, true, true)
	if len(events) == 0 || ctx.Agent() == nil {
		return resumeDecision{}
	}
	if len(events) > 1 {
		if d := decideResume(ctx, events, tools); d.action != resumeActionContinue {
			return d
		}
	}
	// A trailing call is replayed only when this agent wrote it: a session
	// accepts caller-supplied events, and a call from anyone else would run a
	// tool with no model turn behind it.
	last := events[len(events)-1]
	if isInvocationResumable(ctx) && !last.Partial && len(utils.FunctionCalls(utils.Content(last))) > 0 && last.Author == ctx.Agent().Name() {
		return resumeDecision{action: resumeActionReplayCalls, event: last}
	}
	return resumeDecision{}
}

// decideResume is the multi-event core of decideStepResume; events holds at
// least two events. In a resumable run, a call still waiting for its answer
// pauses the step. Otherwise the last call event of ctx's agent that names one
// of tools is replayed whole when its answer came back on a branch the call
// opened, cut down to the calls without a response under their own ID and
// name when a response carries another name, and cut down to the calls that
// never ran when every long-running call has its answer.
//
// It ports adk-python's decide_resume (flows/llm_flows/core/_resume.py), except
// that a response under another name does not replay the calls already
// answered under their own name.
func decideResume(ctx agent.InvocationContext, events []*session.Event, tools map[string]tool.Tool) resumeDecision {
	resumable := isInvocationResumable(ctx)
	pausedByLast := resumable && ShouldPauseInvocation(ctx, events[len(events)-1])
	if resumable && !pausedByLast && pauseLeftCallsUnanswered(ctx, events) {
		return resumeDecision{action: resumeActionPause}
	}
	pause := pausedByLast
	callEvent, callIdx := findTargetCallEvent(events, tools, ctx.Agent().Name())
	if callEvent == nil {
		return pauseOrContinue(pause)
	}

	callIDs, callNames := map[string]bool{}, map[string]bool{}
	for _, fc := range utils.FunctionCalls(utils.Content(callEvent)) {
		if fc.ID != "" {
			callIDs[fc.ID] = true
		}
		callNames[fc.Name] = true
	}
	lroIDs := map[string]bool{}
	for _, ev := range events[callIdx:] {
		for _, id := range ev.LongRunningToolIDs {
			lroIDs[id] = true
			callIDs[id] = true
		}
	}
	later := events[callIdx+1:]
	answeredIDs := responseIDs(later)
	answerEvent := findAnswerEvent(events, callEvent, callIdx, callIDs, callNames)
	// An answer on a sub-branch resolves the call however its IDs look.
	fromSubBranch := isSubBranchAnswer(answerEvent, callEvent)
	answers := utils.FunctionResponses(utils.Content(answerEvent))
	lroUnanswered := len(lroIDs) > 0 && !anyIn(lroIDs, answeredIDs)
	callUnanswered := !anyIn(callIDs, answeredIDs) && !slices.ContainsFunc(answers, func(fr *genai.FunctionResponse) bool {
		return callNames[fr.Name]
	})

	switch {
	case !fromSubBranch && (lroUnanswered || callUnanswered):
		pause = resumable
	case needsCallReplay(callNames, answers, otherCallIDs(events, callIDs), fromSubBranch):
		if fromSubBranch {
			return resumeDecision{action: resumeActionReplayCalls, event: callEvent}
		}
		if ev := wrongNameReplayEvent(callEvent, later); ev != nil {
			return resumeDecision{action: resumeActionReplayCalls, event: ev}
		}
	case allIn(lroIDs, answeredIDs):
		if ev := unexecutedCallsEvent(callEvent, later); ev != nil {
			return resumeDecision{action: resumeActionReplayCalls, event: ev}
		}
	}
	return pauseOrContinue(pause)
}

func pauseOrContinue(pause bool) resumeDecision {
	if pause {
		return resumeDecision{action: resumeActionPause}
	}
	return resumeDecision{}
}

// pauseLeftCallsUnanswered reports whether an event of events before the last
// paused the invocation (see [ShouldPauseInvocation]) and one of the calls it
// awaits, by function call ID or long-running tool ID, has no function
// response in events. Every earlier event counts, not only the one before the
// last: a long-running call followed by several text events sits further back.
//
// It ports adk-python's _pause_left_calls_unanswered
// (flows/llm_flows/core/_resume.py).
func pauseLeftCallsUnanswered(ctx agent.InvocationContext, events []*session.Event) bool {
	awaited := map[string]bool{}
	for _, ev := range events[:len(events)-1] {
		if !ShouldPauseInvocation(ctx, ev) {
			continue
		}
		for _, fc := range utils.FunctionCalls(utils.Content(ev)) {
			if fc.ID != "" {
				awaited[fc.ID] = true
			}
		}
		for _, id := range ev.LongRunningToolIDs {
			awaited[id] = true
		}
	}
	return len(awaited) > 0 && !allIn(awaited, responseIDs(events))
}

// findTargetCallEvent returns the last event of events before the last one,
// and its index, that agentName authored and that calls at least one of
// tools, or (nil, -1) when there is none.
//
// It ports adk-python's _find_target_call_event
// (flows/llm_flows/core/_resume.py).
func findTargetCallEvent(events []*session.Event, tools map[string]tool.Tool, agentName string) (*session.Event, int) {
	for i := len(events) - 2; i >= 0; i-- {
		ev := events[i]
		if ev.Author != agentName {
			continue
		}
		if slices.ContainsFunc(utils.FunctionCalls(utils.Content(ev)), func(fc *genai.FunctionCall) bool {
			_, ok := tools[fc.Name]
			return ok
		}) {
			return ev, i
		}
	}
	return nil, -1
}

// findAnswerEvent returns the last event of events after callIdx, the index of
// callEvent, with a function response that answers callEvent: one whose ID is
// in callIDs, one without an ID whose name is in callNames, or a HITL request
// answered on a branch that one of callEvent's calls opened. It returns the
// last of events when none does.
//
// It ports adk-python's _find_answer_event (flows/llm_flows/core/_resume.py).
func findAnswerEvent(events []*session.Event, callEvent *session.Event, callIdx int, callIDs, callNames map[string]bool) *session.Event {
	calls := utils.FunctionCalls(utils.Content(callEvent))
	for i := len(events) - 1; i > callIdx; i-- {
		ev := events[i]
		for _, fr := range utils.FunctionResponses(utils.Content(ev)) {
			if (fr.ID != "" && callIDs[fr.ID]) ||
				(fr.ID == "" && callNames[fr.Name]) ||
				(isHITLName(fr.Name) && BranchCarriesCall(ev.Branch, calls)) {
				return ev
			}
		}
	}
	return events[len(events)-1]
}

// isHITLName reports whether name is that of a request for user input or for
// a credential, which a nested agent or tool answers on its own sub-branch.
func isHITLName(name string) bool {
	return name == requestInputFunctionCallName || name == toolauth.RequestCredentialFunctionCallName
}

// isSubBranchAnswer reports whether answerEvent came from the user on a
// branch that one of callEvent's calls opened.
//
// It ports adk-python's _is_sub_branch_answer (flows/llm_flows/core/_resume.py).
func isSubBranchAnswer(answerEvent, callEvent *session.Event) bool {
	return answerEvent != nil && answerEvent.Author == "user" &&
		BranchCarriesCall(answerEvent.Branch, utils.FunctionCalls(utils.Content(callEvent)))
}

// needsCallReplay reports whether the calls named callNames still have to run
// given answers, the responses of their answer event: there are calls and
// either no answer, an answer whose name is not one of callNames, or an
// answer from a sub-branch. An answer whose ID is in otherCallIDs, the IDs of
// calls of other call events, answers one of those calls, so its name says
// nothing about callNames.
//
// It ports adk-python's _needs_call_replay (flows/llm_flows/core/_resume.py),
// which also counts the names of answers to other call events, so that one
// event answering the calls of two call events replays the later one.
func needsCallReplay(callNames map[string]bool, answers []*genai.FunctionResponse, otherCallIDs map[string]bool, fromSubBranch bool) bool {
	if len(callNames) == 0 {
		return false
	}
	return len(answers) == 0 || slices.ContainsFunc(answers, func(fr *genai.FunctionResponse) bool {
		return !callNames[fr.Name] && !otherCallIDs[fr.ID]
	}) || fromSubBranch
}

// otherCallIDs returns the non-empty IDs of the function calls in events that
// are not in own.
func otherCallIDs(events []*session.Event, own map[string]bool) map[string]bool {
	ids := map[string]bool{}
	for _, ev := range events {
		for _, fc := range utils.FunctionCalls(utils.Content(ev)) {
			if fc.ID != "" && !own[fc.ID] {
				ids[fc.ID] = true
			}
		}
	}
	return ids
}

// unexecutedCallsEvent returns callEvent cut down to its non-call parts and the
// calls that never ran, or nil when every call ran or when callEvent's author
// wrote an event with content in laterEvents. A call ran when a response in
// laterEvents carries its ID, whatever its name; each response without an ID
// covers one remaining call of its name, in call order. A long-running call
// is never replayed: a long-running tool writes no response when it starts.
//
// It ports adk-python's _unexecuted_calls_event
// (flows/llm_flows/core/_resume.py).
func unexecutedCallsEvent(callEvent *session.Event, laterEvents []*session.Event) *session.Event {
	// Tool responses and auth or confirmation requests are written only after
	// the whole batch ran, so after any of them a call missing its response
	// ran and lost it, or is pending.
	if slices.ContainsFunc(laterEvents, func(ev *session.Event) bool {
		return ev.Author == callEvent.Author && utils.Content(ev) != nil
	}) {
		return nil
	}
	answeredIDs := responseIDs(laterEvents)
	answeredNames := idLessResponseNames(laterEvents)
	return callEventWith(callEvent, func(fc *genai.FunctionCall) bool {
		// An explicit ID takes precedence and consumes no ID-less response.
		if fc.ID != "" && (answeredIDs[fc.ID] || slices.Contains(callEvent.LongRunningToolIDs, fc.ID)) {
			return false
		}
		if answeredNames[fc.Name] > 0 {
			answeredNames[fc.Name]--
			return false
		}
		return true
	})
}

// wrongNameReplayEvent returns callEvent cut down to its non-call parts and the
// calls without a response in laterEvents under their own ID and name, or nil
// when no call is left. Each response without an ID covers one remaining call
// of its name, in call order. Unlike unexecutedCallsEvent, it keeps
// long-running calls and ignores later events of callEvent's author.
func wrongNameReplayEvent(callEvent *session.Event, laterEvents []*session.Event) *session.Event {
	type idName struct{ id, name string }
	answered := map[idName]bool{}
	for _, ev := range laterEvents {
		for _, fr := range utils.FunctionResponses(utils.Content(ev)) {
			if fr.ID != "" {
				answered[idName{fr.ID, fr.Name}] = true
			}
		}
	}
	answeredNames := idLessResponseNames(laterEvents)
	return callEventWith(callEvent, func(fc *genai.FunctionCall) bool {
		if fc.ID != "" && answered[idName{fc.ID, fc.Name}] {
			return false
		}
		if answeredNames[fc.Name] > 0 {
			answeredNames[fc.Name]--
			return false
		}
		return true
	})
}

// callEventWith returns a shallow copy of callEvent whose content holds every
// non-call part and the calls for which keep is true, in their order, or nil
// when keep is true for no call. keep is called once per call, in order.
// callEvent is not changed.
func callEventWith(callEvent *session.Event, keep func(*genai.FunctionCall) bool) *session.Event {
	if callEvent.Content == nil {
		return nil
	}
	var parts []*genai.Part
	kept := false
	for _, p := range callEvent.Content.Parts {
		if p == nil || p.FunctionCall == nil {
			parts = append(parts, p)
			continue
		}
		if keep(p.FunctionCall) {
			parts = append(parts, p)
			kept = true
		}
	}
	if !kept {
		return nil
	}
	content := *callEvent.Content
	content.Parts = parts
	ev := *callEvent
	ev.Content = &content
	return &ev
}

// responseIDs returns the non-empty IDs of the function responses in events.
func responseIDs(events []*session.Event) map[string]bool {
	ids := map[string]bool{}
	for _, ev := range events {
		for _, fr := range utils.FunctionResponses(utils.Content(ev)) {
			if fr.ID != "" {
				ids[fr.ID] = true
			}
		}
	}
	return ids
}

// idLessResponseNames counts the function responses in events that have no
// ID, by name.
func idLessResponseNames(events []*session.Event) map[string]int {
	names := map[string]int{}
	for _, ev := range events {
		for _, fr := range utils.FunctionResponses(utils.Content(ev)) {
			if fr.ID == "" {
				names[fr.Name]++
			}
		}
	}
	return names
}

// anyIn reports whether some key of set is in other.
func anyIn(set, other map[string]bool) bool {
	for k := range set {
		if other[k] {
			return true
		}
	}
	return false
}

// allIn reports whether every key of set is in other.
func allIn(set, other map[string]bool) bool {
	for k := range set {
		if !other[k] {
			return false
		}
	}
	return true
}
