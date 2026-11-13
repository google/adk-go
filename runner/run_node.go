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

package runner

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log"
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/compactionctx"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	imemory "google.golang.org/adk/v2/internal/memory"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// isLlmAgent reports whether a is an LlmAgent (i.e. backed by the
// internal LLM agent state). The node runtime is used only for LlmAgent
// roots, mirroring adk-python where _run_node_async is reached for an
// LlmAgent.
func isLlmAgent(a agent.Agent) bool {
	_, ok := a.(llminternal.Agent)
	return ok
}

// buildRunnerNode wraps any agent as the workflow node that the node
// runtime drives (Go counterpart of adk-python's build_node). All agent
// kinds share the same wrapper; see newAgentNode.
func buildRunnerNode(a agent.Agent) (workflow.Node, error) {
	if a == nil {
		return nil, fmt.Errorf("node runtime: agent cannot be nil")
	}
	return newAgentNode(a), nil
}

// newStandInAgent returns the agent a node root's invocation context
// carries, named and described as n. Plugins, callbacks and telemetry read
// ctx.Agent(); adk-java gives a node root the same kind of stand-in
// (Runner.NodeViewAgent). The stand-in has no Run function and is never run.
func newStandInAgent(n workflow.Node) (agent.Agent, error) {
	a, err := agent.New(agent.Config{Name: n.Name(), Description: n.Description()})
	if err != nil {
		return nil, fmt.Errorf("failed to create stand-in agent for node %q: %w", n.Name(), err)
	}
	return a, nil
}

// runNode drives an agent through the workflow engine by wrapping it
// in a single-node workflow (START -> node), reusing the agent path's
// session/plugin/persist pipeline. The wrapping node bridges the agent's
// HITL (long-running tools, which emit LongRunningToolIDs) into a
// workflow pause; see newAgentNode. Go equivalent of adk-python
// Runner._run_node_async. rootMode is the root LlmAgent's resolved mode.
// When the runner's root is a node, agentToRun is its stand-in agent and
// the workflow drives the node itself.
func (r *Runner) runNode(
	ctx context.Context,
	storedSession session.Session,
	agentToRun agent.Agent,
	msg *genai.Content,
	cfg agent.RunConfig,
	runMetadata map[string]any,
	rootMode llminternal.Mode,
	opts runOptions,
	yield func(*session.Event, error) bool,
) {
	// An invocation that ended in an error is not a finished turn and must not
	// be summarized: the window would hold a question with no answer, and that
	// summary is stored permanently and degrades every later prompt. Observing
	// it here, rather than at each error site, means no path can forget to.
	// One compaction runtime for the whole invocation, attached before both the
	// invocation context and the post-invocation hook are built from this ctx.
	// Allocating it inside newNodeInvocationContext instead gave the mid-turn
	// processor a different instance from the one this function reads, so the
	// "already compacted" hand-off between the two strategies never arrived.
	ctx = compactionctx.ToContext(ctx, r.compactionRuntime())
	ctx = r.withContextCache(ctx)
	defer r.settleContextCaches(ctx, storedSession)

	invocationFailed := false
	emit := yield
	lc := newRunLifecycle(r.pluginManager)
	yield = func(ev *session.Event, err error) bool {
		if err != nil {
			invocationFailed = true
		}
		ok := emit(ev, err)
		lc.yielded(err, ok)
		return ok
	}

	// Compaction has to happen however iteration ends. Breaking out of the
	// range loop on the terminal event is the ordinary streaming idiom, and
	// what the A2A executor does, so a hook placed only after the loop would
	// never run for those callers and compaction would silently never happen.
	// Deferring makes it unconditional.
	//
	// On an early exit the error cannot be yielded, because yield must not be
	// called once it has returned false, so it is logged instead.
	// Assigned once the invocation context exists, below. The compaction hook
	// runs from a defer, so it reads whatever this holds by then.
	var invocationCtx agent.InvocationContext

	compacted := false
	compactOnce := func() error {
		// An aborted invocation is not a finished turn either (adk-python
		// workflow/_node_runner_utils.py, run_node_async).
		if compacted || invocationFailed || icontext.Aborted(ctx) {
			return nil
		}
		compacted = true
		return r.compactAfterInvocation(ctx, storedSession, invocationCtx)
	}
	defer func() {
		if err := compactOnce(); err != nil {
			log.Printf("adk: %v", err) //nolint:forbidigo // pre-slog call site
		}
	}()

	// The message is validated before its invocation is resolved, as in
	// adk-python's run_node_async, so a mixed message is refused as such and
	// not as an unknown function call.
	if msg != nil {
		if err := r.validateNodeRootMessage(msg); err != nil {
			yield(nil, err)
			return
		}
	}
	invocationID, err := r.runInvocationID(storedSession, msg, opts.invocationID, nodePath)
	if err != nil {
		yield(nil, err)
		return
	}
	joinsTask := false
	if taskInvocationID := joinedTaskInvocationID(storedSession, msg, invocationID); taskInvocationID != "" {
		invocationID, joinsTask = taskInvocationID, true
	}

	if err := r.checkCFCModel(ctx, cfg); err != nil {
		yield(nil, err)
		return
	}
	// UserContent is read by Workflow.Run as the workflow's seed input.
	ictx := r.newNodeInvocationContext(ctx, storedSession, agentToRun, msg, cfg, runMetadata, invocationID, joinsTask)
	invocationCtx = ictx
	priorUserContent := ictx.UserContent()

	// Append the user message to history (also runs the on_user_message
	// plugin callback), same as the agent path.
	ictx, userEvent, replaced, err := r.appendMessageToSession(ictx, storedSession, msg, cfg.SaveInputBlobsAsArtifacts, nodePath, invocationID, r.pluginManager, opts.stateDelta)
	// A stage error from it is a run error, reported with this context.
	lc.bind(ictx)
	if err != nil {
		yield(nil, err)
		lc.stop()
		return
	}
	if replaced && r.usesLegacySubAgentPicker(rootMode) {
		// The agent is picked again from the session that now holds the
		// replacement, which may answer a sub-agent's call. Its branch is
		// restored from the invocation's own events only, as adk-python
		// does, so it never takes a branch of an earlier invocation.
		agentToRun, err = r.findAgentToRun(storedSession, nil)
		if err != nil {
			yield(nil, err)
			return
		}
		branch := r.restoreBranchFromHistory(storedSession, agentToRun, ictx.InvocationID())
		ictx = ictx.WithDelta(&agent.CommonContextDelta{
			InvocationContextDelta: &agent.InvocationContextDelta{Agent: &agentToRun, Branch: &branch},
		})
		invocationCtx = ictx
	}
	// A joined turn keeps the replacement as its content.
	if replaced && !joinsTask {
		ictx = withRestoredUserContent(ictx, storedSession, priorUserContent)
		invocationCtx = ictx
	}

	// Plugin lifecycle: after_run is deferred before the user message is
	// yielded, so a caller that stops ranging on that message still gets it,
	// as in adk-python, whose finally covers the user-message yield.
	pluginManager := r.pluginManager
	lc.start(ictx)
	defer lc.finish(nil)
	// Deferred after the after-run callbacks, so it runs before them, as
	// adk-python seals before after_run in one finally.
	abortState := icontext.AbortStateFrom(ictx)
	if abortState != nil {
		defer r.sealOnStop(ictx, storedSession)
	}
	if opts.yieldUserMessage && userEvent != nil {
		if !yield(userEvent, nil) {
			return
		}
	}

	// Run before_run and honor its early exit.
	if pluginManager != nil {
		if event, err := r.runBeforeRunCallback(ictx, false); event != nil || err != nil {
			lc.shortCircuit()
			if yield(event, err) {
				lc.stop()
				lc.finish(yield)
			}
			return
		}
	}

	var wf *workflow.Workflow
	if r.rootNode != nil {
		// Anonymous, so the node's path is "<name>@1" as in Python, and
		// marked as the node-root wrapper, so the node's events carry no
		// workflow bookkeeping (Python runs a node root directly).
		wf, err = workflow.New("", []workflow.Edge{
			{From: workflow.Start, To: r.rootNode},
		}, workflow.WithNodeRootWrapper())
	} else {
		// Built only now, for the agent picked after the message was taken in:
		// the node runs the agent it wraps, not the context's agent.
		var node workflow.Node
		node, err = buildRunnerNode(agentToRun)
		if err != nil {
			yield(nil, err)
			lc.stop()
			return
		}
		// Architectural note: Unlike Go, Python ADK executes standalone agents
		// directly via agent.run_async. Go wraps top-level agents in a synthetic
		// single-node workflow (START -> node) so all execution rides through
		// a unified graph engine. We pass workflow.WithRootWrapper() to prevent
		// this synthetic workflow from stamping an unwanted namespace prefix
		// ("app/agent@1") onto events, ensuring 100% path parity with Python.
		wf, err = workflow.New(rootWorkflowName(r.appName, agentToRun), []workflow.Edge{
			{From: workflow.Start, To: node},
		}, workflow.WithRootWrapper())
	}
	if err != nil {
		yield(nil, fmt.Errorf("failed to build node workflow: %w", err))
		lc.stop()
		return
	}

	// Resume (HITL continuation) vs Run (fresh): a resume is a turn whose
	// function response answers a node waiting on an open interrupt. The
	// paused state comes from session history, not a persisted blob.
	state, err := wf.ReconstructRunState(storedSession, ictx.InvocationID())
	if err != nil {
		yield(nil, fmt.Errorf("failed to reconstruct workflow run state: %w", err))
		lc.stop()
		return
	}

	// The workflow runs under the engine context, which the abort cancels;
	// events are persisted and sealed under ictx, which it does not.
	engine, stopWatch := watchAbort(ictx)
	defer stopWatch()
	var events iter.Seq2[*session.Event, error]
	responses := buildResumeResponses(turnMessage(ictx), state, storedSession)
	switch {
	case abortState.Aborted():
		events = func(func(*session.Event, error) bool) {}
	case len(responses) > 0:
		events = wf.Resume(engine, state, responses)
	default:
		events = wf.Run(engine)
	}

	// Consume the workflow's event stream, same loop as the agent path:
	// on_event plugin callback, persist non-partial events, yield.
	for event, evErr := range events {
		if evErr != nil {
			// The abort's own cancellation, coming back from the workflow,
			// ends the run as a stop, not as a failure.
			if icontext.IsAbortEcho(ictx, evErr) {
				break
			}
			if !yield(nil, evErr) {
				return
			}
			continue
		}

		// Stamp the agent as author on engine control events that have
		// none, else a later turn's findAgentToRun logs a spurious
		// "unknown agent" warning for the empty author.
		if event.Author == "" {
			event.Author = agentToRun.Name()
		}
		stampTimestamp(ictx, event)

		// An on-event plugin error ends the run, as on the classic path.
		event, err = r.processEventWithPlugins(ictx, event)
		if err != nil {
			yield(nil, err)
			lc.stop()
			return
		}
		event = clearMessageAsOutput(event)

		if shouldAppendEvent(event, false) {
			if err := r.sessionService.AppendEvent(ictx, storedSession, event); err != nil {
				yield(nil, fmt.Errorf("failed to add event to session: %w", err))
				lc.stop()
				return
			}
			event = withoutTempState(event)
		}

		if !yield(event, nil) {
			return
		}
		// A signal closed while the caller handled the event stops the run
		// before the next event is requested.
		if abortState.Aborted() {
			break
		}
	}

	if !r.yieldSealed(ictx, storedSession, yield) {
		lc.stop()
		return
	}

	// After-run runs before post-invocation compaction, as in adk-python's
	// run_node_async (workflow/_node_runner_utils.py).
	if !lc.finish(yield) {
		return
	}

	// Compact once the invocation is done and every event it produced has been
	// persisted. Never mid-invocation: that is tail retention's job.
	//
	// compactOnce is idempotent because the deferred call also runs it.
	// Reaching it here means the consumer drained the stream, so a failure can
	// still be reported.
	if err := compactOnce(); err != nil {
		yield(nil, err)
		return
	}
}

// rootWorkflowName derives the persistence-namespacing name for the
// per-run node workflow. It must be stable across turns (so a paused HITL
// run can be resumed) and unique per agent within a session.
func rootWorkflowName(appName string, a agent.Agent) string {
	return appName + "/" + a.Name()
}

// newNodeInvocationContext builds the node path's InvocationContext,
// mirroring the agent path's wiring (parent map for transfer/tool
// resolution, resolved agent for the flow).
//
// The invocation ID is invocationID, the one runInvocationID decided or the
// one an open task delegation lent, or a minted one when it is empty. When
// invocationID names an invocation whose original user message is stored,
// that message is the user content, which a node root reads as its input;
// otherwise msg is (adk-python workflow/_node_runner_utils.py,
// run_node_async). turnMessage still returns msg for the context. A turn
// that joinsTask borrowed invocationID from an open task delegation and
// keeps msg as its user content.
func (r *Runner) newNodeInvocationContext(
	ctx context.Context,
	storedSession session.Session,
	agentToRun agent.Agent,
	msg *genai.Content,
	cfg agent.RunConfig,
	runMetadata map[string]any,
	invocationID string,
	joinsTask bool,
) agent.Context {
	ctx = parentmap.ToContext(ctx, r.parents)
	ctx = runconfig.ToContext(ctx, &runconfig.RunConfig{
		StreamingMode:  runconfig.StreamingMode(cfg.StreamingMode),
		MaxLLMCalls:    runconfig.ResolveMaxLLMCalls(cfg.MaxLLMCalls),
		HTTPOptions:    cfg.HTTPOptions,
		Labels:         cfg.Labels,
		CustomMetadata: runMetadata,
		Resumable:      r.resumable,
		NodePath:       true,
	})
	ctx = r.withPluginManager(ctx)

	var artifacts agent.Artifacts
	if r.artifactService != nil {
		artifacts = &artifactinternal.Artifacts{
			Service:   r.artifactService,
			SessionID: storedSession.ID(),
			AppName:   storedSession.AppName(),
			UserID:    storedSession.UserID(),
		}
	}

	var memoryImpl agent.Memory
	if r.memoryService != nil {
		memoryImpl = &imemory.Memory{
			Service:   r.memoryService,
			SessionID: storedSession.ID(),
			UserID:    storedSession.UserID(),
			AppName:   storedSession.AppName(),
		}
	}

	if joinsTask {
		ctx = context.WithValue(ctx, joinedTaskKey{}, true)
	}
	userContent := msg
	if invocationID != "" && !joinsTask {
		if stored := utils.FindUserMessageForInvocation(storedSession.Events(), invocationID); stored != nil {
			userContent = stored
			ctx = context.WithValue(ctx, turnMessageKey{}, restoredTurn{message: msg, restored: stored})
		}
	}
	ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Artifacts:    artifacts,
		Memory:       memoryImpl,
		Session:      storedSession,
		Branch:       r.restoreBranchFromHistory(storedSession, agentToRun, invocationID),
		Agent:        agentToRun,
		UserContent:  userContent,
		RunConfig:    &cfg,
		InvocationID: invocationID,
	})
	resCtx := agent.NewContext(ic)
	return resCtx
}

type turnMessageKey struct{}

// joinedTaskKey marks the context of a turn that joined an open task
// delegation: it borrowed the invocation ID and is never a retry.
type joinedTaskKey struct{}

// joinedTaskInvocationID returns the invocation ID a run's message borrows
// from the open task delegation of sess (findActiveTaskIsolationScope), or
// "" when it joins none. Only a message that runInvocationID resolved to no
// invocation (invocationID is "") and that answers no function call joins:
// it stays the user's next turn, under the delegation's invocation so the
// task agent sees it (adk-python workflow/_node_runner_utils.py,
// run_node_async). A message with an unmatched function response keeps the
// new invocation that lenient resolution gives it.
func joinedTaskInvocationID(sess session.Session, msg *genai.Content, invocationID string) string {
	if invocationID != "" || msg == nil || answersFunctionCalls(msg) {
		return ""
	}
	_, taskInvocationID := findActiveTaskIsolationScope(sess)
	return taskInvocationID
}

// isJoinedTurn reports whether ctx is the context of a turn that joined an
// open task delegation.
func isJoinedTurn(ctx context.Context) bool {
	joined, _ := ctx.Value(joinedTaskKey{}).(bool)
	return joined
}

// restoredTurn records, for a node-path turn whose user content is the
// invocation's stored message, the message the turn itself brought.
type restoredTurn struct {
	message, restored *genai.Content
}

// withRestoredUserContent returns ctx for a turn whose message an
// on-user-message callback replaced. When the replacement answers function
// calls it stays the turn's message (turnMessage), and the user content
// becomes the user message that ctx's invocation stored, or prior, the user
// content before the callback, when it stored none (adk-python
// workflow/_node_runner_utils.py, run_node_async). Otherwise it returns ctx.
func withRestoredUserContent(ctx agent.Context, storedSession session.Session, prior *genai.Content) agent.Context {
	replacement := ctx.UserContent()
	if !answersFunctionCalls(replacement) {
		return ctx
	}
	userContent := utils.FindUserMessageForInvocation(storedSession.Events(), ctx.InvocationID())
	if userContent == nil {
		userContent = prior
	}
	base := context.Context(context.WithValue(ctx, turnMessageKey{}, restoredTurn{message: replacement, restored: userContent}))
	return ctx.WithDelta(&agent.CommonContextDelta{
		InvocationContextDelta: &agent.InvocationContextDelta{Context: &base, UserContent: &userContent},
	})
}

// answersFunctionCalls reports whether msg holds a function response with an
// ID, adk-python's resume inputs (Runner._extract_resume_inputs).
func answersFunctionCalls(msg *genai.Content) bool {
	for _, fr := range utils.FunctionResponses(msg) {
		if fr.ID != "" {
			return true
		}
	}
	return false
}

// turnMessage returns the message the current node-path turn brought, whose
// function responses answer pending interrupts. It is ctx's user content,
// except when newNodeInvocationContext replaced that with the invocation's
// stored message and no on-user-message callback replaced it since.
func turnMessage(ctx agent.InvocationContext) *genai.Content {
	userContent := ctx.UserContent()
	if turn, ok := ctx.Value(turnMessageKey{}).(restoredTurn); ok && userContent == turn.restored {
		return turn.message
	}
	return userContent
}

// buildResumeResponses maps msg's function responses to interruptID ->
// payload, keeping only those that answer a still-pending interrupt.
// Returns nil for a fresh turn (nothing answered). Go analog of
// adk-python _extract_resume_inputs.
//
// Pending = interrupts waiting in RunState PLUS any unanswered
// long-running call open in history. The latter matters when one turn
// emitted several long-running calls: RunState records only one as
// waiting, but the human may answer the others; matching open history
// calls lets each answer drive a resume until all are resolved.
func buildResumeResponses(msg *genai.Content, state *workflow.RunState, sess session.Session) map[string]any {
	if msg == nil {
		return nil
	}
	pending := map[string]struct{}{}
	if state != nil {
		for id := range waitingInterruptIDs(state) {
			pending[id] = struct{}{}
		}
	}
	for id := range openLongRunningCallIDs(sess) {
		pending[id] = struct{}{}
	}
	if len(pending) == 0 {
		return nil
	}

	var out map[string]any
	for _, fr := range utils.FunctionResponses(msg) {
		if fr == nil || fr.ID == "" {
			continue
		}
		if _, ok := pending[fr.ID]; !ok {
			continue
		}
		if out == nil {
			out = map[string]any{}
		}
		// Opaque payload; Workflow.Resume validates against any schema.
		out[fr.ID] = decodeResumeResponse(fr)
	}
	// The runner has already persisted this turn's answers. Rehydration
	// unwraps and validates them (adk-python's {"result": v} included), so
	// its value replaces the ingress decoding, as in workflowagent.
	if state != nil {
		for _, ns := range state.Nodes {
			if ns == nil {
				continue
			}
			for id, response := range ns.ResumedInputs {
				if _, matched := out[id]; matched {
					out[id] = response
				}
			}
		}
	}
	return out
}

// openLongRunningCallIDs returns the set of long-running tool call IDs in
// session history that do not yet have a matching FunctionResponse, i.e.
// the interrupts still awaiting a human answer.
func openLongRunningCallIDs(sess session.Session) map[string]struct{} {
	open := map[string]struct{}{}
	if sess == nil || sess.Events() == nil {
		return open
	}
	answered := map[string]struct{}{}
	for _, ev := range utils.ApplyRewinds(slices.Collect(sess.Events().All())) {
		for _, id := range ev.LongRunningToolIDs {
			open[id] = struct{}{}
		}
		for _, fr := range utils.FunctionResponses(ev.Content) {
			if fr != nil && fr.ID != "" {
				answered[fr.ID] = struct{}{}
			}
		}
	}
	for id := range answered {
		delete(open, id)
	}
	return open
}

// decodeResumeResponse extracts the user payload from a function
// response, accepting {"response": v}, {"payload": v}, or the raw map. A
// string under "response" is JSON-parsed when possible.
func decodeResumeResponse(fr *genai.FunctionResponse) any {
	if fr.Response == nil {
		return nil
	}
	if raw, ok := fr.Response["response"]; ok {
		if s, isStr := raw.(string); isStr {
			var decoded any
			if err := json.Unmarshal([]byte(s), &decoded); err == nil {
				return decoded
			}
			return s
		}
		return raw
	}
	if payload, ok := fr.Response["payload"]; ok {
		return payload
	}
	return fr.Response
}

// waitingInterruptIDs returns the set of interrupt IDs for every node
// in state that is currently paused on a long-running interrupt, plus the
// IDs of interrupts that history answers (NodeState.ResumedInputs). The
// runner appends the turn's message before it reconstructs the state, so
// the interrupts this turn answers are in the second set, as they are for
// agent/workflowagent's detectResume.
func waitingInterruptIDs(state *workflow.RunState) map[string]struct{} {
	ids := map[string]struct{}{}
	for _, ns := range state.Nodes {
		if ns == nil {
			continue
		}
		for id := range ns.ResumedInputs {
			ids[id] = struct{}{}
		}
		if ns.Status != workflow.NodeWaiting {
			continue
		}
		for _, id := range ns.Interrupts {
			if id != "" {
				ids[id] = struct{}{}
			}
		}
	}
	return ids
}

// resolveInvocationID returns the invocation whose function calls the
// function responses of msg answer, so a resume turn and the pause it
// answers share one ID. Only responses with an ID count; with none it
// returns "". The calls are looked up in the live events of sess (after
// rewinds), newest first. When a response matches no call, resolveStrict
// returns an error; when the calls belong to more than one invocation,
// resolveStrict and resolveRejectMultiple do, and resolveFirst returns the
// invocation of the first response's call. Otherwise an unattributable
// message returns "", which leaves the run to its caller's ID or a minted
// one. Go analog of adk-python Runner._resolve_invocation_id_from_fr.
func resolveInvocationID(sess session.Session, msg *genai.Content, mode resolveMode) (string, error) {
	pending := map[string]bool{}
	for _, fr := range utils.FunctionResponses(msg) {
		if fr.ID != "" {
			pending[fr.ID] = true
		}
	}
	if len(pending) == 0 || sess == nil || sess.Events() == nil {
		return "", nil
	}
	events := utils.ApplyRewinds(slices.Collect(sess.Events().All()))
	var invocations []string
	callInvocation := map[string]string{}
	for i := len(events) - 1; i >= 0 && len(pending) > 0; i-- {
		for _, fc := range utils.FunctionCalls(utils.Content(events[i])) {
			if !pending[fc.ID] {
				continue
			}
			delete(pending, fc.ID)
			callInvocation[fc.ID] = events[i].InvocationID
			if !slices.Contains(invocations, events[i].InvocationID) {
				invocations = append(invocations, events[i].InvocationID)
			}
		}
	}
	switch {
	case len(pending) > 0:
		if mode == resolveStrict {
			return "", fmt.Errorf("function call not found for %d function response ids: each function response ID must match a function call in the session history", len(pending))
		}
		return "", nil
	case len(invocations) > 1:
		if mode == resolveFirst {
			for _, fr := range utils.FunctionResponses(msg) {
				if id, ok := callInvocation[fr.ID]; ok {
					return id, nil
				}
			}
		}
		if mode != resolveLenient {
			return "", fmt.Errorf("function responses resolve to multiple invocations (%d): all function responses in a single message must belong to the same invocation", len(invocations))
		}
		return "", nil
	}
	return invocations[0], nil
}
