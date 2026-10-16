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

package agentanalytics

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
)

const (
	// maxTrackedInvocations bounds the number of invocations whose spans
	// are tracked at once.
	maxTrackedInvocations = 4096
	// invocationTTL is how long an invocation stays tracked after its last
	// callback. It bounds the memory held by invocations that never reach
	// AfterRunCallback, for example because their caller stopped reading
	// events.
	invocationTTL = time.Hour
	// sweepInterval is the number of new invocations between two sweeps
	// for expired ones.
	sweepInterval = 128
)

var agentInterfaceType = reflect.TypeFor[agent.Agent]()

// spanKind is the kind of operation that a span covers.
type spanKind int

const (
	spanKindInvocation spanKind = iota + 1
	spanKindAgent
	spanKindLLM
	spanKindTool
)

// spanRecord is a span that has started.
type spanRecord struct {
	spanID       string
	parentSpanID string
	kind         spanKind
	// path, name and branch identify the activation of agent spans.
	path   string
	name   string
	branch string
	start  time.Time
	// firstToken is the arrival time of the first streamed chunk of a
	// model call that carries generated content.
	firstToken time.Time
	// hasResponse reports whether the agent activation has produced a model
	// response or an event with content or output, itself or through an
	// activation that ran under it.
	hasResponse bool
}

// rowSpan places a row in the trace of its invocation. It holds the
// trace_id, span_id and parent_span_id columns, the latency_ms object of
// rows that end a span, and the name of the invocation's root agent.
type rowSpan struct {
	traceID       string
	spanID        string
	parentSpanID  string
	latency       map[string]any
	rootAgentName string
}

// invocationTrace holds the spans of one invocation.
type invocationTrace struct {
	// lastActive is the time of the invocation's last callback, in Unix
	// nanoseconds. Eviction reads it without holding mu.
	lastActive atomic.Int64
	// openCalls is the number of in-flight model and tool calls. Eviction
	// reads it without holding mu.
	openCalls atomic.Int64
	// rootAgentAtomic holds the latest rootAgentName string for lock-free
	// reads during eviction.
	rootAgentAtomic atomic.Value

	// traceID, rootSpanID and rootParentSpanID are initialized before inv is
	// published in tm.invocations and are read-only afterwards.
	traceID          string
	rootSpanID       string
	rootParentSpanID string

	mu             sync.Mutex
	rootAgentName  string
	configuredRoot string
	started        bool
	evicted        bool
	// runnerAgent is the root agent of the tree that the runner runs. The
	// agents of AGENT_STARTING rows are looked up in its tree.
	runnerAgent agent.Agent
	// parentAgent maps each sub-agent name to its parent agent name in the
	// configured agent tree.
	parentAgent map[string]string
	// sequentialParent maps each sub-agent name to its parent agent name when
	// that parent is a non-LLM workflow agent (such as a sequential or loop
	// agent) rather than an llmagent whose sub-agents can transfer to peers.
	sequentialParent map[string]string
	// agentStack holds the invocation span, followed by the spans of the
	// running agents in the order they started. Agents that run
	// concurrently, such as the sub-agents of a parallel agent or
	// workflow.RunNode activations, can end in any order, so spans are
	// removed from wherever they are.
	agentStack []*spanRecord
	// llmSpans holds the running model calls, by llmKey.
	llmSpans map[string]*spanRecord
	// completedModels holds the model calls that OnModelErrorCallback
	// ended, so that if a plugin recovers from the error and ADK runs
	// AfterModelCallback for the same call, LLM_RESPONSE reuses the call's
	// span without reporting latency a second time.
	completedModels map[string]*spanRecord
	// toolSpans holds the running tool calls, by toolKey. Concurrent tool
	// calls of an agent are siblings: each is parented to the agent and
	// never to another tool call.
	toolSpans map[string]*spanRecord
	// completedTools holds the tool calls that OnToolErrorCallback ended,
	// so that the AfterToolCallback that ADK runs next for the same call
	// finds its span instead of ending another one. ADK runs
	// OnToolErrorCallback alone for a call of a tool that does not exist;
	// such records last as long as the invocation.
	completedTools map[string]*spanRecord
	// unrecoveredErrors tracks unhandled errors encountered during the
	// invocation keyed by activation / call identity (e.g. llmKey), together
	// with the identity of the agent activation that reported each of them.
	// An agent activation that ends normally clears the errors recorded for
	// it and for the open activations that its end retires.
	unrecoveredErrors map[string]recordedError
}

// recordedError is an unrecovered error of an invocation and the identity of
// the agent activation (or of its model or tool call) that it was recorded
// for.
type recordedError struct {
	err error
	id  identity
}

func newInvocationTrace(ctx context.Context, traceID string, now time.Time) *invocationTrace {
	root := &spanRecord{spanID: newSpanID(), kind: spanKindInvocation, start: now}
	inv := &invocationTrace{
		traceID:           cmp.Or(traceID, newTraceID()),
		rootSpanID:        root.spanID,
		parentAgent:       map[string]string{},
		sequentialParent:  map[string]string{},
		agentStack:        []*spanRecord{root},
		llmSpans:          map[string]*spanRecord{},
		completedModels:   map[string]*spanRecord{},
		toolSpans:         map[string]*spanRecord{},
		completedTools:    map[string]*spanRecord{},
		unrecoveredErrors: map[string]recordedError{},
	}
	// Continue the caller's OpenTelemetry trace, if any. Unlike adk-python,
	// which reuses the ambient span ID as the invocation span ID, ADK Go's
	// runner does not open a per-invocation OpenTelemetry span before
	// invoking plugin callbacks, so a caller's long-lived request or server
	// span would otherwise be reused as the span_id of multiple
	// invocations. Allocating a child invocation span under the ambient
	// span keeps each invocation's span tree distinct while preserving the
	// caller's raw OpenTelemetry IDs in attributes.otel.
	if sc := trace.SpanContextFromContext(ctx); sc.IsValid() {
		inv.traceID = sc.TraceID().String()
		root.parentSpanID = sc.SpanID().String()
		inv.rootParentSpanID = root.parentSpanID
	}
	inv.lastActive.Store(now.UnixNano())
	return inv
}

// hasOpenCalls reports whether the invocation has an in-flight model or tool
// call waiting for its completion callback.
func (inv *invocationTrace) hasOpenCalls() bool {
	return inv.openCalls.Load() > 0
}

func (inv *invocationTrace) syncOpenCallsLocked() {
	inv.openCalls.Store(int64(len(inv.llmSpans) + len(inv.toolSpans)))
}

func (inv *invocationTrace) setRootAgentLocked(name string) {
	inv.rootAgentName = name
	if name != "" {
		inv.rootAgentAtomic.Store(name)
	}
}

// restoreEvicted restores the root span identity and trace metadata saved when
// inv was previously evicted. Called before inv is published in tm.invocations.
func (inv *invocationTrace) restoreEvicted(ev evictedTrace) {
	inv.evicted = true
	if ev.traceID != "" {
		inv.traceID = ev.traceID
	}
	if ev.spanID != "" {
		inv.rootSpanID = ev.spanID
		inv.agentStack[0].spanID = ev.spanID
	}
	inv.rootParentSpanID = ev.parentSpanID
	inv.agentStack[0].parentSpanID = ev.parentSpanID
	inv.setRootAgentLocked(ev.rootAgentName)
}

// rowSpan returns the place of a row that describes rec. latency is set on
// rows that end rec.
func (inv *invocationTrace) rowSpan(rec *spanRecord, latency map[string]any) rowSpan {
	return rowSpan{
		traceID:       inv.traceID,
		spanID:        rec.spanID,
		parentSpanID:  rec.parentSpanID,
		latency:       latency,
		rootAgentName: inv.rootAgentName,
	}
}

// noteRootAgent records the invocation's root agent and agent hierarchy from
// ctx, runner, and any rememberedRoot resolved from tm's agent-instance parent
// map before acquiring inv.mu. runner is nil in callbacks that do not get the
// agent directly.
func (inv *invocationTrace) noteRootAgent(ctx context.Context, id identity, runner, rememberedRoot agent.Agent, useBranchFallback bool) {
	if inv.configuredRoot == "" && len(inv.parentAgent) == 0 {
		if parents := orZero(func() map[string]agent.Agent { return contextParentMap(ctx) }); len(parents) > 0 {
			for child, parent := range parents {
				if pName := orZero(parent.Name); child != "" && pName != "" {
					inv.parentAgent[child] = pName
					if !isLLMAgent(parent) {
						inv.sequentialParent[child] = pName
					}
				}
				inv.recordAgentTree(parent)
			}
			if root := topAgentFromMap(runner, id.agentName, parents); root != nil {
				inv.runnerAgent = root
				if name := orZero(root.Name); name != "" {
					inv.configuredRoot = name
					inv.setRootAgentLocked(name)
				}
			}
		}
	}
	if runner != nil {
		inv.recordAgentTree(runner)
		if rememberedRoot != nil && rememberedRoot != runner {
			inv.recordAgentTree(rememberedRoot)
			if inv.runnerAgent == nil {
				inv.runnerAgent = rememberedRoot
			}
			if name := orZero(rememberedRoot.Name); name != "" && inv.configuredRoot == "" {
				inv.configuredRoot = name
				inv.setRootAgentLocked(name)
			}
		}
		if inv.runnerAgent == nil || inv.isAncestorAgent(orZero(runner.Name), orZero(inv.runnerAgent.Name)) {
			inv.runnerAgent = runner
		}
		if rName := orZero(runner.Name); rName != "" && inv.isAncestorAgent(rName, cmp.Or(inv.configuredRoot, inv.rootAgentName, id.agentName)) {
			inv.configuredRoot = rName
			inv.setRootAgentLocked(rName)
		}
	}
	if inv.configuredRoot != "" {
		inv.setRootAgentLocked(inv.configuredRoot)
		return
	}
	if inv.rootAgentName == "" {
		switch {
		case runner != nil:
			inv.setRootAgentLocked(cmp.Or(id.agentName, orZero(runner.Name)))
		case useBranchFallback && id.branch != "":
			root, _, _ := strings.Cut(id.branch, ".")
			inv.setRootAgentLocked(root)
		default:
			inv.setRootAgentLocked(id.agentName)
		}
	}
}

func isLLMAgent(a agent.Agent) bool {
	if a == nil {
		return false
	}
	_, ok := orZero(func() any { return agentConfig(a) }).(llmagent.Config)
	return ok
}

// recordAgentTree records the child-to-parent edges of root's sub-agent tree.
func (inv *invocationTrace) recordAgentTree(root agent.Agent) {
	if root == nil {
		return
	}
	visited := map[string]bool{}
	var walk func(a agent.Agent)
	walk = func(a agent.Agent) {
		if a == nil {
			return
		}
		parentName := orZero(a.Name)
		if parentName == "" || visited[parentName] {
			return
		}
		visited[parentName] = true
		nonLLM := !isLLMAgent(a)
		for _, sub := range orZero(a.SubAgents) {
			if sub == nil {
				continue
			}
			if subName := orZero(sub.Name); subName != "" {
				inv.parentAgent[subName] = parentName
				if nonLLM {
					inv.sequentialParent[subName] = parentName
				}
				walk(sub)
			}
		}
	}
	walk(root)
}

// topAgentFromMap walks parents upward from start (or agentName) to the
// configured root agent of the runner's tree.
func topAgentFromMap(start agent.Agent, agentName string, parents map[string]agent.Agent) agent.Agent {
	cur := start
	if cur == nil && agentName != "" {
		cur = parents[agentName]
	}
	if cur == nil {
		for _, p := range parents {
			cur = p
			break
		}
	}
	for range len(parents) + 1 {
		if cur == nil {
			break
		}
		next := parents[orZero(cur.Name)]
		if next == nil || next == cur {
			return cur
		}
		cur = next
	}
	return cur
}

// isAncestorAgent reports whether ancestor is a configured ancestor of child
// in inv.parentAgent.
func (inv *invocationTrace) isAncestorAgent(ancestor, child string) bool {
	if ancestor == "" || child == "" {
		return false
	}
	for cur, steps := inv.parentAgent[child], 0; cur != "" && steps <= len(inv.parentAgent); cur, steps = inv.parentAgent[cur], steps+1 {
		if cur == ancestor {
			return true
		}
	}
	return false
}

// findAgent returns the index in agentStack of the most recent span of the
// agent name running on path and branch, falling back to matching branch and
// name, or name alone, or -1. It never returns the invocation span.
func (inv *invocationTrace) findAgent(path, name, branch string) int {
	branchMatch := -1
	nameMatch := -1
	for i := len(inv.agentStack) - 1; i > 0; i-- {
		rec := inv.agentStack[i]
		if rec.kind != spanKindAgent || rec.name != name || (path != "" && rec.path != "" && rec.path != path) {
			continue
		}
		if path != "" && rec.path == path && rec.branch == branch {
			return i
		}
		if rec.branch == branch && branchMatch < 0 {
			if path == "" || rec.path == "" {
				return i
			}
			branchMatch = i
		}
		if nameMatch < 0 {
			nameMatch = i
		}
	}
	if branchMatch >= 0 {
		return branchMatch
	}
	return nameMatch
}

// parentFor returns the span that an agent starting with id runs under:
// the most recent agent whose branch and node path contain id, or the
// invocation. When id has a configured non-LLM parent in inv.sequentialParent,
// sibling spans that are not configured ancestors of id are skipped without
// mutating agentStack on push.
func (inv *invocationTrace) parentFor(id identity) *spanRecord {
	hasSequentialParent := inv.sequentialParent[id.agentName] != ""
	for i := len(inv.agentStack) - 1; i > 0; i-- {
		rec := inv.agentStack[i]
		if !withinBranch(id.branch, rec.branch) || !withinPath(id.path, rec.path, rec.name == id.agentName) {
			continue
		}
		if hasSequentialParent && !inv.isAncestorAgent(rec.name, id.agentName) {
			continue
		}
		return rec
	}
	return inv.agentStack[0]
}

// ownerOf returns the span that the model and tool calls of id run under:
// the agent's own span, if it is known.
func (inv *invocationTrace) ownerOf(id identity) *spanRecord {
	if i := inv.findAgent(id.path, id.agentName, id.branch); i > 0 {
		return inv.agentStack[i]
	}
	return inv.parentFor(id)
}

// withinBranch reports whether branch is ancestor or one of its descendant
// branches. ADK names the branch of a sub-agent of a parallel agent after
// its parent branch: "parent.parallel.sub".
func withinBranch(branch, ancestor string) bool {
	return ancestor == "" || branch == ancestor || strings.HasPrefix(branch, ancestor+".")
}

// withinPath reports whether path runs inside the node activation ancestor.
// When both paths are non-empty, a child agent either runs in a descendant
// node path ("ancestor/child@2") or, for sub-agents invoked directly without
// entering a new workflow node, shares ancestor's path under a different
// agent name.
func withinPath(path, ancestor string, sameAgent bool) bool {
	if path == "" || ancestor == "" {
		return true
	}
	if path == ancestor {
		return !sameAgent
	}
	return strings.HasPrefix(path, ancestor+"/")
}

// traceManager assigns trace, span and parent span IDs to rows and measures
// the latency of invocations, agent runs, model calls and tool calls. A
// traceManager is safe for concurrent use by multiple goroutines.
//
// ADK runs the tool calls of a model response, the sub-agents of a parallel
// agent, and concurrent workflow.RunNode activations in parallel. A single
// stack of spans would let them end each other's spans, so the manager keys
// agent, model and tool spans by node path, agent, branch and function call
// ID, and parents model and tool calls to the span of their agent. Callers
// that fan out the same agent concurrently via workflow.RunNode can also pass
// workflow.WithUseSubBranch when they want isolated session branch history in
// addition to distinct node paths.
type traceManager struct {
	mu          sync.Mutex
	invocations map[string]*invocationTrace
	evicted     map[string]evictedTrace
	knownParent map[agent.Agent]agent.Agent
	traceSeed   string
	inserts     int
	logf        func(string, ...any)

	// maxTracked, ttl and now are fields so that tests can change them.
	maxTracked int
	ttl        time.Duration
	now        func() time.Time
}

// evictedTrace is what is kept of an evicted invocation, so that its later rows
// keep its invocation span and root agent.
type evictedTrace struct {
	traceID       string
	spanID        string
	parentSpanID  string
	rootAgentName string
}

func newTraceManager() *traceManager {
	return &traceManager{
		invocations: map[string]*invocationTrace{},
		evicted:     map[string]evictedTrace{},
		knownParent: map[agent.Agent]agent.Agent{},
		traceSeed:   newTraceID(),
		maxTracked:  maxTrackedInvocations,
		ttl:         invocationTTL,
		now:         time.Now,
	}
}

// traceIDFor returns a deterministic 32-hex-digit trace ID for invocationID
// within tm when invocationID is non-empty, so an invocation whose state was
// evicted mid-run and recreated on a later callback keeps its trace_id.
func (tm *traceManager) traceIDFor(invocationID string) string {
	if invocationID == "" {
		return newTraceID()
	}
	sum := sha256.Sum256([]byte(tm.traceSeed + "\x00" + invocationID))
	return hex.EncodeToString(sum[:16])
}

// rememberAgentTree records the sub-agent-to-parent links of runner by
// agent.Agent instance and returns the highest ancestor known for runner.
// Sub-agents are walked before acquiring tm.mu so user agent code is never
// called under a traceManager lock.
func (tm *traceManager) rememberAgentTree(runner agent.Agent) agent.Agent {
	if runner == nil {
		return nil
	}
	type edge struct {
		child, parent agent.Agent
	}
	var edges []edge
	visited := map[agent.Agent]bool{}
	var walk func(a agent.Agent)
	walk = func(a agent.Agent) {
		if a == nil || visited[a] {
			return
		}
		visited[a] = true
		for _, sub := range orZero(a.SubAgents) {
			if sub == nil {
				continue
			}
			edges = append(edges, edge{child: sub, parent: a})
			walk(sub)
		}
	}
	walk(runner)

	tm.mu.Lock()
	defer tm.mu.Unlock()
	if len(edges) > 0 {
		if len(tm.knownParent)+len(edges) > maxTrackedInvocations {
			clear(tm.knownParent)
		}
		for _, e := range edges {
			tm.knownParent[e.child] = e.parent
		}
	}
	cur := runner
	for range len(tm.knownParent) + 1 {
		next := tm.knownParent[cur]
		if next == nil || next == cur {
			return cur
		}
		cur = next
	}
	return cur
}

// invocation returns the trace of the invocation id belongs to, creating it
// on the invocation's first callback. Callbacks without an invocation ID
// get a trace that is not tracked, so that their rows still have IDs.
func (tm *traceManager) invocation(ctx context.Context, id identity) *invocationTrace {
	now := tm.now()
	if id.invocationID == "" {
		return newInvocationTrace(ctx, "", now)
	}
	tm.mu.Lock()
	defer tm.mu.Unlock()
	inv, ok := tm.invocations[id.invocationID]
	if !ok {
		ev, wasEvicted := tm.evicted[id.invocationID]
		inv = newInvocationTrace(ctx, tm.traceIDFor(id.invocationID), now)
		if wasEvicted {
			inv.restoreEvicted(ev)
		}
		tm.invocations[id.invocationID] = inv
		tm.inserts++
		tm.evictLocked(now, id.invocationID)
	}
	inv.lastActive.Store(now.UnixNano())
	return inv
}

// evictLocked drops invocations idle for longer than the TTL that have no
// open model or tool calls, every sweepInterval insertions or when too many
// are tracked, and then the least recently active ones while too many are
// still tracked. keep is never dropped. tm.mu must be held.
func (tm *traceManager) evictLocked(now time.Time, keep string) {
	cutoff := now.Add(-tm.ttl).UnixNano()
	if tm.inserts%sweepInterval == 0 || len(tm.invocations) > tm.maxTracked {
		for id, inv := range tm.invocations {
			if id != keep && inv.lastActive.Load() < cutoff && !inv.hasOpenCalls() {
				tm.markEvictedLocked(id, inv)
				delete(tm.invocations, id)
			}
		}
	}
	for len(tm.invocations) > tm.maxTracked {
		oldestID, oldest := "", int64(0)
		for id, inv := range tm.invocations {
			if t := inv.lastActive.Load(); id != keep && (oldestID == "" || t < oldest) {
				oldestID, oldest = id, t
			}
		}
		if oldestID == "" {
			return
		}
		evictedInv := tm.invocations[oldestID]
		if (oldest >= cutoff || evictedInv.hasOpenCalls()) && tm.logf != nil {
			tm.logf("agentanalytics: trace manager capacity (%d) exceeded; evicting active invocation %q", tm.maxTracked, oldestID)
		}
		tm.markEvictedLocked(oldestID, evictedInv)
		delete(tm.invocations, oldestID)
	}
}

func (tm *traceManager) markEvictedLocked(invocationID string, inv *invocationTrace) {
	if len(tm.evicted) >= max(tm.maxTracked, 64) {
		clear(tm.evicted)
	}
	var ev evictedTrace
	if inv != nil {
		rootAgent, _ := inv.rootAgentAtomic.Load().(string)
		ev = evictedTrace{
			traceID:       inv.traceID,
			spanID:        inv.rootSpanID,
			parentSpanID:  inv.rootParentSpanID,
			rootAgentName: rootAgent,
		}
	}
	tm.evicted[invocationID] = ev
}

// invocationSpan returns the place of a row written by a callback of the
// whole invocation, such as USER_MESSAGE_RECEIVED. Such callbacks get the
// agent that the runner runs, runner, which is used to record the root
// agent.
func (tm *traceManager) invocationSpan(ctx context.Context, id identity, runner agent.Agent) rowSpan {
	inv := tm.invocation(ctx, id)
	rememberedRoot := tm.rememberAgentTree(runner)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.started = true
	inv.noteRootAgent(ctx, id, runner, rememberedRoot, false)
	return inv.rowSpan(inv.agentStack[0], nil)
}

// endInvocation stops tracking the invocation and returns the place of its
// INVOCATION_COMPLETED row, with the invocation latency if the invocation
// was tracked from its start without being evicted mid-run.
func (tm *traceManager) endInvocation(ctx context.Context, id identity) rowSpan {
	var inv *invocationTrace
	var ev evictedTrace
	var wasEvicted bool
	if id.invocationID != "" {
		tm.mu.Lock()
		inv = tm.invocations[id.invocationID]
		ev, wasEvicted = tm.evicted[id.invocationID]
		delete(tm.invocations, id.invocationID)
		delete(tm.evicted, id.invocationID)
		tm.mu.Unlock()
	}
	if inv == nil {
		inv = newInvocationTrace(ctx, tm.traceIDFor(id.invocationID), tm.now())
		if wasEvicted {
			inv.restoreEvicted(ev)
		}
		inv.noteRootAgent(ctx, id, nil, nil, false)
		return inv.rowSpan(inv.agentStack[0], nil)
	}
	now := tm.now()
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgent(ctx, id, nil, nil, false)
	var latency map[string]any
	if inv.started && !inv.evicted {
		latency = latencyOf(inv.agentStack[0], now)
	}
	return inv.rowSpan(inv.agentStack[0], latency)
}

// pushAgentSpan starts the span of an agent run and returns the place of
// its AGENT_STARTING row.
func (tm *traceManager) pushAgentSpan(ctx context.Context, id identity) rowSpan {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.started = true
	inv.noteRootAgent(ctx, id, nil, nil, true)
	rec := &spanRecord{
		spanID:       newSpanID(),
		parentSpanID: inv.parentFor(id).spanID,
		kind:         spanKindAgent,
		path:         id.path,
		name:         id.agentName,
		branch:       id.branch,
		start:        now,
	}
	inv.agentStack = append(inv.agentStack, rec)
	return inv.rowSpan(rec, nil)
}

// agentNamed returns the agent of id, looked up by name in the tree of the
// agent that the runner runs, or nil if it is not found there.
func (tm *traceManager) agentNamed(ctx context.Context, id identity) agent.Agent {
	inv := tm.invocation(ctx, id)
	inv.mu.Lock()
	inv.noteRootAgent(ctx, id, nil, nil, true)
	runner := inv.runnerAgent
	inv.mu.Unlock()
	if runner == nil {
		return nil
	}
	// FindAgent can run code of custom agents, so mu is not held.
	return orZero(func() agent.Agent { return runner.FindAgent(id.agentName) })
}

// popAgentSpan ends the span of an agent run and returns the place of its
// AGENT_COMPLETED row. It never ends the invocation span: if the agent's
// span is unknown, the row gets a span of its own and no latency.
//
// The errors recorded for the activations that end here are cleared: ADK
// ended them normally, for example because a plugin recovered from the error
// or because the caller kept reading after ADK yielded it, so they no longer
// fail the invocation.
func (tm *traceManager) popAgentSpan(ctx context.Context, id identity) rowSpan {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgent(ctx, id, nil, nil, true)
	i := inv.findAgent(id.path, id.agentName, id.branch)
	if i < 0 {
		return inv.rowSpan(inv.newChild(inv.parentFor(id), spanKindAgent, now), nil)
	}
	rec := inv.agentStack[i]
	// The recorded errors are attributed while the spans are still open.
	errorSpans := make(map[string]string, len(inv.unrecoveredErrors))
	for key, e := range inv.unrecoveredErrors {
		if j := inv.activationOfLocked(e); j > 0 {
			errorSpans[key] = inv.agentStack[j].spanID
		}
	}
	inv.agentStack = slices.Delete(inv.agentStack, i, i+1)
	// Also retire any unclosed descendant spans whose parentSpanID lineage
	// reaches rec.spanID (for example, a sub-agent short-circuited in
	// BeforeAgentCallback that never ran AfterAgentCallback).
	retired := map[string]bool{rec.spanID: true}
	for idx := i; idx < len(inv.agentStack); {
		if retired[inv.agentStack[idx].parentSpanID] {
			retired[inv.agentStack[idx].spanID] = true
			inv.agentStack = slices.Delete(inv.agentStack, idx, idx+1)
			continue
		}
		idx++
	}
	for key, spanID := range errorSpans {
		if retired[spanID] {
			delete(inv.unrecoveredErrors, key)
		}
	}
	return inv.rowSpan(rec, latencyOf(rec, now))
}

// unclosedAgentSpan records an unclosed agent span popped during invocation termination.
type unclosedAgentSpan struct {
	span      rowSpan
	agentName string
	branch    string
	path      string
	err       error
}

// completeResponse records that the author of a final, non-partial session
// event with content or an output responded. The author's activation is
// looked up from the event's author, branch and node path in id, as in
// eventSpan, because the event callback gets the agent that the runner runs
// rather than the author. The model calls still open under the activation end
// without a row: ADK runs no AfterModelCallback for a call that a
// BeforeModelCallback answered, so its response event is all that ends it.
// The activation and the open activations that it runs under are marked as
// responded. An event whose author has no open agent span changes nothing.
func (tm *traceManager) completeResponse(ctx context.Context, id identity, author, branch string) {
	inv := tm.invocation(ctx, id)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	i := inv.findAgent(id.path, author, branch)
	if i <= 0 {
		return
	}
	owner := inv.agentStack[i]
	for key, rec := range inv.llmSpans {
		if rec.parentSpanID == owner.spanID {
			delete(inv.llmSpans, key)
		}
	}
	inv.syncOpenCallsLocked()
	lineage := inv.lineageLocked(i)
	for _, rec := range inv.agentStack[1:] {
		if lineage[rec.spanID] {
			rec.hasResponse = true
		}
	}
}

// recordError marks an unrecovered error on the invocation trace keyed by activation/call key.
func (tm *traceManager) recordError(ctx context.Context, id identity, key string, err error) {
	if err == nil {
		return
	}
	inv := tm.invocation(ctx, id)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	if inv.unrecoveredErrors == nil {
		inv.unrecoveredErrors = make(map[string]recordedError)
	}
	inv.unrecoveredErrors[key] = recordedError{err: err, id: id}
}

// clearError clears any recorded unrecovered error for key on the invocation trace,
// called when a subsequent callback recovers from an error.
func (tm *traceManager) clearError(ctx context.Context, id identity, key string) {
	inv := tm.invocation(ctx, id)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	delete(inv.unrecoveredErrors, key)
}

// popUnclosedAgentSpans ends the invocation's agent spans that are still open
// when the invocation terminates and returns the invocation's terminal error,
// if any, together with the open spans that the error is attributed to.
//
// Plugin callbacks do not observe every failure, so the terminal error covers
// only these detectable ones, in order of precedence:
//
//   - An unrecovered model error that a callback recorded, if any agent span
//     is still open. A response to the failed model call, after a plugin
//     recovered from the error, clears the error, and so does the normal end
//     of its activation or of an activation that it runs under.
//   - The error of ctx when the invocation was canceled. Cancellation is
//     returned even when every agent span is closed, or when no agent ran at
//     all, for example because a BeforeRunCallback canceled the invocation.
//   - An open agent span that still owns an open model or tool call: a
//     BeforeModelCallback or BeforeToolCallback error stops the agent
//     without any error callback.
//   - Otherwise, the most recent open agent span whose activation produced
//     no response, neither a model response nor an event with content or
//     output of its own or of an activation that ran under it, for example
//     because its Run returned an error first.
//
// The last two are reported with a synthetic error, and only if ctx was not
// ended: an agent that a BeforeAgentCallback answered ends its node, so no
// AfterAgentCallback closes its span, yet it did not fail.
//
// Failures that the callbacks cannot tell apart from a success are not
// reported, and the invocation is reported as completed:
//
//   - A run that fails after the failing activation produced a response,
//     such as a custom agent that writes an event and then returns an error,
//     or an AfterAgentCallback that returns an error. The callbacks see the
//     same as for a caller that stopped reading once it had the response.
//   - A model error of an activation that ADK still ends normally, running
//     its AfterAgentCallback, because the caller kept reading after the
//     error.
//
// Only the failing activations, each with its own error, and the open
// activations they ran under are returned; a sibling span left open by a
// shortcut did not fail. When no failing activation is known, every open span
// is returned if the invocation was canceled, and none otherwise, so that the
// error of an activation without an open span is not attributed to others.
func (tm *traceManager) popUnclosedAgentSpans(ctx context.Context, id identity) ([]unclosedAgentSpan, error) {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgent(ctx, id, nil, nil, false)
	var cancelErr error
	if ctx != nil {
		cancelErr = orZero(ctx.Err)
	}
	if len(inv.agentStack) <= 1 {
		inv.agentStack = inv.agentStack[:1]
		return nil, cancelErr
	}

	keys := slices.Sorted(maps.Keys(inv.unrecoveredErrors))
	unionLineage := make(map[string]bool)
	spanErrors := make(map[string]error)
	var primaryErr error

	for _, k := range keys {
		e := inv.unrecoveredErrors[k]
		if e.err == nil {
			continue
		}
		if primaryErr == nil {
			primaryErr = e.err
		}
		if failingIdx := inv.activationOfLocked(e); failingIdx > 0 {
			failingRec := inv.agentStack[failingIdx]
			spanErrors[failingRec.spanID] = e.err
			unionLineage[failingRec.spanID] = true
			for ancestorID := range inv.lineageLocked(failingIdx) {
				unionLineage[ancestorID] = true
				if spanErrors[ancestorID] == nil {
					spanErrors[ancestorID] = e.err
				}
			}
		}
	}

	if primaryErr == nil {
		primaryErr = cancelErr
	}

	if primaryErr == nil && !invocationEnded(ctx) {
		failingIdx := inv.agentWithOpenCallLocked()
		if failingIdx <= 0 {
			for i := len(inv.agentStack) - 1; i > 0; i-- {
				if !inv.agentStack[i].hasResponse {
					failingIdx = i
					break
				}
			}
		}
		if failingIdx > 0 {
			failingAgentName := inv.agentStack[failingIdx].name
			primaryErr = fmt.Errorf("agent %q execution failed before completion", failingAgentName)
			failingRec := inv.agentStack[failingIdx]
			spanErrors[failingRec.spanID] = primaryErr
			unionLineage[failingRec.spanID] = true
			for ancestorID := range inv.lineageLocked(failingIdx) {
				unionLineage[ancestorID] = true
				if spanErrors[ancestorID] == nil {
					spanErrors[ancestorID] = primaryErr
				}
			}
		}
	}

	if primaryErr == nil {
		inv.agentStack = inv.agentStack[:1]
		return nil, nil
	}

	// Without a known failing activation, a canceled invocation interrupted
	// every open span; otherwise the open spans did not fail, and the error
	// is not attributed to them.
	attributeAll := len(unionLineage) == 0 && cancelErr != nil
	var popped []unclosedAgentSpan
	for i := len(inv.agentStack) - 1; i >= 1; i-- {
		rec := inv.agentStack[i]
		if !attributeAll && !unionLineage[rec.spanID] {
			continue
		}
		spanErr := spanErrors[rec.spanID]
		if spanErr == nil {
			spanErr = primaryErr
		}
		popped = append(popped, unclosedAgentSpan{
			span:      inv.rowSpan(rec, latencyOf(rec, now)),
			agentName: rec.name,
			branch:    rec.branch,
			path:      rec.path,
			err:       spanErr,
		})
	}
	inv.agentStack = inv.agentStack[:1]
	return popped, primaryErr
}

// invocationEnded reports whether ctx is an ADK context whose invocation was
// ended, for example by a BeforeAgentCallback that answered for the agent.
func invocationEnded(ctx context.Context) bool {
	ic, ok := ctx.(interface{ Ended() bool })
	return ok && orZero(ic.Ended)
}

// activationOfLocked returns the index in agentStack of the open agent
// activation that the recorded error e is attributed to, or -1.
func (inv *invocationTrace) activationOfLocked(e recordedError) int {
	if e.id.agentName == "" {
		return -1
	}
	return inv.findAgent(e.id.path, e.id.agentName, e.id.branch)
}

// agentWithOpenCallLocked returns the index in agentStack of the most recent
// open agent span that an open model or tool call runs under, or -1.
func (inv *invocationTrace) agentWithOpenCallLocked() int {
	owners := map[string]bool{}
	for _, rec := range inv.llmSpans {
		owners[rec.parentSpanID] = true
	}
	for _, rec := range inv.toolSpans {
		owners[rec.parentSpanID] = true
	}
	for i := len(inv.agentStack) - 1; i > 0; i-- {
		if owners[inv.agentStack[i].spanID] {
			return i
		}
	}
	return -1
}

// lineageLocked returns the span IDs of agentStack[i] and of the open agent
// spans that it runs under, or an empty set if i is not an agent span index.
func (inv *invocationTrace) lineageLocked(i int) map[string]bool {
	lineage := map[string]bool{}
	if i <= 0 || i >= len(inv.agentStack) {
		return lineage
	}
	byID := make(map[string]*spanRecord, len(inv.agentStack))
	for _, rec := range inv.agentStack[1:] {
		byID[rec.spanID] = rec
	}
	for rec := inv.agentStack[i]; rec != nil && !lineage[rec.spanID]; rec = byID[rec.parentSpanID] {
		lineage[rec.spanID] = true
	}
	return lineage
}

// pushLLMSpan starts the span of a model call and returns the place of its
// LLM_REQUEST row. A model call of the same agent that never ended, for
// example because a BeforeModelCallback answered it, is replaced.
func (tm *traceManager) pushLLMSpan(ctx context.Context, id identity) rowSpan {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.started = true
	inv.noteRootAgent(ctx, id, nil, nil, true)
	key := llmKey(id)
	rec := inv.newChild(inv.ownerOf(id), spanKindLLM, now)
	inv.llmSpans[key] = rec
	inv.syncOpenCallsLocked()
	delete(inv.completedModels, key)
	return inv.rowSpan(rec, nil)
}

// recordFirstToken records the arrival of a streamed chunk of generated
// content of the running model call of an agent. Only the first chunk is
// recorded; the caller decides which chunks carry generated content.
func (tm *traceManager) recordFirstToken(ctx context.Context, id identity) {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	inv.mu.Lock()
	defer inv.mu.Unlock()
	if rec := inv.llmSpans[llmKey(id)]; rec != nil && rec.firstToken.IsZero() {
		rec.firstToken = now
	}
}

// popLLMSpan ends the running model call of an agent and returns the place
// of its LLM_RESPONSE or LLM_ERROR row. With responded set, a call whose
// response was not streamed gets a time to first token equal to its total
// latency, as in adk-python. If OnModelErrorCallback already ended the call
// and a plugin recovered from the error, the subsequent AfterModelCallback
// reuses the call's span without latency. If the call is otherwise unknown,
// the row gets a span of its own and no latency.
func (tm *traceManager) popLLMSpan(ctx context.Context, id identity, responded bool) rowSpan {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgent(ctx, id, nil, nil, true)
	key := llmKey(id)
	rec := inv.llmSpans[key]
	if rec == nil {
		if responded {
			if prev := inv.completedModels[key]; prev != nil {
				delete(inv.completedModels, key)
				return inv.rowSpan(prev, nil)
			}
		}
		return inv.rowSpan(inv.newChild(inv.ownerOf(id), spanKindLLM, now), nil)
	}
	delete(inv.llmSpans, key)
	inv.syncOpenCallsLocked()
	if !responded {
		inv.completedModels[key] = rec
	} else {
		if owner := inv.ownerOf(id); owner != nil {
			owner.hasResponse = true
		}
		if rec.firstToken.IsZero() {
			rec.firstToken = now
		}
	}
	return inv.rowSpan(rec, latencyOf(rec, now))
}

// pushToolSpan starts the span of a tool call and returns the place of its
// TOOL_STARTING row.
func (tm *traceManager) pushToolSpan(ctx context.Context, id identity, key string) rowSpan {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.started = true
	inv.noteRootAgent(ctx, id, nil, nil, true)
	rec := inv.newChild(inv.ownerOf(id), spanKindTool, now)
	inv.toolSpans[key] = rec
	inv.syncOpenCallsLocked()
	delete(inv.completedTools, key)
	return inv.rowSpan(rec, nil)
}

// popToolSpan ends the span of a tool call and returns the place of its
// TOOL_COMPLETED or, with fromError set, TOOL_ERROR row.
//
// ADK runs AfterToolCallback after OnToolErrorCallback for a failed call.
// closedByError reports that OnToolErrorCallback already ended the call,
// in which case the returned place reuses its span, without latency. If
// the call is unknown, as for a call of a tool that does not exist, the
// row gets a span of its own and no latency; the agent and invocation
// spans are left alone.
func (tm *traceManager) popToolSpan(ctx context.Context, id identity, key string, fromError bool) (span rowSpan, closedByError bool) {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgent(ctx, id, nil, nil, true)
	if rec := inv.toolSpans[key]; rec != nil {
		delete(inv.toolSpans, key)
		inv.syncOpenCallsLocked()
		if fromError {
			inv.completedTools[key] = rec
		}
		return inv.rowSpan(rec, latencyOf(rec, now)), false
	}
	if !fromError {
		if rec := inv.completedTools[key]; rec != nil {
			delete(inv.completedTools, key)
			return inv.rowSpan(rec, nil), true
		}
	}
	rec := inv.newChild(inv.ownerOf(id), spanKindTool, now)
	if fromError {
		inv.completedTools[key] = rec
	}
	return inv.rowSpan(rec, nil), false
}

// eventSpan returns the place of a row that describes a session event: the
// span of the agent that wrote the event, if it is running, and otherwise
// the invocation span. Like other invocation callbacks, the event callback
// is called with the agent that the runner runs, whose name is recorded as
// the root agent.
func (tm *traceManager) eventSpan(ctx context.Context, id identity, author, branch string) rowSpan {
	inv := tm.invocation(ctx, id)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgent(ctx, id, nil, nil, false)
	if i := inv.findAgent(id.path, author, branch); i > 0 {
		return inv.rowSpan(inv.agentStack[i], nil)
	}
	return inv.rowSpan(inv.agentStack[0], nil)
}

// newChild returns a new span under parent. It is not tracked.
func (inv *invocationTrace) newChild(parent *spanRecord, kind spanKind, now time.Time) *spanRecord {
	return &spanRecord{spanID: newSpanID(), parentSpanID: parent.spanID, kind: kind, start: now}
}

// llmKey identifies the model call of an agent activation: an agent makes
// one model call at a time per node path and branch.
func llmKey(id identity) string {
	return id.path + "\x00" + id.agentName + "\x00" + id.branch
}

// toolKey identifies a tool call within its owning agent activation by
// function call ID, which ADK sets on every call, or else by tool name.
// Scoping functionCallID to the agent activation prevents non-UUID provider
// call IDs (such as "call_0") from colliding across concurrent sub-agents.
func toolKey(id identity, functionCallID, toolName string) string {
	if functionCallID != "" {
		return id.path + "\x00" + id.agentName + "\x00" + id.branch + "\x00" + functionCallID
	}
	return id.path + "\x00" + id.agentName + "\x00" + id.branch + "\x00" + toolName
}

// contextParentMap returns the child-to-parent agent map that ADK's runner
// attaches to ctx via internal/agent/parentmap, or nil if ctx does not carry
// one.
func contextParentMap(ctx context.Context) (res map[string]agent.Agent) {
	defer func() {
		if r := recover(); r != nil {
			res = nil
		}
	}()
	if ctx == nil {
		return nil
	}
	queue := []reflect.Value{reflect.ValueOf(ctx)}
	seen := map[visitedContextStruct]bool{}
	for steps := 0; len(queue) > 0 && steps < 512; steps++ {
		v := queue[0]
		queue = queue[1:]
		var ptr uintptr
		for v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface {
			if v.IsNil() {
				v = reflect.Value{}
				break
			}
			if v.Kind() == reflect.Pointer {
				ptr = v.Pointer()
			}
			v = v.Elem()
		}
		if !v.IsValid() || v.Kind() != reflect.Struct {
			continue
		}
		if ptr != 0 {
			visitKey := visitedContextStruct{ptr: ptr, typ: v.Type()}
			if seen[visitKey] {
				continue
			}
			seen[visitKey] = true
		}
		if key, val := v.FieldByName("key"), v.FieldByName("val"); key.IsValid() && val.IsValid() {
			if val.Kind() == reflect.Interface && !val.IsNil() {
				val = val.Elem()
			}
			if val.IsValid() && val.Kind() == reflect.Map &&
				val.Type().Key().Kind() == reflect.String &&
				val.Type().Elem() == agentInterfaceType {
				if keyVal, ok := exportContextKey(key); ok {
					lookupCtx := ctx
					if v.CanAddr() && v.Addr().CanInterface() {
						if c, ok := v.Addr().Interface().(context.Context); ok {
							lookupCtx = c
						}
					}
					if raw := orZero(func() any { return lookupCtx.Value(keyVal) }); raw != nil {
						if m := reflect.ValueOf(raw); m.IsValid() && m.Kind() == reflect.Map && m.CanInterface() {
							out := make(map[string]agent.Agent, m.Len())
							iter := m.MapRange()
							for iter.Next() {
								if a, ok := iter.Value().Interface().(agent.Agent); ok && a != nil {
									out[iter.Key().String()] = a
								}
							}
							return out
						}
					}
				}
			}
		}
		for _, field := range []string{"context", "Context", "ctx", "invocationContext"} {
			if next := v.FieldByName(field); next.IsValid() {
				queue = append(queue, next)
			}
		}
	}
	return nil
}

func exportContextKey(key reflect.Value) (any, bool) {
	if key.Kind() == reflect.Interface && !key.IsNil() {
		key = key.Elem()
	}
	if !key.IsValid() {
		return nil, false
	}
	if key.CanInterface() {
		return key.Interface(), true
	}
	cp := reflect.New(key.Type()).Elem()
	switch key.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		cp.SetInt(key.Int())
		return cp.Interface(), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		cp.SetUint(key.Uint())
		return cp.Interface(), true
	case reflect.String:
		cp.SetString(key.String())
		return cp.Interface(), true
	case reflect.Bool:
		cp.SetBool(key.Bool())
		return cp.Interface(), true
	case reflect.Struct:
		if key.NumField() == 0 {
			return cp.Interface(), true
		}
	}
	return nil, false
}

// latencyOf returns the latency_ms object of a row that ends rec at end:
// the total latency and, for model calls, the time to first token.
func latencyOf(rec *spanRecord, end time.Time) map[string]any {
	latency := map[string]any{"total_ms": end.Sub(rec.start).Milliseconds()}
	if !rec.firstToken.IsZero() {
		latency["time_to_first_token_ms"] = rec.firstToken.Sub(rec.start).Milliseconds()
	}
	return latency
}

// newTraceID returns a random trace ID of 32 hex digits, the format of
// OpenTelemetry trace IDs and of adk-python's uuid4().hex.
func newTraceID() string {
	id := uuid.New()
	return hex.EncodeToString(id[:])
}

// newSpanID returns a random span ID of 16 hex digits, the format of
// OpenTelemetry span IDs and of adk-python's uuid4().hex[:16].
func newSpanID() string {
	id := uuid.New()
	return hex.EncodeToString(id[:8])
}
