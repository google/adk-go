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
	"container/list"
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
	// streamed reports that a chunk of the streamed response of a model call
	// has arrived. The call is making progress, so the agent that owns it did
	// not fail even if its caller stops reading before the final response.
	streamed bool
	// hasResponse reports whether the agent activation has produced a model
	// response or a final event with content or a typed output, itself or
	// through an activation that ran under it. Such an event of an
	// activation also settles the activations that still run under it in the
	// event's own branch and node path, even if one of them fails later, and
	// an event of an author that has no open span settles the activation
	// that it would have run under.
	//
	// hasResponse alone does not retire an earlier activation on
	// re-entry (see pushAgentSpan): a transfer-back chain sets it on the
	// parent while the parent is still running, so retiring on hasResponse
	// would end a live span. Instead, taskDelegated marks that a child
	// task-mode sub-agent completed via finish_task, which is the only case
	// where re-entry retirement is safe.
	hasResponse bool
	// taskDelegated is set on the delegating agent activation when its child
	// task-mode sub-agent completes via a finish_task function response
	// (see popTaskAgentSpan). It distinguishes the task-mode delegation
	// case — where dispatch interrupts the delegating chat activation and
	// that prior activation must be retired on re-entry — from an
	// ordinary transfer-back chain, where the parent is still running and
	// must not be retired. Only an LLM agent (nonLLM == false) whose
	// configured sub-agents include the task sub-agent can delegate via
	// runChat; a workflow agent (nonLLM == true) runs its own
	// AfterAgentCallback, and an ancestor LLM agent under which a resumed
	// nested task was parented is still active in its own runChat loop.
	taskDelegated bool
	// nonLLM reports that the agent is a workflow agent, such as a sequential
	// or loop agent, rather than an llmagent whose sub-agents can transfer to
	// peers.
	nonLLM bool
	// children holds the names of the sub-agents configured on this agent
	// activation.
	children []string
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
	// pendingSpans protects late state between lookup and registering a span.
	pendingSpans atomic.Int64
	// rootAgentAtomic holds the latest rootAgentName string for lock-free
	// reads during eviction.
	rootAgentAtomic atomic.Value
	// ancestryFromContextAtomic reports whether ancestryFromContext has been
	// set, so callers can skip walking ctx for a parent map before taking mu.
	ancestryFromContextAtomic atomic.Bool
	// adoptedTraceID holds the trace ID of the first OpenTelemetry span that a
	// callback of the invocation ran in, when the invocation's own traceID was
	// not taken from a span: ADK opens its first span after the callbacks that
	// precede the agents, so the trace is only known from then on. Rows of
	// callbacks that run in no span use it. Eviction reads it without holding
	// mu.
	adoptedTraceID atomic.Value

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
	runnerAgent     agent.Agent
	runnerAgentName string
	// parentAgent maps each sub-agent name to its parent agent name in the
	// configured agent tree.
	parentAgent map[string]string
	// sequentialParent maps each sub-agent name to its parent agent name when
	// that parent is a non-LLM workflow agent (such as a sequential or loop
	// agent) rather than an llmagent whose sub-agents can transfer to peers.
	sequentialParent map[string]string
	// ancestryFromContext reports that the agent hierarchy was read from the
	// parent map that the runner attaches to the callback context. That map
	// describes the tree of the runner that runs this invocation, so it
	// stays authoritative even when it is empty, because the runner runs an
	// agent that has no parent: the ancestry that the traceManager learned
	// from other runners, which may have run the same agent instance as a
	// sub-agent, does not replace it.
	ancestryFromContext bool
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
	// with the span ID of the agent activation that reported each of them.
	// An error stays recorded until a response to the failed call, or a final
	// event of its activation, clears it. The end of its activation does not:
	// once ADK has yielded the error, the invocation failed, whatever the
	// caller does next, and ADK still ends the activations above the failed
	// one normally when the caller keeps reading.
	unrecoveredErrors map[string]recordedError
	// lastEventPartial reports whether the most recent session event observed
	// by onEvent was a streamed partial chunk. When the caller stops reading
	// mid-stream on a partial event and no error was recorded, in-flight
	// sibling calls and unresponded activations are not treated as failures.
	lastEventPartial bool
}

// recordedError is an unrecovered error of an invocation and the span ID of
// the agent activation that was active when it was recorded.
type recordedError struct {
	err    error
	spanID string
}

func newInvocationTraceWithSpan(sc trace.SpanContext, traceID string, now time.Time) *invocationTrace {
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
	// Continue the caller's OpenTelemetry trace, if any, with an invocation
	// span that is a child of the caller's span. adk-python reuses the ambient
	// span ID as the invocation span ID, but ADK Go's runner opens no
	// per-invocation OpenTelemetry span before it invokes plugin callbacks,
	// so a caller's long-lived request or server span would be reused as the
	// span_id of every invocation that it serves. A child invocation span
	// keeps the span trees of those invocations apart.
	//
	// Only the trace ID follows OpenTelemetry (see rowSpan), so that the rows
	// keep joining the spans that ADK records for the same run. The span_id
	// and parent_span_id columns are the plugin's own execution tree, as in
	// adk-python, whose rows take them from the plugin's own span stack: the
	// spans that ADK opens are never written to the table, so parents taken
	// from them would point at spans that no row has. The ambient span IDs
	// are only written, to attributes.otel, when EnableOTelCorrelation is
	// set, which it is not by default.
	if sc.IsValid() {
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

func (inv *invocationTrace) hasOpenLateSpans() bool {
	inv.mu.Lock()
	defer inv.mu.Unlock()
	return inv.pendingSpans.Load() > 0 || inv.hasOpenCalls() || len(inv.agentStack) > 1
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
// inv was previously evicted or completed, before it is returned to a callback.
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

// ambientSpanContext returns the OpenTelemetry SpanContext that ctx carries,
// or the zero SpanContext if it carries none.
func ambientSpanContext(ctx context.Context) trace.SpanContext {
	if ctx == nil {
		return trace.SpanContext{}
	}
	return orZero(func() trace.SpanContext { return trace.SpanContextFromContext(ctx) })
}

// ambientTraceID returns the trace ID of the OpenTelemetry span that ctx
// carries, or "" if it carries none.
func ambientTraceID(ctx context.Context) string {
	sc := ambientSpanContext(ctx)
	if !sc.IsValid() {
		return ""
	}
	return sc.TraceID().String()
}

// rowSpan returns the place of a row that describes rec. latency is set on
// rows that end rec. ambient is the trace ID of the OpenTelemetry span that
// the callback writing the row runs in, or "" if it runs in none.
//
// The ambient trace ID wins on every row, as in adk-python, so that the rows
// of an invocation keep joining the spans that ADK records for it by trace
// ID. See rowTraceID for the rows of callbacks that run in no span.
func (inv *invocationTrace) rowSpan(rec *spanRecord, latency map[string]any, ambient string) rowSpan {
	return rowSpan{
		traceID:       inv.rowTraceID(ambient),
		spanID:        rec.spanID,
		parentSpanID:  rec.parentSpanID,
		latency:       latency,
		rootAgentName: inv.rootAgentName,
	}
}

// rowTraceID returns the trace ID of a row that a callback writes, when that
// callback runs in the OpenTelemetry span of the trace ambient, or in no span
// if ambient is empty.
//
// The runner opens no span before the callbacks that precede the agents, and
// ADK's own spans start with the agents', so the invocation has a trace ID of
// its own until a callback runs in a span. The trace of the first such span
// is adopted then, unless the caller's span already gave the invocation its
// trace: the rows of callbacks that run in no span, such as the session
// events and the end of the invocation, join the trace of the spans around
// them instead of a trace that has no other span. Rows written earlier keep
// the trace ID of their time.
func (inv *invocationTrace) rowTraceID(ambient string) string {
	if ambient == "" {
		return inv.currentTraceID()
	}
	if inv.rootParentSpanID == "" {
		inv.adoptedTraceID.CompareAndSwap(nil, ambient)
	}
	return ambient
}

// currentTraceID returns the trace ID of the rows of callbacks that run in no
// OpenTelemetry span. It does not need mu.
func (inv *invocationTrace) currentTraceID() string {
	if adopted, ok := inv.adoptedTraceID.Load().(string); ok {
		return adopted
	}
	return inv.traceID
}

// agentEdge is a child-to-parent link in an agent sub-agent tree.
type agentEdge struct {
	child, parent string
	nonLLM        bool
}

// rootAgentInput holds the agent-tree metadata extracted from ctx, runner, and
// rememberedRoot before acquiring inv.mu, so user agent code and context
// lookups never run under a lock.
type rootAgentInput struct {
	hasContextParents bool
	contextEdges      []agentEdge
	contextRoot       agent.Agent
	contextRootName   string
	runner            agent.Agent
	runnerName        string
	runnerEdges       []agentEdge
	remembered        agent.Agent
	rememberedName    string
	rememberedEdges   []agentEdge
}

func walkAgentEdges(root agent.Agent) []agentEdge {
	if root == nil {
		return nil
	}
	var edges []agentEdge
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
				edges = append(edges, agentEdge{child: subName, parent: parentName, nonLLM: nonLLM})
				walk(sub)
			}
		}
	}
	walk(root)
	return edges
}

func prepareRootAgentInput(ctx context.Context, inv *invocationTrace, id identity, runner, rememberedRoot agent.Agent) rootAgentInput {
	var in rootAgentInput
	if !inv.ancestryFromContextAtomic.Load() {
		if parents := orZero(func() map[string]agent.Agent { return contextParentMap(ctx) }); parents != nil {
			in.hasContextParents = true
			for child, parent := range parents {
				if pName := orZero(parent.Name); child != "" && pName != "" {
					in.contextEdges = append(in.contextEdges, agentEdge{
						child:  child,
						parent: pName,
						nonLLM: !isLLMAgent(parent),
					})
				}
				in.contextEdges = append(in.contextEdges, walkAgentEdges(parent)...)
			}
			if root := topAgentFromMap(runner, id.agentName, parents); root != nil {
				in.contextRoot = root
				in.contextRootName = orZero(root.Name)
			}
		}
	}
	if runner != nil {
		in.runner = runner
		in.runnerName = orZero(runner.Name)
		in.runnerEdges = walkAgentEdges(runner)
		if rememberedRoot != nil && rememberedRoot != runner {
			in.remembered = rememberedRoot
			in.rememberedName = orZero(rememberedRoot.Name)
			in.rememberedEdges = walkAgentEdges(rememberedRoot)
		}
	}
	return in
}

func (inv *invocationTrace) applyEdgesLocked(edges []agentEdge) {
	for _, e := range edges {
		inv.parentAgent[e.child] = e.parent
		if e.nonLLM {
			inv.sequentialParent[e.child] = e.parent
		}
	}
}

// noteRootAgentLocked records the invocation's root agent and agent hierarchy
// from in, which was collected before acquiring inv.mu.
//
// The parent map that the runner attaches to ctx describes the tree of the
// runner of this invocation, so it is authoritative when ctx carries one,
// even an empty map: the runner then runs an agent that has no parent.
// remembered comes from the trees of earlier runners, which may have run
// the same agent instance as a sub-agent, and is used only when ctx carries no
// parent map.
func (inv *invocationTrace) noteRootAgentLocked(id identity, in rootAgentInput, useBranchFallback bool) {
	if !inv.ancestryFromContext && in.hasContextParents {
		inv.ancestryFromContext = true
		inv.ancestryFromContextAtomic.Store(true)
		inv.applyEdgesLocked(in.contextEdges)
		if in.contextRoot != nil {
			inv.runnerAgent = in.contextRoot
			inv.runnerAgentName = in.contextRootName
			if in.contextRootName != "" {
				inv.configuredRoot = in.contextRootName
				inv.setRootAgentLocked(in.contextRootName)
			}
		}
	}
	if in.runner != nil {
		inv.applyEdgesLocked(in.runnerEdges)
		if in.remembered != nil && in.remembered != in.runner && !inv.ancestryFromContext {
			inv.applyEdgesLocked(in.rememberedEdges)
			if inv.runnerAgent == nil {
				inv.runnerAgent = in.remembered
				inv.runnerAgentName = in.rememberedName
			}
			if in.rememberedName != "" && inv.configuredRoot == "" {
				inv.configuredRoot = in.rememberedName
				inv.setRootAgentLocked(in.rememberedName)
			}
		}
		if inv.runnerAgent == nil || inv.isAncestorAgent(in.runnerName, inv.runnerAgentName) {
			inv.runnerAgent = in.runner
			inv.runnerAgentName = in.runnerName
		}
		if in.runnerName != "" && inv.isAncestorAgent(in.runnerName, cmp.Or(inv.configuredRoot, inv.rootAgentName, id.agentName)) {
			inv.configuredRoot = in.runnerName
			inv.setRootAgentLocked(in.runnerName)
		}
	}
	if inv.configuredRoot != "" {
		inv.setRootAgentLocked(inv.configuredRoot)
		return
	}
	if inv.rootAgentName == "" {
		switch {
		case in.runner != nil:
			inv.setRootAgentLocked(cmp.Or(id.agentName, in.runnerName))
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

// agentChildren are the sub-agents that are configured on a running agent.
type agentChildren struct {
	// parent is the name of the running agent.
	parent string
	// nonLLM reports that parent is a workflow agent, such as a sequential or
	// loop agent, rather than an llmagent whose sub-agents can transfer to
	// peers.
	nonLLM bool
	// names holds the names of the sub-agents.
	names []string
}

// runningAgentChildren returns the sub-agents that are configured on the agent
// that ctx runs, if that agent is named name. Agents that run as nodes of a
// workflow graph need not be in the sub-agent tree of the agent that the runner
// runs, so the agent hierarchy that the runner attaches to ctx can lack them.
// It is the zero value if ctx does not expose its agent. The callback context
// logs a warning from Agent, so the agent is read from the invocation context
// that WithICDelta derives from it. SubAgents and Name are user code, so
// callers must not hold a lock.
func runningAgentChildren(ctx context.Context, name string) agentChildren {
	return orZero(func() agentChildren {
		c, ok := ctx.(interface {
			WithICDelta(*agent.InvocationContextDelta) agent.InvocationContext
		})
		if !ok || name == "" {
			return agentChildren{}
		}
		ic := c.WithICDelta(&agent.InvocationContextDelta{})
		if ic == nil {
			return agentChildren{}
		}
		a := ic.Agent()
		if a == nil || a.Name() != name {
			return agentChildren{}
		}
		children := agentChildren{parent: name, nonLLM: !isLLMAgent(a)}
		for _, sub := range a.SubAgents() {
			if sub == nil {
				continue
			}
			if subName := sub.Name(); subName != "" {
				children.names = append(children.names, subName)
			}
		}
		return children
	})
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
// invocation. When a running activation on agentStack declares id.agentName as
// a child, a non-LLM workflow parent skips open sibling spans without mutating
// agentStack on push, while an LLM parent keeps peer transfers under the active
// peer. When no running activation declares id.agentName, parentFor falls back
// to the runner's configured agent tree in inv.sequentialParent and
// inv.parentAgent.
func (inv *invocationTrace) parentFor(id identity) *spanRecord {
	var topCandidate *spanRecord
	for i := len(inv.agentStack) - 1; i > 0; i-- {
		rec := inv.agentStack[i]
		if !withinBranch(id.branch, rec.branch) || !withinPath(id.path, rec.path, rec.name == id.agentName) {
			continue
		}
		if topCandidate == nil {
			topCandidate = rec
		}
		if slices.Contains(rec.children, id.agentName) {
			if rec.nonLLM {
				return rec
			}
			return topCandidate
		}
	}
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
	mu              sync.Mutex
	invocations     map[string]*invocationTrace
	evicted         map[string]evictedTrace
	completed       map[string]*completedTrace
	completionOrder list.List
	knownParent     map[agent.Agent]agent.Agent
	traceSeed       string
	inserts         int
	logf            func(string, ...any)

	// maxTracked, ttl and now are fields so that tests can change them.
	maxTracked int
	ttl        time.Duration
	now        func() time.Time
}

// evictedTrace retains the identity of an invocation after active tracking ends.
type evictedTrace struct {
	traceID       string
	spanID        string
	parentSpanID  string
	rootAgentName string
	lastActive    int64
}

type completedTrace struct {
	identity evictedTrace
	// Late calls share state without retaining the finished run's span stacks.
	late     *invocationTrace
	position *list.Element
}

func newTraceManager() *traceManager {
	return &traceManager{
		invocations: map[string]*invocationTrace{},
		evicted:     map[string]evictedTrace{},
		completed:   map[string]*completedTrace{},
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
// get a trace that is not tracked, so that their rows still have IDs. Completed
// invocations are restored without restarting active tracking.
func (tm *traceManager) invocation(ctx context.Context, id identity) *invocationTrace {
	return tm.lookupInvocation(ctx, id, false, false)
}

// lookupInvocation starts a fresh run only when start is true. Other callbacks
// after AfterRun use the completed identity without restarting active tracking.
func (tm *traceManager) lookupInvocation(ctx context.Context, id identity, start, reserveSpan bool) *invocationTrace {
	now := tm.now()
	sc := ambientSpanContext(ctx)
	if id.invocationID == "" {
		inv := newInvocationTraceWithSpan(sc, "", now)
		if reserveSpan {
			inv.pendingSpans.Add(1)
		}
		return inv
	}
	var warnIDs []string
	var maxTracked int
	var logf func(string, ...any)
	tm.mu.Lock()
	inv, ok := tm.invocations[id.invocationID]
	if !ok {
		if completed := tm.completed[id.invocationID]; completed != nil {
			if !start {
				if completed.late == nil {
					completed.late = newInvocationTraceWithSpan(sc, tm.traceIDFor(id.invocationID), now)
					completed.late.restoreEvicted(completed.identity)
				}
				completed.late.lastActive.Store(now.UnixNano())
				tm.completionOrder.MoveToBack(completed.position)
				if reserveSpan {
					completed.late.pendingSpans.Add(1)
				}
				tm.mu.Unlock()
				return completed.late
			}
			tm.completionOrder.Remove(completed.position)
			delete(tm.completed, id.invocationID)
		}
		ev, wasEvicted := tm.evicted[id.invocationID]
		inv = newInvocationTraceWithSpan(sc, tm.traceIDFor(id.invocationID), now)
		if wasEvicted {
			inv.restoreEvicted(ev)
		}
		tm.invocations[id.invocationID] = inv
		tm.inserts++
		warnIDs = tm.evictLocked(now, id.invocationID)
		maxTracked = tm.maxTracked
		logf = tm.logf
	}
	inv.lastActive.Store(now.UnixNano())
	if reserveSpan {
		inv.pendingSpans.Add(1)
	}
	tm.mu.Unlock()
	if logf != nil {
		for _, evictedID := range warnIDs {
			logf("agentanalytics: trace manager capacity (%d) exceeded; evicting active invocation %q", maxTracked, evictedID)
		}
	}
	return inv
}

// evictLocked drops invocations idle for longer than the TTL that have no
// open model or tool calls, every sweepInterval insertions or when too many
// are tracked, and then the least recently active ones while too many are
// still tracked. keep is never dropped. It returns the IDs of active
// invocations evicted due to capacity so callers can log after releasing
// tm.mu. tm.mu must be held.
func (tm *traceManager) evictLocked(now time.Time, keep string) []string {
	cutoff := now.Add(-tm.ttl).UnixNano()
	if tm.inserts%sweepInterval == 0 || len(tm.invocations) > tm.maxTracked {
		for id, inv := range tm.invocations {
			if id != keep && inv.lastActive.Load() < cutoff && !inv.hasOpenCalls() {
				tm.markEvictedLocked(now, id, inv)
				delete(tm.invocations, id)
			}
		}
	}
	var warnIDs []string
	for len(tm.invocations) > tm.maxTracked {
		oldestID, oldest := "", int64(0)
		for id, inv := range tm.invocations {
			if t := inv.lastActive.Load(); id != keep && (oldestID == "" || t < oldest) {
				oldestID, oldest = id, t
			}
		}
		if oldestID == "" {
			break
		}
		evictedInv := tm.invocations[oldestID]
		if oldest >= cutoff || evictedInv.hasOpenCalls() {
			warnIDs = append(warnIDs, oldestID)
		}
		tm.markEvictedLocked(now, oldestID, evictedInv)
		delete(tm.invocations, oldestID)
	}
	return warnIDs
}

func (tm *traceManager) markEvictedLocked(now time.Time, invocationID string, inv *invocationTrace) {
	limit := max(tm.maxTracked, 64)
	if _, exists := tm.evicted[invocationID]; !exists && len(tm.evicted) >= limit {
		cutoff := now.Add(-tm.ttl).UnixNano()
		for id, ev := range tm.evicted {
			if ev.lastActive < cutoff {
				delete(tm.evicted, id)
			}
		}
		for len(tm.evicted) >= limit {
			oldestID, oldest := "", int64(0)
			for id, ev := range tm.evicted {
				if oldestID == "" || ev.lastActive < oldest {
					oldestID, oldest = id, ev.lastActive
				}
			}
			if oldestID == "" {
				break
			}
			delete(tm.evicted, oldestID)
		}
	}
	var ev evictedTrace
	if inv != nil {
		ev = inv.savedIdentity()
	} else {
		ev.lastActive = now.UnixNano()
	}
	tm.evicted[invocationID] = ev
}

func (inv *invocationTrace) savedIdentity() evictedTrace {
	rootAgent, _ := inv.rootAgentAtomic.Load().(string)
	return evictedTrace{
		traceID:       inv.currentTraceID(),
		spanID:        inv.rootSpanID,
		parentSpanID:  inv.rootParentSpanID,
		rootAgentName: rootAgent,
		lastActive:    inv.lastActive.Load(),
	}
}

// Completed identities have their own capacity: finishing an ordinary run must
// never evict the saved identity of a run that is still in progress. The order
// makes ordinary completion pruning constant-time; late open calls are skipped.
// tm.mu must be held.
func (tm *traceManager) rememberCompletedLocked(invocationID string, inv *invocationTrace) {
	completed := tm.completed[invocationID]
	if completed == nil {
		completed = &completedTrace{position: tm.completionOrder.PushBack(invocationID)}
		tm.completed[invocationID] = completed
	} else {
		tm.completionOrder.MoveToBack(completed.position)
	}
	completed.identity = inv.savedIdentity()
	limit := max(tm.maxTracked, 64)
	for candidate := tm.completionOrder.Front(); len(tm.completed) > limit && candidate != nil; {
		next := candidate.Next()
		id := candidate.Value.(string)
		entry := tm.completed[id]
		if entry.late == nil || !entry.late.hasOpenLateSpans() {
			delete(tm.completed, id)
			tm.completionOrder.Remove(candidate)
		}
		candidate = next
	}
}

// invocationSpan returns the place of a row written by a callback of the
// whole invocation, such as USER_MESSAGE_RECEIVED. Such callbacks get the
// agent that the runner runs, runner, which is used to record the root
// agent.
func (tm *traceManager) invocationSpan(ctx context.Context, id identity, runner agent.Agent) rowSpan {
	inv := tm.lookupInvocation(ctx, id, true, false)
	rememberedRoot := tm.rememberAgentTree(runner)
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, runner, rememberedRoot)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.started = true
	inv.noteRootAgentLocked(id, rootIn, false)
	return inv.rowSpan(inv.agentStack[0], nil, ambient)
}

// endInvocation stops tracking the invocation and returns the place of its
// INVOCATION_COMPLETED row, with the invocation latency if the invocation
// was tracked from its start without being evicted mid-run. Its identity stays
// in a separate bounded cache for late callbacks. A new run with the same ID
// discards that completed state; capacity pruning can remove idle entries earlier.
func (tm *traceManager) endInvocation(ctx context.Context, id identity) rowSpan {
	now := tm.now()
	sc := ambientSpanContext(ctx)
	ambient := ""
	if sc.IsValid() {
		ambient = sc.TraceID().String()
	}
	var inv *invocationTrace
	if id.invocationID != "" {
		tm.mu.Lock()
		inv = tm.invocations[id.invocationID]
		if inv == nil {
			inv = newInvocationTraceWithSpan(sc, tm.traceIDFor(id.invocationID), now)
			if ev, ok := tm.evicted[id.invocationID]; ok {
				inv.restoreEvicted(ev)
			} else if completed := tm.completed[id.invocationID]; completed != nil {
				inv.restoreEvicted(completed.identity)
			}
		}
		tm.mu.Unlock()
	}
	if inv == nil {
		inv = newInvocationTraceWithSpan(sc, tm.traceIDFor(id.invocationID), now)
	}
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	inv.noteRootAgentLocked(id, rootIn, false)
	var latency map[string]any
	if inv.started && !inv.evicted {
		latency = latencyOf(inv.agentStack[0], now)
	}
	row := inv.rowSpan(inv.agentStack[0], latency, ambient)
	inv.mu.Unlock()
	if id.invocationID != "" {
		tm.mu.Lock()
		// Publish the final identity, including any trace adopted by rowSpan,
		// before removing active state. Late Live callbacks may start and end
		// an operation, so their state is shared separately from the finished run.
		tm.rememberCompletedLocked(id.invocationID, inv)
		delete(tm.evicted, id.invocationID)
		delete(tm.invocations, id.invocationID)
		tm.mu.Unlock()
	}
	return row
}

// pushAgentSpan starts the span of an agent run and returns the place of
// its AGENT_STARTING row. The sub-agents that are configured on the starting
// agent are recorded on its activation span before they start, so that agents
// that run under a workflow agent outside the runner's agent tree, such as a
// graph node, are parented like those of the tree without overwriting sibling
// graph nodes. If an agent activation of the same (name, branch, path) is
// re-entered after a task-mode sub-agent completes via finish_task, the prior
// activation interrupted by task dispatch is retired first so the continuation
// turn becomes a direct sibling under the parent span. Only a direct LLM
// parent is marked; workflow agents run their own AfterAgentCallback, so
// nonLLM is defensively excluded.
func (tm *traceManager) pushAgentSpan(ctx context.Context, id identity) rowSpan {
	inv := tm.lookupInvocation(ctx, id, false, true)
	defer inv.pendingSpans.Add(-1)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	// SubAgents is user code, so it is read before mu is taken.
	children := runningAgentChildren(ctx, id.agentName)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.started = true
	inv.noteRootAgentLocked(id, rootIn, true)
	for i := len(inv.agentStack) - 1; i > 0; i-- {
		rec := inv.agentStack[i]
		// Retire the prior activation only when it delegated to a completed
		// task-mode sub-agent (taskDelegated). An ordinary transfer-back
		// chain (planner → helper → planner) leaves the first planner with
		// hasResponse == true (its transfer call counted as a response)
		// while it is still running, so retiring on hasResponse would pop
		// the planner and its descendants prematurely.
		if rec.kind == spanKindAgent && rec.name == id.agentName && rec.branch == id.branch && rec.path == id.path && rec.taskDelegated {
			inv.retireAgentLocked(i)
			break
		}
	}
	rec := &spanRecord{
		spanID:       newSpanID(),
		parentSpanID: inv.parentFor(id).spanID,
		kind:         spanKindAgent,
		path:         id.path,
		name:         id.agentName,
		branch:       id.branch,
		start:        now,
		nonLLM:       children.nonLLM,
		children:     children.names,
	}
	inv.agentStack = append(inv.agentStack, rec)
	return inv.rowSpan(rec, nil, ambient)
}

// agentNamed returns the agent of id, looked up by name in the tree of the
// agent that the runner runs, or nil if it is not found there.
func (tm *traceManager) agentNamed(ctx context.Context, id identity) agent.Agent {
	inv := tm.invocation(ctx, id)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	inv.noteRootAgentLocked(id, rootIn, true)
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
// The activations that still run under the agent end with it: a sub-agent
// that a BeforeAgentCallback answered never gets an AfterAgentCallback of its
// own. A failed activation does not: one that has an unrecovered error
// recorded, or that owns a model or tool call that never ended, stays open,
// and so does the error, so that the end of the invocation reports the
// failure. ADK ends the agents above a failed sub-agent normally when the
// caller keeps reading after it yielded the error, and whether the
// invocation failed must not depend on how the caller iterates.
func (tm *traceManager) popAgentSpan(ctx context.Context, id identity) rowSpan {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgentLocked(id, rootIn, true)
	i := inv.findAgent(id.path, id.agentName, id.branch)
	if i < 0 {
		return inv.rowSpan(inv.newChild(inv.parentFor(id), spanKindAgent, now), nil, ambient)
	}
	rec := inv.agentStack[i]
	inv.retireAgentLocked(i)
	return inv.rowSpan(rec, latencyOf(rec, now), ambient)
}

// popTaskAgentSpan ends the span of a task-mode sub-agent that completed via
// a finish_task function response (which ADK does without running
// AfterAgentCallback), and marks its delegating agent activation as having
// delegated to a completed task. The mark lets pushAgentSpan retire the
// parent's prior activation safely on re-entry, without retiring a parent
// that is still running in an ordinary transfer-back chain. Only a direct
// LLM parent (nonLLM == false) whose configured sub-agents include the task
// sub-agent is marked; a workflow agent (nonLLM == true) runs its own
// AfterAgentCallback, and an ancestor LLM agent under which a resumed nested
// task was parented is still active in its own runChat loop.
func (tm *traceManager) popTaskAgentSpan(ctx context.Context, id identity) rowSpan {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgentLocked(id, rootIn, true)
	i := inv.findAgent(id.path, id.agentName, id.branch)
	if i < 0 {
		return inv.rowSpan(inv.newChild(inv.parentFor(id), spanKindAgent, now), nil, ambient)
	}
	rec := inv.agentStack[i]
	// A resumed task can start before its delegating chat's callback, placing
	// its span under a still-running ancestor instead. Only mark a parent
	// activation that actually has this task configured as its child.
	// Non-LLM workflow agents run their own AfterAgentCallback, so nonLLM is
	// defensively excluded.
	for j := range inv.agentStack {
		if inv.agentStack[j].spanID == rec.parentSpanID &&
			inv.agentStack[j].kind == spanKindAgent &&
			!inv.agentStack[j].nonLLM &&
			slices.Contains(inv.agentStack[j].children, rec.name) {
			inv.agentStack[j].taskDelegated = true
			break
		}
	}
	inv.retireAgentLocked(i)
	return inv.rowSpan(rec, latencyOf(rec, now), ambient)
}

// retireAgentLocked removes agentStack[i] and any unclosed descendant spans
// whose parentSpanID lineage reaches agentStack[i].spanID, except failed
// activations and the open spans they run under.
func (inv *invocationTrace) retireAgentLocked(i int) {
	rec := inv.agentStack[i]
	// The failed activations and the open spans they run under are attributed
	// while the spans are still open.
	failed := inv.openCallOwnersLocked()
	for j := range inv.agentStack {
		if j > 0 && failed[inv.agentStack[j].spanID] {
			maps.Copy(failed, inv.lineageLocked(j))
		}
	}
	for _, e := range inv.unrecoveredErrors {
		if j := inv.activationOfLocked(e); j > 0 {
			maps.Copy(failed, inv.lineageLocked(j))
		}
	}
	inv.agentStack = slices.Delete(inv.agentStack, i, i+1)
	// Also retire any unclosed descendant spans whose parentSpanID lineage
	// reaches rec.spanID (for example, a sub-agent short-circuited in
	// BeforeAgentCallback that never ran AfterAgentCallback), except the
	// failed ones and what runs under them.
	retired := map[string]bool{rec.spanID: true}
	for idx := i; idx < len(inv.agentStack); {
		if d := inv.agentStack[idx]; retired[d.parentSpanID] && !failed[d.spanID] {
			retired[d.spanID] = true
			inv.agentStack = slices.Delete(inv.agentStack, idx, idx+1)
			continue
		}
		idx++
	}
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
// event with content or a typed output responded. The
// author's activation is looked up from the event's author, branch and node
// path in id, as in eventSpan, because the event callback gets the agent that
// the runner runs rather than the author. calls holds the function call ID, or
// else the tool name, of each function response of the event.
//
// If the author has an open agent span, the event settles the activation and
// the open activations that run under it inside the event's own branch and node
// path. An activation on another branch or in another node path is separate
// work that the event says nothing about, for example another node of a
// parallel workflow whose function node writes its events under the workflow
// agent's name, so it stays as it is. The activation and the open activations
// that it runs under are marked as responded, and so are the settled ones. For
// each settled activation:
//
//   - The model calls still open under it end without a row: ADK runs no
//     AfterModelCallback for a call that a BeforeModelCallback answered, so a
//     response event of its agent, or of an agent above it, is all that ends
//     the call.
//   - The unrecovered errors recorded for the author's own activation end with
//     them: it went on to respond, for example after a plugin recovered from a
//     model error and an earlier plugin's AfterModelCallback answered the call,
//     which stops the chain before this plugin's AfterModelCallback clears the
//     error. Unrecovered errors of activations below it stay recorded, because
//     a parent event does not mean that a failed sub-agent recovered.
//   - The tool calls still open under it end without a row in the same way: an
//     earlier plugin's AfterToolCallback that returns a result stops the chain
//     before this plugin's, so a response event of its agent, or of an agent
//     above it, is all that ends them. Only the open tool calls of the author
//     itself are matched against calls, because they alone are answered by the
//     function responses of the event: the others stay open, because they have
//     not responded.
//
// A sub-agent that fails later, without any callback seeing the failure, is
// settled as well, so its failure is not reported: popUnclosedAgentSpans
// documents this trade-off. Errors recorded after the event still count.
//
// If the author has no open agent span, for example because a callback of an
// earlier plugin answered for it before this plugin saw it start, the response
// arrived under the activation that the author would have run under. That
// activation and the open activations that it runs under are marked as
// responded. Nothing else changes.
func (tm *traceManager) completeResponse(ctx context.Context, id identity, author, branch string, calls []string) {
	inv := tm.invocation(ctx, id)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.lastEventPartial = false
	i := inv.findAgent(id.path, author, branch)
	if i <= 0 {
		if author == "" || author == userAuthor {
			return
		}
		parent := inv.parentFor(identity{agentName: author, branch: branch, path: id.path})
		if j := slices.Index(inv.agentStack, parent); j > 0 {
			inv.markRespondedLocked(inv.lineageLocked(j))
		}
		return
	}
	owner := inv.agentStack[i]
	// An activation is created after the one that it runs under, so one pass
	// over the later spans finds all that run under the owner.
	under := map[string]bool{owner.spanID: true}
	for _, rec := range inv.agentStack[i+1:] {
		if under[rec.parentSpanID] && withinBranch(rec.branch, branch) && withinPath(rec.path, id.path, false) {
			under[rec.spanID] = true
		}
	}
	for key, e := range inv.unrecoveredErrors {
		if inv.activationOfLocked(e) == i {
			delete(inv.unrecoveredErrors, key)
		}
	}
	for key, rec := range inv.llmSpans {
		if rec.parentSpanID == owner.spanID {
			delete(inv.llmSpans, key)
		}
	}
	for key, rec := range inv.toolSpans {
		if rec.parentSpanID == owner.spanID && slices.Contains(calls, toolKeyCall(key)) {
			delete(inv.toolSpans, key)
		}
	}
	inv.syncOpenCallsLocked()
	inv.markRespondedLocked(inv.lineageLocked(i))
	inv.markRespondedLocked(under)
}

// markRespondedLocked marks the open agent spans in spans, by span ID, as
// responded.
func (inv *invocationTrace) markRespondedLocked(spans map[string]bool) {
	for _, rec := range inv.agentStack[1:] {
		if spans[rec.spanID] {
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
	var spanID string
	if i := inv.findAgent(id.path, id.agentName, id.branch); i > 0 {
		spanID = inv.agentStack[i].spanID
	}
	inv.unrecoveredErrors[key] = recordedError{err: err, spanID: spanID}
}

// clearError clears any recorded unrecovered error for key on the invocation
// trace when a plugin recovered from the error on the same model call
// (completedModels[key] != nil). A later call on a new activation, such as a
// retry in a loop agent, does not erase an earlier unrecovered error.
func (tm *traceManager) clearError(ctx context.Context, id identity, key string) {
	inv := tm.invocation(ctx, id)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	if inv.completedModels[key] != nil {
		delete(inv.unrecoveredErrors, key)
	}
}

// popUnclosedAgentSpans ends the invocation's agent spans that are still open
// when the invocation terminates and returns the invocation's terminal error,
// if any, together with the open spans that the error is attributed to.
//
// Plugin callbacks do not observe every failure, so the terminal error covers
// only these detectable ones, in order of precedence:
//
//   - An unrecovered model error that a callback recorded. A response to the
//     failed model call, after a plugin recovered from the error, clears the
//     error, and so does a final event of its activation.
//   - The error of ctx when the invocation was canceled. Cancellation is
//     returned even when every agent span is closed, or when no agent ran at
//     all, for example because a BeforeRunCallback canceled the invocation.
//   - An open agent span that still owns an open model or tool call: a
//     BeforeModelCallback or BeforeToolCallback error stops the agent
//     without any error callback.
//   - Otherwise, the most recent open agent span whose activation produced
//     no response, neither a model response nor an event with content or
//     output of its own, of an activation that ran under it or of one that it
//     ran under, for example because its Run returned an error first.
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
//   - A sub-agent that fails without any callback after an agent that it
//     runs under produced content, such as a custom agent that writes an
//     event while its sub-agent runs. The event settles the activations that
//     still run under the agent, so the callbacks cannot tell the failed
//     sub-agent apart from one whose caller stopped reading once it had the
//     response.
//
// Only the failing activations, each with its own error, and the open
// activations they ran under are returned; a sibling span left open by a
// shortcut did not fail. When no failing activation is known, every open span
// is returned if the invocation was canceled, and none otherwise, so that the
// error of an activation without an open span is not attributed to others.
func (tm *traceManager) popUnclosedAgentSpans(ctx context.Context, id identity) ([]unclosedAgentSpan, error) {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgentLocked(id, rootIn, false)
	var cancelErr error
	if ctx != nil {
		cancelErr = orZero(ctx.Err)
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

	if len(inv.agentStack) <= 1 {
		inv.agentStack = inv.agentStack[:1]
		return nil, primaryErr
	}

	if primaryErr == nil && !invocationEnded(ctx) && !inv.lastEventPartial {
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
			span:      inv.rowSpan(rec, latencyOf(rec, now), ambient),
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
	if e.spanID == "" {
		return -1
	}
	for i := len(inv.agentStack) - 1; i > 0; i-- {
		if inv.agentStack[i].spanID == e.spanID {
			return i
		}
	}
	return -1
}

// openCallOwnersLocked returns the set of parent span IDs that currently own
// an open model or tool call. A model call that has already delivered a
// streamed chunk of generated content is excluded, because a caller that stops
// reading mid-stream leaves the streaming call open without failing.
func (inv *invocationTrace) openCallOwnersLocked() map[string]bool {
	owners := map[string]bool{}
	for _, rec := range inv.llmSpans {
		if !rec.streamed {
			owners[rec.parentSpanID] = true
		}
	}
	for _, rec := range inv.toolSpans {
		owners[rec.parentSpanID] = true
	}
	return owners
}

// agentWithOpenCallLocked returns the index in agentStack of the most recent
// open agent span that an open model or tool call runs under, or -1.
func (inv *invocationTrace) agentWithOpenCallLocked() int {
	owners := inv.openCallOwnersLocked()
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
	inv := tm.lookupInvocation(ctx, id, false, true)
	defer inv.pendingSpans.Add(-1)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.started = true
	inv.noteRootAgentLocked(id, rootIn, true)
	key := llmKey(id)
	rec := inv.newChild(inv.ownerOf(id), spanKindLLM, now)
	inv.llmSpans[key] = rec
	inv.syncOpenCallsLocked()
	delete(inv.completedModels, key)
	return inv.rowSpan(rec, nil, ambient)
}

// recordFirstToken records the arrival of a streamed chunk of generated
// content of the running model call of an agent. Only the first chunk's
// timestamp is recorded; every streamed chunk with generated content also marks
// the call and its owning agent lineage as having responded, so a caller that
// stops reading mid-stream is not treated as an unclosed-call failure.
func (tm *traceManager) recordFirstToken(ctx context.Context, id identity) {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	inv.mu.Lock()
	defer inv.mu.Unlock()
	rec := inv.llmSpans[llmKey(id)]
	if rec == nil {
		return
	}
	rec.streamed = true
	if rec.firstToken.IsZero() {
		rec.firstToken = now
	}
	if i := inv.findAgent(id.path, id.agentName, id.branch); i > 0 {
		inv.markRespondedLocked(inv.lineageLocked(i))
	}
}

// recordStreamed marks the running model call of author (if any) and its
// owning agent lineage as having delivered streamed output.
func (tm *traceManager) recordStreamed(ctx context.Context, id identity, author, branch string) {
	inv := tm.invocation(ctx, id)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.lastEventPartial = true
	callID := identity{agentName: author, branch: branch, path: id.path}
	if rec := inv.llmSpans[llmKey(callID)]; rec != nil {
		rec.streamed = true
	}
	if i := inv.findAgent(id.path, author, branch); i > 0 {
		inv.markRespondedLocked(inv.lineageLocked(i))
	}
}

// clearLastPartial clears the lastEventPartial flag when a non-partial session
// event arrives.
func (tm *traceManager) clearLastPartial(ctx context.Context, id identity) {
	inv := tm.invocation(ctx, id)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.lastEventPartial = false
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
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgentLocked(id, rootIn, true)
	key := llmKey(id)
	rec := inv.llmSpans[key]
	if rec == nil {
		if responded {
			if prev := inv.completedModels[key]; prev != nil {
				delete(inv.completedModels, key)
				return inv.rowSpan(prev, nil, ambient)
			}
		}
		return inv.rowSpan(inv.newChild(inv.ownerOf(id), spanKindLLM, now), nil, ambient)
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
	return inv.rowSpan(rec, latencyOf(rec, now), ambient)
}

// pushToolSpan starts the span of a tool call and returns the place of its
// TOOL_STARTING row.
func (tm *traceManager) pushToolSpan(ctx context.Context, id identity, key string) rowSpan {
	inv := tm.lookupInvocation(ctx, id, false, true)
	defer inv.pendingSpans.Add(-1)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.started = true
	inv.noteRootAgentLocked(id, rootIn, true)
	rec := inv.newChild(inv.ownerOf(id), spanKindTool, now)
	inv.toolSpans[key] = rec
	inv.syncOpenCallsLocked()
	delete(inv.completedTools, key)
	return inv.rowSpan(rec, nil, ambient)
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
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgentLocked(id, rootIn, true)
	if rec := inv.toolSpans[key]; rec != nil {
		delete(inv.toolSpans, key)
		inv.syncOpenCallsLocked()
		if fromError {
			inv.completedTools[key] = rec
		}
		return inv.rowSpan(rec, latencyOf(rec, now), ambient), false
	}
	if !fromError {
		if rec := inv.completedTools[key]; rec != nil {
			delete(inv.completedTools, key)
			return inv.rowSpan(rec, nil, ambient), true
		}
	}
	rec := inv.newChild(inv.ownerOf(id), spanKindTool, now)
	if fromError {
		inv.completedTools[key] = rec
	}
	return inv.rowSpan(rec, nil, ambient), false
}

// eventSpan returns the place of a row that describes a session event: the
// span of the agent that wrote the event, if it is running, and otherwise
// the invocation span. Like other invocation callbacks, the event callback
// is called with the agent that the runner runs, whose name is recorded as
// the root agent.
func (tm *traceManager) eventSpan(ctx context.Context, id identity, author, branch string) rowSpan {
	inv := tm.invocation(ctx, id)
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgentLocked(id, rootIn, false)
	if i := inv.findAgent(id.path, author, branch); i > 0 {
		return inv.rowSpan(inv.agentStack[i], nil, ambient)
	}
	return inv.rowSpan(inv.agentStack[0], nil, ambient)
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

// toolKeyCall returns the function call ID, or else the tool name, that
// toolKey put at the end of key.
func toolKeyCall(key string) string {
	return key[strings.LastIndexByte(key, 0)+1:]
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
// the total latency and, for model calls that produced a first token, the
// time to first token. Unlike adk-python, LLM_ERROR rows also include
// time_to_first_token_ms when a streamed chunk with generated content
// arrived before the error.
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
