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
	"errors"
	"fmt"
	"maps"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
	"weak"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/platform"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
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

var (
	agentInterfaceType   = reflect.TypeFor[agent.Agent]()
	contextInterfaceType = reflect.TypeFor[context.Context]()
)

// agentPkgPath and workflowPkgPath are the import paths of ADK's agent and
// workflow packages, and adkPkgPrefix is the import path prefix that all of
// ADK's packages share, as runtime.Stack names their functions.
var (
	agentPkgPath           = agentInterfaceType.PkgPath()
	workflowPkgPath        = reflect.TypeFor[workflow.Edge]().PkgPath()
	adkPkgPrefix           = agentPkgPath[:strings.LastIndexByte(agentPkgPath, '/')+1]
	internalContextPkgPath = adkPkgPrefix + "internal/context"
	internalLLMPkgPath     = adkPkgPrefix + "internal/llminternal"
	remoteAgentPkgPath     = adkPkgPrefix + "agent/remoteagent/v2"
	a2aServerPkgPath       = adkPkgPrefix + "server/adka2a/v2"
)

// schedulerCreator and errgroupCreator are the functions that runtime.Stack
// names as the creators of the goroutines on which ADK's workflow scheduler
// runs each node that it starts (go runNode in scheduler.startNode) and on
// which a ParallelAgent runs each of its branches (errgroup.Group.Go). A
// function that contains a go statement is never inlined, so the names are
// those of the functions themselves.
var schedulerCreator = workflowPkgPath + ".(*scheduler).startNode"

const errgroupCreator = "golang.org/x/sync/errgroup.(*Group).Go"

// Custom metadata keys of the event that ADK's workflow scheduler writes when
// a node run ends without an output (see recordNodeCompletion).
const (
	workflowBasePathPrefix    = "adk.workflow.base_path."
	workflowNodeOutcomeKey    = "adk.workflow.node_outcome"
	workflowNodeCompletedKey  = "adk.workflow.node_completed"
	workflowNodeFailedOutcome = "failed"
	workflowNodeCancelOutcome = "cancelled"
)

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
	// streamed reports that the agent that owns a model call emitted a
	// partial session event while the call was open (see recordStreamed).
	// The call is making progress, so the agent did not fail even if its
	// caller stops reading before the final response. A streamed chunk that
	// an agent's AfterModelCallback rejects is never emitted, so it does not
	// count.
	streamed bool
	// canceled reports whether the context of the callback that started the
	// span was canceled with cause context.Canceled, as when a workflow or a
	// parallel agent cancels running work after the caller stopped reading,
	// and otherwise returns the cause with which the context ended, such as a
	// sibling branch's error in a parallel agent's errgroup. A call that
	// ended, or that the run handed to the late state, drops it, ended,
	// activation, retiredCauses and parent (see dropContexts), so that the
	// record of a call that outlives its run keeps neither the context nor
	// the span above it, whose record keeps the context of its own callback.
	// An agent span retired from a late trace keeps it for as long as it
	// keeps parent (see dropRetiredLateContextsLocked).
	canceled func() (canceled bool, cause error)
	// retiredCauses holds the openFailureCause of each activation that
	// retireAgentLocked removed under this span while this span stayed open,
	// together with the name, branch and node path of the retired activation
	// whose cause it is. It keeps the non-cancellation, non-bare-deadline
	// cause of a retired branch (such as a parallel agent's errgroup cause
	// when a responded branch failed, moving up to each enclosing open agent
	// as inner spans retire) until a final response of the span clears it
	// (see completeResponse), a final response of the retired activation's
	// author clears that activation's own entry (as when its own
	// AfterAgentCallback recovers after the plugin retired it), a failed
	// workflow retry attempt or tool-run activation purges it (see
	// purgeActivationLocked), or the invocation ends.
	retiredCauses []retiredCause
	// parent is the span that the span started under, if it is an agent or
	// the invocation span, and stopPoint reports that a context that the run
	// cancels when its caller stops reading can start at the span: the
	// context of an activation that runs right under the invocation, which
	// the runner runs as a workflow node, of a workflow node, which runs on
	// a node path of its own, or of a branch of a parallel agent (see
	// canceledByStop).
	parent    *spanRecord
	stopPoint bool
	// hasResponse reports whether the agent activation has produced a model
	// response, or a partial or final event with content or a typed output,
	// itself or through an activation that ran under it. A final event of an
	// activation also settles the activations that still run under it in the
	// event's own branch and node path, even if one of them fails later, and
	// a final event of an author that has no open span settles the activation
	// that it would have run under.
	//
	// hasResponse alone does not retire an earlier activation on
	// re-entry (see popDelegatingAgentSpan): a transfer-back chain sets it on
	// the parent while the parent is still running, so retiring on
	// hasResponse would end a live span. Instead, taskDelegated marks that a
	// child task-mode sub-agent completed via finish_task, which is the only
	// case where re-entry retirement is safe.
	hasResponse bool
	// ownResponse reports that this activation itself produced a final event or
	// responded via its own AfterModelCallback / concurrent peer reclaim, rather
	// than only inheriting hasResponse from a child via markRespondedLocked.
	ownResponse bool
	// modelResponded reports that AfterModelCallback (popLLMSpan) set
	// hasResponse on this activation via activationKey, so a concurrent peer on
	// the same branch and node path never reclaims it.
	modelResponded bool
	// extraResponses counts additional final session-event responses attributed
	// to this activation while hasResponse was already set, available for a
	// concurrent peer on the same branch and node path to reclaim on pop.
	extraResponses int
	// awaitingModelEvent and awaitingToolEvent count completed model and tool
	// callbacks on this activation whose session events have not yet reached
	// completeResponse.
	awaitingModelEvent int
	awaitingToolEvent  int
	// concurrentPeer records that another activation of the same (parent, name,
	// branch, path) ran concurrently while this activation was open.
	concurrentPeer bool
	// goroutineID is the runtime goroutine ID on which pushAgentSpan started
	// the activation, origin names the function that created that goroutine
	// and the goroutine that ran it, and attempt tells whether the activation
	// is the agent of a workflow node attempt that runs on that goroutine, as
	// runtime.Stack shows them (see currentGoroutine). They tell whether the
	// activation that it runs under waits for its run to return (see joins),
	// whether a tool call that returned ran it on its own goroutine (see
	// markToolRunsDoneLocked), and whether a later activation is a later
	// attempt of the same workflow node (see pushAgentSpan). One goroutine
	// can hold several agent runs at once, as when a function node runs an
	// agent again inside its loop over the events of an earlier run of it,
	// whose iterator stays suspended in a yield meanwhile, so that an
	// activation ran on the current goroutine does not show by itself that
	// its run is over (see canSettleInSubtreeLocked). They are zero when the
	// context does not show the activation's InvocationContext.
	goroutineID uint64
	origin      goroutineOrigin
	attempt     attemptSite
	// startPlatformTime holds the activation's start timestamp on the platform
	// clock (platform.Now), so session events stamped on the same non-concurrent
	// goroutine strictly after startPlatformTime can prove they did not originate
	// from an already-retired non-custom predecessor.
	startPlatformTime time.Time
	// completed records that the activation's own AfterAgentCallback ran (see
	// popAgentSpan). ADK runs it only once the agent's Run returned all its
	// events, which for a ParallelAgent is once every branch returned (see
	// joins).
	completed bool
	// nodeRunEnded records that a node completion event reported that the
	// node run in which the workflow scheduler started the activation ended
	// with a failed or canceled outcome (see recordNodeCompletion). The
	// scheduler writes that event only after the node's Run returned, but a
	// node can write the same metadata on an event of its own while its run
	// goes on. So the report decides what is written for the activation, but
	// never shows that its run is over: on its strength alone no
	// InvocationContext is read (see laterAttempt in pushAgentSpan and
	// reportedOverLocked).
	nodeRunEnded bool
	// startedAgain records that the workflow scheduler that ran the
	// activation as the agent of an agent node attempt (see attemptNode) ran
	// the same node again, on the same branch and node path but on another
	// goroutine, while the activation was open, as RetryConfig does after a
	// failure, or as a second trigger of a running node does (see
	// pushAgentSpan and endRestartedNodeRuns). An agent that a function node
	// or a custom agent runs is never marked, even when it runs on a
	// goroutine that the scheduler started: the scheduler did not run it as
	// an attempt, and its failure is the code's own to handle.
	startedAgain bool
	// nodeRuns counts, on the activation of a workflow agent, the node runs
	// on each node path and branch whose agent activation its scheduler
	// started and whose end no node completion event reported yet (see
	// recordNodeCompletion).
	nodeRuns map[nodeRunKey]int

	// ended reports whether the activation's own InvocationContext was
	// marked as ended (InvocationContext.Ended), as ADK does when a plugin or
	// agent BeforeAgentCallback answers with non-nil content (skipping the
	// agent body and AfterAgentCallback). That state is written without
	// synchronization on the goroutine that runs the activation, so ended is
	// called only where that goroutine cannot write it concurrently (see
	// settleEndedLocked), and is nil once it reported true.
	ended func() bool
	// contextEnded records that ended reported true. Combining it with
	// hasResponse in responded() recognizes a BeforeAgentCallback-answered
	// sub-agent even when an enclosing workflow.ParallelWorker or custom
	// parent drops its event before OnEventCallback sees it.
	contextEnded bool
	// runDone records that the activation's run is known to be over while its
	// span is open: its own AfterAgentCallback ran, a later attempt of its
	// workflow.ParallelWorker item started on the goroutine on which it ran
	// (see laterAttempt in pushAgentSpan), the workflow agent whose scheduler
	// ran its node again ended (see endRestartedNodeRuns), an activation that
	// waits for it is known to be over in turn (see joins), it started on the
	// goroutine of such an end after the activation or tool call whose
	// callback marks it (see canSettleInSubtreeLocked), or it started on the
	// goroutine of a tool call that returned, after the call started (see
	// markToolRunsDoneLocked). That an activation merely ran on the current
	// goroutine does not count, as its iterator may still be suspended in a
	// yield below the current frame, and neither does a node completion
	// event, which a node can write itself (see nodeRunEnded). Each of these
	// is a point where the goroutine that sets it under mu has synchronized
	// with the end of the activation's run, so ended can be called under mu
	// from then on, unless a tool call or an agent that is still open ran the
	// activation instead of the one that it runs under (see
	// launcherOpenLocked): ended is called once nothing that may have run it
	// is open anymore (see settleFinishedLocked). Retiring a span without
	// such a point leaves runDone unset: a finish_task function response (see
	// popTaskAgentSpan) and the new activation of an agent that delegated to
	// a completed task (see popDelegatingAgentSpan) are events that do not
	// show that the activation's run returned, and neither does the return of
	// a tool call for a run that the tool started on another goroutine (see
	// endToolRunsLocked), so that run may still end its InvocationContext.
	runDone bool
	// seq orders the agent activations and tool calls of an invocation by
	// the order in which they started, which their start times may not tell
	// apart.
	seq uint64
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
	// eventClockStart is platform.Now(ctx) read before inv.mu when a task-mode
	// agent activation opens (zero for non-task agents so ordinary runs do not
	// advance a custom platform TimeProvider), matching the clock with which
	// session.NewEvent stamps events so an event created before this activation
	// started cannot be attributed to it.
	eventClockStart time.Time
	// taskAttempts holds the finish_task calls that this activation's own tool
	// callbacks observed, keeping observed completions (taskAttemptCompleted and
	// taskAttemptFailedEnd) strictly separate from pending/non-ending attempts
	// (taskAttemptPending) until the FunctionResponse event arrives (see
	// noteTaskStart, noteTaskAttempt, retireFailedTaskAttempts, and
	// finishedTaskLocked).
	taskAttempts []taskAttempt
	// nonLLM reports that the agent is a workflow agent, such as a sequential
	// or loop agent, rather than an llmagent whose sub-agents can transfer to
	// peers.
	nonLLM bool
	// propagatesChildError reports that the agent is one of ADK's built-in
	// workflow agents (parallel, sequential, loop, or graph workflow), which
	// always yield sub-agent errors to their caller rather than swallowing
	// them in Run.
	propagatesChildError bool
	// isParallel reports that the agent is ADK's built-in ParallelAgent, which
	// runs each of its declared SubAgents on a goroutine of its own, on
	// parallelBranch(a, child) (see joins).
	isParallel bool
	// isWorkflow reports that the agent is ADK's built-in WorkflowAgent, whose
	// scheduler runs each node that it starts on a goroutine of its own and
	// waits for all of them before the agent's Run returns (see joins).
	isWorkflow bool
	// lastActivity is the time of the latest callback under this agent
	// activation: its own start and end, its model and tool calls, sub-agents
	// run under it, and its errors and events. It stands in for the end of an
	// activation that fails without an AfterAgentCallback, so a purged
	// activation's latency and row time reflect its own work, not the retry
	// delay that follows it (see touchAgentActivityLocked).
	lastActivity time.Time
	// taskMode reports that the agent runs in llmagent.ModeTask: a successful
	// finish_task function response ends its activation, and ADK runs no
	// AfterAgentCallback for it (see popTaskAgentSpan). An agent of another
	// mode may have a tool of its own that is named finish_task.
	taskMode bool
	// hasAfterAgentCallbacks reports that the activation has (or conservatively
	// may have) agent- or plugin-level AfterAgentCallbacks that ADK runs outside
	// this plugin's AfterAgentCallback.
	hasAfterAgentCallbacks bool
	// hasCustomUUIDProvider reports that the activation's context had an active
	// custom platform.WithUUIDProvider at start.
	hasCustomUUIDProvider bool
	// runEndSeq and runEndPlatformTime record the sequence and platform time of
	// the first retirement of an enclosing span that left this activation open
	// after its run was known to be over (runDone): the run created all its
	// events by then, so a later retirement of the activation is dated there
	// (see recordRetiredActivationLocked).
	runEndSeq          uint64
	runEndPlatformTime time.Time
	// callStarted reports that pushLLMSpan or pushToolSpan started the model
	// or tool call, rather than popToolSpan making up a span for a call that
	// this plugin did not see start.
	callStarted bool
	// errorAttempt is, for a tool call that OnToolErrorCallback ended, the
	// attempt whose callback ended it (see toolAttempt and
	// invocationTrace.completedTools).
	errorAttempt toolAttempt
	// handedOver reports, for a model or tool call, that the run left it open
	// at AfterRun, which handed it over to the late state of its completed
	// entry (see rememberCompletedLocked), rather than a callback starting it
	// after the run ended. While one such call is open in the late state, the
	// run waits for late calls (see completedTrace.waitsForLateCalls).
	handedOver bool
	// ownActivation reports, for a model or tool call, that parent is the
	// agent activation that makes the call: the agent run of the context of
	// the callback that started the call started it (see activationKey).
	// When the agent had no open span of that run, as when an enclosing agent
	// already retired it, the call runs under the span that the agent would
	// run under (see ownerOf), whose run tells nothing about the call's (see
	// endedWithRun).
	ownActivation bool
	// children holds the names of the sub-agents configured on this agent
	// activation.
	children []string
	// activation identifies the agent run that started the span: the
	// InvocationContext that ADK creates for each run of an agent, which its
	// agent, model and tool callbacks share (see activationKey). It tells
	// apart concurrent runs of the same agent on one branch and node path,
	// which the name, branch and path of the span do not. It is nil when the
	// context does not show it.
	activation any
	// popRetirement holds transient retirement metadata for
	// recordRetiredActivationLocked, so that the retired activation's entry
	// receives it before any pruning inspects it.
	popRetirement *popRetirement
	// recordedAfterAgentCredit and recordedModelCredits hold monotonic
	// high-water credit grants already transferred from this span into
	// retiredActivations, preventing compacted-hull replay from re-granting
	// previously issued or settled credits.
	recordedAfterAgentCredit bool
	recordedModelCredits     int
}

// popRetirement carries per-retirement metadata into
// recordRetiredActivationLocked: whether the retirement counted as a peer
// retirement, any after-agent or model response credits left behind, the
// action maps of the context in which its after-agent callbacks run and the
// value that keeps them reachable (see callbackAfterAgentActions), and the
// goroutine running those callbacks.
type popRetirement struct {
	peerRetirement       bool
	afterAgentCredit     bool
	modelCredits         int
	afterAgentActions    actionMaps
	afterAgentActionsRef any
	goroutineID          uint64
}

// retiredSwallower identifies a retired custom ancestor that drained a child
// failure without its own context failing, so recovery content from that
// activation's AfterAgentCallback may clear the cause without clearing causes
// owned by other activations of the same agent. anyName and anyBranch mark a
// global cap-overflow hull folded across distinct agent names or branches.
type retiredSwallower struct {
	spanID    string
	seq       uint64
	name      string
	branch    string
	path      string
	anyName   bool
	anyBranch bool
}

// matchesIdentity reports whether (name, branch) matches this swallower's
// agent and branch identity, respecting wildcard flags on cap-overflow hulls.
func (s retiredSwallower) matchesIdentity(name, branch string) bool {
	return (s.anyName || s.name == name) && (s.anyBranch || s.branch == branch)
}

// matches reports whether recovery content authored by (name, branch, path)
// matches this swallower's identity. An empty path on either side matches any
// path, as in findAgent.
func (s retiredSwallower) matches(name, branch, path string) bool {
	return s.matchesIdentity(name, branch) && (path == "" || s.path == "" || s.path == path)
}

// retiredCause is the failure cause of an activation that retired under an
// open span (see spanRecord.retiredCauses), with the name, branch and node
// path of the retired activation. swallowableBySelf reports that the retired
// activation drained an inherited child failure without its own context
// failing and is not a built-in workflow agent, so recovery content from its
// own AfterAgentCallback may clear the cause. swallowers holds the retired
// custom swallower activations whose recovery responses may clear this cause.
type retiredCause struct {
	err               error
	name              string
	branch            string
	path              string
	swallowableBySelf bool
	swallowers        []retiredSwallower
	swallowerShared   *atomic.Int64
}

// addSwallower records rec as a retired custom swallower eligible to clear c on
// a later recovery response from that specific activation of rec.
//
// swallowerShared tracks how many slots of the backing array have been claimed
// so linear ancestor chains grow geometrically, reducing backing-array slice
// copies across an N-ancestor chain from O(N^2) to O(N) while duplicate
// checking remains O(N^2) comparisons across chain length N. Value-copied
// branches allocate a fresh backing array instead of overwriting a sibling's
// elements.
func (c *retiredCause) addSwallower(rec *spanRecord) {
	if rec == nil {
		return
	}
	if rec.spanID != "" {
		for _, s := range c.swallowers {
			if s.spanID == rec.spanID {
				return
			}
		}
	} else {
		for _, s := range c.swallowers {
			if s.spanID == "" && s.name == rec.name && s.branch == rec.branch && s.path == rec.path {
				return
			}
		}
	}
	elem := retiredSwallower{
		spanID: rec.spanID,
		seq:    rec.seq,
		name:   rec.name,
		branch: rec.branch,
		path:   rec.path,
	}
	n := len(c.swallowers)
	if c.swallowerShared != nil && n < cap(c.swallowers) && c.swallowerShared.CompareAndSwap(int64(n), int64(n+1)) {
		c.swallowers = append(c.swallowers, elem)
		return
	}
	next := make([]retiredSwallower, n+1, max(4, 2*(n+1)))
	copy(next, c.swallowers)
	next[n] = elem
	shared := new(atomic.Int64)
	shared.Store(int64(n + 1))
	c.swallowers = next
	c.swallowerShared = shared
}

func (c retiredCause) hasSwallowerSpan(spanID string) bool {
	if spanID == "" {
		return false
	}
	for _, s := range c.swallowers {
		if s.spanID == spanID {
			return true
		}
	}
	return false
}

// uuidProviderState is what the agent activations under one key of a
// uuidProviderLedger started with: a custom platform.UUIDProvider on their
// context, the default generator (no provider, or a nil one installed with
// platform.WithUUIDProvider(ctx, nil), which shadows any provider further up
// ctx, so that session.NewEvent falls back to a UUIDv4), or a mix of both.
type uuidProviderState uint8

const (
	uuidProviderUnseen uuidProviderState = iota
	uuidProviderCustom
	uuidProviderDefault
	uuidProviderMixed
)

// with returns the state of a key that saw both s and o.
func (s uuidProviderState) with(o uuidProviderState) uuidProviderState {
	switch {
	case s == uuidProviderUnseen || s == o:
		return o
	case o == uuidProviderUnseen:
		return s
	default:
		return uuidProviderMixed
	}
}

// uuidProviderKey is a key of a uuidProviderLedger: the agent name, branch and
// node path of the activations it covers, where anyBranch and anyPath make it
// cover the activations of the name on every branch or at every path.
type uuidProviderKey struct {
	name, branch, path string
	anyBranch, anyPath bool
}

const (
	// maxUUIDProviderEntries bounds the number of keys recorded in a live
	// uuidProviderLedger. Once reached, new path-specific keys fold into
	// author/branch summary keys without growing states.
	maxUUIDProviderEntries = 512
	// maxCompletedUUIDProviderEntries bounds the number of summary keys
	// retained on a completedTrace after compactForCompleted drops
	// per-path entries.
	maxCompletedUUIDProviderEntries = 32
	// maxUUIDProviderAncestorSteps bounds the number of node-path prefix
	// lookups in eventUUIDProviderFor before falling back to anyPath.
	maxUUIDProviderAncestorSteps = 8
)

// uuidProviderLedger records, for the agent activations of one invocation,
// whether each started with a custom platform.UUIDProvider on its context (see
// traceManager.eventUUIDProvider). An invocation gets one only once a custom
// provider is seen in it, so one that never sees a custom provider records
// nothing. The invocation's live trace, its evicted identity, its completed
// entry and its late state share it, and mu guards it.
type uuidProviderLedger struct {
	mu sync.Mutex
	// runCustom reports that the context of the run's BeforeRunCallback or
	// OnUserMessageCallback carried a custom provider.
	runCustom bool
	// compacted reports that compactForCompleted dropped per-path keys for
	// completedTrace retention and lowered the states cap to
	// maxCompletedUUIDProviderEntries.
	compacted bool
	// pathOverflowed reports that at least one path-specific (!anyPath)
	// entry was omitted or compacted into the anyPath summary keys.
	pathOverflowed bool
	// branchOverflowed reports that at least one per-branch (anyPath &&
	// !anyBranch) entry was omitted or compacted into the author-wide
	// (anyBranch && anyPath) summary key.
	branchOverflowed bool
	// authorOverflow aggregates observations whose author-wide summary key
	// could not fit in states.
	authorOverflow uuidProviderState
	// branchSummaryOverflow aggregates branch-wide provider observations that
	// could not fit in branchStates.
	branchSummaryOverflow uuidProviderState
	states                map[uuidProviderKey]uuidProviderState
	// branchStates summarizes per-branch provider state across all authors on
	// non-empty branches, bounded separately at capEntries so states retains its
	// author/branch summary budget cleanly.
	branchStates map[string]uuidProviderState
}

// note records that an activation of name on branch at path started with
// (custom) or without a custom provider, under its own key and under the keys
// that cover any branch, any path, or both. Once states reaches its entry cap,
// new finer-grained keys fold conservatively into coarser summary keys without
// growing states.
func (l *uuidProviderLedger) note(name, branch, path string, custom bool) {
	s := uuidProviderDefault
	if custom {
		s = uuidProviderCustom
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	capEntries := maxUUIDProviderEntries
	if l.compacted {
		capEntries = maxCompletedUUIDProviderEntries
	}

	if branch != "" {
		if cur, ok := l.branchStates[branch]; ok {
			l.branchStates[branch] = cur.with(s)
		} else if l.branchSummaryOverflow == uuidProviderUnseen && len(l.branchStates) < capEntries {
			if l.branchStates == nil {
				l.branchStates = map[string]uuidProviderState{}
			}
			l.branchStates[branch] = s
		} else {
			l.branchSummaryOverflow = l.branchSummaryOverflow.with(s)
		}
	}

	authorKey := uuidProviderKey{name: name, anyBranch: true, anyPath: true}
	if cur, ok := l.states[authorKey]; ok {
		l.states[authorKey] = cur.with(s)
	} else if l.authorOverflow == uuidProviderUnseen && len(l.states) < capEntries {
		l.states[authorKey] = s
	} else {
		l.authorOverflow = l.authorOverflow.with(s)
		l.branchOverflowed = true
		l.pathOverflowed = true
		return
	}

	branchKey := uuidProviderKey{name: name, branch: branch, anyPath: true}
	if cur, ok := l.states[branchKey]; ok {
		l.states[branchKey] = cur.with(s)
	} else if !l.branchOverflowed && len(l.states) < capEntries {
		l.states[branchKey] = s
	} else {
		l.branchOverflowed = true
		l.pathOverflowed = true
		return
	}

	if l.pathOverflowed {
		return
	}

	exact := uuidProviderKey{name: name, branch: branch, path: path}
	anyBranchPath := uuidProviderKey{name: name, path: path, anyBranch: true}
	needed := 0
	if _, ok := l.states[exact]; !ok {
		needed++
	}
	if _, ok := l.states[anyBranchPath]; !ok && anyBranchPath != exact {
		needed++
	}
	if len(l.states)+needed > capEntries {
		l.pathOverflowed = true
		return
	}
	l.states[exact] = l.states[exact].with(s)
	l.states[anyBranchPath] = l.states[anyBranchPath].with(s)
}

// compactForCompleted drops per-path entries and caps retained summary entries
// in a freshly allocated small map so a completedTrace holds < 1 KB even after
// tens of thousands of distinct workflow node paths.
func (l *uuidProviderLedger) compactForCompleted() *uuidProviderLedger {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.compacted = true
	l.pathOverflowed = true
	summaryCount := 0
	for k := range l.states {
		if k.anyPath {
			summaryCount++
		}
	}
	compact := make(map[uuidProviderKey]uuidProviderState, min(summaryCount, maxCompletedUUIDProviderEntries))
	for k, s := range l.states {
		if !k.anyPath || !k.anyBranch {
			continue
		}
		if len(compact) < maxCompletedUUIDProviderEntries {
			compact[k] = s
		} else {
			l.authorOverflow = l.authorOverflow.with(s)
			l.branchOverflowed = true
		}
	}
	for k, s := range l.states {
		if !k.anyPath || k.anyBranch {
			continue
		}
		// Retain a branch key only when its author-wide key also fit in compact.
		authorKey := uuidProviderKey{name: k.name, anyBranch: true, anyPath: true}
		if _, ok := compact[authorKey]; ok && len(compact) < maxCompletedUUIDProviderEntries {
			compact[k] = s
		} else {
			l.branchOverflowed = true
		}
	}
	l.states = compact
	if len(l.branchStates) > 0 {
		compactBranches := make(map[string]uuidProviderState, min(len(l.branchStates), maxCompletedUUIDProviderEntries))
		for b, s := range l.branchStates {
			if len(compactBranches) < maxCompletedUUIDProviderEntries {
				compactBranches[b] = s
			} else {
				l.branchSummaryOverflow = l.branchSummaryOverflow.with(s)
			}
		}
		l.branchStates = compactBranches
	}
	return l
}

func (l *uuidProviderLedger) branchState(branch string) uuidProviderState {
	if l == nil || branch == "" {
		return uuidProviderUnseen
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.branchStates[branch]
}

func (l *uuidProviderLedger) branchOverflow() uuidProviderState {
	if l == nil {
		return uuidProviderUnseen
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.branchSummaryOverflow
}

// stateAt returns the state of k, which is uuidProviderUnseen when l is nil.
// When finer-grained keys have overflowed or been compacted, stateAt falls back
// conservatively to the corresponding summary tier so mixed custom/default
// histories return uuidProviderMixed while pure custom histories stay custom.
func (l *uuidProviderLedger) stateAt(k uuidProviderKey) uuidProviderState {
	if l == nil {
		return uuidProviderUnseen
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if !k.anyPath {
		if l.pathOverflowed {
			return uuidProviderUnseen
		}
		return l.states[k]
	}
	if s := l.states[k]; s != uuidProviderUnseen {
		return s
	}
	if !k.anyBranch && l.branchOverflowed {
		authorKey := uuidProviderKey{name: k.name, anyBranch: true, anyPath: true}
		if s := l.states[authorKey]; s != uuidProviderUnseen {
			return s
		}
	}
	if l.authorOverflow != uuidProviderUnseen {
		eff := l.authorOverflow
		if !l.runCustom {
			eff = eff.with(uuidProviderDefault)
		}
		return eff
	}
	return uuidProviderUnseen
}

// isRunCustom reports l.runCustom, which is false when l is nil.
func (l *uuidProviderLedger) isRunCustom() bool {
	if l == nil {
		return false
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.runCustom
}

// actionMaps holds pointers to the StateDelta and ArtifactDelta maps of a
// session.EventActions, nil for a nil map. Storing unsafe.Pointer rather than
// a uintptr address keeps each captured map reachable for as long as the
// actionMaps value is, so the GC can neither free the map nor give its address
// to a new map that an unrelated event would then appear to share. ADK
// initializes the callback context's EventActions with both maps and yields the
// event of an after-agent callback's content with a copy of EventActions.
// Unless a callback replaces both map fields on the live EventActions via
// ctx.WithDelta(nil).Actions(), the yielded event shares at least one of the
// captured maps (and an earlier plugin's OnEventCallback that copies only one
// map leaves the other in place).
type actionMaps struct {
	stateDelta, artifactDelta unsafe.Pointer
}

// shares reports whether a and b hold a common non-nil map.
func (a actionMaps) shares(b actionMaps) bool {
	return (a.stateDelta != nil && a.stateDelta == b.stateDelta) ||
		(a.artifactDelta != nil && a.artifactDelta == b.artifactDelta)
}

// firstRetiredCause returns the first failure cause that an activation that
// retired under rec left on it, or nil.
func (rec *spanRecord) firstRetiredCause() error {
	for _, c := range rec.retiredCauses {
		if c.err != nil {
			return c.err
		}
	}
	return nil
}

// clearRetiredCauseFor removes the swallowable failure causes that activations
// of the agent name on branch, and on path, retired under rec with. An empty
// path on either side matches any path, as in findAgent.
func (rec *spanRecord) clearRetiredCauseFor(name, branch, path string) {
	rec.retiredCauses = slices.DeleteFunc(rec.retiredCauses, func(c retiredCause) bool {
		return c.swallowableBySelf && c.name == name && c.branch == branch && (path == "" || c.path == "" || c.path == path)
	})
}

type agentIdentityKey struct {
	name   string
	branch string
}

// retiredActivation records a retired agent activation after it leaves
// agentStack, along with any custom ancestors that were still open on
// agentStack at retirement and might buffer its events across later runs, plus
// the sequence and platform-clock metadata used to prove later session events
// did not originate from it.
type retiredActivation struct {
	swallower    retiredSwallower
	parentSpanID string
	guardSpanIDs []string
	// ancestorSpanIDs holds the span IDs along the enclosing agent chain from
	// rec's immediate parent up to the top-level agent at retirement, keeping
	// settlement scoped to descendants of the recovering ancestor.
	ancestorSpanIDs []string
	// openEnclosingSpanIDs holds the sorted, deduplicated span IDs of the
	// lowest still-open enclosing agent ancestors tracking when this
	// activation's enclosing run(s) return, and enclosingRetiredUnjoined
	// records that an enclosing chain exited without synchronizing (completed
	// or runDone) so parentRunOver cannot later fire across it.
	openEnclosingSpanIDs     []string
	enclosingRetiredUnjoined bool
	retiredAtSeq             uint64
	completedOrTask          bool
	// joinedAtRetire reports that the activation's run synchronized and exited
	// at retirement (completed || runDone), excluding finish_task retirements
	// whose goroutine is not guaranteed to have returned.
	joinedAtRetire bool
	// hasUnjoinedTask records monotonically across compaction folds whether this
	// entry (or any member folded into it) retired as an unjoined task
	// (taskMode && !joinedAtRetire) whose goroutine may still emit events.
	hasUnjoinedTask bool
	// settledByParent reports that this activation retired cleanly or had every
	// cause cleared by an ancestor with no surviving failures along the chain;
	// exclusion still requires event ID and post-start/post-retirement ordering.
	settledByParent        bool
	concurrentPeer         bool
	unrespondedAtPop       bool
	hasAfterAgentCallbacks bool
	hasCustomUUIDProvider  bool
	// afterAgentActions holds the action maps of the context in which the
	// activation's after-agent callbacks ran, which the event that ADK builds
	// from their content carries (see actionMaps), and afterAgentActionsRef
	// holds the EventActions that holds both maps (see
	// callbackAfterAgentActions), which keeps them reachable so that their
	// addresses cannot be reused while they are compared, without keeping the
	// callback context or anything else of the run. They are recorded for
	// every activation with after-agent callbacks and kept until that event
	// arrives or nothing can take it for another activation's event anymore
	// (see settleAfterAgentActionsLocked).
	afterAgentActions    actionMaps
	afterAgentActionsRef any
	// afterAgentEventSeen records that the event that ADK built from the
	// activation's after-agent callbacks was delivered. ADK builds at most one
	// such event per activation, so no event created after the activation
	// retired is its own from then on (see canOwnEventCreatedAt).
	afterAgentEventSeen bool
	// afterAgentCredit records that the activation retired unresponded while
	// an open peer of the same agent, branch and node path ran, with no model
	// or tool event pending, so that reconcileConcurrentResponseOnPopLocked
	// added one unrespondedRetired credit for the response that it may still
	// deliver. Its own after-agent event settles that credit and no other (see
	// consumeAfterAgentCreditLocked).
	afterAgentCredit bool
	// modelCredits records how many pending model-event credits in
	// unrespondedRetired the activation contributed when it retired, so that
	// its own buffered run events settle at most those credits and never steal a
	// peer's credit.
	modelCredits int
	// creditSettled records that an event of the activation settled the
	// credit that afterAgentCredit recorded, its after-agent event or an
	// event of its run, so that no other event of the activation settles a
	// credit (see settleRetiredResponseCreditLocked). creditGroupID links
	// overlapping credit-bearing entries whose ambiguous events share a
	// bounded group credit pool (see invocationTrace.creditGroupRem).
	creditSettled bool
	creditGroupID uint64
	// foldedCredits preserves credit-group memberships and remaining credit
	// capacities per exact (name, branch, path) when credit-bearing entries are
	// folded into a cap-128 hull so surviving group peers keep consistent
	// shared creditGroupRem pools and hulls never leak or over-consume credits.
	foldedCredits []foldedCredit
	// peerRetirement records that reconcileConcurrentResponseOnPopLocked
	// counted the activation's retirement in retiredPeerCount, so that the
	// retired peers that still have entries of their own can be told from
	// those that were pruned or compacted (see soleOpenAuthorLocked).
	peerRetirement bool
	// popGoroutineID is the goroutine on which popAgentSpan retired the
	// activation, 0 when unknown. ADK runs the activation's after-agent
	// callbacks and builds their event on that goroutine right after
	// popAgentSpan returns, before the goroutine starts or ends another
	// agent other than one that a callback runs, which starts and ends
	// inside them (see markAfterAgentCallbacksReturnedOnLocked).
	popGoroutineID uint64
	// afterAgentDone records that the activation's after-agent callbacks
	// returned, or that the run of its parent returned, so that their event,
	// if they built one and anyone can still deliver it, was created by then
	// (see settleAfterAgentActionsLocked). afterAgentDoneAt is the platform
	// time at which that was seen, zero when unknown: the activation created
	// no event after it, its after-agent event included (see
	// canOwnEventCreatedAt).
	afterAgentDone   bool
	afterAgentDoneAt time.Time
	// afterAgentPending marks a compacted hull that stands for activations
	// whose after-agent events may still come and whose action maps it no
	// longer holds (see foldPendingAfterAgentLocked): any event of its agent
	// that one of them can have created may be such an event, so the hull can
	// own an event created after its members retired, and no later activation
	// excludes it by event-creation timestamp provenance.
	afterAgentPending           bool
	compactedHull               bool
	grantedAfterAgentCredit     bool
	grantedModelCredits         int
	goroutineID                 uint64
	startPlatformTime           time.Time
	retiredAtPlatformTime       time.Time
	parentRetiredAtPlatformTime time.Time
}

type foldedCredit struct {
	swallower        retiredSwallower
	credits          int
	creditGroupID    uint64
	afterAgentCredit bool
}

func appendFoldedCredit(dst []foldedCredit, sw retiredSwallower, credits int, gid uint64, afterAgentCredit bool) []foldedCredit {
	if credits <= 0 || gid == 0 {
		return dst
	}
	exact := retiredSwallower{name: sw.name, branch: sw.branch, path: sw.path}
	for i := range dst {
		if dst[i].creditGroupID == gid && dst[i].swallower.name == exact.name && dst[i].swallower.branch == exact.branch && dst[i].swallower.path == exact.path {
			dst[i].credits += credits
			dst[i].afterAgentCredit = dst[i].afterAgentCredit || afterAgentCredit
			return dst
		}
	}
	return append(dst, foldedCredit{
		swallower:        exact,
		credits:          credits,
		creditGroupID:    gid,
		afterAgentCredit: afterAgentCredit,
	})
}

func (ra retiredActivation) remainingCredits() int {
	n := ra.modelCredits
	if ra.afterAgentCredit {
		n++
	}
	return n
}

func (ra *retiredActivation) normalizeGrantedCredits() {
	if ra.afterAgentCredit {
		ra.grantedAfterAgentCredit = true
	}
	if ra.modelCredits > ra.grantedModelCredits {
		ra.grantedModelCredits = ra.modelCredits
	}
}

func (inv *invocationTrace) allocCreditGroupIDLocked() uint64 {
	inv.nextCreditGroupID++
	if inv.creditGroupRem == nil {
		inv.creditGroupRem = make(map[uint64]int)
	}
	return inv.nextCreditGroupID
}

func (inv *invocationTrace) settleCreditGroupIfExhaustedLocked(gid uint64, name, branch, path string) {
	if gid == 0 || inv.creditGroupRem == nil {
		return
	}
	if rem, ok := inv.creditGroupRem[gid]; ok && rem > 0 {
		return
	}
	delete(inv.creditGroupRem, gid)
	var settled bool
	for i := range inv.retiredActivations {
		ra := &inv.retiredActivations[i]
		if ra.compactedHull {
			if len(ra.foldedCredits) > 0 {
				prevLen := len(ra.foldedCredits)
				ra.foldedCredits = slices.DeleteFunc(ra.foldedCredits, func(fc foldedCredit) bool {
					return fc.creditGroupID == gid
				})
				if len(ra.foldedCredits) != prevLen {
					settled = true
					if len(ra.foldedCredits) == 0 {
						ra.foldedCredits = nil
						ra.creditSettled = true
					}
				}
			}
			continue
		}
		if ra.creditGroupID == gid {
			ra.normalizeGrantedCredits()
			ra.afterAgentCredit = false
			ra.modelCredits = 0
			ra.creditSettled = true
			ra.creditGroupID = 0
			settled = true
		}
	}
	if settled {
		inv.settleAfterAgentActionsLocked(func(ra *retiredActivation) bool {
			return ra.swallower.matches(name, branch, path)
		})
	}
}

func (inv *invocationTrace) mergeCreditGroupsLocked(indices []int, name, branch, path string) uint64 {
	if len(indices) == 0 {
		return 0
	}
	exactKey := func(sw retiredSwallower) bool {
		return sw.name == name && sw.branch == branch && sw.path == path
	}
	var targetGID uint64
	oldGIDs := make(map[uint64]bool)
	totalRem := 0
	for _, idx := range indices {
		ra := &inv.retiredActivations[idx]
		if ra.compactedHull {
			for _, fc := range ra.foldedCredits {
				if !exactKey(fc.swallower) || fc.credits <= 0 {
					continue
				}
				if fc.creditGroupID != 0 {
					if targetGID == 0 {
						targetGID = fc.creditGroupID
					}
					if !oldGIDs[fc.creditGroupID] {
						oldGIDs[fc.creditGroupID] = true
						totalRem += inv.creditGroupRem[fc.creditGroupID]
					}
				} else {
					totalRem += fc.credits
				}
			}
			continue
		}
		ra.normalizeGrantedCredits()
		if ra.creditGroupID != 0 {
			if targetGID == 0 {
				targetGID = ra.creditGroupID
			}
			if !oldGIDs[ra.creditGroupID] {
				oldGIDs[ra.creditGroupID] = true
				totalRem += inv.creditGroupRem[ra.creditGroupID]
			}
		} else {
			totalRem += ra.remainingCredits()
		}
	}
	if targetGID == 0 {
		targetGID = inv.allocCreditGroupIDLocked()
	}
	for gid := range oldGIDs {
		if gid != targetGID {
			delete(inv.creditGroupRem, gid)
		}
	}
	memberCap := 0
	for i := range inv.retiredActivations {
		ra := &inv.retiredActivations[i]
		if ra.compactedHull {
			if len(ra.foldedCredits) == 0 {
				continue
			}
			var merged []foldedCredit
			remapped := false
			for _, fc := range ra.foldedCredits {
				if !exactKey(fc.swallower) || fc.credits <= 0 {
					merged = append(merged, fc)
					continue
				}
				if oldGIDs[fc.creditGroupID] {
					fc.creditGroupID = targetGID
					remapped = true
				}
				if fc.creditGroupID == targetGID {
					memberCap += fc.credits
				}
				if fc.creditGroupID == 0 {
					merged = append(merged, fc)
				} else {
					merged = appendFoldedCredit(merged, fc.swallower, fc.credits, fc.creditGroupID, fc.afterAgentCredit)
				}
			}
			if remapped {
				ra.foldedCredits = merged
			}
			continue
		}
		if oldGIDs[ra.creditGroupID] {
			ra.creditGroupID = targetGID
		}
		if ra.creditGroupID == targetGID {
			memberCap += ra.remainingCredits()
		}
	}
	for _, idx := range indices {
		ra := &inv.retiredActivations[idx]
		if ra.compactedHull {
			if len(ra.foldedCredits) == 0 {
				continue
			}
			var merged []foldedCredit
			assigned := false
			for _, fc := range ra.foldedCredits {
				if exactKey(fc.swallower) && fc.credits > 0 && fc.creditGroupID == 0 {
					fc.creditGroupID = targetGID
					memberCap += fc.credits
					assigned = true
				}
				if fc.creditGroupID == 0 {
					merged = append(merged, fc)
				} else {
					merged = appendFoldedCredit(merged, fc.swallower, fc.credits, fc.creditGroupID, fc.afterAgentCredit)
				}
			}
			if assigned {
				ra.foldedCredits = merged
			}
			continue
		}
		if ra.creditGroupID == 0 {
			ra.creditGroupID = targetGID
			memberCap += ra.remainingCredits()
		}
	}
	if inv.creditGroupRem == nil {
		inv.creditGroupRem = make(map[uint64]int)
	}
	inv.creditGroupRem[targetGID] = min(totalRem, memberCap)
	return targetGID
}

// clearExhaustedAfterAgentCreditsLocked settles any remaining afterAgentCredit
// or modelCredits on exact-key non-hull entries of (name, branch, path) once
// unrespondedRetired[k] has reached 0, and releases any after-agent action maps
// that matching entries no longer need. inv.mu must be held.
func (inv *invocationTrace) clearExhaustedAfterAgentCreditsLocked(name, branch, path string) {
	for i := range inv.retiredActivations {
		ra := &inv.retiredActivations[i]
		if ra.compactedHull {
			if len(ra.foldedCredits) > 0 {
				kept := ra.foldedCredits[:0]
				for _, fc := range ra.foldedCredits {
					if fc.swallower.name == name && fc.swallower.branch == branch && fc.swallower.path == path {
						if fc.creditGroupID != 0 {
							delete(inv.creditGroupRem, fc.creditGroupID)
						}
						continue
					}
					kept = append(kept, fc)
				}
				if len(kept) == 0 {
					ra.foldedCredits = nil
					ra.creditSettled = true
				} else {
					ra.foldedCredits = kept
				}
			}
			continue
		}
		if (ra.afterAgentCredit || ra.modelCredits > 0 || ra.creditGroupID != 0) &&
			ra.swallower.name == name && ra.swallower.branch == branch && ra.swallower.path == path {
			ra.normalizeGrantedCredits()
			if ra.creditGroupID != 0 {
				delete(inv.creditGroupRem, ra.creditGroupID)
				ra.creditGroupID = 0
			}
			ra.afterAgentCredit = false
			ra.modelCredits = 0
			ra.creditSettled = true
		}
	}
	inv.settleAfterAgentActionsLocked(func(ra *retiredActivation) bool {
		return ra.swallower.matches(name, branch, path)
	})
}

// consumeAfterAgentCreditLocked settles, for the after-agent event of
// retiredActivations[idx], the unrespondedRetired credit that the activation
// left when it retired (see retiredActivation.afterAgentCredit), and nothing
// else: the model and tool credits that it left stand for events of its run
// that may still come, whatever the after-agent event's content, and a credit
// of another activation of the same agent stands for that activation's
// response. It reports whether a credit was consumed. inv.mu must be held.
func (inv *invocationTrace) consumeAfterAgentCreditLocked(idx int) bool {
	if idx < 0 || idx >= len(inv.retiredActivations) {
		return false
	}
	ra := &inv.retiredActivations[idx]
	if !ra.afterAgentCredit {
		return false
	}
	ra.normalizeGrantedCredits()
	name, branch, path := ra.swallower.name, ra.swallower.branch, ra.swallower.path
	k := llmKey(identity{path: path, agentName: name, branch: branch})
	if ra.creditGroupID != 0 && inv.creditGroupRem[ra.creditGroupID] <= 0 {
		inv.settleCreditGroupIfExhaustedLocked(ra.creditGroupID, name, branch, path)
		return false
	}
	ra.afterAgentCredit = false
	if ra.modelCredits == 0 {
		ra.creditSettled = true
	}
	if gid := ra.creditGroupID; gid != 0 {
		inv.creditGroupRem[gid]--
		inv.settleCreditGroupIfExhaustedLocked(gid, name, branch, path)
	}
	if inv.unrespondedRetired[k] > 0 {
		inv.unrespondedRetired[k]--
	}
	if inv.unrespondedRetired[k] <= 0 {
		inv.clearExhaustedAfterAgentCreditsLocked(name, branch, path)
	}
	return true
}

// settleRetiredResponseCreditLocked settles one unrespondedRetired credit of
// the key k, the llmKey of (name, branch, path), for a session event that
// completeResponse absorbs as the response of a retired activation and that
// carries no action map of a retired activation's after-agent callbacks. If
// exactly one active credit-bearing retired activation matching (name, branch,
// path) can have created the event in its run (either as the unique matching
// entry or when all credits on k are tracked without wildcard ambiguity and
// only one credit-holder matches), the event settles one credit of that
// activation. Overlapping credit-holders merge into a bounded credit group so
// a single wave can never consume disjoint peer credits; when no matching
// credit-holder exists and trackedGroupCredits >= unrespondedRetired[k], the
// event settles nothing. It reports whether a credit was settled. inv.mu must
// be held.
func (inv *invocationTrace) settleRetiredResponseCreditLocked(k, name, branch, path string, evCreatedAt time.Time, hasEventID, evNonEmptyID bool) bool {
	if inv.unrespondedRetired[k] <= 0 {
		return false
	}
	author := -1
	trackedGroupCredits := 0
	var seenGroup map[uint64]bool
	var matchedHolders []int
	matchedWildcard := false
	hasExactFoldedCredit := func(fc foldedCredit) bool {
		return fc.swallower.name == name && fc.swallower.branch == branch && fc.swallower.path == path &&
			fc.credits > 0 && (fc.creditGroupID == 0 || inv.creditGroupRem[fc.creditGroupID] > 0)
	}
	hasActiveCredit := func(ra retiredActivation) bool {
		if ra.compactedHull {
			return slices.ContainsFunc(ra.foldedCredits, hasExactFoldedCredit)
		}
		return ra.swallower.name == name && ra.swallower.branch == branch && ra.swallower.path == path &&
			(ra.afterAgentCredit || ra.modelCredits > 0) &&
			(ra.creditGroupID == 0 || inv.creditGroupRem[ra.creditGroupID] > 0)
	}
	addGroupCredits := func(ra retiredActivation, skipGroups map[uint64]bool, seen *map[uint64]bool) int {
		total := 0
		addOne := func(gid uint64, fallback int) {
			if gid != 0 {
				if skipGroups[gid] {
					return
				}
				if *seen == nil {
					*seen = make(map[uint64]bool)
				}
				if !(*seen)[gid] {
					(*seen)[gid] = true
					total += inv.creditGroupRem[gid]
				}
				return
			}
			total += fallback
		}
		if ra.compactedHull {
			for _, fc := range ra.foldedCredits {
				if hasExactFoldedCredit(fc) {
					addOne(fc.creditGroupID, fc.credits)
				}
			}
			return total
		}
		addOne(ra.creditGroupID, ra.remainingCredits())
		return total
	}
	for j, ra := range inv.retiredActivations {
		activeCredit := hasActiveCredit(ra)
		if activeCredit {
			trackedGroupCredits += addGroupCredits(ra, nil, &seenGroup)
		}
		if !ra.swallower.matches(name, branch, path) {
			continue
		}
		hasTimestamp := hasEventID || (evNonEmptyID && ra.hasCustomUUIDProvider)
		canOwnMaplessAfterAgent := false
		if activeCredit {
			if ra.compactedHull {
				canOwnMaplessAfterAgent = slices.ContainsFunc(ra.foldedCredits, func(fc foldedCredit) bool {
					return hasExactFoldedCredit(fc) && fc.afterAgentCredit
				})
			} else {
				canOwnMaplessAfterAgent = ra.afterAgentCredit && ra.afterAgentActions == (actionMaps{})
			}
		}
		if !ra.createdInRun(evCreatedAt, hasTimestamp) && !(canOwnMaplessAfterAgent && hasTimestamp && ra.canOwnEventCreatedAt(evCreatedAt, true)) {
			continue
		}
		if ra.swallower.path != path {
			matchedWildcard = true
		}
		if activeCredit {
			matchedHolders = append(matchedHolders, j)
		}
		if ra.compactedHull || author != -1 {
			author = -2
		} else {
			author = j
		}
	}
	if (author >= 0 && inv.retiredActivations[author].swallower.path == path) ||
		(trackedGroupCredits >= inv.unrespondedRetired[k] && !matchedWildcard && len(matchedHolders) == 1 && !inv.retiredActivations[matchedHolders[0]].compactedHull) {
		hIdx := author
		if hIdx < 0 {
			hIdx = matchedHolders[0]
		}
		ra := &inv.retiredActivations[hIdx]
		switch {
		case ra.afterAgentCredit:
			return inv.consumeAfterAgentCreditLocked(hIdx)
		case ra.modelCredits > 0:
			if gid := ra.creditGroupID; gid != 0 && inv.creditGroupRem[gid] <= 0 {
				inv.settleCreditGroupIfExhaustedLocked(gid, name, branch, path)
				return false
			}
			ra.normalizeGrantedCredits()
			ra.modelCredits--
			if !ra.afterAgentCredit && ra.modelCredits == 0 {
				ra.creditSettled = true
			}
			if gid := ra.creditGroupID; gid != 0 {
				inv.creditGroupRem[gid]--
				inv.settleCreditGroupIfExhaustedLocked(gid, name, branch, path)
			}
			inv.unrespondedRetired[k]--
			if inv.unrespondedRetired[k] <= 0 {
				inv.clearExhaustedAfterAgentCreditsLocked(name, branch, path)
			}
			return true
		default:
			return false
		}
	}
	if len(matchedHolders) == 0 {
		if trackedGroupCredits >= inv.unrespondedRetired[k] {
			return false
		}
		inv.unrespondedRetired[k]--
		if inv.unrespondedRetired[k] <= 0 {
			inv.clearExhaustedAfterAgentCreditsLocked(name, branch, path)
		}
		return true
	}

	matchedIdxSet := make(map[int]bool, len(matchedHolders))
	matchedGroups := make(map[uint64]bool, len(matchedHolders))
	for _, idx := range matchedHolders {
		matchedIdxSet[idx] = true
		ra := inv.retiredActivations[idx]
		if ra.compactedHull {
			for _, fc := range ra.foldedCredits {
				if hasExactFoldedCredit(fc) && fc.creditGroupID != 0 {
					matchedGroups[fc.creditGroupID] = true
				}
			}
			continue
		}
		if ra.creditGroupID != 0 {
			matchedGroups[ra.creditGroupID] = true
		}
	}
	unmatchedGroupCredits := 0
	var seenUnmatchedGroup map[uint64]bool
	for j, ra := range inv.retiredActivations {
		if matchedIdxSet[j] || !hasActiveCredit(ra) {
			continue
		}
		unmatchedGroupCredits += addGroupCredits(ra, matchedGroups, &seenUnmatchedGroup)
	}

	var targetGID uint64
	if len(matchedHolders) >= 2 || trackedGroupCredits >= inv.unrespondedRetired[k] {
		targetGID = inv.mergeCreditGroupsLocked(matchedHolders, name, branch, path)
	}
	if trackedGroupCredits >= inv.unrespondedRetired[k] && targetGID != 0 {
		inv.creditGroupRem[targetGID]--
		inv.settleCreditGroupIfExhaustedLocked(targetGID, name, branch, path)
	}
	inv.unrespondedRetired[k]--
	if inv.unrespondedRetired[k] <= 0 {
		inv.clearExhaustedAfterAgentCreditsLocked(name, branch, path)
	} else if inv.unrespondedRetired[k] <= unmatchedGroupCredits {
		if targetGID != 0 && inv.creditGroupRem != nil {
			inv.creditGroupRem[targetGID] = 0
			inv.settleCreditGroupIfExhaustedLocked(targetGID, name, branch, path)
		} else {
			for gid := range matchedGroups {
				if inv.creditGroupRem != nil {
					inv.creditGroupRem[gid] = 0
					inv.settleCreditGroupIfExhaustedLocked(gid, name, branch, path)
				}
			}
			for _, idx := range matchedHolders {
				mra := &inv.retiredActivations[idx]
				if mra.compactedHull {
					continue
				}
				mra.normalizeGrantedCredits()
				mra.afterAgentCredit = false
				mra.modelCredits = 0
				mra.creditSettled = true
			}
			inv.settleAfterAgentActionsLocked(func(ra *retiredActivation) bool {
				return ra.swallower.matches(name, branch, path)
			})
		}
	}
	return true
}

// createdInRun reports whether the retired activation ra can have
// created, in its run, an event created at evCreatedAt, which hasEventID
// reports to be its creation time: the event was created after ra started
// and no later than ra retired. An event without creation evidence can be
// any activation's.
func (ra retiredActivation) createdInRun(evCreatedAt time.Time, hasEventID bool) bool {
	if evCreatedAt.IsZero() || !hasEventID {
		return true
	}
	if !ra.startPlatformTime.IsZero() && evCreatedAt.Before(ra.startPlatformTime) {
		return false
	}
	return ra.retiredAtPlatformTime.IsZero() || ra.unjoinedTaskRetirement() || !evCreatedAt.After(ra.retiredAtPlatformTime)
}

// activeSwallower pairs a retired activation's slice index in
// retiredActivations with its swallower identity for per-agent lookup.
type activeSwallower struct {
	idx       int
	swallower retiredSwallower
}

// afterAgentState holds what tells whether a retired activation still needs
// the action maps of its after-agent callbacks (see
// needsAfterAgentActionsLocked): the spans that a recorded cause names as
// swallowers, and the spans that an open agent span runs under, which may
// have retired already. When such an open span retires later, its failure
// cause moves up with the activation among its swallowers, so the activation's
// after-agent event may still have to recover it.
type afterAgentState struct {
	swallowers    map[string]bool
	underOpen     map[string]bool
	openSet       map[string]bool
	activeByAgent map[agentIdentityKey][]activeSwallower
}

func (inv *invocationTrace) afterAgentStateLocked() afterAgentState {
	st := afterAgentState{
		swallowers: make(map[string]bool),
		underOpen:  make(map[string]bool),
		openSet:    inv.openSpanSetLocked(),
	}
	if st.openSet == nil {
		st.openSet = map[string]bool{}
	}
	for _, rec := range inv.agentStack {
		for _, c := range rec.retiredCauses {
			for _, s := range c.swallowers {
				if s.spanID != "" {
					st.swallowers[s.spanID] = true
				}
			}
		}
	}
	if len(inv.agentStack) > 1 {
		for _, r := range inv.agentStack[1:] {
			for p := r.parent; p != nil && p.kind == spanKindAgent; p = p.parent {
				if p.spanID != "" {
					st.underOpen[p.spanID] = true
				}
			}
		}
	}
	if len(st.swallowers) > 0 || len(st.underOpen) > 0 {
		st.activeByAgent = make(map[agentIdentityKey][]activeSwallower)
		for idx, ra := range inv.retiredActivations {
			if !ra.compactedHull && (st.swallowers[ra.swallower.spanID] || st.underOpen[ra.swallower.spanID]) {
				k := agentIdentityKey{name: ra.swallower.name, branch: ra.swallower.branch}
				st.activeByAgent[k] = append(st.activeByAgent[k], activeSwallower{idx: idx, swallower: ra.swallower})
			}
		}
	}
	return st
}

// holdsAfterAgentEvent reports whether ra still recognizes an after-agent
// event that may come: it keeps the action maps of its after-agent callbacks,
// or it is a hull that stands for activations whose events may (see
// afterAgentPending).
func (ra retiredActivation) holdsAfterAgentEvent() bool {
	return ra.afterAgentActions != (actionMaps{}) || ra.afterAgentPending
}

// markAfterAgentDone records that ra's after-agent callbacks returned, as
// seen at the platform time at, zero when unknown. Each time at which that is
// seen bounds the events of ra, so the earliest known one is kept.
func (ra *retiredActivation) markAfterAgentDone(at time.Time) {
	ra.afterAgentDone = true
	if !at.IsZero() && (ra.afterAgentDoneAt.IsZero() || at.Before(ra.afterAgentDoneAt)) {
		ra.afterAgentDoneAt = at
	}
}

// dropAfterAgentEvent forgets the after-agent event that ra recognizes (see
// holdsAfterAgentEvent).
func (ra *retiredActivation) dropAfterAgentEvent() {
	ra.afterAgentActions = actionMaps{}
	ra.afterAgentActionsRef = nil
	ra.afterAgentPending = false
}

type unrespondedPathSet struct {
	hasWildcard bool
	paths       map[string]struct{}
}

func (ps unrespondedPathSet) matchesPath(path string) bool {
	if !ps.hasWildcard && len(ps.paths) == 0 {
		return false
	}
	if path == "" || ps.hasWildcard {
		return true
	}
	_, ok := ps.paths[path]
	return ok
}

func addUnrespondedPath(ps *unrespondedPathSet, kp string) {
	if kp == "" {
		ps.hasWildcard = true
		return
	}
	if ps.paths == nil {
		ps.paths = make(map[string]struct{}, 1)
	}
	ps.paths[kp] = struct{}{}
}

// unrespondedRetiredIndex indexes inv.unrespondedRetired once per pruning or
// compaction pass so retired-activation matching runs in O(R + U) instead of
// scanning and parsing inv.unrespondedRetired for every retired entry.
type unrespondedRetiredIndex struct {
	nonEmpty bool
	allPaths unrespondedPathSet
	byAgent  map[agentIdentityKey]unrespondedPathSet
	byName   map[string]unrespondedPathSet
	byBranch map[string]unrespondedPathSet
}

func (inv *invocationTrace) newUnrespondedRetiredIndexLocked() unrespondedRetiredIndex {
	if len(inv.unrespondedRetired) == 0 {
		return unrespondedRetiredIndex{}
	}
	var idx unrespondedRetiredIndex
	for k, count := range inv.unrespondedRetired {
		if count <= 0 {
			continue
		}
		kp, kn, kb, ok := parseLLMKey(k)
		if !ok {
			continue
		}
		if !idx.nonEmpty {
			idx.nonEmpty = true
			n := len(inv.unrespondedRetired)
			idx.byAgent = make(map[agentIdentityKey]unrespondedPathSet, n)
			idx.byName = make(map[string]unrespondedPathSet, n)
			idx.byBranch = make(map[string]unrespondedPathSet, n)
		}
		addUnrespondedPath(&idx.allPaths, kp)
		ak := agentIdentityKey{name: kn, branch: kb}
		ps := idx.byAgent[ak]
		addUnrespondedPath(&ps, kp)
		idx.byAgent[ak] = ps
		nps := idx.byName[kn]
		addUnrespondedPath(&nps, kp)
		idx.byName[kn] = nps
		bps := idx.byBranch[kb]
		addUnrespondedPath(&bps, kp)
		idx.byBranch[kb] = bps
	}
	return idx
}

func (idx unrespondedRetiredIndex) matches(s retiredSwallower) bool {
	if !idx.nonEmpty {
		return false
	}
	switch {
	case s.anyName && s.anyBranch:
		return idx.allPaths.matchesPath(s.path)
	case s.anyBranch:
		return idx.byName[s.name].matchesPath(s.path)
	case s.anyName:
		return idx.byBranch[s.branch].matchesPath(s.path)
	default:
		return idx.byAgent[agentIdentityKey{name: s.name, branch: s.branch}].matchesPath(s.path)
	}
}

func (inv *invocationTrace) shrinkRetiredActivationsLocked() {
	n := len(inv.retiredActivations)
	if n == 0 {
		inv.retiredActivations = nil
		return
	}
	c := cap(inv.retiredActivations)
	if c > 2*maxRetiredActivationsPerInvocation && n*4 <= c {
		shrunk := make([]retiredActivation, n, max(n, maxRetiredActivationsPerInvocation))
		copy(shrunk, inv.retiredActivations)
		inv.retiredActivations = shrunk
	}
}

func (inv *invocationTrace) hasUnrespondedRetiredSwallowerLocked(s retiredSwallower, includeTools bool) bool {
	matchMap := func(m map[string]int) bool {
		for k, count := range m {
			if count <= 0 {
				continue
			}
			kp, kn, kb, ok := parseLLMKey(k)
			if ok && s.matches(kn, kb, kp) {
				return true
			}
		}
		return false
	}
	return matchMap(inv.unrespondedRetired) || (includeTools && matchMap(inv.unrespondedRetiredTools))
}

// hasUnrespondedRetiredForLocked reports whether inv.unrespondedRetired (or,
// when includeTools is true, inv.unrespondedRetiredTools) holds a positive
// credit for a key matching (name, branch, path), using wildcard path
// semantics where an empty path on either side matches any path. inv.mu must
// be held.
func (inv *invocationTrace) hasUnrespondedRetiredForLocked(name, branch, path string, includeTools bool) bool {
	return inv.hasUnrespondedRetiredSwallowerLocked(retiredSwallower{name: name, branch: branch, path: path}, includeTools)
}

// needsAfterAgentActionsLocked reports whether retiredActivations[idx] keeps
// the action maps of its after-agent callbacks (see
// retiredActivation.afterAgentActions), or, for a hull, keeps standing for
// the after-agent events of its members (see afterAgentPending). Its
// after-agent event is recognized by them alone, so that an event that it
// builds after the activation retired is never taken for another
// activation's: one that started meanwhile, or one whose run the event's
// timestamp falls in. The maps are kept while
//
//   - the event may recover a failure that the activation swallowed, or one
//     that an open span under it carries and that moves up with it among its
//     swallowers once that span retires (see afterAgentState);
//   - its after-agent callbacks may still run (see afterAgentDone), so that
//     an activation of the same agent that starts meanwhile could be taken
//     for the event's author;
//   - the event may settle the credit that the activation left (see
//     afterAgentCredit);
//   - or, once the callbacks returned, another activation of the agent could
//     be taken for the event's author or lose something to it: one is open,
//     one swallowed a failure that is still recorded or has an open span
//     under it, or a credit of the agent is outstanding.
//
// An activation of the agent that starts after the callbacks returned
// started after their event was created, so it does not own that event,
// unless both fall in one tick of a coarse clock, which the maps cover while
// that activation is open, and an event that a retired activation of the
// agent, other than a swallower, is taken to have created recovers nothing.
// Nor is the activation a possible author of an event created after its
// callbacks were seen to return (see afterAgentDoneAt), whether its
// after-agent event was delivered unrecognized or never came, so an open
// activation's later recovery counts without the maps. That the activation
// retired without a response, or that a custom ancestor may still hold its
// after-agent event, keeps nothing by itself once its callbacks returned:
// the event would only show that it responded, which no failure depends on,
// and keeping the maps until such an event arrives would keep them for every
// activation whose callbacks build none, while a custom ancestor runs.
// inv.mu must be held.
func (inv *invocationTrace) needsAfterAgentActionsLocked(idx int, st afterAgentState) bool {
	if idx < 0 || idx >= len(inv.retiredActivations) {
		return false
	}
	ra := &inv.retiredActivations[idx]
	if !ra.holdsAfterAgentEvent() {
		return false
	}
	if !ra.compactedHull && (st.swallowers[ra.swallower.spanID] || st.underOpen[ra.swallower.spanID]) {
		return true
	}
	openSet := st.openSet
	if openSet == nil {
		openSet = inv.openSpanSetLocked()
	}
	canHoldEvent := openSet[ra.parentSpanID] || slices.ContainsFunc(ra.guardSpanIDs, func(g string) bool {
		return openSet[g]
	})
	name, branch, path := ra.swallower.name, ra.swallower.branch, ra.swallower.path
	if (!ra.afterAgentDone && canHoldEvent) || (ra.afterAgentCredit && inv.hasUnrespondedRetiredSwallowerLocked(ra.swallower, false)) {
		return true
	}
	if inv.hasUnrespondedRetiredSwallowerLocked(ra.swallower, true) {
		return true
	}
	if !canHoldEvent {
		return false
	}
	if len(inv.agentStack) > 1 {
		for _, rec := range inv.agentStack[1:] {
			if rec.kind == spanKindAgent && ra.swallower.matches(rec.name, rec.branch, rec.path) {
				return true
			}
		}
	}
	if len(st.swallowers) > 0 || len(st.underOpen) > 0 {
		if st.activeByAgent != nil && !ra.swallower.anyName && !ra.swallower.anyBranch {
			for _, other := range st.activeByAgent[agentIdentityKey{name: name, branch: branch}] {
				if other.idx != idx && other.swallower.matches(name, branch, path) {
					return true
				}
			}
		} else {
			for j, other := range inv.retiredActivations {
				if j != idx && !other.compactedHull && ra.swallower.matches(other.swallower.name, other.swallower.branch, other.swallower.path) && (st.swallowers[other.swallower.spanID] || st.underOpen[other.swallower.spanID]) {
					return true
				}
			}
		}
	}
	return false
}

// settleAfterAgentActionsLocked drops the after-agent action maps of the
// retired activations for which match reports true that no longer need them
// (see needsAfterAgentActionsLocked). inv.mu must be held.
func (inv *invocationTrace) settleAfterAgentActionsLocked(match func(ra *retiredActivation) bool) {
	if match == nil {
		return
	}
	var st afterAgentState
	for idx := range inv.retiredActivations {
		ra := &inv.retiredActivations[idx]
		if !ra.holdsAfterAgentEvent() || !match(ra) {
			continue
		}
		if st.swallowers == nil {
			st = inv.afterAgentStateLocked()
		}
		if !inv.needsAfterAgentActionsLocked(idx, st) {
			ra.dropAfterAgentEvent()
		}
	}
}

// afterAgentCallbacksReturnedOnLocked records, when an agent starts or ends
// on the goroutine goid at the platform time at, that the after-agent
// callbacks of the activations that popAgentSpan retired on that goroutine
// earlier returned, and by when (see markAfterAgentCallbacksReturnedOnLocked),
// and drops the action maps that they no longer need. inv.mu must be held.
func (inv *invocationTrace) afterAgentCallbacksReturnedOnLocked(goid uint64, at time.Time, running int) {
	if inv.markAfterAgentCallbacksReturnedOnLocked(goid, at, running) {
		inv.settleAfterAgentActionsOnLocked(goid)
	}
}

// markAfterAgentCallbacksReturnedOnLocked records, when an agent starts or
// ends on the goroutine goid at the platform time at, that the after-agent
// callbacks of the activations that popAgentSpan retired on that goroutine
// earlier returned (see retiredActivation.popGoroutineID), and by when (see
// retiredActivation.afterAgentDoneAt), except for those that are still
// running: running is the number of runs of after-agent callbacks that
// enclose the start or end on that goroutine, other than the ending
// activation's own (see afterAgentCallbackRunsOfTrace), as when a callback
// runs an agent itself. Each start or end on goid records every activation
// retired there except the ones whose callbacks enclose it, and those runs
// nest, so the activations whose callbacks still run are the ones that
// retired first among those on goid not recorded yet: the activation retired
// last may have returned from its callbacks or still run them, and every
// other one encloses it. It reports whether it recorded any. goid 0 is
// unknown and shows nothing. inv.mu must be held.
func (inv *invocationTrace) markAfterAgentCallbacksReturnedOnLocked(goid uint64, at time.Time, running int) bool {
	if goid == 0 {
		return false
	}
	running = max(running, 0)
	var pending []int
	for idx := range inv.retiredActivations {
		if ra := &inv.retiredActivations[idx]; ra.popGoroutineID == goid && !ra.afterAgentDone {
			pending = append(pending, idx)
		}
	}
	if len(pending) <= running {
		return false
	}
	slices.SortStableFunc(pending, func(a, b int) int {
		return cmp.Compare(inv.retiredActivations[a].retiredAtSeq, inv.retiredActivations[b].retiredAtSeq)
	})
	for _, idx := range pending[running:] {
		inv.retiredActivations[idx].markAfterAgentDone(at)
	}
	return true
}

// settleAfterAgentActionsOnLocked drops the after-agent action maps of the
// activations that popAgentSpan retired on the goroutine goid that no longer
// need them (see needsAfterAgentActionsLocked). inv.mu must be held.
func (inv *invocationTrace) settleAfterAgentActionsOnLocked(goid uint64) {
	if goid == 0 {
		return
	}
	inv.settleAfterAgentActionsLocked(func(ra *retiredActivation) bool {
		return ra.popGoroutineID == goid
	})
}

func (inv *invocationTrace) openSpanSetLocked() map[string]bool {
	if len(inv.agentStack) <= 1 {
		return nil
	}
	openSet := make(map[string]bool, len(inv.agentStack)-1)
	for _, r := range inv.agentStack[1:] {
		openSet[r.spanID] = true
	}
	return openSet
}

func customGuardSpanIDsLocked(rec *spanRecord, openSet map[string]bool) []string {
	if rec == nil {
		return nil
	}
	var guards []string
	for cur, p := rec, rec.parent; p != nil && p.kind == spanKindAgent; cur, p = p, p.parent {
		if openSet[p.spanID] && (!p.propagatesChildError || (p.isWorkflow && cur.goroutineID != 0 && !cur.isNodeAgentOf(p))) {
			guards = append(guards, p.spanID)
		}
	}
	return guards
}

func filterOpenSpanIDs(ids []string, openSet map[string]bool) []string {
	var open []string
	for _, g := range ids {
		if openSet[g] && !slices.Contains(open, g) {
			open = append(open, g)
		}
	}
	slices.Sort(open)
	return open
}

func unionOpenSpanIDs(a, b []string, openSet map[string]bool) []string {
	var merged []string
	for _, id := range a {
		if openSet[id] && !slices.Contains(merged, id) {
			merged = append(merged, id)
		}
	}
	for _, id := range b {
		if openSet[id] && !slices.Contains(merged, id) {
			merged = append(merged, id)
		}
	}
	slices.Sort(merged)
	return merged
}

func (inv *invocationTrace) findRetiredActivationIndexLocked(spanID string) int {
	if spanID == "" {
		return -1
	}
	for i := range inv.retiredActivations {
		if !inv.retiredActivations[i].compactedHull && inv.retiredActivations[i].swallower.spanID == spanID {
			return i
		}
	}
	return -1
}

func (inv *invocationTrace) recordRetiredActivationEntryLocked(entry retiredActivation) {
	if entry.swallower.spanID == "" {
		return
	}
	for i := range inv.retiredActivations {
		if !inv.retiredActivations[i].compactedHull && !entry.compactedHull && inv.retiredActivations[i].swallower.spanID == entry.swallower.spanID {
			for _, g := range entry.guardSpanIDs {
				if !slices.Contains(inv.retiredActivations[i].guardSpanIDs, g) {
					inv.retiredActivations[i].guardSpanIDs = append(inv.retiredActivations[i].guardSpanIDs, g)
				}
			}
			for _, a := range entry.ancestorSpanIDs {
				if !slices.Contains(inv.retiredActivations[i].ancestorSpanIDs, a) {
					inv.retiredActivations[i].ancestorSpanIDs = append(inv.retiredActivations[i].ancestorSpanIDs, a)
				}
			}
			for _, id := range entry.openEnclosingSpanIDs {
				if !slices.Contains(inv.retiredActivations[i].openEnclosingSpanIDs, id) {
					inv.retiredActivations[i].openEnclosingSpanIDs = append(inv.retiredActivations[i].openEnclosingSpanIDs, id)
				}
			}
			slices.Sort(inv.retiredActivations[i].openEnclosingSpanIDs)
			ra := &inv.retiredActivations[i]
			ra.enclosingRetiredUnjoined = ra.enclosingRetiredUnjoined || entry.enclosingRetiredUnjoined
			ra.completedOrTask = ra.completedOrTask || entry.completedOrTask
			ra.joinedAtRetire = ra.joinedAtRetire || entry.joinedAtRetire
			ra.hasUnjoinedTask = (ra.hasUnjoinedTask || entry.hasUnjoinedTask || (entry.completedOrTask && !entry.joinedAtRetire)) && !ra.joinedAtRetire
			ra.settledByParent = ra.settledByParent || entry.settledByParent
			ra.peerRetirement = ra.peerRetirement || entry.peerRetirement
			ra.retiredAtSeq = max(ra.retiredAtSeq, entry.retiredAtSeq)
			if !entry.startPlatformTime.IsZero() && (ra.startPlatformTime.IsZero() || entry.startPlatformTime.Before(ra.startPlatformTime)) {
				ra.startPlatformTime = entry.startPlatformTime
			}
			if entry.retiredAtPlatformTime.After(ra.retiredAtPlatformTime) {
				ra.retiredAtPlatformTime = entry.retiredAtPlatformTime
			}
			if entry.parentRetiredAtPlatformTime.After(ra.parentRetiredAtPlatformTime) {
				ra.parentRetiredAtPlatformTime = entry.parentRetiredAtPlatformTime
			}
			if entry.afterAgentDone {
				ra.markAfterAgentDone(entry.afterAgentDoneAt)
			}
			if (entry.completedOrTask || entry.joinedAtRetire) && !entry.unrespondedAtPop {
				ra.unrespondedAtPop = false
			}
			ra.hasAfterAgentCallbacks = ra.hasAfterAgentCallbacks || entry.hasAfterAgentCallbacks
			ra.hasCustomUUIDProvider = ra.hasCustomUUIDProvider || entry.hasCustomUUIDProvider
			ra.normalizeGrantedCredits()
			entry.normalizeGrantedCredits()
			if entry.creditSettled && !ra.creditSettled {
				if gid := ra.creditGroupID; gid != 0 {
					rem := ra.remainingCredits()
					ra.afterAgentCredit = false
					ra.modelCredits = 0
					ra.creditSettled = true
					ra.creditGroupID = 0
					if inv.creditGroupRem != nil {
						inv.creditGroupRem[gid] = max(0, inv.creditGroupRem[gid]-rem)
						inv.settleCreditGroupIfExhaustedLocked(gid, ra.swallower.name, ra.swallower.branch, ra.swallower.path)
					}
				} else {
					ra.afterAgentCredit = false
					ra.modelCredits = 0
					ra.creditSettled = true
				}
			} else if !entry.creditSettled && (!ra.creditSettled || ra.grantedAfterAgentCredit || ra.grantedModelCredits > 0) {
				hadUnassignedCredits := ra.creditGroupID == 0 && ra.remainingCredits() > 0
				delta := 0
				if entry.grantedAfterAgentCredit && !ra.grantedAfterAgentCredit {
					ra.grantedAfterAgentCredit = true
					ra.afterAgentCredit = true
					delta++
				}
				if entry.grantedModelCredits > ra.grantedModelCredits {
					inc := entry.grantedModelCredits - ra.grantedModelCredits
					ra.grantedModelCredits = entry.grantedModelCredits
					ra.modelCredits += inc
					delta += inc
				}
				if delta > 0 {
					ra.creditSettled = false
				}
				if delta > 0 || hadUnassignedCredits {
					if ra.creditGroupID == 0 {
						ra.creditGroupID = inv.allocCreditGroupIDLocked()
						inv.creditGroupRem[ra.creditGroupID] = ra.remainingCredits()
					} else {
						if inv.creditGroupRem == nil {
							inv.creditGroupRem = make(map[uint64]int)
						}
						inv.creditGroupRem[ra.creditGroupID] += delta
					}
				}
			}
			if ra.popGoroutineID == 0 {
				ra.popGoroutineID = entry.popGoroutineID
			}
			if ra.afterAgentActions == (actionMaps{}) && !ra.afterAgentEventSeen {
				ra.afterAgentActions, ra.afterAgentActionsRef = entry.afterAgentActions, entry.afterAgentActionsRef
			}
			return
		}
	}
	entry.normalizeGrantedCredits()
	if entry.completedOrTask && !entry.joinedAtRetire {
		entry.hasUnjoinedTask = true
	}
	entry.guardSpanIDs = slices.Clone(entry.guardSpanIDs)
	entry.ancestorSpanIDs = slices.Clone(entry.ancestorSpanIDs)
	entry.openEnclosingSpanIDs = slices.Clone(entry.openEnclosingSpanIDs)
	slices.Sort(entry.openEnclosingSpanIDs)
	if !entry.compactedHull {
		if credits := entry.remainingCredits(); credits > 0 && !entry.creditSettled && entry.creditGroupID == 0 {
			gid := inv.allocCreditGroupIDLocked()
			entry.creditGroupID = gid
			inv.creditGroupRem[gid] = credits
		}
	} else {
		entry.creditGroupID = 0
	}
	inv.retiredActivations = append(inv.retiredActivations, entry)
}

// parentJoinsChildOnceReturned reports whether p's return guarantees that c's
// run and callbacks have also finished: c's own run ended without remaining
// after-agent callbacks still running after popAgentSpan ((c.runDone &&
// !c.completed) || (c.completed && !c.hasAfterAgentCallbacks)), c ran on the
// same goroutine as p (or without goroutine tracking in synthetic tests), or p
// is a structured ADK parent that joins c before returning (see
// joinsOnceReturned). A custom parent returning on its own goroutine without
// joining a detached child goroutine (including one that popped this plugin's
// AfterAgentCallback while another AfterAgentCallback is still running on that
// child goroutine) returns false.
func parentJoinsChildOnceReturned(p, c *spanRecord) bool {
	if p == nil || c == nil {
		return false
	}
	if (c.runDone && !c.completed) || (c.completed && !c.hasAfterAgentCallbacks) {
		return true
	}
	if p.goroutineID == 0 || c.goroutineID == 0 || p.goroutineID == c.goroutineID {
		return true
	}
	return p.joinsOnceReturned(c)
}

// unjoinedDirectParentRetiring reports whether ra has no tracked
// openEnclosingSpanIDs yet its direct parent pRec is a custom agent retiring on
// another goroutine without having joined it. Entries recorded via
// recordRetiredActivationLocked whose hop to an open parent satisfied
// parentJoinsChildOnceReturned (including non-pop runDone children and
// callback-free completed children on another goroutine) carry that parent in
// openEnclosingSpanIDs, and entries retiring in the same batch are already
// classified in recordRetiredActivationLocked.
func (ra *retiredActivation) unjoinedDirectParentRetiring(retiredBySpan map[string]*spanRecord) bool {
	if ra.enclosingRetiredUnjoined || len(ra.openEnclosingSpanIDs) > 0 || ra.afterAgentDone || !ra.parentRetiredAtPlatformTime.IsZero() {
		return false
	}
	if retiredBySpan[ra.swallower.spanID] != nil {
		return false
	}
	pRec := retiredBySpan[ra.parentSpanID]
	return pRec != nil &&
		ra.goroutineID != 0 && pRec.goroutineID != 0 && ra.goroutineID != pRec.goroutineID &&
		!pRec.propagatesChildError && !pRec.isParallel && !pRec.isWorkflow
}

func (inv *invocationTrace) recordRetiredActivationsAtLocked(openSet map[string]bool, retiredAt time.Time, recs ...*spanRecord) {
	if len(recs) == 0 {
		return
	}
	retiredParents := make(map[string]bool, len(recs))
	retiredBySpan := make(map[string]*spanRecord, len(recs))
	retiredAncestorAgents := make(map[agentIdentityKey]bool)
	// runOver holds the retired parents whose runs returned. When every hop
	// from a descendant up to such a parent joined the activation below it (see
	// parentJoinsChildOnceReturned), the descendant's run and after-agent
	// callbacks have also finished by then; an unjoined detached child goroutine
	// under a custom parent is excluded via enclosingRetiredUnjoined because an
	// outer ancestor or callback on that child goroutine can still emit events.
	runOver := make(map[string]bool, len(recs))
	var pruneKeys []agentIdentityKey
	addPruneKey := func(name, branch string) {
		k := agentIdentityKey{name: name, branch: branch}
		if !slices.Contains(pruneKeys, k) {
			pruneKeys = append(pruneKeys, k)
		}
	}
	for _, rec := range recs {
		if rec != nil && rec.kind == spanKindAgent && rec.spanID != "" {
			retiredParents[rec.spanID] = true
			retiredBySpan[rec.spanID] = rec
			if rec.completed || rec.runDone {
				runOver[rec.spanID] = true
			}
			addPruneKey(rec.name, rec.branch)
			for p := rec.parent; p != nil && p.kind == spanKindAgent; p = p.parent {
				if p.spanID != "" && !openSet[p.spanID] {
					retiredAncestorAgents[agentIdentityKey{name: p.name, branch: p.branch}] = true
					addPruneKey(p.name, p.branch)
				}
			}
		}
	}
	for _, rec := range recs {
		inv.recordRetiredActivationLocked(rec, openSet, runOver, retiredAt)
	}
	unrespIdx := inv.newUnrespondedRetiredIndexLocked()
	if len(retiredParents) > 0 {
		affectedByRetired := func(ra *retiredActivation) bool {
			if retiredParents[ra.parentSpanID] {
				return true
			}
			for _, g := range ra.guardSpanIDs {
				if retiredParents[g] {
					return true
				}
			}
			for _, a := range ra.ancestorSpanIDs {
				if retiredParents[a] {
					return true
				}
			}
			for _, x := range ra.openEnclosingSpanIDs {
				if retiredParents[x] || !openSet[x] {
					return true
				}
			}
			return false
		}
		var st afterAgentState
		for i := range inv.retiredActivations {
			ra := &inv.retiredActivations[i]
			if !affectedByRetired(ra) {
				continue
			}
			if ra.unjoinedDirectParentRetiring(retiredBySpan) {
				ra.enclosingRetiredUnjoined = true
			}
			anySatisfiedByRunOver := false
			if len(ra.openEnclosingSpanIDs) > 0 {
				var nextOpen []string
				addOpen := func(id string) {
					if id != "" && !slices.Contains(nextOpen, id) {
						nextOpen = append(nextOpen, id)
					}
				}
				for _, x := range ra.openEnclosingSpanIDs {
					if openSet[x] {
						addOpen(x)
						continue
					}
					if runOver[x] {
						anySatisfiedByRunOver = true
						continue
					}
					if r := retiredBySpan[x]; r != nil {
						resolved := false
						for c, p := r, r.parent; p != nil && p.kind == spanKindAgent; c, p = p, p.parent {
							if !parentJoinsChildOnceReturned(p, c) {
								break
							}
							if p.spanID != "" && openSet[p.spanID] {
								addOpen(p.spanID)
								resolved = true
								break
							}
							if runOver[p.spanID] {
								anySatisfiedByRunOver = true
								resolved = true
								break
							}
						}
						if !resolved {
							ra.enclosingRetiredUnjoined = true
						}
					} else {
						xIdx := slices.Index(ra.ancestorSpanIDs, x)
						if xIdx < 0 {
							// x was intersected away from ra.ancestorSpanIDs (a multi-parent
							// hull) and disappeared from openSet outside
							// recordRetiredActivationsAtLocked, so intermediate ancestors
							// below ra.ancestorSpanIDs[0] may still be open.
							ra.enclosingRetiredUnjoined = true
							continue
						}
						resolved := false
						for _, a := range ra.ancestorSpanIDs[xIdx+1:] {
							if runOver[a] {
								anySatisfiedByRunOver = true
								resolved = true
								break
							}
							if openSet[a] {
								addOpen(a)
								resolved = true
								break
							}
						}
						if !resolved {
							ra.enclosingRetiredUnjoined = true
						}
					}
				}
				slices.Sort(nextOpen)
				ra.openEnclosingSpanIDs = nextOpen
			}
			parentRunOver := !ra.enclosingRetiredUnjoined &&
				!openSet[ra.parentSpanID] &&
				len(ra.openEnclosingSpanIDs) == 0 &&
				(anySatisfiedByRunOver || runOver[ra.parentSpanID])
			if !retiredAt.IsZero() && ra.parentRetiredAtPlatformTime.IsZero() && (retiredParents[ra.parentSpanID] || !openSet[ra.parentSpanID]) && parentRunOver {
				ra.parentRetiredAtPlatformTime = retiredAt
			}
			if parentRunOver {
				ra.markAfterAgentDone(retiredAt)
			}
			if st.swallowers == nil {
				st = inv.afterAgentStateLocked()
			}
			if (ra.swallower.anyName || ra.swallower.anyBranch) && len(ra.guardSpanIDs) > 0 {
				ra.guardSpanIDs = filterOpenSpanIDs(ra.guardSpanIDs, openSet)
			}
			hasOpenGuard := slices.ContainsFunc(ra.guardSpanIDs, func(g string) bool {
				return openSet[g]
			})
			// Once the parent retired, only an open custom ancestor that holds
			// the activation's events can still deliver its after-agent event,
			// so without one the maps go, unless the event may still recover a
			// cause that the activation swallowed, or one that an open span under
			// it carries and moves up with it among its swallowers once that span
			// retires (see afterAgentState). With one, they stay as long
			// as the activation needs them (see needsAfterAgentActionsLocked).
			if !hasOpenGuard && (retiredParents[ra.parentSpanID] || !openSet[ra.parentSpanID]) && (ra.compactedHull || (!st.swallowers[ra.swallower.spanID] && !st.underOpen[ra.swallower.spanID])) {
				ra.dropAfterAgentEvent()
			}
			if !ra.swallower.anyName && !ra.swallower.anyBranch {
				addPruneKey(
					ra.swallower.name,
					ra.swallower.branch,
				)
			}
		}
		inv.settleAfterAgentActionsLocked(func(ra *retiredActivation) bool {
			return affectedByRetired(ra) || (!ra.swallower.anyName && !ra.swallower.anyBranch && retiredAncestorAgents[agentIdentityKey{name: ra.swallower.name, branch: ra.swallower.branch}])
		})
		inv.retiredActivations = slices.DeleteFunc(inv.retiredActivations, func(ra retiredActivation) bool {
			if !ra.swallower.anyName && !ra.swallower.anyBranch {
				return false
			}
			if ra.holdsAfterAgentEvent() || (ra.peerRetirement && inv.hasOpenPeerLocked(ra)) || unrespIdx.matches(ra.swallower) {
				return false
			}
			return !slices.ContainsFunc(ra.guardSpanIDs, func(g string) bool {
				return openSet[g]
			})
		})
	}
	for _, k := range pruneKeys {
		inv.pruneSettledRetiredActivationsWithIndexLocked(k.name, k.branch, unrespIdx)
	}
	inv.enforceRetiredActivationCapWithIndexLocked(unrespIdx)
	inv.shrinkRetiredActivationsLocked()
}

func (inv *invocationTrace) recordRetiredActivationLocked(rec *spanRecord, openSet, runOver map[string]bool, retiredAt time.Time) {
	if rec == nil || rec.kind != spanKindAgent || rec.spanID == "" {
		return
	}
	var ancestorSpanIDs []string
	for p := rec.parent; p != nil && p.kind == spanKindAgent; p = p.parent {
		if p.spanID != "" {
			ancestorSpanIDs = append(ancestorSpanIDs, p.spanID)
		}
	}
	// An activation that an enclosing retirement left open after its run was
	// over (see spanRecord.runEndSeq) created all its events by then, so its
	// retirement is dated there. Dated when a buffered event later lets a
	// parent retire it, which can come after a later activation of the same
	// agent started, it would look like a peer that could have created the
	// later activation's events.
	retiredAtSeq := inv.nextSeqLocked()
	if rec.runEndSeq != 0 && rec.runEndSeq < retiredAtSeq {
		retiredAtSeq = rec.runEndSeq
		if !rec.runEndPlatformTime.IsZero() {
			retiredAt = rec.runEndPlatformTime
		}
	}
	// Any transient popRetirement metadata attached to rec goes into the entry
	// now, before recordRetiredActivationsAtLocked prunes the entries of rec's
	// agent, branch and node path, which keeps an entry that still expects
	// its after-agent event or holds a credit (see
	// pruneSettledRetiredActivationsLocked).
	var pop popRetirement
	if rec.popRetirement != nil {
		pop = *rec.popRetirement
		rec.popRetirement = nil
	}
	newAfterAgentCredit := pop.afterAgentCredit && !rec.recordedAfterAgentCredit
	if pop.afterAgentCredit {
		rec.recordedAfterAgentCredit = true
	}
	newModelCredits := max(0, pop.modelCredits-rec.recordedModelCredits)
	if pop.modelCredits > rec.recordedModelCredits {
		rec.recordedModelCredits = pop.modelCredits
	}
	var openEnclosing []string
	var enclosingRetiredUnjoined bool
	var parentDone bool
	var earlierAncestorDone bool
	for c, p := rec, rec.parent; p != nil && p.kind == spanKindAgent; c, p = p, p.parent {
		if !parentJoinsChildOnceReturned(p, c) {
			openEnclosing = nil
			parentDone = false
			earlierAncestorDone = false
			enclosingRetiredUnjoined = true
			break
		}
		if p.spanID != "" && openSet[p.spanID] {
			openEnclosing = []string{p.spanID}
			break
		}
		if runOver[p.spanID] {
			openEnclosing = nil
			parentDone = true
			break
		}
		if rec.runEndSeq != 0 && (p.completed || p.runDone) {
			earlierAncestorDone = true
		}
	}
	if !parentDone && len(openEnclosing) == 0 && earlierAncestorDone {
		parentDone = true
	}
	if !parentDone && len(openEnclosing) == 0 && rec.parent != nil && rec.parent.kind == spanKindAgent {
		enclosingRetiredUnjoined = true
	}
	entry := retiredActivation{
		swallower: retiredSwallower{
			spanID: rec.spanID,
			seq:    rec.seq,
			name:   rec.name,
			branch: rec.branch,
			path:   rec.path,
		},
		parentSpanID:             rec.parentSpanID,
		guardSpanIDs:             customGuardSpanIDsLocked(rec, openSet),
		ancestorSpanIDs:          ancestorSpanIDs,
		openEnclosingSpanIDs:     openEnclosing,
		enclosingRetiredUnjoined: enclosingRetiredUnjoined,
		retiredAtSeq:             retiredAtSeq,
		completedOrTask:          rec.completed || rec.taskMode,
		joinedAtRetire:           rec.completed || rec.runDone,
		hasUnjoinedTask:          rec.taskMode && !rec.completed && !rec.runDone,
		concurrentPeer:           rec.concurrentPeer,
		unrespondedAtPop:         !rec.ownResponse && !rec.contextEnded && !rec.taskMode,
		hasAfterAgentCallbacks:   rec.hasAfterAgentCallbacks,
		hasCustomUUIDProvider:    rec.hasCustomUUIDProvider,
		afterAgentActions:        pop.afterAgentActions,
		afterAgentActionsRef:     pop.afterAgentActionsRef,
		afterAgentCredit:         newAfterAgentCredit,
		modelCredits:             newModelCredits,
		grantedAfterAgentCredit:  rec.recordedAfterAgentCredit,
		grantedModelCredits:      rec.recordedModelCredits,
		peerRetirement:           pop.peerRetirement,
		popGoroutineID:           pop.goroutineID,
		goroutineID:              rec.goroutineID,
		startPlatformTime:        rec.startPlatformTime,
		retiredAtPlatformTime:    retiredAt,
	}
	if parentDone {
		if !retiredAt.IsZero() {
			entry.parentRetiredAtPlatformTime = retiredAt
		}
		entry.markAfterAgentDone(retiredAt)
	}
	inv.recordRetiredActivationEntryLocked(entry)
}

func (inv *invocationTrace) markSettledRetiredActivationsLocked(recoveringSpanID string, includeSelf bool) {
	if recoveringSpanID == "" || len(inv.retiredActivations) == 0 {
		return
	}
	openBySpan := make(map[string]*spanRecord, len(inv.agentStack))
	survivingSwallowers := make(map[string]bool)
	for _, rec := range inv.agentStack {
		if rec.kind == spanKindAgent && rec.spanID != "" {
			openBySpan[rec.spanID] = rec
		}
		for _, c := range rec.retiredCauses {
			for _, s := range c.swallowers {
				if s.spanID != "" {
					survivingSwallowers[s.spanID] = true
				}
			}
		}
	}
	openCalls := inv.openCallOwnersLocked(nil)
	unrecBySpan := make(map[string]bool)
	for _, e := range inv.unrecoveredErrors {
		if !e.collateral && e.spanID != "" {
			unrecBySpan[e.spanID] = true
		}
	}
	spanHasSurvivingFailure := func(spanID string, allowOpen bool) bool {
		if spanID == "" || survivingSwallowers[spanID] || unrecBySpan[spanID] || openCalls[spanID] {
			return true
		}
		if rec := openBySpan[spanID]; rec != nil {
			if !allowOpen || len(rec.retiredCauses) > 0 || rec.contextFailureCause() != nil {
				return true
			}
		}
		return false
	}

	var pruneKeys []agentIdentityKey
	for i := range inv.retiredActivations {
		ra := &inv.retiredActivations[i]
		// Fail closed on compacted hulls: a hull merges multiple activations
		// and intersects their ancestor chains, so neither individual member
		// failures nor unshared intermediate ancestor failures can be verified.
		if !ra.joinedAtRetire || ra.settledByParent || ra.compactedHull {
			continue
		}
		isSelf := ra.swallower.spanID == recoveringSpanID
		if !((includeSelf && isSelf) || slices.Contains(ra.ancestorSpanIDs, recoveringSpanID)) {
			continue
		}
		if spanHasSurvivingFailure(ra.swallower.spanID, false) {
			continue
		}
		blocked := false
		if !isSelf {
			for _, ancID := range ra.ancestorSpanIDs {
				isRecovering := ancID == recoveringSpanID
				if spanHasSurvivingFailure(ancID, isRecovering) {
					blocked = true
					break
				}
				if isRecovering {
					break
				}
			}
		}
		if blocked {
			continue
		}
		ra.settledByParent = true
		k := agentIdentityKey{name: ra.swallower.name, branch: ra.swallower.branch}
		if !slices.Contains(pruneKeys, k) {
			pruneKeys = append(pruneKeys, k)
		}
	}
	for _, k := range pruneKeys {
		inv.pruneSettledRetiredActivationsLocked(k.name, k.branch)
	}
}

// unjoinedTaskRetirement reports that a finish_task function response retired
// the activation (completedOrTask without completed) before its run
// synchronized (!joinedAtRetire), so its goroutine may still be creating
// events after retiredAtPlatformTime. Both fields are part of compactKey, so
// a compacted hull has the same value as each of its members.
func (ra retiredActivation) unjoinedTaskRetirement() bool {
	return ra.hasUnjoinedTask || (ra.completedOrTask && !ra.joinedAtRetire)
}

// canOwnEventCreatedAt reports whether ra can have created an event created
// at evCreatedAt. An activation without after-agent callbacks creates no event
// after it retired, unless it is an unjoined finish_task retirement (see
// unjoinedTaskRetirement), whose run may outlast its retirement, and neither
// does one whose after-agent event was already delivered (see
// afterAgentEventSeen). The enclosing activation's run returns only after ra's
// run did, so its retirement bounds ra's events in either case. So does the
// time at which ra's after-agent callbacks were seen to have returned (see
// afterAgentDoneAt), which is only seen once ra's run returned: ADK builds
// their event before the goroutine that ran them starts or ends another
// agent, and before the run of ra's parent returns, so an event created later
// is not that event, even if that event never comes or is not recognized. An
// agent that the callbacks run on that goroutine shows nothing (see
// markAfterAgentCallbacksReturnedOnLocked).
func (ra retiredActivation) canOwnEventCreatedAt(evCreatedAt time.Time, hasEventID bool) bool {
	return ra.canOwnEventAt(evCreatedAt, hasEventID, false)
}

// canOwnEventAt is canOwnEventCreatedAt for an event that, if afterAgentEvent
// is set, carries an action map of the context in which ra's after-agent
// callbacks ran (see actionMaps). Such an event is ra's after-agent event
// whenever it was created, so post-retirement upper bounds
// (retiredAtPlatformTime, afterAgentDoneAt, and parentRetiredAtPlatformTime)
// do not bound it, as for callbacks that run an agent on another goroutine and
// wait for it, which the goroutine that ran them does not show.
func (ra retiredActivation) canOwnEventAt(evCreatedAt time.Time, hasEventID, afterAgentEvent bool) bool {
	if evCreatedAt.IsZero() {
		return true
	}
	if !hasEventID {
		return false
	}
	if !ra.startPlatformTime.IsZero() && evCreatedAt.Before(ra.startPlatformTime) {
		return false
	}
	if !afterAgentEvent {
		if (!ra.hasAfterAgentCallbacks || ra.afterAgentEventSeen) && !ra.unjoinedTaskRetirement() && !ra.retiredAtPlatformTime.IsZero() && evCreatedAt.After(ra.retiredAtPlatformTime) {
			return false
		}
		if !ra.afterAgentDoneAt.IsZero() && evCreatedAt.After(ra.afterAgentDoneAt) {
			return false
		}
		if !ra.parentRetiredAtPlatformTime.IsZero() && evCreatedAt.After(ra.parentRetiredAtPlatformTime) {
			return false
		}
	}
	return true
}

func (ra retiredActivation) excludedByEventTimestamp(laterSpanID string, laterSeq, laterGoroutineID uint64, laterConcurrentPeer bool, laterStartPlatformTime, evCreatedAt time.Time, hasEventID bool) bool {
	// Events created before a pending hull's callbacks or parent return, or
	// while a mapless entry/hull with after-agent callbacks (whose event is
	// unrecognized without action maps) can still create events, cannot be
	// excluded.
	if (ra.afterAgentPending || (ra.hasAfterAgentCallbacks && ra.afterAgentActions == (actionMaps{}))) && ra.canOwnEventCreatedAt(evCreatedAt, hasEventID) {
		return false
	}
	settledMatch := ra.joinedAtRetire && ra.settledByParent
	joinedResponded := ra.joinedAtRetire && !ra.unrespondedAtPop
	if (!ra.compactedHull && ra.swallower.spanID == laterSpanID) ||
		!(ra.completedOrTask || settledMatch || joinedResponded) ||
		ra.concurrentPeer ||
		laterConcurrentPeer ||
		(ra.unrespondedAtPop && !settledMatch) ||
		(!(settledMatch || joinedResponded) && (laterGoroutineID == 0 || ra.goroutineID != laterGoroutineID)) ||
		ra.swallower.seq >= laterSeq ||
		ra.retiredAtSeq >= laterSeq ||
		!hasEventID ||
		evCreatedAt.IsZero() ||
		laterStartPlatformTime.IsZero() {
		return false
	}
	if evCreatedAt.After(laterStartPlatformTime) {
		return true
	}
	// On a coarse clock the event can share the tick in which the later
	// activation started. A joined activation's run returned before it
	// retired, so an event that ra could not have created by then (see
	// canOwnEventCreatedAt) is the later activation's.
	return (settledMatch || joinedResponded) &&
		!evCreatedAt.Before(laterStartPlatformTime) &&
		!ra.canOwnEventCreatedAt(evCreatedAt, hasEventID)
}

func minPlatformTime(a, b time.Time) time.Time {
	if a.IsZero() || b.IsZero() {
		return time.Time{}
	}
	if a.Before(b) {
		return a
	}
	return b
}

func maxPlatformTime(a, b time.Time) time.Time {
	if a.IsZero() || b.IsZero() {
		return time.Time{}
	}
	if a.After(b) {
		return a
	}
	return b
}

// pruneOrphanedCreditGroupsLocked clears stale creditGroupIDs on zero-credit
// entries and deletes creditGroupRem entries with no remaining live holder.
func (inv *invocationTrace) pruneOrphanedCreditGroupsLocked() {
	if len(inv.creditGroupRem) == 0 {
		return
	}
	live := make(map[uint64]bool, len(inv.creditGroupRem))
	for i := range inv.retiredActivations {
		ra := &inv.retiredActivations[i]
		if ra.compactedHull {
			if len(ra.foldedCredits) > 0 {
				kept := ra.foldedCredits[:0]
				for _, fc := range ra.foldedCredits {
					if fc.creditGroupID != 0 && fc.credits > 0 {
						live[fc.creditGroupID] = true
						kept = append(kept, fc)
					}
				}
				if len(kept) == 0 {
					ra.foldedCredits = nil
				} else {
					ra.foldedCredits = kept
				}
			}
			continue
		}
		if ra.creditGroupID != 0 {
			if ra.afterAgentCredit || ra.modelCredits > 0 {
				live[ra.creditGroupID] = true
			} else {
				ra.creditGroupID = 0
			}
		}
	}
	for gid := range inv.creditGroupRem {
		if !live[gid] {
			delete(inv.creditGroupRem, gid)
		}
	}
}

// pruneSettledRetiredActivationsLocked drops the entries of (name, branch) in
// retiredActivations across all node paths that nothing needs anymore and
// compacts the rest per (name, branch, path) bucket (see below). An entry
// stays as long as a recorded cause names it as a swallower, or an open custom
// ancestor of it may still deliver its events (guardSpanIDs). An entry also
// stays, whatever its ancestors, while it still recognizes an after-agent
// event that may come (see needsAfterAgentActionsLocked), has an open span
// under it, whose failure moves up with it among its swallowers once that span
// retires, while a path-compatible unrespondedRetired credit remains
// outstanding (so credit attribution cannot be stranded or stolen), or while
// it counted as a peer retirement and a peer of it is open (see
// soleOpenAuthorLocked), as for the agent of a workflow node or the agent that
// the runner runs, which have no custom ancestor. Such an entry is not
// compacted either while it keeps its maps, holds or settled a credit with
// compatible unrespondedRetired credits still outstanding, has an open span
// under it, or, while a peer of it is open, counted as a peer retirement, so
// that the entries that the peer's own events depend on stay as many as the
// activations that retired while it runs. An entry's maps go once nothing
// needs them, so that, as for the activations of an agent that a custom agent
// runs one after another, a wave of activations whose runs are over and whose
// after-agent events cannot be taken for another activation's leaves no more
// entries than compaction keeps, and the entries that keep their maps only
// because their after-agent events may still come are bounded too (see
// foldPendingAfterAgentLocked and enforceRetiredActivationCapLocked). inv.mu
// must be held.
func (inv *invocationTrace) pruneSettledRetiredActivationsLocked(name, branch string) {
	inv.pruneSettledRetiredActivationsWithIndexLocked(name, branch, inv.newUnrespondedRetiredIndexLocked())
}

func (inv *invocationTrace) pruneSettledRetiredActivationsWithIndexLocked(name, branch string, unrespIdx unrespondedRetiredIndex) {
	defer inv.pruneOrphanedCreditGroupsLocked()
	if len(inv.retiredActivations) == 0 {
		return
	}
	openSet := inv.openSpanSetLocked()
	for idx := range inv.retiredActivations {
		ra := &inv.retiredActivations[idx]
		if !ra.swallower.anyName && !ra.swallower.anyBranch && ra.swallower.name == name && ra.swallower.branch == branch && len(ra.guardSpanIDs) > 0 {
			ra.guardSpanIDs = filterOpenSpanIDs(ra.guardSpanIDs, openSet)
		}
	}
	inv.settleAfterAgentActionsLocked(func(ra *retiredActivation) bool {
		if ra.swallower.anyName || ra.swallower.anyBranch || ra.swallower.name != name || ra.swallower.branch != branch {
			return false
		}
		if ra.afterAgentDone {
			return true
		}
		return !openSet[ra.parentSpanID] && !slices.ContainsFunc(ra.guardSpanIDs, func(g string) bool {
			return openSet[g]
		})
	})
	st := inv.afterAgentStateLocked()
	surviving := st.swallowers
	openPeer := inv.hasOpenPeerLocked
	// pinned reports that ra stays as it is, neither dropped nor compacted.
	// A peer retirement stays uncompacted while a peer of it is open, as it
	// stays undeleted below: soleOpenAuthorLocked gives an open activation an
	// event only while every retired peer that retiredPeerCount counts has
	// an entry of its own, so a hull that took the place of such entries
	// would keep the open activation's recovery from counting. Credit-bearing
	// or creditSettled entries stay pinned while compatible
	// unrespondedRetired credits remain outstanding.
	pinned := func(ra retiredActivation) bool {
		return ra.holdsAfterAgentEvent() ||
			((ra.afterAgentCredit || ra.modelCredits > 0 || ra.creditSettled || len(ra.foldedCredits) > 0) && unrespIdx.matches(ra.swallower)) ||
			(!ra.compactedHull && st.underOpen[ra.swallower.spanID]) ||
			(ra.peerRetirement && openPeer(ra))
	}
	inv.retiredActivations = slices.DeleteFunc(inv.retiredActivations, func(ra retiredActivation) bool {
		if ra.swallower.anyName || ra.swallower.anyBranch || ra.swallower.name != name || ra.swallower.branch != branch || (!ra.compactedHull && surviving[ra.swallower.spanID]) ||
			pinned(ra) ||
			unrespIdx.matches(ra.swallower) {
			return false
		}
		return !slices.ContainsFunc(ra.guardSpanIDs, func(g string) bool {
			return openSet[g]
		})
	})
	inv.foldPendingAfterAgentWithIndexLocked(name, branch, st, unrespIdx)

	type compactKey struct {
		name                     string
		branch                   string
		path                     string
		guards                   string
		openEnclosing            string
		openParentSpanID         string
		completedOrTask          bool
		joinedAtRetire           bool
		hasUnjoinedTask          bool
		settledByParent          bool
		enclosingRetiredUnjoined bool
		concurrentPeer           bool
		unrespondedAtPop         bool
		hasAfterAgentCallbacks   bool
		afterAgentEventSeen      bool
		hasCustomUUIDProvider    bool
		hasStartTime             bool
		hasRetiredTime           bool
		hasParentRetiredTime     bool
	}
	type bucket struct {
		firstIdx    int
		lastIdx     int
		anyFirstIdx int
		anyLastIdx  int
		members     []int
	}
	buckets := make(map[compactKey]*bucket)
	beforeEntry := func(a, b retiredActivation, aIdx, bIdx int) bool {
		if a.retiredAtSeq != b.retiredAtSeq {
			return a.retiredAtSeq < b.retiredAtSeq
		}
		if a.swallower.seq != b.swallower.seq {
			return a.swallower.seq < b.swallower.seq
		}
		return aIdx < bIdx
	}
	for idx := range inv.retiredActivations {
		ra := &inv.retiredActivations[idx]
		if ra.swallower.anyName || ra.swallower.anyBranch || ra.swallower.name != name || ra.swallower.branch != branch || (!ra.compactedHull && surviving[ra.swallower.spanID]) || pinned(*ra) {
			continue
		}
		openParent := ""
		if openSet[ra.parentSpanID] {
			openParent = ra.parentSpanID
		}
		ck := compactKey{
			name:                     ra.swallower.name,
			branch:                   ra.swallower.branch,
			path:                     ra.swallower.path,
			guards:                   strings.Join(ra.guardSpanIDs, "\x00"),
			openEnclosing:            strings.Join(ra.openEnclosingSpanIDs, "\x00"),
			openParentSpanID:         openParent,
			completedOrTask:          ra.completedOrTask,
			joinedAtRetire:           ra.joinedAtRetire,
			hasUnjoinedTask:          ra.unjoinedTaskRetirement(),
			settledByParent:          ra.settledByParent,
			enclosingRetiredUnjoined: ra.enclosingRetiredUnjoined,
			concurrentPeer:           ra.concurrentPeer,
			unrespondedAtPop:         ra.unrespondedAtPop,
			hasAfterAgentCallbacks:   ra.hasAfterAgentCallbacks,
			afterAgentEventSeen:      ra.afterAgentEventSeen,
			hasCustomUUIDProvider:    ra.hasCustomUUIDProvider,
			hasStartTime:             !ra.startPlatformTime.IsZero(),
			hasRetiredTime:           !ra.retiredAtPlatformTime.IsZero(),
			hasParentRetiredTime:     !ra.parentRetiredAtPlatformTime.IsZero(),
		}
		b, ok := buckets[ck]
		if !ok {
			b = &bucket{
				firstIdx:    -1,
				lastIdx:     -1,
				anyFirstIdx: idx,
				anyLastIdx:  idx,
			}
			buckets[ck] = b
		} else {
			if beforeEntry(*ra, inv.retiredActivations[b.anyFirstIdx], idx, b.anyFirstIdx) {
				b.anyFirstIdx = idx
			}
			if beforeEntry(inv.retiredActivations[b.anyLastIdx], *ra, b.anyLastIdx, idx) {
				b.anyLastIdx = idx
			}
		}
		if !ra.compactedHull {
			if b.firstIdx == -1 || beforeEntry(*ra, inv.retiredActivations[b.firstIdx], idx, b.firstIdx) {
				b.firstIdx = idx
			}
			if b.lastIdx == -1 || beforeEntry(inv.retiredActivations[b.lastIdx], *ra, b.lastIdx, idx) {
				b.lastIdx = idx
			}
		}
		b.members = append(b.members, idx)
	}
	needsCompaction := false
	for _, b := range buckets {
		if len(b.members) > 3 {
			needsCompaction = true
			break
		}
	}
	if !needsCompaction {
		inv.enforceRetiredActivationCapWithIndexLocked(unrespIdx)
		inv.shrinkRetiredActivationsLocked()
		return
	}
	keep := make(map[int]bool, 3*len(buckets))
	midHullAt := make(map[int]retiredActivation, len(buckets))
	for _, b := range buckets {
		firstIdx, lastIdx := b.firstIdx, b.lastIdx
		if firstIdx == -1 {
			firstIdx, lastIdx = b.anyFirstIdx, b.anyLastIdx
		}
		keep[firstIdx] = true
		keep[lastIdx] = true
		if len(b.members) <= 3 {
			for _, idx := range b.members {
				keep[idx] = true
			}
			continue
		}
		firstMidIdx := -1
		var hull retiredActivation
		for _, idx := range b.members {
			if idx == firstIdx || idx == lastIdx {
				continue
			}
			m := inv.retiredActivations[idx]
			if firstMidIdx == -1 {
				// The fields of compactKey — joinedAtRetire, settledByParent,
				// afterAgentEventSeen, and hasCustomUUIDProvider among them —
				// are the same for every member of the bucket, so the hull
				// keeps those of its first member.
				firstMidIdx = idx
				hull = m
				hull.guardSpanIDs = slices.Clone(m.guardSpanIDs)
				hull.ancestorSpanIDs = slices.Clone(m.ancestorSpanIDs)
				hull.openEnclosingSpanIDs = filterOpenSpanIDs(m.openEnclosingSpanIDs, openSet)
				hull.compactedHull = true
				hull.afterAgentActions = actionMaps{}
				hull.afterAgentActionsRef = nil
				hull.afterAgentCredit = false
				hull.modelCredits = 0
				hull.creditSettled = false
				hull.creditGroupID = 0
				hull.peerRetirement = false
				hull.popGoroutineID = 0
				continue
			}
			if m.swallower.seq > hull.swallower.seq {
				hull.swallower.seq = m.swallower.seq
				if m.swallower.spanID != "" {
					hull.swallower.spanID = m.swallower.spanID
				}
			}
			hull.retiredAtSeq = max(hull.retiredAtSeq, m.retiredAtSeq)
			if hull.parentSpanID != m.parentSpanID {
				hull.parentSpanID = ""
			}
			if hull.goroutineID != m.goroutineID {
				hull.goroutineID = 0
			}
			hull.ancestorSpanIDs = slices.DeleteFunc(hull.ancestorSpanIDs, func(id string) bool {
				return !slices.Contains(m.ancestorSpanIDs, id)
			})
			hull.openEnclosingSpanIDs = unionOpenSpanIDs(hull.openEnclosingSpanIDs, m.openEnclosingSpanIDs, openSet)
			hull.enclosingRetiredUnjoined = hull.enclosingRetiredUnjoined || m.enclosingRetiredUnjoined
			hull.afterAgentEventSeen = hull.afterAgentEventSeen && m.afterAgentEventSeen
			hull.hasCustomUUIDProvider = hull.hasCustomUUIDProvider && m.hasCustomUUIDProvider
			hull.startPlatformTime = minPlatformTime(hull.startPlatformTime, m.startPlatformTime)
			hull.retiredAtPlatformTime = maxPlatformTime(hull.retiredAtPlatformTime, m.retiredAtPlatformTime)
			hull.parentRetiredAtPlatformTime = maxPlatformTime(hull.parentRetiredAtPlatformTime, m.parentRetiredAtPlatformTime)
			hull.afterAgentDone = hull.afterAgentDone && m.afterAgentDone
			hull.afterAgentDoneAt = maxPlatformTime(hull.afterAgentDoneAt, m.afterAgentDoneAt)
		}
		if firstMidIdx != -1 {
			midHullAt[firstMidIdx] = hull
		}
	}
	out := inv.retiredActivations[:0]
	for idx, ra := range inv.retiredActivations {
		if hull, ok := midHullAt[idx]; ok {
			out = append(out, hull)
			continue
		}
		if ra.swallower.anyName || ra.swallower.anyBranch || ra.swallower.name != name || ra.swallower.branch != branch || (!ra.compactedHull && surviving[ra.swallower.spanID]) || keep[idx] || pinned(ra) {
			out = append(out, ra)
		}
	}
	clear(inv.retiredActivations[len(out):])
	inv.retiredActivations = out
	inv.enforceRetiredActivationCapWithIndexLocked(unrespIdx)
	inv.shrinkRetiredActivationsLocked()
}

// maxRetiredActivationsPerInvocation bounds the total number of
// retiredActivation entries retained per invocation across all agents,
// branches, and node paths.
const maxRetiredActivationsPerInvocation = 128

func foldRetiredCapHull(members []retiredActivation, openSet map[string]bool) retiredActivation {
	m0 := members[0]
	hull := m0
	hull.swallower.spanID = ""
	hull.guardSpanIDs = filterOpenSpanIDs(m0.guardSpanIDs, openSet)
	hull.ancestorSpanIDs = slices.Clone(m0.ancestorSpanIDs)
	hull.openEnclosingSpanIDs = filterOpenSpanIDs(m0.openEnclosingSpanIDs, openSet)
	hull.hasUnjoinedTask = m0.unjoinedTaskRetirement()
	hull.afterAgentPending = m0.holdsAfterAgentEvent()
	hull.afterAgentEventSeen = !hull.afterAgentPending && m0.afterAgentEventSeen
	hull.creditSettled = m0.creditSettled
	hull.compactedHull = true
	hull.afterAgentActions = actionMaps{}
	hull.afterAgentActionsRef = nil
	hull.afterAgentCredit = false
	hull.modelCredits = 0
	hull.grantedAfterAgentCredit = false
	hull.grantedModelCredits = 0
	hull.creditGroupID = 0
	hull.foldedCredits = nil
	hull.popGoroutineID = 0
	var foldedCredits []foldedCredit
	collectCredits := func(m retiredActivation) {
		if m.compactedHull {
			for _, fc := range m.foldedCredits {
				foldedCredits = appendFoldedCredit(foldedCredits, fc.swallower, fc.credits, fc.creditGroupID, fc.afterAgentCredit)
			}
			return
		}
		if !m.creditSettled {
			if rem := m.remainingCredits(); rem > 0 {
				foldedCredits = appendFoldedCredit(foldedCredits, m.swallower, rem, m.creditGroupID, m.afterAgentCredit)
			}
		}
	}
	collectCredits(m0)
	for _, m := range members[1:] {
		if !hull.swallower.anyName && (m.swallower.anyName || m.swallower.name != hull.swallower.name) {
			hull.swallower.name = ""
			hull.swallower.anyName = true
		}
		if !hull.swallower.anyBranch && (m.swallower.anyBranch || m.swallower.branch != hull.swallower.branch) {
			hull.swallower.branch = ""
			hull.swallower.anyBranch = true
		}
		if hull.swallower.path != m.swallower.path {
			hull.swallower.path = ""
		}
		hull.swallower.seq = max(hull.swallower.seq, m.swallower.seq)
		hull.retiredAtSeq = max(hull.retiredAtSeq, m.retiredAtSeq)
		if hull.parentSpanID != m.parentSpanID {
			hull.parentSpanID = ""
		}
		if hull.goroutineID != m.goroutineID {
			hull.goroutineID = 0
		}
		for _, g := range m.guardSpanIDs {
			if openSet[g] && !slices.Contains(hull.guardSpanIDs, g) {
				hull.guardSpanIDs = append(hull.guardSpanIDs, g)
			}
		}
		hull.ancestorSpanIDs = slices.DeleteFunc(hull.ancestorSpanIDs, func(id string) bool {
			return !slices.Contains(m.ancestorSpanIDs, id)
		})
		hull.openEnclosingSpanIDs = unionOpenSpanIDs(hull.openEnclosingSpanIDs, m.openEnclosingSpanIDs, openSet)
		hull.enclosingRetiredUnjoined = hull.enclosingRetiredUnjoined || m.enclosingRetiredUnjoined
		hull.completedOrTask = hull.completedOrTask && m.completedOrTask
		hull.joinedAtRetire = hull.joinedAtRetire && m.joinedAtRetire
		hull.hasUnjoinedTask = hull.hasUnjoinedTask || m.unjoinedTaskRetirement()
		hull.settledByParent = hull.settledByParent && m.settledByParent
		hull.concurrentPeer = hull.concurrentPeer || m.concurrentPeer
		hull.unrespondedAtPop = hull.unrespondedAtPop || m.unrespondedAtPop
		hull.peerRetirement = hull.peerRetirement || m.peerRetirement
		hull.hasAfterAgentCallbacks = hull.hasAfterAgentCallbacks || m.hasAfterAgentCallbacks
		hull.hasCustomUUIDProvider = hull.hasCustomUUIDProvider && m.hasCustomUUIDProvider
		hull.afterAgentPending = hull.afterAgentPending || m.holdsAfterAgentEvent()
		hull.afterAgentEventSeen = !hull.afterAgentPending && hull.afterAgentEventSeen && m.afterAgentEventSeen
		hull.creditSettled = hull.creditSettled || m.creditSettled
		collectCredits(m)
		hull.startPlatformTime = minPlatformTime(hull.startPlatformTime, m.startPlatformTime)
		hull.retiredAtPlatformTime = maxPlatformTime(hull.retiredAtPlatformTime, m.retiredAtPlatformTime)
		hull.parentRetiredAtPlatformTime = maxPlatformTime(hull.parentRetiredAtPlatformTime, m.parentRetiredAtPlatformTime)
		hull.afterAgentDone = hull.afterAgentDone && m.afterAgentDone
		if hull.afterAgentDone {
			hull.afterAgentDoneAt = maxPlatformTime(hull.afterAgentDoneAt, m.afterAgentDoneAt)
		} else {
			hull.afterAgentDoneAt = time.Time{}
		}
	}
	hull.foldedCredits = foldedCredits
	slices.Sort(hull.guardSpanIDs)
	return hull
}

// enforceRetiredActivationCapLocked bounds len(inv.retiredActivations) at
// maxRetiredActivationsPerInvocation by dropping fully settled entries and
// folding the oldest overflow entries into a conservative summary hull while
// keeping recent entries uncompacted. inv.mu must be held.
func (inv *invocationTrace) enforceRetiredActivationCapLocked() {
	if len(inv.retiredActivations) <= maxRetiredActivationsPerInvocation {
		return
	}
	inv.enforceRetiredActivationCapWithIndexLocked(inv.newUnrespondedRetiredIndexLocked())
}

func (inv *invocationTrace) enforceRetiredActivationCapWithIndexLocked(unrespIdx unrespondedRetiredIndex) {
	if len(inv.retiredActivations) <= maxRetiredActivationsPerInvocation {
		return
	}
	defer inv.pruneOrphanedCreditGroupsLocked()
	st := inv.afterAgentStateLocked()
	openSet := st.openSet
	surviving := st.swallowers
	inv.retiredActivations = slices.DeleteFunc(inv.retiredActivations, func(ra retiredActivation) bool {
		if (!ra.compactedHull && (surviving[ra.swallower.spanID] || st.underOpen[ra.swallower.spanID])) ||
			ra.holdsAfterAgentEvent() ||
			(ra.peerRetirement && inv.hasOpenPeerLocked(ra)) ||
			unrespIdx.matches(ra.swallower) {
			return false
		}
		return !slices.ContainsFunc(ra.guardSpanIDs, func(g string) bool {
			return openSet[g]
		})
	})
	if len(inv.retiredActivations) <= maxRetiredActivationsPerInvocation {
		inv.shrinkRetiredActivationsLocked()
		return
	}
	need := len(inv.retiredActivations) - maxRetiredActivationsPerInvocation + 1
	var tier1, tier2, tier3, tier4 []int
	for idx, ra := range inv.retiredActivations {
		switch {
		case !ra.compactedHull && (surviving[ra.swallower.spanID] || st.underOpen[ra.swallower.spanID]):
			tier4 = append(tier4, idx)
		case ra.remainingCredits() > 0 || len(ra.foldedCredits) > 0:
			tier3 = append(tier3, idx)
		case ra.peerRetirement && inv.hasOpenPeerLocked(ra):
			tier2 = append(tier2, idx)
		default:
			tier1 = append(tier1, idx)
		}
	}
	selected := make([]int, 0, need)
	for _, tier := range [][]int{tier1, tier2, tier3, tier4} {
		for _, idx := range tier {
			if len(selected) >= need {
				break
			}
			selected = append(selected, idx)
		}
		if len(selected) >= need {
			break
		}
	}
	if len(selected) < 2 {
		return
	}
	slices.Sort(selected)
	foldedSet := make(map[int]bool, len(selected))
	members := make([]retiredActivation, 0, len(selected))
	for _, idx := range selected {
		foldedSet[idx] = true
		ra := &inv.retiredActivations[idx]
		if !ra.compactedHull && !ra.creditSettled && ra.remainingCredits() > 0 && ra.creditGroupID == 0 {
			ra.creditGroupID = inv.allocCreditGroupIDLocked()
			inv.creditGroupRem[ra.creditGroupID] = ra.remainingCredits()
		}
		members = append(members, *ra)
	}
	firstIdx := selected[0]
	hull := foldRetiredCapHull(members, openSet)
	out := inv.retiredActivations[:0]
	for idx, ra := range inv.retiredActivations {
		if idx == firstIdx {
			out = append(out, hull)
			continue
		}
		if !foldedSet[idx] {
			out = append(out, ra)
		}
	}
	clear(inv.retiredActivations[len(out):])
	inv.retiredActivations = out
	inv.shrinkRetiredActivationsLocked()
}

// hasOpenPeerLocked reports whether an activation of the agent and branch of
// the retired activation ra with a wildcard-compatible node path is open, a
// peer of ra in the sense of retiredPeerCount (see soleOpenAuthorLocked).
// inv.mu must be held.
func (inv *invocationTrace) hasOpenPeerLocked(ra retiredActivation) bool {
	if len(inv.agentStack) <= 1 {
		return false
	}
	for _, rec := range inv.agentStack[1:] {
		if rec.kind == spanKindAgent && ra.swallower.matches(rec.name, rec.branch, rec.path) {
			return true
		}
	}
	return false
}

// maxPendingAfterAgent bounds, per parent activation, agent and branch, the
// retired activations that keep the action maps of their after-agent
// callbacks only because their after-agent events may still come (see
// foldPendingAfterAgentLocked).
const maxPendingAfterAgent = 8

// foldPendingAfterAgentLocked bounds the entries of the agent name on branch
// that keep their after-agent action maps only because their after-agent
// events may still come: those that hold no credit and that no recorded
// cause names as a swallower or has an open span under it. Nothing may show
// when the after-agent callbacks of such an activation return, as for the
// activations that a parent runs on goroutines of their own and does not wait
// for, until the parent's run returns (see recordRetiredActivationsAtLocked).
// Of those under one parent, at most maxPendingAfterAgent stay; the oldest of
// the others fold into one compacted hull that stands for their after-agent
// events (see afterAgentPending). An event that one of them can have created
// may then be such an event, so it is no one's alone and clears no failure,
// and an open activation of the agent that an ancestor of theirs may hold
// events for clears no failure with its own events either (see
// hasGuardedRetiredPeerForEventLocked), until the hull no longer needs to
// stand for them (see needsAfterAgentActionsLocked).
// A hull of activations on different node paths matches any path.
//
// While a peer of an entry is open (see hasOpenPeerLocked), the entry is not
// folded if it counted as a peer retirement, since soleOpenAuthorLocked gives
// the open peer its events only while every retired peer that
// retiredPeerCount counts has an entry of its own, nor if its callbacks were
// seen to return (afterAgentDone): it keeps its maps then because the open
// peer could be taken for the author of its after-agent event, not because
// that event may still come, and a hull would stand for it as one that may.
// Such entries stay as they are until no peer is open, as pruning keeps them
// (see pruneSettledRetiredActivationsLocked). inv.mu must be held.
func (inv *invocationTrace) foldPendingAfterAgentLocked(name, branch string, st afterAgentState) {
	inv.foldPendingAfterAgentWithIndexLocked(name, branch, st, inv.newUnrespondedRetiredIndexLocked())
}

func (inv *invocationTrace) foldPendingAfterAgentWithIndexLocked(name, branch string, st afterAgentState, unrespIdx unrespondedRetiredIndex) {
	defer inv.pruneOrphanedCreditGroupsLocked()
	openSet := st.openSet
	if openSet == nil {
		openSet = inv.openSpanSetLocked()
	}
	type foldGroupKey struct {
		openParent           string
		hasParentRetiredTime bool
	}
	groups := make(map[foldGroupKey][]int)
	for idx, ra := range inv.retiredActivations {
		if ra.swallower.anyName || ra.swallower.anyBranch || ra.swallower.name != name || ra.swallower.branch != branch || !ra.holdsAfterAgentEvent() ||
			((ra.afterAgentCredit || ra.modelCredits > 0 || ra.creditSettled || len(ra.foldedCredits) > 0) && unrespIdx.matches(ra.swallower)) ||
			(!ra.compactedHull && (st.swallowers[ra.swallower.spanID] || st.underOpen[ra.swallower.spanID])) {
			continue
		}
		if (ra.peerRetirement || (ra.afterAgentDone && !ra.compactedHull)) && inv.hasOpenPeerLocked(ra) {
			continue
		}
		openParent := ""
		if openSet[ra.parentSpanID] {
			openParent = ra.parentSpanID
		}
		gk := foldGroupKey{
			openParent:           openParent,
			hasParentRetiredTime: !ra.parentRetiredAtPlatformTime.IsZero(),
		}
		groups[gk] = append(groups[gk], idx)
	}
	folded := make(map[int]bool)
	for _, members := range groups {
		if len(members) <= maxPendingAfterAgent {
			continue
		}
		slices.SortFunc(members, func(a, b int) int {
			return cmp.Or(
				cmp.Compare(inv.retiredActivations[a].retiredAtSeq, inv.retiredActivations[b].retiredAtSeq),
				cmp.Compare(inv.retiredActivations[a].swallower.seq, inv.retiredActivations[b].swallower.seq),
				cmp.Compare(a, b),
			)
		})
		var hulls, nonHulls []int
		for _, idx := range members {
			if inv.retiredActivations[idx].compactedHull {
				hulls = append(hulls, idx)
			} else {
				nonHulls = append(nonHulls, idx)
			}
		}
		var fold []int
		if len(hulls) > 0 {
			fold = append(fold, hulls...)
			if extraNonHulls := len(members) - maxPendingAfterAgent - (len(hulls) - 1); extraNonHulls > 0 {
				fold = append(fold, nonHulls[:extraNonHulls]...)
			}
		} else {
			fold = members[:len(members)-maxPendingAfterAgent+1]
		}
		hull := &inv.retiredActivations[fold[0]]
		hull.compactedHull = true
		hull.hasAfterAgentCallbacks = true
		hull.hasUnjoinedTask = hull.unjoinedTaskRetirement()
		hull.guardSpanIDs = filterOpenSpanIDs(hull.guardSpanIDs, openSet)
		hull.openEnclosingSpanIDs = filterOpenSpanIDs(hull.openEnclosingSpanIDs, openSet)
		for _, idx := range fold[1:] {
			m := inv.retiredActivations[idx]
			if m.swallower.seq > hull.swallower.seq {
				hull.swallower.seq = m.swallower.seq
				hull.swallower.spanID = m.swallower.spanID
			}
			if hull.swallower.path != m.swallower.path {
				hull.swallower.path = ""
			}
			if hull.parentSpanID != m.parentSpanID {
				hull.parentSpanID = ""
			}
			hull.retiredAtSeq = max(hull.retiredAtSeq, m.retiredAtSeq)
			for _, g := range m.guardSpanIDs {
				if openSet[g] && !slices.Contains(hull.guardSpanIDs, g) {
					hull.guardSpanIDs = append(hull.guardSpanIDs, g)
				}
			}
			hull.ancestorSpanIDs = slices.DeleteFunc(hull.ancestorSpanIDs, func(id string) bool {
				return !slices.Contains(m.ancestorSpanIDs, id)
			})
			hull.openEnclosingSpanIDs = unionOpenSpanIDs(hull.openEnclosingSpanIDs, m.openEnclosingSpanIDs, openSet)
			hull.enclosingRetiredUnjoined = hull.enclosingRetiredUnjoined || m.enclosingRetiredUnjoined
			if hull.goroutineID != m.goroutineID {
				hull.goroutineID = 0
			}
			hull.startPlatformTime = minPlatformTime(hull.startPlatformTime, m.startPlatformTime)
			hull.retiredAtPlatformTime = maxPlatformTime(hull.retiredAtPlatformTime, m.retiredAtPlatformTime)
			hull.parentRetiredAtPlatformTime = maxPlatformTime(hull.parentRetiredAtPlatformTime, m.parentRetiredAtPlatformTime)
			hull.completedOrTask = hull.completedOrTask && m.completedOrTask
			hull.joinedAtRetire = hull.joinedAtRetire && m.joinedAtRetire
			hull.hasUnjoinedTask = hull.hasUnjoinedTask || m.unjoinedTaskRetirement()
			hull.settledByParent = hull.settledByParent && m.settledByParent
			hull.concurrentPeer = hull.concurrentPeer || m.concurrentPeer
			hull.unrespondedAtPop = hull.unrespondedAtPop || m.unrespondedAtPop
			hull.afterAgentDone = hull.afterAgentDone && m.afterAgentDone
			hull.afterAgentDoneAt = maxPlatformTime(hull.afterAgentDoneAt, m.afterAgentDoneAt)
			hull.hasCustomUUIDProvider = hull.hasCustomUUIDProvider && m.hasCustomUUIDProvider
			folded[idx] = true
		}
		slices.Sort(hull.guardSpanIDs)
		hull.ancestorSpanIDs = slices.Clone(hull.ancestorSpanIDs)
		hull.afterAgentActions = actionMaps{}
		hull.afterAgentActionsRef = nil
		hull.afterAgentPending = true
		hull.afterAgentEventSeen = false
		hull.afterAgentCredit = false
		hull.modelCredits = 0
		hull.creditSettled = false
		hull.creditGroupID = 0
		hull.peerRetirement = false
		hull.popGoroutineID = 0
	}
	if len(folded) == 0 {
		return
	}
	out := inv.retiredActivations[:0]
	for idx, ra := range inv.retiredActivations {
		if !folded[idx] {
			out = append(out, ra)
		}
	}
	clear(inv.retiredActivations[len(out):])
	inv.retiredActivations = out
	inv.shrinkRetiredActivationsLocked()
}

// hasGuardedRetiredPeerForEventLocked reports whether an earlier activation
// matching rec's (name, branch, path) retired while a non-propagating ancestor
// remained open to buffer and later replay its events, or left a retiredCause
// swallow token on agentStack, and cannot be excluded by event-creation
// timestamp provenance. When true, an event arriving while rec is still open on
// agentStack cannot safely clear rec's failure state. An ancestor that buffers
// events replays them with the time at which they were created, so a retired
// activation that cannot have created an event at that time (see
// canOwnEventCreatedAt), such as one without after-agent callbacks that had
// retired by then, does not guard it. inv.mu must be held.
func (inv *invocationTrace) hasGuardedRetiredPeerForEventLocked(rec *spanRecord, path string, evCreatedAt time.Time, hasEventID, evNonEmptyID bool) bool {
	if rec == nil || rec.kind != spanKindAgent {
		return false
	}
	openSet := inv.openSpanSetLocked()
	for _, ra := range inv.retiredActivations {
		hasPeerID := hasEventID || (evNonEmptyID && ra.hasCustomUUIDProvider)
		hasBothID := hasEventID || (evNonEmptyID && rec.hasCustomUUIDProvider && ra.hasCustomUUIDProvider)
		if (ra.compactedHull || ra.swallower.spanID != rec.spanID) && ra.swallower.matches(rec.name, rec.branch, path) &&
			slices.ContainsFunc(ra.guardSpanIDs, func(g string) bool {
				return openSet[g]
			}) &&
			(!hasPeerID || ra.canOwnEventCreatedAt(evCreatedAt, hasPeerID)) &&
			!ra.excludedByEventTimestamp(rec.spanID, rec.seq, rec.goroutineID, rec.concurrentPeer, rec.startPlatformTime, evCreatedAt, hasBothID) {
			return true
		}
	}
	for _, s := range inv.agentStack {
		for _, c := range s.retiredCauses {
			for _, sw := range c.swallowers {
				if sw.spanID != rec.spanID && sw.matches(rec.name, rec.branch, path) {
					return true
				}
			}
		}
	}
	return false
}

// soleOpenAuthorLocked returns the index on agentStack of the only activation
// that can have created a session event created at evCreatedAt, if that is
// one of the open activations at peerIdxs, or -1. Open activations that
// started after the event was created did not create it. A retired activation
// of the same agent, branch and node path did not either if it cannot own the
// event (see canOwnEventCreatedAt): the event was created before it started,
// after its enclosing activation retired, after its after-agent callbacks
// were seen to return, or after it retired itself once its run returned,
// unless its after-agent event may still come. An event with no
// creation evidence, a retired peer that no longer has its own entry in
// retiredActivations (see retiredPeerCount), or a swallower that a retired
// cause names leaves the author unproven. inv.mu must be held.
func (inv *invocationTrace) soleOpenAuthorLocked(peerIdxs []int, path string, evCreatedAt time.Time, hasIDFor func(*spanRecord) bool, hasIDForRetired func(retiredActivation) bool) int {
	if evCreatedAt.IsZero() || len(peerIdxs) == 0 || hasIDFor == nil || hasIDForRetired == nil {
		return -1
	}
	author := -1
	for _, j := range peerIdxs {
		if j <= 0 || j >= len(inv.agentStack) {
			continue
		}
		rec := inv.agentStack[j]
		if hasIDFor(rec) && !rec.startPlatformTime.IsZero() && evCreatedAt.Before(rec.startPlatformTime) {
			continue
		}
		if author >= 0 {
			return -1
		}
		author = j
	}
	if author < 0 || !hasIDFor(inv.agentStack[author]) {
		return -1
	}
	rec := inv.agentStack[author]
	var trackedByKey map[string]int
	if len(inv.retiredPeerCount) > 0 {
		trackedByKey = make(map[string]int)
	}
	for _, ra := range inv.retiredActivations {
		if trackedByKey != nil && ra.peerRetirement && !ra.compactedHull && ra.swallower.matches(rec.name, rec.branch, path) {
			pk := llmKey(identity{path: ra.swallower.path, agentName: ra.swallower.name, branch: ra.swallower.branch})
			trackedByKey[pk]++
		}
		if (ra.compactedHull || ra.swallower.spanID != rec.spanID) && ra.swallower.matches(rec.name, rec.branch, path) && (!hasIDForRetired(ra) || ra.canOwnEventCreatedAt(evCreatedAt, true)) {
			return -1
		}
	}
	for pk, want := range inv.retiredPeerCount {
		if want <= 0 {
			continue
		}
		kp, kn, kb, ok := parseLLMKey(pk)
		if ok && kn == rec.name && kb == rec.branch && (path == "" || kp == "" || kp == path) && trackedByKey[pk] < want {
			return -1
		}
	}
	for _, s := range inv.agentStack {
		for _, c := range s.retiredCauses {
			for _, sw := range c.swallowers {
				if sw.spanID != rec.spanID && sw.matches(rec.name, rec.branch, path) {
					return -1
				}
			}
		}
	}
	return author
}

// provenAfterAllRetiredLocked reports whether at least one retired activation
// matches rec's (name, branch, path), every matching retired activation is
// excluded by event-creation timestamp provenance against rec, and none carries a
// surviving retiredCause swallow token on agentStack. inv.mu must be held.
func (inv *invocationTrace) provenAfterAllRetiredLocked(rec *spanRecord, path string, evCreatedAt time.Time, hasEventID, evNonEmptyID bool) bool {
	if rec == nil || rec.kind != spanKindAgent || len(inv.retiredActivations) == 0 {
		return false
	}
	sawMatch := false
	for _, ra := range inv.retiredActivations {
		if (!ra.compactedHull && ra.swallower.spanID == rec.spanID) || !ra.swallower.matches(rec.name, rec.branch, path) {
			continue
		}
		sawMatch = true
		hasBothID := hasEventID || (evNonEmptyID && rec.hasCustomUUIDProvider && ra.hasCustomUUIDProvider)
		if !ra.excludedByEventTimestamp(rec.spanID, rec.seq, rec.goroutineID, rec.concurrentPeer, rec.startPlatformTime, evCreatedAt, hasBothID) {
			return false
		}
	}
	if !sawMatch {
		return false
	}
	for _, s := range inv.agentStack {
		for _, c := range s.retiredCauses {
			for _, sw := range c.swallowers {
				if sw.spanID != rec.spanID && sw.matches(rec.name, rec.branch, path) {
					return false
				}
			}
		}
	}
	return true
}

// rowSpan places a row in the trace of its invocation. It holds the
// trace_id, span_id and parent_span_id columns, the latency_ms object of
// rows that end a span, the name of the invocation's root agent, and the
// timestamp at which the callback that produced the row read the trace
// manager's clock. Callbacks that read tm.now() set timestamp on the returned
// rowSpan; unclosedAgentSpan rows leave it zero so that purged activations use
// eventData.timestamp (lastActivity) and end-of-run AGENT_ERROR rows use
// time.Now() in buildRow.
type rowSpan struct {
	traceID       string
	spanID        string
	parentSpanID  string
	latency       map[string]any
	rootAgentName string
	timestamp     time.Time
}

// invocationTrace holds the spans of one invocation.
type invocationTrace struct {
	// lastActive is the time of the invocation's last callback, in Unix
	// nanoseconds. Eviction reads it without holding mu.
	lastActive atomic.Int64
	// openCalls is the number of in-flight model and tool calls. Eviction
	// reads it without holding mu.
	openCalls atomic.Int64
	// openHandedOver is the number of in-flight model and tool calls that
	// AfterRun handed over to this late state (see spanRecord.handedOver).
	// Pruning reads it without holding mu.
	openHandedOver atomic.Int64
	// openAgents reports whether agentStack holds an activation besides the
	// invocation's own span. Pruning reads it without holding mu.
	openAgents atomic.Bool
	// pendingSpans protects late state between lookup and registering a span.
	// It is a standalone allocation so that completedTrace.livePending can
	// point to it without retaining the rest of invocationTrace after the run
	// finishes.
	pendingSpans *atomic.Int64
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
	// finished reports that endInvocation moved the calls of the invocation to
	// its completed entry (see rememberCompletedLocked and lockLive).
	finished bool
	// isLate reports that this invocationTrace is the late state of a completed
	// run (see lateLocked). AfterRun has already run on the live trace, so
	// popUnclosedAgentSpans will never run on a late trace.
	isLate bool
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
	// uuidProviders is nil until the run or one of its agent activations
	// starts with a custom platform.UUIDProvider (see noteUUIDProviderLocked),
	// except that a late state starts with the ledger of the finished run (see
	// lateLocked). noteUUIDProviderLocked sets it with mu held, and it is read
	// without mu (see traceManager.eventUUIDProvider).
	uuidProviders atomic.Pointer[uuidProviderLedger]
	// agentStack holds the invocation span, followed by the spans of the
	// running agents in the order they started. Agents that run
	// concurrently, such as the sub-agents of a parallel agent or
	// workflow.RunNode activations, can end in any order, so spans are
	// removed from wherever they are.
	agentStack []*spanRecord
	// lookalikes counts the open agent activations on agentStack by agent
	// name, branch and node path (see lookalikeKey), and lookalikeDups counts
	// the keys that two or more of them share, so that parentForRunLocked
	// skips resolving concurrent lookalikes without scanning agentStack while
	// every open activation is distinct. pushAgentLocked, removeAgentLocked,
	// removeAgentsLocked and truncateAgentsLocked, through which agentStack
	// changes, keep them up to date and set lookalikesLen to the length of
	// agentStack. A stack whose length differs from lookalikesLen was built
	// some other way, and is counted afresh (see syncLookalikesLocked).
	lookalikes    map[lookalikeKey]int
	lookalikeDups int
	lookalikesLen int
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
	// lastSeq is the seq of the agent activation or tool call that started
	// last (see spanRecord.seq).
	lastSeq uint64
	// completedTools holds the tool calls that OnToolErrorCallback ended,
	// so that the AfterToolCallback that ADK runs next for the same call
	// finds its span instead of ending another one. ADK skips
	// AfterToolCallback when an unanswered missing-tool lookup or a
	// non-FunctionTool fails; completeResponseWithActions clears any matching
	// entry whose live attempt matches the response's EventActions maps (or
	// whose tool context is no longer live), pushToolSpan clears any remaining
	// entry when a later call starts, and popToolSpan consumes any entry read
	// by AfterToolCallback. A stored record answers only the AfterToolCallback
	// of the attempt that it ended (see toolAttempt): a later call that reuses
	// the function call ID, and whose start an earlier plugin hid from this
	// one, is a call of its own.
	completedTools map[string]*spanRecord
	// unrecoveredErrors tracks unhandled errors encountered during the
	// invocation keyed by activation / call identity (e.g. llmKey), together
	// with the span ID of the agent activation that reported each of them.
	// An error stays recorded until a response to the failed call, or a final
	// event of its activation, clears it, or until ADK handles the failure
	// itself without yielding it, by running a workflow node again or by
	// returning the failure of an agent that runs as a tool as the tool's
	// error (see purgeActivationLocked). The end of its activation does not:
	// once ADK has yielded the error, the invocation failed, whatever the
	// caller does next, and ADK still ends the activations above the failed
	// one normally when the caller keeps reading. An error that agent code
	// handles without yielding it, such as a custom agent that runs a failed
	// sub-agent again or goes on without it, stays recorded as well, and the
	// invocation is reported as failed: no callback tells it apart from an
	// error that reached the caller.
	unrecoveredErrors map[string]recordedError
	// unrespondedRetired counts clean AfterAgentCallback retirements (or
	// buffered model/tool events of retired activations) per llmKey while a
	// peer activation of the same (path, agentName, branch) remains open, so a
	// buffered event yielded after its author retired does not mask a failing
	// peer activation.
	unrespondedRetired map[string]int
	// unrespondedRetiredTools counts completed tool-event credits
	// (awaitingToolEvent) left unyielded when a retired activation leaves a
	// peer open or retires under a buffering custom guard, kept separate from
	// unrespondedRetired so a buffered FunctionResponse consumes a retired tool
	// credit rather than a dropped model credit or an open peer's toolSpan.
	unrespondedRetiredTools map[string]int
	// retiredPeerCount counts retired peer activations per llmKey when
	// concurrent activations of the same (path, agentName, branch) overlapped,
	// so trailer events emitted from a retired peer's own AfterAgentCallback
	// are absorbed rather than attributed to a remaining peer with an
	// unrecovered model or tool error. soleOpenAuthorLocked consumes this count
	// when checking whether an event can only belong to a remaining open peer.
	retiredPeerCount map[string]int
	// retiredActivations records retired agent activations by span ID while an
	// enclosing custom ancestor remains open on agentStack, or while an
	// unguarded entry is pinned by after-agent state, compatible
	// unrespondedRetired credits, an open span under it, or an open peer (and
	// is dropped or compacted once none of those holds). creditGroupRem tracks
	// the remaining credits of each overlapping component credit group.
	retiredActivations []retiredActivation
	creditGroupRem     map[uint64]int
	nextCreditGroupID  uint64
	// lastSessionEventPartial reports that the most recent session event that
	// onEvent observed was a partial event. lastEventPartial reports that, in
	// addition, no work started since that shows that the run went on after
	// the event (see clearLastPartialUnderLocked), and lastPartialSpanID is
	// the span of the event's author, if it has one open. When the caller
	// stops reading mid-stream on a partial event and no error was recorded,
	// in-flight sibling calls and unresponded activations are not treated as
	// failures. While lastSessionEventPartial is set, the work that a
	// workflow or a parallel agent canceled, as they do for a caller that
	// stopped reading, is not treated as a failure either, even if it started
	// after the event, unlike work whose context an agent, a callback or a
	// tool canceled itself (see spanRecord.canceledByStop and
	// popUnclosedAgentSpans).
	lastSessionEventPartial bool
	lastEventPartial        bool
	lastPartialSpanID       string
	// partialLineage holds the span IDs of the activations that wrote a
	// partial session event since the last session event that was not
	// partial, themselves or through an activation that ran under them: if
	// the caller stopped reading on such an event, they did not fail.
	partialLineage map[string]bool
}

// recordedError is an unrecovered error of an invocation and the span ID of
// the agent activation that was active when it was recorded. canceled
// reports that the context of the callback that recorded the error was
// canceled by then by a parallel agent or a workflow with cause
// context.Canceled, as they cancel the work that still runs when the caller
// stops reading (see canceledByStop), rather than by the call or an agent
// canceling a context of its own. When a context.Canceled error's callback
// context ended with a cause other than context.Canceled or bare
// context.DeadlineExceeded, cause holds that error and collateral reports
// whether the cancellation started at a workflow, runner-root or
// parallel-branch stop point (see collateralFailureCause), as when a parallel
// agent's errgroup cancels sibling branches with a failing branch's error.
type recordedError struct {
	err        error
	cause      error
	spanID     string
	bareKey    string
	seq        uint64
	canceled   bool
	collateral bool
}

func newInvocationTraceWithSpan(sc trace.SpanContext, traceID string, now time.Time) *invocationTrace {
	return newInvocationTraceWithKnownSpan(sc, traceID, "", now)
}

func newInvocationTraceWithKnownSpan(sc trace.SpanContext, traceID, spanID string, now time.Time) *invocationTrace {
	if spanID == "" {
		spanID = newSpanID()
	}
	if traceID == "" && !sc.IsValid() {
		traceID = newTraceID()
	}
	root := &spanRecord{spanID: spanID, kind: spanKindInvocation, start: now}
	inv := &invocationTrace{
		traceID:           traceID,
		rootSpanID:        root.spanID,
		pendingSpans:      new(atomic.Int64),
		parentAgent:       map[string]string{},
		sequentialParent:  map[string]string{},
		agentStack:        []*spanRecord{root},
		lookalikesLen:     1,
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
	// Apart from the parent_span_id of the invocation span, which is the
	// caller's span ID when the invocation starts in a span, only the trace
	// ID follows OpenTelemetry (see rowSpan), so that the rows keep joining
	// the spans that ADK records for the same run. The other span_id and
	// parent_span_id columns are the plugin's own execution tree, as in
	// adk-python, whose rows take them from the plugin's own span stack: the
	// spans that ADK opens are never written to the table, so parents taken
	// from them would point at spans that no row has. The IDs of the spans
	// that the callbacks run in are only written, to attributes.otel, when
	// EnableOTelCorrelation is set, which it is not by default.
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

// hasOpenLateSpans reports whether a callback is starting a span, or a model
// call, tool call or activation is open. It reads only atomic state, so it
// never takes mu.
func (inv *invocationTrace) hasOpenLateSpans() bool {
	return inv.pendingSpans.Load() > 0 || inv.hasOpenCalls() || inv.openAgents.Load()
}

// syncOpenCallsLocked publishes the number of open model and tool calls, and of
// those that AfterRun handed over (see spanRecord.handedOver), for the readers
// that do not hold mu. Every change to llmSpans or toolSpans is followed by a
// call to it: a call that starts, that its completion callback or an error
// callback ends, that completeResponse ends, that a purge removes, that a new
// call of the same key replaces, or that AfterRun moves to the late state.
func (inv *invocationTrace) syncOpenCallsLocked() {
	inv.openCalls.Store(int64(len(inv.llmSpans) + len(inv.toolSpans)))
	var handedOver int64
	for _, calls := range []map[string]*spanRecord{inv.llmSpans, inv.toolSpans} {
		for _, rec := range calls {
			if rec.handedOver {
				handedOver++
			}
		}
	}
	inv.openHandedOver.Store(handedOver)
}

func (inv *invocationTrace) syncOpenAgentsLocked() {
	inv.openAgents.Store(len(inv.agentStack) > 1)
}

func (inv *invocationTrace) setRootAgentLocked(name string) {
	inv.rootAgentName = name
	if name != "" {
		inv.rootAgentAtomic.Store(name)
	}
}

// restoreEvicted restores the root span identity, trace metadata and UUID
// provider ledger saved when inv was previously evicted or completed, before it
// is returned to a callback.
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
	if ev.uuidProviders != nil && inv.uuidProviders.Load() == nil {
		inv.uuidProviders.Store(ev.uuidProviders)
	}
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
// the trace ID of their time, so when no span is active as the invocation
// starts, USER_MESSAGE_RECEIVED and INVOCATION_STARTING are on another trace
// than the rows after them. adk-python keeps every row of an invocation on
// one trace instead.
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

// noteUUIDProviderLocked records in inv's uuidProviderLedger whether the agent
// activation of id started with a custom platform.UUIDProvider on its context
// (custom), or, when run is true, whether the run did. Until a custom provider
// is seen, inv has no ledger and this records nothing: without one, every
// event of the invocation is taken to have been created without a custom
// provider anyway. Because uuidProviders stays nil until a custom
// platform.UUIDProvider appears in the invocation, any activation of an author
// A that started earlier in the invocation under the default provider left no
// ledger entry; if another activation of A later starts with a custom provider
// in the same invocation (either re-running on the same (name, branch, path) or
// running concurrently or later on another path or branch), a hand-built
// non-UUIDv4 event from that earlier unrecorded activation evaluated once the
// custom activation has started can match uuidProviderCustom either on the
// shared key or through eventUUIDProvider fallback keys (non-empty ancestor
// paths, anyPath, and anyBranch when ev.Branch == ""). inv.mu must be held.
func (inv *invocationTrace) noteUUIDProviderLocked(id identity, custom, run bool) {
	l := inv.uuidProviders.Load()
	if l == nil {
		if !custom {
			return
		}
		l = &uuidProviderLedger{
			compacted:      inv.isLate,
			pathOverflowed: inv.isLate,
			states:         map[uuidProviderKey]uuidProviderState{},
		}
		inv.uuidProviders.Store(l)
	}
	switch {
	case run:
		l.mu.Lock()
		l.runCustom = l.runCustom || custom
		l.mu.Unlock()
	case id.agentName != "":
		l.note(id.agentName, id.branch, id.path, custom)
	}
}

// isTrackedInvocation reports whether invID is currently tracked in
// tm.invocations, tm.completed, or tm.evicted.
//
// isTrackedInvocation acquires tm.mu, so callers must not hold tm.mu.
func (tm *traceManager) isTrackedInvocation(invID string) bool {
	if tm == nil || invID == "" {
		return false
	}
	tm.mu.Lock()
	defer tm.mu.Unlock()
	if tm.invocations[invID] != nil || tm.completed[invID] != nil {
		return true
	}
	_, evicted := tm.evicted[invID]
	return evicted
}

// eventUUIDProvider reports whether an event that author created in the
// invocation invID, on branch at the node path path, was created under a
// custom platform.UUIDProvider, judging by what the activations of author in
// that invocation started with (see uuidProviderLedger). It looks at the
// activations on branch, or on any branch when branch is empty, as a custom
// agent's session.NewEvent leaves the event's branch empty; and at path, or
// else at the nearest non-empty '/' ancestor of path (up to
// maxUUIDProviderAncestorSteps lookups, stopping before "" so a nested workflow
// path such as wf@1/fn@1 never falls back to a direct top-level activation at
// path == ""), as the events of function and dynamic workflow nodes carry the
// node's path and the name of the workflow agent that runs them, or else at
// any path, as the events of a nested workflow agent carry paths that start at
// it (inner@1/fn@1) while its own path is that of its node (outer@1/inner@1).
// The first of these that has any recorded activation decides, and only if all
// recorded activations under that key started alike: a key that covers
// recorded activations with and without a custom provider, as of a worker that
// ran on the same branch and path once with a custom provider and once
// without, gives false, so that an ID that one activation built by hand is not
// taken for one that the other's provider made. Because uuidProviders stays
// nil until a custom platform.UUIDProvider appears in the invocation, any
// activation of author A that started earlier in the invocation under the
// default provider left no ledger entry; if another activation of A later
// starts with a custom provider in the same invocation (either re-running on
// the same (name, branch, path) or running concurrently or later on another
// path or branch), a hand-built non-UUIDv4 event from that earlier unrecorded
// activation evaluated once the custom activation has started can match
// uuidProviderCustom either on the shared key or through the fallback keys
// above (non-empty ancestor paths, anyPath, and anyBranch when branch == "").
// An event of no recorded activation is judged by how the run started. tracked
// reports that invID is a live, evicted or completed invocation; custom is
// false when it is not.
//
// eventUUIDProvider acquires tm.mu, so callers must not hold tm.mu.
func (tm *traceManager) eventUUIDProvider(invID, author, branch, path string) (custom, tracked bool) {
	return tm.eventUUIDProviderFor(invID, author, branch, path, "", "")
}

// eventUUIDProviderFor is eventUUIDProvider with an optional fallbackAuthor and
// fallbackBranch from the delivering context's identity. When author is empty
// or "user" (such as a synthesized task response or user-role content yielded
// by a worker activation), fallbackAuthor is checked before falling back to how
// the run started so a worker that called platform.WithUUIDProvider(ctx, nil)
// fails closed rather than inheriting run.isRunCustom().
//
// eventUUIDProviderFor acquires tm.mu, so callers must not hold tm.mu.
func (tm *traceManager) eventUUIDProviderFor(invID, author, branch, path, fallbackAuthor, fallbackBranch string) (custom, tracked bool) {
	if tm == nil || invID == "" {
		return false, false
	}
	var run, late *uuidProviderLedger
	tm.mu.Lock()
	if inv := tm.invocations[invID]; inv != nil {
		tracked = true
		run = inv.uuidProviders.Load()
	} else if c := tm.completed[invID]; c != nil {
		tracked = true
		run = c.uuidProviders
		if c.late != nil {
			late = c.late.uuidProviders.Load()
		}
	} else if ev, ok := tm.evicted[invID]; ok {
		tracked = true
		run = ev.uuidProviders
	}
	tm.mu.Unlock()
	if late == run {
		late = nil
	}
	if run == nil && late == nil {
		return false, tracked
	}
	stateAt := func(k uuidProviderKey) uuidProviderState {
		return run.stateAt(k).with(late.stateAt(k))
	}
	lookupAuthor := func(name, br, p string) uuidProviderState {
		if name == "" {
			return uuidProviderUnseen
		}
		k := uuidProviderKey{name: name, branch: br, path: p, anyBranch: br == ""}
		for range maxUUIDProviderAncestorSteps {
			if s := stateAt(k); s != uuidProviderUnseen {
				return s
			}
			slash := strings.LastIndexByte(k.path, '/')
			if slash <= 0 {
				break
			}
			k.path = k.path[:slash]
		}
		k.path = ""
		k.anyPath = true
		return stateAt(k)
	}
	if s := lookupAuthor(author, branch, path); s != uuidProviderUnseen {
		return s == uuidProviderCustom, true
	}
	if (author == "" || author == userAuthor) && fallbackAuthor != "" && fallbackAuthor != author {
		if s := lookupAuthor(fallbackAuthor, cmp.Or(branch, fallbackBranch), path); s != uuidProviderUnseen {
			return s == uuidProviderCustom, true
		}
	}
	if effectiveBranch := cmp.Or(branch, fallbackBranch); effectiveBranch != "" {
		cur := effectiveBranch
		var branchHit uuidProviderState
		for {
			if bs := run.branchState(cur).with(late.branchState(cur)); bs != uuidProviderUnseen {
				branchHit = bs
				break
			}
			dot := strings.LastIndexByte(cur, '.')
			if dot <= 0 {
				break
			}
			cur = cur[:dot]
		}
		if branchHit != uuidProviderUnseen {
			if branchHit != uuidProviderCustom {
				return false, true
			}
		} else if ov := run.branchOverflow().with(late.branchOverflow()); ov != uuidProviderUnseen && ov != uuidProviderCustom {
			return false, true
		}
	}
	return run.isRunCustom() || late.isRunCustom(), true
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

func isBuiltInWorkflowAgent(a agent.Agent) bool {
	if a == nil {
		return false
	}
	switch orZero(func() string { return agentTypeOf(a) }) {
	case "ParallelAgent", "SequentialAgent", "LoopAgent", "WorkflowAgent":
		return true
	default:
		return false
	}
}

func agentTypeOf(a agent.Agent) string {
	for range 4 {
		v := reflect.ValueOf(a)
		for v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface {
			if v.IsNil() {
				return ""
			}
			v = v.Elem()
		}
		if v.Kind() != reflect.Struct {
			return ""
		}
		if f := v.FieldByName("AgentType"); f.IsValid() && f.Kind() == reflect.String {
			if s := f.String(); s != "" {
				return s
			}
		}
		inner := v.FieldByName("Agent")
		if !inner.IsValid() || !inner.CanInterface() {
			return ""
		}
		next, ok := reflect.TypeAssert[agent.Agent](inner)
		if !ok {
			return ""
		}
		a = next
	}
	return ""
}

// agentChildren are the sub-agents that are configured on a running agent.
type agentChildren struct {
	// parent is the name of the running agent.
	parent string
	// nonLLM reports that parent is a workflow agent, such as a sequential or
	// loop agent, rather than an llmagent whose sub-agents can transfer to
	// peers.
	nonLLM bool
	// propagatesChildError reports that parent is a built-in ADK workflow
	// agent whose Run always yields sub-agent errors rather than swallowing
	// them.
	propagatesChildError bool
	// isParallel reports that parent is ADK's built-in ParallelAgent.
	isParallel bool
	// isWorkflow reports that parent is ADK's built-in WorkflowAgent.
	isWorkflow bool
	// taskMode reports that parent is an llmagent configured with
	// llmagent.ModeTask. Neither a placement nor the runner binds task mode
	// for an agent that does not declare it.
	taskMode bool
	// hasAfterAgentCallbacks reports that the agent configures (or
	// conservatively may configure) agent-level AfterAgentCallbacks or ctx
	// carries (or conservatively may carry) another plugin with an
	// AfterAgentCallback, any of which can populate the single post-retirement
	// after-agent event that ADK builds after all AfterAgentCallbacks run.
	hasAfterAgentCallbacks bool
	// names holds the names of the sub-agents.
	names []string
}

// runningAgentChildren returns the sub-agents that are configured on the agent
// that ctx runs, and the kind of that agent, if it is named name. Agents that
// run as nodes of a workflow graph need not be in the sub-agent tree of the
// agent that the runner runs, so the agent hierarchy that the runner attaches
// to ctx can lack them. It is the zero value if ctx does not expose its agent.
// The callback context logs a warning from Agent, so the agent is read from
// the invocation context that WithICDelta derives from it. SubAgents and Name
// are user code, so callers must not hold a lock.
func runningAgentChildren(ctx context.Context, name string, selfPlugin *baseplugin.Plugin) agentChildren {
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
		cfg, isLLM := agentConfig(a).(llmagent.Config)
		agType := orZero(func() string { return agentTypeOf(a) })
		children := agentChildren{
			parent:                 name,
			nonLLM:                 !isLLM,
			propagatesChildError:   isBuiltInWorkflowAgent(a),
			isParallel:             agType == "ParallelAgent",
			isWorkflow:             agType == "WorkflowAgent",
			taskMode:               isLLM && cfg.Mode == llmagent.ModeTask,
			hasAfterAgentCallbacks: agentHasAfterAgentCallbacks(a) || contextHasAfterAgentPlugin(ctx, selfPlugin),
		}
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

// parentOutsideLocked returns the span that an agent starting with id would
// run under if the spans in exclude had ended (see parentFor). inv.mu must be
// held.
func (inv *invocationTrace) parentOutsideLocked(id identity, exclude map[string]bool) *spanRecord {
	saved := inv.agentStack
	inv.agentStack = slices.DeleteFunc(slices.Clone(saved), func(r *spanRecord) bool { return exclude[r.spanID] })
	defer func() { inv.agentStack = saved }()
	return inv.parentFor(id)
}

// lookalikeKey identifies concurrent activations of the same agent on the
// same branch and node path.
type lookalikeKey struct {
	name, branch, path string
}

// pushAgentLocked appends rec to agentStack (see lookalikes).
func (inv *invocationTrace) pushAgentLocked(rec *spanRecord) {
	inv.syncLookalikesLocked()
	inv.agentStack = append(inv.agentStack, rec)
	inv.countLookalikeLocked(rec, 1)
	inv.lookalikesLen = len(inv.agentStack)
}

// removeAgentLocked removes agentStack[i] (see lookalikes).
func (inv *invocationTrace) removeAgentLocked(i int) {
	inv.syncLookalikesLocked()
	inv.countLookalikeLocked(inv.agentStack[i], -1)
	inv.agentStack = slices.Delete(inv.agentStack, i, i+1)
	inv.lookalikesLen = len(inv.agentStack)
}

// removeAgentsLocked removes the spans of agentStack for which del reports
// true (see lookalikes).
func (inv *invocationTrace) removeAgentsLocked(del func(*spanRecord) bool) {
	inv.syncLookalikesLocked()
	inv.agentStack = slices.DeleteFunc(inv.agentStack, func(rec *spanRecord) bool {
		if !del(rec) {
			return false
		}
		inv.countLookalikeLocked(rec, -1)
		return true
	})
	inv.lookalikesLen = len(inv.agentStack)
}

// truncateAgentsLocked removes every span of agentStack but the invocation
// span (see lookalikes).
func (inv *invocationTrace) truncateAgentsLocked() {
	inv.agentStack = inv.agentStack[:1]
	clear(inv.lookalikes)
	inv.lookalikeDups = 0
	inv.lookalikesLen = len(inv.agentStack)
}

// countLookalikeLocked adds delta to the count of open activations of rec's
// lookalikeKey, if rec is an agent span, keeping lookalikeDups in step.
func (inv *invocationTrace) countLookalikeLocked(rec *spanRecord, delta int) {
	if rec.kind != spanKindAgent {
		return
	}
	k := lookalikeKey{name: rec.name, branch: rec.branch, path: rec.path}
	before := inv.lookalikes[k]
	after := before + delta
	if before < 2 && after >= 2 {
		inv.lookalikeDups++
	} else if before >= 2 && after < 2 {
		inv.lookalikeDups--
	}
	if after <= 0 {
		delete(inv.lookalikes, k)
		return
	}
	if inv.lookalikes == nil {
		inv.lookalikes = make(map[lookalikeKey]int)
	}
	inv.lookalikes[k] = after
}

// syncLookalikesLocked counts the open activations on agentStack afresh if
// agentStack changed other than through the methods that keep lookalikes up
// to date (see lookalikes).
func (inv *invocationTrace) syncLookalikesLocked() {
	if inv.lookalikesLen == len(inv.agentStack) {
		return
	}
	clear(inv.lookalikes)
	inv.lookalikeDups = 0
	for i, rec := range inv.agentStack {
		if i > 0 {
			inv.countLookalikeLocked(rec, 1)
		}
	}
	inv.lookalikesLen = len(inv.agentStack)
}

func isThinContextWrapper(t reflect.Type) bool {
	if t.NumField() != 1 {
		return false
	}
	f := t.Field(0)
	return f.Anonymous && f.Name == "Context" && f.Type == contextInterfaceType
}

func extractContextField(v reflect.Value, name string) (any, bool) {
	f := directStructField(v, name)
	if !f.IsValid() || f.Kind() != reflect.Interface || f.IsNil() || !f.CanInterface() {
		return nil, false
	}
	return f.Interface(), true
}

func launchLineageNext(v reflect.Value) (any, bool) {
	vt := v.Type()
	switch {
	case vt.PkgPath() == "context":
		if next, ok := extractContextField(v, "Context"); ok {
			return next, true
		}
		if cc := directStructField(v, "cancelCtx"); cc.IsValid() {
			if cc.Kind() == reflect.Pointer {
				if cc.IsNil() {
					return nil, false
				}
				cc = cc.Elem()
			}
			if cc.Kind() == reflect.Struct {
				return extractContextField(cc, "Context")
			}
		}
		return nil, false
	case isADKType(vt):
		return extractContextField(v, "Context")
	case isThinContextWrapper(vt):
		f := v.Field(0)
		if f.IsNil() || !f.CanInterface() {
			return nil, false
		}
		return f.Interface(), true
	default:
		return nil, false
	}
}

// launcherLocked returns the index in agentStack of the open agent activation,
// not excluded, whose context the agent run with activation key act (see
// activationKey) was started with, or 0 if no such activation is known. ADK
// derives the context of each agent run from the context passed to its Run,
// and standard context wrappers (except context.WithoutCancel), ADK contexts,
// and pointer wrappers embedding only context.Context with no other fields
// expose the parent they derive from, so the chain from act passes through the
// activation key of every run that the new run's context derives from, nearest
// first. The first one of an open activation tells which activation started the
// run, whichever goroutine runs it. Only those allowlisted types and direct
// fields are read, so no user fields or methods run while inv.mu is held.
func (inv *invocationTrace) launcherLocked(act any, excluded []bool) int {
	if act == nil {
		return 0
	}
	byActivation := make(map[any]int)
	for i, rec := range inv.agentStack {
		if i == 0 || excluded[i] || rec.kind != spanKindAgent || rec.activation == nil {
			continue
		}
		if reflect.ValueOf(rec.activation).Kind() == reflect.Pointer {
			byActivation[rec.activation] = i
		}
	}
	if len(byActivation) == 0 {
		return 0
	}
	cur := act
	var seen map[any]struct{}
	for step := 0; ; step++ {
		v := reflect.ValueOf(cur)
		if v.Kind() != reflect.Pointer || v.IsNil() {
			return 0
		}
		if step > 0 {
			if i, ok := byActivation[cur]; ok {
				return i
			}
		}
		if step >= 32 {
			if seen == nil {
				seen = make(map[any]struct{})
			}
			if _, dup := seen[cur]; dup {
				return 0
			}
			seen[cur] = struct{}{}
		}
		if v = v.Elem(); v.Kind() != reflect.Struct {
			return 0
		}
		next, ok := launchLineageNext(v)
		if !ok {
			return 0
		}
		cur = next
	}
}

// parentForRunLocked returns the span that an agent starting with id on
// goroutine goid, whose origin is origin and whose activation key is
// activation, runs under (see parentFor), leaving out any spans in exclude
// plus:
//   - open activations of id's agent, branch and node path that the scheduler
//     of their workflow agent (or of any open lookalike workflow activation
//     sharing its name, branch and node path) started on other goroutines,
//     together with the agents under them; and
//   - when concurrent open activations share (name, branch, path), every one
//     of them that neither is nor encloses the activation whose context the
//     run was started with (see launcherLocked), together with the agents
//     under them; or, when no open activation is on that context's chain, the
//     peers of the member that runs on goid (or has a non-excluded open span
//     in its subtree that does) or, failing that, started goid (origin.parent),
//     keeping the winning member, its ancestors, and any earlier activations
//     on its goroutine unexcluded.
//
// When no open activation is on the chain and the resolved parent itself (or
// any ancestor of it) belongs to an unmatched multi-member concurrent
// lookalike group (no member or descendant matches goid or origin.parent), it
// moves up to the nearest open span enclosing the members of that group that
// could hold the child (see appendPossibleOwnersLocked), so a helper-goroutine
// activation cannot land inside one peer lookalike's open child span while the
// child's branch and node path still pick out the only member that holds a
// span they fit. Non-concurrent lookalike chains (nested re-entry,
// same-goroutine stale spans, or suspended iterators) are left to parentFor
// without exclusion or ambiguity demotion unless a cross-goroutine nested
// ancestor directly matches goid or origin.parent.
func (inv *invocationTrace) parentForRunLocked(id identity, exclude map[string]bool, goid uint64, origin goroutineOrigin, activation any) *spanRecord {
	callerExcluded := exclude
	var clonedExclude bool
	ensureExcludeMutable := func() {
		if clonedExclude {
			return
		}
		exclude = maps.Clone(exclude)
		if exclude == nil {
			exclude = make(map[string]bool)
		}
		clonedExclude = true
	}
	if origin.creator == schedulerCreator && origin.parent != 0 {
		var wfStarterKeys map[lookalikeKey]bool
		for _, candidate := range inv.agentStack[1:] {
			if candidate.kind == spanKindAgent && candidate.isWorkflow && !callerExcluded[candidate.spanID] &&
				origin.startedBy(schedulerCreator, candidate) {
				if wfStarterKeys == nil {
					wfStarterKeys = make(map[lookalikeKey]bool)
				}
				wfStarterKeys[lookalikeKey{name: candidate.name, branch: candidate.branch, path: candidate.path}] = true
			}
		}
		var addedWfExclusion bool
		for i := 1; i < len(inv.agentStack); i++ {
			other := inv.agentStack[i]
			if exclude[other.spanID] || other.kind != spanKindAgent ||
				other.name != id.agentName || other.branch != id.branch || other.path != id.path ||
				other.goroutineID == goid || other.parent == nil || !other.parent.isWorkflow ||
				!other.origin.startedBy(schedulerCreator, other.parent) {
				continue
			}
			if !origin.startedBy(schedulerCreator, other.parent) &&
				!wfStarterKeys[lookalikeKey{name: other.parent.name, branch: other.parent.branch, path: other.parent.path}] {
				continue
			}
			ensureExcludeMutable()
			exclude[other.spanID] = true
			addedWfExclusion = true
		}
		if addedWfExclusion {
			inv.propagateExcludedLocked(exclude)
		}
	}
	var unmatched map[lookalikeKey][]*spanRecord
	var unmatchedByRec map[*spanRecord][]*spanRecord
	var tree *openSpanTree
	if goid != 0 {
		exclude, unmatched, unmatchedByRec, tree = inv.excludePeerLookalikesLocked(exclude, &clonedExclude, goid, origin, activation)
	}
	var parent *spanRecord
	if len(exclude) > 0 {
		parent = inv.parentOutsideLocked(id, exclude)
	} else {
		parent = inv.parentFor(id)
	}
	selected := parent
	var seenAmbiguous map[*spanRecord]bool
	var ambiguous []*spanRecord
	for parent != nil && parent.kind == spanKindAgent && len(unmatched) > 0 {
		ambiguous = ambiguous[:0]
		for cur := parent; cur != nil && cur.kind == spanKindAgent; cur = cur.parent {
			if recs := unmatchedByRec[cur]; len(recs) > 1 && !seenAmbiguous[recs[0]] {
				if seenAmbiguous == nil {
					seenAmbiguous = make(map[*spanRecord]bool)
				}
				seenAmbiguous[recs[0]] = true
				ambiguous = inv.appendPossibleOwnersLocked(ambiguous, tree, exclude, recs, id, selected)
			}
		}
		if len(ambiguous) == 0 {
			break
		}
		parent = tree.enclosing(inv, ambiguous)
	}
	return parent
}

// appendPossibleOwnersLocked appends to dst the members of the unmatched
// lookalike group recs that could hold a child agent starting with id whose
// parent would otherwise be selected: the members with an open span, not
// excluded, in their subtree (themselves included) that id's branch and node
// path fit as they fit a parent (see parentFor) at or below selected's branch
// and node path. A child cannot run under a member without such a span, so
// when fewer than two members have one, their branch and node path already
// tell which member holds the child, and nothing is appended. inv.mu must be
// held.
func (inv *invocationTrace) appendPossibleOwnersLocked(dst []*spanRecord, tree *openSpanTree, exclude map[string]bool, recs []*spanRecord, id identity, selected *spanRecord) []*spanRecord {
	start := len(dst)
	for _, m := range recs {
		mIdx, ok := tree.stackIdx[m]
		if !ok {
			continue
		}
		for _, u := range tree.dfsSeq[tree.tin[mIdx]-1 : tree.tout[mIdx]] {
			rec := inv.agentStack[u]
			if exclude[rec.spanID] ||
				!withinBranch(id.branch, rec.branch) || !withinPath(id.path, rec.path, rec.name == id.agentName) ||
				!withinBranch(rec.branch, selected.branch) || !atOrBelowPath(rec.path, selected.path) {
				continue
			}
			dst = append(dst, m)
			break
		}
	}
	if len(dst)-start < 2 {
		return dst[:start]
	}
	return dst
}

// atOrBelowPath reports whether node path path is ancestor or one of its
// descendant node paths. An empty ancestor holds every path.
func atOrBelowPath(path, ancestor string) bool {
	return ancestor == "" || path == ancestor || strings.HasPrefix(path, ancestor+"/")
}

// propagateExcludedLocked marks every open agent span in inv.agentStack[1:]
// whose ancestor along spanRecord.parent is in exclude. It runs in O(N) time
// using memoized ancestor states. inv.mu must be held and exclude must be
// mutable.
func (inv *invocationTrace) propagateExcludedLocked(exclude map[string]bool) {
	ancState := make(map[*spanRecord]uint8, len(inv.agentStack))
	var chain []*spanRecord
	for i := 1; i < len(inv.agentStack); i++ {
		rec := inv.agentStack[i]
		if rec.kind != spanKindAgent {
			continue
		}
		if exclude[rec.spanID] {
			ancState[rec] = 2
			continue
		}
		chain = chain[:0]
		var state uint8 = 1
		for cur := rec.parent; cur != nil && cur.kind == spanKindAgent; cur = cur.parent {
			if exclude[cur.spanID] || ancState[cur] == 2 {
				state = 2
				break
			}
			if ancState[cur] == 1 {
				state = 1
				break
			}
			chain = append(chain, cur)
		}
		ancState[rec] = state
		for _, c := range chain {
			ancState[c] = state
		}
		if state == 2 {
			exclude[rec.spanID] = true
		}
	}
}

// openSpanTree indexes inv.agentStack by pointer and open-parent Euler-tour
// intervals in O(N) time so ancestor, enclosing-span, and subtree-exclusion
// queries avoid repeated parent-chain scans.
type openSpanTree struct {
	stackIdx    map[*spanRecord]int
	openParent  []int
	childrenOf  [][]int
	tin         []int
	tout        []int
	depth       []int
	dfsSeq      []int
	preExcluded []bool
}

func (inv *invocationTrace) buildOpenSpanTreeLocked(exclude map[string]bool) *openSpanTree {
	n := len(inv.agentStack)
	stackIdx := make(map[*spanRecord]int, n)
	for i, rec := range inv.agentStack {
		stackIdx[rec] = i
	}
	openParent := make([]int, n)
	childrenOf := make([][]int, n)
	var preExcluded []bool
	if len(exclude) > 0 {
		preExcluded = make([]bool, n)
	}
	var retiredAnc map[*spanRecord]int
	var retiredExcl map[*spanRecord]bool

	for i := 1; i < n; i++ {
		rec := inv.agentStack[i]
		if preExcluded != nil && exclude[rec.spanID] {
			preExcluded[i] = true
		}
		p := rec.parent
		if p == nil || p.kind != spanKindAgent {
			openParent[i] = 0
			childrenOf[0] = append(childrenOf[0], i)
			continue
		}
		if pIdx, ok := stackIdx[p]; ok {
			openParent[i] = pIdx
			childrenOf[pIdx] = append(childrenOf[pIdx], i)
			continue
		}
		var path []*spanRecord
		ancIdx := 0
		baseExcl := false
		for cur := p; cur != nil && cur.kind == spanKindAgent; cur = cur.parent {
			if idx, ok := stackIdx[cur]; ok {
				ancIdx = idx
				break
			}
			if idx, ok := retiredAnc[cur]; ok {
				ancIdx = idx
				baseExcl = retiredExcl[cur]
				break
			}
			path = append(path, cur)
		}
		curExcl := baseExcl
		if len(path) > 0 {
			if retiredAnc == nil {
				retiredAnc = make(map[*spanRecord]int, len(path))
			}
			for _, node := range slices.Backward(path) {
				retiredAnc[node] = ancIdx
				if len(exclude) > 0 && exclude[node.spanID] {
					curExcl = true
				}
				if curExcl {
					if retiredExcl == nil {
						retiredExcl = make(map[*spanRecord]bool)
					}
					retiredExcl[node] = true
				}
			}
		}
		openParent[i] = ancIdx
		childrenOf[ancIdx] = append(childrenOf[ancIdx], i)
		if curExcl && preExcluded != nil {
			preExcluded[i] = true
		}
	}

	type dfsFrame struct {
		u       int
		entered bool
	}
	stack := make([]dfsFrame, 1, n)
	stack[0] = dfsFrame{u: 0}
	tin := make([]int, n)
	tout := make([]int, n)
	depth := make([]int, n)
	dfsSeq := make([]int, 0, n)
	timer := 0
	for len(stack) > 0 {
		top := &stack[len(stack)-1]
		u := top.u
		if !top.entered {
			top.entered = true
			timer++
			tin[u] = timer
			dfsSeq = append(dfsSeq, u)
			for _, v := range slices.Backward(childrenOf[u]) {
				depth[v] = depth[u] + 1
				stack = append(stack, dfsFrame{u: v})
			}
		} else {
			tout[u] = timer
			stack = stack[:len(stack)-1]
		}
	}

	return &openSpanTree{
		stackIdx:    stackIdx,
		openParent:  openParent,
		childrenOf:  childrenOf,
		tin:         tin,
		tout:        tout,
		depth:       depth,
		dfsSeq:      dfsSeq,
		preExcluded: preExcluded,
	}
}

func (t *openSpanTree) isOpenAncestor(ancIdx, descIdx int) bool {
	return ancIdx > 0 && descIdx > 0 && ancIdx != descIdx && t.tin[ancIdx] < t.tin[descIdx] && t.tout[descIdx] <= t.tout[ancIdx]
}

// enclosing returns the nearest open span on agentStack that each of recs runs
// under, or the invocation span when there is none.
func (t *openSpanTree) enclosing(inv *invocationTrace, recs []*spanRecord) *spanRecord {
	if len(recs) == 0 || len(inv.agentStack) == 0 {
		if len(inv.agentStack) == 0 {
			return nil
		}
		return inv.agentStack[0]
	}
	n := len(inv.agentStack)
	minStrictTin := n + 1
	minAncTin := n + 1
	maxTout := 0
	for _, rec := range recs {
		if idx, ok := t.stackIdx[rec]; ok {
			if t.tin[idx] < minStrictTin {
				minStrictTin = t.tin[idx]
			}
			if t.tout[idx] > maxTout {
				maxTout = t.tout[idx]
			}
			continue
		}
		var ancIdx int
		for cur := rec.parent; cur != nil && cur.kind == spanKindAgent; cur = cur.parent {
			if idx, ok := t.stackIdx[cur]; ok {
				ancIdx = idx
				break
			}
		}
		if ancIdx == 0 {
			return inv.agentStack[0]
		}
		if t.tin[ancIdx] < minAncTin {
			minAncTin = t.tin[ancIdx]
		}
		if t.tout[ancIdx] > maxTout {
			maxTout = t.tout[ancIdx]
		}
	}
	var startAnc int
	if idx, ok := t.stackIdx[recs[0]]; ok {
		startAnc = t.openParent[idx]
	} else {
		for cur := recs[0].parent; cur != nil && cur.kind == spanKindAgent; cur = cur.parent {
			if idx, ok := t.stackIdx[cur]; ok {
				startAnc = idx
				break
			}
		}
	}
	for u := startAnc; u > 0; u = t.openParent[u] {
		if t.tin[u] < minStrictTin && t.tin[u] <= minAncTin && t.tout[u] >= maxTout {
			return inv.agentStack[u]
		}
	}
	return inv.agentStack[0]
}

// pushIntMinHeap pushes v onto a binary min-heap stored in h.
func pushIntMinHeap(h []int, v int) []int {
	h = append(h, v)
	i := len(h) - 1
	for i > 0 {
		p := (i - 1) / 2
		if h[p] <= h[i] {
			break
		}
		h[p], h[i] = h[i], h[p]
		i = p
	}
	return h
}

// popIntMinHeap pops the smallest element from a non-empty binary min-heap.
func popIntMinHeap(h []int) ([]int, int) {
	top := h[0]
	last := h[len(h)-1]
	h = h[:len(h)-1]
	if len(h) > 0 {
		h[0] = last
		i := 0
		for {
			l := 2*i + 1
			if l >= len(h) {
				break
			}
			small := l
			if r := l + 1; r < len(h) && h[r] < h[l] {
				small = r
			}
			if h[i] <= h[small] {
				break
			}
			h[i], h[small] = h[small], h[i]
			i = small
		}
	}
	return h, top
}

// maxSegTree is a bottom-up iterative binary segment tree over [0, n) storing
// maximum values for O(log N) range-max subtree queries and point updates.
type maxSegTree struct {
	n    int
	tree []int
}

func newMaxSegTree(n int) maxSegTree {
	return maxSegTree{n: n, tree: make([]int, 2*n)}
}

func (s maxSegTree) build() {
	for i := s.n - 1; i >= 1; i-- {
		s.tree[i] = max(s.tree[2*i], s.tree[2*i+1])
	}
}

func (s maxSegTree) set(pos, val int) {
	i := pos + s.n
	s.tree[i] = val
	for i > 1 {
		i >>= 1
		s.tree[i] = max(s.tree[2*i], s.tree[2*i+1])
	}
}

func (s maxSegTree) query(l, r int) int {
	res := 0
	l += s.n
	r += s.n
	for l <= r {
		if l&1 == 1 {
			res = max(res, s.tree[l])
			l++
		}
		if r&1 == 0 {
			res = max(res, s.tree[r])
			r--
		}
		l >>= 1
		r >>= 1
	}
	return res
}

// excludePeerLookalikesLocked inspects non-excluded open agent activations on
// inv.agentStack[1:] grouped by lookalikeKey in O(N log N) worst-case time
// (O(N + G log G) when no witness node is excluded mid-pass), and in O(1)
// time without allocating while no two open activations share a lookalikeKey
// (see lookalikes). When an open activation is on the context chain of the
// run whose activation key is activation (see launcherLocked), every member
// of every group that neither is nor encloses it is excluded with the agents
// under it, and no group is left unmatched: the context that the run was
// started with tells the members apart better than any goroutine does, as a
// lookalike can stay open on a goroutine while suspended in yield or after
// its BeforeAgentCallback answered for it, and a goroutine can run work that
// any activation hands it after the one that created it. Otherwise, groups
// with a live goroutine match (goid in Pass 1, then origin.parent in Pass 2)
// are resolved in outer-to-inner topological order over the SCC condensation
// DAG of group dependencies, excluding losing peers (while keeping the winner,
// its ancestors, and earlier activations on the winner's non-zero goroutine
// unexcluded, and scoping any unresolved inner peers to the matched subtree
// below ancestor lookalikes). Groups with no goroutine match filter out
// superseded same-goroutine activations and then collapse nested ancestor chains
// (countEffectivePeers); when at least two effective concurrent peers remain,
// all live members within the group's matched scope are returned in unmatched
// (re-filtered and re-collapsed against final exclusions). inv.mu must be held.
func (inv *invocationTrace) excludePeerLookalikesLocked(exclude map[string]bool, clonedExclude *bool, goid uint64, origin goroutineOrigin, activation any) (map[string]bool, map[lookalikeKey][]*spanRecord, map[*spanRecord][]*spanRecord, *openSpanTree) {
	if goid == 0 || len(inv.agentStack) <= 2 {
		return exclude, nil, nil, nil
	}
	if inv.syncLookalikesLocked(); inv.lookalikeDups == 0 {
		return exclude, nil, nil, nil
	}
	n := len(inv.agentStack)
	firstIdx := make(map[lookalikeKey]int, n-1)
	var multiKeys []lookalikeKey
	var groups map[lookalikeKey][]int
	for i := 1; i < n; i++ {
		rec := inv.agentStack[i]
		if exclude[rec.spanID] || rec.kind != spanKindAgent {
			continue
		}
		k := lookalikeKey{name: rec.name, branch: rec.branch, path: rec.path}
		if first, ok := firstIdx[k]; !ok {
			firstIdx[k] = i
		} else {
			if groups == nil {
				groups = make(map[lookalikeKey][]int)
			}
			if len(groups[k]) == 0 {
				multiKeys = append(multiKeys, k)
				groups[k] = []int{first, i}
			} else {
				groups[k] = append(groups[k], i)
			}
		}
	}
	if len(multiKeys) == 0 {
		return exclude, nil, nil, nil
	}

	slices.SortFunc(multiKeys, func(a, b lookalikeKey) int {
		return cmp.Compare(firstIdx[a], firstIdx[b])
	})
	numGroups := len(multiKeys)
	nodeGroupID := make([]int, n)
	for i := range nodeGroupID {
		nodeGroupID[i] = -1
	}
	for id, k := range multiKeys {
		for _, mIdx := range groups[k] {
			nodeGroupID[mIdx] = id
		}
	}

	tree := inv.buildOpenSpanTreeLocked(exclude)
	excluded := make([]bool, n)
	if tree.preExcluded != nil {
		for _, u := range tree.dfsSeq[1:] {
			if tree.preExcluded[u] || excluded[tree.openParent[u]] {
				excluded[u] = true
			}
		}
	}

	ensureExcludeMutable := func() {
		if *clonedExclude {
			return
		}
		exclude = maps.Clone(exclude)
		if exclude == nil {
			exclude = make(map[string]bool)
		}
		*clonedExclude = true
	}

	if launcher := inv.launcherLocked(activation, excluded); launcher > 0 {
		for _, k := range multiKeys {
			for _, mIdx := range groups[k] {
				if excluded[mIdx] || mIdx == launcher || tree.isOpenAncestor(mIdx, launcher) {
					continue
				}
				ensureExcludeMutable()
				for _, u := range tree.dfsSeq[tree.tin[mIdx]-1 : tree.tout[mIdx]] {
					excluded[u] = true
					exclude[inv.agentStack[u].spanID] = true
				}
			}
		}
		return exclude, nil, nil, tree
	}

	groupsByTin := make([][]int, numGroups)
	for _, u := range tree.dfsSeq[1:] {
		if excluded[u] {
			continue
		}
		if id := nodeGroupID[u]; id >= 0 {
			groupsByTin[id] = append(groupsByTin[id], u)
		}
	}

	var topoIDs []int
	if numGroups == 1 {
		topoIDs = []int{0}
	} else {
		adj := make([][]int, numGroups)
		type groupEdge struct{ from, to int }
		seenEdges := make(map[groupEdge]bool)
		type ancFrame struct {
			u       int
			ancGrp  int
			childAt int
		}
		dfsStack := make([]ancFrame, 1, n)
		dfsStack[0] = ancFrame{u: 0, ancGrp: -1}
		for len(dfsStack) > 0 {
			top := &dfsStack[len(dfsStack)-1]
			if top.childAt < len(tree.childrenOf[top.u]) {
				v := tree.childrenOf[top.u][top.childAt]
				top.childAt++
				if excluded[v] {
					continue
				}
				nextAnc := top.ancGrp
				if ku := nodeGroupID[v]; ku >= 0 {
					if top.ancGrp >= 0 && top.ancGrp != ku {
						e := groupEdge{from: top.ancGrp, to: ku}
						if !seenEdges[e] {
							seenEdges[e] = true
							adj[top.ancGrp] = append(adj[top.ancGrp], ku)
						}
					}
					nextAnc = ku
				}
				dfsStack = append(dfsStack, ancFrame{u: v, ancGrp: nextAnc})
			} else {
				dfsStack = dfsStack[:len(dfsStack)-1]
			}
		}

		// Condense cycles via iterative Tarjan SCC so every outer ancestor SCC
		// resolves strictly before any downstream inner group SCC.
		index := make([]int, numGroups)
		lowlink := make([]int, numGroups)
		onStack := make([]bool, numGroups)
		tarjanStack := make([]int, 0, numGroups)
		sccID := make([]int, numGroups)
		var sccs [][]int
		nextIndex := 1
		type tarjanFrame struct {
			u      int
			edgeAt int
		}
		callStack := make([]tarjanFrame, 0, numGroups)
		for start := range numGroups {
			if index[start] != 0 {
				continue
			}
			index[start] = nextIndex
			lowlink[start] = nextIndex
			nextIndex++
			onStack[start] = true
			tarjanStack = append(tarjanStack, start)
			callStack = append(callStack, tarjanFrame{u: start})
			for len(callStack) > 0 {
				top := &callStack[len(callStack)-1]
				u := top.u
				if top.edgeAt < len(adj[u]) {
					v := adj[u][top.edgeAt]
					top.edgeAt++
					if index[v] == 0 {
						index[v] = nextIndex
						lowlink[v] = nextIndex
						nextIndex++
						onStack[v] = true
						tarjanStack = append(tarjanStack, v)
						callStack = append(callStack, tarjanFrame{u: v})
					} else if onStack[v] && index[v] < lowlink[u] {
						lowlink[u] = index[v]
					}
					continue
				}
				callStack = callStack[:len(callStack)-1]
				if len(callStack) > 0 {
					p := callStack[len(callStack)-1].u
					if lowlink[u] < lowlink[p] {
						lowlink[p] = lowlink[u]
					}
				}
				if lowlink[u] == index[u] {
					cid := len(sccs)
					var comp []int
					for {
						w := tarjanStack[len(tarjanStack)-1]
						tarjanStack = tarjanStack[:len(tarjanStack)-1]
						onStack[w] = false
						sccID[w] = cid
						comp = append(comp, w)
						if w == u {
							break
						}
					}
					slices.Sort(comp)
					sccs = append(sccs, comp)
				}
			}
		}

		numSCC := len(sccs)
		sccAdj := make([][]int, numSCC)
		sccIndeg := make([]int, numSCC)
		seenSCCEdges := make(map[groupEdge]bool, len(seenEdges))
		for u := range numGroups {
			su := sccID[u]
			for _, v := range adj[u] {
				sv := sccID[v]
				if su == sv {
					continue
				}
				e := groupEdge{from: su, to: sv}
				if !seenSCCEdges[e] {
					seenSCCEdges[e] = true
					sccAdj[su] = append(sccAdj[su], sv)
					sccIndeg[sv]++
				}
			}
		}

		topoIDs = make([]int, 0, numGroups)
		var heap []int
		for cid := range numSCC {
			if sccIndeg[cid] == 0 {
				// sccs[cid][0] is unique across SCCs and maps back via sccID[minID] == cid.
				heap = pushIntMinHeap(heap, sccs[cid][0])
			}
		}
		for len(heap) > 0 {
			var minID int
			heap, minID = popIntMinHeap(heap)
			cid := sccID[minID]
			topoIDs = append(topoIDs, sccs[cid]...)
			for _, dst := range sccAdj[cid] {
				sccIndeg[dst]--
				if sccIndeg[dst] == 0 {
					heap = pushIntMinHeap(heap, sccs[dst][0])
				}
			}
		}
	}

	var seg maxSegTree
	var segActive bool
	var segTargetGoid uint64
	activateSeg := func(target uint64) {
		segTargetGoid = target
		if cap(seg.tree) < 2*n {
			seg = newMaxSegTree(n)
		} else {
			seg.n = n
			seg.tree = seg.tree[:2*n]
			clear(seg.tree)
		}
		for u := 1; u < n; u++ {
			if !excluded[u] && inv.agentStack[u].goroutineID == target {
				seg.tree[n+tree.tin[u]-1] = u
			}
		}
		seg.build()
		segActive = true
	}

	var excludeQueue []int
	propagateExclude := func(u int) {
		if excluded[u] {
			return
		}
		ensureExcludeMutable()
		excluded[u] = true
		exclude[inv.agentStack[u].spanID] = true
		if segActive && inv.agentStack[u].goroutineID == segTargetGoid {
			seg.set(tree.tin[u]-1, 0)
		}
		excludeQueue = append(excludeQueue[:0], u)
		for len(excludeQueue) > 0 {
			cur := excludeQueue[len(excludeQueue)-1]
			excludeQueue = excludeQueue[:len(excludeQueue)-1]
			for _, ch := range tree.childrenOf[cur] {
				if !excluded[ch] {
					excluded[ch] = true
					exclude[inv.agentStack[ch].spanID] = true
					if segActive && inv.agentStack[ch].goroutineID == segTargetGoid {
						seg.set(tree.tin[ch]-1, 0)
					}
					excludeQueue = append(excludeQueue, ch)
				}
			}
		}
	}

	computeBestMatch := func(targetGoid uint64, dst []int) {
		clear(dst)
		if targetGoid == 0 {
			return
		}
		for _, u := range tree.dfsSeq[1:] {
			if !excluded[u] && inv.agentStack[u].goroutineID == targetGoid {
				dst[u] = u
			}
		}
		for idx := len(tree.dfsSeq) - 1; idx >= 1; idx-- {
			u := tree.dfsSeq[idx]
			if excluded[u] || dst[u] == 0 {
				continue
			}
			p := tree.openParent[u]
			if p > 0 && !excluded[p] && dst[u] > dst[p] {
				dst[p] = dst[u]
			}
		}
	}

	pickWinner := func(liveIdxs, best []int, useSeg bool, scopeIdx int) (int, int) {
		winnerIdx := 0
		bestMatch := 0
		bestDepth := -1
		for _, mIdx := range liveIdxs {
			if scopeIdx != 0 && mIdx != scopeIdx && !tree.isOpenAncestor(scopeIdx, mIdx) {
				continue
			}
			var bm int
			if useSeg {
				bm = seg.query(tree.tin[mIdx]-1, tree.tout[mIdx]-1)
			} else {
				bm = best[mIdx]
			}
			if bm == 0 {
				continue
			}
			if bm > bestMatch || (bm == bestMatch && tree.depth[mIdx] > bestDepth) {
				bestMatch = bm
				bestDepth = tree.depth[mIdx]
				winnerIdx = mIdx
			}
		}
		return winnerIdx, bestMatch
	}

	var effectiveScratch []int
	countEffectivePeers := func(liveByTin []int) int {
		if len(liveByTin) <= 1 {
			return len(liveByTin)
		}
		var lastForGoid map[uint64]int
		for _, u := range liveByTin {
			if g := inv.agentStack[u].goroutineID; g != 0 {
				if lastForGoid == nil {
					lastForGoid = make(map[uint64]int, len(liveByTin))
				}
				if prev, ok := lastForGoid[g]; !ok || u > prev {
					lastForGoid[g] = u
				}
			}
		}
		compacted := effectiveScratch[:0]
		for _, u := range liveByTin {
			if g := inv.agentStack[u].goroutineID; g != 0 && lastForGoid[g] != u {
				continue
			}
			compacted = append(compacted, u)
		}
		effectiveScratch = compacted
		if len(compacted) <= 1 {
			return len(compacted)
		}
		effective := 0
		for i, u := range compacted {
			if i+1 < len(compacted) && tree.tout[compacted[i+1]] <= tree.tout[u] {
				continue
			}
			effective++
		}
		return effective
	}

	excludeLosers := func(liveIdxs []int, winnerIdx, bm int, matchGoid uint64) bool {
		winner := inv.agentStack[winnerIdx]
		var keptInsideBm bool
		for _, mIdx := range liveIdxs {
			if mIdx == winnerIdx || tree.isOpenAncestor(mIdx, winnerIdx) {
				continue
			}
			// Only keep inner peers inside a descendant witness bm when the winner
			// itself is on a different goroutine from matchGoid; a direct
			// matchGoid winner must still exclude cross-goroutine inner peers even
			// if an inline child frame shares winner's goroutine.
			if bm != winnerIdx && winner.goroutineID != matchGoid && tree.isOpenAncestor(bm, mIdx) {
				keptInsideBm = true
				continue
			}
			other := inv.agentStack[mIdx]
			if other.goroutineID != 0 && other.goroutineID == winner.goroutineID && mIdx < winnerIdx {
				continue
			}
			propagateExclude(mIdx)
		}
		return keptInsideBm
	}

	// Pass 1: resolve groups with a live goid match before starter-only groups.
	bestGoid := make([]int, n)
	computeBestMatch(goid, bestGoid)
	resolved := make([]bool, numGroups)
	matchedScope := make([]int, numGroups)
	var hasUnresolved bool
	for _, id := range topoIDs {
		k := multiKeys[id]
		rawIdxs := groups[k]
		liveIdxs := rawIdxs[:0]
		var needRecompute bool
		for _, mIdx := range rawIdxs {
			if !excluded[mIdx] {
				liveIdxs = append(liveIdxs, mIdx)
				if !segActive {
					if bm := bestGoid[mIdx]; bm != 0 && excluded[bm] {
						needRecompute = true
					}
				}
			}
		}
		groups[k] = liveIdxs
		if len(liveIdxs) <= 1 {
			resolved[id] = true
			continue
		}
		if !segActive && needRecompute {
			activateSeg(goid)
		}
		winnerIdx, bm := pickWinner(liveIdxs, bestGoid, segActive, 0)
		if winnerIdx == 0 {
			hasUnresolved = true
			continue
		}
		if excludeLosers(liveIdxs, winnerIdx, bm, goid) {
			matchedScope[id] = bm
			hasUnresolved = true
		} else {
			resolved[id] = true
		}
	}

	var unmatched map[lookalikeKey][]*spanRecord
	var unmatchedRecKey map[int]lookalikeKey
	if hasUnresolved {
		bestStarter := make([]int, n)
		computeBestMatch(origin.parent, bestStarter)
		segActive = false

		deepestGoidAnc := make([]int, n)
		var deepestStarterAnc []int
		if origin.parent != 0 {
			deepestStarterAnc = make([]int, n)
		}
		for _, u := range tree.dfsSeq[1:] {
			if excluded[u] {
				continue
			}
			p := tree.openParent[u]
			g := inv.agentStack[u].goroutineID
			if g == goid {
				deepestGoidAnc[u] = u
			} else {
				deepestGoidAnc[u] = deepestGoidAnc[p]
			}
			if deepestStarterAnc != nil {
				if g == origin.parent {
					deepestStarterAnc[u] = u
				} else {
					deepestStarterAnc[u] = deepestStarterAnc[p]
				}
			}
		}

		for _, id := range topoIDs {
			if resolved[id] {
				continue
			}
			k := multiKeys[id]
			rawIdxs := groups[k]
			liveIdxs := rawIdxs[:0]
			var needRecompute bool
			for _, mIdx := range rawIdxs {
				if !excluded[mIdx] {
					liveIdxs = append(liveIdxs, mIdx)
					if !segActive {
						if bm := bestStarter[mIdx]; bm != 0 && excluded[bm] {
							needRecompute = true
						}
					}
				}
			}
			groups[k] = liveIdxs
			if len(liveIdxs) <= 1 {
				continue
			}
			effScope := matchedScope[id]
			goidScope := 0
			for _, mIdx := range liveIdxs {
				if effScope != 0 && mIdx != effScope && !tree.isOpenAncestor(effScope, mIdx) {
					continue
				}
				if anc := deepestGoidAnc[mIdx]; anc > goidScope {
					goidScope = anc
				}
			}
			if goidScope != 0 && (effScope == 0 || tree.isOpenAncestor(effScope, goidScope)) {
				effScope = goidScope
			}
			if !segActive && needRecompute {
				activateSeg(origin.parent)
			}
			winnerIdx, bm := pickWinner(liveIdxs, bestStarter, segActive, effScope)
			if winnerIdx != 0 {
				if !excludeLosers(liveIdxs, winnerIdx, bm, origin.parent) {
					continue
				}
				matchedScope[id] = bm
			}

			if matchedScope[id] == 0 {
				ancM := effScope
				if ancM == 0 && deepestStarterAnc != nil {
					for _, mIdx := range liveIdxs {
						if anc := deepestStarterAnc[mIdx]; anc > ancM {
							ancM = anc
						}
					}
				}
				if ancM != 0 {
					var hasInside, hasSiblingOutside bool
					for _, mIdx := range liveIdxs {
						if tree.isOpenAncestor(ancM, mIdx) {
							hasInside = true
						} else {
							hasSiblingOutside = true
						}
					}
					if hasInside && hasSiblingOutside {
						kept := liveIdxs[:0]
						for _, mIdx := range liveIdxs {
							if tree.isOpenAncestor(ancM, mIdx) {
								kept = append(kept, mIdx)
							} else {
								propagateExclude(mIdx)
							}
						}
						liveIdxs = kept
						groups[k] = liveIdxs
						if len(liveIdxs) <= 1 {
							continue
						}
						matchedScope[id] = ancM
					}
				}
			}

			scope := matchedScope[id]
			rawTin := groupsByTin[id]
			liveByTin := rawTin[:0]
			for _, u := range rawTin {
				if !excluded[u] && (scope == 0 || u == scope || tree.isOpenAncestor(scope, u)) {
					liveByTin = append(liveByTin, u)
				}
			}
			groupsByTin[id] = liveByTin
			if countEffectivePeers(liveByTin) <= 1 {
				continue
			}
			if unmatched == nil {
				unmatched = make(map[lookalikeKey][]*spanRecord)
				unmatchedRecKey = make(map[int]lookalikeKey, len(liveByTin))
			}
			unmatched[k] = nil
			for _, u := range liveByTin {
				unmatchedRecKey[u] = k
			}
		}
	}

	var unmatchedByRec map[*spanRecord][]*spanRecord
	if len(unmatched) > 0 {
		keptByTin := make(map[lookalikeKey][]int, len(unmatched))
		for _, u := range tree.dfsSeq[1:] {
			if excluded[u] {
				continue
			}
			if k, ok := unmatchedRecKey[u]; ok {
				keptByTin[k] = append(keptByTin[k], u)
			}
		}
		unmatchedByRec = make(map[*spanRecord][]*spanRecord, len(unmatchedRecKey))
		for k := range unmatched {
			keptIdxs := keptByTin[k]
			if countEffectivePeers(keptIdxs) <= 1 {
				delete(unmatched, k)
				continue
			}
			recs := make([]*spanRecord, len(keptIdxs))
			for i, u := range keptIdxs {
				recs[i] = inv.agentStack[u]
			}
			unmatched[k] = recs
			for _, r := range recs {
				unmatchedByRec[r] = recs
			}
		}
	}
	return exclude, unmatched, unmatchedByRec, tree
}

// ownerOf returns the span that the model and tool calls of id run under:
// the agent's own span, if it is known, preferring the one that the agent run
// of ctx started (see findAgentForContext), or else the span that it would run
// under (see parentFor). It never returns nil: parentFor falls back to
// agentStack[0], the invocation span.
func (inv *invocationTrace) ownerOf(ctx context.Context, id identity) *spanRecord {
	if i := inv.findAgentForContext(ctx, id.path, id.agentName, id.branch); i > 0 {
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
// agent name. An activation of the same agent on the same path, including an
// empty one, runs beside ancestor rather than inside it, as when a custom
// agent runs the same sub-agent concurrently.
func withinPath(path, ancestor string, sameAgent bool) bool {
	if path == ancestor {
		return !sameAgent
	}
	if path == "" || ancestor == "" {
		return true
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
//
// Invocations are keyed by invocation ID alone. A run that reuses the ID of a
// finished run, as a HITL resume reuses the ID of the paused run, starts
// tracking state, and an invocation span, of its own; without ambient spans,
// the rows of both runs have the trace_id that is derived from the invocation
// ID. Two runs that overlap with one invocation ID are not supported: no
// callback carries a handle of its run, so they share tracking state and can
// end each other's spans.
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
	selfPlugin      weak.Pointer[baseplugin.Plugin]

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
	uuidProviders *uuidProviderLedger
}

// completedTrace is what the callbacks that run after a run ended need of it.
type completedTrace struct {
	identity evictedTrace
	// livePending points to the finished run's pendingSpans counter when a
	// callback reserved a span on the live invocation before AfterRun and has
	// not yet re-entered lookupInvocation on the late state, keeping
	// pruneCompletedLocked from dropping the entry during the handoff.
	livePending *atomic.Int64
	// late holds the model and tool calls that the run left open, and those
	// that an error callback ended, for the callbacks that end them after
	// AfterRun. It does not retain the finished run's span stacks, and it is
	// made only for a run that left open calls behind, or for the first
	// callback after AfterRun.
	late *invocationTrace
	// endedModels and endedTools hold, until late is made, the calls that
	// OnModelErrorCallback or OnToolErrorCallback ended, for the
	// AfterModelCallback or AfterToolCallback that ADK may still run for them
	// (see popLLMSpan and popToolSpan). They are nil when there are none.
	endedModels, endedTools map[string]*spanRecord
	// uuidProviders is the compacted uuidProviderLedger of the finished run,
	// nil if it had none. The late state starts with the same ledger (see
	// lateLocked).
	uuidProviders *uuidProviderLedger
	position      *list.Element
}

func (c *completedTrace) hasPendingOrOpenLateSpans() bool {
	if c.livePending != nil && c.livePending.Load() > 0 {
		return true
	}
	return c.late != nil && c.late.hasOpenLateSpans()
}

func (c *completedTrace) hasPendingSpans() bool {
	if c.livePending != nil && c.livePending.Load() > 0 {
		return true
	}
	return c.late != nil && c.late.pendingSpans.Load() > 0
}

// waitsForLateCalls reports whether c is kept only for the calls that AfterRun
// handed over to its late state (see spanRecord.handedOver) while at least one
// of them is still open, and no callback is starting a span in it. Calls that
// callbacks started after the run ended do not count, nor does a handed-over
// call once its completion callback, completeResponse or a purge ended it.
// tm.mu must be held.
func (c *completedTrace) waitsForLateCalls() bool {
	return !c.hasPendingSpans() && c.late != nil && c.late.openHandedOver.Load() > 0
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

// lookupInvocation starts a fresh run only when start is true, as for the
// callbacks that begin a run, so that a run that reuses the ID of a finished
// run, such as a HITL resume, gets tracking state, and an invocation span, of
// its own; without ambient spans, the rows of both runs have the trace_id
// that is derived from the invocation ID. Other callbacks after AfterRun use
// the completed identity without restarting active tracking.
func (tm *traceManager) lookupInvocation(ctx context.Context, id identity, start, reserveSpan bool) *invocationTrace {
	return tm.lookupInvocationWithUUID(ctx, id, start, reserveSpan, false)
}

func (tm *traceManager) lookupInvocationWithUUID(ctx context.Context, id identity, start, reserveSpan, customUUIDs bool) *invocationTrace {
	now := tm.now()
	sc := ambientSpanContext(ctx)
	if id.invocationID == "" {
		inv := newInvocationTraceWithSpan(sc, "", now)
		if customUUIDs {
			inv.uuidProviders.Store(&uuidProviderLedger{states: map[uuidProviderKey]uuidProviderState{}})
		}
		if reserveSpan {
			inv.pendingSpans.Add(1)
		}
		return inv
	}
	var warnIDs []string
	var maxTracked int
	var logf func(string, ...any)
	inv, returned := func() (*invocationTrace, bool) {
		tm.mu.Lock()
		defer tm.mu.Unlock()
		inv, ok := tm.invocations[id.invocationID]
		if !ok {
			if completed := tm.completed[id.invocationID]; completed != nil {
				if !start {
					late := tm.lateLocked(completed, sc, tm.traceIDFor(id.invocationID), now)
					late.lastActive.Store(now.UnixNano())
					tm.completionOrder.MoveToBack(completed.position)
					if customUUIDs && late.uuidProviders.Load() == nil {
						late.uuidProviders.Store(&uuidProviderLedger{
							compacted:      late.isLate,
							pathOverflowed: late.isLate,
							states:         map[uuidProviderKey]uuidProviderState{},
						})
					}
					if reserveSpan {
						late.pendingSpans.Add(1)
					}
					return late, true
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
		if customUUIDs && inv.uuidProviders.Load() == nil {
			inv.uuidProviders.Store(&uuidProviderLedger{
				compacted:      inv.isLate,
				pathOverflowed: inv.isLate,
				states:         map[uuidProviderKey]uuidProviderState{},
			})
		}
		inv.lastActive.Store(now.UnixNano())
		if reserveSpan {
			inv.pendingSpans.Add(1)
		}
		return inv, false
	}()
	if returned {
		return inv
	}
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
		if ev.uuidProviders != nil {
			ev.uuidProviders = ev.uuidProviders.compactForCompleted()
		}
	} else {
		ev.lastActive = now.UnixNano()
	}
	if prev, ok := tm.evicted[invocationID]; ok && ev.uuidProviders == nil {
		ev.uuidProviders = prev.uuidProviders
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
		uuidProviders: inv.uuidProviders.Load(),
	}
}

// rememberCompletedLocked records invocation completion using tm.now() when
// creating late state; see rememberCompletedAtLocked. tm.mu must be held.
func (tm *traceManager) rememberCompletedLocked(invocationID, runner string, inv *invocationTrace) {
	tm.rememberCompletedAtLocked(invocationID, runner, inv, time.Time{})
}

// Completed identities have their own capacity, max(maxTracked, 64):
// finishing an ordinary run must never evict the saved identity of a run that
// is still in progress. The model and tool calls that the run started and that
// may still get a callback, such as a tool call of RunLive that ends after
// AfterRun, move to the late state, so that the callbacks end the calls' own
// spans. Two kinds of open calls do not: an open model call that streamed a
// partial event, and a call of an agent activation that finished running,
// which ADK left without a completion callback, as when a BeforeModelCallback
// of the agent or of a later plugin returned an error, or the model-call
// budget ran out, after this plugin's own BeforeModelCallback started the call
// (see endedWithRun, for which runner names the agent that the runner ran). A
// caller that stops reading a stream leaves its model call open for good. When
// the caller of runner.Run stops at a partial event, the call's activation is
// the runner's agent or one that it waits for, so endedWithRun leaves the call
// behind as well; the partial event alone decides for an activation that
// nothing is known to wait for, as a sub-agent whose custom parent stops
// reading it on a goroutine that the parent does not wait for. If such a call
// does end after AfterRun, as when a custom agent returned while a sub-agent
// that it did not wait for still streams, its AfterModelCallback gets a span
// of its own. The calls that an error
// callback ended move along with them, or, if the run left no call open, wait
// in the completed entry until a callback after AfterRun makes the late state.
// inv is marked finished, so that a model or tool callback that looked inv up
// before then acts on the late state instead (see lockLive), and inv drops all
// its calls, so that none of them is reachable from both inv and the late
// state.
//
// Beyond capacity, the least recently active runs are pruned. The callbacks
// do not show the end of every activation that no callback can end anymore,
// as of a sub-agent that a custom agent ran on another goroutine and waited
// for after the sub-agent stopped at an error, so while more runs than half
// of the capacity wait for calls that AfterRun handed over to their late
// state, the least recently active of them go first (see
// pruneWaitingLocked): a run keeps its identity when it completes however
// many such runs there are, and they leave at least half of the capacity to
// the runs that complete after them. A run counts only while one of the calls
// that AfterRun handed over is open (see waitsForLateCalls): not once those
// calls ended, even if callbacks started other calls after AfterRun. Then come
// the runs that have no pending span reservation and nothing open in their
// late state, and then, if more runs than the capacity still wait for late
// calls, the others too, except runs in which a callback is starting a span.
// Each pass walks the runs from the least recently active one and stops once
// the capacity is met; a run that a pass checks costs a check of atomic state
// (see hasPendingOrOpenLateSpans and waitsForLateCalls), without taking its
// lock. A callback of a pruned run gets a span of its own. If now is zero,
// rememberCompletedAtLocked calls tm.now() when creating late state; callers
// such as endInvocation and forceEndInvocation pass a non-zero now. tm.mu must
// be held.
//
// Divergence from adk-python (google/adk-python@25d191710985,
// after_run_callback lines 10102-10145): Python ADK clears the current
// task's contextvars span stack in after_run_callback. Go needs explicit
// shared late-call bookkeeping because goroutines do not inherit
// task-local state and RunLive tool goroutines can outlive
// AfterRunCallback, so open calls are handed to the late state instead.
func (tm *traceManager) rememberCompletedAtLocked(invocationID, runner string, inv *invocationTrace, now time.Time) {
	savedID := inv.savedIdentity()
	existing := tm.completed[invocationID]
	var panicVal any
	var panicked bool

	func() {
		inv.mu.Lock()
		defer inv.mu.Unlock()
		_ = inv.agentStack[0]
		lastSeq := inv.lastSeq
		// A completed entry has a ledger already only if AfterRun ran again for an
		// invocation that endInvocation had removed, or forceEndInvocation ran for
		// one after endInvocation had removed it. Then inv is a fresh trace that
		// recorded nothing (or restored completed.identity), so the compacted
		// ledger of the earlier run stays.
		var compactedProviders *uuidProviderLedger
		if existing != nil && existing.uuidProviders != nil {
			compactedProviders = existing.uuidProviders
		} else {
			compactedProviders = inv.uuidProviders.Load().compactForCompleted()
		}
		savedID.uuidProviders = compactedProviders
		models, endedModels := startedCalls(inv.llmSpans), startedCalls(inv.completedModels)
		maps.DeleteFunc(models, func(_ string, rec *spanRecord) bool { return rec.streamed || rec.endedWithRun(runner) })
		tools, endedTools := startedCalls(inv.toolSpans), startedCalls(inv.completedTools)
		maps.DeleteFunc(tools, func(_ string, rec *spanRecord) bool { return rec.endedWithRun(runner) })

		// Prepare and populate the late destination (or merged ended-call maps)
		// before unlinking inv from tm or clearing inv's call maps, so a panic
		// while allocating or populating late leaves the active trace and its open
		// calls intact.
		var preparedLate *invocationTrace
		var nextEndedModels, nextEndedTools map[string]*spanRecord
		if len(models)+len(tools) > 0 {
			if now.IsZero() {
				func() {
					defer func() {
						if r := recover(); r != nil {
							panicVal = r
							panicked = true
							now = time.Now()
						}
					}()
					now = tm.now()
				}()
			}
			if existing != nil && existing.late != nil {
				preparedLate = existing.late
				if preparedLate.uuidProviders.Load() == nil && compactedProviders != nil {
					preparedLate.uuidProviders.Store(compactedProviders)
				}
			} else {
				preparedLate = newInvocationTraceWithKnownSpan(trace.SpanContext{}, savedID.traceID, savedID.spanID, now)
				preparedLate.isLate = true
				preparedLate.restoreEvicted(savedID)
				preparedLate.uuidProviders.Store(compactedProviders)
				if existing != nil {
					maps.Copy(preparedLate.completedModels, existing.endedModels)
					maps.Copy(preparedLate.completedTools, existing.endedTools)
				}
			}
		} else if existing != nil && existing.late != nil {
			preparedLate = existing.late
			if preparedLate.uuidProviders.Load() == nil && compactedProviders != nil {
				preparedLate.uuidProviders.Store(compactedProviders)
			}
		}

		if preparedLate != nil {
			func() {
				preparedLate.mu.Lock()
				defer preparedLate.mu.Unlock()
				preparedLate.lastSeq = max(preparedLate.lastSeq, lastSeq)
				maps.Copy(preparedLate.llmSpans, models)
				maps.Copy(preparedLate.completedModels, endedModels)
				maps.Copy(preparedLate.toolSpans, tools)
				maps.Copy(preparedLate.completedTools, endedTools)
				for _, rec := range models {
					rec.handedOver = true
				}
				for _, rec := range tools {
					rec.handedOver = true
				}
				preparedLate.syncOpenCallsLocked()
			}()
		} else {
			var prevModels, prevTools map[string]*spanRecord
			if existing != nil {
				prevModels, prevTools = existing.endedModels, existing.endedTools
			}
			nextEndedModels = withCalls(prevModels, endedModels)
			nextEndedTools = withCalls(prevTools, endedTools)
		}

		if existing == nil {
			existing = &completedTrace{position: tm.completionOrder.PushBack(invocationID)}
			tm.completed[invocationID] = existing
		} else {
			tm.completionOrder.MoveToBack(existing.position)
		}

		// Unlink active and evicted entries before marking inv finished so a panic
		// after handover can never leave a finished invocation reachable from
		// tm.invocations.
		if cur := tm.invocations[invocationID]; cur == nil || cur == inv {
			if ev, ok := tm.evicted[invocationID]; ok && (cur == inv || savedID.spanID == "" || ev.spanID == "" || ev.spanID == savedID.spanID) {
				delete(tm.evicted, invocationID)
			}
			delete(tm.invocations, invocationID)
		}
		if existing.identity.spanID != "" && existing.identity.spanID == savedID.spanID {
			savedID.traceID = cmp.Or(savedID.traceID, existing.identity.traceID)
			savedID.rootAgentName = cmp.Or(savedID.rootAgentName, existing.identity.rootAgentName)
		}
		existing.uuidProviders = compactedProviders
		existing.identity = savedID
		if inv.pendingSpans.Load() > 0 {
			existing.livePending = inv.pendingSpans
		} else if existing.livePending != nil && existing.livePending.Load() == 0 {
			existing.livePending = nil
		}
		if preparedLate != nil {
			existing.endedModels, existing.endedTools = nil, nil
			existing.late = preparedLate
		} else {
			existing.endedModels = nextEndedModels
			existing.endedTools = nextEndedTools
		}

		inv.agentStack[0].dropContexts()
		inv.runnerAgent = nil
		// The late state keeps no context of the run's callbacks: the calls that
		// are open drop theirs here, and those that ended did when they ended.
		// The open ones are marked as handed over, so that the run waits for late
		// calls only while one of them is open (see waitsForLateCalls).
		for _, rec := range models {
			rec.dropContexts()
		}
		for _, rec := range tools {
			rec.dropContexts()
		}
		inv.finished = true
		clear(inv.llmSpans)
		clear(inv.completedModels)
		clear(inv.toolSpans)
		clear(inv.completedTools)
		inv.syncOpenCallsLocked()
	}()
	if panicked {
		panic(panicVal)
	}
	tm.pruneCompletedToLimitLocked()
}

func (tm *traceManager) publishCompletedIdentityLocked(invocationID string, saved evictedTrace) {
	delete(tm.evicted, invocationID)
	delete(tm.invocations, invocationID)
	completed := tm.completed[invocationID]
	if completed == nil {
		completed = &completedTrace{position: tm.completionOrder.PushBack(invocationID)}
		tm.completed[invocationID] = completed
	} else {
		tm.completionOrder.MoveToBack(completed.position)
		if completed.livePending != nil && completed.livePending.Load() == 0 {
			completed.livePending = nil
		}
	}
	if completed.uuidProviders == nil && saved.uuidProviders != nil {
		completed.uuidProviders = saved.uuidProviders.compactForCompleted()
	}
	saved.uuidProviders = completed.uuidProviders
	completed.identity = saved
}

func (tm *traceManager) pruneCompletedToLimitLocked() {
	limit := max(tm.maxTracked, 64)
	tm.pruneWaitingLocked(limit, limit/2)
	tm.pruneCompletedLocked(limit, func(c *completedTrace) bool { return !c.hasPendingOrOpenLateSpans() })
	tm.pruneCompletedLocked(limit, func(c *completedTrace) bool { return !c.hasPendingSpans() })
}

// lateLocked returns the late state of completed, and makes it, with the
// ended calls that completed holds, if it does not exist yet. tm.mu must be
// held.
func (tm *traceManager) lateLocked(completed *completedTrace, sc trace.SpanContext, traceID string, now time.Time) *invocationTrace {
	if completed.livePending != nil && completed.livePending.Load() == 0 {
		completed.livePending = nil
	}
	if completed.late == nil {
		late := newInvocationTraceWithKnownSpan(sc, cmp.Or(traceID, completed.identity.traceID), completed.identity.spanID, now)
		late.isLate = true
		late.restoreEvicted(completed.identity)
		// Sharing the run's ledger lets the activations that start after
		// AfterRun mark the keys they reuse as mixed (see eventUUIDProvider).
		late.uuidProviders.Store(completed.uuidProviders)
		maps.Copy(late.completedModels, completed.endedModels)
		maps.Copy(late.completedTools, completed.endedTools)
		completed.endedModels, completed.endedTools = nil, nil
		completed.late = late
	}
	return completed.late
}

// withCalls returns calls with the calls in more added, and nil if both are
// empty.
func withCalls(calls, more map[string]*spanRecord) map[string]*spanRecord {
	if len(more) == 0 {
		return calls
	}
	if calls == nil {
		return maps.Clone(more)
	}
	maps.Copy(calls, more)
	return calls
}

// pruneCompletedLocked drops the least recently active completed runs for
// which drop reports true until at most limit remain. tm.mu must be held.
func (tm *traceManager) pruneCompletedLocked(limit int, drop func(*completedTrace) bool) {
	for candidate := tm.completionOrder.Front(); len(tm.completed) > limit && candidate != nil; {
		next := candidate.Next()
		id := candidate.Value.(string)
		if drop(tm.completed[id]) {
			delete(tm.completed, id)
			tm.completionOrder.Remove(candidate)
		}
		candidate = next
	}
}

// pruneWaitingLocked drops, while more than limit completed runs remain, the
// least recently active runs that wait for late calls (see
// waitsForLateCalls), as long as more than keep of them remain. tm.mu must be
// held.
func (tm *traceManager) pruneWaitingLocked(limit, keep int) {
	if len(tm.completed) <= limit {
		return
	}
	waiting := 0
	for _, c := range tm.completed {
		if c.waitsForLateCalls() {
			waiting++
		}
	}
	for candidate := tm.completionOrder.Front(); len(tm.completed) > limit && waiting > keep && candidate != nil; {
		next := candidate.Next()
		id := candidate.Value.(string)
		if tm.completed[id].waitsForLateCalls() {
			delete(tm.completed, id)
			tm.completionOrder.Remove(candidate)
			waiting--
		}
		candidate = next
	}
}

// startedCalls returns the calls in calls that pushLLMSpan or pushToolSpan
// started, without the spans that popToolSpan made up for calls whose start
// this plugin did not see, or nil if there are none.
func startedCalls(calls map[string]*spanRecord) map[string]*spanRecord {
	var started map[string]*spanRecord
	for key, rec := range calls {
		if !rec.callStarted {
			continue
		}
		if started == nil {
			started = make(map[string]*spanRecord, len(calls))
		}
		started[key] = rec
	}
	return started
}

// endedWithRun reports whether no callback can end rec, a model or tool call
// that is still open when AfterRun ends the run of runner, the agent that the
// runner ran, because the agent activation that rec runs under finished
// running: ADK runs the callbacks of a call within the run of the activation
// that makes the call. The activation finished running if its run is known to
// be over (see spanRecord.runDone), or if it ran under an activation of runner
// right under the invocation, through activations each of which ran on the
// goroutine of the one above it, or that the one above it waits for before
// its own run returns (see joinsOnceReturned): runner.Run runs AfterRun only
// once the run of runner returned. RunLive runs no agent callbacks for the
// agent that it runs, so an activation of runner right under the invocation
// of a RunLive run is one that a RunLive tool started under runner's name,
// and it is taken for runner's own. Any other call may still end: a call
// under the invocation span itself, as a tool call of RunLive; a call of an
// agent that a RunLive tool runs, or that a custom agent, a function node of
// a workflow agent or a tool runs on a goroutine that it does not wait for;
// a call of an activation that its callback context did not show, as in
// direct traceManager tests; and a call that started under the span that its
// agent would run under rather than its own, because the agent had no open
// span of its run then (see spanRecord.ownActivation). The mu of rec's
// invocation must be held.
func (rec *spanRecord) endedWithRun(runner string) bool {
	a := rec.parent
	if a == nil || !rec.ownActivation || a.kind != spanKindAgent || a.activation == nil || a.goroutineID == 0 {
		return false
	}
	if a.runDone {
		return true
	}
	for p := a.parent; p != nil; a, p = p, p.parent {
		switch {
		case p.kind == spanKindInvocation:
			return a.name == runner && a.activation != nil && a.goroutineID != 0
		case p.kind != spanKindAgent, a.goroutineID == 0,
			a.goroutineID != p.goroutineID && !p.joinsOnceReturned(a):
			return false
		}
	}
	return false
}

// joinsOnceReturned reports whether the run of the agent activation p, once it
// is known to have returned, has also outlasted the run of c, an activation
// that runs right under it on another goroutine: p joins c (see joins); p is
// a ParallelAgent that started c's goroutine for a declared sub-agent, on the
// branch that it assigns to that sub-agent; or p is a workflow agent and c is
// the agent of a workflow.ParallelWorker item (see attemptWorker), whose
// node's Run returns only once the goroutine of every item ended, and whose
// node run the scheduler of p waits for. Since ADK v2.1.0, a ParallelAgent
// waits for every branch before its Run returns, even when its caller stops
// reading early; with ADK v2.0.0 it waits for them only when it returns all
// their events, so a call of a branch that an early stop left running and
// that ends after AfterRun gets a span of its own. The callbacks do not see
// the goroutine that runs the worker's Run, so the items of a worker that a
// function node of p runs on a goroutine that it does not wait for are
// taken for those of a node of p; endedWithRun reads no InvocationContext, so
// a call of such an item that ends after AfterRun then gets a span of its
// own.
func (p *spanRecord) joinsOnceReturned(c *spanRecord) bool {
	return p.joins(c, false) || (p.isParallel && p.goroutineID != 0 && c.origin.startedBy(errgroupCreator, p) &&
		slices.Contains(p.children, c.name) && c.branch == parallelBranch(p, c.name)) ||
		(p.isWorkflow && c.attempt == attemptWorker)
}

// invocationSpan returns the place of a row written by a callback of the
// whole invocation, such as USER_MESSAGE_RECEIVED. Such callbacks get the
// agent that the runner runs, runner, which is used to record the root
// agent.
func (tm *traceManager) invocationSpan(ctx context.Context, id identity, runner agent.Agent) rowSpan {
	customUUIDs := contextHasCustomUUIDProvider(ctx)
	inv := tm.lookupInvocationWithUUID(ctx, id, true, false, customUUIDs)
	now := tm.now()
	rememberedRoot := tm.rememberAgentTree(runner)
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, runner, rememberedRoot)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.started = true
	inv.noteUUIDProviderLocked(id, customUUIDs, true)
	inv.noteRootAgentLocked(id, rootIn, false)
	inv.agentStack[0].canceled = canceledFunc(ctx)
	span := inv.rowSpan(inv.agentStack[0], nil, ambient)
	span.timestamp = now
	return span
}

// runEndTarget binds afterRun's completion and fallback cleanup to the
// invocation trace (or saved evicted identity) of the run being ended, so a
// retry after a recovered panic cannot finish a resumed run that started under
// the same invocation ID in the meantime.
type runEndTarget struct {
	inv        *invocationTrace
	evicted    evictedTrace
	hasEvicted bool
	bound      bool
}

func (t runEndTarget) rootSpanID() string {
	if t.inv != nil {
		return t.inv.rootSpanID
	}
	if t.hasEvicted {
		return t.evicted.spanID
	}
	return ""
}

func (tm *traceManager) captureRunEndTarget(invocationID string) runEndTarget {
	if tm == nil || invocationID == "" {
		return runEndTarget{}
	}
	tm.mu.Lock()
	defer tm.mu.Unlock()
	if inv := tm.invocations[invocationID]; inv != nil {
		return runEndTarget{inv: inv, bound: true}
	}
	if ev, ok := tm.evicted[invocationID]; ok {
		return runEndTarget{evicted: ev, hasEvicted: true, bound: true}
	}
	return runEndTarget{}
}

func (tm *traceManager) refreshRunEndTarget(invocationID string, cur runEndTarget) runEndTarget {
	if tm == nil || invocationID == "" {
		return cur
	}
	tm.mu.Lock()
	defer tm.mu.Unlock()
	inv := tm.invocations[invocationID]
	if inv == nil {
		return cur
	}
	if !cur.bound || inv == cur.inv || (cur.rootSpanID() != "" && inv.rootSpanID == cur.rootSpanID()) {
		return runEndTarget{inv: inv, bound: true}
	}
	return cur
}

func finishDetachedTrace(inv *invocationTrace) {
	inv.mu.Lock()
	defer inv.mu.Unlock()
	if inv.finished {
		return
	}
	if len(inv.agentStack) > 0 && inv.agentStack[0] != nil {
		inv.agentStack[0].dropContexts()
	}
	inv.runnerAgent = nil
	for _, rec := range inv.llmSpans {
		rec.dropContexts()
	}
	for _, rec := range inv.toolSpans {
		rec.dropContexts()
	}
	inv.finished = true
	clear(inv.llmSpans)
	clear(inv.completedModels)
	clear(inv.toolSpans)
	clear(inv.completedTools)
	inv.syncOpenCallsLocked()
}

// endInvocation stops tracking the invocation and returns the place of its
// INVOCATION_COMPLETED row, with the invocation latency if the invocation
// was tracked from its start without being evicted mid-run. Its identity stays
// in a separate bounded cache for late callbacks. A new run with the same ID
// discards that completed state; capacity pruning can remove idle entries earlier.
func (tm *traceManager) endInvocation(ctx context.Context, id identity) rowSpan {
	return tm.endInvocationTarget(ctx, id, nil)
}

func (tm *traceManager) endInvocationTarget(ctx context.Context, id identity, target *runEndTarget) rowSpan {
	now := tm.now()
	sc := ambientSpanContext(ctx)
	ambient := ""
	if sc.IsValid() {
		ambient = sc.TraceID().String()
	}
	var inv *invocationTrace
	if id.invocationID != "" {
		func() {
			tm.mu.Lock()
			defer tm.mu.Unlock()
			inv = tm.invocations[id.invocationID]
			if inv != nil && target != nil && target.bound {
				wantSpanID := target.rootSpanID()
				if inv != target.inv && (wantSpanID == "" || inv.rootSpanID != wantSpanID) {
					inv = target.inv
				}
			}
			if inv == nil && target != nil && target.inv != nil {
				inv = target.inv
			}
			if inv == nil {
				if ev, ok := tm.evicted[id.invocationID]; ok {
					inv = newInvocationTraceWithKnownSpan(sc, cmp.Or(ev.traceID, tm.traceIDFor(id.invocationID)), ev.spanID, now)
					inv.restoreEvicted(ev)
				} else if target != nil && target.hasEvicted {
					inv = newInvocationTraceWithKnownSpan(sc, cmp.Or(target.evicted.traceID, tm.traceIDFor(id.invocationID)), target.evicted.spanID, now)
					inv.restoreEvicted(target.evicted)
				} else if completed := tm.completed[id.invocationID]; completed != nil {
					inv = newInvocationTraceWithKnownSpan(sc, cmp.Or(completed.identity.traceID, tm.traceIDFor(id.invocationID)), completed.identity.spanID, now)
					inv.restoreEvicted(completed.identity)
				} else {
					inv = newInvocationTraceWithSpan(sc, tm.traceIDFor(id.invocationID), now)
				}
			}
			if target != nil {
				if !target.bound {
					*target = runEndTarget{inv: inv, bound: true}
				} else if target.inv == nil {
					target.inv = inv
				}
			}
		}()
	}
	if inv == nil {
		inv = newInvocationTraceWithSpan(sc, tm.traceIDFor(id.invocationID), now)
	}
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	var row rowSpan
	func() {
		inv.mu.Lock()
		defer inv.mu.Unlock()
		inv.noteRootAgentLocked(id, rootIn, false)
		var latency map[string]any
		if inv.started && !inv.evicted {
			latency = latencyOf(inv.agentStack[0], now)
		}
		row = inv.rowSpan(inv.agentStack[0], latency, ambient)
		row.timestamp = now
	}()
	if id.invocationID != "" {
		func() {
			tm.mu.Lock()
			defer tm.mu.Unlock()
			if cur := tm.invocations[id.invocationID]; cur != nil && cur != inv {
				if cur.rootSpanID != inv.rootSpanID {
					finishDetachedTrace(inv)
					return
				}
				// The same root span was evicted and restored into cur between lookup
				// and publication: publish inv first (keeping any adopted trace ID and
				// calls from inv), then publish and unlink cur so cur does not remain
				// active in tm.invocations or tm.evicted.
				tm.rememberCompletedAtLocked(id.invocationID, id.agentName, inv, now)
				tm.rememberCompletedAtLocked(id.invocationID, id.agentName, cur, now)
				return
			}
			// Publish the final identity, including any trace adopted by rowSpan,
			// and remove active state before marking inv finished. Late Live
			// callbacks may start and end an operation, so their state is shared
			// separately from the finished run.
			tm.rememberCompletedAtLocked(id.invocationID, id.agentName, inv, now)
		}()
	}
	return row
}

func (tm *traceManager) endInvocationRetry(ctx context.Context, id identity, target runEndTarget) {
	if tm == nil || id.invocationID == "" {
		return
	}
	if target.inv != nil {
		var alreadyFinished bool
		func() {
			target.inv.mu.Lock()
			defer target.inv.mu.Unlock()
			alreadyFinished = target.inv.finished
		}()
		if alreadyFinished {
			return
		}
	}
	now := tm.now()
	if target.inv != nil {
		rootIn := prepareRootAgentInput(ctx, target.inv, id, nil, nil)
		func() {
			target.inv.mu.Lock()
			defer target.inv.mu.Unlock()
			if !target.inv.finished {
				target.inv.noteRootAgentLocked(id, rootIn, false)
			}
		}()
	}
	tm.completeRunEndTarget(id, target, now)
}

// forceEndInvocation publishes completion bookkeeping and clears active and
// evicted tracking for id without calling ctx methods or tm.now, used as a
// panic-safe fallback when endInvocation itself panics.
func (tm *traceManager) forceEndInvocation(id identity) {
	if tm == nil || id.invocationID == "" {
		return
	}
	now := time.Now()
	tm.mu.Lock()
	defer tm.mu.Unlock()
	if inv := tm.invocations[id.invocationID]; inv != nil {
		tm.rememberCompletedAtLocked(id.invocationID, id.agentName, inv, now)
		return
	}
	var saved evictedTrace
	if ev, ok := tm.evicted[id.invocationID]; ok {
		saved = ev
		if saved.traceID == "" {
			saved.traceID = tm.traceIDFor(id.invocationID)
		}
	} else if completed := tm.completed[id.invocationID]; completed != nil {
		saved = completed.identity
	} else {
		saved = evictedTrace{
			traceID:       tm.traceIDFor(id.invocationID),
			rootAgentName: id.agentName,
			lastActive:    now.UnixNano(),
		}
	}
	tm.publishCompletedIdentityLocked(id.invocationID, saved)
	tm.pruneCompletedToLimitLocked()
}

func (tm *traceManager) forceEndInvocationTarget(id identity, target runEndTarget) {
	if tm == nil || id.invocationID == "" {
		return
	}
	tm.completeRunEndTarget(id, target, time.Now())
}

func (tm *traceManager) completeRunEndTarget(id identity, target runEndTarget, now time.Time) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	if target.inv != nil {
		var alreadyFinished bool
		func() {
			target.inv.mu.Lock()
			defer target.inv.mu.Unlock()
			alreadyFinished = target.inv.finished
		}()
		if alreadyFinished {
			return
		}
		if cur := tm.invocations[id.invocationID]; cur != nil {
			if cur == target.inv {
				tm.rememberCompletedAtLocked(id.invocationID, id.agentName, cur, now)
				return
			}
			if target.inv.rootSpanID != "" && cur.rootSpanID == target.inv.rootSpanID {
				tm.rememberCompletedAtLocked(id.invocationID, id.agentName, target.inv, now)
				tm.rememberCompletedAtLocked(id.invocationID, id.agentName, cur, now)
				return
			}
			finishDetachedTrace(target.inv)
			return
		}
		tm.rememberCompletedAtLocked(id.invocationID, id.agentName, target.inv, now)
		return
	}
	if target.hasEvicted {
		if cur := tm.invocations[id.invocationID]; cur != nil {
			if target.evicted.spanID != "" && cur.rootSpanID == target.evicted.spanID {
				tm.rememberCompletedAtLocked(id.invocationID, id.agentName, cur, now)
			}
			return
		}
		if completed := tm.completed[id.invocationID]; completed != nil && completed.identity.spanID == target.evicted.spanID {
			if ev, ok := tm.evicted[id.invocationID]; ok && ev.spanID == target.evicted.spanID {
				delete(tm.evicted, id.invocationID)
			}
			return
		}
		saved := target.evicted
		if ev, ok := tm.evicted[id.invocationID]; ok {
			if ev.spanID != target.evicted.spanID {
				return
			}
			saved = ev
		}
		if saved.traceID == "" {
			saved.traceID = tm.traceIDFor(id.invocationID)
		}
		tm.publishCompletedIdentityLocked(id.invocationID, saved)
		tm.pruneCompletedToLimitLocked()
		return
	}
	if tm.invocations[id.invocationID] != nil || tm.completed[id.invocationID] != nil {
		return
	}
	if _, ok := tm.evicted[id.invocationID]; ok {
		return
	}
	tm.publishCompletedIdentityLocked(id.invocationID, evictedTrace{
		traceID:       tm.traceIDFor(id.invocationID),
		rootAgentName: id.agentName,
		lastActive:    now.UnixNano(),
	})
	tm.pruneCompletedToLimitLocked()
}

// popDelegatingAgentSpan ends the activation of the agent of id that a task
// delegation interrupted, as the agent starts again after its task-mode
// sub-agent completed via finish_task, and returns the place of the
// activation's AGENT_COMPLETED row. ADK's chat loop runs the task while the
// delegating activation waits, then stops reading that activation, so ADK
// runs no AfterAgentCallback for it, and runs the agent again. Only an
// activation that popTaskAgentSpan marked as taskDelegated ends: an ordinary
// transfer-back chain (planner → helper → planner) leaves the first planner
// with hasResponse == true (its transfer call counted as a response) while it
// is still running, so ending it on hasResponse would end a live span. The
// agent's new activation then starts under the parent, as a sibling of the
// ended one. ok is false if no such activation is open.
//
// Divergence from adk-python (google/adk-python@25d191710985): Python ADK
// closes successful agent activations in after_agent_callback (lines
// 10243-10263) and failed ones in on_agent_error_callback (lines 10827-10869,
// which pops the agent span at line 10850), a callback that Go ADK does not
// have. Go emits AGENT_COMPLETED here on task-mode delegation interruption and
// in popTaskAgentSpan on finish_task because llmagent.ModeTask sub-agents
// complete without firing AfterAgentCallback.
func (tm *traceManager) popDelegatingAgentSpan(ctx context.Context, id identity) (span rowSpan, ok bool) {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	ended := endedFunc(ctx)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgentLocked(id, rootIn, true)
	for i := len(inv.agentStack) - 1; i > 0; i-- {
		rec := inv.agentStack[i]
		if rec.kind == spanKindAgent && rec.name == id.agentName && rec.branch == id.branch && rec.path == id.path && rec.taskDelegated {
			defer inv.retireLateSettledLocked()
			if rec.parent != nil && rec.parent.kind == spanKindAgent {
				inv.touchAgentActivityLocked(rec.parent, now)
			}
			// As in pushAgentSpan, which runs in the same callback, the runs that
			// started on this goroutine before are over only if its trace shows
			// a node attempt (see sameGoroutineSince there).
			since := noFrame
			if ended != nil {
				if _, _, attempt := currentGoroutine(); attempt != attemptNone {
					since = 0
				}
			}
			// The mark comes from the task's finish_task event and the start
			// of a new activation of the agent, neither of which shows that
			// the delegating activation's run returned, so its
			// InvocationContext is not read (see popTaskAgentSpan).
			inv.retireAgentLocked(i, false, since)
			span = inv.rowSpan(rec, latencyOf(rec, now), ambient)
			span.timestamp = now
			return span, true
		}
	}
	return rowSpan{}, false
}

// pushAgentSpan starts the span of an agent run and returns the place of
// its AGENT_STARTING row together with any prior failed activations purged on
// re-entry. The sub-agents that are configured on the starting agent are
// recorded on its activation span before they start, so that agents that run
// under a workflow agent outside the runner's agent tree, such as a graph
// node, are parented like those of the tree without overwriting sibling graph
// nodes. If a workflow node is re-entered on the same node path by a later
// attempt of the node (see laterAttempt below) while the earlier attempt
// failed under the activation that the new one runs under, below that
// activation's node path, as when RetryConfig runs a failed node again in one
// run of its workflow agent, the earlier attempt is purged: ADK handled that
// failure without yielding it, so the attempt's failed activations are
// returned for the caller to write as AGENT_ERROR rows before the new
// attempt's AGENT_STARTING row without failing the invocation (see
// purgeActivationLocked). An attempt that failed under another activation, as
// when the workflow agent itself runs again, is kept, and so is its error.
// Re-entering the same node path under a new parent activation — for example
// when a caller runs a workflow agent again after receiving an error from its
// previous run — keeps the yielded failure. Only the agent of a node attempt
// counts as an attempt (see attemptSite): only an agent of an agent node or
// ParallelWorker attempt counts, not an agent that a function node or a custom
// agent runs itself, whether the function or custom agent ran it again after
// recovering from its error or inside its loop over the events of an earlier
// run, or RetryConfig retried the function node after the function returned
// the agent's error. Such an earlier run is the code's own to handle. An
// earlier attempt that nothing shows to be one, such as one for which the
// workflow reported no completion (see recordNodeCompletion), is a concurrent
// peer: it is neither read nor purged here, and a failed earlier attempt of a
// node that the workflow started again is purged when the workflow agent ends
// instead (see endRestartedNodeRuns). A node completion event decides whether
// an earlier attempt is purged, but does not show that its run is over, as a
// node can write the same metadata on an event of its own while it runs, so
// the earlier attempt's InvocationContext is not read on its strength (see
// spanRecord.nodeRunEnded).
func (tm *traceManager) pushAgentSpan(ctx context.Context, id identity) (rowSpan, []unclosedAgentSpan) {
	customUUIDs := contextHasCustomUUIDProvider(ctx)
	inv := tm.lookupInvocationWithUUID(ctx, id, false, true, customUUIDs)
	defer inv.pendingSpans.Add(-1)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	selfPlugin := tm.selfPlugin.Value()
	// SubAgents and any custom platform TimeProvider are user code, so both are
	// read before mu is taken; non-task agents skip platform.Now so registering
	// the plugin does not advance an application clock on ordinary runs.
	children := runningAgentChildren(ctx, id.agentName, selfPlugin)
	var eventClockStart time.Time
	if children.taskMode {
		eventClockStart = platform.Now(ctx)
	}
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	activation := activationKey(ctx)
	ended := endedFunc(ctx)
	var goid uint64
	var origin goroutineOrigin
	var attempt attemptSite
	if ended != nil {
		goid, origin, attempt = currentGoroutine()
	}
	// The trace of a node attempt shows no other agent run on this goroutine
	// (see attemptSiteOfTrace), so every activation that ran on it before is
	// over; otherwise an agent run that started on it earlier may still be
	// suspended below this callback, inside its loop over the events of
	// another run (see canSettleInSubtreeLocked).
	sameGoroutineSince := noFrame
	if attempt != attemptNone {
		sameGoroutineSince = 0
	}
	// As for a model or tool call, an agent that starts while AfterRun ends
	// the run starts in the late state, where its later callbacks look for
	// it. Once rememberCompletedLocked marked inv finished, lockLive leaves
	// inv unlocked and reports false, and the recursive call looks id up
	// again (see lookupInvocation), which finds the late state of the
	// completed entry while that entry remains. The reservation of the first
	// lookup stays held until its deferred Add, after the recursive call
	// returns, so that completed.livePending stays set meanwhile, and the
	// completed entry counts as one in which a callback is starting a span
	// when completed runs are pruned (see rememberCompletedLocked). An agent
	// that starts after AfterRun's popUnclosedAgentSpans but before
	// rememberCompletedLocked marks inv finished still starts on the live
	// trace, which hands over only its open model and tool calls: the
	// agent's AfterAgentCallback then finds the late state and writes a span
	// of its own, without latency (see popAgentSpan).
	if !inv.lockLive() {
		return tm.pushAgentSpan(ctx, id)
	}
	defer inv.mu.Unlock()
	inv.started = true
	inv.noteRootAgentLocked(id, rootIn, true)
	inv.noteUUIDProviderLocked(id, customUUIDs, false)
	// Register concurrent peers before the workflow-retry completion-inference
	// loop so the retry guard can check !rec.concurrentPeer. Only peers that
	// are NOT earlier attempts of the node are marked (see laterAttempt).
	parent := inv.parentForRunLocked(id, nil, goid, origin, activation)
	var concurrentPeer bool
	isSyntheticRetry := func(other *spanRecord) bool {
		return activation == nil && other.activation == nil && other.parent != nil && other.parent.activation == nil
	}
	// laterAttempt reports whether the activation of id is a later attempt
	// of the workflow node that other ran as its agent (see attemptSite), so
	// that other's failure is handled: other runs, with another activation,
	// under a parent span whose node path lies above id.path, and either
	//
	//   - both are the agent of a workflow.ParallelWorker item that ran on
	//     this goroutine, as when RetryConfig runs the item's node again on
	//     the worker's goroutine (even when other.parent is not the workflow
	//     agent, as when concurrent lookalike activations moved other up to
	//     the span that encloses them; see parentForRunLocked). The stack of
	//     this callback shows no other agent run (see attemptSiteOfTrace), so
	//     other's run, and every run that started on this goroutine before,
	//     has returned, and synced reports that other's InvocationContext can
	//     be read; or
	//   - both are the agent of a node attempt under a workflow agent
	//     (other.parent.isWorkflow) whose scheduler started both attempts on
	//     goroutines of their own (see startedBy), and a node completion event
	//     reported that the node run that ran other ended (see
	//     spanRecord.nodeRunEnded). A node can write the same metadata on an
	//     event of its own while it still runs, so the report does not show
	//     that other's run is over: synced is false, other's InvocationContext
	//     is not read, and other is purged now only for a failure that was
	//     recorded for it, not for having no response yet or for a model or
	//     tool call that it or a run under it still has open.
	//
	// Nothing else shows that other is an earlier attempt: the agent runs
	// that a function node or a custom agent makes, one after another or one
	// inside its loop over the events of another, share the parent, node
	// path, branch and often the goroutine of an earlier attempt, and may
	// each have a context of their own. Such an earlier activation is a
	// concurrent peer, which is neither read nor purged here. The
	// synthetic-retry fallback (activation == nil on every span in the chain)
	// applies only to direct traceManager unit tests, not to real Live runs
	// whose invocation-root span also has activation == nil.
	laterAttempt := func(other *spanRecord) (later, synced bool) {
		if other.parent == nil || !isSubNodePath(id.path, other.parent.path) || (activation != nil && other.activation == activation) {
			return false, false
		}
		if isSyntheticRetry(other) {
			return true, true
		}
		switch {
		case attempt == attemptWorker && other.attempt == attemptWorker && goid != 0 && other.goroutineID == goid:
			// A ParallelWorker item retry runs sequentially on the exact same
			// runWorker goroutine, even when concurrent lookalike activations
			// moved other up to a span that is not the workflow agent.
			return true, true
		case attempt == attemptNode && other.attempt == attemptNode && other.parent.isWorkflow && other.nodeRunEnded &&
			origin.startedBy(schedulerCreator, other.parent) && other.origin.startedBy(schedulerCreator, other.parent):
			return true, false
		}
		return false, false
	}
	// The workflow scheduler starts a node again, on the same branch and node
	// path, when RetryConfig retries it or when a running node is triggered
	// again.
	restarted := attempt == attemptNode && parent.isWorkflow && origin.startedBy(schedulerCreator, parent)
	for _, other := range inv.agentStack[1:] {
		if other.kind != spanKindAgent ||
			other.parent != parent || other.name != id.agentName || other.branch != id.branch || other.path != id.path {
			continue
		}
		if restarted && other.attempt == attemptNode && other.goroutineID != goid && other.origin.startedBy(schedulerCreator, parent) {
			other.startedAgain = true
		}
		if later, _ := laterAttempt(other); later {
			continue
		}
		other.concurrentPeer = true
		concurrentPeer = true
	}
	// Workflow-retry completion-inference loop.
	var purged []unclosedAgentSpan
	var didPurge bool
	for i := len(inv.agentStack) - 1; i > 0; i-- {
		rec := inv.agentStack[i]
		if rec.kind != spanKindAgent || rec.name != id.agentName || rec.branch != id.branch || rec.path != id.path {
			continue
		}
		// A workflow node stops reading its agent at the first error, so a
		// failed attempt gets no AfterAgentCallback, and RetryConfig runs the
		// node again on the same path under the same live workflow activation
		// once ADK has handled the failure without yielding it. A new run of
		// an enclosing workflow agent reuses the same node paths under a new
		// workflow activation after yielding the earlier node error to its
		// caller, so rec.parent differs from the new activation's parent (with
		// the failed attempt's own subtree excluded) and the yielded failure
		// is kept. A healthy earlier activation, such as the first planner of
		// a transfer-back chain, a failed one outside a node, concurrent runs
		// under a non-WorkflowAgent parent (even when PromoteWithDelta assigns
		// a sub-node path), the runs that a function node or a custom agent
		// makes, and an earlier attempt that nothing shows to be one (see
		// laterAttempt) are kept too: they may still run, or their failure is
		// the code's own to handle, so neither their InvocationContext nor
		// whether they failed is read here.
		later, synced := laterAttempt(rec)
		sameWorker := attempt == attemptWorker && rec.attempt == attemptWorker && goid != 0 && rec.goroutineID == goid
		if !later || (!sameWorker && (rec.concurrentPeer || rec.parent != inv.parentForRunLocked(id, inv.subtreeLocked(i), goid, origin, activation))) {
			continue
		}
		if synced {
			// The earlier attempt's run returned (see laterAttempt).
			inv.settleEndedInSubtreeLocked(i, true, sameGoroutineSince)
		}
		// Without that, only a report shows that the earlier attempt's run
		// ended, and a node can write one while its run goes on, so the
		// attempt's InvocationContext is not read and the attempt fails here
		// only by what was recorded for it, such as an error or a failure
		// cause, not by having no response yet, nor by a model or tool call
		// that it or a run under it has open: the run may still be making
		// that call, which then ends as usual. If it failed silently, or its
		// open call is a failure, the workflow agent's end purges it (see
		// endRestartedNodeRuns). Once a recorded failure decides the purge,
		// the report also shows which runs under the attempt ended with it
		// without responding or with a call open (see reportedOverLocked).
		if failures := inv.failuresLocked(i, false, nil, synced); len(failures) > 0 {
			if !synced {
				failures = inv.failuresLocked(i, false, inv.reportedOverLocked(i), false)
			}
			purged = inv.purgeActivationLocked(i, failures, now, ambient)
			didPurge = true
			break
		}
	}
	// Mark concurrent peers on any remaining matching spans (workflow
	// children that did not fail and were not purged by the retry loop).
	if didPurge {
		parent = inv.parentForRunLocked(id, nil, goid, origin, activation)
	}
	for _, other := range inv.agentStack[1:] {
		if other.kind == spanKindAgent &&
			other.parent == parent && other.name == id.agentName && other.branch == id.branch && other.path == id.path {
			other.concurrentPeer = true
			concurrentPeer = true
		}
	}
	rec := &spanRecord{
		spanID:                 newSpanID(),
		parentSpanID:           parent.spanID,
		kind:                   spanKindAgent,
		path:                   id.path,
		name:                   id.agentName,
		branch:                 id.branch,
		start:                  now,
		lastActivity:           now,
		eventClockStart:        eventClockStart,
		canceled:               canceledFunc(ctx),
		ended:                  ended,
		goroutineID:            goid,
		origin:                 origin,
		attempt:                attempt,
		activation:             activation,
		parent:                 parent,
		stopPoint:              parent.kind == spanKindInvocation || id.path != parent.path || id.branch != parent.branch,
		nonLLM:                 children.nonLLM,
		propagatesChildError:   children.propagatesChildError,
		isParallel:             children.isParallel,
		isWorkflow:             children.isWorkflow,
		concurrentPeer:         concurrentPeer,
		taskMode:               children.taskMode,
		hasAfterAgentCallbacks: children.hasAfterAgentCallbacks || (selfPlugin != nil && children.parent == ""),
		children:               children.names,
		seq:                    inv.nextSeqLocked(),
	}
	inv.pushAgentLocked(rec)
	if rec.isNodeAgentOf(parent) {
		if parent.nodeRuns == nil {
			parent.nodeRuns = make(map[nodeRunKey]int)
		}
		parent.nodeRuns[nodeRunKey{path: rec.path, branch: rec.branch}]++
	}
	inv.syncOpenAgentsLocked()
	inv.clearLastPartialUnderLocked(rec)
	if parent != nil && parent.kind == spanKindAgent {
		inv.touchAgentActivityLocked(parent, now)
	}
	span := inv.rowSpan(rec, nil, ambient)
	span.timestamp = now
	return span, purged
}

// agentNamed returns the agent of id, looked up by name in the tree of the
// agent that the runner runs, or nil if it is not found there.
func (tm *traceManager) agentNamed(ctx context.Context, id identity) agent.Agent {
	inv := tm.invocation(ctx, id)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	for !inv.lockLive() {
		inv = tm.invocation(ctx, id)
	}
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
// recorded, that owns a model or tool call that never ended, or that never
// responded, such as a custom agent whose Run returned an error before any
// event, stays open, and so does the error, so that the end of the invocation
// reports the failure. ADK ends the agents above a failed sub-agent normally
// when the caller keeps reading after it yielded the error, and whether the
// invocation failed must not depend on how the caller iterates.
//
// The end of the agent that the runner runs directly, a workflow agent or a
// custom agent as opposed to an llmagent, which runs as a workflow node, also
// shows that the caller read past the last partial event: ADK runs its
// AfterAgentCallback only if every event it yielded, an error included, was
// read on, so a run that fails while another branch streams does not look
// like a caller that stopped reading mid-stream (see clearLastPartialLocked).
func (tm *traceManager) popAgentSpan(ctx context.Context, id identity) rowSpan {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	defer inv.retireLateSettledLocked()
	inv.noteRootAgentLocked(id, rootIn, true)
	i := inv.findAgentForContext(ctx, id.path, id.agentName, id.branch)
	if i < 0 {
		span := inv.rowSpan(inv.newChild(inv.parentFor(id), spanKindAgent, now), nil, ambient)
		span.timestamp = now
		return span
	}
	rec := inv.agentStack[i]
	rec.completed = true
	if rec.parent != nil && rec.parent.kind == spanKindAgent {
		inv.touchAgentActivityLocked(rec.parent, now)
	}
	if rec.nonLLM && rec.parent != nil && rec.parent.kind == spanKindInvocation {
		inv.clearLastPartialLocked()
	}
	inv.reconcileConcurrentResponseOnPopLocked(i, rec.seq)
	inv.retireAgentLocked(i, true, rec.seq)
	span := inv.rowSpan(rec, latencyOf(rec, now), ambient)
	span.timestamp = now
	return span
}

// endRestartedNodeRuns purges, when the activation of a workflow agent that id
// and ctx identify ends, the failed activations that its scheduler started as
// nodes and then started again on the same branch and node path (see
// spanRecord.startedAgain), and returns them for the caller to write as
// AGENT_ERROR rows before the workflow agent's AGENT_COMPLETED row (see
// purgeActivationLocked). The scheduler waits for every node run that it
// started before the workflow agent's Run returns, and ADK runs the agent's
// AfterAgentCallback, which calls this, only after that. Such an activation
// is the agent of a node attempt that ran right under the node's Run on a
// goroutine that the scheduler started (see attemptNode), so its run is over
// too, and so is the run of each activation under it that the agents between
// them wait for (see settleEndedInSubtreeLocked and reportedOverLocked).
// RetryConfig handled the failure of an earlier attempt without yielding it,
// but pushAgentSpan purges such an attempt when the next one starts only if a
// node completion event reported that its run ended (see laterAttempt) and a
// failure was recorded for it. ADK reports no node runs in releases before
// node completion events, nor for a node that the scheduler started again
// while it ran, and an attempt that failed without a recorded failure, by not
// responding, is not known to have stopped until here.
func (tm *traceManager) endRestartedNodeRuns(ctx context.Context, id identity) []unclosedAgentSpan {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	i := inv.findAgentForContext(ctx, id.path, id.agentName, id.branch)
	if i < 0 {
		return nil
	}
	w := inv.agentStack[i]
	if !w.isWorkflow {
		return nil
	}
	var purged []unclosedAgentSpan
	// purgeActivationLocked removes agentStack[j] and the spans that run under
	// it, which all come after it, so the scan goes backwards.
	for j := len(inv.agentStack) - 1; j > i; j-- {
		if j >= len(inv.agentStack) {
			continue
		}
		rec := inv.agentStack[j]
		if rec.kind != spanKindAgent || rec.parent != w || !rec.startedAgain || !rec.origin.startedBy(schedulerCreator, w) {
			continue
		}
		inv.settleEndedInSubtreeLocked(j, true, w.seq)
		if failures := inv.failuresLocked(j, false, inv.reportedOverLocked(j), true); len(failures) > 0 {
			purged = append(purged, inv.purgeActivationLocked(j, failures, now, ambient)...)
		}
	}
	return purged
}

// recordNodeCompletion records the end of a node run that ADK's workflow
// scheduler reports with a node completion event: an event with the node's
// path and branch whose custom metadata holds a node outcome or a completed
// flag (see workflowNodeOutcomeKey) and the base path of the workflow that
// ran the node, the parent of the node's path, under
// workflowBasePathPrefix+workflowName. The scheduler writes it only after the
// node's Run returned, but a node can write the same metadata on an event of
// its own while it runs, and the two look alike here, so the event reports
// how a run ended without showing that it is over: nothing reads the
// InvocationContext of an activation on its strength. The run is attributed
// only when that is unambiguous: one open activation of a workflow agent
// named workflowName counts one node run on the path and branch whose end no
// completion reported yet (see spanRecord.nodeRuns). A failed or canceled run
// then marks the open activation that the scheduler started as the node's
// agent for it (see isNodeAgentOf), if there is exactly one, as reported over
// (see spanRecord.nodeRunEnded); if no such activation is open anymore, its
// run ended earlier. A run that ends with an output gets no completion event
// that is read here. Only a resumed run that ends with an output gets a
// record, under a key of its own (adk.workflow.node_resume_completed) that
// nodeCompletion does not count. So once a node completed with an output,
// later runs of it on the same path and branch are not attributed; neither are runs of a node that the
// scheduler started again while it ran, as a node that two join nodes route
// to may be.
func (tm *traceManager) recordNodeCompletion(ctx context.Context, id identity, workflowName, path, branch string, failed bool) {
	inv := tm.invocation(ctx, id)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	key := nodeRunKey{path: path, branch: branch}
	var w *spanRecord
	for _, rec := range inv.agentStack[1:] {
		if rec.kind != spanKindAgent || !rec.isWorkflow || rec.name != workflowName || rec.nodeRuns[key] == 0 {
			continue
		}
		if w != nil {
			return
		}
		w = rec
	}
	if w == nil || w.nodeRuns[key] != 1 {
		return
	}
	var run *spanRecord
	for _, rec := range inv.agentStack[1:] {
		if rec.parent != w || rec.path != path || rec.branch != branch || rec.nodeRunEnded || !rec.isNodeAgentOf(w) {
			continue
		}
		if run != nil {
			return
		}
		run = rec
	}
	switch {
	case run == nil:
		delete(w.nodeRuns, key)
	case failed:
		run.nodeRunEnded = true
		delete(w.nodeRuns, key)
	}
}

// nodeRunKey is the node path and branch of a workflow node run.
type nodeRunKey struct {
	path, branch string
}

// isNodeAgentOf reports whether the scheduler of w, a workflow agent
// activation, started rec as the agent of one of w's nodes: rec runs under w
// as the agent of a node attempt (see attemptNode) on a goroutine that w's
// scheduler started (see startedBy), and the last element of rec's node path
// names a node of rec's own name, as the path of an agent node does. An agent
// that a function node of the same name runs is not one.
func (rec *spanRecord) isNodeAgentOf(w *spanRecord) bool {
	if w == nil || !w.isWorkflow || rec.parent != w || rec.attempt != attemptNode || !rec.origin.startedBy(schedulerCreator, w) {
		return false
	}
	node := rec.path[strings.LastIndexByte(rec.path, '/')+1:]
	if at := strings.LastIndexByte(node, '@'); at >= 0 {
		node = node[:at]
	}
	return node == rec.name
}

// inWorkflowNodeSubtreeLocked reports whether rec is at or under an open
// workflow node activation (see isNodeAgentOf) whose run is known to be over
// (runDone). When a caller stops reading on a partial event, ADK's workflow
// scheduler cancels every running node with context.Canceled (so
// canceledByStop holds on the node and its subtree) and waits for all node
// goroutines before Workflow.Run returns, whereas a node that returns an
// error is removed from runCancels before cancelAll and keeps an uncanceled
// context even on ADK releases before workflow node completion events.
func (inv *invocationTrace) inWorkflowNodeSubtreeLocked(rec *spanRecord) bool {
	if rec == nil || !rec.runDone {
		return false
	}
	for cur := rec; cur != nil && cur.kind == spanKindAgent; cur = cur.parent {
		if cur.isNodeAgentOf(cur.parent) {
			return true
		}
	}
	return false
}

// reconcileConcurrentResponseOnPopLocked conserves session-event response
// credits when agentStack[i] retires cleanly while one or more peer
// activations of the same (path, name, branch) are still open on agentStack.
// If completeResponse earlier routed rec's session event to a newer open peer,
// the credit is moved back to rec; otherwise, if rec retired before its
// buffered session event reached completeResponse, unrespondedRetired records
// the pending event so completeResponse does not mark a failing peer responded.
// Silent custom activations that emit no content retire normally via
// AfterAgentCallback without remaining open on agentStack. since bounds the
// activations that settleEndedInSubtreeLocked takes to be over because they
// ran on the current goroutine.
func (inv *invocationTrace) reconcileConcurrentResponseOnPopLocked(i int, since uint64) {
	rec := inv.agentStack[i]
	inv.settleEndedInSubtreeLocked(i, true, since)
	var livePeers []*spanRecord
	for j, other := range inv.agentStack {
		if j == 0 || j == i {
			continue
		}
		if other.kind == spanKindAgent && other.path == rec.path && other.name == rec.name && other.branch == rec.branch {
			livePeers = append(livePeers, other)
		}
	}
	k := llmKey(identity{path: rec.path, agentName: rec.name, branch: rec.branch})
	if rec.concurrentPeer || len(livePeers) > 0 {
		if inv.retiredPeerCount == nil {
			inv.retiredPeerCount = make(map[string]int)
		}
		inv.retiredPeerCount[k]++
	}
	if len(livePeers) == 0 {
		return
	}
	if !rec.responded() {
		for _, other := range livePeers {
			if other.extraResponses > 0 {
				rec.hasResponse = true
				other.extraResponses--
				return
			}
		}
		for _, other := range livePeers {
			if other.hasResponse && !other.modelResponded {
				rec.hasResponse = true
				other.hasResponse = false
				return
			}
		}
		if inv.unrespondedRetired == nil {
			inv.unrespondedRetired = make(map[string]int)
		}
		inv.unrespondedRetired[k]++
		return
	}
	if pending := rec.awaitingModelEvent + rec.awaitingToolEvent; pending > 0 {
		if inv.unrespondedRetired == nil {
			inv.unrespondedRetired = make(map[string]int)
		}
		inv.unrespondedRetired[k] += pending
	}
}

type taskAttemptKind uint8

const (
	// taskAttemptPending is a finish_task call that has started or whose
	// observed result in afterTool did not carry the task-ending sentinel,
	// which a later AfterToolCallback may still promote before the
	// FunctionResponse event is emitted.
	taskAttemptPending taskAttemptKind = iota
	// taskAttemptCompleted is a finish_task call whose observed result in
	// afterTool carried the task-ending sentinel without a classified tool
	// result error.
	taskAttemptCompleted
	// taskAttemptFailedEnd is a finish_task call whose observed result in
	// afterTool carried the task-ending sentinel together with a classified
	// tool result error.
	taskAttemptFailedEnd
)

// taskDispatchMaps holds weak references to the StateDelta and ArtifactDelta
// map headers allocated for a tool call's EventActions. ADK copies EventActions
// by struct value onto the resulting FunctionResponse session.Event
// (ev.Actions = *toolCtx.Actions()), so both map headers remain reachable on
// any buffered single-part or leading-part FunctionResponse event without
// taskAttempt retaining user state payloads or risking recycled uintptr
// addresses after garbage collection.
type taskDispatchMaps struct {
	stateDelta    weak.Pointer[byte]
	artifactDelta weak.Pointer[byte]
}

func mapHeaderWeakPtr[K comparable, V any](m map[K]V) weak.Pointer[byte] {
	if m == nil {
		return weak.Pointer[byte]{}
	}
	p := (*byte)(reflect.ValueOf(m).UnsafePointer())
	if p == nil {
		return weak.Pointer[byte]{}
	}
	return weak.Make(p)
}

func dispatchMapsOf(actions *session.EventActions) taskDispatchMaps {
	if actions == nil {
		return taskDispatchMaps{}
	}
	return taskDispatchMaps{
		stateDelta:    mapHeaderWeakPtr(actions.StateDelta),
		artifactDelta: mapHeaderWeakPtr(actions.ArtifactDelta),
	}
}

func (d taskDispatchMaps) hasLiveMap() bool {
	return d.stateDelta.Value() != nil || d.artifactDelta.Value() != nil
}

func (d taskDispatchMaps) matches(other taskDispatchMaps) bool {
	if d.stateDelta.Value() != nil && d.stateDelta == other.stateDelta {
		return true
	}
	return d.artifactDelta.Value() != nil && d.artifactDelta == other.artifactDelta
}

type taskAttempt struct {
	callID      string
	kind        taskAttemptKind
	startedOnly bool
	recordedAt  time.Time
	seq         uint64
	dispatch    taskDispatchMaps
}

func isObservedEndingAttempt(kind taskAttemptKind) bool {
	return kind == taskAttemptCompleted || kind == taskAttemptFailedEnd
}

type taskMatchStatus uint8

const (
	taskMatchUnobserved taskMatchStatus = iota
	taskMatchFound
	taskMatchAmbiguous
)

func earlierTaskAttempt(aTime time.Time, aSeq uint64, bTime time.Time, bSeq uint64) bool {
	if !aTime.IsZero() && !bTime.IsZero() && !aTime.Equal(bTime) {
		return aTime.Before(bTime)
	}
	if aSeq != bSeq && (aSeq == 0) == (bSeq == 0) {
		return aSeq < bSeq
	}
	if aSeq != 0 && bSeq == 0 {
		return true
	}
	return false
}

// popTaskAgentSpan ends the span of a task-mode sub-agent that completed via
// a finish_task function response (which ADK does without running
// AfterAgentCallback), and marks its delegating agent activation as having
// delegated to a completed task. ok is false, and nothing changes, if the
// agent has no open span or does not run in task mode: an agent of another
// mode may have a tool of its own named finish_task, and ADK runs its
// AfterAgentCallback as usual. The mark lets popDelegatingAgentSpan retire
// the parent's prior activation safely on re-entry, without retiring a parent
// that is still running in an ordinary transfer-back chain. Only a direct
// LLM parent (nonLLM == false) whose configured sub-agents include the task
// sub-agent is marked; a workflow agent (nonLLM == true) runs its own
// AfterAgentCallback, and an ancestor LLM agent under which a resumed nested
// task was parented is still active in its own runChat loop.
//
// failed reports that the response also carries a classified tool result
// error, with which ADK ends the activation as well, but which fails its task.
// All ending call IDs on the event are evaluated together before any attempt is
// consumed (see resolveFinishedTaskLocked): single-call ties resolve to the
// earliest-recorded attempt so overlapping runs reusing call_0 all close
// cleanly, while disjoint multi-call matches leave spans open rather than
// guessing across conflicting call IDs. An unobserved call falls back to
// unobservedTaskLocked. OnEventCallback runs on the runner's goroutine with
// the runner's context, and the event does not show that the run that wrote it
// returned: a custom agent can forward the event of a sub-agent that it runs
// on another goroutine and wait for that goroutine later. So the span is
// retired without the activation's InvocationContext being read and without
// its run being taken to be over (see retireAgentLocked).
func (tm *traceManager) popTaskAgentSpan(ctx context.Context, id identity, callID string, failed bool, eventTime time.Time) (span rowSpan, ok bool) {
	var endingCallIDs []string
	if callID != "" {
		endingCallIDs = []string{callID}
	}
	return tm.popTaskAgentSpanForCalls(ctx, id, endingCallIDs, failed, eventTime)
}

func (tm *traceManager) popTaskAgentSpanForCalls(ctx context.Context, id identity, endingCallIDs []string, failed bool, eventTime time.Time) (span rowSpan, ok bool) {
	return tm.popTaskAgentSpanForEvent(ctx, id, endingCallIDs, "", taskDispatchMaps{}, failed, len(endingCallIDs) > 1, eventTime)
}

func (tm *traceManager) popTaskAgentSpanForEvent(ctx context.Context, id identity, endingCallIDs []string, leadingActionCallID string, evDispatch taskDispatchMaps, failed, mergedResponse bool, eventTime time.Time) (span rowSpan, ok bool) {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgentLocked(id, rootIn, true)
	i, status := inv.resolveFinishedTaskLocked(id, endingCallIDs, leadingActionCallID, evDispatch, failed, mergedResponse, eventTime)
	switch status {
	case taskMatchAmbiguous:
		return rowSpan{}, false
	case taskMatchUnobserved:
		fallbackCallID := ""
		for _, cid := range endingCallIDs {
			if cid != "" {
				fallbackCallID = cid
				break
			}
		}
		i = inv.unobservedTaskLocked(ctx, id, fallbackCallID, eventTime)
	}
	if i < 0 {
		return rowSpan{}, false
	}
	rec := inv.agentStack[i]
	if !rec.taskMode {
		return rowSpan{}, false
	}
	defer inv.retireLateSettledLocked()
	if rec.parent != nil && rec.parent.kind == spanKindAgent {
		inv.touchAgentActivityLocked(rec.parent, now)
	}
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
	inv.retireAgentLocked(i, false, noFrame)
	span = inv.rowSpan(rec, latencyOf(rec, now), ambient)
	span.timestamp = now
	return span, true
}

func (inv *invocationTrace) matchesTaskAgentLocked(rec *spanRecord, id identity) bool {
	return rec.kind == spanKindAgent && rec.taskMode &&
		rec.name == id.agentName && rec.branch == id.branch &&
		(id.path == "" || rec.path == "" || rec.path == id.path)
}

// finishedTaskLocked resolves and consumes a single ending finish_task callID
// on id, delegating to resolveFinishedTaskLocked. inv.mu must be held.
func (inv *invocationTrace) finishedTaskLocked(id identity, callID string, failed bool, eventTime time.Time) (int, taskMatchStatus) {
	if callID == "" {
		return -1, taskMatchUnobserved
	}
	return inv.resolveFinishedTaskLocked(id, []string{callID}, "", taskDispatchMaps{}, failed, false, eventTime)
}

// resolveFinishedTaskLocked evaluates all non-empty endingCallIDs of a
// finish_task completion event before consuming any attempt:
//
//  1. Leading-part EventActions map identity (weak references to StateDelta and
//     ArtifactDelta on the first FunctionResponse part) selects the exact
//     producing activation immediately when matched, and deprioritizes
//     disproved attempts when another candidate or open activation exists.
//  2. Timestamp plausibility (excluding activations started after eventTime, and
//     excluding attempts recorded after eventTime on single-response events
//     while keeping parallel merged-response attempts whose sibling part
//     stamped an earlier event timestamp) and O(N log N) suffix-maximum causal
//     supersession narrow candidates across all endingCallIDs.
//  3. Across multiple observed call IDs in one merged event, an activation must
//     account for every surviving observed call ID; disjoint matches across
//     different activations return taskMatchAmbiguous without consuming any
//     attempt. Same-set ties choose the highest tier (1: matching failed, 2:
//     opposite terminal flag, 3: pending) and then the earliest-recorded
//     attempt (recordedAt, seq, stack order) so overlapping call_0 runs all
//     close cleanly.
//
// inv.mu must be held.
func (inv *invocationTrace) resolveFinishedTaskLocked(id identity, endingCallIDs []string, leadingActionCallID string, evDispatch taskDispatchMaps, failed, mergedResponse bool, eventTime time.Time) (int, taskMatchStatus) {
	var callIDs []string
	for _, cid := range endingCallIDs {
		if cid != "" && !slices.Contains(callIDs, cid) {
			callIDs = append(callIDs, cid)
		}
	}
	if len(callIDs) == 0 {
		return -1, taskMatchUnobserved
	}

	wantKind := taskAttemptCompleted
	altEndKind := taskAttemptFailedEnd
	if failed {
		wantKind = taskAttemptFailedEnd
		altEndKind = taskAttemptCompleted
	}

	var matchingOpen []int
	hasPlausibleOpenWithoutEnding := false
	for i := 1; i < len(inv.agentStack); i++ {
		rec := inv.agentStack[i]
		if !inv.matchesTaskAgentLocked(rec, id) {
			continue
		}
		matchingOpen = append(matchingOpen, i)
		if !eventTime.IsZero() && !rec.eventClockStart.IsZero() && eventTime.Before(rec.eventClockStart) {
			continue
		}
		hasEnding := slices.ContainsFunc(rec.taskAttempts, func(att taskAttempt) bool {
			return isObservedEndingAttempt(att.kind)
		})
		if !hasEnding {
			hasPlausibleOpenWithoutEnding = true
		}
	}
	if len(matchingOpen) == 0 {
		return -1, taskMatchUnobserved
	}

	type tierChoice struct {
		ok              bool
		recordedAt      time.Time
		seq             uint64
		actionMatched   bool
		actionDisproved bool
	}
	betterAttempt := func(cur, cand tierChoice) bool {
		if !cur.ok {
			return true
		}
		if cand.actionMatched != cur.actionMatched {
			return cand.actionMatched
		}
		if cand.actionDisproved != cur.actionDisproved {
			return !cand.actionDisproved
		}
		return earlierTaskAttempt(cand.recordedAt, cand.seq, cur.recordedAt, cur.seq)
	}

	type taskCandidate struct {
		idx             int
		callID          string
		kind            taskAttemptKind
		tier            uint8
		recordedAt      time.Time
		seq             uint64
		actionMatched   bool
		actionDisproved bool
	}

	var candidates []taskCandidate
	for _, cid := range callIDs {
		checkAction := cid == leadingActionCallID && evDispatch.hasLiveMap()
		for _, i := range matchingOpen {
			rec := inv.agentStack[i]
			var wantC, altC, pendingC tierChoice
			for _, att := range rec.taskAttempts {
				if att.callID != cid {
					continue
				}
				matched := checkAction && att.dispatch.matches(evDispatch)
				disproved := checkAction && !matched && att.dispatch.hasLiveMap()
				cand := tierChoice{
					ok:              true,
					recordedAt:      att.recordedAt,
					seq:             att.seq,
					actionMatched:   matched,
					actionDisproved: disproved,
				}
				switch att.kind {
				case wantKind:
					if betterAttempt(wantC, cand) {
						wantC = cand
					}
				case altEndKind:
					if betterAttempt(altC, cand) {
						altC = cand
					}
				case taskAttemptPending:
					if betterAttempt(pendingC, cand) {
						pendingC = cand
					}
				}
			}
			// An exact dispatch-map match on a lower tier outranks a stale
			// non-matching attempt on the same activation; otherwise keep tier
			// precedence (wantKind -> altEndKind -> pending).
			switch {
			case wantC.ok && (wantC.actionMatched || (!altC.actionMatched && !pendingC.actionMatched)):
				candidates = append(candidates, taskCandidate{
					idx: i, callID: cid, kind: wantKind, tier: 1,
					recordedAt: wantC.recordedAt, seq: wantC.seq,
					actionMatched: wantC.actionMatched, actionDisproved: wantC.actionDisproved,
				})
			case altC.ok && (altC.actionMatched || !pendingC.actionMatched):
				candidates = append(candidates, taskCandidate{
					idx: i, callID: cid, kind: altEndKind, tier: 2,
					recordedAt: altC.recordedAt, seq: altC.seq,
					actionMatched: altC.actionMatched, actionDisproved: altC.actionDisproved,
				})
			case pendingC.ok:
				candidates = append(candidates, taskCandidate{
					idx: i, callID: cid, kind: taskAttemptPending, tier: 3,
					recordedAt: pendingC.recordedAt, seq: pendingC.seq,
					actionMatched: pendingC.actionMatched, actionDisproved: pendingC.actionDisproved,
				})
			}
		}
	}
	if len(candidates) == 0 {
		return -1, taskMatchUnobserved
	}

	consumeWinner := func(winnerIdx int, source []taskCandidate) (int, taskMatchStatus) {
		rec := inv.agentStack[winnerIdx]
		for _, cid := range callIDs {
			var wantConsumeKind taskAttemptKind
			foundKind := false
			for _, c := range source {
				if c.idx == winnerIdx && c.callID == cid {
					wantConsumeKind = c.kind
					foundKind = true
					break
				}
			}
			if !foundKind {
				continue
			}
			checkAction := cid == leadingActionCallID && evDispatch.hasLiveMap()
			bestJ := -1
			for j, att := range rec.taskAttempts {
				if att.callID != cid || att.kind != wantConsumeKind {
					continue
				}
				if checkAction && att.dispatch.matches(evDispatch) {
					bestJ = j
					break
				}
				if bestJ < 0 || earlierTaskAttempt(att.recordedAt, att.seq, rec.taskAttempts[bestJ].recordedAt, rec.taskAttempts[bestJ].seq) {
					bestJ = j
				}
			}
			if bestJ >= 0 {
				rec.taskAttempts = slices.Delete(rec.taskAttempts, bestJ, bestJ+1)
			}
		}
		return winnerIdx, taskMatchFound
	}

	// 1. Exact leading-part EventActions map match identifies the producing
	// activation directly, even on frozen clocks.
	var exactMatches []taskCandidate
	for _, c := range candidates {
		if c.actionMatched {
			exactMatches = append(exactMatches, c)
		}
	}
	if len(exactMatches) > 0 {
		best := exactMatches[0]
		for _, c := range exactMatches[1:] {
			if c.tier < best.tier ||
				(c.tier == best.tier && (earlierTaskAttempt(c.recordedAt, c.seq, best.recordedAt, best.seq) ||
					(c.recordedAt.Equal(best.recordedAt) && c.seq == best.seq && c.idx < best.idx))) {
				best = c
			}
		}
		return consumeWinner(best.idx, candidates)
	}

	// 2. Drop candidates whose recorded live dispatch maps conflict with the
	// event's leading-part maps when another candidate or an open activation
	// without a recorded ending attempt explains the event.
	hasDisproved := slices.ContainsFunc(candidates, func(c taskCandidate) bool { return c.actionDisproved })
	if hasDisproved {
		nonDisproved := make([]taskCandidate, 0, len(candidates))
		for _, c := range candidates {
			if !c.actionDisproved {
				nonDisproved = append(nonDisproved, c)
			}
		}
		if len(nonDisproved) > 0 {
			candidates = nonDisproved
		} else if hasPlausibleOpenWithoutEnding {
			return -1, taskMatchUnobserved
		}
	}

	// 3. Timestamp plausibility filter. For merged multi-part FunctionResponse
	// events, ADK stamps ev.Timestamp from events[0], which can precede a
	// concurrent sibling tool's AfterToolCallback recordedAt, so only
	// rec.eventClockStart bounds merged events.
	if !eventTime.IsZero() && (len(candidates) >= 2 || hasPlausibleOpenWithoutEnding) {
		plausible := make([]taskCandidate, 0, len(candidates))
		for _, c := range candidates {
			rec := inv.agentStack[c.idx]
			if !rec.eventClockStart.IsZero() && eventTime.Before(rec.eventClockStart) {
				continue
			}
			if !mergedResponse && !c.recordedAt.IsZero() && eventTime.Before(c.recordedAt) {
				continue
			}
			plausible = append(plausible, c)
		}
		if len(plausible) > 0 {
			candidates = plausible
		} else if len(candidates) == 1 && hasPlausibleOpenWithoutEnding {
			return -1, taskMatchUnobserved
		}
	}

	// 4. O(N log N) suffix-maximum causal supersession across all surviving
	// candidates: a candidate cOld with a non-zero recordedAt is superseded when
	// another activation idx != cOld.idx has a plausible strictly-later
	// recordedAt and either started after cOld (recNew.seq > cOld.seq) or opened
	// its event clock after cOld was recorded (cOld.recordedAt < recNew.eventClockStart).
	if len(candidates) >= 2 && !eventTime.IsZero() {
		order := make([]int, len(candidates))
		for k := range order {
			order[k] = k
		}
		slices.SortFunc(order, func(a, b int) int {
			ta, tb := candidates[a].recordedAt, candidates[b].recordedAt
			switch {
			case ta.Equal(tb):
				return cmp.Compare(a, b)
			case ta.After(tb):
				return -1
			default:
				return 1
			}
		})

		type idxSeq struct {
			idx int
			seq uint64
		}
		type idxClk struct {
			idx int
			clk time.Time
		}
		var topSeq1, topSeq2 idxSeq
		var topClk1, topClk2 idxClk
		addWitness := func(c taskCandidate) {
			if c.recordedAt.IsZero() {
				return
			}
			recNew := inv.agentStack[c.idx]
			if mergedResponse {
				if !recNew.eventClockStart.IsZero() && eventTime.Before(recNew.eventClockStart) {
					return
				}
			} else if eventTime.Before(c.recordedAt) {
				return
			}
			if recNew.seq > 0 {
				if topSeq1.idx == c.idx {
					topSeq1.seq = max(topSeq1.seq, recNew.seq)
				} else if recNew.seq > topSeq1.seq {
					topSeq2 = topSeq1
					topSeq1 = idxSeq{idx: c.idx, seq: recNew.seq}
				} else if topSeq2.idx == c.idx {
					topSeq2.seq = max(topSeq2.seq, recNew.seq)
				} else if recNew.seq > topSeq2.seq {
					topSeq2 = idxSeq{idx: c.idx, seq: recNew.seq}
				}
			}
			if !recNew.eventClockStart.IsZero() {
				if topClk1.idx == c.idx {
					if recNew.eventClockStart.After(topClk1.clk) {
						topClk1.clk = recNew.eventClockStart
					}
				} else if topClk1.clk.IsZero() || recNew.eventClockStart.After(topClk1.clk) {
					topClk2 = topClk1
					topClk1 = idxClk{idx: c.idx, clk: recNew.eventClockStart}
				} else if topClk2.idx == c.idx {
					if recNew.eventClockStart.After(topClk2.clk) {
						topClk2.clk = recNew.eventClockStart
					}
				} else if topClk2.clk.IsZero() || recNew.eventClockStart.After(topClk2.clk) {
					topClk2 = idxClk{idx: c.idx, clk: recNew.eventClockStart}
				}
			}
		}

		superseded := make([]bool, len(candidates))
		for start := 0; start < len(order); {
			batchTime := candidates[order[start]].recordedAt
			end := start + 1
			for end < len(order) && candidates[order[end]].recordedAt.Equal(batchTime) {
				end++
			}
			if !batchTime.IsZero() {
				for _, pos := range order[start:end] {
					cOld := candidates[pos]
					maxSeq := topSeq1.seq
					if topSeq1.idx == cOld.idx {
						maxSeq = topSeq2.seq
					}
					maxClk := topClk1.clk
					if topClk1.idx == cOld.idx {
						maxClk = topClk2.clk
					}
					if (maxSeq > 0 && cOld.seq < maxSeq) || (!maxClk.IsZero() && cOld.recordedAt.Before(maxClk)) {
						superseded[pos] = true
					}
				}
			}
			for _, pos := range order[start:end] {
				addWitness(candidates[pos])
			}
			start = end
		}

		unsuperseded := make([]taskCandidate, 0, len(candidates))
		for k, c := range candidates {
			if !superseded[k] {
				unsuperseded = append(unsuperseded, c)
			}
		}
		if len(unsuperseded) > 0 {
			candidates = unsuperseded
		}
	}

	// 5. Aggregate across observed call IDs in the event. Every call ID in one
	// session.Event was emitted by a single activation; if surviving observed
	// call IDs split across disjoint activations so no single activation covers
	// all of them, leave spans open rather than guessing.
	observedCallIDs := make(map[string]struct{}, len(callIDs))
	type actSummary struct {
		idx        int
		matchCount int
		bestTier   uint8
		recordedAt time.Time
		seq        uint64
	}
	var summaries []actSummary
	summaryByIdx := make(map[int]int, len(candidates))
	for _, c := range candidates {
		observedCallIDs[c.callID] = struct{}{}
		pos, ok := summaryByIdx[c.idx]
		if !ok {
			summaryByIdx[c.idx] = len(summaries)
			summaries = append(summaries, actSummary{
				idx:        c.idx,
				matchCount: 1,
				bestTier:   c.tier,
				recordedAt: c.recordedAt,
				seq:        c.seq,
			})
			continue
		}
		s := &summaries[pos]
		s.matchCount++
		if c.tier < s.bestTier || (c.tier == s.bestTier && earlierTaskAttempt(c.recordedAt, c.seq, s.recordedAt, s.seq)) {
			s.bestTier = c.tier
			s.recordedAt = c.recordedAt
			s.seq = c.seq
		}
	}

	wantMatches := len(observedCallIDs)
	var fullMatches []actSummary
	for _, s := range summaries {
		if s.matchCount == wantMatches {
			fullMatches = append(fullMatches, s)
		}
	}
	if len(fullMatches) == 0 {
		return -1, taskMatchAmbiguous
	}

	best := fullMatches[0]
	for _, s := range fullMatches[1:] {
		if s.bestTier < best.bestTier {
			best = s
			continue
		}
		if s.bestTier > best.bestTier {
			continue
		}
		if earlierTaskAttempt(s.recordedAt, s.seq, best.recordedAt, best.seq) ||
			(s.recordedAt.Equal(best.recordedAt) && s.seq == best.seq && s.idx < best.idx) {
			best = s
		}
	}
	return consumeWinner(best.idx, candidates)
}

// unobservedTaskLocked resolves the open task-mode activation of id for a
// finish_task completion event whose callID was not observed by any open
// activation's tool callbacks (for example, when an earlier plugin answered
// BeforeToolCallback and AfterToolCallback, or in synthetic event-only tests
// without call IDs). Activations that already hold an unconsumed observed
// ending attempt for another call are skipped when another candidate exists,
// eventTime excludes activations started after the event was created, and ties
// fall back to ctx's activation (when in pool) or the newest plausible
// matching activation as on main. inv.mu must be held.
func (inv *invocationTrace) unobservedTaskLocked(ctx context.Context, id identity, _ string, eventTime time.Time) int {
	var withoutEnding, allMatching []int
	for i := 1; i < len(inv.agentStack); i++ {
		rec := inv.agentStack[i]
		if !inv.matchesTaskAgentLocked(rec, id) {
			continue
		}
		allMatching = append(allMatching, i)
		hasEnding := slices.ContainsFunc(rec.taskAttempts, func(att taskAttempt) bool {
			return isObservedEndingAttempt(att.kind)
		})
		if !hasEnding {
			withoutEnding = append(withoutEnding, i)
		}
	}
	pool := withoutEnding
	if len(pool) == 0 {
		pool = allMatching
	}
	if len(pool) == 0 {
		return -1
	}
	if len(pool) >= 2 && !eventTime.IsZero() {
		plausible := make([]int, 0, len(pool))
		for _, idx := range pool {
			rec := inv.agentStack[idx]
			if rec.eventClockStart.IsZero() || !eventTime.Before(rec.eventClockStart) {
				plausible = append(plausible, idx)
			}
		}
		if len(plausible) > 0 {
			pool = plausible
		}
	}
	if len(pool) == 1 {
		return pool[0]
	}
	if ctxIdx := inv.findAgentForContext(ctx, id.path, id.agentName, id.branch); ctxIdx > 0 && slices.Contains(pool, ctxIdx) {
		return ctxIdx
	}
	return pool[len(pool)-1]
}

// noteTaskStart records that the activation of ctx and id started a
// finish_task call with callID, so the pending attempt is still attributed to
// that activation if an earlier plugin's AfterToolCallback short-circuits
// before noteTaskAttempt runs.
func (tm *traceManager) noteTaskStart(ctx context.Context, id identity, callID string) {
	if callID == "" {
		return
	}
	recordedAt := platform.Now(ctx)
	dispatch := dispatchMapsOf(toolActionsOf(ctx))
	inv := tm.invocation(ctx, id)
	for !inv.lockLive() {
		inv = tm.invocation(ctx, id)
	}
	defer inv.mu.Unlock()
	if i := inv.findAgentForContext(ctx, id.path, id.agentName, id.branch); i > 0 {
		rec := inv.agentStack[i]
		for j := range rec.taskAttempts {
			if rec.taskAttempts[j].callID == callID && rec.taskAttempts[j].startedOnly {
				if dispatch.hasLiveMap() {
					rec.taskAttempts[j].dispatch = dispatch
				}
				return
			}
		}
		rec.taskAttempts = append(rec.taskAttempts, taskAttempt{
			callID:      callID,
			kind:        taskAttemptPending,
			startedOnly: true,
			recordedAt:  recordedAt,
			seq:         inv.lastSeq,
			dispatch:    dispatch,
		})
	}
}

// noteTaskAttempt records the outcome of a finish_task call on the agent
// activation that ctx and id identify, distinguishing calls whose result
// already ended the task in afterTool (ends), with a classified tool result
// error or not (failed), from attempts that did not end it, which a later
// AfterToolCallback may still promote before the FunctionResponse event is
// emitted (see popTaskAgentSpan and retireFailedTaskAttempts).
func (tm *traceManager) noteTaskAttempt(ctx context.Context, id identity, callID string, ends, failed bool) {
	if callID == "" {
		return
	}
	recordedAt := platform.Now(ctx)
	dispatch := dispatchMapsOf(toolActionsOf(ctx))
	kind := taskAttemptPending
	if ends {
		if failed {
			kind = taskAttemptFailedEnd
		} else {
			kind = taskAttemptCompleted
		}
	}
	inv := tm.invocation(ctx, id)
	for !inv.lockLive() {
		inv = tm.invocation(ctx, id)
	}
	defer inv.mu.Unlock()
	if i := inv.findAgentForContext(ctx, id.path, id.agentName, id.branch); i > 0 {
		rec := inv.agentStack[i]
		for j := range rec.taskAttempts {
			if rec.taskAttempts[j].callID == callID && rec.taskAttempts[j].startedOnly {
				rec.taskAttempts[j].kind = kind
				rec.taskAttempts[j].startedOnly = false
				rec.taskAttempts[j].recordedAt = recordedAt
				rec.taskAttempts[j].seq = inv.lastSeq
				if dispatch.hasLiveMap() {
					rec.taskAttempts[j].dispatch = dispatch
				}
				return
			}
		}
		rec.taskAttempts = append(rec.taskAttempts, taskAttempt{
			callID:     callID,
			kind:       kind,
			recordedAt: recordedAt,
			seq:        inv.lastSeq,
			dispatch:   dispatch,
		})
	}
}

// retireFailedTaskAttempts removes pending (non-ending) finish_task attempts
// for failedCalls once their FunctionResponse event arrives without ending the
// task. Observed ending attempts (taskAttemptCompleted and
// taskAttemptFailedEnd) are never deleted for failedCalls, so an unobserved
// failure on a newer activation cannot wipe out another open activation's
// observed completion; overruledEndingCalls (sentinel-bearing responses behind
// a failed first finish_task response in the same merged event) are retired on
// the activation where failedCalls matched, or when no pending failed ID
// matched (for example when the first failed response had an empty call ID), on
// a plausible matching activation holding an observed ending attempt for that ID.
func (tm *traceManager) retireFailedTaskAttempts(ctx context.Context, id identity, failedCalls, overruledEndingCalls []string) {
	tm.retireFailedTaskAttemptsForEvent(ctx, id, failedCalls, overruledEndingCalls, "", taskDispatchMaps{}, len(failedCalls)+len(overruledEndingCalls) > 1, time.Time{})
}

func (tm *traceManager) retireFailedTaskAttemptsForEvent(ctx context.Context, id identity, failedCalls, overruledEndingCalls []string, leadingActionCallID string, evDispatch taskDispatchMaps, mergedResponse bool, eventTime time.Time) {
	if len(failedCalls) == 0 && len(overruledEndingCalls) == 0 {
		return
	}
	remaining := make(map[string]int, len(failedCalls))
	total := 0
	for _, call := range failedCalls {
		if call != "" {
			remaining[call]++
			total++
		}
	}
	overruledTotal := 0
	var overruled map[string]int
	if len(overruledEndingCalls) > 0 {
		overruled = make(map[string]int, len(overruledEndingCalls))
		for _, call := range overruledEndingCalls {
			if call != "" {
				overruled[call]++
				overruledTotal++
			}
		}
	}
	if total == 0 && overruledTotal == 0 {
		return
	}
	inv := tm.invocation(ctx, id)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.retireFailedTaskAttemptsAtLocked(ctx, id, remaining, total, overruled, overruledTotal, leadingActionCallID, evDispatch, mergedResponse, eventTime)
}

func (inv *invocationTrace) retireFailedTaskAttemptsLocked(ctx context.Context, id identity, remaining map[string]int, total int, overruled map[string]int, overruledTotal int) {
	inv.retireFailedTaskAttemptsAtLocked(ctx, id, remaining, total, overruled, overruledTotal, "", taskDispatchMaps{}, total+overruledTotal > 1, time.Time{})
}

// retireFailedTaskAttemptsAtLocked compacts pending finish_task attempts
// matching remaining across plausible open activations of id, preferring an
// exact leading-part EventActions map match first, excluding activations and
// single-call attempts recorded after eventTime so a delayed older failure
// cannot erase a newer activation's pending attempt, and retiring
// overruledEndingCalls on the activation where remaining matched or on a
// plausible matching activation when remaining was empty. inv.mu must be held.
func (inv *invocationTrace) retireFailedTaskAttemptsAtLocked(ctx context.Context, id identity, remaining map[string]int, total int, overruled map[string]int, overruledTotal int, leadingActionCallID string, evDispatch taskDispatchMaps, mergedResponse bool, eventTime time.Time) {
	matchesAgent := func(rec *spanRecord) bool {
		if rec.kind != spanKindAgent || rec.name != id.agentName || rec.branch != id.branch ||
			(id.path != "" && rec.path != "" && rec.path != id.path) || len(rec.taskAttempts) == 0 {
			return false
		}
		if !eventTime.IsZero() && !rec.eventClockStart.IsZero() && eventTime.Before(rec.eventClockStart) {
			return false
		}
		return true
	}
	plausibleAttempt := func(att taskAttempt, skipDisproved bool) bool {
		if !eventTime.IsZero() && !mergedResponse && !att.recordedAt.IsZero() && eventTime.Before(att.recordedAt) {
			return false
		}
		if skipDisproved && att.callID == leadingActionCallID && evDispatch.hasLiveMap() && att.dispatch.hasLiveMap() && !att.dispatch.matches(evDispatch) {
			return false
		}
		return true
	}
	retireOnRec := func(rec *spanRecord, skipDisproved bool) {
		matchedFailedOnRec := false
		for _, att := range rec.taskAttempts {
			if att.kind == taskAttemptPending && remaining[att.callID] > 0 && plausibleAttempt(att, skipDisproved) {
				matchedFailedOnRec = true
				break
			}
		}
		if !matchedFailedOnRec {
			return
		}
		write := 0
		for _, att := range rec.taskAttempts {
			if total > 0 && att.kind == taskAttemptPending && remaining[att.callID] > 0 && plausibleAttempt(att, skipDisproved) {
				remaining[att.callID]--
				total--
				continue
			}
			if overruledTotal > 0 && overruled[att.callID] > 0 && plausibleAttempt(att, false) {
				overruled[att.callID]--
				overruledTotal--
				continue
			}
			rec.taskAttempts[write] = att
			write++
		}
		clear(rec.taskAttempts[write:])
		rec.taskAttempts = rec.taskAttempts[:write]
	}

	// Pass 1: exact leading-part EventActions map match identifies the
	// activation that produced the failed response even on equal timestamps.
	if total > 0 && leadingActionCallID != "" && remaining[leadingActionCallID] > 0 && evDispatch.hasLiveMap() {
		for i := len(inv.agentStack) - 1; i > 0 && total > 0; i-- {
			rec := inv.agentStack[i]
			if !matchesAgent(rec) {
				continue
			}
			hasExact := slices.ContainsFunc(rec.taskAttempts, func(att taskAttempt) bool {
				return att.kind == taskAttemptPending && att.callID == leadingActionCallID && att.dispatch.matches(evDispatch)
			})
			if hasExact {
				retireOnRec(rec, true)
				break
			}
		}
	}

	// Pass 2: plausible non-disproved pending attempts, then fallback if needed.
	for _, skipDisproved := range []bool{true, false} {
		for i := len(inv.agentStack) - 1; i > 0 && total > 0; i-- {
			rec := inv.agentStack[i]
			if !matchesAgent(rec) {
				continue
			}
			retireOnRec(rec, skipDisproved)
		}
		if total == 0 || !evDispatch.hasLiveMap() {
			break
		}
	}

	if overruledTotal <= 0 {
		return
	}
	retireOverruledOn := func(rec *spanRecord) {
		hasMatch := false
		for _, att := range rec.taskAttempts {
			if isObservedEndingAttempt(att.kind) && overruled[att.callID] > 0 && plausibleAttempt(att, false) {
				hasMatch = true
				break
			}
		}
		if !hasMatch {
			return
		}
		write := 0
		for _, att := range rec.taskAttempts {
			if overruledTotal > 0 && isObservedEndingAttempt(att.kind) && overruled[att.callID] > 0 && plausibleAttempt(att, false) {
				overruled[att.callID]--
				overruledTotal--
				continue
			}
			rec.taskAttempts[write] = att
			write++
		}
		clear(rec.taskAttempts[write:])
		rec.taskAttempts = rec.taskAttempts[:write]
	}
	ctxIdx := inv.findAgentForContext(ctx, id.path, id.agentName, id.branch)
	if ctxIdx > 0 && activationKey(ctx) != nil && matchesAgent(inv.agentStack[ctxIdx]) {
		retireOverruledOn(inv.agentStack[ctxIdx])
	}
	// When no concrete activation context matched, prefer the earliest-recorded
	// plausible overruled attempt so a failed merged batch on an older open run
	// retires its own overruled ending call before a concurrent newer run's.
	for i := 1; i < len(inv.agentStack) && overruledTotal > 0; i++ {
		if i == ctxIdx && activationKey(ctx) != nil {
			continue
		}
		if rec := inv.agentStack[i]; matchesAgent(rec) {
			retireOverruledOn(rec)
		}
	}
}

// retireAgentLocked removes agentStack[i] and any unclosed descendant spans
// whose parentSpanID lineage reaches agentStack[i].spanID, except failed
// activations and the open spans they run under. An activation failed if it
// owns an open model or tool call, has a non-collateral unrecovered error
// recorded, or never responded (see spanRecord.responded): a sub-agent that a
// BeforeAgentCallback answered marks its own InvocationContext as ended, so it
// counts as responded even when an enclosing workflow.ParallelWorker or custom
// parent consumes its event before OnEventCallback sees it. If a tool call or
// an agent that is still running may have run the sub-agent instead (see
// launcherOpenLocked), that state is read only once they end (see
// settleFinishedLocked), and the sub-agent stays open until then. A sub-agent
// that a custom parent stopped iterating before it produced content or ended
// its own InvocationContext, or whose silent failure a custom parent swallowed
// without emitting a settling final event on the same branch and node path
// while the ancestor spans between them remain open, has responded() == false
// and stays open (see popUnclosedAgentSpans). Any openFailureCause on the
// retired span or its retired descendants is preserved on the nearest live
// ancestor in agentStack (retiredCauses), linked to the retired span, so
// draining a workflow or parallel root reports the same terminal error as
// stopping at the error. done reports that agentStack[i]'s run is known to be
// over, as popAgentSpan knows after the activation's own AfterAgentCallback
// and endToolRunsLocked knows for an activation that ran on the goroutine of
// a tool call that returned (see settleEndedInSubtreeLocked). Without it, as
// when an event retires the span — a finish_task function response
// (popTaskAgentSpan) or the new activation of an agent that delegated to a
// completed task (popDelegatingAgentSpan) — or for an activation that a tool
// started on a goroutine that the tool did not wait for, its InvocationContext
// is not read and its run is not taken to be over. since bounds the
// activations under it that are taken to be over because they ran on the
// current goroutine (see canSettleInSubtreeLocked).
func (inv *invocationTrace) retireAgentLocked(i int, done bool, since uint64) {
	rec := inv.agentStack[i]
	subtree := inv.subtreeLocked(i)
	// The activations in the subtree whose runs are known to be over are
	// settled, unless a tool call or an agent that is still open ran them
	// instead of the activation that they run under: an agent ends after the
	// runs of the sub-agents that it ran on its own goroutine, and ADK's
	// built-in workflow and parallel agents after those that they started on
	// goroutines that they wait for (see spanRecord.joins).
	inv.settleEndedInSubtreeLocked(i, done, since)
	// The failed activations and the open spans they run under are attributed
	// while the spans are still open.
	failed := inv.openCallOwnersLocked(nil)
	for j, d := range inv.agentStack {
		unresponded := (d.runDone || inv.launcherOpenLocked(d)) && !d.responded()
		if subtree[d.spanID] && (failed[d.spanID] || unresponded) {
			maps.Copy(failed, inv.lineageLocked(j))
		}
	}
	for _, e := range inv.unrecoveredErrors {
		if e.collateral {
			continue
		}
		if j := inv.activationOfLocked(e); j > 0 {
			maps.Copy(failed, inv.lineageLocked(j))
		}
	}
	ownCtxCause := rec.contextFailureCause()
	inheritedCause := rec.firstRetiredCause()
	idx := i
	inv.removeAgentLocked(i)
	// Also retire any unclosed descendant spans in rec's subtree (for example,
	// a sub-agent short-circuited in BeforeAgentCallback that never ran
	// AfterAgentCallback, including one whose direct parent already retired
	// earlier while launcherOpenLocked held it open), except the failed ones
	// and what runs under them.
	var descendantOpenCause error
	var retired []*spanRecord
	for idx < len(inv.agentStack) {
		d := inv.agentStack[idx]
		canRetire := subtree[d.spanID] && !failed[d.spanID]
		for p := d.parent; canRetire && p != nil && p != rec && p.kind == spanKindAgent; p = p.parent {
			if failed[p.spanID] {
				canRetire = false
			}
		}
		if canRetire {
			drainedParent := d.parent != nil && d.parent != rec
			for p := d.parent; drainedParent && p != nil && p != rec && p.kind == spanKindAgent; p = p.parent {
				if subtree[p.spanID] {
					drainedParent = false
				}
			}
			if drainedParent {
				inheritedCause = cmp.Or(inheritedCause, d.openFailureCause())
			} else if descendantOpenCause == nil {
				descendantOpenCause = d.openFailureCause()
			}
			if inv.isLate {
				retired = append(retired, d)
			}
			inv.removeAgentLocked(idx)
			continue
		}
		idx++
	}
	if cause := cmp.Or(ownCtxCause, inheritedCause, descendantOpenCause); cause != nil {
		swallowable := !rec.propagatesChildError && ownCtxCause == nil && inheritedCause != nil && descendantOpenCause == nil
		for anc := rec.parent; anc != nil; anc = anc.parent {
			if slices.Contains(inv.agentStack, anc) {
				// The cause stays linked to rec, so that recovery content from
				// a custom agent's own AfterAgentCallback, which ADK runs
				// after this plugin's, clears it when rec drained an inherited
				// failure without its own context failing (see
				// completeResponse).
				anc.retiredCauses = append(anc.retiredCauses, retiredCause{
					err:               cause,
					name:              rec.name,
					branch:            rec.branch,
					path:              rec.path,
					swallowableBySelf: swallowable,
				})
				break
			}
		}
	}
	if inv.isLate {
		// Nothing can end a retired late activation anymore, so once its
		// cause moved up, its record keeps no callback context except for
		// canceled and parent while an open span or call still reaches it (see
		// dropRetiredLateContextsLocked). When the last thing reaching rec is
		// an open tool or model call, or a span drained by a response event
		// (see retireDrainedDescendantsLocked), nothing revisits rec to clear
		// canceled and parent; once nothing reachable still holds rec via
		// parent, the record is unreachable (garbage), so those leftover fields
		// do not matter.
		retired = append(retired, rec)
		inv.dropRetiredLateContextsLocked(retired...)
	}
	inv.syncOpenAgentsLocked()
}

// subtreeLocked returns the span IDs of agentStack[i] and of the open agent
// spans that run under it, following spanRecord.parent even across an
// intermediate parent span that already retired while launcherOpenLocked kept
// its descendant open.
func (inv *invocationTrace) subtreeLocked(i int) map[string]bool {
	subtree := map[string]bool{inv.agentStack[i].spanID: true}
	for _, rec := range inv.agentStack[i+1:] {
		for p := rec.parent; p != nil && p.kind == spanKindAgent; p = p.parent {
			if subtree[p.spanID] {
				subtree[rec.spanID] = true
				break
			}
		}
	}
	return subtree
}

// failuresLocked returns the activations of the subtree of agentStack[i] (see
// subtreeLocked) that failed, together with those they run under in the
// subtree, each with its error, keyed by span ID. The result is empty if
// nothing failed. An activation failed if it never responded, owns an open
// model or tool call (see openCalls below), has a non-collateral unrecovered
// error recorded, or (when agentStack[i]'s parent context itself did not fail)
// carries a retired cause (see spanRecord.retiredCauses) or subtree failure
// cause. An activation that did not fail is not in the result, even if a
// sibling of it did: it may have ended without an AfterAgentCallback, as an
// agent that its BeforeAgentCallback answered does.
// The error of an activation is the one recorded for it, or else the first one
// recorded or propagated in the subtree, or else a synthetic one. An activation
// whose run is not known to be over (see spanRecord.runDone), such as one that
// a custom agent or a tool started on a goroutine that it did not wait for, is
// not known to have failed by not responding, as its InvocationContext cannot
// be read yet (see settleEndedInSubtreeLocked), and neither is one under
// agentStack[i] that a tool call or an agent that is still open may have run
// instead of the one that it runs under (see launcherOpenLocked). The
// activations in over, which may be nil, failed by not responding too: a
// report shows that their runs ended, although their InvocationContext was
// not read, so only what was recorded for them shows a response (see
// reportedOverLocked).
//
// With openCalls set, an activation that owns an open model or tool call
// failed. Without it, only one whose own run is known to be over or that is
// in over did: a node can write the metadata of a completion event while its
// run goes on (see spanRecord.nodeRunEnded), so a call that is still open
// under such a report may yet end as usual, and does not show a failure on
// its own (see laterAttempt in pushAgentSpan).
//
// With stopped set, the caller stopped reading on a partial event, which stops
// the workflow that the run goes through and any parallel agent, and they
// cancel the work that still runs. Work that ended with that cancellation did
// not fail, as in popUnclosedAgentSpans (see spanRecord.canceledByStop): a call
// that they canceled does not make its agent fail, nor does a context.Canceled
// error that such a call recorded, nor does an activation that they canceled
// before it responded.
func (inv *invocationTrace) failuresLocked(i int, stopped bool, over map[string]bool, openCalls bool) map[string]error {
	subtree := inv.subtreeLocked(i)
	var ignore func(*spanRecord) bool
	if stopped {
		ignore = (*spanRecord).canceledByStop
	}
	owners := inv.openCallOwnersLocked(ignore)
	errs := make(map[string]error)
	var firstErr, collateralCause error
	for _, k := range inv.sortedErrorKeysLocked() {
		e := inv.unrecoveredErrors[k]
		if !subtree[e.spanID] || (stopped && e.canceled && errors.Is(e.err, context.Canceled)) {
			continue
		}
		if e.collateral {
			collateralCause = cmp.Or(collateralCause, e.cause)
			continue
		}
		if errs[e.spanID] == nil {
			errs[e.spanID] = e.err
		}
		if firstErr == nil {
			firstErr = e.err
		}
	}
	parentAlive := inv.agentStack[i].parent == nil || inv.agentStack[i].parent.contextFailureCause() == nil
	failed := make(map[string]bool)
	var subtreeCause error
	for j := i; j < len(inv.agentStack); j++ {
		rec := inv.agentStack[j]
		if !subtree[rec.spanID] {
			continue
		}
		if cause := rec.firstRetiredCause(); parentAlive && cause != nil && errs[rec.spanID] == nil {
			errs[rec.spanID] = cause
		}
		subtreeCause = cmp.Or(subtreeCause, rec.openFailureCause())
		unresponded := (rec.runDone || over[rec.spanID]) && !rec.responded() && !(stopped && rec.canceledByStop()) &&
			!(j != i && rec.ended != nil && inv.launcherOpenLocked(rec))
		openCall := owners[rec.spanID] && (openCalls || rec.runDone || over[rec.spanID])
		if errs[rec.spanID] != nil || openCall || unresponded {
			maps.Copy(failed, inv.lineageLocked(j))
		}
	}
	firstErr = cmp.Or(firstErr, subtreeCause, collateralCause)
	if parentAlive && len(failed) == 0 && firstErr != nil {
		failed[inv.agentStack[i].spanID] = true
	}
	// An activation is created after the ones that it runs under, so the
	// lineage of a failed one leaves the subtree only above agentStack[i].
	failures := make(map[string]error, len(failed))
	for _, rec := range inv.agentStack[i:] {
		if !failed[rec.spanID] {
			continue
		}
		err := cmp.Or(errs[rec.spanID], firstErr)
		if err == nil {
			err = fmt.Errorf("agent %q execution failed before completion", rec.name)
		}
		failures[rec.spanID] = err
	}
	return failures
}

// reportedOverLocked returns the span IDs of agentStack[i], the agent of an
// earlier attempt of a workflow node whose end a node completion event
// reported (see spanRecord.nodeRunEnded), and of the open activations under it
// whose runs end with it: those whose runs are known to be over (see
// spanRecord.runDone) stay out, as failuresLocked reads them anyway, and each
// other one is in if it has no activation or goroutine information, as in
// direct traceManager tests, or if each agent between it and agentStack[i]
// waits for the run of the one below it (see spanRecord.joins), up to one that
// is over, counting the report of a ParallelAgent's node run as the end of the
// branches that it waited for. A node can write the metadata of a completion
// event on an event of its own while its run goes on, so the result decides
// which of these runs failed by not responding, or by a model or tool call
// that is still open (see failuresLocked), only once something else shows
// that the attempt failed or is over: a failure recorded for it (see
// laterAttempt in pushAgentSpan) or the end of the workflow agent whose
// scheduler ran it (see endRestartedNodeRuns). It reads no InvocationContext
// and records nothing.
func (inv *invocationTrace) reportedOverLocked(i int) map[string]bool {
	subtree := inv.subtreeLocked(i)
	over := map[string]bool{inv.agentStack[i].spanID: true}
	for _, d := range inv.agentStack[i+1:] {
		if !subtree[d.spanID] || d.runDone {
			continue
		}
		if d.activation == nil || d.goroutineID == 0 {
			over[d.spanID] = true
			continue
		}
		for c, p := d, d.parent; p != nil && p.kind == spanKindAgent && p.joins(c, true); c, p = p, p.parent {
			if p.runDone || over[p.spanID] {
				over[d.spanID] = true
				break
			}
		}
	}
	return over
}

// purgeActivationLocked removes agentStack[i] and the open spans that run
// under it, with their model and tool calls and their unrecovered errors. It
// is for a failure that ADK handled without yielding it: the failed
// activations never get an AfterAgentCallback, and their failure must not fail
// the invocation. It returns the failed activations that were removed, those
// in failures (see failuresLocked), each with its error, so the caller can
// write their AGENT_ERROR rows immediately; the others end without a row, as
// retireAgentLocked ends them.
func (inv *invocationTrace) purgeActivationLocked(i int, failures map[string]error, now time.Time, ambient string) []unclosedAgentSpan {
	purged := inv.subtreeLocked(i)
	var purgedSpans []unclosedAgentSpan
	for j := len(inv.agentStack) - 1; j >= i; j-- {
		rec := inv.agentStack[j]
		err := failures[rec.spanID]
		if err == nil {
			continue
		}
		end := now
		if !rec.lastActivity.IsZero() {
			end = rec.lastActivity
		}
		purgedSpans = append(purgedSpans, unclosedAgentSpan{
			span:      inv.rowSpan(rec, latencyOf(rec, end), ambient),
			agentName: rec.name,
			branch:    rec.branch,
			path:      rec.path,
			err:       err,
			timestamp: end,
		})
	}
	// In the late state, nothing can end a purged activation anymore, so
	// once it leaves agentStack, its record keeps no callback context except
	// for canceled and parent while an open span or call still reaches it (see
	// dropRetiredLateContextsLocked). The purge removes every open span that
	// runs under rec, but not an open model or tool call owned by an activation
	// that retired earlier under rec: the call cleanup below checks only a
	// call's direct parent span ID. When such a call is the last thing reaching
	// rec, nothing revisits rec to clear canceled and parent; once nothing
	// reachable still holds rec via parent, the record is unreachable
	// (garbage), so those leftover fields do not matter. Only the removed
	// records qualify: the span that agentStack[i] runs under may still be
	// open.
	var latePurged []*spanRecord
	if inv.isLate {
		for _, rec := range inv.agentStack[i:] {
			if purged[rec.spanID] {
				latePurged = append(latePurged, rec)
			}
		}
	}
	inv.removeAgentsLocked(func(rec *spanRecord) bool { return purged[rec.spanID] })
	inv.syncOpenAgentsLocked()
	for _, calls := range []map[string]*spanRecord{inv.llmSpans, inv.completedModels, inv.toolSpans, inv.completedTools} {
		maps.DeleteFunc(calls, func(_ string, rec *spanRecord) bool { return purged[rec.parentSpanID] })
	}
	maps.DeleteFunc(inv.unrecoveredErrors, func(_ string, e recordedError) bool { return purged[e.spanID] })
	if purged[inv.lastPartialSpanID] {
		// Purging a handled activation is not observing that the consumer
		// read a non-partial session event past the partial one. Reset only
		// the per-span partial state, keeping the session-level state so
		// that consumer-stop suppression survives the purge.
		inv.lastEventPartial = false
		inv.lastPartialSpanID = ""
	}
	inv.syncOpenCallsLocked()
	if len(latePurged) > 0 {
		inv.dropRetiredLateContextsLocked(latePurged...)
	}
	return purgedSpans
}

// endToolRunsLocked ends the agent activations that the tool calls of the
// agent of id ran and that are still open, once that agent has no tool call
// left open. An agent that runs as a tool, such as a single_turn sub-agent,
// runs through workflow.RunNode on a node path below its owner's. When it
// fails, ADK returns the failure to the owner's model as the tool's error
// instead of failing the run, and the failed activation never gets an
// AfterAgentCallback. Such a failed activation (see failuresLocked) is purged
// and returned so that the plugin writes its AGENT_ERROR row without failing
// the invocation. An activation that did not fail, such as one that its
// BeforeAgentCallback answered and that therefore never gets an
// AfterAgentCallback either, is retired quietly, without an AGENT_ERROR row.
// So is one that ended with the cancellation of a caller that stopped reading
// mid-stream: stopped reports that the last session event was partial, and
// the tool run that still goes when the caller stops reading on it ends with
// the cancellation, which is no failure of its own. The tool calls of an agent
// can run concurrently and an activation does not record which call ran it,
// so nothing is ended while another tool call of the agent is still open.
// Whether an activation failed is decided only if its run is known to be over
// because it ran on the goroutine of a tool call that returned, after the
// call started (see markToolRunsDoneLocked): a tool can return while a run
// that it started on another goroutine goes on, makes model and tool calls,
// and still ends its InvocationContext. Such an activation is retired,
// without its InvocationContext being read, only if it owns no open call and
// has no failure recorded (see failuresLocked); otherwise it stays open, with
// its calls, until its own callbacks, the end of a run that waits for it, or
// the end of the invocation end it. since is the seq of the
// tool call that the current callback ends, or noFrame if it is not known:
// the activations that started on the current goroutine after the call did
// returned before it (see canSettleInSubtreeLocked).
func (inv *invocationTrace) endToolRunsLocked(ctx context.Context, id identity, stopped bool, now time.Time, ambient string, since uint64) []unclosedAgentSpan {
	// In RunLive the root agent has no span on agentStack, so
	// findAgentForContext returns -1 and Live-tool child activations parent
	// under agentStack[0] (spanKindInvocation). Using max(0, ...) lets
	// endToolRunsLocked end the activations that a Live tool ran once the
	// tool finishes.
	i := max(0, inv.findAgentForContext(ctx, id.path, id.agentName, id.branch))
	owner := inv.agentStack[i]
	for _, rec := range inv.toolSpans {
		if rec.parentSpanID == owner.spanID {
			return nil
		}
	}
	var purged []unclosedAgentSpan
	// Purging or retiring agentStack[j] removes only agentStack[j] and spans
	// after it, so the scan goes backwards.
	for j := len(inv.agentStack) - 1; j > i; j-- {
		rec := inv.agentStack[j]
		if rec.parentSpanID != owner.spanID || !isSubNodePath(rec.path, owner.path) {
			continue
		}
		// The tool call that ran the activation returned, but only a run on
		// the tool call's own goroutine is known to be over with it.
		done := rec.runDone || rec.activation == nil || rec.goroutineID == 0
		inv.settleEndedInSubtreeLocked(j, done, since)
		failures := inv.failuresLocked(j, stopped, nil, true)
		switch {
		case len(failures) == 0:
			inv.retireAgentLocked(j, done, since)
		case done:
			purged = append(purged, inv.purgeActivationLocked(j, failures, now, ambient)...)
		}
		// A run that is not known to be over and that still owns an open
		// call or has a failure recorded stays open: the tool's return does
		// not show that it failed, and its own callbacks, the end of a run
		// that waits for it, or the end of the invocation end it.
	}
	return purged
}

// markToolRunsDoneLocked records that the agent activations that started on
// the current goroutine after call, the tool call that a callback on that
// goroutine ends, finished running (see spanRecord.runDone). ADK runs a tool
// and its tool callbacks on one goroutine, so a run that the tool made on
// that goroutine returned before the tool did, and the callback synchronized
// with its end. A run that the tool started on another goroutine is not
// marked: the tool may have returned without waiting for it.
func (inv *invocationTrace) markToolRunsDoneLocked(call *spanRecord) {
	var goid uint64
	for _, rec := range inv.agentStack[1:] {
		if rec.runDone || rec.goroutineID == 0 || rec.seq <= call.seq {
			continue
		}
		if goid == 0 {
			if goid = currentGoroutineID(); goid == 0 {
				return
			}
		}
		if rec.goroutineID == goid {
			rec.runDone = true
		}
	}
}

// isSubNodePath reports whether path is the node path of a node that runs
// below the node path owner, as workflow.RunNode names the children it runs.
func isSubNodePath(path, owner string) bool {
	if owner == "" {
		return path != ""
	}
	return strings.HasPrefix(path, owner+"/")
}

// clearLastPartialLocked records that the caller read past the last partial
// event.
func (inv *invocationTrace) clearLastPartialLocked() {
	inv.lastSessionEventPartial = false
	inv.lastEventPartial = false
	inv.lastPartialSpanID = ""
	inv.partialLineage = nil
}

// clearLastPartialUnderLocked clears lastEventPartial when work starts in rec
// after the last partial event, if rec runs under the event's author or the
// author already ended, as a sibling does that a sequential agent starts
// next: the author went on, so if the work then fails while the caller reads
// on, the failure is reported. A workflow, which LLM and workflow agents run
// through, does not wait for its caller to read a partial event, and a
// parallel agent lets the author go on even when its caller stopped reading
// on the event, so the work may also have started after the caller stopped:
// both cancel the work that still runs then, and lastSessionEventPartial
// stays set, so that popUnclosedAgentSpans does not take the work that they
// canceled for a failure. Work whose context the author or the work itself
// canceled is still taken for one (see spanRecord.canceledByStop), so a
// failure of work that the author went on to run is reported. Work in a
// concurrent sibling of a running author shows nothing. partialLineage is kept
// until clearLastPartialLocked, because an async root such as a workflowagent
// whose FunctionNode emitted a partial event on the root branch and node path
// can start another node before its caller stops reading, and
// hasCanceledByStopUnderLocked separates that consumer-stop cancellation from a
// synchronous custom root that failed after draining its child (see
// popUnclosedAgentSpans).
func (inv *invocationTrace) clearLastPartialUnderLocked(rec *spanRecord) {
	if inv.lastPartialSpanID == "" {
		return
	}
	authorIdx := slices.IndexFunc(inv.agentStack, func(r *spanRecord) bool { return r.spanID == inv.lastPartialSpanID })
	if authorIdx < 0 || inv.lineageLocked(slices.Index(inv.agentStack, rec))[inv.lastPartialSpanID] {
		inv.lastEventPartial = false
		inv.lastPartialSpanID = ""
	}
}

// unclosedAgentSpan records an unclosed agent span popped during invocation termination.
type unclosedAgentSpan struct {
	span      rowSpan
	agentName string
	branch    string
	path      string
	err       error
	// timestamp is the time of the latest callback under the activation,
	// for the row's timestamp column and latency.
	timestamp time.Time
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
// that it runs under are marked as responded, and so are the settled ones. Of
// the author's own activation:
//
//   - The model calls that are still open end without a row: ADK runs no
//     AfterModelCallback for a call that a BeforeModelCallback answered, so a
//     response event of its agent is all that ends the call. The model calls
//     of the settled activations under it stay open, because they may still
//     be running, as when a custom agent writes an event while the model call
//     of its sub-agent is in flight: their own callbacks end them with their
//     spans and latency.
//   - The unrecovered errors recorded for it end: it went on to respond, for
//     example after a plugin recovered from a model error and an earlier
//     plugin's AfterModelCallback answered the call, which stops the chain
//     before this plugin's AfterModelCallback clears the error. Unrecovered
//     errors of activations below it stay recorded, because a parent event
//     does not mean that a failed sub-agent recovered.
//   - The open tool calls that the function responses of the event answer
//     end without a row in the same way: an earlier plugin's
//     AfterToolCallback that returns a result stops the chain before this
//     plugin's, so the response event is all that ends them. The other open
//     tool calls stay open, because the event does not answer them.
//
// A sub-agent that fails later, without any callback seeing the failure, is
// settled as well, so its failure is not reported: popUnclosedAgentSpans
// documents this trade-off. Errors recorded after the event still count.
//
// If the author has no open agent span, for example because a callback of an
// earlier plugin answered for it before this plugin saw it start, the response
// arrived under the activation that the author would have run under. That
// activation and the open activations that it runs under are marked as
// responded. Nothing else changes. The agent that the runner runs, which the
// event callback gets in id, runs right under the invocation, so an event of
// it marks nothing when it has no open span, as in a live run, where it runs
// without agent callbacks: a tool of it may already run a sub-agent on its
// branch and node path while the callback of the event that called the tool
// runs, which would be taken for the activation that the agent runs under.
func (tm *traceManager) completeResponse(ctx context.Context, id identity, author, branch string, calls []string) {
	tm.completeResponseWithActions(ctx, id, author, branch, calls, nil)
}

func (tm *traceManager) completeResponseWithActions(ctx context.Context, id identity, author, branch string, calls []string, eventActions *session.EventActions) {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	inv.mu.Lock()
	defer inv.mu.Unlock()
	// Any return, early ones included, can leave a late activation that
	// nothing can end anymore (see retireLateSettledLocked).
	defer inv.retireLateSettledLocked()
	wasPartial := inv.lastSessionEventPartial
	partialSpanID := inv.lastPartialSpanID
	inv.clearLastPartialLocked()
	i := inv.findAgent(id.path, author, branch)
	if i <= 0 {
		if author == "" || author == userAuthor {
			return
		}
		// ADK runs an agent's own AfterAgentCallback after this plugin's
		// retired the agent's activation, so recovery content that a custom
		// agent returns arrives once the activation is gone: scan all open
		// spans because a concurrent undeclared sibling started before the
		// custom agent retired can sit above the ancestor carrying the
		// swallowable retiredCause (see retireAgentLocked).
		for _, rec := range inv.agentStack {
			rec.clearRetiredCauseFor(author, branch, id.path)
		}
		inv.retireDrainedDescendantsLocked(func(p *spanRecord, openSet map[string]bool) bool {
			return !openSet[p.spanID] && !p.propagatesChildError && p.contextFailureCause() == nil &&
				p.name == author && p.branch == branch && (id.path == "" || p.path == "" || p.path == id.path)
		})
		parent := inv.parentFor(identity{agentName: author, branch: branch, path: id.path})
		if author == id.agentName {
			return
		}
		if j := slices.Index(inv.agentStack, parent); j > 0 {
			inv.touchAgentActivityLocked(inv.agentStack[j], now)
			inv.markRespondedLocked(inv.lineageLocked(j))
		}
		return
	}
	var matchedPending, absorbed bool
	i, matchedPending, absorbed = inv.selectResponseOwnerLocked(i, calls, wasPartial, partialSpanID, now)
	if absorbed {
		return
	}
	owner := inv.agentStack[i]
	inv.touchAgentActivityLocked(owner, now)
	// An activation is created after the one that it runs under, so one pass
	// over the later spans finds all that run under the owner.
	under := map[string]bool{owner.spanID: true}
	for _, rec := range inv.agentStack[i+1:] {
		if under[rec.parentSpanID] && withinBranch(rec.branch, branch) && withinPath(rec.path, id.path, false) {
			under[rec.spanID] = true
		}
	}
	for _, rec := range inv.agentStack[i:] {
		if under[rec.spanID] {
			rec.retiredCauses = nil
		}
	}
	if !owner.concurrentPeer || matchedPending {
		for key, e := range inv.unrecoveredErrors {
			if inv.activationOfLocked(e) == i && !e.collateral {
				delete(inv.unrecoveredErrors, key)
			}
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
	for key, rec := range inv.completedTools {
		if rec.parentSpanID == owner.spanID && slices.Contains(calls, toolKeyCall(key)) && eventActionsMatchLiveAttempt(eventActions, rec.errorAttempt.Value()) {
			delete(inv.completedTools, key)
		}
	}
	inv.syncOpenCallsLocked()
	inv.markRespondedLocked(inv.lineageLocked(i))
	inv.markRespondedLocked(under)
	if !owner.propagatesChildError && owner.contextFailureCause() == nil {
		inv.retireDrainedDescendantsLocked(func(p *spanRecord, _ map[string]bool) bool {
			return p == owner
		})
	}
	// Removing the matched tool calls may leave no launcher open for an
	// activation whose run was already known to be over (see
	// launcherOpenLocked). The event does not show that any run that those
	// calls made returned, so none is marked as finished here (see
	// markToolRunsDoneLocked).
	inv.settleFinishedLocked()
}

func sameMapPtr[K comparable, V any](a, b map[K]V) bool {
	if a == nil || b == nil {
		return false
	}
	return reflect.ValueOf(a).UnsafePointer() == reflect.ValueOf(b).UnsafePointer()
}

func eventActionsMatchLiveAttempt(eventActions, live *session.EventActions) bool {
	if live == nil {
		return true
	}
	if live.StateDelta != nil || live.ArtifactDelta != nil {
		if eventActions == nil {
			return false
		}
		return sameMapPtr(eventActions.StateDelta, live.StateDelta) ||
			sameMapPtr(eventActions.ArtifactDelta, live.ArtifactDelta)
	}
	return eventActions == nil || (eventActions.StateDelta == nil && eventActions.ArtifactDelta == nil)
}

// retireDrainedDescendantsLocked removes healthy open agent spans that stayed
// on agentStack only because their buffered session events arrived after their
// enclosing workflow or parallel parent had already retired, once a custom
// ancestor matching matchAnc emits a final response that settles the drained
// subtree.
func (inv *invocationTrace) retireDrainedDescendantsLocked(matchAnc func(p *spanRecord, openSet map[string]bool) bool) {
	openSet := make(map[string]bool, len(inv.agentStack))
	for _, r := range inv.agentStack[1:] {
		openSet[r.spanID] = true
	}
	failed := inv.openCallOwnersLocked(nil)
	for j, r := range inv.agentStack {
		unresponded := (r.runDone || inv.launcherOpenLocked(r)) && !r.responded()
		if j > 0 && (failed[r.spanID] || unresponded) {
			maps.Copy(failed, inv.lineageLocked(j))
		}
	}
	for _, e := range inv.unrecoveredErrors {
		if e.collateral {
			continue
		}
		if j := inv.activationOfLocked(e); j > 0 {
			maps.Copy(failed, inv.lineageLocked(j))
		}
	}
	inv.removeAgentsLocked(func(d *spanRecord) bool {
		if d.kind != spanKindAgent || failed[d.spanID] || d.parent == nil || d.parent.kind != spanKindAgent || openSet[d.parent.spanID] {
			return false
		}
		for p := d.parent; p != nil && p.kind == spanKindAgent; p = p.parent {
			if p != d.parent && matchAnc(p, openSet) {
				return true
			}
			if openSet[p.spanID] {
				return false
			}
		}
		return false
	})
	inv.syncOpenAgentsLocked()
}

// selectResponseOwnerLocked picks which open activation of agentStack[i]'s
// (path, name, branch) owns a session event in completeResponse, reporting
// whether it matched pending model/tool callback evidence, or absorbs the event
// when its author already retired while a peer remained open.
func (inv *invocationTrace) selectResponseOwnerLocked(i int, calls []string, wasPartial bool, partialSpanID string, now time.Time) (int, bool, bool) {
	base := inv.agentStack[i]
	k := llmKey(identity{path: base.path, agentName: base.name, branch: base.branch})
	absorbOntoParent := func() (int, bool, bool) {
		if base.parent != nil && base.parent.kind == spanKindAgent {
			inv.touchAgentActivityLocked(base.parent, now)
			if pIdx := slices.Index(inv.agentStack, base.parent); pIdx > 0 {
				inv.markRespondedLocked(inv.lineageLocked(pIdx))
			}
		}
		return 0, false, true
	}
	var peerIdxs []int
	for j := len(inv.agentStack) - 1; j > 0; j-- {
		rec := inv.agentStack[j]
		if rec.kind == spanKindAgent && rec.path == base.path && rec.name == base.name && rec.branch == base.branch {
			peerIdxs = append(peerIdxs, j)
		}
	}
	if len(peerIdxs) <= 1 && !base.concurrentPeer && inv.unrespondedRetired[k] == 0 && inv.retiredPeerCount[k] == 0 {
		if len(calls) > 0 {
			base.awaitingToolEvent = max(0, base.awaitingToolEvent-len(calls))
		} else if base.awaitingModelEvent > 0 {
			base.awaitingModelEvent--
		}
		return i, true, false
	}
	// hasUnrecErr reports whether activation idx has a non-collateral
	// unrecovered error recorded, so a model-event credit should not let a
	// peer's response claim its events (see P1-3).
	hasUnrecErr := func(idx int) bool {
		for _, e := range inv.unrecoveredErrors {
			if !e.collateral && inv.activationOfLocked(e) == idx {
				return true
			}
		}
		return false
	}
	if len(calls) > 0 {
		for _, j := range peerIdxs {
			cand := inv.agentStack[j]
			if cand.awaitingToolEvent > 0 && !hasUnrecErr(j) {
				cand.awaitingToolEvent = max(0, cand.awaitingToolEvent-len(calls))
				return j, true, false
			}
			for key, rec := range inv.toolSpans {
				if rec.parentSpanID == cand.spanID && slices.Contains(calls, toolKeyCall(key)) {
					return j, true, false
				}
			}
		}
	} else {
		for _, j := range peerIdxs {
			cand := inv.agentStack[j]
			if cand.awaitingModelEvent > 0 && !hasUnrecErr(j) {
				cand.awaitingModelEvent--
				return j, true, false
			}
		}
	}
	if inv.unrespondedRetired[k] > 0 {
		inv.unrespondedRetired[k]--
		return absorbOntoParent()
	}
	if wasPartial && partialSpanID != "" {
		for _, j := range peerIdxs {
			if cand := inv.agentStack[j]; cand.spanID == partialSpanID && cand.hasResponse && !cand.modelResponded {
				return j, false, false
			}
		}
	}
	for _, j := range peerIdxs {
		if !inv.agentStack[j].hasResponse && !hasUnrecErr(j) {
			return j, false, false
		}
	}
	for _, j := range peerIdxs {
		if !hasUnrecErr(j) {
			inv.agentStack[j].extraResponses++
			return j, false, false
		}
	}
	if inv.retiredPeerCount[k] > 0 {
		for _, rec := range inv.agentStack {
			rec.clearRetiredCauseFor(base.name, base.branch, base.path)
		}
		return absorbOntoParent()
	}
	for _, j := range peerIdxs {
		if !inv.agentStack[j].hasResponse {
			return j, false, false
		}
	}
	base.extraResponses++
	return i, false, false
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
	bareKey := key
	key = callKey(ctx, key)
	inv := tm.invocation(ctx, id)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	if inv.unrecoveredErrors == nil {
		inv.unrecoveredErrors = make(map[string]recordedError)
	}
	var spanID string
	if i := inv.findAgentForContext(ctx, id.path, id.agentName, id.branch); i > 0 {
		spanID = inv.agentStack[i].spanID
	}
	seq := inv.nextSeqLocked()
	if prev, exists := inv.unrecoveredErrors[key]; exists && (prev.spanID == "" || prev.spanID == spanID) {
		seq = prev.seq
	}
	if prev, exists := inv.unrecoveredErrors[key]; exists && prev.spanID != "" && prev.spanID != spanID {
		inv.unrecoveredErrors[prev.spanID+"\x00"+key] = prev
	}
	// The error is checked like a call that ctx starts under the agent. A
	// context cancels its children before cancel returns, so the contexts
	// that a workflow or a parallel agent canceled are canceled by now.
	owner := inv.ownerOf(ctx, id)
	if owner.kind == spanKindAgent {
		inv.touchAgentActivityLocked(owner, tm.now())
		// A recorded error clears stale model- and tool-event credits so a
		// peer's response cannot claim this activation's events (see
		// selectResponseOwnerLocked).
		owner.awaitingModelEvent = 0
		owner.awaitingToolEvent = 0
	}
	call := &spanRecord{canceled: canceledFunc(ctx), parent: owner, stopPoint: owner.kind == spanKindInvocation}
	var cause error
	var collateral bool
	if errors.Is(err, context.Canceled) {
		cause, collateral = call.collateralFailureCause()
		if cause != nil && !errors.Is(err, cause) {
			err = fmt.Errorf("%w: %w", err, cause)
		}
	}
	inv.unrecoveredErrors[key] = recordedError{
		err:        err,
		cause:      cause,
		spanID:     spanID,
		bareKey:    bareKey,
		seq:        seq,
		canceled:   call.canceledByStop(),
		collateral: collateral,
	}
}

// clearError clears any recorded unrecovered error for key on the invocation
// trace when a plugin recovered from the error on the same model call
// (completedModels[key] != nil). A later call on a new activation, such as a
// retry in a loop agent, does not erase an earlier unrecovered error.
func (tm *traceManager) clearError(ctx context.Context, id identity, key string) {
	key = callKey(ctx, key)
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
//     error, and so does a final event of its activation. When the last
//     session event was partial and ctx was not canceled, a context.Canceled
//     error does not count if the context of the callback that recorded it
//     was canceled by a workflow or a parallel agent with cause
//     context.Canceled (see spanRecord.canceledByStop): a caller that stops
//     reading on a partial event stops the workflow that the run goes
//     through, and any parallel agent, which cancel the work that still runs,
//     and a model ends a canceled call with the context's error. When ctx was
//     not canceled and a context.Canceled error's callback context ended at a
//     stop point with another cause (recordedError.collateral, see
//     spanRecord.collateralFailureCause), as when a parallel agent's errgroup
//     cancels sibling branches with a failing branch's error, the canceled
//     sibling's context.Canceled is set aside while its cause is kept for the
//     invocation and any attributed failure rows. A context.Canceled error
//     still fails the invocation directly (wrapping any custom cause) if the
//     call canceled a context of its own, or if an agent, a callback or a
//     tool canceled the context that the call runs under, such as a custom
//     agent that cancels the context of a sub-agent on a timeout of its own
//     while the caller reads on.
//   - The error of ctx when the invocation was canceled. Cancellation is
//     returned even when every agent span is closed, or when no agent ran at
//     all, for example because a BeforeRunCallback canceled the invocation.
//   - An open agent span that still owns an open model or tool call: a
//     BeforeModelCallback or BeforeToolCallback error stops the agent
//     without any error callback.
//   - An open agent span (most recent first) whose activation produced no
//     response (see spanRecord.responded) — neither ending its own
//     InvocationContext via BeforeAgentCallback nor producing a model response
//     or an event with content or output of its own, of an activation that ran
//     under it or of one that it ran under — for example because its Run
//     returned an error first.
//   - Otherwise, the non-context.Canceled, non-bare-context.DeadlineExceeded
//     cause with which a span's context ended (see openFailureCauseLocked and
//     spanRecord.retiredCauses), when every open activation had already
//     responded or been retired.
//
// The open-call and unresponded-span checks use the open failure cause when
// one is present, or else a synthetic error, and run only if ctx was not
// ended and either the last session event was not partial, work that shows
// that the run went on started after it (see invocationTrace.lastEventPartial),
// or a span's context ended with a cause other than context.Canceled or bare
// context.DeadlineExceeded (see spanRecord.openFailureCause and
// openFailureCauseLocked; bare context.DeadlineExceeded is excluded because
// ADK's workflow scheduler leaves finished NodeConfig.Timeout contexts
// uncanceled, so an open BeforeAgentCallback shortcut span reaches
// context.DeadlineExceeded after succeeding, whereas parallelagent's errgroup
// wraps branch errors): an agent that a BeforeAgentCallback answered
// ends its node, so no AfterAgentCallback closes its span, yet it did not
// fail, and a caller that stops reading mid-stream leaves the streaming
// activations and their siblings open. A workflow does not wait for its
// caller to read a partial event, and a parallel agent lets the author of a
// partial event go on even when the caller stopped reading on it; both then
// cancel the work that still runs with cause context.Canceled. So while the
// last session event is partial and ctx was not canceled, neither an
// activation nor a call whose callback's context a workflow or a parallel
// agent canceled with cause context.Canceled is reported, and neither is an
// activation that wrote a partial event since the last session event that was
// not partial, itself or through an activation that ran under it, such as the
// parallel agent itself, whose context the run need not cancel. When the
// last session event is still partial without such a sibling cause, only
// activations at or under an open workflow node whose run is known to be over
// (see inWorkflowNodeSubtreeLocked) are still checked, because the workflow
// scheduler cancels every running node on consumer stop (canceledByStop) and
// removes a failing node from runCancels before cancelAll even on ADK releases
// that do not emit node completion events, leaving the failing node's context
// uncanceled. When a branch of a parallel agent fails instead, its errgroup
// cancels the branch contexts with the failing branch's error as the cause:
// collateral context.Canceled model errors on canceled siblings are set
// aside while their cause is kept, openFailureCauseLocked (together with
// retiredCauses when a root retired responded branches on drain) overrides
// lastEventPartial, and both INVOCATION_ERROR and any attributed AGENT_ERROR
// rows carry that errgroup cause. Because every branch context shares the
// errgroup cause, AGENT_ERROR is attributed by the open-call and unresponded
// scan: it names the failing branch in a two-branch run whose sibling already
// streamed content and ended its call, can name a canceled sibling that still
// has an open call or has not responded (for example when three or more
// branches run or when the streaming branch itself fails after emitting
// partial content), and emits no AGENT_ERROR row when every branch had
// already responded and closed its calls.
//
// Failures that the callbacks cannot tell apart from a success are not
// reported, and the invocation is reported as completed:
//
//   - Outside a parallel branch whose Run yields an error (which leaves the
//     branch span open with the parallel agent's errgroup cause on its own
//     context even without a running sibling) or whose failure cancels a
//     still-running sibling's context with the branch error, a run that fails
//     after the failing activation produced a response, such as a custom
//     agent outside a parallel branch that writes an event and then returns
//     an error, or an AfterAgentCallback that returns an error when no
//     parallel sibling is still open. Because contextFailureCause returns a
//     cause only when the outermost span sharing it is a parallel branch
//     (top.isParallelBranch), any cause recorded on a custom agent's own
//     WithCancelCause is ignored (and is not promoted into retiredCauses by
//     retireAgentLocked), whether the custom agent handles the sub-agent's
//     failure or propagates it after the sub-agent already emitted content,
//     and including when a ParallelAgent runs under such a canceled outer
//     context. The callbacks see the same as for a caller that stopped
//     reading once it had the response.
//   - A run that fails without a recorded error, and without starting more
//     work, after a partial event, such as a custom agent that writes a
//     progress event without content and then returns an error. The
//     callbacks see the same as for a caller that stopped reading on the
//     partial event.
//   - A sub-agent that fails without any callback after an agent that it
//     runs under produced content, such as a custom agent that writes an
//     event while its sub-agent runs. The event settles the activations that
//     still run under the agent, so the callbacks cannot tell the failed
//     sub-agent apart from one whose caller stopped reading once it had the
//     response.
//   - Work that started after the last session event, a partial one, and
//     whose failure without a recorded error never reaches a workflow node
//     or a parallel agent as an error (for example a sub-agent error that a
//     custom parent swallows while the last session event remains partial,
//     as when the caller stopped on that partial event or the custom parent
//     runs under an LLM root; draining a non-LLM root clears the partial
//     marker in popAgentSpan). The callbacks cannot tell the failure apart
//     from work that started after its caller stopped reading.
//   - Work that ends with context.Canceled after the last session event, a
//     partial one, because code canceled a context of its own that it ran a
//     workflow node or a branch of a parallel agent with, or that it ran the
//     agent that the runner runs with. The callbacks cannot tell the
//     cancellation apart from that of a workflow or a parallel agent whose
//     caller stopped reading.
//
// Conversely, a model error that agent code handles without yielding it is
// reported, as no callback tells it apart from one that reached the caller
// (see unrecoveredErrors). Likewise, the callbacks cannot tell a custom agent
// that stops iterating a sub-agent before the sub-agent produced content or
// ended its own InvocationContext (without emitting a settling final event on
// the same branch and node path while every ancestor span between it and the
// sub-agent is still open), or that swallows a sub-agent's silent error and
// recovers through a fallback sub-agent, apart from a sub-agent of a workflow
// agent that failed silently: retireAgentLocked keeps the unresponded
// sub-agent open when its parent ends, and completeResponse follows
// parentSpanID through open spans only (so reading an intermediate sub-agent
// such as a sequential agent to its end after its child failed without
// responding retires the intermediate span and leaves the child open beyond
// the reach of the outer custom agent's subsequent final event), so the run is
// reported with AGENT_ERROR and INVOCATION_ERROR. Likewise, a custom agent
// that handles a parallel agent's error is reported as failed
// (retireAgentLocked moves the parallel branches' retired causes onto the
// enclosing agent when the parallel agent ends, or keeps the parallel agent
// and its branches open if the custom agent broke out early), unless every
// branch of the parallel agent had responded, no canceled sibling recorded a
// collateral context.Canceled error in unrecoveredErrors, and the custom agent
// reads the parallel agent to its end and then writes a final event of its own
// (which clears retiredCauses in completeResponse, but does not settle an
// unresponded sub-branch or clear collateral errors recorded on a child
// branch), from its Run or from its own AfterAgentCallback (which ADK runs
// after this plugin's retired the custom agent, so the event clears the
// causes that the custom agent's retirement moved up). Runs that overlap with
// one invocation ID share the spans that this ends (see traceManager).
//
// Three RetryConfig shapes still end INVOCATION_ERROR after the retry succeeds,
// as they do on main. First, a function node with RetryConfig whose function
// runs an agent and returns its error: only the agent of an agent node or
// ParallelWorker attempt counts as an attempt (see attemptSite), so the
// function's earlier run of the agent is not purged. Second, an agent node with
// RetryConfig inside a nested workflow.NewWorkflowNode, whose inner scheduler has
// no workflow agent span on agentStack (so its goroutines are not started by the
// outer workflow agent's scheduler, and its completion events name the inner
// workflow rather than the outer). Third, a workflow retry whose earlier
// attempt relies on endRestartedNodeRuns (when no node completion event and
// recorded failure purged it in pushAgentSpan) if the caller stops reading
// before the workflow agent ends, so AfterAgentCallback and endRestartedNodeRuns
// never run.
//
// The scan does not read whether an activation ended its own
// InvocationContext, which is written without synchronization on the
// goroutine that runs the activation, which may still run: it relies on what
// was read once the activation's run was known to be over, after its own run,
// the run of an enclosing agent that waits for it (see spanRecord.joins), or
// the tool call that ran it on the call's own goroutine finished (see
// settleEndedLocked and settleEndedInSubtreeLocked). Nothing is read while a
// tool call or an agent that may still run the activation is open (see
// launcherOpenLocked), or before an enclosing agent that dropped the
// activation's event finishes without emitting a final response event of its
// own: when the invocation ends while a tool call or enclosing agent still
// runs a sub-agent that a BeforeAgentCallback answered (as when a live run is
// closed during the call, or a caller stops reading before the enclosing
// agent ends), the callbacks cannot tell the sub-agent apart from one that
// failed silently while still running, and the run is reported with
// AGENT_ERROR for the sub-agent and INVOCATION_ERROR.
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
	// A caller that stopped reading on the last session event, a partial one,
	// stopped the run's workflow and any parallel agent too, which cancel the
	// work that still runs, including work that started after the event. A
	// context that an agent canceled itself is no such cancellation.
	stoppedMidStream := cancelErr == nil && inv.lastSessionEventPartial
	canceledByStop := func(rec *spanRecord) bool { return stoppedMidStream && rec.canceledByStop() }

	keys := inv.sortedErrorKeysLocked()
	unionLineage := make(map[string]bool)
	spanErrors := make(map[string]error)
	var primaryErr, collateralCause error

	for _, k := range keys {
		e := inv.unrecoveredErrors[k]
		if e.err == nil || (stoppedMidStream && e.canceled && errors.Is(e.err, context.Canceled)) {
			continue
		}
		if e.collateral && cancelErr == nil {
			collateralCause = cmp.Or(collateralCause, e.cause)
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

	siblingCause := cmp.Or(inv.openFailureCauseLocked(), collateralCause)
	if len(inv.agentStack) <= 1 {
		if primaryErr == nil && !invocationEnded(ctx) && siblingCause != nil {
			primaryErr = siblingCause
		}
		inv.truncateAgentsLocked()
		return nil, primaryErr
	}

	if primaryErr == nil && !invocationEnded(ctx) {
		// When AfterRun runs on the goroutine that started agentStack[1] (the
		// runner's root agent, or an activation that stayed open after the
		// root agent's span retired), settleEndedInSubtreeLocked(1, true, 0)
		// records that agentStack[1] finished running, settles it, and
		// settles the activations under it whose runs are known to be over
		// before the reverse-order loop below pops any remaining unclosed
		// spans. done and since 0 hold here: AfterRun runs only after the
		// root agent's Run returned, and every agent run that started on this
		// goroutine ran inside that Run, so it has returned too.
		if len(inv.agentStack) > 1 {
			if goid := currentGoroutineID(); goid != 0 && inv.agentStack[1].goroutineID == goid {
				inv.settleEndedInSubtreeLocked(1, true, 0)
			}
		}
		partialSuppress := inv.lastEventPartial && siblingCause == nil
		ignoreCall := func(call *spanRecord) bool {
			return canceledByStop(call) || (partialSuppress && !inv.inWorkflowNodeSubtreeLocked(call.parent))
		}
		failingIdx := inv.agentWithOpenCallLocked(ignoreCall)
		if failingIdx <= 0 {
			for i := len(inv.agentStack) - 1; i > 0; i-- {
				// partialLineage is empty unless the last session event is
				// partial, and a canceled ctx is the error by now. Once
				// work that shows that the run went on started after it
				// (!lastEventPartial), an activation that wrote it is taken
				// for one whose caller stopped reading only if a workflow or
				// a parallel agent canceled work under it, as they do then.
				rec := inv.agentStack[i]
				if partialSuppress && !inv.inWorkflowNodeSubtreeLocked(rec) {
					continue
				}
				stoppedOn := inv.partialLineage[rec.spanID] && (inv.lastEventPartial || inv.hasCanceledByStopUnderLocked(i, canceledByStop))
				if !rec.responded() && !canceledByStop(rec) && !stoppedOn {
					failingIdx = i
					break
				}
			}
		}
		if failingIdx > 0 {
			primaryErr = siblingCause
			if primaryErr == nil {
				failingAgentName := inv.agentStack[failingIdx].name
				primaryErr = fmt.Errorf("agent %q execution failed before completion", failingAgentName)
			}
			failingRec := inv.agentStack[failingIdx]
			spanErrors[failingRec.spanID] = primaryErr
			unionLineage[failingRec.spanID] = true
			for ancestorID := range inv.lineageLocked(failingIdx) {
				unionLineage[ancestorID] = true
				if spanErrors[ancestorID] == nil {
					spanErrors[ancestorID] = primaryErr
				}
			}
		} else if siblingCause != nil {
			primaryErr = siblingCause
		}
	}

	if primaryErr == nil {
		inv.truncateAgentsLocked()
		inv.syncOpenAgentsLocked()
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
	inv.truncateAgentsLocked()
	inv.syncOpenAgentsLocked()
	return popped, primaryErr
}

// responded reports whether rec produced a response or was seen to have ended
// its own InvocationContext cleanly, as when a BeforeAgentCallback answered for
// it (see settleEndedLocked). It does not call rec.ended: the goroutine that
// runs the activation may still be writing that state, for example when the
// invocation ends while a tool call that runs the activation is still open.
func (rec *spanRecord) responded() bool {
	return rec.hasResponse || rec.contextEnded
}

// settleEndedLocked calls rec.ended and, once it reports true, records that in
// contextEnded and drops rec.ended. An activation's InvocationContext is
// marked as ended on the goroutine that runs the activation, without
// synchronization: by ADK when a BeforeAgentCallback answers for the
// activation, before the activation yields its first event, or by a custom
// agent that calls EndInvocation while it runs, which it can do after it
// wrote its last event. So the callers call it only for an activation whose
// run is known to be over (see spanRecord.runDone, which neither a node
// completion event nor a run on the current goroutine shows by itself), on a
// goroutine that has synchronized with the end of that run (see
// settleEndedInSubtreeLocked and settleFinishedLocked), and a false result is
// checked again when such a point is reached for an activation that it runs
// under. An activation whose span an event retires, such as a finish_task
// function response, is not settled on that event (see
// settleEndedInSubtreeLocked).
func (rec *spanRecord) settleEndedLocked() {
	if rec.ended != nil && rec.ended() {
		rec.contextEnded = true
		rec.ended = nil
	}
}

// noFrame is the since bound of settleEndedInSubtreeLocked that takes no
// activation to be over because it ran on the current goroutine.
const noFrame = ^uint64(0)

// settleEndedInSubtreeLocked records that agentStack[i] finished running and
// settles it (see settleEndedLocked) if done is set, records that the open
// activations under it whose runs are known to be over finished running (see
// spanRecord.runDone and canSettleInSubtreeLocked), and settles those (see
// settleFinishedLocked).
//
// done reports that the caller has synchronized with the end of
// agentStack[i]'s run. The callers that pass true are popAgentSpan, after the
// activation's own AfterAgentCallback ran (directly through
// reconcileConcurrentResponseOnPopLocked and through retireAgentLocked);
// endToolRunsLocked, for an activation that ran on the goroutine of a tool
// call that returned, after the call started (see markToolRunsDoneLocked);
// pushAgentSpan, when a later attempt of the activation's
// workflow.ParallelWorker item started on the goroutine on which it ran (see
// laterAttempt there); endRestartedNodeRuns, when the workflow agent whose
// scheduler ran it as the agent of a node attempt, and started that node
// again, ended; and popUnclosedAgentSpans, for agentStack[1] when AfterRun
// runs on the goroutine that started it, after the root agent's Run returned.
//
// Retiring a span on an event passes done == false, as the event does not
// show that the run that wrote it returned: popTaskAgentSpan, through
// retireAgentLocked, for a finish_task function response, which a custom
// agent can forward for a sub-agent that it runs on another goroutine before
// that sub-agent ends its InvocationContext; and popDelegatingAgentSpan,
// through retireAgentLocked, for the earlier activation of an agent that
// delegated to a completed task (see spanRecord.taskDelegated). So does
// endToolRunsLocked for an activation that a tool started on another
// goroutine, which is not known to have finished when the tool returns. A
// node completion event does not count either: a node can write one itself
// while it runs (see spanRecord.nodeRunEnded).
//
// The run of an activation under agentStack[i] is known to be over only if it
// started on the current goroutine after the frame that since names, or if
// each agent between it and an activation whose run is over waits for the run
// of the one below it (see spanRecord.joins): a custom agent or a workflow
// function node can return while a run that it started on another goroutine
// still goes on and may still mark its InvocationContext as ended, so such a
// run is left unsettled, with its InvocationContext unread, until something
// else shows that it is over, such as its own AfterAgentCallback. A sub-agent
// that a tool call or an agent that is still open may have run instead (see
// launcherOpenLocked) is settled only once they end.
//
// since is the seq of the activation or tool call on whose goroutine, and
// inside whose run, the calling callback runs, 0 if the callback's stack shows
// no agent run below it, or noFrame: an agent run that started on a goroutine
// after an activation or a tool call started there runs inside it, so it has
// returned by the time the activation's or call's own callback runs after its
// end, whereas one that started earlier may still be suspended below that
// callback, as when a function node runs an agent inside its loop over the
// events of another run.
func (inv *invocationTrace) settleEndedInSubtreeLocked(i int, done bool, since uint64) {
	root := inv.agentStack[i]
	if done {
		root.settleEndedLocked()
		root.runDone = true
	}
	subtree := inv.subtreeLocked(i)
	var needDescendantCheck bool
	for _, rec := range inv.agentStack[i+1:] {
		if subtree[rec.spanID] && !rec.runDone {
			needDescendantCheck = true
			break
		}
	}
	if needDescendantCheck {
		var curGoid uint64
		if since != noFrame {
			curGoid = currentGoroutineID()
		}
		for _, rec := range inv.agentStack[i+1:] {
			if subtree[rec.spanID] && !rec.runDone && inv.canSettleInSubtreeLocked(i, rec, curGoid, since) {
				rec.runDone = true
			}
		}
	}
	inv.settleFinishedLocked()
}

// canSettleInSubtreeLocked reports whether the run of d, an open activation
// under agentStack[rootIdx], is known to be over: d started on the current
// goroutine curGoid after the frame that since names (see
// settleEndedInSubtreeLocked), or each agent from d's parent up to one that
// finished running (runDone) or started on the current goroutine after that
// frame, and at most up to agentStack[rootIdx], waits for the run of the
// activation below it (see spanRecord.joins), so that the end of that agent's
// run is also the end of d's. An activation without activation or goroutine
// information, as in direct traceManager tests, counts as over. Activations
// are created after the ones that they run under, so
// settleEndedInSubtreeLocked, which visits agentStack in order, decides for
// the activations that d runs under before it decides for d.
func (inv *invocationTrace) canSettleInSubtreeLocked(rootIdx int, d *spanRecord, curGoid, since uint64) bool {
	root := inv.agentStack[rootIdx]
	returnedHere := func(rec *spanRecord) bool {
		return curGoid != 0 && rec.goroutineID == curGoid && rec.seq > since
	}
	if d.activation == nil || d.goroutineID == 0 || returnedHere(d) {
		return true
	}
	for c, p := d, d.parent; p != nil && p.kind == spanKindAgent; c, p = p, p.parent {
		if !p.joins(c, false) {
			return false
		}
		if p.runDone || returnedHere(p) {
			return true
		}
		if p == root {
			return false
		}
	}
	return false
}

// joins reports whether the run of the agent activation p ends only after the
// run of c, an activation that runs right under it, which holds when p is one
// of ADK's built-in agents that yield the errors of their sub-agents (see
// propagatesChildError) and
//
//   - c ran on p's goroutine, as the sub-agents of a SequentialAgent or a
//     LoopAgent do: p's Run runs them before it returns;
//   - p is a workflow agent whose scheduler started c's goroutine (see
//     goroutineOrigin.startedBy), as it does for the agent of an agent node:
//     the scheduler waits for every node run that it started before the
//     workflow agent's Run returns; or
//   - p is a ParallelAgent that started c's goroutine for a declared
//     sub-agent, on the branch that it assigns to that sub-agent, and p's run
//     is known to have returned all its events, after which ADK runs its
//     AfterAgentCallback (see completed): a ParallelAgent waits for every
//     branch before then. Since ADK v2.1.0 it also waits before its Run
//     returns when it stops early, which only the node completion events of
//     later releases report (see nodeRunEnded). With reported set, as when a
//     recorded failure of p or the end of the workflow agent whose scheduler
//     ran p as a node decides which runs under p failed (see
//     reportedOverLocked), such a report counts too.
//
// A custom agent, or a function node that runs an agent, can return while a
// run that it started on another goroutine still goes on, and so can an agent
// whose goroutine is not known, so joins reports false for them.
func (p *spanRecord) joins(c *spanRecord, reported bool) bool {
	if !p.propagatesChildError || p.goroutineID == 0 || c.goroutineID == 0 {
		return false
	}
	switch {
	case c.goroutineID == p.goroutineID:
		return true
	case p.isWorkflow:
		return c.origin.startedBy(schedulerCreator, p)
	case p.isParallel:
		return c.origin.startedBy(errgroupCreator, p) &&
			slices.Contains(p.children, c.name) && c.branch == parallelBranch(p, c.name) &&
			(p.completed || (reported && p.nodeRunEnded))
	}
	return false
}

// settleFinishedLocked settles (see settleEndedLocked) the open activations
// that are already known to have finished running (see spanRecord.runDone)
// and that no tool call or agent that is still open may have run (see
// launcherOpenLocked). It marks no run as finished itself, so the callers call
// it after something marked one or ended a tool call or an agent that may have
// held one back: settleEndedInSubtreeLocked when an agent ends, popToolSpan
// when a tool call returns, after markToolRunsDoneLocked marked the runs that
// the call made on its own goroutine, and completeResponse when a function
// response of an agent's event ends the agent's tool calls. Such an event
// shows that the calls answered, not that any run that they made is over, so
// it only releases activations that were marked before.
func (inv *invocationTrace) settleFinishedLocked() {
	for _, rec := range inv.agentStack[1:] {
		if rec.runDone && !inv.launcherOpenLocked(rec) {
			rec.settleEndedLocked()
		}
	}
}

// retireLateSettledLocked retires the activations of a late trace that
// nothing can end anymore. An activation that its BeforeAgentCallback
// answered never gets an AfterAgentCallback, and in the late state, where
// AfterRun has already run, popUnclosedAgentSpans never ends it either. Such
// an activation is retired once it is settled (its run is over and it ended
// its InvocationContext, see settleFinishedLocked), no tool call or agent
// that is still open may have run it (see launcherOpenLocked), and the agent
// that it runs under is no longer open: that agent ended while such a tool
// call or agent still held the activation open, or the activation runs right
// under the invocation span. While that agent is open, its own end retires
// the activation with it, even one that carries a failure cause, and moves a
// failure cause of what it retires up to the nearest span above it that is
// still open (see retireAgentLocked). As there, an activation that failed,
// and those that it runs under, stay. Here, so does one that carries a
// failure cause (see spanRecord.openFailureCause) or runs an activation that
// carries one, but only while something can still read the cause through
// it: an agent above it that is still open, through which failure reporting
// (see failuresLocked) reads the cause and whose end moves the cause up, or,
// for an activation that runs right under the invocation span on a node path
// below the invocation span's, an open or later tool call that may end
// without its agent on agentStack (as the root agent's calls do in RunLive,
// or when an agent ends before its tool call does), where the end of such a
// call reports the failures of those activations and of the ones under them
// (see endToolRunsLocked). Failure reporting reads the causes in
// the subtree of the activation that it reports for, and writes the rows of
// the activations that failed along their lineage on agentStack, so the
// activations between the one that carries the cause and the one through
// which it is read stay too, and for a failure cause the rows do not depend
// on whether the reading comes as soon as the activations settle or only
// later. That includes the cause of a collateral error (see recordedError)
// whose cancellation started at a parallel branch, as when a parallel agent
// cancels its other branches: the cause is then that branch's context
// failure cause, which keeps the activations as above. When such a
// cancellation started at another stop point, such as a workflow node,
// spanRecord.openFailureCause does not return the cause and the scan of the
// unrecovered errors in the body skips the error, so nothing keeps an
// activation for it, and the rows that failure reporting writes from that
// cause depend on whether the reading comes before the activation that the
// error is attributed to retires or only later. Unlike a failure, a failure
// cause keeps no activation once nothing that can read it remains (for example,
// when an open caller above it ends and it is not right under the invocation
// span on a sub-node path); once reported by endToolRunsLocked or cleared by
// recovery, the activation then retires, as do those above it that the cause
// kept. The callbacks that can release such an activation (the end of an
// agent, of a tool call or of a model call, and a response event, see
// completeResponse) call it last, once they no longer hold an index into
// agentStack and have reported the failures that they found.
func (inv *invocationTrace) retireLateSettledLocked() {
	if !inv.isLate || len(inv.agentStack) <= 1 {
		return
	}
	hasSettled := func() bool {
		return slices.ContainsFunc(inv.agentStack[1:], func(rec *spanRecord) bool {
			return rec.runDone && rec.contextEnded
		})
	}
	if !hasSettled() {
		return
	}
	// The lineage of a failed activation follows parent even across an
	// activation that already retired, as subtreeLocked does.
	failed := inv.openCallOwnersLocked(nil)
	markFailed := func(idx int) {
		for cur := inv.agentStack[idx]; cur != nil && cur.kind == spanKindAgent; cur = cur.parent {
			failed[cur.spanID] = true
		}
	}
	for j := 1; j < len(inv.agentStack); j++ {
		d := inv.agentStack[j]
		unresponded := (d.runDone || inv.launcherOpenLocked(d)) && !d.responded()
		if failed[d.spanID] || unresponded {
			markFailed(j)
		}
	}
	for _, e := range inv.unrecoveredErrors {
		if e.collateral {
			continue
		}
		if j := inv.activationOfLocked(e); j > 0 {
			markFailed(j)
		}
	}
	// In RunLive the root agent has no span on agentStack, and an agent on
	// agentStack can end before a tool call that it started does, or start one
	// later. In those cases findAgentForContext returns -1 when the call ends, so
	// endToolRunsLocked resolves the owner to the invocation span and reports
	// the failures of the activations right under the invocation span on a
	// node path below that span's, and of the ones under them.
	invSpan := inv.agentStack[0]
	var retired []*spanRecord
	for hasSettled() {
		openAgents := make(map[string]bool, len(inv.agentStack))
		for _, rec := range inv.agentStack[1:] {
			openAgents[rec.spanID] = true
		}
		// heldByCause reports whether agentStack[idx], or an activation that
		// runs under it (see subtreeLocked), carries a failure cause that can
		// still be read through agentStack[idx]: an agent that it runs under
		// and that is still open (following parent across already-retired
		// agents), or—for agentStack[idx] right under the invocation span on a
		// node path below that span's—an open or later tool call whose owner
		// resolves to the invocation span in endToolRunsLocked. Keeping such an
		// activation directly on agentStack (along with each one between it and
		// the reader) preserves the lineage along which failure reporting writes
		// rows, routes later child activations, retries, responses, and errors,
		// and preserves open late span tracking until a tool call reports it or
		// recovery clears the cause. Once nothing able to read it remains, such
		// activations still retire here.
		heldByCause := func(idx int) bool {
			rec := inv.agentStack[idx]
			readable := rec.parentSpanID == invSpan.spanID && isSubNodePath(rec.path, invSpan.path)
			for a := rec.parent; !readable && a != nil && a.kind == spanKindAgent; a = a.parent {
				readable = openAgents[a.spanID]
			}
			if !readable {
				return false
			}
			subtree := inv.subtreeLocked(idx)
			return slices.ContainsFunc(inv.agentStack[idx:], func(d *spanRecord) bool {
				return subtree[d.spanID] && d.openFailureCause() != nil
			})
		}
		changed := false
		// Retiring an activation can release the activations that run under
		// it, which come after it on agentStack, so the scan starts over.
		for idx := len(inv.agentStack) - 1; idx >= 1; idx-- {
			rec := inv.agentStack[idx]
			parentOpen := rec.parent != nil && rec.parent.kind == spanKindAgent && openAgents[rec.parent.spanID]
			if !parentOpen && rec.runDone && rec.contextEnded && !inv.launcherOpenLocked(rec) && !failed[rec.spanID] && !heldByCause(idx) {
				retired = append(retired, rec)
				if rec.parent != nil && rec.parent.kind == spanKindAgent {
					retired = append(retired, rec.parent)
				}
				inv.removeAgentLocked(idx)
				changed = true
				break
			}
		}
		if !changed {
			break
		}
	}
	if len(retired) > 0 {
		inv.dropRetiredLateContextsLocked(retired...)
		inv.syncOpenAgentsLocked()
	}
}

// launcherOpenLocked reports whether a tool call or an agent that is still open
// may have run rec instead of the activation that rec runs under (see
// parentFor), so that rec's run may go on after that activation ended:
//
//   - A tool, a custom agent and a workflow agent can run an agent on another
//     branch, as with agent.InvocationContextDelta.Branch or
//     workflow.WithOverrideBranch, where it runs under the open agent whose
//     branch and node path contain its own, which does not wait for it.
//   - A custom or workflow agent that did not declare its sub-agents in
//     SubAgents and runs multiple sub-agents concurrently can start rec, on its
//     own branch and node path or on a descendant one (as with
//     agent.InvocationContextDelta.Branch or workflow.WithOverrideBranch),
//     while an earlier sibling sub-agent is open on its own or the same
//     descendant branch and node path, causing parentFor to place rec (or an
//     agent between rec and the launcher) under that sibling rather than under
//     the enclosing agent that actually launched and waits for rec. Branch and
//     node path containment does not show which of them launched rec, so rec is
//     held back while an agent between rec and the enclosing agent sits under
//     another agent inside the enclosing agent's branch and node path (other
//     than one of the enclosing agent's own parallel-agent child branches) that
//     did not declare it in SubAgents.
//
// Only a tool call or a non-LLM agent that started before rec (seq < rec.seq)
// and whose own run is not over (!a.runDone) can still be running rec. An
// llmagent runs other agents through its tool calls or sequentially via
// transfer, and an agent that declared rec in its direct parent's children
// ran rec under that direct parent. A tool that functiontool.NewStreaming
// makes runs without tool callbacks, so an agent that it runs is not
// recognized.
func (inv *invocationTrace) launcherOpenLocked(rec *spanRecord) bool {
	for _, call := range inv.toolSpans {
		if call.seq < rec.seq {
			return true
		}
	}
	for _, a := range inv.agentStack[1:] {
		if !a.nonLLM || a.runDone || a.seq >= rec.seq {
			continue
		}
		if !(withinBranch(rec.branch, a.branch) && withinPath(rec.path, a.path, false)) {
			return true
		}
		for anc := rec; anc != nil && anc != a && anc.kind == spanKindAgent && anc.seq > a.seq; anc = anc.parent {
			p := anc.parent
			if p != nil && p != a && p.kind == spanKindAgent &&
				withinBranch(p.branch, a.branch) && withinPath(p.path, a.path, a.name == p.name) &&
				!(p.parent == a && a.isParallel && slices.Contains(a.children, p.name) && p.branch == parallelBranch(a, p.name)) &&
				!slices.Contains(p.children, anc.name) {
				return true
			}
		}
	}
	return false
}

// parallelBranch returns the branch name ADK assigns to childName when run as
// a direct sub-agent of a parallel agent activation a.
func parallelBranch(a *spanRecord, childName string) string {
	if a.branch == "" {
		return a.name + "." + childName
	}
	return a.branch + "." + a.name + "." + childName
}

// nextSeqLocked returns the seq of an agent activation or tool call that starts
// now (see spanRecord.seq).
func (inv *invocationTrace) nextSeqLocked() uint64 {
	inv.lastSeq++
	return inv.lastSeq
}

// sortedErrorKeysLocked returns the keys of inv.unrecoveredErrors ordered by
// (bareKey, seq, key) so that primaryErr and firstErr selection is
// deterministic and independent of the %p pointer prefix in callKey.
func (inv *invocationTrace) sortedErrorKeysLocked() []string {
	keys := slices.SortedFunc(maps.Keys(inv.unrecoveredErrors), func(a, b string) int {
		ea, eb := inv.unrecoveredErrors[a], inv.unrecoveredErrors[b]
		return cmp.Or(
			cmp.Compare(ea.bareKey, eb.bareKey),
			cmp.Compare(ea.seq, eb.seq),
			cmp.Compare(a, b),
		)
	})
	return keys
}

// endedFunc returns a function that checks whether the InvocationContext of an
// agent activation started with ctx was marked ended. Agent callbacks receive
// a callbackContextWrapper whose Ended method logs a warning and returns
// false, whereas WithICDelta(nil) unwraps the underlying commonContext sharing
// the activation's invocationContext without allocating a copy.
func endedFunc(ctx context.Context) func() bool {
	if ctx == nil {
		return nil
	}
	if c, ok := ctx.(interface {
		WithICDelta(*agent.InvocationContextDelta) agent.InvocationContext
	}); ok {
		if ic := orZero(func() agent.InvocationContext { return c.WithICDelta(nil) }); ic != nil {
			return func() bool { return orZero(ic.Ended) }
		}
	}
	if ic, ok := ctx.(interface{ Ended() bool }); ok {
		return func() bool { return orZero(ic.Ended) }
	}
	return nil
}

// maxActivationKeyDepth bounds the InvocationContext chain that activationKey
// follows.
const maxActivationKeyDepth = 8

// activationKey returns the per-activation context pointer that ADK created for
// the agent run that ctx belongs to, or nil if ctx does not show it. Each
// agent.Run allocates a fresh run context via ctx.WithContext(spanCtx) and
// stores it in the base InvocationContext's embedded Context field; llmagent.run
// then derives a second base InvocationContext via WithICDelta that copies the
// same embedded Context pointer. Unwrapping to the innermost InvocationContext
// and returning its embedded InvocationContext pointer when present gives an
// identical key across BeforeAgentCallback, model and tool callbacks, and
// AfterAgentCallback of one run while keeping concurrent runs of the same agent
// on one branch and node path distinct. An agent callback receives a
// callbackContextWrapper whose InvocationContext method logs a warning and
// returns nil, so the chain starts at WithICDelta(nil).
func activationKey(ctx context.Context) any {
	c, ok := ctx.(interface {
		WithICDelta(*agent.InvocationContextDelta) agent.InvocationContext
	})
	if !ok {
		return nil
	}
	return orZero(func() any {
		var key any
		var cur any = c.WithICDelta(nil)
		for steps := 0; cur != nil && steps < maxActivationKeyDepth; steps++ {
			if v := reflect.ValueOf(cur); v.Kind() != reflect.Pointer || v.IsNil() {
				break
			}
			key = cur
			next, ok := cur.(interface {
				InvocationContext() agent.InvocationContext
			})
			if !ok {
				break
			}
			inner := next.InvocationContext()
			if inner == nil || any(inner) == cur {
				break
			}
			cur = inner
		}
		if key != nil {
			if v := reflect.ValueOf(key); v.Kind() == reflect.Pointer && !v.IsNil() && v.Elem().Kind() == reflect.Struct {
				elem := v.Elem()
				if isADKType(elem.Type()) {
					if f := directStructField(elem, "Context"); f.IsValid() && f.CanInterface() {
						if inner, ok := reflect.TypeAssert[agent.InvocationContext](f); ok && inner != nil {
							if iv := reflect.ValueOf(inner); iv.Kind() == reflect.Pointer && !iv.IsNil() {
								return inner
							}
						}
					}
				}
			}
		}
		return key
	})
}

// isActivationOf reports whether rec is the span of the agent activation
// whose run ctx belongs to (see activationKey).
func (rec *spanRecord) isActivationOf(ctx context.Context) bool {
	return rec.kind == spanKindAgent && rec.activation != nil && rec.activation == activationKey(ctx)
}

// findAgentForContext returns the index in agentStack of the open span of the
// agent name on branch that the agent run of ctx started (see activationKey),
// or else, if no open span of it carries the key, as findAgent does. Concurrent
// runs of the same agent on one branch and node path are then told apart by
// the run that each callback belongs to rather than by which started last.
func (inv *invocationTrace) findAgentForContext(ctx context.Context, path, name, branch string) int {
	if act := activationKey(ctx); act != nil {
		for i := len(inv.agentStack) - 1; i > 0; i-- {
			rec := inv.agentStack[i]
			if rec.kind == spanKindAgent && rec.activation == act && rec.name == name && rec.branch == branch {
				return i
			}
		}
	}
	return inv.findAgent(path, name, branch)
}

// currentGoroutineID returns the runtime goroutine ID of the caller, or 0 if
// unavailable.
func currentGoroutineID() uint64 {
	var buf [64]byte
	n := runtime.Stack(buf[:], false)
	const prefix = "goroutine "
	if n <= len(prefix) || string(buf[:len(prefix)]) != prefix {
		return 0
	}
	var id uint64
	for _, b := range buf[len(prefix):n] {
		if b == ' ' {
			return id
		}
		if b < '0' || b > '9' {
			return 0
		}
		id = id*10 + uint64(b-'0')
	}
	return 0
}

// goroutineOrigin names the function that created a goroutine and the
// goroutine that ran that function, as the "created by F in goroutine N" line
// of runtime.Stack shows them.
type goroutineOrigin struct {
	creator string
	parent  uint64
}

// startedBy reports whether creator created the goroutine that o describes
// while it ran on the goroutine on which the activation p started, as the
// workflow scheduler of a workflow agent does for each node that it starts
// (see schedulerCreator) and a ParallelAgent does for each branch (see
// errgroupCreator). Goroutine IDs are never reused within a process.
func (o goroutineOrigin) startedBy(creator string, p *spanRecord) bool {
	return p != nil && p.goroutineID != 0 && o.parent == p.goroutineID && o.creator == creator
}

// attemptSite tells whether an agent activation is the agent of a workflow
// node attempt that runs on the goroutine on which the activation started,
// as the stack trace of its BeforeAgentCallback shows (see
// attemptSiteOfTrace).
type attemptSite uint8

const (
	// attemptNone is any other activation: the agent that the runner runs,
	// an agent that a function node, a custom agent, a built-in agent or a
	// tool runs, one that runs on the goroutine of an iterator that iter.Pull
	// made, and one whose stack trace does not show its goroutine in full.
	attemptNone attemptSite = iota
	// attemptNode is the agent of an agent node that the workflow scheduler
	// runs on the goroutine that it started for the node run (go runNode in
	// scheduler.startNode).
	attemptNode
	// attemptWorker is the agent of an agent node that a
	// workflow.ParallelWorker runs for one of its items on the goroutine that
	// it started for the item (go runWorker in ParallelWorker.Run), where it
	// runs the node again when RetryConfig retries the item.
	attemptWorker
)

// attemptSiteOfTrace returns the attempt site (see attemptSite) of the agent
// activation whose BeforeAgentCallback runs on the goroutine whose stack trace
// runtime.Stack(buf, false) wrote, complete reporting whether the trace fit in
// buf. The trace shows a node attempt only when the goroutine runs nothing
// but that attempt: from the bottom up, the function that the go statement
// started (workflow.runNode, or ParallelWorker.runWorker with its
// runWrappedOnce), the AgentNode's Run, any frames of ADK's own packages that
// adapt the agent to the node (as llmagent's RunLLMAgentAsNode does), the
// agent's Run, and the runBeforeAgentCallbacks call that runs the callback.
// An agent's Run appears exactly once in the trace, so no earlier agent run on
// the goroutine is suspended below the activation in a yield, as one is when a
// function node or a custom agent runs an agent inside its loop over the
// events of another run. attemptNone is returned for anything else, including
// a trace that is not complete or that the runtime elided frames from.
func attemptSiteOfTrace(trace []byte, complete bool) attemptSite {
	if !complete {
		return attemptNone
	}
	_, frames, ok := strings.Cut(string(trace), "\n")
	if !ok {
		return attemptNone
	}
	// The trace lists the frames from the innermost one down, each as a
	// "function(arguments)" line followed by a tab-indented "file:line" one,
	// and ends with a "created by" line for the goroutine.
	var names []string
	if !visitTraceFrameNames(frames, func(name string) {
		names = append(names, name)
	}) {
		return attemptNone
	}
	agentRun := agentPkgPath + ".(*agent).Run"
	i := len(names) - 1
	// At higher GOTRACEBACK levels the trace also shows the runtime's frames,
	// such as runtime.goexit at the bottom, and the wrapper function that a
	// go statement with arguments may start.
	for i >= 0 && strings.HasPrefix(names[i], "runtime.") {
		i--
	}
	if i >= 0 && (strings.HasPrefix(names[i], workflowPkgPath+".(*scheduler).startNode.") ||
		strings.HasPrefix(names[i], workflowPkgPath+".(*ParallelWorker).Run.")) {
		i--
	}
	if i < 0 {
		return attemptNone
	}
	var site attemptSite
	switch names[i] {
	case workflowPkgPath + ".runNode":
		site = attemptNode
	case workflowPkgPath + ".(*ParallelWorker).runWorker":
		site = attemptWorker
		if i > 0 && names[i-1] == workflowPkgPath+".(*ParallelWorker).runWrappedOnce" {
			i--
		}
	default:
		return attemptNone
	}
	i--
	if i < 0 || !isFrameOf(names[i], workflowPkgPath+".(*AgentNode).Run") {
		return attemptNone
	}
	i--
	for i >= 0 && strings.HasPrefix(names[i], adkPkgPrefix) && !isFrameOf(names[i], agentRun) {
		i--
	}
	if i < 1 || !isFrameOf(names[i], agentRun) || names[i-1] != agentPkgPath+".runBeforeAgentCallbacks" {
		return attemptNone
	}
	for j, name := range names {
		if j != i && isFrameOf(name, agentRun) {
			return attemptNone
		}
	}
	return site
}

// isFrameOf reports whether name, a function name as runtime.Stack prints it,
// names fn or a function literal, range-over-func body or method value
// wrapper inside it, as "fn.func1", "fn.func1-range1" and "fn-fm" do.
func isFrameOf(name, fn string) bool {
	rest, ok := strings.CutPrefix(name, fn)
	return ok && (rest == "" || rest[0] == '.' || rest[0] == '-')
}

// maxGoroutineTraceSize bounds the buffer that currentGoroutine reads the
// caller's stack trace into. The runtime prints at most 100 frames of a
// goroutine, so the trace fits well within it.
const maxGoroutineTraceSize = 1 << 20

// currentGoroutine returns the runtime goroutine ID of the caller, the origin
// of its goroutine (see goroutineOrigin) and, for a caller that runs in a
// BeforeAgentCallback, the attempt site of the agent activation that starts
// (see attemptSiteOfTrace), or zero values for what the stack trace does not
// show.
//
// runtime.Stack is called directly here rather than through
// currentGoroutineTrace so that no extra helper frame pushes a stack right at
// Go's 100-frame traceback limit into "...frames elided...".
func currentGoroutine() (uint64, goroutineOrigin, attemptSite) {
	buf := make([]byte, 8<<10)
	var n int
	for {
		n = runtime.Stack(buf, false)
		if n < len(buf) || len(buf) >= maxGoroutineTraceSize {
			break
		}
		buf = make([]byte, 2*len(buf))
	}
	trace, complete := buf[:n], n < len(buf)
	id, origin := parseGoroutineTrace(trace, complete)
	return id, origin, attemptSiteOfTrace(trace, complete)
}

// currentGoroutineTrace returns the stack trace of the caller's goroutine as
// runtime.Stack(buf, false) writes it, and whether it fit in the buffer.
func currentGoroutineTrace() (trace []byte, complete bool) {
	buf := make([]byte, 8<<10)
	for {
		n := runtime.Stack(buf, false)
		if n < len(buf) || len(buf) >= maxGoroutineTraceSize {
			return buf[:n], n < len(buf)
		}
		buf = make([]byte, 2*len(buf))
	}
}

// visitTraceFrameNames parses frames written by runtime.Stack (lines after the
// goroutine header) up to "created by ", calling visit with each frame's
// function name. It returns false if frames were elided by the runtime or a
// frame line is malformed.
func visitTraceFrameNames(frames string, visit func(name string)) bool {
	for line := range strings.SplitSeq(frames, "\n") {
		if line == "" || line[0] == '\t' {
			continue
		}
		if strings.HasPrefix(line, "created by ") {
			break
		}
		if strings.HasPrefix(line, "...") {
			return false
		}
		paren := strings.LastIndexByte(line, '(')
		if paren <= 0 {
			return false
		}
		visit(line[:paren])
	}
	return true
}

// isGoroutineTraceHeader reports whether header is the line with which
// runtime.Stack starts the trace of a goroutine:
//
//	goroutine 7 [chan receive, 2 minutes]:
//	goroutine 7 gp=0xc000002380 m=0 mp=0x5b7e60 [running]:
//	goroutine 7 [running labels:{"k": "v"}]:
//	goroutine 7 [running] {k: "v w"}:
//
// The second form is that of GOTRACEBACK=system and above. The last two show
// pprof labels: Go 1.26 prints them inside the bracket when
// GODEBUG=tracebacklabels=1, and Go 1.27 after it, by default in a main module
// that declares go 1.27 or later. A label key or value can hold brackets and
// braces, which the runtime then quotes (Go 1.26 quotes every one), so labels
// are read as quoted strings rather than searched for delimiters.
func isGoroutineTraceHeader(header string) bool {
	rest, ok := strings.CutPrefix(header, "goroutine ")
	if !ok {
		return false
	}
	// A header with no space after the ID has no bracket either, and so
	// fails below, as does one with no bracket.
	idText, rest, _ := strings.Cut(rest, " ")
	if _, err := strconv.ParseUint(idText, 10, 64); err != nil {
		return false
	}
	fields, rest, _ := strings.Cut(rest, "[")
	if (fields != "" && !strings.HasSuffix(fields, " ")) || strings.ContainsAny(fields, `]{}"`) {
		return false
	}
	// The status, such as "chan receive, 2 minutes", runs up to the closing
	// bracket, or up to the labels that Go 1.26 prints inside it.
	end := strings.IndexAny(rest, `[]{}"`)
	if end <= 0 {
		return false
	}
	if status, ok := strings.CutSuffix(rest[:end], " labels:"); ok {
		return status != "" && cutTracebackLabels(rest[end:], true) == "]:"
	}
	if labels, ok := strings.CutPrefix(rest[end:], "] "); ok {
		return cutTracebackLabels(labels, false) == ":"
	}
	return rest[end:] == "]:"
}

// cutTracebackLabels returns what follows the pprof labels that s starts with
// in a goroutine header (see isGoroutineTraceHeader), "{" and "}" around
// "key: value" pairs that ", " separates, or "" if s does not start with
// labels. quoted requires every key and value to be quoted, as in Go 1.26.
func cutTracebackLabels(s string, quoted bool) string {
	s, ok := strings.CutPrefix(s, "{")
	if !ok {
		return ""
	}
	for {
		if s, ok = strings.CutPrefix(cutTracebackLabelString(s, quoted), ": "); !ok {
			return ""
		}
		s = cutTracebackLabelString(s, quoted)
		if rest, ok := strings.CutPrefix(s, "}"); ok {
			return rest
		}
		if s, ok = strings.CutPrefix(s, ", "); !ok {
			return ""
		}
	}
}

// cutTracebackLabelString returns what follows the label key or value that s
// starts with: a string in double quotes, in which a backslash escapes the
// character after it, or, unless quoted is true, a possibly empty run of the
// letters, digits, '.', '/' and '_' that Go 1.27 leaves unquoted. It returns
// "" if s starts with a quote that is not closed, or with no quote although
// quoted is true.
func cutTracebackLabelString(s string, quoted bool) string {
	if s, ok := strings.CutPrefix(s, `"`); ok {
		for i := 0; i < len(s); i++ {
			switch s[i] {
			case '\\':
				i++
			case '"':
				return s[i+1:]
			}
		}
		return ""
	}
	if quoted {
		return ""
	}
	return strings.TrimLeftFunc(s, func(r rune) bool {
		return 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z' || '0' <= r && r <= '9' || r == '.' || r == '/' || r == '_'
	})
}

// afterAgentCallbackRunsOfTrace returns how many runs of after-agent
// callbacks the goroutine whose stack trace runtime.Stack(buf, false) wrote is
// in, complete reporting whether the trace fit in buf: the frames of
// agent.runAfterAgentCallbacks, which runs an activation's plugin and agent
// after-agent callbacks on the goroutine that ran the activation until all
// callbacks finish or one returns non-nil content or an error (skipping later
// callbacks, including this plugin's AfterAgentCallback when an earlier plugin
// returns content or an error). An agent run started synchronously on that
// same goroutine (such as a custom agent.New runner) runs inside that frame,
// whereas a runner-driven llmagent executes as a workflow node on its own
// scheduler goroutine and therefore sees 0 outer runAfterAgentCallbacks frames
// during BeforeAgentCallbacks/Run and 1 (its own) during its
// AfterAgentCallback rather than 2. The second result is false when the trace
// is not a valid goroutine stack trace (including "goroutine N [...]:" or
// "goroutine N [...] {...}:"), is incomplete, or had frames elided by the
// runtime.
func afterAgentCallbackRunsOfTrace(trace []byte, complete bool) (int, bool) {
	if !complete {
		return 0, false
	}
	header, frames, ok := strings.Cut(string(trace), "\n")
	if !ok || !isGoroutineTraceHeader(header) {
		return 0, false
	}
	fn := agentPkgPath + ".runAfterAgentCallbacks"
	runs := 0
	if !visitTraceFrameNames(frames, func(name string) {
		if name == fn {
			runs++
		}
	}) {
		return 0, false
	}
	return runs, true
}

// parseGoroutineTrace parses the stack trace of one goroutine as
// runtime.Stack(buf, false) writes it: the goroutine ID from the header
// ("goroutine N [...]:", with more fields before the bracket at higher
// GOTRACEBACK levels) and the origin from the first "created by F in
// goroutine M" line. Later "created by" lines, which GODEBUG's
// tracebackancestors setting adds, describe ancestor goroutines. The origin
// is zero unless the trace is complete.
func parseGoroutineTrace(trace []byte, complete bool) (uint64, goroutineOrigin) {
	s := string(trace)
	rest, ok := strings.CutPrefix(s, "goroutine ")
	if !ok {
		return 0, goroutineOrigin{}
	}
	idText, _, ok := strings.Cut(rest, " ")
	if !ok {
		return 0, goroutineOrigin{}
	}
	id, err := strconv.ParseUint(idText, 10, 64)
	if err != nil {
		return 0, goroutineOrigin{}
	}
	if !complete {
		return id, goroutineOrigin{}
	}
	const createdBy = "\ncreated by "
	_, after, ok := strings.Cut(s, createdBy)
	if !ok {
		return id, goroutineOrigin{}
	}
	line, _, _ := strings.Cut(after, "\n")
	const in = " in goroutine "
	sep := strings.LastIndex(line, in)
	if sep <= 0 {
		return id, goroutineOrigin{}
	}
	parent, err := strconv.ParseUint(line[sep+len(in):], 10, 64)
	if err != nil {
		return id, goroutineOrigin{}
	}
	return id, goroutineOrigin{creator: line[:sep], parent: parent}
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
// an open model or tool call other than one for which ignore, if not nil,
// reports true. A model call whose agent emitted a partial session event
// while the call was open (see spanRecord.streamed) is excluded, because a
// caller that stops reading mid-stream leaves the streaming call open
// without failing.
func (inv *invocationTrace) openCallOwnersLocked(ignore func(*spanRecord) bool) map[string]bool {
	owners := map[string]bool{}
	for _, rec := range inv.llmSpans {
		if !rec.streamed && (ignore == nil || !ignore(rec)) {
			owners[rec.parentSpanID] = true
		}
	}
	for _, rec := range inv.toolSpans {
		if ignore == nil || !ignore(rec) {
			owners[rec.parentSpanID] = true
		}
	}
	return owners
}

// agentWithOpenCallLocked returns the index in agentStack of the most recent
// open agent span that an open model or tool call runs under, or -1. A call
// for which ignore, if not nil, reports true does not count.
func (inv *invocationTrace) agentWithOpenCallLocked(ignore func(*spanRecord) bool) int {
	owners := inv.openCallOwnersLocked(ignore)
	for i := len(inv.agentStack) - 1; i > 0; i-- {
		if owners[inv.agentStack[i].spanID] {
			return i
		}
	}
	return -1
}

// hasCanceledByStopUnderLocked reports whether canceledByStop holds for an
// open activation in the subtree of agentStack[i] (see subtreeLocked) or for
// an open model or tool call of one of them.
func (inv *invocationTrace) hasCanceledByStopUnderLocked(i int, canceledByStop func(*spanRecord) bool) bool {
	subtree := inv.subtreeLocked(i)
	for _, rec := range inv.agentStack[i:] {
		if subtree[rec.spanID] && canceledByStop(rec) {
			return true
		}
	}
	for _, calls := range []map[string]*spanRecord{inv.llmSpans, inv.toolSpans} {
		for _, call := range calls {
			if subtree[call.parentSpanID] && canceledByStop(call) {
				return true
			}
		}
	}
	return false
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
	if !inv.lockLive() {
		return tm.pushLLMSpan(ctx, id)
	}
	defer inv.mu.Unlock()
	inv.started = true
	inv.noteRootAgentLocked(id, rootIn, true)
	key := callKey(ctx, llmKey(id))
	owner := inv.ownerOf(ctx, id)
	inv.clearLastPartialUnderLocked(owner)
	if owner.kind == spanKindAgent {
		inv.touchAgentActivityLocked(owner, now)
		// A new model call clears stale pending-event credits from earlier
		// tool-response or model-response events that the agent's parent
		// consumed without forwarding, so a peer cannot use them to claim
		// this activation's final response event (see selectResponseOwnerLocked).
		owner.awaitingModelEvent = 0
		owner.awaitingToolEvent = 0
	}
	rec := inv.newChild(owner, spanKindLLM, now)
	rec.canceled = canceledFunc(ctx)
	rec.callStarted = true
	rec.ownActivation = owner.isActivationOf(ctx)
	inv.llmSpans[key] = rec
	inv.syncOpenCallsLocked()
	delete(inv.completedModels, key)
	span := inv.rowSpan(rec, nil, ambient)
	span.timestamp = now
	return span
}

// recordFirstToken records the arrival time of the first streamed chunk of
// generated content of the running model call of an agent. It does not mark
// the call as streaming: an agent's AfterModelCallback may still reject the
// chunk, so only the partial session event that ADK emits once the chunk is
// accepted does (see recordStreamed).
func (tm *traceManager) recordFirstToken(ctx context.Context, id identity) {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	for !inv.lockLive() {
		inv = tm.invocation(ctx, id)
	}
	defer inv.mu.Unlock()
	if rec := inv.llmSpans[callKey(ctx, llmKey(id))]; rec != nil && rec.firstToken.IsZero() {
		rec.firstToken = now
		if owner := inv.ownerOf(ctx, id); owner.kind == spanKindAgent {
			inv.touchAgentActivityLocked(owner, now)
		}
	}
}

// recordStreamed records a partial session event of author: it marks the
// running model call of author (if any) as streaming and adds the owning agent
// lineage to partialLineage, and, if the event has content or a typed output,
// it marks that lineage as having responded. It sets lastSessionEventPartial
// and lastEventPartial. A partial event without content, such as a progress
// event, is not a response: it only keeps a caller that stops reading on it
// from being taken for a failure. The node path of the event can differ from
// the path of the context of the model callbacks, as for the agent that
// runner.Run runs, so the call is also found as one that the author's
// activation owns.
func (tm *traceManager) recordStreamed(ctx context.Context, id identity, author, branch string, withContent bool) {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.lastSessionEventPartial = true
	inv.lastEventPartial = true
	inv.lastPartialSpanID = ""
	callID := identity{agentName: author, branch: branch, path: id.path}
	if rec := inv.llmSpans[llmKey(callID)]; rec != nil {
		rec.streamed = true
	}
	if i := inv.findAgent(id.path, author, branch); i > 0 {
		rec := inv.agentStack[i]
		for _, llm := range inv.llmSpans {
			if llm.parentSpanID == rec.spanID {
				llm.streamed = true
			}
		}
		inv.lastPartialSpanID = rec.spanID
		inv.touchAgentActivityLocked(rec, now)
		lineage := inv.lineageLocked(i)
		if inv.partialLineage == nil {
			inv.partialLineage = make(map[string]bool, len(lineage))
		}
		maps.Copy(inv.partialLineage, lineage)
		if withContent {
			inv.markRespondedLocked(lineage)
		}
	}
}

// clearLastPartial clears the lastEventPartial flag when a non-partial session
// event without content or a typed output arrives.
func (tm *traceManager) clearLastPartial(ctx context.Context, id identity) {
	inv := tm.invocation(ctx, id)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.clearLastPartialLocked()
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
	if !inv.lockLive() {
		return tm.popLLMSpan(ctx, id, responded)
	}
	defer inv.mu.Unlock()
	// The ended call, including a recovered call completed from completedModels,
	// no longer keeps its agent or the agents above it from retiring in the
	// late state (see retireLateSettledLocked).
	defer inv.retireLateSettledLocked()
	inv.noteRootAgentLocked(id, rootIn, true)
	key := callKey(ctx, llmKey(id))
	rec := inv.llmSpans[key]
	if rec == nil {
		if responded {
			if prev := inv.completedModels[key]; prev != nil {
				delete(inv.completedModels, key)
				if owner := inv.ownerOf(ctx, id); owner.kind == spanKindAgent {
					owner.hasResponse = true
					owner.modelResponded = true
					owner.awaitingModelEvent++
				}
				span := inv.rowSpan(prev, nil, ambient)
				span.timestamp = now
				return span
			}
		}
		span := inv.rowSpan(inv.newChild(inv.ownerOf(ctx, id), spanKindLLM, now), nil, ambient)
		span.timestamp = now
		return span
	}
	delete(inv.llmSpans, key)
	inv.syncOpenCallsLocked()
	rec.dropContexts()
	if owner := inv.ownerOf(ctx, id); owner.kind == spanKindAgent {
		inv.touchAgentActivityLocked(owner, now)
	}
	if !responded {
		// A model call that did not respond leaves no pending model-event
		// credit: clear stale credits so a peer's response cannot claim
		// this activation's events (see selectResponseOwnerLocked).
		inv.completedModels[key] = rec
		if owner := inv.ownerOf(ctx, id); owner.kind == spanKindAgent {
			owner.awaitingModelEvent = 0
		}
	} else {
		if owner := inv.ownerOf(ctx, id); owner.kind == spanKindAgent {
			owner.hasResponse = true
			owner.modelResponded = true
			owner.awaitingModelEvent++
		}
		if rec.firstToken.IsZero() {
			rec.firstToken = now
		}
	}
	span := inv.rowSpan(rec, latencyOf(rec, now), ambient)
	span.timestamp = now
	return span
}

// pushToolSpan starts the span of a tool call and returns the place of its
// TOOL_STARTING row.
func (tm *traceManager) pushToolSpan(ctx context.Context, id identity, key string) rowSpan {
	inv := tm.lookupInvocation(ctx, id, false, true)
	defer inv.pendingSpans.Add(-1)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	if !inv.lockLive() {
		return tm.pushToolSpan(ctx, id, key)
	}
	defer inv.mu.Unlock()
	inv.started = true
	inv.noteRootAgentLocked(id, rootIn, true)
	key = callKey(ctx, key)
	owner := inv.ownerOf(ctx, id)
	inv.clearLastPartialUnderLocked(owner)
	if owner.kind == spanKindAgent {
		inv.touchAgentActivityLocked(owner, now)
		// A new tool call clears stale model-event credits from earlier
		// model-response events that the agent's parent consumed without
		// forwarding, so a peer cannot use them to claim a later event. It
		// keeps awaitingToolEvent: ADK runs the tool calls of one model
		// response concurrently by default, so another call of the same
		// response may already have ended, and the merged function-response
		// event of the response still needs that call's credit.
		owner.awaitingModelEvent = 0
	}
	rec := inv.newChild(owner, spanKindTool, now)
	rec.canceled = canceledFunc(ctx)
	rec.seq = inv.nextSeqLocked()
	rec.callStarted = true
	rec.ownActivation = owner.isActivationOf(ctx)
	inv.toolSpans[key] = rec
	inv.syncOpenCallsLocked()
	delete(inv.completedTools, key)
	span := inv.rowSpan(rec, nil, ambient)
	span.timestamp = now
	return span
}

// popToolSpan ends the span of a tool call and returns the place of its
// TOOL_COMPLETED or, with fromError set, TOOL_ERROR row.
//
// ADK runs AfterToolCallback after OnToolErrorCallback for a failed call.
// closedByError reports that the OnToolErrorCallback of the same attempt
// (see toolAttempt) already ended the call, in which case the returned
// place reuses its span, without latency. If the call is unknown, as for a
// call of a tool that does not exist, the row gets a span of its own and no
// latency; the agent and invocation spans are left alone. Once the agent
// has no tool call left open, the activations that its tool calls ran and
// that never ended are ended—purged if they failed, or retired quietly
// otherwise (see endToolRunsLocked).
func (tm *traceManager) popToolSpan(ctx context.Context, id identity, key string, fromError bool) (span rowSpan, purged []unclosedAgentSpan, closedByError bool) {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	actions := toolActionsOf(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	if !inv.lockLive() {
		return tm.popToolSpan(ctx, id, key, fromError)
	}
	defer inv.mu.Unlock()
	inv.noteRootAgentLocked(id, rootIn, true)
	key = callKey(ctx, key)
	open := inv.toolSpans[key]
	delete(inv.toolSpans, key)
	inv.syncOpenCallsLocked()
	if open != nil {
		inv.markToolRunsDoneLocked(open)
	}
	if !fromError {
		if i := inv.findAgentForContext(ctx, id.path, id.agentName, id.branch); i > 0 {
			inv.agentStack[i].awaitingToolEvent++
		}
	}
	if open != nil && open.parent != nil && open.parent.kind == spanKindAgent {
		inv.touchAgentActivityLocked(open.parent, now)
	}
	since := noFrame
	if open != nil {
		since = open.seq
	}
	purged = inv.endToolRunsLocked(ctx, id, inv.lastSessionEventPartial, now, ambient, since)
	// The ended tool call may have run open activations under other spans.
	inv.settleFinishedLocked()
	inv.retireLateSettledLocked()
	if open != nil {
		open.dropContexts()
		if fromError {
			open.errorAttempt = attemptOf(actions)
			inv.completedTools[key] = open
		}
		span = inv.rowSpan(open, latencyOf(open, now), ambient)
		span.timestamp = now
		return span, purged, false
	}
	if !fromError {
		if rec := inv.completedTools[key]; rec != nil && rec.errorAttempt == attemptOf(actions) {
			delete(inv.completedTools, key)
			span = inv.rowSpan(rec, nil, ambient)
			span.timestamp = now
			return span, purged, true
		}
	}
	rec := inv.newChild(inv.ownerOf(ctx, id), spanKindTool, now)
	if fromError {
		rec.dropContexts()
		rec.errorAttempt = attemptOf(actions)
		inv.completedTools[key] = rec
	}
	span = inv.rowSpan(rec, nil, ambient)
	span.timestamp = now
	return span, purged, false
}

// eventSpan returns the place of a row that describes a session event: the
// span of the agent that wrote the event, if it is running, and otherwise
// the invocation span. Like other invocation callbacks, the event callback
// is called with the agent that the runner runs, whose name is recorded as
// the root agent.
func (tm *traceManager) eventSpan(ctx context.Context, id identity, author, branch string) rowSpan {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgentLocked(id, rootIn, false)
	rec := inv.agentStack[0]
	if i := inv.findAgent(id.path, author, branch); i > 0 {
		rec = inv.agentStack[i]
		inv.touchAgentActivityLocked(rec, now)
	}
	span := inv.rowSpan(rec, nil, ambient)
	span.timestamp = now
	return span
}

// newChild returns a new span under parent. It is not tracked.
func (inv *invocationTrace) newChild(parent *spanRecord, kind spanKind, now time.Time) *spanRecord {
	return &spanRecord{
		spanID:       newSpanID(),
		parentSpanID: parent.spanID,
		kind:         kind,
		start:        now,
		parent:       parent,
		stopPoint:    parent.kind == spanKindInvocation,
	}
}

// lockLive locks inv for a callback that looked inv up and reports true,
// unless the run of inv finished in between: the run's calls then are in the
// late state of its completed entry, where the callbacks that follow look for
// them, so lockLive leaves inv unlocked and reports false, and the callback
// looks the invocation up again.
func (inv *invocationTrace) lockLive() bool {
	inv.mu.Lock()
	if inv.finished {
		inv.mu.Unlock()
		return false
	}
	return true
}

// canceledFunc returns the spanRecord.canceled function of a span that a
// callback with ctx starts. It reports canceled == true only when ctx was
// canceled with the sentinel context.Canceled as its cause, as when a
// consumer stop cancels through a plain CancelFunc. When ctx ended with any
// other cause, such as a sibling branch's error in a parallel agent's
// errgroup (errgroup.WithContext) or context.DeadlineExceeded, it reports
// canceled == false and returns that cause.
func canceledFunc(ctx context.Context) func() (canceled bool, cause error) {
	if ctx == nil {
		return nil
	}
	return func() (bool, error) {
		err := orZero(ctx.Err)
		if err == nil {
			return false, nil
		}
		cause := orZero(func() error { return context.Cause(ctx) })
		if errors.Is(err, context.Canceled) && cause == context.Canceled {
			return true, nil
		}
		return false, cmp.Or(cause, err)
	}
}

// wasCanceled reports whether the context of the callback that started rec
// was canceled with cause context.Canceled.
func (rec *spanRecord) wasCanceled() bool {
	if rec.canceled == nil {
		return false
	}
	c, _ := rec.canceled()
	return c
}

// failureCause returns the non-context.Canceled cause with which the context
// of the callback that started rec ended, or nil if the context is still live
// or was canceled with cause context.Canceled.
func (rec *spanRecord) failureCause() error {
	if rec.canceled == nil {
		return nil
	}
	_, cause := rec.canceled()
	return cause
}

// rawContextFailureCause returns the failureCause of rec's own context when it
// is not bare context.DeadlineExceeded. ADK's workflow scheduler creates
// NodeConfig.Timeout contexts with WithAgentTimeout and deletes their cancel
// func on node completion without calling it, so a finished node whose
// BeforeAgentCallback answered leaves an open shortcut span whose context later
// reaches bare context.DeadlineExceeded; a parallel agent's errgroup instead
// wraps every branch error, which is never equal to context.DeadlineExceeded.
func (rec *spanRecord) rawContextFailureCause() error {
	if cause := rec.failureCause(); cause != nil && cause != context.DeadlineExceeded {
		return cause
	}
	return nil
}

// isParallelBranch reports whether rec is the span of a branch of a
// ParallelAgent. In direct traceManager tests without goroutine information
// (goroutineID == 0), any agent span on a branch below its parent's branch
// counts as a parallel branch.
func (rec *spanRecord) isParallelBranch() bool {
	if rec.kind != spanKindAgent || rec.parent == nil {
		return false
	}
	if rec.goroutineID == 0 {
		return rec.branch != rec.parent.branch
	}
	p := rec.parent
	return p.isParallel &&
		rec.origin.startedBy(errgroupCreator, p) &&
		slices.Contains(p.children, rec.name) &&
		rec.branch == parallelBranch(p, rec.name)
}

// contextFailureCause returns the rawContextFailureCause of rec's own context
// scoped to a parallel agent branch. When rec's context ended with a
// qualifying cause (neither context.Canceled nor bare
// context.DeadlineExceeded), it walks up rec's ancestors to the outermost
// span sharing such a context cause and returns that cause only when that
// outermost span is a parallel branch (top.isParallelBranch). Because
// top.isParallelBranch is false for custom agents and their sub-agents when a
// custom agent cancels its own context.WithCancelCause(ic), any cause
// recorded on a custom agent's WithCancelCause is ignored by
// contextFailureCause (and is therefore not promoted into retiredCauses by
// retireAgentLocked either), whether the custom agent handles the sub-agent's
// failure or propagates it after the sub-agent already emitted content, and
// including when a ParallelAgent runs under such a canceled outer context.
func (rec *spanRecord) contextFailureCause() error {
	cause := rec.rawContextFailureCause()
	if cause == nil {
		return nil
	}
	top := rec
	for top.parent != nil && top.parent.rawContextFailureCause() != nil {
		top = top.parent
	}
	if !top.isParallelBranch() {
		return nil
	}
	return cause
}

// openFailureCause returns rec's contextFailureCause, or else the first of
// rec.retiredCauses.
func (rec *spanRecord) openFailureCause() error {
	return cmp.Or(rec.contextFailureCause(), rec.firstRetiredCause())
}

// openFailureCauseLocked returns the first non-nil openFailureCause across the
// invocation's agentStack (most recent first, including retiredCauses on the
// invocation root at index 0), or nil if none ended with a qualifying cause.
// Open model and tool call contexts derive from their owner activation's
// context, so scanning agentStack covers them too.
func (inv *invocationTrace) openFailureCauseLocked() error {
	for _, v := range slices.Backward(inv.agentStack) {
		if cause := v.openFailureCause(); cause != nil {
			return cause
		}
	}
	return nil
}

// canceledByStop reports whether the context of the callback that started
// rec was canceled where the run cancels work when its caller stops reading.
// Two checks separate a consumer stop from failures that cancel a context:
//
//   - First, rec.wasCanceled (via canceledFunc) requires context.Cause(ctx)
//     to be context.Canceled, separating a consumer stop's plain CancelFunc
//     from a parallel agent's errgroup canceling sibling branches with a
//     failing branch's error as the cause.
//   - Second, walking rec.parent to the outermost span whose context was
//     canceled in that way checks top.stopPoint, separating a cancellation
//     that started at a workflow node, runner root or parallel branch from
//     an agent, a callback or a tool canceling a context of its own (such as
//     a custom agent canceling a sub-agent's context on a timeout, even if
//     the parent it canceled the context under has already ended).
func (rec *spanRecord) canceledByStop() bool {
	if !rec.wasCanceled() {
		return false
	}
	top := rec
	for top.parent != nil && top.parent.wasCanceled() {
		top = top.parent
	}
	return top.stopPoint
}

// collateralFailureCause checks whether rec's context ended with a qualifying
// rawContextFailureCause (neither context.Canceled nor bare
// context.DeadlineExceeded) and reports whether the outermost ancestor sharing
// such a context cause is a stop point (workflow node, runner root, or parallel
// branch). A context.Canceled error recorded on a call with collateral == true
// while the invocation context itself is still live came from sibling-failure
// teardown (such as a parallel agent's errgroup) rather than the call or an
// enclosing agent canceling a private sub-context of its own. Unlike
// contextFailureCause, this uses rawContextFailureCause so that a custom
// agent's own cancel cause on a non-parallel span still wraps a recorded
// context.Canceled error (see TestTraceManager_RecordErrorWrapsNonCollateralCustomCancelCause).
func (rec *spanRecord) collateralFailureCause() (cause error, collateral bool) {
	cause = rec.rawContextFailureCause()
	if cause == nil {
		return nil, false
	}
	top := rec
	for top.parent != nil && top.parent.rawContextFailureCause() != nil {
		top = top.parent
	}
	return cause, top.stopPoint
}

// toolAttempt identifies one dispatch of a tool call by the EventActions of
// its tool context. ADK makes the EventActions of a tool context anew for
// every call, so two attempts never share them, even when a model or
// provider reuses a function call ID, and passes the one tool context of a
// call to its BeforeToolCallback, OnToolErrorCallback and AfterToolCallback.
// The contexts that ADK derives from a tool context (WithAgentTimeout,
// WithAgentCancel, WithDelta) keep its EventActions. A context whose Actions
// method returns other EventActions on each call, which ADK's contexts never
// do, makes each callback an attempt of its own, so that its
// AfterToolCallback does not find the call that its OnToolErrorCallback
// ended. The reference is weak: it keeps nothing of the call alive and,
// unlike an address, never compares equal to the reference of other
// EventActions that the runtime later puts at the same address. The zero
// value is the attempt of every context that exposes no EventActions, which
// ADK's tool dispatch never builds, so the callbacks of such contexts match
// by call key alone.
type toolAttempt = weak.Pointer[session.EventActions]

// toolActionsOf returns the EventActions of the tool callback context ctx, or
// nil if ctx has no Actions method, or the method panics or returns nil. It
// runs code of the context, so it is called without inv.mu held.
func toolActionsOf(ctx context.Context) *session.EventActions {
	c, ok := ctx.(interface{ Actions() *session.EventActions })
	if !ok {
		return nil
	}
	return orZero(c.Actions)
}

// attemptOf returns the attempt that actions identify, or the zero value for
// nil.
func attemptOf(actions *session.EventActions) toolAttempt {
	if actions == nil {
		return toolAttempt{}
	}
	return weak.Make(actions)
}

// callKey prefixes key with the pointer identity of ctx's agent-run
// InvocationContext when present (see activationKey), isolating concurrent
// activations of the same agent on one branch and node path without retaining
// ctx. Because toolKeyCall reads after the final NUL byte, prefixing before
// toolKey preserves toolKeyCall unchanged.
func callKey(ctx context.Context, key string) string {
	if act := activationKey(ctx); act != nil {
		return fmt.Sprintf("%p\x00%s", act, key)
	}
	return key
}

// dropContexts drops what rec keeps of the callbacks of its run: the context
// of the callback that started it, the InvocationContext of its activation
// (see ended and activation), any retired causes, and the span that it started
// under, whose record keeps the context of its own callback. It is for a call
// that ended, or that the run handed to the late state: its record outlives
// the run, so it must keep no context of it. wasCanceled and canceledByStop
// report false, contextFailureCause and openFailureCause report nil, and
// collateralFailureCause reports (nil, false) for such a record.
func (rec *spanRecord) dropContexts() {
	rec.canceled, rec.ended, rec.activation, rec.parent = nil, nil, nil, nil
	rec.retiredCauses = nil
}

// dropRetiredLateContextsLocked drops what agent spans retired from a late
// trace keep of the callbacks of their runs. It clears ended, activation and
// retiredCauses on each of them, and canceled and parent on each one that no
// remaining open span or call reaches through its parent chain (see
// subtreeLocked and launcherOpenLocked) and on each retired span above it, up
// to the first span that is still open or still reached. One that is still
// reached keeps both: the cancellation checks of the span that reaches it
// (see contextFailureCause, canceledByStop and collateralFailureCause) walk
// up that chain and read canceled on each record in it.
func (inv *invocationTrace) dropRetiredLateContextsLocked(retired ...*spanRecord) {
	for _, r := range retired {
		r.ended, r.activation, r.retiredCauses = nil, nil, nil
	}
	keepParent := make(map[*spanRecord]bool)
	markChain := func(s *spanRecord) {
		for p := s.parent; p != nil && p.kind == spanKindAgent && !keepParent[p]; p = p.parent {
			keepParent[p] = true
		}
	}
	for _, s := range inv.agentStack[1:] {
		markChain(s)
	}
	for _, s := range inv.llmSpans {
		markChain(s)
	}
	for _, s := range inv.toolSpans {
		markChain(s)
	}
	for _, r := range retired {
		// Walk up through the retired spans that no open span or call
		// reaches anymore.
		for s := r; !keepParent[s]; {
			p := s.parent
			s.canceled, s.parent = nil, nil
			if p == nil || p.kind != spanKindAgent || slices.Contains(inv.agentStack, p) {
				break
			}
			s = p
		}
	}
}

// llmKey identifies the model call of an agent activation: an agent makes
// one model call at a time per node path and branch. Length-prefixed encoding
// keeps parseLLMKey exact even when path, agentName, or branch contains NUL
// bytes or colons.
func llmKey(id identity) string {
	var b strings.Builder
	b.Grow(len(id.path) + len(id.agentName) + len(id.branch) + 12)
	b.WriteString(strconv.Itoa(len(id.path)))
	b.WriteByte(':')
	b.WriteString(id.path)
	b.WriteString(strconv.Itoa(len(id.agentName)))
	b.WriteByte(':')
	b.WriteString(id.agentName)
	b.WriteByte(':')
	b.WriteString(id.branch)
	return b.String()
}

// parseLLMKey extracts (path, agentName, branch) from a key created by llmKey.
func parseLLMKey(k string) (path, name, branch string, ok bool) {
	pathLenStr, rest, ok := strings.Cut(k, ":")
	if !ok {
		return "", "", "", false
	}
	pathLen, err := strconv.Atoi(pathLenStr)
	if err != nil || pathLen < 0 || pathLen > len(rest) {
		return "", "", "", false
	}
	path, rest = rest[:pathLen], rest[pathLen:]
	nameLenStr, rest, ok := strings.Cut(rest, ":")
	if !ok {
		return "", "", "", false
	}
	nameLen, err := strconv.Atoi(nameLenStr)
	if err != nil || nameLen < 0 || nameLen+1 > len(rest) || rest[nameLen] != ':' {
		return "", "", "", false
	}
	return path, rest[:nameLen], rest[nameLen+1:], true
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

type contextStructQueue struct {
	inline   [16]reflect.Value
	head     int
	tail     int
	count    int
	overflow []reflect.Value
	overHead int
}

func (q *contextStructQueue) push(v reflect.Value) {
	if !v.IsValid() || ((v.Kind() == reflect.Interface || v.Kind() == reflect.Pointer) && v.IsNil()) {
		return
	}
	if q.overflow != nil {
		q.overflow = append(q.overflow, v)
		return
	}
	if q.count == len(q.inline) {
		q.overflow = make([]reflect.Value, len(q.inline)+1, len(q.inline)*2)
		for i := range q.count {
			idx := (q.head + i) & (len(q.inline) - 1)
			q.overflow[i] = q.inline[idx]
			q.inline[idx] = reflect.Value{}
		}
		q.overflow[q.count] = v
		q.head, q.tail, q.count = 0, 0, 0
		return
	}
	q.inline[q.tail] = v
	q.tail = (q.tail + 1) & (len(q.inline) - 1)
	q.count++
}

func (q *contextStructQueue) pop() (reflect.Value, bool) {
	if q.overflow != nil {
		if q.overHead >= len(q.overflow) {
			return reflect.Value{}, false
		}
		v := q.overflow[q.overHead]
		q.overflow[q.overHead] = reflect.Value{}
		q.overHead++
		return v, true
	}
	if q.count == 0 {
		return reflect.Value{}, false
	}
	v := q.inline[q.head]
	q.inline[q.head] = reflect.Value{}
	q.head = (q.head + 1) & (len(q.inline) - 1)
	q.count--
	return v, true
}

func sameContextChild(a, b reflect.Value) bool {
	if !a.IsValid() || !b.IsValid() {
		return false
	}
	if a.Kind() == reflect.Interface && !a.IsNil() {
		a = a.Elem()
	}
	if b.Kind() == reflect.Interface && !b.IsNil() {
		b = b.Elem()
	}
	return a.Kind() == reflect.Pointer && b.Kind() == reflect.Pointer && a.Pointer() == b.Pointer() && a.Type() == b.Type()
}

// adkContextChildren returns up to two inner context fields from allowlisted
// Go stdlib and ADK context wrapper structs, ignoring application-owned types.
func adkContextChildren(v reflect.Value) (reflect.Value, reflect.Value) {
	vt := v.Type()
	switch vt.PkgPath() {
	case "context":
		if f := directStructField(v, "Context"); f.IsValid() {
			return f, reflect.Value{}
		}
		if f := directStructField(v, "cancelCtx"); f.IsValid() {
			return f, reflect.Value{}
		}
		return directStructField(v, "c"), reflect.Value{}
	case agentPkgPath:
		switch vt.Name() {
		case "commonContext":
			c1, c2 := directStructField(v, "Context"), directStructField(v, "invocationContext")
			if sameContextChild(c1, c2) {
				return c1, reflect.Value{}
			}
			return c1, c2
		case "invocationContext":
			return directStructField(v, "Context"), reflect.Value{}
		case "callbackContextWrapper", "toolContextWrapper":
			return directStructField(v, "context"), reflect.Value{}
		}
	case internalContextPkgPath:
		switch vt.Name() {
		case "InvocationContext":
			return directStructField(v, "Context"), reflect.Value{}
		case "ReadonlyContext":
			c1, c2 := directStructField(v, "Context"), directStructField(v, "InvocationContext")
			if sameContextChild(c1, c2) {
				return c1, reflect.Value{}
			}
			return c1, c2
		}
	case internalLLMPkgPath:
		if vt.Name() == "cancelledToolContext" {
			return directStructField(v, "Context"), directStructField(v, "cancelCtx")
		}
	case remoteAgentPkgPath:
		if vt.Name() == "authContext" {
			return directStructField(v, "InvocationContext"), directStructField(v, "ctx")
		}
	case a2aServerPkgPath:
		if vt.Name() == "executorContext" {
			return directStructField(v, "Context"), reflect.Value{}
		}
	}
	return reflect.Value{}, reflect.Value{}
}

// contextParentMap returns the child-to-parent agent map that ADK's runner
// attaches to ctx via internal/agent/parentmap, or nil if ctx does not carry
// one. Only Go stdlib and ADK context wrappers are unwrapped; fields of
// third-party or user context types are never read.
func contextParentMap(ctx context.Context) (res map[string]agent.Agent) {
	defer func() {
		if r := recover(); r != nil {
			res = nil
		}
	}()
	if ctx == nil {
		return nil
	}
	var q contextStructQueue
	q.push(reflect.ValueOf(ctx))
	var seen map[visitedContextStruct]bool
	for steps := range 512 {
		item, ok := q.pop()
		if !ok {
			break
		}
		v, ok := derefContextStructValueLazy(item, &seen, steps >= 32)
		if !ok {
			continue
		}
		if v.Type().PkgPath() == "context" {
			if key, val := directStructField(v, "key"), directStructField(v, "val"); key.IsValid() && val.IsValid() {
				if val.Kind() == reflect.Interface && !val.IsNil() {
					val = val.Elem()
				}
				if val.IsValid() && val.Kind() == reflect.Map &&
					val.Type().Key().Kind() == reflect.String &&
					val.Type().Elem() == agentInterfaceType {
					if keyVal, ok := exportContextKey(key); ok {
						lookupCtx := ctx
						if v.CanAddr() && v.Addr().CanInterface() {
							if c, ok := reflect.TypeAssert[context.Context](v.Addr()); ok {
								lookupCtx = c
							}
						}
						if raw := orZero(func() any { return lookupCtx.Value(keyVal) }); raw != nil {
							if m := reflect.ValueOf(raw); m.IsValid() && m.Kind() == reflect.Map && m.CanInterface() {
								out := make(map[string]agent.Agent, m.Len())
								iter := m.MapRange()
								for iter.Next() {
									if a, ok := reflect.TypeAssert[agent.Agent](iter.Value()); ok && a != nil {
										out[iter.Key().String()] = a
									}
								}
								return out
							}
						}
					}
				}
			}
		}
		c1, c2 := adkContextChildren(v)
		q.push(c1)
		q.push(c2)
	}
	return nil
}

var (
	pluginPtrType        = reflect.TypeFor[*baseplugin.Plugin]()
	uuidProviderType     = reflect.TypeFor[platform.UUIDProvider]()
	pluginManagerPkgPath = adkPkgPrefix + "internal/plugininternal"

	uuidProviderKeyOnce sync.Once
	uuidProviderCtxKey  any

	pluginManagerKeyMu        sync.Mutex
	pluginManagerCtxKey       any
	pluginManagerStructType   reflect.Type
	pluginManagerPluginsField = -1
)

// directStructField returns the direct field of v named name without traversing
// embedded struct pointers, so a nil embedded pointer cannot panic.
func directStructField(v reflect.Value, name string) reflect.Value {
	t := v.Type()
	for i, n := 0, t.NumField(); i < n; i++ {
		if t.Field(i).Name == name {
			return v.Field(i)
		}
	}
	return reflect.Value{}
}

func isPluginManagerType(t reflect.Type) bool {
	return t != nil && t.Kind() == reflect.Struct &&
		t.Name() == "PluginManager" && t.PkgPath() == pluginManagerPkgPath
}

func discoverUUIDProviderCtxKey() any {
	uuidProviderKeyOnce.Do(func() {
		probe := platform.WithUUIDProvider(context.Background(), func() string { return "" })
		cur := reflect.ValueOf(probe)
		for cur.IsValid() {
			for cur.Kind() == reflect.Pointer || cur.Kind() == reflect.Interface {
				if cur.IsNil() {
					return
				}
				cur = cur.Elem()
			}
			if !cur.IsValid() || cur.Kind() != reflect.Struct {
				return
			}
			if key, val := directStructField(cur, "key"), directStructField(cur, "val"); key.IsValid() && val.IsValid() {
				if val.Kind() == reflect.Interface && !val.IsNil() {
					val = val.Elem()
				}
				if val.IsValid() && val.Type() == uuidProviderType {
					if k, ok := exportContextKey(key); ok {
						uuidProviderCtxKey = k
						return
					}
				}
			}
			cur = directStructField(cur, "Context")
		}
	})
	return uuidProviderCtxKey
}

func discoverPluginManagerCtxKey(ctx context.Context) any {
	pluginManagerKeyMu.Lock()
	defer pluginManagerKeyMu.Unlock()
	if pluginManagerCtxKey == nil {
		key, structType, pluginsField := probePluginManagerCtxKey(ctx)
		if key != nil && structType != nil && pluginsField >= 0 {
			pluginManagerStructType = structType
			pluginManagerPluginsField = pluginsField
			pluginManagerCtxKey = key
		}
	}
	return pluginManagerCtxKey
}

// probePluginManagerCtxKey reads ADK's unexported PluginManager context key
// and layout from an existing callback context. Starting a runner for discovery
// would emit an invocation span, even if its agent returned immediately.
func probePluginManagerCtxKey(ctx context.Context) (key any, structType reflect.Type, pluginsField int) {
	pluginsField = -1
	defer func() {
		if r := recover(); r != nil {
			key, structType, pluginsField = nil, nil, -1
		}
	}()
	queue := []reflect.Value{reflect.ValueOf(ctx)}
	seen := map[visitedContextStruct]bool{}
	for len(queue) > 0 {
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
		if v.Type().PkgPath() == "context" && v.Type().Name() == "valueCtx" {
			if kVal, val := directStructField(v, "key"), directStructField(v, "val"); kVal.IsValid() && val.IsValid() {
				if val.Kind() == reflect.Interface && !val.IsNil() {
					val = val.Elem()
				}
				if val.Kind() == reflect.Pointer && !val.IsNil() && isPluginManagerType(val.Type().Elem()) {
					vt := val.Type().Elem()
					for sf := range vt.Fields() {
						if len(sf.Index) == 1 && sf.Name == "plugins" && sf.Type.Kind() == reflect.Slice && sf.Type.Elem() == pluginPtrType {
							if k, ok := exportContextKey(kVal); ok && reflect.TypeOf(k).PkgPath() == pluginManagerPkgPath+"/plugincontext" {
								return k, vt, sf.Index[0]
							}
						}
					}
				}
			}
		}
		c1, c2 := adkContextChildren(v)
		queue = append(queue, c1, c2)
	}
	return nil, nil, -1
}

// contextHasAfterAgentPlugin queries ctx for ADK's PluginManager using the
// once-discovered context key and reports whether any other registered plugin
// (before or after self) defines an AfterAgentCallback, since ADK builds the
// single after-agent event after all AfterAgentCallbacks run. When self != nil
// and PluginManager or self cannot be resolved (for example, when self is
// wrapped inside a composing plugin), it conservatively reports true so
// post-retirement after-agent events are never falsely excluded. When self is
// nil (as in direct traceManager unit tests), it falls back to matching
// pluginName once and reports false if ctx carries no PluginManager.
func contextHasAfterAgentPlugin(ctx context.Context, self *baseplugin.Plugin) (res bool) {
	unknown := self != nil
	if ctx == nil {
		return unknown
	}
	key := discoverPluginManagerCtxKey(ctx)
	if key == nil || pluginManagerStructType == nil || pluginManagerPluginsField < 0 {
		return unknown
	}
	raw, ok := func() (val any, ok bool) {
		defer func() {
			if recover() != nil {
				val, ok = nil, false
			}
		}()
		return ctx.Value(key), true
	}()
	if !ok || raw == nil {
		return unknown
	}
	defer func() {
		if r := recover(); r != nil {
			res = true
		}
	}()
	v := reflect.ValueOf(raw)
	if !v.IsValid() || v.Kind() != reflect.Pointer || v.IsNil() {
		return unknown
	}
	elem := v.Elem()
	if elem.Type() != pluginManagerStructType {
		return unknown
	}
	plugins := elem.Field(pluginManagerPluginsField)
	n := plugins.Len()
	pluginElemType := pluginPtrType.Elem()
	selfIdx := -1
	if self != nil {
		selfPtr := reflect.ValueOf(self).Pointer()
		for i := range n {
			elem := plugins.Index(i)
			if elem.Kind() == reflect.Pointer && !elem.IsNil() && elem.Pointer() == selfPtr {
				selfIdx = i
				break
			}
		}
		if selfIdx < 0 {
			return true
		}
	} else {
		for i := range n {
			elem := plugins.Index(i)
			if elem.Kind() != reflect.Pointer || elem.IsNil() || elem.Elem().Type() != pluginElemType {
				return true
			}
			nameField := directStructField(elem.Elem(), "name")
			if !nameField.IsValid() || nameField.Kind() != reflect.String {
				return true
			}
			if nameField.String() == pluginName {
				selfIdx = i
				break
			}
		}
	}
	for k := range n {
		if k == selfIdx {
			continue
		}
		elem := plugins.Index(k)
		if elem.Kind() != reflect.Pointer || elem.IsNil() || elem.Elem().Type() != pluginElemType {
			return true
		}
		cb := directStructField(elem.Elem(), "afterAgentCallback")
		if !cb.IsValid() || cb.Kind() != reflect.Func {
			return true
		}
		if !cb.IsNil() {
			return true
		}
	}
	return false
}

// contextHasCustomUUIDProvider queries ctx for ADK's platform.WithUUIDProvider
// entry using the once-discovered context key and reports whether the active
// provider is non-nil without invoking it.
func contextHasCustomUUIDProvider(ctx context.Context) (res bool) {
	if ctx == nil {
		return false
	}
	key := discoverUUIDProviderCtxKey()
	if key == nil {
		return false
	}
	defer func() {
		if r := recover(); r != nil {
			res = false
		}
	}()
	p, ok := ctx.Value(key).(platform.UUIDProvider)
	return ok && p != nil
}

// callbackAfterAgentActionMaps returns the action maps of the EventActions of
// ADK's callback context ctx (see actionMaps), so that onEvent can recognize
// the event that runAfterAgentCallbacks builds from that EventActions.
// On a callback context, Actions() logs that it is unsupported and returns nil;
// a callback can still reach the live *session.EventActions via
// ctx.WithDelta(nil).Actions() and replace StateDelta or ArtifactDelta (in
// which case the yielded event carries the replacement map rather than the one
// captured here). Only immutable Go stdlib context and ADK agent context
// structs are unwrapped—never fields of arbitrary user-defined context types.
func callbackAfterAgentActionMaps(ctx context.Context) actionMaps {
	am, _ := extractCallbackAfterAgentActions(ctx)
	return am
}

// callbackAfterAgentActions returns callbackAfterAgentActionMaps(ctx) and, when
// they are not zero, a value that keeps the captured StateDelta and
// ArtifactDelta maps reachable directly (as [2]unsafe.Pointer, referring to
// nothing of ctx or the EventActions struct). Retaining keep prevents garbage
// collection and address reuse of those captured map objects while keep is held
// (see actionMaps), even if a callback subsequently replaces StateDelta or
// ArtifactDelta on ctx.WithDelta(nil).Actions() with new maps (in which case the
// yielded event carries the replacement maps rather than the captured ones).
func callbackAfterAgentActions(ctx context.Context) (actionMaps, any) {
	am, pins := extractCallbackAfterAgentActions(ctx)
	if am == (actionMaps{}) {
		return actionMaps{}, nil
	}
	return am, pins
}

func extractCallbackAfterAgentActions(ctx context.Context) (actionMaps, [2]unsafe.Pointer) {
	if ctx == nil {
		return actionMaps{}, [2]unsafe.Pointer{}
	}
	return extractCallbackAfterAgentActionsValue(reflect.ValueOf(ctx))
}

func extractCallbackAfterAgentActionsValue(cur reflect.Value) (am actionMaps, pins [2]unsafe.Pointer) {
	defer func() {
		if r := recover(); r != nil {
			am = actionMaps{}
			pins = [2]unsafe.Pointer{}
		}
	}()
	var seen map[visitedContextStruct]bool
	for steps := 0; cur.IsValid() && steps < 512; steps++ {
		var ok bool
		cur, ok = derefContextStructValueLazy(cur, &seen, steps >= 32)
		if !ok {
			return actionMaps{}, [2]unsafe.Pointer{}
		}
		vt := cur.Type()
		if vt.PkgPath() == agentPkgPath && vt.Name() == "commonContext" {
			actions, ok := derefContextStructValueLazy(directStructField(cur, "actions"), &seen, steps >= 32)
			if !ok {
				return actionMaps{}, [2]unsafe.Pointer{}
			}
			pins = [2]unsafe.Pointer{
				mapPointer(directStructField(actions, "StateDelta")),
				mapPointer(directStructField(actions, "ArtifactDelta")),
			}
			return actionMaps{
				stateDelta:    pins[0],
				artifactDelta: pins[1],
			}, pins
		}
		var next reflect.Value
		switch vt.PkgPath() {
		case "context":
			if f := directStructField(cur, "Context"); f.IsValid() {
				next = f
			} else if f := directStructField(cur, "cancelCtx"); f.IsValid() {
				next = f
			} else if f := directStructField(cur, "c"); f.IsValid() {
				next = f
			}
		case agentPkgPath:
			if f := directStructField(cur, "context"); f.IsValid() {
				next = f
			}
		default:
			// Never read fields of third-party or user context structs without
			// their synchronization.
			return actionMaps{}, [2]unsafe.Pointer{}
		}
		cur = next
	}
	return actionMaps{}, [2]unsafe.Pointer{}
}

func derefContextStructValue(v reflect.Value, seen map[visitedContextStruct]bool) (reflect.Value, bool) {
	return derefContextStructValueLazy(v, &seen, seen != nil)
}

func derefContextStructValueLazy(v reflect.Value, seen *map[visitedContextStruct]bool, track bool) (reflect.Value, bool) {
	for derefs := 0; v.IsValid() && (v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface); derefs++ {
		if derefs >= 512 || v.IsNil() {
			return reflect.Value{}, false
		}
		if v.Kind() == reflect.Pointer && (*seen != nil || track || derefs >= 4) {
			if *seen == nil {
				*seen = make(map[visitedContextStruct]bool, 8)
			}
			visitKey := visitedContextStruct{ptr: v.Pointer(), typ: v.Type()}
			if (*seen)[visitKey] {
				return reflect.Value{}, false
			}
			(*seen)[visitKey] = true
		}
		v = v.Elem()
	}
	if !v.IsValid() || v.Kind() != reflect.Struct {
		return reflect.Value{}, false
	}
	return v, true
}

// mapPointer returns an unsafe.Pointer to the map that v holds, or nil when v
// is not a non-nil map. Storing it keeps the map object live for the GC.
func mapPointer(v reflect.Value) unsafe.Pointer {
	if !v.IsValid() || v.Kind() != reflect.Map || v.IsNil() {
		return nil
	}
	return v.UnsafePointer()
}

// eventActionMaps returns the action maps of ev (see actionMaps), zero when ev
// is nil.
func eventActionMaps(ev *session.Event) actionMaps {
	if ev == nil {
		return actionMaps{}
	}
	return actionMaps{
		stateDelta:    mapPointer(reflect.ValueOf(ev.Actions.StateDelta)),
		artifactDelta: mapPointer(reflect.ValueOf(ev.Actions.ArtifactDelta)),
	}
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

// touchAgentActivityLocked updates the lastActivity of rec and every agent
// activation above it up to (not including) the first non-agent span, so a
// purged activation's latency and row time reflect the latest work under it
// rather than the retry delay that follows it. It never moves lastActivity
// backward: a callback that runs at a wall-clock time earlier than the last
// one, as a reordered event may, leaves lastActivity unchanged.
func (inv *invocationTrace) touchAgentActivityLocked(rec *spanRecord, now time.Time) {
	for ; rec != nil && rec.kind == spanKindAgent; rec = rec.parent {
		if now.After(rec.lastActivity) {
			rec.lastActivity = now
		}
	}
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
