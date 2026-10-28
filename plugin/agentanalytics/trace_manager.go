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

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
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

var agentInterfaceType = reflect.TypeFor[agent.Agent]()

// agentPkgPath and workflowPkgPath are the import paths of ADK's agent and
// workflow packages, and adkPkgPrefix is the import path prefix that all of
// ADK's packages share, as runtime.Stack names their functions.
var (
	agentPkgPath    = agentInterfaceType.PkgPath()
	workflowPkgPath = reflect.TypeFor[workflow.Edge]().PkgPath()
	adkPkgPrefix    = agentPkgPath[:strings.LastIndexByte(agentPkgPath, '/')+1]
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
	// finishedTasks holds the function call IDs of the finish_task calls
	// that the activation's own tool callbacks saw complete (see
	// noteTaskFinished), so that the function response event that reports
	// one of them ends this activation rather than another run of the same
	// agent on the same branch, on another node path or, if no other run on
	// the event's node path saw a call with that ID complete, on the same one
	// (see finishedTaskLocked and popTaskAgentSpan).
	finishedTasks []string
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
	// callStarted reports that pushLLMSpan or pushToolSpan started the model
	// or tool call, rather than popToolSpan making up a span for a call that
	// this plugin did not see start.
	callStarted bool
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
}

// retiredCause is the failure cause of an activation that retired under an
// open span (see spanRecord.retiredCauses), with the name, branch and node
// path of the retired activation. swallowableBySelf reports that the retired
// activation drained an inherited child failure without its own context
// failing and is not a built-in workflow agent, so recovery content from its
// own AfterAgentCallback may clear the cause.
type retiredCause struct {
	err               error
	name              string
	branch            string
	path              string
	swallowableBySelf bool
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
	// lastSeq is the seq of the agent activation or tool call that started
	// last (see spanRecord.seq).
	lastSeq uint64
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
	// retiredPeerCount counts retired peer activations per llmKey when
	// concurrent activations of the same (path, agentName, branch) overlapped,
	// so trailer events emitted from a retired peer's own AfterAgentCallback
	// are absorbed rather than attributed to a remaining peer with an
	// unrecovered model or tool error.
	retiredPeerCount map[string]int
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
	root := &spanRecord{spanID: newSpanID(), kind: spanKindInvocation, start: now}
	inv := &invocationTrace{
		traceID:           cmp.Or(traceID, newTraceID()),
		rootSpanID:        root.spanID,
		pendingSpans:      new(atomic.Int64),
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
		next, ok := inner.Interface().(agent.Agent)
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
		cfg, isLLM := agentConfig(a).(llmagent.Config)
		agType := orZero(func() string { return agentTypeOf(a) })
		children := agentChildren{
			parent:               name,
			nonLLM:               !isLLM,
			propagatesChildError: isBuiltInWorkflowAgent(a),
			isParallel:           agType == "ParallelAgent",
			isWorkflow:           agType == "WorkflowAgent",
			taskMode:             isLLM && cfg.Mode == llmagent.ModeTask,
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

// parentForRunLocked returns the span that an agent starting with id on
// goroutine goid, whose origin is origin, runs under (see parentFor), leaving
// out the open activations of id's agent, branch and node path that the
// scheduler of their workflow agent started on other goroutines, as it
// started goid, together with the agents under them (see subtreeLocked). The
// scheduler runs each node run on a goroutine of its own right under the
// workflow agent, so such an earlier run of the node does not enclose the new
// one even when an agent under it shares id's branch and node path, as a
// sub-agent of a sequential agent that a failed attempt left open does.
func (inv *invocationTrace) parentForRunLocked(id identity, goid uint64, origin goroutineOrigin) *spanRecord {
	var exclude map[string]bool
	for i := 1; i < len(inv.agentStack); i++ {
		other := inv.agentStack[i]
		if other.kind != spanKindAgent || other.name != id.agentName || other.branch != id.branch || other.path != id.path ||
			other.goroutineID == goid || other.parent == nil || !other.parent.isWorkflow ||
			!origin.startedBy(schedulerCreator, other.parent) || !other.origin.startedBy(schedulerCreator, other.parent) {
			continue
		}
		if exclude == nil {
			exclude = make(map[string]bool)
		}
		maps.Copy(exclude, inv.subtreeLocked(i))
	}
	if exclude == nil {
		return inv.parentFor(id)
	}
	return inv.parentOutsideLocked(id, exclude)
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
	position                *list.Element
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
				late := tm.lateLocked(completed, sc, tm.traceIDFor(id.invocationID), now)
				late.lastActive.Store(now.UnixNano())
				tm.completionOrder.MoveToBack(completed.position)
				if reserveSpan {
					late.pendingSpans.Add(1)
				}
				tm.mu.Unlock()
				return late
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
// lock. A callback of a pruned run gets a span of its own.
// tm.mu must be held.
//
// Divergence from adk-python (google/adk-python@25d191710985,
// after_run_callback lines 10102-10145): Python ADK clears the current
// task's contextvars span stack in after_run_callback. Go needs explicit
// shared late-call bookkeeping because goroutines do not inherit
// task-local state and RunLive tool goroutines can outlive
// AfterRunCallback, so open calls are handed to the late state instead.
func (tm *traceManager) rememberCompletedLocked(invocationID, runner string, inv *invocationTrace) {
	completed := tm.completed[invocationID]
	if completed == nil {
		completed = &completedTrace{position: tm.completionOrder.PushBack(invocationID)}
		tm.completed[invocationID] = completed
	} else {
		tm.completionOrder.MoveToBack(completed.position)
	}
	completed.identity = inv.savedIdentity()
	if inv.pendingSpans.Load() > 0 {
		completed.livePending = inv.pendingSpans
	} else if completed.livePending != nil && completed.livePending.Load() == 0 {
		completed.livePending = nil
	}
	inv.mu.Lock()
	inv.finished = true
	inv.agentStack[0].dropContexts()
	inv.runnerAgent = nil
	lastSeq := inv.lastSeq
	models, endedModels := startedCalls(inv.llmSpans), startedCalls(inv.completedModels)
	maps.DeleteFunc(models, func(_ string, rec *spanRecord) bool { return rec.streamed || rec.endedWithRun(runner) })
	tools, endedTools := startedCalls(inv.toolSpans), startedCalls(inv.completedTools)
	maps.DeleteFunc(tools, func(_ string, rec *spanRecord) bool { return rec.endedWithRun(runner) })
	// The late state keeps no context of the run's callbacks: the calls that
	// are open drop theirs here, and those that ended did when they ended.
	// The open ones are marked as handed over, so that the run waits for late
	// calls only while one of them is open (see waitsForLateCalls).
	for _, rec := range models {
		rec.dropContexts()
		rec.handedOver = true
	}
	for _, rec := range tools {
		rec.dropContexts()
		rec.handedOver = true
	}
	// The calls leave inv, so that no spanRecord is reachable from both inv,
	// guarded by inv.mu, and the late state, guarded by its own mu.
	clear(inv.llmSpans)
	clear(inv.completedModels)
	clear(inv.toolSpans)
	clear(inv.completedTools)
	inv.syncOpenCallsLocked()
	inv.mu.Unlock()
	if len(models)+len(tools) > 0 {
		tm.lateLocked(completed, trace.SpanContext{}, completed.identity.traceID, tm.now())
	}
	if late := completed.late; late != nil {
		late.mu.Lock()
		late.lastSeq = max(late.lastSeq, lastSeq)
		maps.Copy(late.llmSpans, models)
		maps.Copy(late.completedModels, endedModels)
		maps.Copy(late.toolSpans, tools)
		maps.Copy(late.completedTools, endedTools)
		late.syncOpenCallsLocked()
		late.mu.Unlock()
	} else {
		completed.endedModels = withCalls(completed.endedModels, endedModels)
		completed.endedTools = withCalls(completed.endedTools, endedTools)
	}
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
		late := newInvocationTraceWithSpan(sc, traceID, now)
		late.isLate = true
		late.restoreEvicted(completed.identity)
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
	inv := tm.lookupInvocation(ctx, id, true, false)
	now := tm.now()
	rememberedRoot := tm.rememberAgentTree(runner)
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, runner, rememberedRoot)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.started = true
	inv.noteRootAgentLocked(id, rootIn, false)
	inv.agentStack[0].canceled = canceledFunc(ctx)
	span := inv.rowSpan(inv.agentStack[0], nil, ambient)
	span.timestamp = now
	return span
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
	row.timestamp = now
	inv.mu.Unlock()
	if id.invocationID != "" {
		tm.mu.Lock()
		// Publish the final identity, including any trace adopted by rowSpan,
		// before removing active state. Late Live callbacks may start and end
		// an operation, so their state is shared separately from the finished run.
		tm.rememberCompletedLocked(id.invocationID, id.agentName, inv)
		delete(tm.evicted, id.invocationID)
		delete(tm.invocations, id.invocationID)
		tm.mu.Unlock()
	}
	return row
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
			inv.retireLateSettledLocked()
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
	inv := tm.lookupInvocation(ctx, id, false, true)
	defer inv.pendingSpans.Add(-1)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	// SubAgents is user code, so it is read before mu is taken.
	children := runningAgentChildren(ctx, id.agentName)
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
	// Register concurrent peers before the workflow-retry completion-inference
	// loop so the retry guard can check !rec.concurrentPeer. Only peers that
	// are NOT earlier attempts of the node are marked (see laterAttempt).
	parent := inv.parentForRunLocked(id, goid, origin)
	var concurrentPeer bool
	isSyntheticRetry := func(other *spanRecord) bool {
		return activation == nil && other.activation == nil && other.parent != nil && other.parent.activation == nil
	}
	// laterAttempt reports whether the activation of id is a later attempt
	// of the workflow node that other ran as its agent (see attemptSite), so
	// that other's failure is handled: other runs, with another activation,
	// under a workflow agent whose node path lies above id.path, and either
	//
	//   - both are the agent of a workflow.ParallelWorker item that ran on
	//     this goroutine, as when RetryConfig runs the item's node again on
	//     the worker's goroutine. The stack of this callback shows no other
	//     agent run (see attemptSiteOfTrace), so other's run, and every run
	//     that started on this goroutine before, has returned, and synced
	//     reports that other's InvocationContext can be read; or
	//   - both are the agent of a node attempt that the workflow agent's
	//     scheduler started on a goroutine of its own (see startedBy), and a
	//     node completion event reported that the node run that ran other
	//     ended (see spanRecord.nodeRunEnded). A node can write the same
	//     metadata on an event of its own while it still runs, so the report
	//     does not show that other's run is over: synced is false, other's
	//     InvocationContext is not read, and other is purged now only for a
	//     failure that was recorded for it, not for having no response yet
	//     or for a model or tool call that it or a run under it still has
	//     open.
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
		if !other.parent.isWorkflow {
			return false, false
		}
		switch {
		case attempt == attemptWorker && other.attempt == attemptWorker && goid != 0 && other.goroutineID == goid:
			return true, true
		case attempt == attemptNode && other.attempt == attemptNode && other.nodeRunEnded &&
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
		if !later || rec.concurrentPeer || rec.parent != inv.parentOutsideLocked(id, inv.subtreeLocked(i)) {
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
			break
		}
	}
	// Mark concurrent peers on any remaining matching spans (workflow
	// children that did not fail and were not purged by the retry loop).
	parent = inv.parentForRunLocked(id, goid, origin)
	for _, other := range inv.agentStack[1:] {
		if other.kind == spanKindAgent &&
			other.parent == parent && other.name == id.agentName && other.branch == id.branch && other.path == id.path {
			other.concurrentPeer = true
			concurrentPeer = true
		}
	}
	rec := &spanRecord{
		spanID:               newSpanID(),
		parentSpanID:         parent.spanID,
		kind:                 spanKindAgent,
		path:                 id.path,
		name:                 id.agentName,
		branch:               id.branch,
		start:                now,
		lastActivity:         now,
		canceled:             canceledFunc(ctx),
		ended:                ended,
		goroutineID:          goid,
		origin:               origin,
		attempt:              attempt,
		activation:           activation,
		parent:               parent,
		stopPoint:            parent.kind == spanKindInvocation || id.path != parent.path || id.branch != parent.branch,
		nonLLM:               children.nonLLM,
		propagatesChildError: children.propagatesChildError,
		isParallel:           children.isParallel,
		isWorkflow:           children.isWorkflow,
		concurrentPeer:       concurrentPeer,
		taskMode:             children.taskMode,
		children:             children.names,
		seq:                  inv.nextSeqLocked(),
	}
	inv.agentStack = append(inv.agentStack, rec)
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
	inv.retireLateSettledLocked()
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
// run ended earlier. A run that ends with an output gets no completion event,
// so once a node completed with an output, later runs of it on the same path
// and branch are not attributed; neither are runs of a node that the
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
// The response ends the activation on the event's node path whose finish_task
// call with one of the call IDs in calls completed (see finishedTaskLocked),
// and otherwise, when no such activation is open or more than one is, as for
// a response whose call this plugin did not see complete, the activation that
// findAgentForContext finds. OnEventCallback runs on the runner's goroutine
// with the runner's context, and the event does not show that the run that
// wrote it returned: a custom agent can forward the event of a sub-agent that
// it runs on another goroutine and wait for that goroutine later. So the span
// is retired without the activation's InvocationContext being read and
// without its run being taken to be over (see retireAgentLocked).
func (tm *traceManager) popTaskAgentSpan(ctx context.Context, id identity, calls []string) (span rowSpan, ok bool) {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	ambient := ambientTraceID(ctx)
	rootIn := prepareRootAgentInput(ctx, inv, id, nil, nil)
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.noteRootAgentLocked(id, rootIn, true)
	i := inv.finishedTaskLocked(id, calls)
	if i < 0 {
		i = inv.findAgentForContext(ctx, id.path, id.agentName, id.branch)
	}
	if i < 0 {
		return rowSpan{}, false
	}
	rec := inv.agentStack[i]
	if !rec.taskMode {
		return rowSpan{}, false
	}
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
	inv.retireLateSettledLocked()
	span = inv.rowSpan(rec, latencyOf(rec, now), ambient)
	span.timestamp = now
	return span, true
}

// finishedTaskLocked returns the index in agentStack of the open activation
// of the agent of id, on the node path of id, that saw one of the finish_task
// calls in calls complete (see noteTaskFinished), or -1 if there is none, or
// more than one. A function call ID is scoped to the activation that made the
// call, not unique: ADK keeps the ID that a model gives a call, and providers
// reuse IDs such as "call_0" (see toolKey). So another run of the agent on the
// same branch may have seen a call with the same ID complete: one on another
// node path did not write the event, and among several on the event's node
// path the call ID does not tell which one did. As in findAgent, a node path
// that either side lacks matches any. inv.mu must be held.
func (inv *invocationTrace) finishedTaskLocked(id identity, calls []string) int {
	found := -1
	for i := len(inv.agentStack) - 1; i > 0; i-- {
		rec := inv.agentStack[i]
		if rec.kind != spanKindAgent || rec.name != id.agentName || rec.branch != id.branch ||
			(id.path != "" && rec.path != "" && rec.path != id.path) ||
			!slices.ContainsFunc(calls, func(call string) bool { return slices.Contains(rec.finishedTasks, call) }) {
			continue
		}
		if found >= 0 {
			return -1
		}
		found = i
	}
	return found
}

// noteTaskFinished records that the finish_task call callID of the agent
// activation that ctx and id identify completed, so that the function response
// event that reports it ends that activation (see popTaskAgentSpan). The tool
// callbacks get the context of the activation that made the call, which the
// event callback does not.
func (tm *traceManager) noteTaskFinished(ctx context.Context, id identity, callID string) {
	if callID == "" {
		return
	}
	inv := tm.invocation(ctx, id)
	for !inv.lockLive() {
		inv = tm.invocation(ctx, id)
	}
	defer inv.mu.Unlock()
	if i := inv.findAgentForContext(ctx, id.path, id.agentName, id.branch); i > 0 {
		inv.agentStack[i].finishedTasks = append(inv.agentStack[i].finishedTasks, callID)
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
	inv.agentStack = slices.Delete(inv.agentStack, i, i+1)
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
			inv.agentStack = slices.Delete(inv.agentStack, idx, idx+1)
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
		// cause moved up, its record keeps no callback context (see
		// dropRetiredLateContextsLocked).
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
	// its record keeps no callback context once it leaves agentStack (see
	// dropRetiredLateContextsLocked). Only the removed records qualify: the
	// span that agentStack[i] runs under may still be open.
	var latePurged []*spanRecord
	if inv.isLate {
		for _, rec := range inv.agentStack[i:] {
			if purged[rec.spanID] {
				latePurged = append(latePurged, rec)
			}
		}
	}
	inv.agentStack = slices.DeleteFunc(inv.agentStack, func(rec *spanRecord) bool { return purged[rec.spanID] })
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
	inv := tm.invocation(ctx, id)
	now := tm.now()
	inv.mu.Lock()
	defer inv.mu.Unlock()
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
	inv.retireLateSettledLocked()
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
	inv.agentStack = slices.DeleteFunc(inv.agentStack, func(d *spanRecord) bool {
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
//     cause only when some span on the walk sharing it is a parallel branch,
//     any cause recorded on a custom agent's own WithCancelCause with no
//     parallel branch on that walk is ignored (and is not promoted into
//     retiredCauses by retireAgentLocked), whether the custom agent handles
//     the sub-agent's failure or propagates it after the sub-agent already
//     emitted content. When a ParallelAgent runs under such a canceled outer
//     context, a parallel branch on the walk keeps the cause (#1984). The
//     callbacks otherwise see the same as for a caller that stopped reading
//     once it had the response.
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
		inv.agentStack = inv.agentStack[:1]
		return nil, primaryErr
	}

	if primaryErr == nil && !invocationEnded(ctx) {
		// When AfterRun runs on the same goroutine that started the runner's
		// root agent, settleEndedInSubtreeLocked(i, true, 0) settles the
		// descendants in that root's subtree that canSettleInSubtreeLocked
		// reaches before the reverse-order loop below pops any remaining
		// unclosed spans.
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
		inv.agentStack = inv.agentStack[:1]
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
	inv.agentStack = inv.agentStack[:1]
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
// laterAttempt there); and endRestartedNodeRuns, when the workflow agent whose
// scheduler ran it as the agent of a node attempt, and started that node
// again, ended.
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
// the activation with it (see retireAgentLocked). As there, an activation
// that failed, and those that it runs under, stay, and so does one that
// carries a failure cause (see spanRecord.openFailureCause), which failure
// reporting reads as a failure too (see failuresLocked). The callbacks that
// can release such an activation (the end of an agent, of a tool call or of
// a model call, and a function response) call it last, once they no longer
// hold an index into agentStack and have reported the failures that they
// found.
func (inv *invocationTrace) retireLateSettledLocked() {
	if !inv.isLate || len(inv.agentStack) <= 1 {
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
		if failed[d.spanID] || unresponded || d.openFailureCause() != nil {
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
	var retired []*spanRecord
	for {
		openAgents := make(map[string]bool, len(inv.agentStack))
		for _, rec := range inv.agentStack[1:] {
			openAgents[rec.spanID] = true
		}
		changed := false
		// Retiring an activation can release the activations that run under
		// it, which come after it on agentStack, so the scan starts over.
		for idx := len(inv.agentStack) - 1; idx >= 1; idx-- {
			rec := inv.agentStack[idx]
			parentOpen := rec.parent != nil && rec.parent.kind == spanKindAgent && openAgents[rec.parent.spanID]
			if !parentOpen && rec.runDone && rec.contextEnded && !inv.launcherOpenLocked(rec) && !failed[rec.spanID] {
				retired = append(retired, rec)
				if rec.parent != nil && rec.parent.kind == spanKindAgent {
					retired = append(retired, rec.parent)
				}
				inv.agentStack = slices.Delete(inv.agentStack, idx, idx+1)
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
	keys := slices.Collect(maps.Keys(inv.unrecoveredErrors))
	slices.SortFunc(keys, func(a, b string) int {
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
				if f := v.Elem().FieldByName("Context"); f.IsValid() && f.CanInterface() {
					if inner, ok := f.Interface().(agent.InvocationContext); ok && inner != nil {
						if iv := reflect.ValueOf(inner); iv.Kind() == reflect.Pointer && !iv.IsNil() {
							return inner
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
	for line := range strings.SplitSeq(frames, "\n") {
		if line == "" || line[0] == '\t' {
			continue
		}
		if strings.HasPrefix(line, "created by ") {
			break
		}
		if strings.HasPrefix(line, "...") {
			return attemptNone
		}
		paren := strings.LastIndexByte(line, '(')
		if paren <= 0 {
			return attemptNone
		}
		names = append(names, line[:paren])
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
func currentGoroutine() (uint64, goroutineOrigin, attemptSite) {
	buf := make([]byte, 8<<10)
	for {
		n := runtime.Stack(buf, false)
		if n < len(buf) || len(buf) >= maxGoroutineTraceSize {
			complete := n < len(buf)
			id, origin := parseGoroutineTrace(buf[:n], complete)
			return id, origin, attemptSiteOfTrace(buf[:n], complete)
		}
		buf = make([]byte, 2*len(buf))
	}
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
	at := strings.Index(s, createdBy)
	if at < 0 {
		return id, goroutineOrigin{}
	}
	line, _, _ := strings.Cut(s[at+len(createdBy):], "\n")
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
	// The ended call no longer keeps its agent, or the agents that its agent
	// runs under, from retiring in the late state (see
	// retireLateSettledLocked).
	inv.retireLateSettledLocked()
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
// closedByError reports that OnToolErrorCallback already ended the call,
// in which case the returned place reuses its span, without latency. If
// the call is unknown, as for a call of a tool that does not exist, the
// row gets a span of its own and no latency; the agent and invocation
// spans are left alone. Once the agent has no tool call left open, the
// activations that its tool calls ran and that never ended are ended—purged
// if they failed, or retired quietly otherwise (see endToolRunsLocked).
func (tm *traceManager) popToolSpan(ctx context.Context, id identity, key string, fromError bool) (span rowSpan, purged []unclosedAgentSpan, closedByError bool) {
	inv := tm.invocation(ctx, id)
	now := tm.now()
	ambient := ambientTraceID(ctx)
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
			inv.completedTools[key] = open
		}
		span = inv.rowSpan(open, latencyOf(open, now), ambient)
		span.timestamp = now
		return span, purged, false
	}
	if !fromError {
		if rec := inv.completedTools[key]; rec != nil {
			delete(inv.completedTools, key)
			span = inv.rowSpan(rec, nil, ambient)
			span.timestamp = now
			return span, purged, true
		}
	}
	rec := inv.newChild(inv.ownerOf(ctx, id), spanKindTool, now)
	if fromError {
		rec.dropContexts()
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
// context.DeadlineExceeded), it walks up rec's ancestors while they share such
// a context cause and returns that cause if any span on the walk is a parallel
// branch. Checking only the outermost span used to drop a parallel branch's
// cause when a custom WithCancelCause parent sat above the ParallelAgent
// (#1984). A custom agent that cancels its own context.WithCancelCause without
// a parallel branch on the walk still returns nil (and the cause is not
// promoted into retiredCauses by retireAgentLocked), whether the custom agent
// handles the sub-agent's failure or propagates it after the sub-agent already
// emitted content.
func (rec *spanRecord) contextFailureCause() error {
	cause := rec.rawContextFailureCause()
	if cause == nil {
		return nil
	}
	// Keep the cause when any span from rec up through the outermost span that
	// shares it is a parallel branch. Climbing past a ParallelAgent to a custom
	// WithCancelCause parent used to drop the branch's cause because only the
	// outermost span was checked (see #1984). A custom agent alone still
	// returns nil: no span on the walk is a parallel branch.
	for cur := rec; ; cur = cur.parent {
		if cur.isParallelBranch() {
			return cause
		}
		if cur.parent == nil || cur.parent.rawContextFailureCause() == nil {
			return nil
		}
	}
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
	for i := len(inv.agentStack) - 1; i >= 0; i-- {
		if cause := inv.agentStack[i].openFailureCause(); cause != nil {
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

// dropRetiredLateContextsLocked clears the callback contexts of agent spans
// retired from a late trace, and clears parent on any retired ancestor that
// is no longer reachable from the remaining open spans or calls. It avoids
// clearing parent on a retired span that a remaining open span or call still
// reaches through its parent chain (see subtreeLocked and
// launcherOpenLocked).
func (inv *invocationTrace) dropRetiredLateContextsLocked(retired ...*spanRecord) {
	for _, r := range retired {
		r.canceled, r.ended, r.activation, r.retiredCauses = nil, nil, nil, nil
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
		if !keepParent[r] {
			r.parent = nil
		}
	}
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
