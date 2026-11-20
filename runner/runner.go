// Copyright 2025 Google LLC
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

// Package runner provides a runtime for ADK agents.
package runner

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log"
	"maps"
	"slices"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/internal/agent/compactionctx"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
	"google.golang.org/adk/v2/internal/compactioninternal"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/eventmetadata"
	"google.golang.org/adk/v2/internal/llminternal"
	imemory "google.golang.org/adk/v2/internal/memory"
	"google.golang.org/adk/v2/internal/plugininternal"
	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/internal/workflowinternal"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
	"google.golang.org/adk/v2/workflow"
)

// Config is used to create a [Runner].
//
// Exactly one of [Config.Agent] or [Config.Node] must be non-nil.
type Config struct {
	// AppName identifies the application in session, artifact, and memory
	// requests. When Node is set and AppName is empty, New defaults AppName
	// to Node.Name().
	AppName string
	// Agent is the root agent which starts execution. Mutually exclusive
	// with Node.
	Agent agent.Agent
	// Node is the root workflow node (such as a [workflow.BaseNode]
	// embedder, [workflow.FunctionNode], a node built by
	// [workflow.NewDynamicNode], or a [workflow.WorkflowNode]) which starts
	// execution on the node runtime path. Mutually exclusive with Agent.
	//
	// A node root that is not a [workflow.WorkflowNode] receives the user
	// message as a *genai.Content with Role "user"; on a turn that resumes
	// it, it receives the invocation's original user message. A
	// [workflow.WorkflowNode] root passes the message on to each successor
	// of its [workflow.Start] in the form that successor declares (see
	// [workflow.NodeConfig.ContentInput]). The node's events carry the root
	// path "<Node.Name()>@1" in [session.NodeInfo].
	Node           workflow.Node
	SessionService session.Service

	// optional
	ArtifactService artifact.Service
	// optional
	MemoryService memory.Service
	// optional
	PluginConfig PluginConfig
	// optional. When true, Run and RunLive create the session if it does not
	// exist. The session is created when the session ID is empty, so the
	// service generates one, or when the session service's Get reports
	// [session.ErrNotFound]. Any other Get error is returned as is, so a
	// transient storage failure never replaces an existing session.
	AutoCreateSession bool

	// Compaction enables context compaction for the sessions this
	// runner drives, replacing older events with summaries. Nil, the default,
	// disables compaction.
	//
	// The sliding window reduces prompt size by a constant factor rather than
	// bounding it. Only tail retention bounds growth, and it only fires when
	// more events accumulate between sliding-window compactions than
	// EventRetentionSize holds back, so a short interval with a large retention
	// size leaves it idle. See [compaction.Config].
	//
	// When the config names no Summarizer, the runner installs a
	// [compaction.LLMSummarizer] over the root agent's model, which then has to
	// be an LLM agent.
	//
	// optional
	Compaction *compaction.Config

	// See [model.ContextCacheConfig] for when a cache is created and reused,
	// and which models and session services support it.
	//
	// optional
	ContextCache *model.ContextCacheConfig

	// Resumable makes the runner's invocations resumable: an invocation pauses
	// on a long-running tool call, agents record their state and their end in
	// session events, and a later Run with [WithInvocationID], or on the
	// classic path with a function response that answers one of the
	// invocation's calls, resumes the invocation from its last event.
	//
	// A sequential or loop agent stops after the sub-agent whose event
	// pauses the invocation, once that sub-agent's events end; resuming the
	// invocation runs that sub-agent again, but not the sub-agents or loop
	// passes after it. On a resume, an LlmAgent first runs the function
	// calls it made in the invocation that never got a response, other than
	// long-running ones, and pauses again without calling the model while a
	// long-running call is unanswered.
	//
	// Resumption is best effort: in-memory state is lost, and a tool call
	// that ran before the pause may run again after a resume, so tools must
	// tolerate at-least-once calls.
	//
	// EXPERIMENTAL: Config.Resumable is experimental; its API and behavior may change or be removed in a later version.
	Resumable bool
}

// PluginConfig configures the plugins a [Runner] applies and how long it waits
// for them to close.
type PluginConfig struct {
	Plugins      []*plugin.Plugin
	CloseTimeout time.Duration
}

// RunOption customizes a single run invocation.
type RunOption func(*runOptions)

type runOptions struct {
	stateDelta       map[string]any
	yieldUserMessage bool
	invocationID     string
}

// WithStateDelta sets a state delta for the run invocation.
func WithStateDelta(delta map[string]any) RunOption {
	return func(o *runOptions) {
		o.stateDelta = delta
	}
}

// WithYieldUserMessage makes Run yield the user message event (after it is
// appended to the session) before any agent/node events. Mirrors
// adk-python's yield_user_message. Currently honored by the node path.
func WithYieldUserMessage() RunOption {
	return func(o *runOptions) {
		o.yieldUserMessage = true
	}
}

// WithInvocationID makes the run continue the invocation id instead of
// starting a new one. An empty id means a new invocation. It applies to Run;
// RunLive returns an error when it is given.
//
// On a runner whose root is an LlmAgent or a [Config.Node] (the node path),
// the run uses id as its invocation ID and runs the root with the
// invocation's original user message; with no message it appends no user
// event and runs no on-user-message callback. It does not skip an agent that
// already ended.
//
// On any other runner (the classic path), if [Config.Resumable] is true, the
// run resumes the invocation: it restores the original user message when msg
// is nil, appends msg under id when msg is given, rebuilds the agents' states
// from the invocation's events, routes to the agent that was active, and
// yields nothing if that agent already ended. If [Config.Resumable] is false,
// the run's events carry id.
//
// With no message and a non-empty [WithStateDelta], both paths append a user
// event with no content that carries the delta.
//
// On the resumable classic path, Run yields an error and adds no event when
// the session has no events ("session <id> has no events to resume") or when
// no user message of the invocation is stored and msg is nil ("no user
// message available for resuming invocation: <id>").
//
// The option is safe to use from concurrent Run calls.
func WithInvocationID(id string) RunOption {
	return func(o *runOptions) {
		o.invocationID = id
	}
}

// New creates a new [Runner].
//
// Exactly one of cfg.Agent or cfg.Node must be non-nil, and
// cfg.SessionService must be non-nil. When cfg.Node is set, its name must
// not be empty, and an empty cfg.AppName defaults to cfg.Node.Name(). When
// cfg.Compaction is set, it must be valid, and either name a Summarizer or
// have an LLM agent with a model as the root. New returns an error
// describing the first precondition that does not hold. It does not modify
// cfg or cfg.Compaction.
func New(cfg Config) (*Runner, error) {
	if cfg.Agent == nil && cfg.Node == nil {
		return nil, fmt.Errorf("root agent is required")
	}
	if cfg.Agent != nil && cfg.Node != nil {
		return nil, errors.New("only one of Agent or Node may be provided")
	}

	if cfg.SessionService == nil {
		return nil, fmt.Errorf("session service is required")
	}

	appName := cfg.AppName
	var parents parentmap.Map
	var standIn agent.Agent
	if cfg.Agent != nil {
		var err error
		parents, err = parentmap.New(cfg.Agent)
		if err != nil {
			return nil, fmt.Errorf("failed to create agent tree: %w", err)
		}
	} else {
		if cfg.Node.Name() == "" {
			return nil, errors.New("node name must not be empty")
		}
		// adk-python names the app after a node root (Runner._resolve_app).
		if appName == "" {
			appName = cfg.Node.Name()
		}
		var err error
		standIn, err = newStandInAgent(cfg.Node)
		if err != nil {
			return nil, err
		}
	}

	pluginManager, err := plugininternal.NewPluginManager(plugininternal.PluginConfig{
		Plugins:      cfg.PluginConfig.Plugins,
		CloseTimeout: cfg.PluginConfig.CloseTimeout,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create plugin manager: %w", err)
	}

	compactionConfig, err := resolveCompactionConfig(cfg.Compaction, cfg.Agent)
	if err != nil {
		return nil, err
	}

	if err := cfg.ContextCache.Validate(); err != nil {
		return nil, fmt.Errorf("invalid ContextCache: %w", err)
	}
	// A copy of the caller's config, its HTTP options included, so that a
	// later change to the caller's headers, timeout or the maps in its extra
	// body neither reaches running invocations nor races with them.
	contextCache := llminternal.CloneContextCacheConfig(cfg.ContextCache)

	return &Runner{
		appName:           appName,
		rootAgent:         cfg.Agent,
		rootNode:          cfg.Node,
		standInAgent:      standIn,
		sessionService:    cfg.SessionService,
		artifactService:   cfg.ArtifactService,
		memoryService:     cfg.MemoryService,
		parents:           parents,
		pluginManager:     pluginManager,
		autoCreateSession: cfg.AutoCreateSession,
		compactionConfig:  compactionConfig,
		contextCache:      contextCache,
		resumable:         cfg.Resumable,
	}, nil
}

// defaultSummarizerTimeout bounds the summarization call the runner installs
// when an application enables compaction without naming a Summarizer. A var so
// a test can shorten it rather than waiting a minute to prove the bound exists.
var defaultSummarizerTimeout = 60 * time.Second

// resolveCompactionConfig validates cfg and fills in the default summarizer.
// rootAgent is nil when the runner's root is a node.
//
// Resolving at construction time means a misconfigured runner fails fast at
// New, rather than silently skipping compaction turns later, or blowing up
// mid-conversation the first time a compaction triggers.
func resolveCompactionConfig(cfg *compaction.Config, rootAgent agent.Agent) (*compaction.Config, error) {
	if cfg == nil {
		return nil, nil
	}
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid Compaction: %w", err)
	}
	// Copy so the caller's config is not mutated by the summarizer default.
	resolved := *cfg
	if resolved.Summarizer != nil {
		return &resolved, nil
	}

	if rootAgent == nil {
		return nil, errors.New("the Compaction config needs a Summarizer: root is a node, so no default model is available")
	}
	llmAgent, ok := rootAgent.(llminternal.Agent)
	if !ok {
		return nil, fmt.Errorf("the Compaction config needs a Summarizer: root agent %q is not an LLM agent, so no default model is available", rootAgent.Name())
	}
	m := llminternal.Reveal(llmAgent).Model
	if m == nil {
		return nil, fmt.Errorf("the Compaction config needs a Summarizer: root agent %q has no model", rootAgent.Name())
	}
	summarizer, err := compaction.NewLLMSummarizer(compaction.LLMSummarizerConfig{
		Model: m,
		// The settings the summarizer keeps, such as safety settings, govern the
		// summarization call too, rather than it silently falling back to
		// provider defaults for the one call that sees the whole transcript.
		// The summarizer ignores the rest of the config, output limits included.
		GenerateContentConfig: llminternal.Reveal(llmAgent).GenerateContentConfig,
		// A bound on the one call the application did not ask for. Compaction
		// runs inside the run loop, and the post-invocation pass runs from a
		// defer, so a provider that never answers holds the turn open with
		// nothing to show for it. Compaction is an optimisation, so giving up
		// on it is the cheap outcome. An application that wants a different
		// bound supplies its own Summarizer.
		Timeout: defaultSummarizerTimeout,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create the default compaction summarizer: %w", err)
	}
	resolved.Summarizer = summarizer
	return &resolved, nil
}

// NewInMemory creates a [Runner] backed entirely by in-memory session,
// artifact, and memory services, with session auto-creation enabled. It mirrors
// adk-python's InMemoryRunner and is intended for local development and tests,
// not production. Use [New] when you need to supply your own services or
// plugins.
func NewInMemory(appName string, a agent.Agent) (*Runner, error) {
	return New(Config{
		AppName:           appName,
		Agent:             a,
		SessionService:    session.InMemoryService(),
		ArtifactService:   artifact.InMemoryService(),
		MemoryService:     memory.InMemoryService(),
		AutoCreateSession: true,
	})
}

// Runner manages the execution of the agent within a session, handling message
// processing, event generation, and interaction with various services like
// artifact storage, session management, and memory.
type Runner struct {
	appName   string
	rootAgent agent.Agent
	// rootNode is Config.Node; exactly one of rootAgent and rootNode is set.
	rootNode workflow.Node
	// standInAgent is the invocation context's agent when rootNode is set,
	// so that plugins, callbacks and telemetry that read ctx.Agent() see the
	// node's name. It is never run.
	standInAgent    agent.Agent
	sessionService  session.Service
	artifactService artifact.Service
	memoryService   memory.Service

	parents           parentmap.Map
	pluginManager     *plugininternal.PluginManager
	autoCreateSession bool

	// compactionConfig is nil when compaction is disabled. Otherwise it is a
	// validated copy of Config.Compaction with the summarizer
	// resolved.
	compactionConfig *compaction.Config

	// contextCache is nil when context caching is disabled. Otherwise it is
	// a validated copy of Config.ContextCache.
	contextCache *model.ContextCacheConfig

	// resumable is Config.Resumable.
	resumable bool
}

// invocationIDOf returns the invocation an InvocationContext names, or "".
func invocationIDOf(ictx agent.InvocationContext) string {
	if ictx == nil {
		return ""
	}
	return ictx.InvocationID()
}

// compactAfterInvocation runs post-invocation sliding-window compaction and
// persists the summary, if one was produced.
//
// It runs once an invocation has finished and every event it produced has been
// appended, so the compactor sees a complete turn.
//
// A failure is returned rather than swallowed. The turn's own events are
// already committed, so the caller keeps everything the agent produced, and the
// error reports only that history did not shrink. Hiding that would let a
// session grow unbounded, and the first visible symptom would be prompts
// failing against the model's context limit, far from the cause.
//
// The summary itself is deliberately not yielded to the caller. It is
// bookkeeping for the next prompt, not part of the conversation.
func (r *Runner) compactAfterInvocation(ctx context.Context, storedSession session.Session, ictx agent.InvocationContext) (err error) {
	if !compactioninternal.HasSlidingWindow(r.compactionConfig) {
		return nil
	}
	// A Summarizer is third-party code and may panic. summarizeTraced marks the
	// span and lets the panic continue so it stays visible, but every caller of
	// this reaches it from a defer, after the invocation is over and its answer
	// has shipped, so unwinding would take the process down on behalf of an
	// optimisation that had already failed. Tail retention never had the
	// problem, because the workflow scheduler turns a node panic into an error.
	//
	// Here rather than in the callers because there are two of them, one per
	// entry point, and they are copies of each other.
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("%w: post-invocation: compaction panicked: %v", compaction.ErrCompaction, rec)
		}
	}()
	// Tail retention may already have compacted this turn from inside the
	// invocation. Summarizing again the moment it ends would pay for a second
	// model call to re-summarize what was just summarized, and would leave two
	// ranges over the same span. The reference implementation reaches the same
	// outcome by evaluating both strategies in one place and returning early.
	if compactionctx.FromContext(ctx).AlreadyCompacted() {
		return nil
	}
	// Compaction is an optimisation, so a canceled or expired run should not
	// spend a model call on it, nor write a summary the caller never waited
	// for.
	if ctx.Err() != nil {
		return nil
	}

	// Re-read rather than reusing the session handle the invocation began with.
	// That handle is a snapshot taken before the turn ran, so a concurrent
	// invocation on the same session may have appended events it cannot see.
	// Summarizing against it would record a range covering those events without
	// having summarized them, and prompt assembly would then drop them: a lost
	// update rather than a data race, so -race stays clean while conversation
	// goes missing.
	current, err := r.reloadSession(ctx, storedSession)
	if err != nil {
		return fmt.Errorf("%w: post-invocation: %w", compaction.ErrCompaction, err)
	}

	summary, finish, err := compactioninternal.SlidingWindow(ctx, r.compactionConfig, current, invocationIDOf(ictx))
	if err != nil {
		return fmt.Errorf("%w: post-invocation: %w", compaction.ErrCompaction, err)
	}
	if summary == nil {
		return nil
	}

	// Summarizing takes a model call, which is long enough for another
	// invocation to append inside the range just chosen. Re-read once more and
	// abandon the summary if anything landed inside it. Skipping costs one
	// wasted call, where recording it would silently drop those turns from
	// every later prompt.
	if ctx.Err() != nil {
		finish(nil, "the run ended before the summary could be stored")
		return nil
	}
	// raced re-reads the session and reports whether anything landed inside the
	// range since the summary was chosen. It runs twice: here, so a doomed
	// summary does not cost a plugin pass, and again immediately before the
	// append.
	//
	// Once is not enough because a plugin runs in between and that is arbitrary
	// code. Anything appended while it runs falls inside the recorded range and
	// is named by nothing, so prompt assembly drops it. This does not need a
	// hostile plugin or even a concurrent invocation to reach: an event carries
	// the timestamp it was created at rather than the one it was stored at, so
	// parallel tool responses and sub-agent events funneled through a channel
	// are routinely created before the range ends and appended after it.
	// The identities present when the window was chosen, captured once. The
	// race guard compares against the snapshot session; the repair after the
	// append needs the same information once that session has moved on.
	known := compactioninternal.KnownEventIDs(current)
	raced := func() (bool, error) {
		latest, err := r.reloadSession(ctx, storedSession)
		if err != nil {
			return false, err
		}
		return compactioninternal.RangeRaced(latest, current, summary), nil
	}
	discardRaced := func() {
		finish(nil, "another compaction covering the same events landed while summarizing")
		log.Printf("adk: discarding a context compaction summary because the session changed inside its range while summarizing") //nolint:forbidigo // pre-slog call site
	}

	lost, err := raced()
	if err != nil {
		finish(err, "")
		return fmt.Errorf("%w: post-invocation: %w", compaction.ErrCompaction, err)
	}
	if lost {
		discardRaced()
		return nil
	}

	// Plugins see the summary before it is stored, like every other event the
	// runner persists.
	//
	// Appending straight from the compactor skips the one hook that lets a
	// plugin see, rewrite or reject what goes into a session, and a summary is
	// exactly the kind of derived content a redaction plugin would care about.
	//
	// This is deliberately more than the reference does. adk-python appends its
	// compaction summaries from the compactor, straight to the session service,
	// so they never reach run_on_event_callback on either of its paths. An
	// earlier note here claimed the reference yields the event to its runner;
	// it does not. Running the hook is a divergence, in the direction where
	// being wrong is visible: a plugin sees content it might not have, rather
	// than missing content it exists to redact.
	//
	// It applies to this path only. Tail retention appends from inside the
	// invocation and does not run plugins, which is documented on
	// compaction.Config.TokenThreshold because nothing reports it at runtime.
	if ictx != nil && r.pluginManager != nil {
		modified, err := r.pluginManager.RunOnEventCallback(ictx, summary)
		if err != nil {
			finish(err, "")
			return fmt.Errorf("%w: plugin rejected the summary event: %w", compaction.ErrCompaction, err)
		}
		if modified != nil {
			// Re-checked, because a replacement did not go through the builder
			// that filters a summarizer's output. A plugin is trusted to see
			// and rewrite a summary, not to put a function call into the next
			// prompt.
			if !compactioninternal.SanitizeSummary(modified) {
				finish(nil, "a plugin left the summary with nothing usable in it")
				log.Printf("adk: discarding a context compaction summary because a plugin left no usable content in it") //nolint:forbidigo // pre-slog call site
				return nil
			}
			summary = modified
		}
	}

	// The plugin pass above is the widest part of the window, so the check is
	// repeated now that it has run. What remains between here and the append is
	// not closed: shutting that too needs the append itself to be conditional
	// on a session version, which the session.Service interface has no way to
	// express.
	lost, err = raced()
	if err != nil {
		finish(err, "")
		return fmt.Errorf("%w: post-invocation: %w", compaction.ErrCompaction, err)
	}
	if lost {
		discardRaced()
		return nil
	}

	if err := r.sessionService.AppendEvent(ctx, current, summary); err != nil {
		finish(err, "")
		return fmt.Errorf("%w: failed to append the summary event: %w", compaction.ErrCompaction, err)
	}
	finish(nil, "")

	// The append itself cannot be guarded, so what lands during it is repaired
	// after the fact. An event stored between the check above and this line
	// sits inside the recorded range named by nothing, and prompt assembly
	// drops it from every later prompt with no summary standing in for it.
	//
	// A failure here leaves the summary stored and correct for everything
	// except a straggler, which is the state we were in before this existed, so
	// it is logged rather than returned: the turn is long over and its answer
	// was delivered.
	//
	// Detached from the caller's cancellation: the summary is already stored
	// and claims a range wider than it summarized until this lands.
	repairCtx, cancelRepair := compactioninternal.RepairContext(ctx)
	defer cancelRepair()
	if latest, err := r.reloadSession(repairCtx, storedSession); err != nil {
		log.Printf("adk: could not re-read the session to check a stored compaction for stragglers: %v", err) //nolint:forbidigo // pre-slog call site
	} else if repair := compactioninternal.RepairAfterAppend(summary, known, latest); repair != nil {
		if err := r.sessionService.AppendEvent(repairCtx, current, repair); err != nil {
			log.Printf("adk: could not store a corrected compaction record: %v", err) //nolint:forbidigo // pre-slog call site
		} else {
			log.Printf("adk: corrected a compaction record that would have covered %d event(s) it did not summarize", //nolint:forbidigo // pre-slog call site
				len(repair.Actions.Compaction.ExcludedEvents)-len(summary.Actions.Compaction.ExcludedEvents))
		}
	}
	return nil
}

// fromPlugin returns the event a plugin's OnEventCallback leaves behind, with
// the compaction record the framework put on the original rather than any the
// plugin supplied.
//
// When modified is nil or original itself, it returns original. Otherwise it
// returns a new event: a copy of original with every field that modified sets
// laid over it, mirroring adk-python's Runner._get_output_event
// (runners.py). ID, InvocationID and Timestamp always stay the original's,
// and an empty Author falls back to the original's. A field counts as set
// when it is not its zero value; Partial counts as set when it is true or
// PartialSet is true, so a plugin can downgrade a partial event. A non-nil
// CustomMetadata, an empty map included, replaces the original's public keys
// while the original's ADK-internal keys survive (eventmetadata.WithInternalFrom);
// a nil one keeps the original's map. fromPlugin mutates neither modified nor
// any map of original or modified.
//
// A compaction record is not content. It says which stored events every later
// prompt drops and what stands in for them, so planting one erases history and
// substitutes text of the planter's choosing, and the substituted text is not
// filtered the way a summary is. session.EventActions.Compaction says the
// framework writes this field, and for tools and callbacks that is enforced:
// agent.eventActionsFrom, the tool path in base_flow, and workflow.ToolNode all
// clear it. Plugins were the one hook where a returned event was persisted as
// given.
//
// The record is restored rather than cleared, because an ordinary event should
// carry none and a summary carries the real one. Both cases are then the same
// rule: whatever the framework decided, not whatever came back.
//
// A replacement also keeps the identity of the event it replaces: its ID,
// invocation ID and timestamp, and its author when the replacement names none,
// as adk-python's Runner._get_output_event does. adk-python keeps every other
// field the replacement did not set too; a Go struct cannot tell an unset field
// from a zero one, so here the replacement's other fields win as given.
//
// The replacement itself is not written to: the result is a copy, so a plugin
// that returns one shared event for several originals leaves it as it built it.
//
// The post-invocation summary path does not use this. A plugin is invited to
// rewrite a summary there, and SanitizeSummary re-checks the result.
func fromPlugin(original, modified *session.Event, record *session.EventCompaction) *session.Event {
	// The record is restored on whatever comes back, including the original,
	// before any early return.
	//
	// Returning early first meant the strip only ran when the plugin handed
	// back a different pointer. A plugin that mutates the event in place and
	// returns nil, or returns the same pointer, is the ordinary way to write
	// this hook and was the pre-change idiom, and either kept a record the
	// plugin had planted. The framework decides what a compaction record says,
	// not a plugin.
	if modified == nil || modified == original {
		original.Actions.Compaction = record
		return original
	}
	out := *original
	mergeLLMResponse(&out.LLMResponse, &modified.LLMResponse)
	if modified.Author != "" {
		out.Author = modified.Author
	}
	if modified.Branch != "" {
		out.Branch = modified.Branch
	}
	if modified.IsolationScope != "" {
		out.IsolationScope = modified.IsolationScope
	}
	mergeEventActions(&out.Actions, &modified.Actions, record)
	if len(modified.LongRunningToolIDs) > 0 {
		out.LongRunningToolIDs = slices.Clone(modified.LongRunningToolIDs)
	}
	if len(modified.Routes) > 0 {
		out.Routes = slices.Clone(modified.Routes)
	}
	if modified.RequestedInput != nil {
		out.RequestedInput = modified.RequestedInput
	}
	if modified.Output != nil {
		out.Output = modified.Output
	}
	if modified.NodeInfo != nil {
		out.NodeInfo = modified.NodeInfo
	}
	return &out
}

// mergeLLMResponse lays the fields src sets over dst, for fromPlugin.
func mergeLLMResponse(dst, src *model.LLMResponse) {
	if src.Content != nil {
		dst.Content = src.Content
	}
	if src.CitationMetadata != nil {
		dst.CitationMetadata = src.CitationMetadata
	}
	if src.GroundingMetadata != nil {
		dst.GroundingMetadata = src.GroundingMetadata
	}
	if src.UsageMetadata != nil {
		dst.UsageMetadata = src.UsageMetadata
	}
	if src.CacheMetadata != nil {
		dst.CacheMetadata = src.CacheMetadata
	}
	if src.CustomMetadata != nil {
		dst.CustomMetadata = eventmetadata.WithInternalFrom(src.CustomMetadata, dst.CustomMetadata)
	}
	if src.LogprobsResult != nil {
		dst.LogprobsResult = src.LogprobsResult
	}
	if src.InputTranscription != nil {
		dst.InputTranscription = src.InputTranscription
	}
	if src.OutputTranscription != nil {
		dst.OutputTranscription = src.OutputTranscription
	}
	if src.ModelVersion != "" {
		dst.ModelVersion = src.ModelVersion
	}
	// A Go bool cannot say whether a struct literal set it to false, so an
	// explicit downgrade needs PartialSet (model.LLMResponse.PartialSet).
	if src.Partial || src.PartialSet {
		dst.Partial = src.Partial
		dst.PartialSet = src.PartialSet
	}
	if src.TurnComplete {
		dst.TurnComplete = true
	}
	if src.Interrupted {
		dst.Interrupted = true
	}
	if src.SessionResumptionHandle != "" {
		dst.SessionResumptionHandle = src.SessionResumptionHandle
	}
	if src.ErrorCode != "" {
		dst.ErrorCode = src.ErrorCode
	}
	if src.ErrorMessage != "" {
		dst.ErrorMessage = src.ErrorMessage
	}
	if src.FinishReason != "" {
		dst.FinishReason = src.FinishReason
	}
	if src.AvgLogprobs != 0 {
		dst.AvgLogprobs = src.AvgLogprobs
	}
}

// mergeEventActions lays the actions src sets over dst, for fromPlugin, and
// sets dst.Compaction to record. The delta and agent-state maps dst ends up
// with are copies, because the session service applies and keeps them.
func mergeEventActions(dst, src *session.EventActions, record *session.EventCompaction) {
	if len(src.StateDelta) > 0 {
		dst.StateDelta = maps.Clone(src.StateDelta)
	} else if dst.StateDelta != nil {
		dst.StateDelta = maps.Clone(dst.StateDelta)
	}
	if len(src.ArtifactDelta) > 0 {
		dst.ArtifactDelta = maps.Clone(src.ArtifactDelta)
	} else if dst.ArtifactDelta != nil {
		dst.ArtifactDelta = maps.Clone(dst.ArtifactDelta)
	}
	if len(src.RequestedToolConfirmations) > 0 {
		dst.RequestedToolConfirmations = maps.Clone(src.RequestedToolConfirmations)
	}
	if len(src.RequestedAuthConfigs) > 0 {
		dst.RequestedAuthConfigs = maps.Clone(src.RequestedAuthConfigs)
	}
	if src.SkipSummarization {
		dst.SkipSummarization = true
	}
	if src.TransferToAgent != "" {
		dst.TransferToAgent = src.TransferToAgent
	}
	if src.Escalate {
		dst.Escalate = true
	}
	if src.SetModelResponse != nil {
		dst.SetModelResponse = src.SetModelResponse
	}
	if src.RewindBeforeInvocationID != "" {
		dst.RewindBeforeInvocationID = src.RewindBeforeInvocationID
	}
	if src.EndOfAgent != nil {
		dst.EndOfAgent = src.EndOfAgent
	}
	if src.AgentState != nil {
		dst.AgentState = maps.Clone(src.AgentState)
	} else if dst.AgentState != nil {
		dst.AgentState = maps.Clone(dst.AgentState)
	}
	dst.Compaction = record
}

// processEventWithPlugins stamps the event's timestamp if it is zero, merges
// the run's custom metadata into event, runs the OnEventCallback plugins and
// applies fromPlugin. When a plugin returns a non-nil event, a replacement or
// the same event, the run's custom metadata is merged into that event again,
// its own keys winning; a nil return keeps the plugin's in-place edits as they
// are. This mirrors adk-python's Runner._process_event_with_plugin_callbacks
// and _get_output_event (runners.py).
//
// It does not persist the event: the caller decides on the returned event, with
// shouldAppendEvent, whether to append it to the session. A runner-synthesized event goes through
// this function so it gets the run's custom metadata before OnEventCallback,
// like agent output events. event must be non-nil. On a plugin error it
// returns (nil, err) unchanged.
func (r *Runner) processEventWithPlugins(ctx agent.InvocationContext, event *session.Event) (*session.Event, error) {
	stampTimestamp(ctx, event)
	runMetadata := runMetadataFromContext(ctx)
	eventmetadata.MergeIntoEvent(event, runMetadata)
	if r.pluginManager == nil {
		return event, nil
	}
	// Captured before the plugin runs. A plugin that mutates the event in
	// place has already overwritten this field by the time the callback
	// returns, so reading it afterwards reads the plugin's value rather than
	// the framework's.
	record := event.Actions.Compaction
	modifiedEvent, err := r.pluginManager.RunOnEventCallback(ctx, event)
	if err != nil {
		return nil, err
	}
	event = fromPlugin(event, modifiedEvent, record)
	stampTimestamp(ctx, event) // a plugin may have cleared it in place
	// adk-python merges again for every returned event, the same object
	// included, and only a None return skips it (runners.py,
	// _get_output_event).
	if modifiedEvent != nil {
		eventmetadata.MergeIntoEvent(event, runMetadata)
	}
	return event, nil
}

// runMetadataFromContext returns the invocation's custom metadata, which Run
// or RunLive filtered with eventmetadata.WithoutInternal and stored on the
// context's runconfig.RunConfig. It returns nil when ctx carries none.
func runMetadataFromContext(ctx context.Context) map[string]any {
	if rc := runconfig.FromContext(ctx); rc != nil {
		return rc.CustomMetadata
	}
	return nil
}

// stampTimestamp gives an event that an agent yielded, or a plugin returned in
// its place, without a timestamp the time the runner received it, as
// adk-python's Event does on construction.
//
// An event built as a struct literal never passes through session.NewEvent.
// Stored with a zero timestamp it moves the session's last update time back to
// year 1, and a backend that orders events by time, such as session/database,
// reads it back ahead of every message before it.
func stampTimestamp(ctx context.Context, event *session.Event) {
	if event.Timestamp.IsZero() {
		event.Timestamp = platform.Now(ctx)
	}
}

// compactionRuntime returns the runtime that the request processors read off
// the context, both to gate prompt assembly on compaction being configured and
// to run intra-invocation compaction. It is nil when compaction is disabled for
// this runner.
func (r *Runner) compactionRuntime() *compactionctx.Runtime {
	return compactionctx.New(r.compactionConfig, r.sessionService)
}

// withContextCache returns ctx carrying this runner's context cache config and
// a ledger of its own for the invocation. Both are set even when caching is
// off, so an invocation never sees the config or ledger of a run it is nested
// in, such as an agent tool's caller.
func (r *Runner) withContextCache(ctx context.Context) context.Context {
	ctx = llminternal.WithContextCacheConfig(ctx, r.contextCache)
	var ledger *llminternal.ContextCacheLedger
	if r.contextCache != nil {
		ledger = llminternal.NewContextCacheLedger()
	}
	return llminternal.WithContextCacheLedger(ctx, ledger)
}

// contextCacheSettleTimeout bounds reading the stored session once an
// invocation has ended, which a hung store would otherwise hold open. A var so
// a test can shorten it rather than waiting for the bound to pass.
var contextCacheSettleTimeout = 30 * time.Second

// settleContextCaches deletes the context caches the invocation's model calls
// created or stopped using, except those the latest stored record of an agent
// names. It runs once the invocation has ended, however it ended. A canceled
// invocation still settles, or its caches would stay on the provider until
// their TTL runs out.
//
// It judges against the session as the session service holds it, not against
// the invocation's own handle. A service may add an event to the handle before
// it writes the event, and leave it there when the write fails, as the
// database service does, so the handle can name a cache the store never
// recorded. Compaction re-reads the session for the same reason. The read
// happens only when a cache was created or stopped being used, so most
// invocations make none.
func (r *Runner) settleContextCaches(ctx context.Context, sess session.Session) {
	ledger := llminternal.ContextCacheLedgerFromContext(ctx)
	if ledger == nil || sess == nil {
		return
	}
	ledger.Settle(context.WithoutCancel(ctx), func(ctx context.Context) (session.Events, error) {
		ctx, cancel := context.WithTimeout(ctx, contextCacheSettleTimeout)
		defer cancel()
		stored, err := r.reloadSession(ctx, sess)
		if err != nil {
			return nil, err
		}
		return stored.Events(), nil
	})
}

// reloadSession re-fetches a session so that compaction and context cache
// cleanup work against current state rather than the snapshot the invocation
// began with.
func (r *Runner) reloadSession(ctx context.Context, s session.Session) (session.Session, error) {
	resp, err := r.sessionService.Get(ctx, &session.GetRequest{
		AppName:   s.AppName(),
		UserID:    s.UserID(),
		SessionID: s.ID(),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to re-read the session: %w", err)
	}
	if resp == nil || resp.Session == nil {
		return nil, fmt.Errorf("session %q disappeared before it could be re-read", s.ID())
	}
	return resp.Session, nil
}

// getOrCreateSession loads the session with the event limits of sessionCfg,
// creating it when it does not exist and the runner auto-creates sessions.
func (r *Runner) getOrCreateSession(ctx context.Context, userID, sessionID string, sessionCfg agent.GetSessionConfig) (session.Session, error) {
	getResp, err := r.sessionService.Get(ctx, &session.GetRequest{
		AppName:         r.appName,
		UserID:          userID,
		SessionID:       sessionID,
		NumRecentEvents: sessionCfg.NumRecentEvents,
		After:           sessionCfg.After,
	})
	if err == nil {
		return getResp.Session, nil
	}
	// Services reject an empty ID in Get with a validation error, not
	// ErrNotFound. Create generates an ID for it.
	if !r.autoCreateSession || (sessionID != "" && !errors.Is(err, session.ErrNotFound)) {
		return nil, err
	}
	createResp, err := r.sessionService.Create(ctx, &session.CreateRequest{
		AppName:   r.appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		return nil, err
	}
	return createResp.Session, nil
}

// Run runs the agent for the given user input, yielding events from agents.
// For each user message it finds the proper agent within an agent tree to
// continue the conversation within the session.
//
// A root LlmAgent must run in chat mode (the default) or task mode; any other
// mode ends the run with an error before the session is read or created. The
// session is loaded with the event window of cfg.GetSessionConfig, read when
// Run is called. The on-user-message plugin callbacks run before the route is
// final, so a replacement message that answers a sub-agent's call is routed to
// that sub-agent. The message they leave is refused, with nothing stored, when
// it carries a function call ("user message cannot contain function calls"),
// or when it has no parts and the root agent is not an LlmAgent ("no parts in
// the new_message").
//
// A root node (Config.Node) runs on the node path; see [Config.Node] for the
// input it receives and the paths its events carry. On a node root, Run also
// refuses, with nothing stored, a message that holds both text and a
// function response with an ID ("message cannot contain both function
// responses and text: ..."); a BeforeRunCallback reply is authored "model";
// and the node's events carry no workflow bookkeeping metadata. On any root,
// a nil msg with a non-empty [WithStateDelta] stores a user event without
// content that carries the delta, which [WithYieldUserMessage] yields on the
// node path.
//
// A user message whose first function response answers a call made in the
// same invocation is stored with that call's branch, and with the call's
// isolation scope when the message has none of its own. A sub-agent that a
// function response or a resume sends back to work continues on the branch
// of its newest event in the invocation it continues that a rewind did not
// remove and that is not on a tool's message branch.
func (r *Runner) Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg agent.RunConfig, opts ...RunOption) iter.Seq2[*session.Event, error] {
	// Copied now: the sequence below runs only when ranged, and later
	// changes to the caller's map must not reach the invocation's events.
	var runMetadata map[string]any
	if cfg.CustomMetadata != nil {
		runMetadata = eventmetadata.WithoutInternal(*cfg.CustomMetadata)
	}
	sessionCfg := getSessionConfigOf(cfg.GetSessionConfig)
	return telemetry.InstrumentInvocation(ctx, r.invocationRoot(), sessionID, func(ctx context.Context) iter.Seq2[*session.Event, error] {
		return r.run(ctx, userID, sessionID, msg, cfg, runMetadata, sessionCfg, opts...)
	})
}

// invocationRoot returns the root whose name and description the invocation
// span records: the root node when the runner has one, else the root agent.
func (r *Runner) invocationRoot() telemetry.AgentLike {
	if r.rootNode != nil {
		return r.rootNode
	}
	return r.rootAgent
}

// getSessionConfigOf returns a copy of *cfg, or the zero GetSessionConfig
// (no event window) when cfg is nil.
func getSessionConfigOf(cfg *agent.GetSessionConfig) agent.GetSessionConfig {
	if cfg == nil {
		return agent.GetSessionConfig{}
	}
	return *cfg
}

// run is the body of Run. runMetadata is the caller's RunConfig.CustomMetadata
// without ADK-internal keys, and sessionCfg its RunConfig.GetSessionConfig,
// both copied when Run was called.
func (r *Runner) run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg agent.RunConfig, runMetadata map[string]any, sessionCfg agent.GetSessionConfig, opts ...RunOption) iter.Seq2[*session.Event, error] {
	// TODO(hakim): we need to validate whether cfg is compatible with the Agent.
	//   see adk-python/src/google/adk/runners.py Runner._new_invocation_context.
	// TODO: setup tracer.
	return func(yield func(*session.Event, error) bool) {
		// An invocation that ended in an error is not a finished turn and must
		// not be summarized: the window would hold a question with no answer,
		// and that summary is stored permanently and degrades every later
		// prompt. Observing it here, rather than at each error site, means no
		// path can forget to.
		invocationFailed := false
		emit := yield
		yield = func(ev *session.Event, err error) bool {
			if err != nil {
				invocationFailed = true
			}
			return emit(ev, err)
		}

		options := runOptions{}
		for _, opt := range opts {
			opt(&options)
		}

		// The inbound message is always a user turn, whatever Role the caller
		// set. Normalize it once here so that findAgentToRun, the invocation
		// context and the stored event all see the same message.
		msg = asUserContent(msg)

		// A root LlmAgent runs in chat mode (the default) or task mode. The
		// mode is checked before the session is read or created, as
		// adk-python does (runners.py, Runner.run_async), so a misconfigured
		// root leaves no session behind.
		var llmInternalState *llminternal.State
		var rootMode llminternal.Mode
		if llmInternalAgent, ok := r.rootAgent.(llminternal.Agent); ok {
			llmInternalState = llminternal.Reveal(llmInternalAgent)
			rootMode = llminternal.ResolveMode(llmInternalState.Mode, llminternal.ModeChat)
			if rootMode != llminternal.ModeChat && rootMode != llminternal.ModeTask {
				yield(nil, fmt.Errorf("LlmAgent as root agent must have mode='chat' or 'task', but got mode='%s'", rootMode))
				return
			}
		}

		storedSession, err := r.getOrCreateSession(ctx, userID, sessionID, sessionCfg)
		if err != nil {
			yield(nil, err)
			return
		}

		// A node root always runs on the node path, as adk-python's
		// Runner.run_async sends a BaseNode root to _run_node_async.
		if r.rootNode != nil {
			r.runNode(ctx, storedSession, r.standInAgent, msg, cfg, runMetadata, llminternal.ModeUnset, options, yield)
			return
		}

		// Node path: an LlmAgent runs through the ADK 2.0 node runtime
		// (the Go equivalent of adk-python's _run_node_async, reached for
		// an LlmAgent root).
		if llmInternalState != nil {
			// Record the root's placement, as every other placement does, so
			// the runner is not the one placement that resolves a mode and
			// then keeps it to itself.
			//
			// Kept in a local rather than assigned back to ctx, this closure's
			// captured parameter, so ranging the returned iterator twice cannot
			// accumulate wrappers. The non-LlmAgent path below does assign back,
			// but this branch returns before reaching it, so on this path the
			// iterator really is clean for a re-range.
			rootCtx := llminternal.WithBoundMode(ctx, r.rootAgent.Name(), llmInternalState, rootMode)

			agentToRun := r.rootAgent
			if r.usesLegacySubAgentPicker(rootMode) {
				agentToRun, err = r.findAgentToRun(storedSession, msg)
				if err != nil {
					yield(nil, err)
					return
				}
			}

			r.runNode(rootCtx, storedSession, agentToRun, msg, cfg, runMetadata, rootMode, options, yield)
			return
		}

		// A resumable runner treats an invocation ID, given by the caller or
		// resolved from a function response, as a resume. Its refusals come
		// before anything is stored (adk-python Runner.run_async and
		// Runner._setup_context_for_resumed_invocation).
		userContent := msg
		invocationID, spansInvocations := resolveInvocationID(storedSession, msg)
		if options.invocationID != "" {
			invocationID = options.invocationID
		}
		resuming := r.resumable && invocationID != ""
		var agentStates map[string]map[string]any
		var ended map[string]bool
		if resuming {
			userContent, err = resumedUserContent(storedSession, msg, invocationID)
			if err != nil {
				yield(nil, err)
				return
			}
			agentStates, ended = rebuildAgentStates(storedSession.Events(), invocationID)
		}

		ctx = parentmap.ToContext(ctx, r.parents)
		ctx = runconfig.ToContext(ctx, &runconfig.RunConfig{
			StreamingMode:  runconfig.StreamingMode(cfg.StreamingMode),
			MaxLLMCalls:    runconfig.ResolveMaxLLMCalls(cfg.MaxLLMCalls),
			HTTPOptions:    cfg.HTTPOptions,
			Labels:         cfg.Labels,
			CustomMetadata: runMetadata,
			Resumable:      r.resumable,
			AgentStates:    agentStates,
		})
		ctx = r.withPluginManager(ctx)
		ctx = compactionctx.ToContext(ctx, r.compactionRuntime())
		ctx = r.withContextCache(ctx)
		defer r.settleContextCaches(ctx, storedSession)

		// Compaction has to happen however iteration ends. Breaking out of the
		// range loop on the terminal event is the ordinary streaming idiom, and
		// what the A2A executor does, so a hook placed only after the loop
		// would never run for those callers and compaction would silently never
		// happen. Deferring makes it unconditional.
		//
		// On an early exit the error cannot be yielded, because yield must not
		// be called once it has returned false, so it is logged instead.
		// Assigned once the invocation context exists, below. The compaction
		// hook runs from a defer, so it reads whatever this holds by then.
		var invocationCtx agent.InvocationContext

		compacted := false
		compactOnce := func() error {
			if compacted || invocationFailed {
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

		// A resume routes once the message is stored, below.
		agentToRun, branch := r.rootAgent, ""
		if !resuming {
			agentToRun, branch = r.classicAgentToRun(storedSession, msg)
			// Answers to calls from several invocations start a new invocation
			// when they go back to the calling agent, as adk-python does for a
			// BaseAgent root of an app that is not resumable. A graph workflow
			// resumes a paused run from its invocation's events, so it keeps the
			// first call's invocation; adk-python raises for a Workflow root. So
			// does an answer whose caller is not in the agent tree, since that
			// caller can be a workflow node agent missing from the workflow's
			// sub-agents; adk-python starts a new invocation there. An ID the
			// caller gives with WithInvocationID is kept.
			if spansInvocations && options.invocationID == "" && agentToRun != r.rootAgent {
				invocationID = ""
			}
		}

		ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
			Artifacts:    artifacts,
			Memory:       memoryImpl,
			Session:      storedSession,
			Branch:       branch,
			Agent:        agentToRun,
			UserContent:  userContent,
			RunConfig:    &cfg,
			InvocationID: invocationID,
		})
		invocationCtx = ic
		ctx := agent.NewContext(ic)
		ctx, _, replaced, err := r.appendMessageToSession(ctx, storedSession, msg, cfg.SaveInputBlobsAsArtifacts, true, r.pluginManager, options.stateDelta)
		if err != nil {
			yield(nil, err)
			return
		}
		switch {
		case resuming:
			agentToRun, branch, err = r.resumedAgentToRun(storedSession, invocationID, ended)
			if err != nil {
				yield(nil, err)
				return
			}
			// An agent that already ended its run in the invocation is not run
			// again (adk-python Runner.run_async).
			if ended[agentToRun.Name()] {
				return
			}
			ctx = ctx.WithDelta(&agent.CommonContextDelta{
				InvocationContextDelta: &agent.InvocationContextDelta{Agent: &agentToRun, Branch: &branch},
			})
			invocationCtx = ctx
		case replaced:
			// A plugin's replacement can answer a call another agent made, so
			// the route is taken again from the replacement, now in history.
			agentToRun, branch = r.classicAgentToRun(storedSession, ctx.UserContent())
			ctx = ctx.WithDelta(&agent.CommonContextDelta{
				InvocationContextDelta: &agent.InvocationContextDelta{Agent: &agentToRun, Branch: &branch},
			})
			invocationCtx = ctx
		}

		pluginManager := r.pluginManager
		if pluginManager != nil {
			// Defer the after run callbacks to perform global cleanup tasks or finalizing logs and metrics data.
			// This does NOT emit any event.
			defer pluginManager.RunAfterRunCallback(ctx)

			if event, err := r.runBeforeRunCallback(ctx, false); event != nil || err != nil {
				yield(event, err)
				return
			}
		}

		for event, err := range agentToRun.Run(ctx) {
			if err != nil {
				// The model-call budget is spent for the whole invocation, so
				// nothing that runs after this can reach a model. Ending here
				// stops a root agent that carries on after a sub-agent error
				// from turning the rest of the run into a stream of limit
				// errors, as adk-python does.
				if !yield(event, err) || errors.Is(err, agent.ErrLLMCallsLimitExceeded) {
					return
				}
				continue
			}

			if event == nil {
				err := fmt.Errorf("adk: agent %q yielded a nil event", agentToRun.Name())
				log.Printf("%v", err) //nolint:forbidigo // pre-slog call site
				yield(nil, err)
				return
			}
			stampTimestamp(ctx, event)

			event, err = r.processEventWithPlugins(ctx, event)
			if err != nil {
				if !yield(nil, err) {
					return
				}
				continue
			}
			event = clearMessageAsOutput(event)

			if shouldAppendEvent(event, false) {
				if err := r.sessionService.AppendEvent(ctx, storedSession, event); err != nil {
					yield(nil, fmt.Errorf("failed to add event to session: %w", err))
					return
				}
			}

			if !yield(event, nil) {
				return
			}
		}

		// Compact once the invocation is done and every event it produced has
		// been persisted. Never mid-invocation: that is tail retention's job.
		//
		// compactOnce is idempotent because the deferred call above also runs
		// it. Reaching it here means the consumer drained the stream, so a
		// failure can still be reported.
		if err := compactOnce(); err != nil {
			yield(nil, err)
			return
		}
	}
}

// runBeforeRunCallback processes and persists a plugin's replacement reply.
// A nil event and nil error mean agent execution should continue. The reply
// goes through the OnEventCallback plugins and is appended to the session
// when shouldAppendEvent(reply, isLive) holds; isLive is true for RunLive.
func (r *Runner) runBeforeRunCallback(ctx agent.InvocationContext, isLive bool) (*session.Event, error) {
	content, err := r.pluginManager.RunBeforeRunCallback(ctx)
	if err != nil || content == nil {
		return nil, err
	}

	event := session.NewEvent(ctx, ctx.InvocationID())
	if r.rootNode != nil {
		// adk-python authors a node root's before-run reply "model"
		// (workflow/_node_runner_utils.py).
		event.Author = genai.RoleModel
	} else {
		// Match RunLive's attribution so later turns can resolve the agent from history.
		event.Author = ctx.Agent().Name()
	}
	event.Content = content

	event, err = r.processEventWithPlugins(ctx, event)
	if err != nil {
		return nil, err
	}
	// Stamp after OnEvent so a replacement keeps its BeforeRun origin, without
	// changing plugin-owned metadata.
	reply := *event
	reply.CustomMetadata = make(map[string]any, len(event.CustomMetadata)+1)
	maps.Copy(reply.CustomMetadata, event.CustomMetadata)
	reply.CustomMetadata[plugininternal.BeforeRunReplyKey] = true
	event = &reply
	stampTimestamp(ctx, event) // a replacement may be a struct literal
	if shouldAppendEvent(event, isLive) {
		if err := r.sessionService.AppendEvent(ctx, ctx.Session(), event); err != nil {
			return nil, fmt.Errorf("failed to add event to session: %w", err)
		}
	}
	return event, nil
}

// errUserFunctionCall rejects a user message that carries a function call.
// Only a model makes function calls, so storing one would put a call no model
// produced into the history.
var errUserFunctionCall = errors.New("user message cannot contain function calls")

type liveAgent interface {
	RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
}

// RunLive runs a live session for the agent, supporting bidirectional streaming.
type runnerLiveSession struct {
	sess          agent.LiveSession
	r             *Runner
	iCtx          agent.InvocationContext
	storedSession session.Session
}

func (s *runnerLiveSession) Send(req agent.LiveRequest) error {
	if req.Content != nil && slices.Contains(req.Content.Parts, nil) {
		// A nil part carries no content, so it is dropped; adk-python rejects
		// the request instead.
		content := *req.Content
		content.Parts = slices.DeleteFunc(slices.Clone(content.Parts), func(p *genai.Part) bool {
			return p == nil
		})
		req.Content = &content
	}
	if len(utils.FunctionCalls(req.Content)) > 0 {
		return errUserFunctionCall
	}
	err := s.sess.Send(req)
	if err != nil {
		return err
	}

	// Save user text content to session history
	if req.Content != nil && len(req.Content.Parts) > 0 {
		// Skip function responses - they are handled separately
		isFunctionResponse := false
		for _, part := range req.Content.Parts {
			if part.FunctionResponse != nil {
				isFunctionResponse = true
				break
			}
		}

		if !isFunctionResponse {
			event := session.NewEvent(s.iCtx, s.iCtx.InvocationID())
			event.Author = "user"
			event.LLMResponse = model.LLMResponse{
				Content: req.Content,
			}
			if err := s.r.sessionService.AppendEvent(s.iCtx, s.storedSession, event); err != nil {
				return fmt.Errorf("failed to add user event to session: %w", err)
			}
		}
	}

	return nil
}

func (s *runnerLiveSession) Close() error {
	return s.sess.Close()
}

type closedLiveSession struct{}

func (s *closedLiveSession) Send(req agent.LiveRequest) error {
	return fmt.Errorf("session is closed")
}

func (s *closedLiveSession) Close() error {
	return nil
}

// RunLive starts a live, bidirectional agent session, returning the session
// and a stream of events from the agents. It returns an error, before the
// session is read, when the runner's root is a node or when opts include
// [WithInvocationID].
func (r *Runner) RunLive(ctx context.Context, userID, sessionID string, cfg agent.LiveRunConfig, opts ...RunOption) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	if r.rootNode != nil {
		return nil, nil, fmt.Errorf("runner root %q is a node, not an agent", r.rootNode.Name())
	}
	options := runOptions{}
	for _, opt := range opts {
		opt(&options)
	}
	// adk-python's run_live takes no invocation ID.
	if options.invocationID != "" {
		return nil, nil, errors.New("RunLive does not support WithInvocationID; resume an invocation with Run")
	}

	storedSession, err := r.getOrCreateSession(ctx, userID, sessionID, getSessionConfigOf(cfg.GetSessionConfig))
	if err != nil {
		return nil, nil, err
	}

	// msg is nil for Live run as it's streaming
	agentToRun, err := r.findAgentToRun(storedSession, nil)
	if err != nil {
		return nil, nil, err
	}

	lAgent, ok := agentToRun.(liveAgent)
	if !ok {
		return nil, nil, fmt.Errorf("agent %s does not support Live Run", agentToRun.Name())
	}

	ctx = parentmap.ToContext(ctx, r.parents)
	ctx = runconfig.ToContext(ctx, &runconfig.RunConfig{
		StreamingMode:  runconfig.StreamingModeBidi, // Live is always bidirectional streaming
		Live:           &cfg,
		CustomMetadata: eventmetadata.WithoutInternal(cfg.CustomMetadata),
		Resumable:      r.resumable,
	})
	ctx = r.withPluginManager(ctx)
	// Deliberately no compactionctx here: context compaction does not apply to
	// live runs. A live session streams over a persistent connection instead of
	// re-sending assembled history each turn, so replacing older events with a
	// summary would not shrink anything.

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

	iCtx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Artifacts:   artifacts,
		Memory:      memoryImpl,
		Session:     storedSession,
		Agent:       agentToRun,
		UserContent: nil,
	})

	if r.pluginManager != nil {
		// The reply goes through OnEvent and the run through AfterRun, as on
		// Run. adk-python runs a live agent through the same
		// Runner._exec_with_plugin as run_async (runners.py).
		earlyExitEvent, err := r.runBeforeRunCallback(iCtx, true)
		if err != nil {
			return nil, nil, err
		}
		if earlyExitEvent != nil {
			earlyExitIter := func(yield func(*session.Event, error) bool) {
				defer r.pluginManager.RunAfterRunCallback(iCtx)
				yield(earlyExitEvent, nil)
			}
			return &closedLiveSession{}, earlyExitIter, nil
		}
	}

	agentSess, innerIter, err := lAgent.RunLive(iCtx)
	if err != nil {
		return nil, nil, err
	}

	wrappedIter := func(yield func(*session.Event, error) bool) {
		if r.pluginManager != nil {
			defer r.pluginManager.RunAfterRunCallback(iCtx)
		}

		var bufferedEvents []*session.Event
		isTranscribing := false

		for event, err := range innerIter {
			if err != nil {
				if !yield(nil, err) {
					return
				}
				continue
			}

			if event == nil {
				err := fmt.Errorf("adk: agent %q yielded a nil event", agentToRun.Name())
				if len(bufferedEvents) > 0 {
					ids := make([]string, 0, len(bufferedEvents))
					for _, bufferedEvent := range bufferedEvents {
						ids = append(ids, bufferedEvent.ID)
					}
					err = fmt.Errorf("%w; discarded buffered event IDs: %q", err, ids)
				}
				log.Printf("%v", err) //nolint:forbidigo // pre-slog call site
				yield(nil, err)
				return
			}
			stampTimestamp(iCtx, event)

			event, err = r.processEventWithPlugins(iCtx, event)
			if err != nil {
				if !yield(nil, err) {
					return
				}
				continue
			}
			event = clearMessageAsOutput(event)

			// Chronological event buffering logic for Live streaming.
			// Holds back tool calls/responses if they arrive before the transcription finishes.
			if event.LLMResponse.Partial && (event.LLMResponse.InputTranscription != nil || event.LLMResponse.OutputTranscription != nil) {
				isTranscribing = true
			}

			isToolCallOrResp := false
			if event.LLMResponse.Content != nil {
				for _, part := range event.LLMResponse.Content.Parts {
					if part != nil && (part.FunctionCall != nil || part.FunctionResponse != nil) {
						isToolCallOrResp = true
						break
					}
				}
			}

			if isTranscribing && isToolCallOrResp {
				bufferedEvents = append(bufferedEvents, event)
				continue
			}

			if !event.LLMResponse.Partial {
				if event.LLMResponse.InputTranscription != nil || event.LLMResponse.OutputTranscription != nil {
					isTranscribing = false

					if shouldAppendEvent(event, true) {
						if err := r.sessionService.AppendEvent(iCtx, storedSession, event); err != nil {
							if !yield(nil, fmt.Errorf("failed to add event to session: %w", err)) {
								return
							}
							continue
						}
					}
					if !yield(event, nil) {
						return
					}

					for _, bufferedEvent := range bufferedEvents {
						if shouldAppendEvent(bufferedEvent, true) {
							if err := r.sessionService.AppendEvent(iCtx, storedSession, bufferedEvent); err != nil {
								if !yield(nil, fmt.Errorf("failed to add event to session: %w", err)) {
									return
								}
								continue
							}
						}
						if !yield(bufferedEvent, nil) {
							return
						}
					}
					bufferedEvents = nil
					continue
				}
			}

			if shouldAppendEvent(event, true) {
				if err := r.sessionService.AppendEvent(iCtx, storedSession, event); err != nil {
					if !yield(nil, fmt.Errorf("failed to add event to session: %w", err)) {
						return
					}
					continue
				}
			}

			if !yield(event, nil) {
				return
			}
		}

		// innerIter has returned; every downstream stop above returns from wrappedIter.
		// Live agents reach here when their session closes, including on cancellation.
		// Python persists live events as they arrive; Go buffers them during transcription.
		// Detach cancellation so session teardown can still persist that buffer.
		// Like the mid-stream flush, store each buffered event that
		// shouldAppendEvent accepts, and yield every one.
		flushCtx := context.WithoutCancel(iCtx)
		for _, bufferedEvent := range bufferedEvents {
			if shouldAppendEvent(bufferedEvent, true) {
				if err := r.sessionService.AppendEvent(flushCtx, storedSession, bufferedEvent); err != nil {
					if !yield(nil, fmt.Errorf("failed to add event to session: %w", err)) {
						return
					}
					continue
				}
			}
			if !yield(bufferedEvent, nil) {
				return
			}
		}
	}

	return &runnerLiveSession{
		sess:          agentSess,
		r:             r,
		iCtx:          iCtx,
		storedSession: storedSession,
	}, wrappedIter, nil
}

// appendMessageToSession runs the on-user-message callbacks on msg, refuses
// the resulting message if it is not a valid user turn, and appends it to
// storedSession as the user event. requireParts refuses a message with no
// parts, which only the classic path does (adk-python
// Runner._append_new_message_to_session; the node path's _append_user_event
// stores it). replaced reports whether a callback returned a replacement; the
// replacement is then ctx's UserContent, and when it answers a function call
// in history, ctx carries that call's invocation ID. On a node root, a
// message that holds both text and a function response with an ID is
// refused before and after the callbacks. A nil msg runs no callback and
// appends a content-less user event carrying stateDelta when stateDelta is
// not empty, and nothing otherwise (adk-python
// Runner._append_state_delta_event).
func (r *Runner) appendMessageToSession(ctx agent.Context, storedSession session.Session, msg *genai.Content, saveInputBlobsAsArtifacts, requireParts bool, pluginManager *plugininternal.PluginManager, stateDelta map[string]any) (_ agent.Context, _ *session.Event, replaced bool, _ error) {
	if msg == nil {
		if len(stateDelta) == 0 {
			return ctx, nil, false, nil
		}
		event, err := r.appendUserEvent(ctx, storedSession, nil, stateDelta)
		return ctx, event, false, err
	}
	if err := r.validateNodeRootMessage(msg); err != nil {
		return ctx, nil, false, err
	}
	if pluginManager != nil {
		modifiedMsg, err := pluginManager.RunOnUserMessageCallback(ctx, msg)
		if err != nil {
			return ctx, nil, false, fmt.Errorf("error running on run user message callback : %w", err)
		}
		if modifiedMsg != nil {
			replaced = true
			// A plugin can return a message with any Role. Normalize it before
			// it becomes the invocation context's UserContent: WithDelta copies
			// the value it points to, so a later reassignment would not reach it.
			msg = asUserContent(modifiedMsg)

			// A replacement that answers an earlier call continues that call's
			// invocation (adk-python workflow/_node_runner_utils.py). The ID is
			// set before WithDelta, which copies the invocation context.
			if id, _ := resolveInvocationID(storedSession, msg); id != "" {
				if err := setInvocationID(ctx, id); err != nil {
					return ctx, nil, replaced, err
				}
			}

			ctx = ctx.WithDelta(&agent.CommonContextDelta{
				InvocationContextDelta: &agent.InvocationContextDelta{
					UserContent: &msg,
				},
			})
			if err := r.validateNodeRootMessage(msg); err != nil {
				return ctx, nil, replaced, err
			}
		}
	}

	if requireParts && len(msg.Parts) == 0 {
		return ctx, nil, replaced, errors.New("no parts in the new_message")
	}
	if len(utils.FunctionCalls(msg)) > 0 {
		return ctx, nil, replaced, errUserFunctionCall
	}

	artifactsService := ctx.Artifacts()
	if artifactsService != nil && saveInputBlobsAsArtifacts {
		for i, part := range msg.Parts {
			// A nil part carries no content, so it is skipped; adk-python rejects
			// the request instead.
			if part == nil || part.InlineData == nil {
				continue
			}
			fileName := fmt.Sprintf("artifact_%s_%d", ctx.InvocationID(), i)
			if _, err := artifactsService.Save(ctx, fileName, part); err != nil {
				return ctx, nil, replaced, fmt.Errorf("failed to save artifact %s: %w", fileName, err)
			}
			// Replace the part with a text placeholder
			msg.Parts[i] = &genai.Part{
				Text: fmt.Sprintf("Uploaded file: %s. It has been saved to the artifacts", fileName),
			}
		}
	}

	event, err := r.appendUserEvent(ctx, storedSession, msg, stateDelta)
	return ctx, event, replaced, err
}

// appendUserEvent appends a user event with content msg, which may be nil,
// and stateDelta to storedSession, and returns it.
func (r *Runner) appendUserEvent(ctx agent.Context, storedSession session.Session, msg *genai.Content, stateDelta map[string]any) (*session.Event, error) {
	event := session.NewEvent(ctx, ctx.InvocationID())

	event.Author = "user"
	event.LLMResponse = model.LLMResponse{
		Content: msg,
	}
	if stateDelta != nil {
		event.Actions.StateDelta = stateDelta
	}
	if event.IsolationScope == "" {
		if iso, _ := findActiveTaskIsolationScope(storedSession); iso != "" {
			event.IsolationScope = iso
		}
	}
	eventmetadata.MergeIntoEvent(event, runMetadataFromContext(ctx))
	stampCallBranch(ctx, event)

	if err := r.sessionService.AppendEvent(ctx, storedSession, event); err != nil {
		return nil, fmt.Errorf("failed to append event to sessionService: %w", err)
	}
	return event, nil
}

// stampCallBranch gives ev, a user event not yet appended, the branch of the
// function call its first function response answers and, when ev has no
// isolation scope, that call's scope. The call is searched among the events
// of ctx's invocation only, newest first. ev is left as it is when it has no
// function response, its first response has no ID, or no event of the
// invocation holds the call (adk-python runners.py
// _stamp_event_branch_context).
func stampCallBranch(ctx agent.InvocationContext, ev *session.Event) {
	responses := utils.FunctionResponses(ev.Content)
	if len(responses) == 0 || responses[0].ID == "" {
		return
	}
	events := llminternal.InvocationEvents(ctx, true, false)
	for _, event := range slices.Backward(events) {
		for _, call := range utils.FunctionCalls(utils.Content(event)) {
			if call.ID != responses[0].ID {
				continue
			}
			ev.Branch = event.Branch
			if ev.IsolationScope == "" {
				ev.IsolationScope = event.IsolationScope
			}
			return
		}
	}
}

// validateNodeRootMessage refuses, on a node root, a message that holds
// both non-empty text and a function response with an ID: the response
// resumes an existing invocation while the text starts a new one
// (adk-python Runner._validate_new_message). Agent roots accept such a
// message, which the A2A input-required flow sends.
func (r *Runner) validateNodeRootMessage(msg *genai.Content) error {
	if r.rootNode == nil {
		return nil
	}
	hasResumeResponse, hasText := false, false
	for _, p := range msg.Parts {
		if p == nil {
			continue
		}
		if p.FunctionResponse != nil && p.FunctionResponse.ID != "" {
			hasResumeResponse = true
		}
		if p.Text != "" {
			hasText = true
		}
	}
	if hasResumeResponse && hasText {
		return errors.New("message cannot contain both function responses and text: function responses resume an existing invocation while text starts a new one")
	}
	return nil
}

// setInvocationID replaces the invocation ID of the invocation context ctx
// wraps. ctx must be the context the runner built with agent.NewContext from
// an icontext.InvocationContext, before any context is derived from it.
func setInvocationID(ctx agent.Context, id string) error {
	holder, ok := ctx.(interface {
		InvocationContext() agent.InvocationContext
	})
	if !ok {
		return fmt.Errorf("invocation context %T cannot take a resolved id", ctx)
	}
	ic, ok := holder.InvocationContext().(*icontext.InvocationContext)
	if !ok {
		return fmt.Errorf("invocation context %T cannot take a resolved id", holder.InvocationContext())
	}
	ic.SetInvocationID(id)
	return nil
}

// asUserContent returns a copy of msg with Role set to [genai.RoleUser] and its
// own Parts slice, so the caller's struct is never modified. The inbound
// message of a run is a user turn: an empty Role would be skipped when the
// model request is built, and a "model" Role would present the caller's text
// as the model's own earlier turn. A nil msg is returned as is.
func asUserContent(msg *genai.Content) *genai.Content {
	if msg == nil {
		return nil
	}
	out := *msg
	out.Role = genai.RoleUser
	out.Parts = slices.Clone(msg.Parts)
	return &out
}

// findActiveTaskIsolationScope returns the open task delegation of sess: the
// isolation scope of its newest live event (see utils.ApplyRewinds) whose
// scope no finish_task response has closed, and that event's invocation ID.
// It returns two empty strings when sess is nil or no scope is open.
//
// A scope is closed when the first finish_task function response of a live
// event in that scope has the result FinishTaskSuccessResult or
// FinishTaskErrorResult, wherever the response sits; events after it in the
// same scope do not reopen it. A response with an error instead of a result
// leaves the scope open, because the task agent retries. Port of adk-python's
// _find_active_task_scope (runners.py).
func findActiveTaskIsolationScope(sess session.Session) (scope, invocationID string) {
	if sess == nil {
		return "", ""
	}
	// closesTask reports whether the first finish_task function response of
	// ev ends its task.
	closesTask := func(ev *session.Event) bool {
		for _, fr := range utils.FunctionResponses(ev.Content) {
			if fr == nil || fr.Name != workflowinternal.FinishTaskToolName {
				continue
			}
			result, _ := fr.Response["result"].(string)
			return result == workflowinternal.FinishTaskSuccessResult || result == workflowinternal.FinishTaskErrorResult
		}
		return false
	}
	events := utils.ApplyRewinds(slices.Collect(sess.Events().All()))
	finished := map[string]struct{}{}
	for _, ev := range events {
		if ev.IsolationScope != "" && closesTask(ev) {
			finished[ev.IsolationScope] = struct{}{}
		}
	}
	for _, ev := range slices.Backward(events) {
		if ev.IsolationScope == "" {
			continue
		}
		if _, done := finished[ev.IsolationScope]; !done {
			return ev.IsolationScope, ev.InvocationID
		}
	}
	return "", ""
}

// classicAgentToRun returns the agent the classic path runs for msg and the
// branch it runs on.
//
// A function response goes back to the agent that made the call, on the
// branch restoreBranchFromHistory gives it within the invocation the
// responses resolve to leniently, or within the whole session when they
// resolve to none (adk-python find_agent_to_run, and
// _setup_context_for_new_invocation with restore_branch_from_history).
// Re-running the root would run agents that
// never saw the call, and they fail on the unmatched response. A call made
// inside a graph workflow is the exception: the root runs, and sequential,
// parallel and loop agents on the way narrow to the sub-agents holding the
// callers (utils.GraphResumeIndexes), provided the workflow's node agents are
// listed in its SubAgents. Sub-agents after it in a sequence still run, and an
// LLM agent among them sees the user's answer as context, unless the node's
// call is on a branch it cannot see, as when the workflow fans out inside a
// parallel agent (#2204). The workflow resumes the interrupted node only for
// an adk_request_input response; any other response restarts the graph. The
// first response in the message decides the route.
func (r *Runner) classicAgentToRun(storedSession session.Session, msg *genai.Content) (agent.Agent, string) {
	responses := utils.FunctionResponses(msg)
	if len(responses) == 0 {
		return r.rootAgent, ""
	}
	// The call is looked up in the live events only (adk-python
	// find_agent_to_run reads _apply_rewinds).
	answers := func(fc *genai.FunctionCall) bool { return fc.ID == responses[0].ID }
	for _, ev := range slices.Backward(utils.ApplyRewinds(slices.Collect(storedSession.Events().All()))) {
		if !slices.ContainsFunc(utils.FunctionCalls(ev.Content), answers) {
			continue
		}
		sub := r.rootAgent.FindAgent(ev.Author)
		if sub == nil || r.isInGraphWorkflow(sub) {
			return r.rootAgent, ""
		}
		// Answers to calls of several invocations search the whole session.
		invocationID, spansInvocations := resolveInvocationID(storedSession, msg)
		if spansInvocations {
			invocationID = ""
		}
		return sub, r.restoreBranchFromHistory(storedSession, sub, invocationID)
	}
	return r.rootAgent, ""
}

// usesLegacySubAgentPicker reports whether a root LlmAgent running in rootMode
// picks the agent to run from session history with findAgentToRun: only a chat
// root with no task-mode sub-agent does. A chat coordinator with task-mode
// sub-agents delegates through its own wrapper, so the picker must not bypass
// it on a resume (adk-python Runner._uses_legacy_sub_agent_picker).
func (r *Runner) usesLegacySubAgentPicker(rootMode llminternal.Mode) bool {
	if rootMode != llminternal.ModeChat {
		return false
	}
	for _, subAgent := range r.rootAgent.SubAgents() {
		sub, ok := subAgent.(llminternal.Agent)
		if ok && llminternal.Reveal(sub).Mode == llminternal.ModeTask {
			return false
		}
	}
	return true
}

// restoreBranchFromHistory returns the branch agentToRun last ran on, taken
// from its most recent live event in sess (see utils.ApplyRewinds) that has a
// branch. A non-empty invocationID limits the search to that invocation's
// events, so an agent does not continue on a branch of an earlier invocation.
// An event on a tool's message branch (llminternal.IsToolBranch, over the
// function calls of the live events) is skipped: the tool published it under
// the agent's name, but the agent does not run there. The root agent, a node
// root's stand-in agent, and an agent with no such event, run on the root
// branch "" (adk-python agents/_agent_router.py restore_branch_from_history;
// adk-python restores no branch for a node root).
func (r *Runner) restoreBranchFromHistory(sess session.Session, agentToRun agent.Agent, invocationID string) string {
	if sess == nil || agentToRun == nil || agentToRun == r.rootAgent || agentToRun == r.standInAgent {
		return ""
	}
	events := utils.ApplyRewinds(slices.Collect(sess.Events().All()))
	callIDs := llminternal.FunctionCallIDs(events)
	for _, ev := range slices.Backward(events) {
		if ev.Branch == "" {
			continue
		}
		if invocationID != "" && ev.InvocationID != invocationID {
			continue
		}
		if llminternal.IsToolBranch(ev.Branch, agentToRun.Name(), callIDs) {
			continue
		}
		if ev.Author == agentToRun.Name() {
			return ev.Branch
		}
	}
	return ""
}

// findAgentToRun returns the agent that should handle the next request based on
// session history.
//
// It decides on the routing view: the session's live events (see
// utils.ApplyRewinds), followed by msg as a user event when msg carries
// function responses. When the view ends with an answer to a function call,
// llminternal.DecideResumeAction decides whether the call's author runs,
// which an answer authored "user" always routes to and an agent's own answer
// does on a resumable runner. Otherwise the most recent agent that may
// transfer across the agent tree runs, skipping user events and the events
// that record an agent's state or end, and the root agent when there is none.
// It ports adk-python's find_agent_to_run (agents/_agent_router.py) with
// Runner._session_for_routing (runners.py).
func (r *Runner) findAgentToRun(sess session.Session, msg *genai.Content) (agent.Agent, error) {
	if isGraphWorkflowAgent(r.rootAgent) {
		return r.rootAgent, nil
	}
	events := utils.ApplyRewinds(slices.Collect(sess.Events().All()))
	if len(utils.FunctionResponses(msg)) > 0 {
		events = append(events, &session.Event{Author: "user", LLMResponse: model.LLMResponse{Content: msg}})
	}
	if subAgent := r.answeredCallAgent(events); subAgent != nil {
		if entry := r.graphWorkflowEntrypoint(subAgent); entry != nil {
			return entry, nil
		}
		return subAgent, nil
	}

	for _, event := range slices.Backward(events) {
		if event.Author == "user" || event.Actions.AgentState != nil ||
			(event.Actions.EndOfAgent != nil && *event.Actions.EndOfAgent) {
			continue
		}

		subAgent := r.rootAgent.FindAgent(event.Author)
		// Agent not found, continue looking for the other event.
		if subAgent == nil {
			log.Printf("Event from an unknown agent: %s, event id: %s", event.Author, event.ID) //nolint:forbidigo // pre-slog call site
			continue
		}

		if r.isTransferableAcrossAgentTree(subAgent) {
			return subAgent, nil
		}
	}

	// Falls back to root agent if no suitable agents are found in the session.
	return r.rootAgent, nil
}

// answeredCallAgent returns the agent that made the function call the last
// of events answers, when llminternal.DecideResumeAction routes the answer to
// that call's author and the author is in the agent tree; nil otherwise.
func (r *Runner) answeredCallAgent(events []*session.Event) agent.Agent {
	if len(events) == 0 {
		return nil
	}
	answer := events[len(events)-1]
	responses := utils.FunctionResponses(answer.Content)
	if len(responses) == 0 {
		return nil
	}
	call := llminternal.FindCallEvent(events[:len(events)-1], responses[0].ID)
	if call == nil || call.Author == "" {
		return nil
	}
	switch llminternal.DecideResumeAction(call, answer, r.resumable) {
	case llminternal.ResumeRouteToAuthor, llminternal.ResumeReplayCalls:
		if subAgent := r.rootAgent.FindAgent(call.Author); subAgent != nil {
			return subAgent
		}
		log.Printf("Function call from an unknown agent: %s, event id: %s", call.Author, call.ID) //nolint:forbidigo // pre-slog call site
	}
	return nil
}

// resumedUserContent returns the user content of a resume of invocationID on
// the classic path: msg when it is given, else the invocation's stored user
// message (utils.FindUserMessageForInvocation). It returns an error when sess
// has no events, or when msg is nil and no user message of the invocation is
// stored (adk-python Runner._setup_context_for_resumed_invocation).
func resumedUserContent(sess session.Session, msg *genai.Content, invocationID string) (*genai.Content, error) {
	if sess.Events().Len() == 0 {
		return nil, fmt.Errorf("session %s has no events to resume", sess.ID())
	}
	if msg != nil {
		return msg, nil
	}
	if stored := utils.FindUserMessageForInvocation(sess.Events(), invocationID); stored != nil {
		return stored, nil
	}
	return nil, fmt.Errorf("no user message available for resuming invocation: %s", invocationID)
}

// rebuildAgentStates returns the agent states and the ended agents of the
// invocation invocationID, keyed by author, from its events in order: an
// end-of-agent event ends its author and drops its state; an event with an
// agent state sets its author's state and reopens it; any other event with
// content not authored "user" gives an author without a non-empty state an
// empty one and reopens it. An author absent from ended has neither ended nor
// recorded anything. It ports adk-python's
// InvocationContext.populate_invocation_agent_states
// (agents/invocation_context.py).
func rebuildAgentStates(events session.Events, invocationID string) (states map[string]map[string]any, ended map[string]bool) {
	states, ended = map[string]map[string]any{}, map[string]bool{}
	for _, ev := range utils.ApplyRewinds(slices.Collect(events.All())) {
		if ev.InvocationID != invocationID {
			continue
		}
		switch {
		case ev.Actions.EndOfAgent != nil && *ev.Actions.EndOfAgent:
			ended[ev.Author] = true
			delete(states, ev.Author)
		case ev.Actions.AgentState != nil:
			states[ev.Author] = ev.Actions.AgentState
			ended[ev.Author] = false
		case ev.Author != "user" && ev.Content != nil && len(states[ev.Author]) == 0:
			states[ev.Author] = map[string]any{}
			ended[ev.Author] = false
		}
	}
	return states, ended
}

// resumedAgentToRun returns the agent a resume of invocationID on the
// classic path runs, and its branch. When the root agent has recorded its
// state or its end in the invocation (it is in ended), that is the root;
// otherwise findAgentToRun picks the agent from sess, which already holds the
// turn's message, and it continues on the branch it last ran on in the
// invocation (adk-python Runner._setup_context_for_resumed_invocation).
func (r *Runner) resumedAgentToRun(sess session.Session, invocationID string, ended map[string]bool) (agent.Agent, string, error) {
	if _, ok := ended[r.rootAgent.Name()]; ok {
		return r.rootAgent, "", nil
	}
	agentToRun, err := r.findAgentToRun(sess, nil)
	if err != nil {
		return nil, "", err
	}
	return agentToRun, r.restoreBranchFromHistory(sess, agentToRun, invocationID), nil
}

// isInGraphWorkflow reports whether a or one of its ancestors is a graph
// workflow agent built by workflowagent.New.
func (r *Runner) isInGraphWorkflow(a agent.Agent) bool {
	return r.outermostGraphWorkflow(a) != nil
}

func isGraphWorkflowAgent(a agent.Agent) bool {
	ia, ok := a.(agentinternal.Agent)
	return ok && agentinternal.Reveal(ia).AgentType == agentinternal.TypeWorkflowAgent
}

func isCompositeWorkflowAgent(a agent.Agent) bool {
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

// outermostGraphWorkflow returns the highest graph workflow agent
// (workflowagent.New) enclosing a, or nil when none does.
func (r *Runner) outermostGraphWorkflow(a agent.Agent) agent.Agent {
	var outermost agent.Agent
	for cur := a; cur != nil; cur = r.parents[cur.Name()] {
		if isGraphWorkflowAgent(cur) {
			outermost = cur
		}
	}
	return outermost
}

// graphWorkflowEntrypoint returns the agent findAgentToRun dispatches to when a
// belongs to a graph workflow: the highest contiguous composite workflow
// ancestor above the outermost graph workflow (so SequentialAgent, LoopAgent
// and ParallelAgent narrow via utils.GraphResumeIndexes and still run sibling
// sub-agents), stopping before any non-workflow ancestor such as an LlmAgent.
func (r *Runner) graphWorkflowEntrypoint(a agent.Agent) agent.Agent {
	wf := r.outermostGraphWorkflow(a)
	if wf == nil {
		return nil
	}
	entry := wf
	for parent := r.parents[entry.Name()]; parent != nil && isCompositeWorkflowAgent(parent); parent = r.parents[entry.Name()] {
		entry = parent
	}
	return entry
}

// checks if the agent and its parent chain allow transfer up the tree.
func (r *Runner) isTransferableAcrossAgentTree(agentToRun agent.Agent) bool {
	for curAgent := agentToRun; curAgent != nil; curAgent = r.parents[curAgent.Name()] {
		llmAgent, ok := curAgent.(llminternal.Agent)
		if !ok {
			return false
		}

		if llminternal.Reveal(llmAgent).DisallowTransferToParent {
			return false
		}
	}

	return true
}

// shouldAppendEvent reports whether the runner appends event, as the
// OnEventCallback plugins left it, to the session: never a partial event, and
// in live mode no event carrying inline model media. This mirrors adk-python's
// Runner._process_and_append_event and _should_append_event (runners.py).
func shouldAppendEvent(event *session.Event, isLive bool) bool {
	if event == nil || event.LLMResponse.Partial {
		return false
	}
	if isLive && isLiveMediaWithInlineData(event) {
		return false
	}
	return true
}

// isLiveMediaWithInlineData reports whether event's content has a part with
// inline audio, video or image data, matched on the MIME type without regard
// to case. Such data stays out of a live session to keep large blobs out of
// storage; a FileData reference to the same media is kept. This mirrors
// adk-python's _is_live_model_media_event_with_inline_data
// (flows/llm_flows/context/_contents.py).
func isLiveMediaWithInlineData(event *session.Event) bool {
	if event == nil || event.LLMResponse.Content == nil {
		return false
	}
	for _, part := range event.LLMResponse.Content.Parts {
		if part == nil || part.InlineData == nil {
			continue
		}
		mime := strings.ToLower(part.InlineData.MIMEType)
		if strings.HasPrefix(mime, "audio/") ||
			strings.HasPrefix(mime, "video/") ||
			strings.HasPrefix(mime, "image/") {
			return true
		}
	}
	return false
}

// clearMessageAsOutput returns event with Output cleared when event is a
// final model message that also serves as its node's output
// (NodeInfo.MessageAsOutput), so the message is stored once, as content. It
// returns a copy rather than editing event, and event itself otherwise. The
// loops call it after the OnEventCallback plugins, which therefore still see
// Output, as adk-python's Runner does (runners.py).
func clearMessageAsOutput(event *session.Event) *session.Event {
	if !event.LLMResponse.Partial && event.NodeInfo != nil && event.NodeInfo.MessageAsOutput && event.LLMResponse.Content != nil {
		clone := *event
		clone.Output = nil
		return &clone
	}
	return event
}

// withPluginManager puts this runner's plugin manager in ctx, which is where
// tools and callbacks look it up. The one exception is a runner without plugins
// that agenttool starts for a sub-agent: it keeps the plugin manager of the run
// that called the tool, so that run's plugins also see the sub-agent. Any other
// runner uses only its own plugins. Run-level callbacks always stay with
// r.pluginManager.
func (r *Runner) withPluginManager(ctx context.Context) context.Context {
	ctx, inherited := plugininternal.TakeInheritedPlugins(ctx)
	if !r.pluginManager.HasPlugins() && inherited.HasPlugins() {
		return plugininternal.ToContext(ctx, inherited)
	}
	return plugininternal.ToContext(ctx, r.pluginManager)
}
