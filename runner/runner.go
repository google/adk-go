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
	"fmt"
	"iter"
	"log"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/agent/compactionctx"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
	"google.golang.org/adk/v2/internal/compactioninternal"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	imemory "google.golang.org/adk/v2/internal/memory"
	"google.golang.org/adk/v2/internal/plugininternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/internal/workflowinternal"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// Config is used to create a [Runner].
type Config struct {
	AppName string
	// Root agent which starts the execution.
	Agent          agent.Agent
	SessionService session.Service

	// optional
	ArtifactService artifact.Service
	// optional
	MemoryService memory.Service
	// optional
	PluginConfig PluginConfig
	// optional
	AutoCreateSession bool

	// EventsCompactionConfig enables context compaction for the sessions this
	// runner drives: older events are periodically summarized so prompts stay
	// small as a conversation grows. Nil, the default, disables compaction.
	//
	// When the config names no Summarizer, the runner installs a
	// [compaction.LLMSummarizer] over the root agent's model, which then has to
	// be an LLM agent.
	//
	// optional
	EventsCompactionConfig *compaction.Config
}

type PluginConfig struct {
	Plugins      []*plugin.Plugin
	CloseTimeout time.Duration
}

type RunOption func(*runOptions)

type runOptions struct {
	stateDelta       map[string]any
	yieldUserMessage bool
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

// New creates a new [Runner].
func New(cfg Config) (*Runner, error) {
	if cfg.Agent == nil {
		return nil, fmt.Errorf("root agent is required")
	}

	if cfg.SessionService == nil {
		return nil, fmt.Errorf("session service is required")
	}

	parents, err := parentmap.New(cfg.Agent)
	if err != nil {
		return nil, fmt.Errorf("failed to create agent tree: %w", err)
	}

	pluginManager, err := plugininternal.NewPluginManager(plugininternal.PluginConfig{
		Plugins:      cfg.PluginConfig.Plugins,
		CloseTimeout: cfg.PluginConfig.CloseTimeout,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create plugin manager: %w", err)
	}

	compactionConfig, err := resolveCompactionConfig(cfg.EventsCompactionConfig, cfg.Agent)
	if err != nil {
		return nil, err
	}

	return &Runner{
		appName:           cfg.AppName,
		rootAgent:         cfg.Agent,
		sessionService:    cfg.SessionService,
		artifactService:   cfg.ArtifactService,
		memoryService:     cfg.MemoryService,
		parents:           parents,
		pluginManager:     pluginManager,
		autoCreateSession: cfg.AutoCreateSession,
		compactionConfig:  compactionConfig,
	}, nil
}

// defaultSummarizerTimeout bounds the summarization call the runner installs
// when an application enables compaction without naming a Summarizer. A var so
// a test can shorten it rather than waiting a minute to prove the bound exists.
var defaultSummarizerTimeout = 60 * time.Second

// resolveCompactionConfig validates cfg and fills in the default summarizer.
//
// Resolving at construction time means a misconfigured runner fails fast at
// New, rather than silently skipping compaction turns later, or blowing up
// mid-conversation the first time a compaction triggers.
func resolveCompactionConfig(cfg *compaction.Config, rootAgent agent.Agent) (*compaction.Config, error) {
	if cfg == nil {
		return nil, nil
	}
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid EventsCompactionConfig: %w", err)
	}
	// Copy so the caller's config is not mutated by the summarizer default.
	resolved := *cfg
	if resolved.Summarizer != nil {
		return &resolved, nil
	}

	llmAgent, ok := rootAgent.(llminternal.Agent)
	if !ok {
		return nil, fmt.Errorf("EventsCompactionConfig needs a Summarizer: root agent %q is not an LLM agent, so no default model is available", rootAgent.Name())
	}
	m := llminternal.Reveal(llmAgent).Model
	if m == nil {
		return nil, fmt.Errorf("EventsCompactionConfig needs a Summarizer: root agent %q has no model", rootAgent.Name())
	}
	summarizer, err := compaction.NewLLMSummarizer(compaction.LLMSummarizerConfig{
		Model: m,
		// Safety settings and output limits the application configured govern
		// the summarization call too, rather than it silently falling back to
		// provider defaults for the one call that sees the whole transcript.
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
	appName         string
	rootAgent       agent.Agent
	sessionService  session.Service
	artifactService artifact.Service
	memoryService   memory.Service

	parents           parentmap.Map
	pluginManager     *plugininternal.PluginManager
	autoCreateSession bool

	// compactionConfig is nil when compaction is disabled. Otherwise it is a
	// validated copy of Config.EventsCompactionConfig with the summarizer
	// resolved.
	compactionConfig *compaction.Config
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
func (r *Runner) compactAfterInvocation(ctx context.Context, storedSession session.Session, ictx agent.InvocationContext) error {
	if !compactioninternal.HasSlidingWindow(r.compactionConfig) {
		return nil
	}
	// Tail retention may already have compacted this turn from inside the
	// invocation. Summarizing again the moment it ends would pay for a second
	// model call to re-summarize what was just summarized, and would leave two
	// ranges over the same span. The reference implementation reaches the same
	// outcome by evaluating both strategies in one place and returning early.
	if compactionctx.FromContext(ctx).AlreadyCompacted() {
		return nil
	}
	// Compaction is an optimisation, so a cancelled or expired run should not
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
	// parallel tool responses and sub-agent events funnelled through a channel
	// are routinely created before the range ends and appended after it.
	raced := func() (bool, error) {
		latest, err := r.reloadSession(ctx, storedSession)
		if err != nil {
			return false, err
		}
		return compactioninternal.RangeRaced(latest, current, summary), nil
	}
	discardRaced := func() {
		finish(nil, "another compaction covering the same events landed while summarizing")
		log.Printf("adk: discarding a context compaction summary because the session changed inside its range while summarizing")
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
	// The reference implementation reaches the same place from the other
	// direction: its sliding window yields the event and lets the runner append
	// it, so that persistence stays at the runtime's synchronisation point.
	// Appending straight from the compactor skipped the one hook that lets a
	// plugin see, rewrite or reject what goes into a session, and a summary is
	// exactly the kind of derived content a redaction plugin would care about.
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
				log.Printf("adk: discarding a context compaction summary because a plugin left no usable content in it")
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
	return nil
}

// fromPlugin returns the event a plugin gave back, with the compaction record
// the framework put on the original rather than any the plugin supplied.
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
// The post-invocation summary path does not use this. A plugin is invited to
// rewrite a summary there, and SanitizeSummary re-checks the result.
func fromPlugin(original, modified *session.Event) *session.Event {
	if modified == nil || modified == original {
		return original
	}
	modified.Actions.Compaction = original.Actions.Compaction
	return modified
}

// compactionRuntime returns the runtime that the request processors read off
// the context, both to gate prompt assembly on compaction being configured and
// to run intra-invocation compaction. It is nil when compaction is disabled for
// this runner.
func (r *Runner) compactionRuntime() *compactionctx.Runtime {
	return compactionctx.New(r.compactionConfig, r.sessionService)
}

// reloadSession re-fetches a session so compaction works against current state
// rather than the snapshot the invocation began with.
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
		return nil, fmt.Errorf("session %q disappeared while compacting", s.ID())
	}
	return resp.Session, nil
}

func (r *Runner) getOrCreateSession(ctx context.Context, userID, sessionID string) (session.Session, error) {
	getResp, err := r.sessionService.Get(ctx, &session.GetRequest{
		AppName:   r.appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err == nil {
		return getResp.Session, nil
	}
	if !r.autoCreateSession {
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
func (r *Runner) Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg agent.RunConfig, opts ...RunOption) iter.Seq2[*session.Event, error] {
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

		storedSession, err := r.getOrCreateSession(ctx, userID, sessionID)
		if err != nil {
			yield(nil, err)
			return
		}

		// Node path: an LlmAgent runs through the ADK 2.0 node runtime
		// (the Go equivalent of adk-python's _run_node_async, reached for
		// an LlmAgent root).

		if isLlmAgent(r.rootAgent) {
			llmInternalAgent, ok := r.rootAgent.(llminternal.Agent)
			if !ok {
				yield(nil, fmt.Errorf("agent %s is not an LlmAgent", r.rootAgent.Name()))
				return
			}

			llmInternalState := llminternal.Reveal(llmInternalAgent)

			if llmInternalState.Mode == "" {
				// LlmAgent as root agent must have chat mode.
				llmInternalState.Mode = llminternal.ModeChat
			}

			if llmInternalState.Mode != llminternal.ModeChat {
				yield(nil, fmt.Errorf("root agent %s must be a chat LlmAgent, but has mode %s", r.rootAgent.Name(), llmInternalState.Mode))
				return
			}

			hasTaskSubAgent := func() bool {
				for _, subAgent := range r.rootAgent.SubAgents() {
					if !isLlmAgent(subAgent) {
						continue
					}
					llmInternalSubAgent := llminternal.Reveal(subAgent.(llminternal.Agent))
					if llmInternalSubAgent.Mode == llminternal.ModeTask {
						return true
					}
				}
				return false
			}

			var agentToRun agent.Agent

			// when the chat coordinator has task-mode sub-agents,
			// the wrapper handles delegation via ctx.run_node. Don't let
			// the legacy sub-agent picker bypass the coordinator on resume.
			if hasTaskSubAgent() {
				agentToRun = r.rootAgent
			} else {
				agentToRun, err = r.findAgentToRun(storedSession, msg)
				if err != nil {
					yield(nil, err)
					return
				}
			}

			r.runNode(ctx, storedSession, agentToRun, msg, cfg, options, yield)
			return
		}

		ctx = parentmap.ToContext(ctx, r.parents)
		ctx = runconfig.ToContext(ctx, &runconfig.RunConfig{
			StreamingMode: runconfig.StreamingMode(cfg.StreamingMode),
		})
		ctx = plugininternal.ToContext(ctx, r.pluginManager)
		ctx = compactionctx.ToContext(ctx, r.compactionRuntime())

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
				log.Printf("adk: %v", err)
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

		var memoryImpl agent.Memory = nil
		if r.memoryService != nil {
			memoryImpl = &imemory.Memory{
				Service:   r.memoryService,
				SessionID: storedSession.ID(),
				UserID:    storedSession.UserID(),
				AppName:   storedSession.AppName(),
			}
		}

		ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
			Artifacts:    artifacts,
			Memory:       memoryImpl,
			Session:      storedSession,
			Agent:        r.rootAgent,
			UserContent:  msg,
			RunConfig:    &cfg,
			InvocationID: resolveInvocationID(storedSession, msg),
		})
		invocationCtx = ic
		ctx := agent.NewContext(ic)
		ctx, _, err = r.appendMessageToSession(ctx, storedSession, msg, cfg.SaveInputBlobsAsArtifacts, r.pluginManager, options.stateDelta)
		if err != nil {
			yield(nil, err)
			return
		}

		pluginManager := r.pluginManager
		if pluginManager != nil {
			// Defer the after run callbacks to perform global cleanup tasks or finalizing logs and metrics data.
			// This does NOT emit any event.
			defer pluginManager.RunAfterRunCallback(ctx)

			earlyExitResult, err := pluginManager.RunBeforeRunCallback(ctx)
			if earlyExitResult != nil || err != nil {
				earlyExitEvent := session.NewEvent(ctx, ctx.InvocationID())
				earlyExitEvent.Author = "user"
				earlyExitEvent.LLMResponse = model.LLMResponse{
					Content: msg,
				}
				if err := r.sessionService.AppendEvent(ctx, storedSession, earlyExitEvent); err != nil {
					yield(nil, fmt.Errorf("failed to add event to session: %w", err))
					return
				}
				yield(earlyExitEvent, err)
				return
			}
		}

		for event, err := range r.rootAgent.Run(ctx) {
			if err != nil {
				if !yield(event, err) {
					return
				}
				continue
			}

			if event != nil && !event.LLMResponse.Partial {
				if event.NodeInfo != nil && event.NodeInfo.MessageAsOutput && event.LLMResponse.Content != nil {
					clone := *event
					clone.Output = nil
					event = &clone
				}
			}

			if pluginManager != nil {
				modifiedEvent, err := pluginManager.RunOnEventCallback(ctx, event)
				if err != nil {
					if !yield(nil, err) {
						return
					}
					continue
				}
				event = fromPlugin(event, modifiedEvent)
			}

			// only commit non-partial event to a session service
			if !event.LLMResponse.Partial {
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

func (r *Runner) RunLive(ctx context.Context, userID, sessionID string, cfg agent.LiveRunConfig, opts ...RunOption) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	options := runOptions{}
	for _, opt := range opts {
		opt(&options)
	}

	storedSession, err := r.getOrCreateSession(ctx, userID, sessionID)
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
		StreamingMode: runconfig.StreamingModeBidi, // Live is always bidirectional streaming
		Live:          &cfg,
	})
	ctx = plugininternal.ToContext(ctx, r.pluginManager)
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

	var memoryImpl agent.Memory = nil
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
		earlyExitResult, err := r.pluginManager.RunBeforeRunCallback(iCtx)
		if err != nil {
			return nil, nil, err
		}
		if earlyExitResult != nil {
			earlyExitEvent := session.NewEvent(iCtx, iCtx.InvocationID())
			earlyExitEvent.Author = agentToRun.Name()
			earlyExitEvent.LLMResponse = model.LLMResponse{
				Content: earlyExitResult,
			}
			if err := r.sessionService.AppendEvent(iCtx, storedSession, earlyExitEvent); err != nil {
				return nil, nil, fmt.Errorf("failed to add event to session: %w", err)
			}

			earlyExitIter := func(yield func(*session.Event, error) bool) {
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

			if event != nil && !event.LLMResponse.Partial {
				if event.NodeInfo != nil && event.NodeInfo.MessageAsOutput && event.LLMResponse.Content != nil {
					clone := *event
					clone.Output = nil
					event = &clone
				}
			}

			if r.pluginManager != nil {
				modifiedEvent, pluginErr := r.pluginManager.RunOnEventCallback(iCtx, event)
				if pluginErr != nil {
					if !yield(nil, pluginErr) {
						return
					}
					continue
				}
				event = fromPlugin(event, modifiedEvent)
			}

			// Chronological event buffering logic for Live streaming.
			// Holds back tool calls/responses if they arrive before the transcription finishes.
			if event.LLMResponse.Partial && (event.LLMResponse.InputTranscription != nil || event.LLMResponse.OutputTranscription != nil) {
				isTranscribing = true
			}

			isToolCallOrResp := false
			if event.LLMResponse.Content != nil {
				for _, part := range event.LLMResponse.Content.Parts {
					if part.FunctionCall != nil || part.FunctionResponse != nil {
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

					if err := r.sessionService.AppendEvent(iCtx, storedSession, event); err != nil {
						if !yield(nil, fmt.Errorf("failed to add event to session: %w", err)) {
							return
						}
						continue
					}
					if !yield(event, nil) {
						return
					}

					for _, bufferedEvent := range bufferedEvents {
						if err := r.sessionService.AppendEvent(iCtx, storedSession, bufferedEvent); err != nil {
							if !yield(nil, fmt.Errorf("failed to add event to session: %w", err)) {
								return
							}
							continue
						}
						if !yield(bufferedEvent, nil) {
							return
						}
					}
					bufferedEvents = nil
					continue
				}
			}

			if !event.LLMResponse.Partial && !hasInlineData(event) {
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
	}

	return &runnerLiveSession{
		sess:          agentSess,
		r:             r,
		iCtx:          iCtx,
		storedSession: storedSession,
	}, wrappedIter, nil
}

func (r *Runner) appendMessageToSession(ctx agent.Context, storedSession session.Session, msg *genai.Content, saveInputBlobsAsArtifacts bool, pluginManager *plugininternal.PluginManager, stateDelta map[string]any) (agent.Context, *session.Event, error) {
	if msg == nil {
		return ctx, nil, nil
	}
	if pluginManager != nil {
		modifiedMsg, err := pluginManager.RunOnUserMessageCallback(ctx, msg)
		if err != nil {
			return ctx, nil, fmt.Errorf("error running on run user message callback : %w", err)
		}
		if modifiedMsg != nil {
			msg = modifiedMsg

			ctx = ctx.WithDelta(&agent.CommonContextDelta{
				InvocationContextDelta: &agent.InvocationContextDelta{
					UserContent: &msg,
				},
			})
		}
	}

	artifactsService := ctx.Artifacts()
	if artifactsService != nil && saveInputBlobsAsArtifacts {
		for i, part := range msg.Parts {
			if part.InlineData == nil {
				continue
			}
			fileName := fmt.Sprintf("artifact_%s_%d", ctx.InvocationID(), i)
			if _, err := artifactsService.Save(ctx, fileName, part); err != nil {
				return ctx, nil, fmt.Errorf("failed to save artifact %s: %w", fileName, err)
			}
			// Replace the part with a text placeholder
			msg.Parts[i] = &genai.Part{
				Text: fmt.Sprintf("Uploaded file: %s. It has been saved to the artifacts", fileName),
			}
		}
	}

	event := session.NewEvent(ctx, ctx.InvocationID())

	event.Author = "user"
	event.LLMResponse = model.LLMResponse{
		Content: msg,
	}
	if stateDelta != nil {
		event.Actions.StateDelta = stateDelta
	}
	if event.IsolationScope == "" {
		if iso := findActiveTaskIsolationScope(storedSession); iso != "" {
			event.IsolationScope = iso
		}
	}

	if err := r.sessionService.AppendEvent(ctx, storedSession, event); err != nil {
		return ctx, nil, fmt.Errorf("failed to append event to sessionService: %w", err)
	}
	return ctx, event, nil
}

// findActiveTaskIsolationScope returns the most recent isolation_scope that has
// not yet been closed by a successful finish_task FunctionResponse.
func findActiveTaskIsolationScope(sess session.Session) string {
	if sess == nil {
		return ""
	}
	events := sess.Events()
	finished := map[string]struct{}{}
	for i := events.Len() - 1; i >= 0; i-- {
		ev := events.At(i)
		if ev == nil || ev.IsolationScope == "" {
			continue
		}
		scope := ev.IsolationScope
		for _, fr := range utils.FunctionResponses(ev.Content) {
			if fr == nil || fr.Name != workflowinternal.FinishTaskToolName {
				continue
			}
			if result, ok := fr.Response["result"]; ok {
				if s, ok := result.(string); ok && s == workflowinternal.FinishTaskSuccessResult {
					finished[scope] = struct{}{}
				}
			}
			break
		}
		if _, done := finished[scope]; done {
			continue
		}
		return scope
	}
	return ""
}

// findAgentToRun returns the agent that should handle the next request based on
// session history.
func (r *Runner) findAgentToRun(session session.Session, msg *genai.Content) (agent.Agent, error) {
	if event := handleUserFunctionCallResponse(session.Events(), msg); event != nil {
		subAgent := r.rootAgent.FindAgent(event.Author)
		if subAgent != nil {
			return subAgent, nil
		}
		log.Printf("Function call from an unknown agent: %s, event id: %s", event.Author, event.ID)
	}

	events := session.Events()
	for i := events.Len() - 1; i >= 0; i-- {
		event := events.At(i)

		if event.Author == "user" {
			continue
		}

		subAgent := r.rootAgent.FindAgent(event.Author)
		// Agent not found, continue looking for the other event.
		if subAgent == nil {
			log.Printf("Event from an unknown agent: %s, event id: %s", event.Author, event.ID)
			continue
		}

		if r.isTransferableAcrossAgentTree(subAgent) {
			return subAgent, nil
		}
	}

	// Falls back to root agent if no suitable agents are found in the session.
	return r.rootAgent, nil
}

// handleUserFunctionCallResponse finds the function call event that matches the function response id
// delivered by the user in the latest event.
func handleUserFunctionCallResponse(events session.Events, msg *genai.Content) *session.Event {
	if events.Len() == 0 {
		return nil
	}

	functionResponses := utils.FunctionResponses(msg)
	if len(functionResponses) == 0 {
		return nil
	}

	// This assumes that even if user provides multiple function responses, all the function calls
	// were made by the same agent. Otherwise it would be impossible to rearrange session events
	// such that every function response has a corresponding call filtering by author.
	callID := functionResponses[0].ID
	for i := events.Len() - 1; i >= 0; i-- {
		event := events.At(i)
		for _, part := range utils.FunctionCalls(event.Content) {
			if part.ID == callID {
				return event
			}
		}
	}
	return nil
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

func hasInlineData(event *session.Event) bool {
	if event.LLMResponse.Content == nil {
		return false
	}
	for _, part := range event.LLMResponse.Content.Parts {
		if part.InlineData != nil {
			return true
		}
	}
	return false
}
