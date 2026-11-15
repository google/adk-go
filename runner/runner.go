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
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/internal/agent/parentmap"
	"google.golang.org/adk/internal/agent/runconfig"
	artifactinternal "google.golang.org/adk/internal/artifact"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/llminternal"
	imemory "google.golang.org/adk/internal/memory"
	"google.golang.org/adk/internal/plugininternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/platform"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
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
	// optional. When true, Run and RunLive create the session if it does not
	// exist. The session is created when the session ID is empty, so the
	// service generates one, or when the session service's Get reports
	// [session.ErrNotFound]. Any other Get error is returned as is, so a
	// transient storage failure never replaces an existing session.
	AutoCreateSession bool
}

type PluginConfig struct {
	Plugins      []*plugin.Plugin
	CloseTimeout time.Duration
}

type RunOption func(*runOptions)

type runOptions struct {
	stateDelta map[string]any
}

// WithStateDelta sets a state delta for the run invocation.
func WithStateDelta(delta map[string]any) RunOption {
	return func(o *runOptions) {
		o.stateDelta = delta
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

	return &Runner{
		appName:           cfg.AppName,
		rootAgent:         cfg.Agent,
		sessionService:    cfg.SessionService,
		artifactService:   cfg.ArtifactService,
		memoryService:     cfg.MemoryService,
		parents:           parents,
		pluginManager:     pluginManager,
		autoCreateSession: cfg.AutoCreateSession,
	}, nil
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
}

// Run runs the agent for the given user input, yielding events from agents.
// For each user message it finds the proper agent within an agent tree to
// continue the conversation within the session.
func (r *Runner) Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg agent.RunConfig, opts ...RunOption) iter.Seq2[*session.Event, error] {
	// TODO(hakim): we need to validate whether cfg is compatible with the Agent.
	//   see adk-python/src/google/adk/runners.py Runner._new_invocation_context.
	// TODO: setup tracer.
	return func(yield func(*session.Event, error) bool) {
		options := runOptions{}
		for _, opt := range opts {
			opt(&options)
		}

		// The inbound message is always a user turn, whatever Role the caller
		// set. Normalize it once here so that findAgentToRun, the invocation
		// context and the stored event all see the same message.
		msg = asUserContent(msg)

		storedSession, err := r.getOrCreateSession(ctx, userID, sessionID)
		if err != nil {
			yield(nil, err)
			return
		}

		agentToRun, branch, err := r.findAgentToRun(storedSession, msg)
		if err != nil {
			yield(nil, err)
			return
		}

		ctx = parentmap.ToContext(ctx, r.parents)
		ctx = runconfig.ToContext(ctx, &runconfig.RunConfig{
			StreamingMode: runconfig.StreamingMode(cfg.StreamingMode),
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

		var memoryImpl agent.Memory = nil
		if r.memoryService != nil {
			memoryImpl = &imemory.Memory{
				Service:   r.memoryService,
				SessionID: storedSession.ID(),
				UserID:    storedSession.UserID(),
				AppName:   storedSession.AppName(),
			}
		}

		ctx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
			Artifacts:   artifacts,
			Memory:      memoryImpl,
			Session:     storedSession,
			Branch:      branch,
			Agent:       agentToRun,
			UserContent: msg,
			RunConfig:   &cfg,
		})
		ctx, err = r.appendMessageToSession(ctx, storedSession, msg, cfg.SaveInputBlobsAsArtifacts, r.pluginManager, options.stateDelta)
		if err != nil {
			yield(nil, err)
			return
		}

		pluginManager := r.pluginManager
		if pluginManager != nil {
			// Defer the after run callbacks to perform global cleanup tasks or finalizing logs and metrics data.
			// This does NOT emit any event.
			defer pluginManager.RunAfterRunCallback(ctx)

			if event, err := r.runBeforeRunCallback(ctx); event != nil || err != nil {
				yield(event, err)
				return
			}
		}

		for event, err := range agentToRun.Run(ctx) {
			if err != nil {
				if !yield(event, err) {
					return
				}
				continue
			}

			if event == nil {
				err := fmt.Errorf("adk: agent %q yielded a nil event", agentToRun.Name())
				log.Printf("%v", err)
				yield(nil, err)
				return
			}
			stampTimestamp(ctx, event)

			if pluginManager != nil {
				modifiedEvent, err := pluginManager.RunOnEventCallback(ctx, event)
				if err != nil {
					if !yield(nil, err) {
						return
					}
					continue
				}
				if modifiedEvent != nil {
					event = modifiedEvent
					stampTimestamp(ctx, event) // a replacement may be a struct literal too
				}
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
	}
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

// runBeforeRunCallback processes and persists a plugin's replacement reply.
// A nil event and nil error mean agent execution should continue.
func (r *Runner) runBeforeRunCallback(ctx agent.InvocationContext) (*session.Event, error) {
	content, err := r.pluginManager.RunBeforeRunCallback(ctx)
	if err != nil || content == nil {
		return nil, err
	}

	event := session.NewEventWithContext(ctx, ctx.InvocationID())
	// Match RunLive's attribution so later turns can resolve the agent from history.
	event.Author = ctx.Agent().Name()
	event.Content = content

	modifiedEvent, err := r.pluginManager.RunOnEventCallback(ctx, event)
	if err != nil {
		return nil, err
	}
	// Stamp after OnEvent so a replacement keeps its BeforeRun origin, without
	// changing plugin-owned metadata.
	if modifiedEvent != nil {
		event = modifiedEvent
	}
	reply := *event
	reply.CustomMetadata = make(map[string]any, len(event.CustomMetadata)+1)
	maps.Copy(reply.CustomMetadata, event.CustomMetadata)
	reply.CustomMetadata[plugininternal.BeforeRunReplyKey] = true
	event = &reply
	stampTimestamp(ctx, event) // a replacement may be a struct literal
	if !event.Partial {
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
			event := session.NewEventWithContext(s.iCtx, s.iCtx.InvocationID())
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
	agentToRun, _, err := r.findAgentToRun(storedSession, nil)
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
		// The reply goes through OnEvent and the run through AfterRun, as on
		// Run. adk-python runs a live agent through the same
		// Runner._exec_with_plugin as run_async (runners.py).
		earlyExitEvent, err := r.runBeforeRunCallback(iCtx)
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
				log.Printf("%v", err)
				yield(nil, err)
				return
			}
			stampTimestamp(iCtx, event)

			if r.pluginManager != nil {
				modifiedEvent, pluginErr := r.pluginManager.RunOnEventCallback(iCtx, event)
				if pluginErr != nil {
					if !yield(nil, pluginErr) {
						return
					}
					continue
				}
				if modifiedEvent != nil {
					event = modifiedEvent
					stampTimestamp(iCtx, event) // a replacement may be a struct literal too
				}
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

		// innerIter has returned; every downstream stop above returns from wrappedIter.
		// Live agents reach here when their session closes, including on cancellation.
		// Python persists live events as they arrive; Go buffers them during transcription.
		// Detach cancellation so session teardown can still persist that buffer.
		// Match the mid-stream flush's persistence policy for buffered tool events,
		// including those carrying inline data.
		flushCtx := context.WithoutCancel(iCtx)
		for _, bufferedEvent := range bufferedEvents {
			if err := r.sessionService.AppendEvent(flushCtx, storedSession, bufferedEvent); err != nil {
				if !yield(nil, fmt.Errorf("failed to add event to session: %w", err)) {
					return
				}
				continue
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

func (r *Runner) appendMessageToSession(ctx agent.InvocationContext, storedSession session.Session, msg *genai.Content, saveInputBlobsAsArtifacts bool, pluginManager *plugininternal.PluginManager, stateDelta map[string]any) (agent.InvocationContext, error) {
	if msg == nil {
		return ctx, nil
	}
	if pluginManager != nil {
		modifiedMsg, err := pluginManager.RunOnUserMessageCallback(ctx, msg)
		if err != nil {
			return ctx, fmt.Errorf("error running on run user message callback : %w", err)
		}
		if modifiedMsg != nil {
			// A plugin can return a message with any Role. Normalize it before
			// it becomes the invocation context's UserContent: NewInvocationContext
			// copies the struct value, so a later reassignment would not reach it.
			msg = asUserContent(modifiedMsg)
			// update ctx user message
			ctx = icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
				Artifacts:    ctx.Artifacts(),
				Memory:       ctx.Memory(),
				Session:      ctx.Session(),
				Branch:       ctx.Branch(),
				Agent:        ctx.Agent(),
				UserContent:  msg,
				RunConfig:    ctx.RunConfig(),
				InvocationID: ctx.InvocationID(),
			})
		}
	}

	if len(utils.FunctionCalls(msg)) > 0 {
		return ctx, errUserFunctionCall
	}

	artifactsService := ctx.Artifacts()
	if artifactsService != nil && saveInputBlobsAsArtifacts {
		for i, part := range msg.Parts {
			if part.InlineData == nil {
				continue
			}
			fileName := fmt.Sprintf("artifact_%s_%d", ctx.InvocationID(), i)
			if _, err := artifactsService.Save(ctx, fileName, part); err != nil {
				return ctx, fmt.Errorf("failed to save artifact %s: %w", fileName, err)
			}
			// Replace the part with a text placeholder
			msg.Parts[i] = &genai.Part{
				Text: fmt.Sprintf("Uploaded file: %s. It has been saved to the artifacts", fileName),
			}
		}
	}

	event := session.NewEventWithContext(ctx, ctx.InvocationID())

	event.Author = "user"
	event.LLMResponse = model.LLMResponse{
		Content: msg,
	}
	if stateDelta != nil {
		event.Actions.StateDelta = stateDelta
	}

	if err := r.sessionService.AppendEvent(ctx, storedSession, event); err != nil {
		return ctx, fmt.Errorf("failed to append event to sessionService: %w", err)
	}
	return ctx, nil
}

// findAgentToRun returns the agent that should handle the next request based on
// session history, along with the branch it should run on when resuming a
// sub-agent's function call (matching adk-python's find_agent_to_run and
// restore_branch_from_history).
func (r *Runner) findAgentToRun(session session.Session, msg *genai.Content) (agent.Agent, string, error) {
	if event := handleUserFunctionCallResponse(session.Events(), msg); event != nil {
		subAgent := r.rootAgent.FindAgent(event.Author)
		if subAgent != nil {
			return subAgent, event.Branch, nil
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
			return subAgent, "", nil
		}
	}

	// Falls back to root agent if no suitable agents are found in the session.
	return r.rootAgent, "", nil
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
