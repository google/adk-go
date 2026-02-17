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

	"strings"

	"github.com/rs/zerolog/log"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/internal/agent/parentmap"
	"google.golang.org/adk/internal/agent/runconfig"
	artifactinternal "google.golang.org/adk/internal/artifact"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/llminternal"
	imemory "google.golang.org/adk/internal/memory"
	"google.golang.org/adk/internal/sessioninternal"
	"google.golang.org/adk/memory"
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

	ResumabilityConfig *agent.ResumabilityConfig
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

	return &Runner{
		appName:            cfg.AppName,
		rootAgent:          cfg.Agent,
		sessionService:     cfg.SessionService,
		artifactService:    cfg.ArtifactService,
		memoryService:      cfg.MemoryService,
		parents:            parents,
		resumabilityConfig: cfg.ResumabilityConfig,
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

	parents            parentmap.Map
	resumabilityConfig *agent.ResumabilityConfig
}

/*
Runs the agent in live mode (experimental feature).

The `run_live` method yields a stream of `Event` objects, but not all
yielded events are saved to the session. Here's a breakdown:

**Events Yielded to Callers:**
  - **Live Model Audio Events with Inline Data:** Events containing raw
    audio `Blob` data(`inline_data`).
  - **Live Model Audio Events with File Data:** Both input and ouput audio
    data are aggregated into an audio file saved into artifacts. The
    reference to the file is saved in the event as `file_data`.
  - **Usage Metadata:** Events containing token usage.
  - **Transcription Events:** Both partial and non-partial transcription
    events are yielded.
  - **Function Call and Response Events:** Always saved.
  - **Other Control Events:** Most control events are saved.

**Events Saved to the Session:**
  - **Live Model Audio Events with File Data:** Both input and ouput audio
    data are aggregated into an audio file saved into artifacts. The
    reference to the file is saved as event in the `file_data` to session
    if RunConfig.save_live_model_audio_to_session is True.
  - **Usage Metadata Events:** Saved to the session.
  - **Non-Partial Transcription Events:** Non-partial transcription events
    are saved.
  - **Function Call and Response Events:** Always saved.
  - **Other Control Events:** Most control events are saved.

**Events Not Saved to the Session:**
  - **Live Model Audio Events with Inline Data:** Events containing raw
    audio `Blob` data are **not** saved to the session.

Args:

	user_id: The user ID for the session. Required if `session` is None.
	session_id: The session ID for the session. Required if `session` is
		None.
	live_request_queue: The queue for live requests.
	run_config: The run config for the agent.

Yields:

	AsyncGenerator[Event, None]: An asynchronous generator that yields
	`Event`
	objects as they are produced by the agent during its live execution.

.. warning::

	This feature is **experimental** and its API or behavior may change
	in future releases.
*/
func (r *Runner) RunLive(ctx context.Context, userID, sessionID string, liveRequestQueue *agent.LiveRequestQueue, cfg agent.RunConfig) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if len(cfg.ResponseModalities) == 0 {
			cfg.ResponseModalities = []genai.Modality{genai.ModalityAudio}
		}

		if cfg.SpeechConfig == nil {
			cfg.SpeechConfig = &genai.SpeechConfig{
				VoiceConfig: &genai.VoiceConfig{
					PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
						VoiceName: "Aoede",
					},
				},
			}
		}

		if strings.TrimSpace(userID) == "" || strings.TrimSpace(sessionID) == "" {
			yield(nil, fmt.Errorf("userID and sessionID must be provided."))
			return
		}

		if liveRequestQueue == nil {
			yield(nil, fmt.Errorf("live request queue must be provided"))
			return
		}

		resp, err := r.sessionService.Get(ctx, &session.GetRequest{
			AppName:   r.appName,
			UserID:    userID,
			SessionID: sessionID,
		})
		if err != nil {
			yield(nil, err)
			return
		}

		// For live multi-agents system, we need model's text transcription as
		// context for the transferred agent.
		if len(r.rootAgent.SubAgents()) > 0 {
			found := false
			for _, modality := range cfg.ResponseModalities {
				if modality == genai.ModalityAudio {
					found = true
					break
				}
			}
			if found {
				if cfg.InputAudioTranscription == nil {
					cfg.InputAudioTranscription = &genai.AudioTranscriptionConfig{}
				}
				if cfg.OutputAudioTranscription == nil {
					cfg.OutputAudioTranscription = &genai.AudioTranscriptionConfig{}
				}
			}
		}

		storedSession := resp.Session

		agentToRun, err := r.findAgentToRun(storedSession)
		if err != nil {
			yield(nil, err)
			return
		}

		invCtx := r.newInvocationContextForLive(ctx, userID, sessionID, liveRequestQueue, cfg, agentToRun, storedSession)

		for event, err := range agentToRun.RunLive(invCtx) {
			if err != nil {
				if !yield(event, err) {
					return
				}
				continue
			}

			// if r.shouldAppendEvent(event, true) {
			if err := r.sessionService.AppendEvent(invCtx, storedSession, event); err != nil {
				yield(nil, fmt.Errorf("failed to add event to session: %w", err))
				return
			}
			// }

			if !yield(event, nil) {
				return
			}
		}
	}
}

func (r *Runner) newInvocationContextForLive(ctx context.Context, userID, sessionID string, liveRequestQueue *agent.LiveRequestQueue, cfg agent.RunConfig, agentToRun agent.Agent, session session.Session) agent.InvocationContext {
	liveConnectConfig := &genai.LiveConnectConfig{
		ResponseModalities:       cfg.ResponseModalities,
		SpeechConfig:             cfg.SpeechConfig,
		InputAudioTranscription:  cfg.InputAudioTranscription,
		OutputAudioTranscription: cfg.OutputAudioTranscription,
		RealtimeInputConfig:      cfg.RealtimeInputConfig,
		ContextWindowCompression: cfg.ContextWindowCompression,
		Proactivity:              cfg.Proactivity,
	}

	if cfg.ExplicitVADSignal {
		liveConnectConfig.ExplicitVADSignal = &cfg.ExplicitVADSignal
	}
	if cfg.EnableAffectiveDialog {
		liveConnectConfig.EnableAffectiveDialog = &cfg.EnableAffectiveDialog
	}

	if r.resumabilityConfig != nil && r.resumabilityConfig.IsResumable {
		if cfg.SessionResumption != nil {
			liveConnectConfig.SessionResumption = cfg.SessionResumption
		} else {
			liveConnectConfig.SessionResumption = &genai.SessionResumptionConfig{
				Handle:      fmt.Sprintf("%s_%s", sessionID, userID),
				Transparent: true,
			}
		}
	}

	ctx = parentmap.ToContext(ctx, r.parents)
	ctx = runconfig.ToContext(ctx, &runconfig.RunConfig{
		StreamingMode:     runconfig.StreamingMode(cfg.StreamingMode),
		LiveConnectConfig: liveConnectConfig,
	})

	var artifacts agent.Artifacts
	if r.artifactService != nil {
		artifacts = &artifactinternal.Artifacts{
			Service:   r.artifactService,
			SessionID: session.ID(),
			AppName:   session.AppName(),
			UserID:    session.UserID(),
		}
	}

	var memoryImpl agent.Memory = nil
	if r.memoryService != nil {
		memoryImpl = &imemory.Memory{
			Service:   r.memoryService,
			SessionID: session.ID(),
			UserID:    session.UserID(),
			AppName:   session.AppName(),
		}
	}

	invCtx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Artifacts:                   artifacts,
		Memory:                      memoryImpl,
		Session:                     sessioninternal.NewMutableSession(r.sessionService, session),
		Agent:                       agentToRun,
		RunConfig:                   &cfg,
		LiveRequestQueue:            liveRequestQueue,
		LiveSessionResumptionHandle: "", //TODO how we get this from session?
	})
	return invCtx
}

func (r *Runner) shouldAppendEvent(event *session.Event, isLiveCall bool) bool {
	if isLiveCall && isLiveModelAudioEventWithInlineData(event) {
		return false
	}
	return !event.Partial
}

func isLiveModelAudioEventWithInlineData(event *session.Event) bool {
	if event.Content == nil {
		return false
	}
	for _, part := range event.Content.Parts {
		if part.InlineData != nil && strings.HasPrefix(part.InlineData.MIMEType, "audio/") {
			return true
		}
	}
	return false
}

// Run runs the agent for the given user input, yielding events from agents.
// For each user message it finds the proper agent within an agent tree to
// continue the conversation within the session.
func (r *Runner) Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg agent.RunConfig) iter.Seq2[*session.Event, error] {
	// TODO(hakim): we need to validate whether cfg is compatible with the Agent.
	//   see adk-python/src/google/adk/runners.py Runner._new_invocation_context.
	// TODO: setup tracer.
	return func(yield func(*session.Event, error) bool) {
		resp, err := r.sessionService.Get(ctx, &session.GetRequest{
			AppName:   r.appName,
			UserID:    userID,
			SessionID: sessionID,
		})
		if err != nil {
			yield(nil, err)
			return
		}

		session := resp.Session

		agentToRun, err := r.findAgentToRun(session)
		if err != nil {
			yield(nil, err)
			return
		}

		ctx = parentmap.ToContext(ctx, r.parents)
		ctx = runconfig.ToContext(ctx, &runconfig.RunConfig{
			StreamingMode: runconfig.StreamingMode(cfg.StreamingMode),
		})

		var artifacts agent.Artifacts
		if r.artifactService != nil {
			artifacts = &artifactinternal.Artifacts{
				Service:   r.artifactService,
				SessionID: session.ID(),
				AppName:   session.AppName(),
				UserID:    session.UserID(),
			}
		}

		var memoryImpl agent.Memory = nil
		if r.memoryService != nil {
			memoryImpl = &imemory.Memory{
				Service:   r.memoryService,
				SessionID: session.ID(),
				UserID:    session.UserID(),
				AppName:   session.AppName(),
			}
		}

		ctx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
			Artifacts:   artifacts,
			Memory:      memoryImpl,
			Session:     sessioninternal.NewMutableSession(r.sessionService, session),
			Agent:       agentToRun,
			UserContent: msg,
			RunConfig:   &cfg,
		})

		for event, err := range agentToRun.Run(ctx) {
			if err != nil {
				if !yield(event, err) {
					return
				}
				continue
			}

			// only commit non-partial event to a session service
			if !event.LLMResponse.Partial {
				if err := r.sessionService.AppendEvent(ctx, session, event); err != nil {
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

// findAgentToRun returns the agent that should handle the next request based on
// session history.
func (r *Runner) findAgentToRun(session session.Session) (agent.Agent, error) {
	events := session.Events()
	for i := events.Len() - 1; i >= 0; i-- {
		event := events.At(i)

		// TODO: findMatchingFunctionCall.

		if event.Author == "user" {
			continue
		}

		subAgent := findAgent(r.rootAgent, event.Author)
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

func findAgent(curAgent agent.Agent, targetName string) agent.Agent {
	if curAgent == nil || curAgent.Name() == targetName {
		return curAgent
	}

	for _, subAgent := range curAgent.SubAgents() {
		if agent := findAgent(subAgent, targetName); agent != nil {
			return agent
		}
	}
	return nil
}
