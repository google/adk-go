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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/internal/agent/parentmap"
	"google.golang.org/adk/internal/agent/runconfig"
	artifactinternal "google.golang.org/adk/internal/artifact"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/llminternal"
	imemory "google.golang.org/adk/internal/memory"
	"google.golang.org/adk/internal/plugininternal"
	"google.golang.org/adk/internal/sessioninternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
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
}

type PluginConfig struct {
	Plugins      []*plugin.Plugin
	CloseTimeout time.Duration
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
		appName:         cfg.AppName,
		rootAgent:       cfg.Agent,
		sessionService:  cfg.SessionService,
		artifactService: cfg.ArtifactService,
		memoryService:   cfg.MemoryService,
		parents:         parents,
		pluginManager:   pluginManager,
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

	parents       parentmap.Map
	pluginManager *plugininternal.PluginManager
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

		storedSession := resp.Session

		agentToRun, err := r.findAgentToRun(storedSession, msg)
		if err != nil {
			yield(nil, err)
			return
		}

		ctx = parentmap.ToContext(ctx, r.parents)
		ctx = runconfig.ToContext(ctx, &runconfig.RunConfig{
			StreamingMode: runconfig.StreamingMode(cfg.StreamingMode),
		})
		ctx = plugininternal.ToContext(ctx, r.pluginManager)

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
			Session:     sessioninternal.NewMutableSession(r.sessionService, storedSession),
			Agent:       agentToRun,
			UserContent: msg,
			RunConfig:   &cfg,
		})
		ctx, err = r.appendMessageToSession(ctx, storedSession, msg, cfg.SaveInputBlobsAsArtifacts, r.pluginManager)
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
				earlyExitEvent := session.NewEvent(ctx.InvocationID())
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

		for event, err := range agentToRun.Run(ctx) {
			if err != nil {
				if !yield(event, err) {
					return
				}
				continue
			}

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

func (r *Runner) appendMessageToSession(ctx agent.InvocationContext, storedSession session.Session, msg *genai.Content, saveInputBlobsAsArtifacts bool, pluginManager *plugininternal.PluginManager) (agent.InvocationContext, error) {
	if msg == nil {
		return ctx, nil
	}
	if pluginManager != nil {
		modifiedMsg, err := pluginManager.RunOnUserMessageCallback(ctx, msg)
		if err != nil {
			return ctx, fmt.Errorf("error running on run user message callback : %w", err)
		}
		if modifiedMsg != nil {
			msg = modifiedMsg
			// update ctx user message
			ctx = icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
				Artifacts:   ctx.Artifacts(),
				Memory:      ctx.Memory(),
				Session:     ctx.Session(),
				Agent:       ctx.Agent(),
				UserContent: msg,
				RunConfig:   ctx.RunConfig(),
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
				return ctx, fmt.Errorf("failed to save artifact %s: %w", fileName, err)
			}
			// Replace the part with a text placeholder
			msg.Parts[i] = &genai.Part{
				Text: fmt.Sprintf("Uploaded file: %s. It has been saved to the artifacts", fileName),
			}
		}
	}

	event := session.NewEvent(ctx.InvocationID())

	event.Author = "user"
	event.LLMResponse = model.LLMResponse{
		Content: msg,
	}

	if err := r.sessionService.AppendEvent(ctx, storedSession, event); err != nil {
		return ctx, fmt.Errorf("failed to append event to sessionService: %w", err)
	}
	return ctx, nil
}

// findAgentToRun returns the agent that should handle the next request based on
// session history.
func (r *Runner) findAgentToRun(session session.Session, msg *genai.Content) (agent.Agent, error) {
	if event := handleUserFunctionCallResponse(session.Events(), msg); event != nil {
		subAgent := findAgent(r.rootAgent, event.Author)
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
