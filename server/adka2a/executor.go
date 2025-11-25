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

package adka2a

import (
	"context"
	"fmt"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

// ExecutorConfig represents mandatory Executor dependencies.
type ExecutorConfig struct {
	// RunnerConfig is the configuration which will be used for [runner.New] during A2A Execute invocation.
	RunnerConfig runner.Config
	// RunConfig is the configuration which will be passed to [runner.Runner.Run] during A2A Execute invocation.
	RunConfig agent.RunConfig
}

var _ a2asrv.AgentExecutor = (*Executor)(nil)

// Executor invokes an ADK agent and translates [session.Event]s to [a2a.Event]s according to the following rules:
//   - If the input doesn't reference any a2a.Task, produce a TaskStatusUpdateEvent with TaskStateSubmitted.
//   - Right before runner.Runner invocation, produce TaskStatusUpdateEvent with TaskStateWorking.
//   - For every session.Event produce a TaskArtifactUpdateEvent{Append=true} with transformed parts.
//   - After the last session.Event is processed produce an empty TaskArtifactUpdateEvent{Append=true} with LastChunk=true,
//     if at least one artifact update was produced during the run.
//   - If there was an LLMResponse with non-zero error code, produce a TaskStatusUpdateEvent with TaskStateFailed.
//     Else if there was an LLMResponse with long-running tool invocation, produce a TaskStatusUpdateEvent with TaskStateInputRequired.
//     Else produce a TaskStatusUpdateEvent with TaskStateCompleted.
type Executor struct {
	config ExecutorConfig
}

// NewExecutor creates an initialized [Executor] instance.
func NewExecutor(config ExecutorConfig) *Executor {
	return &Executor{config: config}
}

func (e *Executor) Execute(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	msg := reqCtx.Message
	if msg == nil {
		return fmt.Errorf("message not provided")
	}
	content, err := toGenAIContent(msg)
	if err != nil {
		return fmt.Errorf("a2a message conversion failed: %w", err)
	}
	r, err := runner.New(e.config.RunnerConfig)
	if err != nil {
		return fmt.Errorf("failed to create a runner: %w", err)
	}

	if reqCtx.StoredTask == nil {
		event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateSubmitted, nil)
		if err := queue.Write(ctx, event); err != nil {
			return fmt.Errorf("failed to setup a task: %w", err)
		}
	}

	invocationMeta := toInvocationMeta(ctx, e.config, reqCtx)

	if err := e.prepareSession(ctx, invocationMeta); err != nil {
		event := toTaskFailedUpdateEvent(reqCtx, err, invocationMeta.eventMeta)
		if err := queue.Write(ctx, event); err != nil {
			return err
		}
		return nil
	}

	event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateWorking, nil)
	event.Metadata = invocationMeta.eventMeta
	if err := queue.Write(ctx, event); err != nil {
		return err
	}

	processor := newEventProcessor(reqCtx, invocationMeta)
	if err := e.process(ctx, r, processor, content, queue); err != nil {
		return err
	}

	return nil
}

func (e *Executor) Cancel(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCanceled, nil)
	if err := queue.Write(ctx, event); err != nil {
		return err
	}
	return nil
}

// Processing failures should be delivered as Task failed events. An error is returned from this method if an event write fails.
func (e *Executor) process(ctx context.Context, r *runner.Runner, processor *eventProcessor, content *genai.Content, q eventqueue.Queue) error {
	meta := processor.meta
	for event, err := range r.Run(ctx, meta.userID, meta.sessionID, content, e.config.RunConfig) {
		if err != nil {
			event := processor.makeTaskFailedEvent(fmt.Errorf("agent run failed: %w", err), nil)
			if eventSendErr := q.Write(ctx, event); eventSendErr != nil {
				return fmt.Errorf("error event write failed: %w, %w", err, eventSendErr)
			}
			return nil
		}

		a2aEvent, err := processor.process(ctx, event)
		if err != nil {
			event := processor.makeTaskFailedEvent(fmt.Errorf("processor failed: %w", err), event)
			if eventSendErr := q.Write(ctx, event); eventSendErr != nil {
				return fmt.Errorf("processor error event write failed: %w, %w", err, eventSendErr)
			}
			return nil
		}

		if a2aEvent != nil {
			if err := q.Write(ctx, a2aEvent); err != nil {
				return fmt.Errorf("send event failed: %w", err)
			}
		}
	}

	for _, ev := range processor.makeTerminalEvents() {
		if err := q.Write(ctx, ev); err != nil {
			return fmt.Errorf("terminal event send failed: %w", err)
		}
	}

	return nil
}

func (e *Executor) prepareSession(ctx context.Context, meta invocationMeta) error {
	service := e.config.RunnerConfig.SessionService

	resp, err := service.Get(ctx, &session.GetRequest{
		AppName:   e.config.RunnerConfig.AppName,
		UserID:    meta.userID,
		SessionID: meta.sessionID,
	})
	if err == nil && resp != nil {
		return nil
	}

	_, err = service.Create(ctx, &session.CreateRequest{
		AppName:   e.config.RunnerConfig.AppName,
		UserID:    meta.userID,
		SessionID: meta.sessionID,
		State:     make(map[string]any),
	})
	if err != nil {
		return fmt.Errorf("failed to create a session: %w", err)
	}
	return nil
}
