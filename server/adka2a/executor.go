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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

// BeforeExecuteCallback is the callback which will be called before an execution is started.
type BeforeExecuteCallback func(ctx context.Context, reqCtx *a2asrv.RequestContext) (context.Context, error)

// AfterEventCallback is the callback which will be called after an ADK event is converted to an A2A event.
type AfterEventCallback func(ctx ExecutorContext, event *session.Event, processed *a2a.TaskArtifactUpdateEvent) error

// AfterExecuteCallback is the callback which will be called after an execution resolved into a completed or failed task.
type AfterExecuteCallback func(ctx ExecutorContext, finalEvent *a2a.TaskStatusUpdateEvent, err error) error

// ExecutorConfig allows to configure Executor.
type ExecutorConfig struct {
	// RunnerConfig is the configuration which will be used for [runner.New] during A2A Execute invocation.
	RunnerConfig runner.Config

	// RunConfig is the configuration which will be passed to [runner.Runner.Run] during A2A Execute invocation.
	RunConfig agent.RunConfig

	// BeforeExecuteCallback is the callback which will be called before an execution is started.
	// It can be used to instrument a context or prevent the execution by returning an error.
	BeforeExecuteCallback BeforeExecuteCallback

	// AfterEventCallback is the callback which will be called after an ADK event is successfully converted to an A2A event.
	// This gives an opportunity to enrich the event with additional metadata or abort the execution by returning an error.
	// The callback is not invoked for errors originating from ADK or event processing. Such errors are converted to
	// TaskStatusUpdateEvent-s with TaskStateFailed state. If needed these can be intercepted using AfterExecuteCallback.
	AfterEventCallback AfterEventCallback

	// AfterExecuteCallback is the callback which will be called after an execution resolved into a completed or failed task.
	// This gives an opportunity to enrich the event with additional metadata or log it.
	AfterExecuteCallback AfterExecuteCallback
}

var _ a2asrv.AgentExecutor = (*Executor)(nil)

// Executor invokes an ADK agent and translates [session.Event]-s to [a2a.Event]-s according to the following rules:
//   - If the input doesn't reference any a2a.Task, produce a Task with TaskStateSubmitted state.
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
	if e.config.BeforeExecuteCallback != nil {
		ctx, err = e.config.BeforeExecuteCallback(ctx, reqCtx)
		if err != nil {
			return fmt.Errorf("before execute: %w", err)
		}
	}

	if event, err := handleInputRequired(reqCtx, content); event != nil || err != nil {
		if err != nil {
			return err
		}
		return queue.Write(ctx, event)
	}

	if reqCtx.StoredTask == nil {
		event := a2a.NewSubmittedTask(reqCtx, msg)
		if err := queue.Write(ctx, event); err != nil {
			return fmt.Errorf("failed to submit a task: %w", err)
		}
	}

	invocationMeta := toInvocationMeta(ctx, e.config, reqCtx)

	session, err := e.prepareSession(ctx, invocationMeta)
	if err != nil {
		event := toTaskFailedUpdateEvent(reqCtx, err, invocationMeta.eventMeta)
		execCtx := newExecutorContext(ctx, invocationMeta, emptySessionState{}, content)
		return e.writeFinalTaskStatus(execCtx, queue, event, err)
	}

	event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateWorking, nil)
	event.Metadata = invocationMeta.eventMeta
	if err := queue.Write(ctx, event); err != nil {
		return err
	}

	processor := newEventProcessor(reqCtx, invocationMeta)
	executorContext := newExecutorContext(ctx, invocationMeta, session.State(), content)
	return e.process(executorContext, r, processor, queue)
}

func (e *Executor) Cancel(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCanceled, nil)
	event.Final = true
	return queue.Write(ctx, event)
}

// Processing failures should be delivered as Task failed events. An error is returned from this method if an event write fails.
func (e *Executor) process(ctx ExecutorContext, r *runner.Runner, processor *eventProcessor, q eventqueue.Queue) error {
	meta := processor.meta
	for adkEvent, adkErr := range r.Run(ctx, meta.userID, meta.sessionID, ctx.UserContent(), e.config.RunConfig) {
		if adkErr != nil {
			event := processor.makeTaskFailedEvent(fmt.Errorf("agent run failed: %w", adkErr), nil)
			return e.writeFinalTaskStatus(ctx, q, event, adkErr)
		}

		a2aEvent, pErr := processor.process(ctx, adkEvent)
		if pErr == nil && a2aEvent != nil && e.config.AfterEventCallback != nil {
			pErr = e.config.AfterEventCallback(ctx, adkEvent, a2aEvent)
		}

		if pErr != nil {
			event := processor.makeTaskFailedEvent(fmt.Errorf("processor failed: %w", pErr), adkEvent)
			return e.writeFinalTaskStatus(ctx, q, event, pErr)
		}

		if a2aEvent != nil {
			if err := q.Write(ctx, a2aEvent); err != nil {
				return fmt.Errorf("event write failed: %w", err)
			}
		}
	}

	if finalChunk, ok := processor.makeFinalArtifactUpdate(); ok {
		if err := q.Write(ctx, finalChunk); err != nil {
			return fmt.Errorf("final artifact update write failed: %w", err)
		}
	}

	finalStatus := processor.makeFinalStatusUpdate()
	return e.writeFinalTaskStatus(ctx, q, finalStatus, nil)
}

func (e *Executor) writeFinalTaskStatus(ctx ExecutorContext, queue eventqueue.Queue, status *a2a.TaskStatusUpdateEvent, err error) error {
	if e.config.AfterExecuteCallback != nil {
		if err = e.config.AfterExecuteCallback(ctx, status, err); err != nil {
			return fmt.Errorf("after execute: %w", err)
		}
	}
	if err := queue.Write(ctx, status); err != nil {
		return fmt.Errorf("%q state update event write failed: %w", status.Status.State, err)
	}
	return nil
}

func (e *Executor) prepareSession(ctx context.Context, meta invocationMeta) (session.Session, error) {
	service := e.config.RunnerConfig.SessionService

	getResp, err := service.Get(ctx, &session.GetRequest{
		AppName:   e.config.RunnerConfig.AppName,
		UserID:    meta.userID,
		SessionID: meta.sessionID,
	})
	if err == nil {
		return getResp.Session, nil
	}

	createResp, err := service.Create(ctx, &session.CreateRequest{
		AppName:   e.config.RunnerConfig.AppName,
		UserID:    meta.userID,
		SessionID: meta.sessionID,
		State:     make(map[string]any),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create a session: %w", err)
	}
	return createResp.Session, nil
}
