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

package v1

import (
	"context"
	"fmt"
	"iter"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

// BeforeExecuteCallback is the callback which will be called before an execution is started.
type BeforeExecuteCallback func(ctx context.Context, reqCtx *a2asrv.ExecutorContext) (context.Context, error)

// AfterEventCallback is the callback which will be called after an ADK event is converted to an A2A event.
type AfterEventCallback func(ctx ExecutorContext, event *session.Event, processed *a2a.TaskArtifactUpdateEvent) error

// AfterExecuteCallback is the callback which will be called after an execution resolved into a completed or failed task.
type AfterExecuteCallback func(ctx ExecutorContext, finalEvent *a2a.TaskStatusUpdateEvent, err error) error

// A2APartConverter is a custom converter for converting A2A parts to GenAI parts.
// Implementations should generally remember to leverage adka2a.ToGenAiPart for default conversions
// nil returns are considered intentionally dropped parts.
type A2APartConverter func(ctx context.Context, a2aEvent a2a.Event, part *a2a.Part) (*genai.Part, error)

// GenAIPartConverter is a custom converter for converting GenAI parts to A2A parts.
// Implementations should generally remember to leverage adka2a.ToA2APart for default conversions
// nil returns are considered intentionally dropped parts.
type GenAIPartConverter func(ctx context.Context, adkEvent *session.Event, part *genai.Part) (*a2a.Part, error)

// OutputMode controls how artifacts are produced.
type OutputMode string

const (
	// OutputArtifactPerRun produces a single artifact per [runner.Runner.Run].
	OutputArtifactPerRun OutputMode = "artifact-per-run"
	// OutputArtifactPerEvent produces an artifact per non-partial [session.Event].
	// While agent is emitting events an artifact is build incrementally (parts are append to it).
	// The next partial event replaces accumulated contents and seals the artifact, meaning
	// the next event from this agent will create a new artifact.
	OutputArtifactPerEvent OutputMode = "artifact-per-event"
)

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

	// A2APartConverter is a custom converter for converting A2A parts to GenAI parts.
	// Implementations should generally remember to leverage [adka2a.ToGenAiPart] for default conversions
	// nil returns are considered intentionally dropped parts.
	A2APartConverter A2APartConverter

	// GenAIPartConverter is a custom converter for converting GenAI parts to A2A parts.
	// Implementations should generally remember to leverage [adka2a.ToA2APart] for default conversions
	// nil returns are considered intentionally dropped parts.
	GenAIPartConverter GenAIPartConverter

	// OutputMode controls how artifacts are produced. Can be [OutputArtifactPerRun] or [OutputArtifactPerEvent].
	// Defaults to [OutputArtifactPerRun].
	OutputMode OutputMode
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

func (e *Executor) Execute(ctx context.Context, execCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		msg := execCtx.Message
		if msg == nil {
			yield(nil, fmt.Errorf("message not provided"))
			return
		}
		content, err := toGenAIContent(ctx, msg, e.config.A2APartConverter)
		if err != nil {
			yield(nil, fmt.Errorf("a2a message conversion failed: %w", err))
			return
		}

		runnerCfg, executorPlugin, err := withExecutorPlugin(e.config.RunnerConfig)
		if err != nil {
			yield(nil, fmt.Errorf("failed to install a2a-executor plugin: %w", err))
			return
		}

		r, err := runner.New(runnerCfg)
		if err != nil {
			yield(nil, fmt.Errorf("failed to create a runner: %w", err))
			return
		}
		if e.config.BeforeExecuteCallback != nil {
			ctx, err = e.config.BeforeExecuteCallback(ctx, execCtx)
			if err != nil {
				yield(nil, fmt.Errorf("before execute: %w", err))
				return
			}
		}

		if event, err := HandleInputRequired(execCtx, content); event != nil || err != nil {
			if err != nil {
				yield(nil, err)
			} else {
				yield(event, nil)
			}
			return
		}

		if execCtx.StoredTask == nil {
			event := a2a.NewSubmittedTask(execCtx, msg)
			if !yield(event, nil) {
				return
			}
		}

		invocationMeta := toInvocationMeta(ctx, e.config, execCtx)

		err = e.prepareSession(ctx, invocationMeta)
		if err != nil {
			statusEvent := toTaskFailedUpdateEvent(execCtx, err, invocationMeta.eventMeta)
			execCtx := newExecutorContext(ctx, invocationMeta, executorPlugin, content)
			e.writeFinalTaskStatus(execCtx, yield, nil, statusEvent, err)
			return
		}

		event := a2a.NewStatusUpdateEvent(execCtx, a2a.TaskStateWorking, nil)
		event.Metadata = invocationMeta.eventMeta
		if !yield(event, nil) {
			return
		}

		var artifactTransform eventToArtifactTransform
		if e.config.OutputMode == OutputArtifactPerEvent {
			artifactTransform = newArtifactMaker(execCtx)
		} else {
			artifactTransform = newLegacyArtifactMaker(execCtx)
		}

		processor := newEventProcessor(execCtx, invocationMeta, e.config.GenAIPartConverter, artifactTransform)
		executorContext := newExecutorContext(ctx, invocationMeta, executorPlugin, content)
		e.process(executorContext, r, processor, yield)
	}
}

func (e *Executor) Cancel(ctx context.Context, execCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		event := a2a.NewStatusUpdateEvent(execCtx, a2a.TaskStateCanceled, nil)
		yield(event, nil)
	}
}

// Processing failures should be delivered as Task failed events. An error is returned from this method if an event write fails.
func (e *Executor) process(ctx ExecutorContext, r *runner.Runner, processor *eventProcessor, yield func(a2a.Event, error) bool) {
	meta := processor.meta
	for adkEvent, adkErr := range r.Run(ctx, meta.userID, meta.sessionID, ctx.UserContent(), e.config.RunConfig) {
		if adkErr != nil {
			event := processor.makeTaskFailedEvent(fmt.Errorf("agent run failed: %w", adkErr), nil)
			e.writeFinalTaskStatus(ctx, yield, processor.makeFinalArtifactUpdate(), event, adkErr)
			return
		}

		a2aEvent, pErr := processor.process(ctx, adkEvent)
		if pErr == nil && a2aEvent != nil && e.config.AfterEventCallback != nil {
			pErr = e.config.AfterEventCallback(ctx, adkEvent, a2aEvent)
		}

		if pErr != nil {
			event := processor.makeTaskFailedEvent(fmt.Errorf("processor failed: %w", pErr), adkEvent)
			e.writeFinalTaskStatus(ctx, yield, processor.makeFinalArtifactUpdate(), event, pErr)
			return
		}

		if a2aEvent != nil {
			if !yield(a2aEvent, nil) {
				return
			}
		}
	}

	finalStatus := processor.makeFinalStatusUpdate()
	e.writeFinalTaskStatus(ctx, yield, processor.makeFinalArtifactUpdate(), finalStatus, nil)
}

func (e *Executor) writeFinalTaskStatus(
	ctx ExecutorContext,
	yield func(a2a.Event, error) bool,
	partialReset *a2a.TaskArtifactUpdateEvent,
	status *a2a.TaskStatusUpdateEvent,
	err error,
) {
	if e.config.AfterExecuteCallback != nil {
		if err = e.config.AfterExecuteCallback(ctx, status, err); err != nil {
			yield(nil, fmt.Errorf("after execute: %w", err))
			return
		}
	}
	if partialReset != nil {
		if !yield(partialReset, nil) {
			return
		}
	}
	yield(status, nil)
}

func (e *Executor) prepareSession(ctx context.Context, meta invocationMeta) error {
	service := e.config.RunnerConfig.SessionService

	_, err := service.Get(ctx, &session.GetRequest{
		AppName:   e.config.RunnerConfig.AppName,
		UserID:    meta.userID,
		SessionID: meta.sessionID,
	})
	if err == nil {
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
