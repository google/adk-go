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

// Package adka2a allows exposing ADK agents via A2A.
//
// Deprecated: Use google.golang.org/adk/server/adka2a/v2 instead.
package adka2a

import (
	"context"
	"errors"
	"fmt"
	"maps"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
	"github.com/a2aproject/a2a-go/log"
	a2av2 "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2acompat/a2av0"
	a2asrvv2 "github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	v2 "google.golang.org/adk/server/adka2a/v2"
	"google.golang.org/adk/session"
)

// BeforeExecuteCallback is the callback which will be called before an execution is started.
type BeforeExecuteCallback func(ctx context.Context, reqCtx *a2asrv.RequestContext) (context.Context, error)

// AfterEventCallback is the callback which will be called after an ADK event is converted to an A2A event.
type AfterEventCallback func(ctx ExecutorContext, event *session.Event, processed *a2a.TaskArtifactUpdateEvent) error

// AfterExecuteCallback is the callback which will be called after an execution resolved into a completed or failed task.
type AfterExecuteCallback func(ctx ExecutorContext, finalEvent *a2a.TaskStatusUpdateEvent, err error) error

// A2APartConverter is a custom converter for converting A2A parts to GenAI parts.
// Implementations should generally remember to leverage adka2a.ToGenAiPart for default conversions
// nil returns are considered intentionally dropped parts.
type A2APartConverter func(ctx context.Context, a2aEvent a2a.Event, part a2a.Part) (*genai.Part, error)

// GenAIPartConverter is a custom converter for converting GenAI parts to A2A parts.
// Implementations should generally remember to leverage adka2a.ToA2APart for default conversions
// nil returns are considered intentionally dropped parts.
type GenAIPartConverter func(ctx context.Context, adkEvent *session.Event, part *genai.Part) (a2a.Part, error)

// A2AExecutionCleanupCallback is a callback which will be called after an execution or cancellation has completed or failed.
type A2AExecutionCleanupCallback func(ctx context.Context, reqCtx *a2asrv.RequestContext, subAgentCards []*a2a.AgentCard, result a2a.SendMessageResult, cause error)

// OutputMode controls how artifacts are produced.
type OutputMode = v2.OutputMode

// Runner is an interface matching [runner.Runner] API.
// It exists to let users use custom runner implementations with A2A agent executor.
type Runner = v2.Runner

// RunnerProvider is a [Runner] factory function. The provided plugin must be installed in the returned [Runner] for
// callbacks taking [ExecutorContext] to work correctly.
type RunnerProvider = v2.RunnerProvider

const (
	// OutputArtifactPerRun produces a single artifact per [runner.Runner.Run].
	OutputArtifactPerRun OutputMode = v2.OutputArtifactPerRun
	// OutputArtifactPerEvent produces an artifact per non-partial [session.Event].
	// While agent is emitting events an artifact is build incrementally (parts are append to it).
	// The next partial event replaces accumulated contents and seals the artifact, meaning
	// the next event from this agent will create a new artifact.
	OutputArtifactPerEvent OutputMode = v2.OutputArtifactPerEvent
)

// RunnerConfig is part of the runner configuration executor code depends on.
// Custom [RunnerProvider] needs to return it back to callers.
type RunnerConfig struct {
	// AppName is the name of the application used in [session.Service] keys and A2A event metadata.
	AppName string
	// Agent is the root agent.
	Agent agent.Agent
	// SessionService is the session service to use.
	SessionService session.Service
}

// ExecutorConfig allows to configure Executor.
type ExecutorConfig struct {
	// RunnerConfig is used for creating a default RunnerProvider. The field is ignored when RunnerProvider is set.
	RunnerConfig runner.Config
	// RunnerProvider is a function which allows to control how a runner is created.
	// If not provided the default provider is used which calls [runner.New] with the RunnerConfig field.
	RunnerProvider RunnerProvider

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

	// A2AExecutionCleanupCallback is a callback which will be called after an execution or cancellation has completed or failed.
	// If not provided, the default behavior is to log the failure cause, if any.
	A2AExecutionCleanupCallback A2AExecutionCleanupCallback
}

var _ a2asrv.AgentExecutor = (*Executor)(nil)

// Executor is the legacy AgentExecutor implementation which delegates to [v2.Executor].
type Executor struct {
	impl *v2.Executor
}

// NewExecutor creates an initialized [Executor] instance.
func NewExecutor(config ExecutorConfig) *Executor {
	v1Config := v2.ExecutorConfig{
		RunnerConfig:   config.RunnerConfig,
		RunnerProvider: config.RunnerProvider,
		RunConfig:      config.RunConfig,
		OutputMode:     v2.OutputMode(config.OutputMode),
	}

	if config.BeforeExecuteCallback != nil {
		v1Config.BeforeExecuteCallback = func(ctx context.Context, execCtx *a2asrvv2.ExecutorContext) (context.Context, error) {
			legacyReqCtx := toRequestContext(execCtx)
			newCtx, err := config.BeforeExecuteCallback(ctx, legacyReqCtx)
			if err != nil {
				return nil, err
			}
			v1ExecCtx, err := toExecutorContext(newCtx, legacyReqCtx)
			if err != nil {
				return nil, err
			}
			*execCtx = *v1ExecCtx
			return newCtx, nil
		}
	}

	if config.AfterEventCallback != nil {
		v1Config.AfterEventCallback = func(ctx v2.ExecutorContext, adkEvent *session.Event, a2aEvent *a2av2.TaskArtifactUpdateEvent) error {
			legacyEvent := a2av0.FromV1TaskArtifactUpdateEvent(a2aEvent)
			if err := config.AfterEventCallback(executorContextWrapper{ctx}, adkEvent, legacyEvent); err != nil {
				return err
			}
			newV1Event, err := a2av0.ToV1Event(legacyEvent)
			if err != nil {
				return err
			}
			if converted, ok := newV1Event.(*a2av2.TaskArtifactUpdateEvent); ok {
				*a2aEvent = *converted
			}
			return nil
		}
	}

	if config.AfterExecuteCallback != nil {
		v1Config.AfterExecuteCallback = func(ctx v2.ExecutorContext, finalEvent *a2av2.TaskStatusUpdateEvent, err error) error {
			legacyEvent := a2av0.FromV1TaskStatusUpdateEvent(finalEvent)
			if cbErr := config.AfterExecuteCallback(executorContextWrapper{ctx}, legacyEvent, err); cbErr != nil {
				return cbErr
			}
			newV1Event, convErr := a2av0.ToV1Event(legacyEvent)
			if convErr != nil {
				return convErr
			}
			if converted, ok := newV1Event.(*a2av2.TaskStatusUpdateEvent); ok {
				*finalEvent = *converted
			}
			return nil
		}
	}

	if config.A2APartConverter != nil {
		v1Config.A2APartConverter = func(ctx context.Context, a2aEvent a2av2.Event, part *a2av2.Part) (*genai.Part, error) {
			legacyEvent, err := a2av0.FromV1Event(a2aEvent)
			if err != nil {
				return nil, fmt.Errorf("a2a event conversion failed: %w", err)
			}
			return config.A2APartConverter(ctx, legacyEvent, a2av0.FromV1Part(part))
		}
	}

	if config.GenAIPartConverter != nil {
		v1Config.GenAIPartConverter = func(ctx context.Context, adkEvent *session.Event, part *genai.Part) (*a2av2.Part, error) {
			legacyPart, err := config.GenAIPartConverter(ctx, adkEvent, part)
			if err != nil {
				return nil, err
			}
			return a2av0.ToV1Part(legacyPart), nil
		}
	}

	if config.A2AExecutionCleanupCallback != nil {
		v1Config.A2AExecutionCleanupCallback = func(ctx context.Context, execCtx *a2asrvv2.ExecutorContext, subAgentCards []*a2av2.AgentCard, result a2av2.SendMessageResult, cause error) {
			legacyReqCtx := toRequestContext(execCtx)
			legacySubAgentCards := make([]*a2a.AgentCard, len(subAgentCards))
			for i, card := range subAgentCards {
				legacySubAgentCards[i] = a2av0.FromV1AgentCard(card)
			}
			legacyEvent, err := a2av0.FromV1Event(result)
			if err != nil {
				log.Warn(ctx, "failed to convert SendMessageResult to legacy format", "error", err)
				config.A2AExecutionCleanupCallback(ctx, legacyReqCtx, legacySubAgentCards, nil, errors.Join(cause, fmt.Errorf("SendMessageResult conversion failed: %w", err)))
				return
			}
			legacyResult, ok := legacyEvent.(a2a.SendMessageResult)
			if !ok {
				log.Warn(ctx, "conversion result is not a2a.SendMessageResult", "type", fmt.Sprintf("%T", legacyResult))
				config.A2AExecutionCleanupCallback(ctx, legacyReqCtx, legacySubAgentCards, nil, errors.Join(cause, fmt.Errorf("conversion result is not a2a.SendMessageResult: %T", legacyEvent)))
				return
			}
			config.A2AExecutionCleanupCallback(ctx, legacyReqCtx, legacySubAgentCards, legacyResult, cause)
		}
	}

	return &Executor{impl: v2.NewExecutor(v1Config)}
}

func (e *Executor) Execute(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	execCtx, err := toExecutorContext(ctx, reqCtx)
	if err != nil {
		return err
	}

	for event, err := range e.impl.Execute(ctx, execCtx) {
		if err != nil {
			return err
		}
		legacyEvent, lErr := a2av0.FromV1Event(event)
		if lErr != nil {
			return lErr
		}
		if err := queue.Write(ctx, legacyEvent); err != nil {
			return err
		}
	}

	return nil
}

func (e *Executor) Cancel(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	v2ReqCtx, err := toExecutorContext(ctx, reqCtx)
	if err != nil {
		return err
	}

	for event, err := range e.impl.Cancel(ctx, v2ReqCtx) {
		if err != nil {
			return err
		}
		legacyEvent, lErr := a2av0.FromV1Event(event)
		if lErr != nil {
			return lErr
		}
		if err := queue.Write(ctx, legacyEvent); err != nil {
			return err
		}
	}

	return nil
}

// ExecutorContext provides read-only information about the context of an A2A agent execution.
// An execution starts with a user message and ends with a task in a terminal or input-required state.
type ExecutorContext interface {
	context.Context

	// SessionID is ID of the session. It is passed as contextID in A2A request.
	SessionID() string
	// UserID is ID of the user who made the request. The information is either extracted from [a2asrv.CallContext]
	// or derived from session ID for unauthenticated requests.
	UserID() string
	// AgentName is the name of the root agent.
	AgentName() string
	// ReadonlyState provides a view of the current session state.
	ReadonlyState() session.ReadonlyState
	// Events provides a readonly view of the current session events.
	Events() session.Events
	// UserContent is a converted A2A message which is passed to runner.Run.
	UserContent() *genai.Content
	// RequestContext contains information about the original A2A Request, the current task and related tasks.
	RequestContext() *a2asrv.RequestContext
}

type executorContextWrapper struct {
	v2.ExecutorContext
}

func (w executorContextWrapper) RequestContext() *a2asrv.RequestContext {
	v1Ctx := w.ExecutorContext.RequestContext()
	return toRequestContext(v1Ctx)
}

func toRequestContext(ctx *a2asrvv2.ExecutorContext) *a2asrv.RequestContext {
	var relatedTasks []*a2a.Task
	for _, t := range ctx.RelatedTasks {
		relatedTasks = append(relatedTasks, a2av0.FromV1Task(t))
	}

	return &a2asrv.RequestContext{
		ContextID:    ctx.ContextID,
		Message:      a2av0.FromV1Message(ctx.Message),
		StoredTask:   a2av0.FromV1Task(ctx.StoredTask),
		TaskID:       a2a.TaskID(ctx.TaskID),
		Metadata:     ctx.Metadata,
		RelatedTasks: relatedTasks,
	}
}

func toExecutorContext(ctx context.Context, reqCtx *a2asrv.RequestContext) (*a2asrvv2.ExecutorContext, error) {
	var user *a2asrvv2.User
	reqMeta := make(map[string][]string)
	if callCtx, ok := a2asrv.CallContextFrom(ctx); ok {
		user = &a2asrvv2.User{Name: callCtx.User.Name(), Authenticated: callCtx.User.Authenticated()}
		maps.Insert(reqMeta, callCtx.RequestMeta().List())
	}

	storedTask, err := a2av0.ToV1Task(reqCtx.StoredTask)
	if err != nil {
		return nil, err
	}

	var relatedTasks []*a2av2.Task
	for _, t := range reqCtx.RelatedTasks {
		v1Task, err := a2av0.ToV1Task(t)
		if err != nil {
			return nil, err
		}
		relatedTasks = append(relatedTasks, v1Task)
	}

	v1Msg, err := a2av0.ToV1Message(reqCtx.Message)
	if err != nil {
		return nil, err
	}

	return &a2asrvv2.ExecutorContext{
		ContextID:     reqCtx.ContextID,
		Message:       v1Msg,
		TaskID:        a2av2.TaskID(reqCtx.TaskID),
		StoredTask:    storedTask,
		RelatedTasks:  relatedTasks,
		Metadata:      reqCtx.Metadata,
		User:          user,
		ServiceParams: a2asrvv2.NewServiceParams(reqMeta),
	}, nil
}
