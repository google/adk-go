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

// Package adka2a allows to expose ADK agents via A2A.
package adka2a

import (
	"context"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
	v2a2a "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2acompat/a2av0"
	v2asrv "github.com/a2aproject/a2a-go/v2/a2asrv"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	v1 "google.golang.org/adk/server/adka2a/v1"
	"google.golang.org/adk/session"
)

// BeforeExecuteCallback is the callback which will be called before an execution is started.
type BeforeExecuteCallback func(ctx context.Context, reqCtx *a2asrv.RequestContext) (context.Context, error)

// AfterEventCallback is the callback which will be called after an ADK event is converted to an A2A event.
type AfterEventCallback func(ctx ExecutorContext, event *session.Event, processed *a2a.TaskArtifactUpdateEvent) error

// AfterExecuteCallback is the callback which will be called after an execution resolved into a completed or failed task.
type AfterExecuteCallback func(ctx ExecutorContext, finalEvent *a2a.TaskStatusUpdateEvent, err error) error

// A2APartConverter is a custom converter for converting A2A parts to GenAI parts.
type A2APartConverter func(ctx context.Context, a2aEvent a2a.Event, part a2a.Part) (*genai.Part, error)

// GenAIPartConverter is a custom converter for converting GenAI parts to A2A parts.
type GenAIPartConverter func(ctx context.Context, adkEvent *session.Event, part *genai.Part) (a2a.Part, error)

// OutputMode controls how artifacts are produced.
type OutputMode string

const (
	OutputArtifactPerRun   OutputMode = "artifact-per-run"
	OutputArtifactPerEvent OutputMode = "artifact-per-event"
)

// ExecutorConfig allows to configure Executor.
type ExecutorConfig struct {
	RunnerConfig          runner.Config
	RunConfig             agent.RunConfig
	BeforeExecuteCallback BeforeExecuteCallback
	AfterEventCallback    AfterEventCallback
	AfterExecuteCallback  AfterExecuteCallback
	A2APartConverter      A2APartConverter
	GenAIPartConverter    GenAIPartConverter
	OutputMode            OutputMode
}

var _ a2asrv.AgentExecutor = (*Executor)(nil)

// Executor invokes an ADK agent and translates [session.Event]-s to [a2a.Event]-s.
type Executor struct {
	impl *v1.Executor
}

// NewExecutor creates an initialized [Executor] instance.
func NewExecutor(config ExecutorConfig) *Executor {
	v1Config := v1.ExecutorConfig{
		RunnerConfig: config.RunnerConfig,
		RunConfig:    config.RunConfig,
		OutputMode:   v1.OutputMode(config.OutputMode),
	}

	if config.BeforeExecuteCallback != nil {
		v1Config.BeforeExecuteCallback = func(ctx context.Context, reqCtx *v2asrv.ExecutorContext) (context.Context, error) {
			// RequestContext conversion might be tricky as there is no direct ToV0RequestContext in a2av0 (it's usually the other way)
			// But for now let's hope it's not strictly needed or provide a placeholder.
			return config.BeforeExecuteCallback(ctx, &a2asrv.RequestContext{
				ContextID:  reqCtx.ContextID,
				Message:    a2av0.FromV1Message(reqCtx.Message),
				StoredTask: a2av0.FromV1Task(reqCtx.StoredTask),
			})
		}
	}

	if config.AfterEventCallback != nil {
		v1Config.AfterEventCallback = func(ctx v1.ExecutorContext, adkEvent *session.Event, a2aEvent *v2a2a.TaskArtifactUpdateEvent) error {
			return config.AfterEventCallback(executorContextWrapper{ctx}, adkEvent, a2av0.FromV1TaskArtifactUpdateEvent(a2aEvent))
		}
	}

	if config.AfterExecuteCallback != nil {
		v1Config.AfterExecuteCallback = func(ctx v1.ExecutorContext, finalEvent *v2a2a.TaskStatusUpdateEvent, err error) error {
			return config.AfterExecuteCallback(executorContextWrapper{ctx}, a2av0.FromV1TaskStatusUpdateEvent(finalEvent), err)
		}
	}

	if config.A2APartConverter != nil {
		v1Config.A2APartConverter = func(ctx context.Context, a2aEvent v2a2a.Event, part *v2a2a.Part) (*genai.Part, error) {
			legacyEvent, _ := a2av0.FromV1Event(a2aEvent)
			return config.A2APartConverter(ctx, legacyEvent, a2av0.FromV1Part(part))
		}
	}

	if config.GenAIPartConverter != nil {
		v1Config.GenAIPartConverter = func(ctx context.Context, adkEvent *session.Event, part *genai.Part) (*v2a2a.Part, error) {
			legacyPart, err := config.GenAIPartConverter(ctx, adkEvent, part)
			if err != nil {
				return nil, err
			}
			return a2av0.ToV1Part(legacyPart), nil
		}
	}

	return &Executor{impl: v1.NewExecutor(v1Config)}
}

func (e *Executor) Execute(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	var user *v2asrv.User
	var reqMeta map[string][]string
	if callCtx, ok := a2asrv.CallContextFrom(ctx); ok {
		user = &v2asrv.User{
			Name:          callCtx.User.Name(),
			Authenticated: callCtx.User.Authenticated(),
			Attributes:    map[string]any{"legacy_user": user},
		}
		for k, v := range callCtx.RequestMeta().List() {
			reqMeta[k] = v
		}
	}

	storedTask, err := a2av0.ToV1Task(reqCtx.StoredTask)
	if err != nil {
		return err
	}

	var relatedTasks []*v2a2a.Task
	for _, t := range reqCtx.RelatedTasks {
		v1Task, err := a2av0.ToV1Task(t)
		if err != nil {
			continue
		}
		relatedTasks = append(relatedTasks, v1Task)
	}

	v1Msg, _ := a2av0.ToV1Message(reqCtx.Message)
	v2ReqCtx := &v2asrv.ExecutorContext{
		ContextID:     reqCtx.ContextID,
		Message:       v1Msg,
		TaskID:        v2a2a.TaskID(reqCtx.TaskID),
		StoredTask:    storedTask,
		RelatedTasks:  relatedTasks,
		Metadata:      reqCtx.Metadata,
		User:          user,
		ServiceParams: v2asrv.NewServiceParams(reqMeta),
	}
	if reqCtx.StoredTask != nil {
		v1Task, _ := a2av0.ToV1Task(reqCtx.StoredTask)
		v2ReqCtx.StoredTask = v1Task
		v2ReqCtx.TaskID = v1Task.ID
	}

	for event, err := range e.impl.Execute(ctx, v2ReqCtx) {
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
	v2ReqCtx := &v2asrv.ExecutorContext{
		ContextID: reqCtx.ContextID,
	}
	if reqCtx.StoredTask != nil {
		v1Task, _ := a2av0.ToV1Task(reqCtx.StoredTask)
		v2ReqCtx.StoredTask = v1Task
		v2ReqCtx.TaskID = v1Task.ID
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
	v1.ExecutorContext
}

func (w executorContextWrapper) RequestContext() *a2asrv.RequestContext {
	v1Ctx := w.ExecutorContext.RequestContext()
	return &a2asrv.RequestContext{
		ContextID:  v1Ctx.ContextID,
		Message:    a2av0.FromV1Message(v1Ctx.Message),
		StoredTask: a2av0.FromV1Task(v1Ctx.StoredTask),
	}
}
