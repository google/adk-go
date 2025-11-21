// Copyright 2025 The A2A Authors
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

package a2asrv

import (
	"context"
	"fmt"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
	"github.com/a2aproject/a2a-go/internal/taskupdate"
)

// AgentExecutor implementations translate agent outputs to A2A events.
// The provided [RequestContext] should be used as a [a2a.TaskInfoProvider] argument for [a2a.Event]-s constructor functions.
// For streaming responses [a2a.TaskArtifactUpdatEvent]-s should be used.
// A2A server stops processing events after one of these events:
//   - An [a2a.Message] with any payload.
//   - An [a2a.TaskStatusUpdateEvent] with Final field set to true.
//   - An [a2a.Task] with a [a2a.TaskState] for which Terminal() method returns true.
//
// The following code can be used as a streaming implementation template with generateOutputs and toParts missing:
//
//	func Execute(ctx context.Context, reqCtx *RequestContext, queue eventqueue.Queue) error {
//		if reqCtx.StoredTask == nil {
//			event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateSubmitted, nil)
//			if err := queue.Write(ctx, event); err != nil {
//				return fmt.Errorf("failed to write state submitted: %w", err)
//			}
//		}
//
//		// perform setup
//
//		event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateWorking, nil)
//		if err := queue.Write(ctx, event); err != nil {
//			return fmt.Errorf("failed to write state working: %w", err)
//		}
//
//		var artifactID a2a.ArtifactID
//		for output, err := range generateOutputs() {
//			if err != nil {
//				event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateFailed, toErrorMessage(err))
//				if err := queue.Write(ctx, event); err != nil {
//					return fmt.Errorf("failed to write state failed: %w", err)
//				}
//			}
//
//			parts := toParts(output)
//			var event *a2a.TaskArtifactUpdateEvent
//			if artifactID == "" {
//				event = a2a.NewArtifactEvent(reqCtx, parts...)
//				artifactID = event.Artifact.ID
//			} else {
//				event = a2a.NewArtifactUpdateEvent(reqCtx, artifactID, parts...)
//			}
//
//			if err := queue.Write(ctx, event); err != nil {
//				return fmt.Errorf("failed to write artifact update: %w", err)
//			}
//		}
//
//		event = a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCompleted, nil)
//		event.Final = true
//		if err := queue.Write(ctx, event); err != nil {
//			return fmt.Errorf("failed to write state working: %w", err)
//		}
//
//		return nil
//	}
type AgentExecutor interface {
	// Execute invokes the agent passing information about the request which triggered execution,
	// translates agent outputs to A2A events and writes them to the event queue.
	// Every invocation runs in a dedicated goroutine.
	//
	// Failures should generally be reported by writing events carrying the cancelation information
	// and task state. An error should be returned in special cases like a failure to write an event.
	Execute(ctx context.Context, reqCtx *RequestContext, queue eventqueue.Queue) error

	// Cancel is called when a client requests the agent to stop working on a task.
	// The simplest implementation can write a cancelation event to the queue and let
	// it be processed by the A2A server. If the events gets applied during an active execution the execution
	// Context gets canceled.
	//
	// An an error should be returned if the cancelation request cannot be processed or a queue write failed.
	Cancel(ctx context.Context, reqCtx *RequestContext, queue eventqueue.Queue) error
}

type executor struct {
	*processor
	taskID          a2a.TaskID
	taskStore       TaskStore
	pushConfigStore PushConfigStore
	agent           AgentExecutor
	params          *a2a.MessageSendParams
	interceptors    []RequestContextInterceptor
}

func (e *executor) Execute(ctx context.Context, q eventqueue.Queue) error {
	reqCtx, task, err := e.loadExecRequestContext(ctx)
	if err != nil {
		return err
	}

	if e.params.Config != nil && e.params.Config.PushConfig != nil {
		if e.pushConfigStore == nil || e.pushSender == nil {
			return a2a.ErrPushNotificationNotSupported
		}
		if _, err := e.pushConfigStore.Save(ctx, e.taskID, e.params.Config.PushConfig); err != nil {
			return fmt.Errorf("failed to save %v: %w", e.params.Config.PushConfig, err)
		}
	}

	e.processor.init(taskupdate.NewManager(e.taskStore, task))

	for _, interceptor := range e.interceptors {
		ctx, err = interceptor.Intercept(ctx, reqCtx)
		if err != nil {
			return fmt.Errorf("interceptor failed: %w", err)
		}
	}

	return e.agent.Execute(ctx, reqCtx, q)
}

// loadExecRequestContext returns the RequestContext for AgentExecutor and a Task for initializing taskupdate.Manager with.
func (e *executor) loadExecRequestContext(ctx context.Context) (*RequestContext, *a2a.Task, error) {
	msg := e.params.Message

	if msg.TaskID == "" {
		contextID := msg.ContextID
		if contextID == "" {
			contextID = a2a.NewContextID()
		}
		reqCtx := &RequestContext{
			Message:   msg,
			TaskID:    e.taskID,
			ContextID: contextID,
			Metadata:  msg.Metadata,
		}
		return reqCtx, a2a.NewSubmittedTask(reqCtx, msg), nil
	}

	if msg.TaskID != e.taskID {
		return nil, nil, fmt.Errorf("bug: message task id different from executor task id")
	}

	storedTask, err := e.taskStore.Get(ctx, msg.TaskID)
	if err != nil {
		return nil, nil, fmt.Errorf("task loading failed: %w", err)
	}
	if storedTask == nil {
		return nil, nil, a2a.ErrTaskNotFound
	}

	if msg.ContextID != "" && msg.ContextID != storedTask.ContextID {
		return nil, nil, fmt.Errorf("message contextID different from task contextID: %w", a2a.ErrInvalidParams)
	}

	if storedTask.Status.State.Terminal() {
		return nil, nil, fmt.Errorf("task in a terminal state %q: %w", storedTask.Status.State, a2a.ErrInvalidParams)
	}

	storedTask.History = append(storedTask.History, msg)
	if err := e.taskStore.Save(ctx, storedTask); err != nil {
		return nil, nil, fmt.Errorf("task message history update failed: %w", err)
	}

	reqCtx := &RequestContext{
		Message:    msg,
		StoredTask: storedTask,
		TaskID:     storedTask.ID,
		ContextID:  storedTask.ContextID,
		Metadata:   msg.Metadata,
	}
	return reqCtx, storedTask, nil
}

type canceler struct {
	*processor
	agent        AgentExecutor
	taskStore    TaskStore
	params       *a2a.TaskIDParams
	interceptors []RequestContextInterceptor
}

func (c *canceler) Cancel(ctx context.Context, q eventqueue.Queue) error {
	task, err := c.taskStore.Get(ctx, c.params.ID)
	if err != nil {
		return fmt.Errorf("failed to load a task: %w", err)
	}
	c.processor.init(taskupdate.NewManager(c.taskStore, task))

	if task.Status.State == a2a.TaskStateCanceled {
		return q.Write(ctx, task)
	}

	if task.Status.State.Terminal() {
		return fmt.Errorf("task in non-cancelable state %s: %w", task.Status.State, a2a.ErrTaskNotCancelable)
	}

	reqCtx := &RequestContext{
		TaskID:     task.ID,
		StoredTask: task,
		ContextID:  task.ContextID,
		Metadata:   c.params.Metadata,
	}

	for _, interceptor := range c.interceptors {
		ctx, err = interceptor.Intercept(ctx, reqCtx)
		if err != nil {
			return fmt.Errorf("interceptor failed: %w", err)
		}
	}

	return c.agent.Cancel(ctx, reqCtx, q)
}

type processor struct {
	// Processor is running in event consumer goroutine, but request context loading
	// happens in event consumer goroutine. Once request context is loaded and validate the processor
	// gets initialized.
	updateManager   *taskupdate.Manager
	pushConfigStore PushConfigStore
	pushSender      PushSender
}

func newProcessor(store PushConfigStore, sender PushSender) *processor {
	return &processor{
		pushConfigStore: store,
		pushSender:      sender,
	}
}

func (p *processor) init(um *taskupdate.Manager) {
	p.updateManager = um
}

// Process implements taskexec.Processor interface.
// A (nil, nil) result means the processing should continue.
// A non-nill result becomes the result of the execution.
func (p *processor) Process(ctx context.Context, event a2a.Event) (*a2a.SendMessageResult, error) {
	// TODO(yarolegovich): handle invalid event sequence where a Message is produced after a Task was created
	if msg, ok := event.(*a2a.Message); ok {
		var result a2a.SendMessageResult = msg
		return &result, nil
	}

	task, err := p.updateManager.Process(ctx, event)
	if err != nil {
		p.updateManager.SetTaskFailed(ctx, err)
		return nil, err
	}

	if err := p.sendPushNotifications(ctx, task); err != nil {
		p.updateManager.SetTaskFailed(ctx, err)
		return nil, err
	}

	if _, ok := event.(*a2a.TaskArtifactUpdateEvent); ok {
		return nil, nil
	}

	if statusUpdate, ok := event.(*a2a.TaskStatusUpdateEvent); ok {
		if statusUpdate.Final {
			var result a2a.SendMessageResult = task
			return &result, nil
		}
		return nil, nil
	}

	if task.Status.State == a2a.TaskStateUnknown {
		return nil, fmt.Errorf("unknown task state: %s", task.Status.State)
	}

	if task.Status.State.Terminal() || task.Status.State == a2a.TaskStateInputRequired {
		var result a2a.SendMessageResult = task
		return &result, nil
	}

	return nil, nil
}

func (p *processor) sendPushNotifications(ctx context.Context, task *a2a.Task) error {
	if p.pushSender == nil || p.pushConfigStore == nil {
		return nil
	}

	configs, err := p.pushConfigStore.List(ctx, task.ID)
	if err != nil {
		return err
	}

	// TODO(yarolegovich): consider dispatching in parallel with max concurrent calls cap
	for _, config := range configs {
		if err := p.pushSender.SendPush(ctx, config, task); err != nil {
			return err
		}
	}
	return nil
}
