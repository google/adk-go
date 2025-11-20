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

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/log"
)

// RequestContextInterceptor defines an extension point for modifying request contexts
// that contain the information needed by [AgentExecutor] implementations to process incoming requests.
type RequestContextInterceptor interface {
	// Intercept has a chance to modify a RequestContext before it gets passed to AgentExecutor.
	Intercept(ctx context.Context, reqCtx *RequestContext) (context.Context, error)
}

// RequestContext provides information about an incoming A2A request to [AgentExecutor].
type RequestContext struct {
	// A message which triggered the execution. nil for cancelation request.
	Message *a2a.Message
	// TaskID is an ID of the task or a newly generated UUIDv4 in case Message did not reference any Task.
	TaskID a2a.TaskID
	// StoredTask is present if request message specified a TaskID.
	StoredTask *a2a.Task
	// RelatedTasks can be present when Message includes Task references and RequestContextBuilder is configured to load them.
	RelatedTasks []*a2a.Task
	// ContextID is a server-generated identifier for maintaining context across multiple related tasks or interactions. Matches the Task ContextID.
	ContextID string
	// Metadata of the request which triggered the call.
	Metadata map[string]any
}

var _ a2a.TaskInfoProvider = (*RequestContext)(nil)

func (rc *RequestContext) TaskInfo() a2a.TaskInfo {
	return a2a.TaskInfo{TaskID: rc.TaskID, ContextID: rc.ContextID}
}

// ReferencedTasksLoader implements RequestContextInterceptor. It populates RelatedTasks field of RequestContext
// with Tasks referenced in the ReferenceTasks field of the Message which triggered the agent execution.
type ReferencedTasksLoader struct {
	Store TaskStore
}

func (ri *ReferencedTasksLoader) Intercept(ctx context.Context, reqCtx *RequestContext) (context.Context, error) {
	msg := reqCtx.Message
	if msg == nil {
		return ctx, nil
	}

	if len(msg.ReferenceTasks) == 0 {
		return ctx, nil
	}

	tasks := make([]*a2a.Task, 0, len(msg.ReferenceTasks))
	for _, taskID := range msg.ReferenceTasks {
		task, err := ri.Store.Get(ctx, taskID)
		if err != nil {
			log.Info(ctx, "failed to get a referenced task", "referenced_task_id", taskID)
			continue
		}
		tasks = append(tasks, task)
	}

	if len(tasks) > 0 {
		reqCtx.RelatedTasks = tasks
	}

	return ctx, nil
}
