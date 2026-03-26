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
	"errors"
	"fmt"
	"iter"
	"slices"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2aclient"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
	"github.com/a2aproject/a2a-go/log"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	iremoteagent "google.golang.org/adk/internal/agent/remoteagent"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
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

// A2AExecutionCleanupCallback is a callback which will be called after an execution or cancellatio has completed or failed.
type A2AExecutionCleanupCallback func(ctx context.Context, reqCtx *a2asrv.RequestContext, subAgentCards []*a2a.AgentCard, result a2a.SendMessageResult, cause error)

// OutputMode controls how artifacts are produced.
type OutputMode string

// Runner is an interface matching [runner.Runner] API.
// It exists to let users use custom runner implementations with A2A agent executor.
type Runner interface {
	// Run runs the agent for the given user input, yielding events from agents.
	Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg agent.RunConfig) iter.Seq2[*session.Event, error]
}

// RunnerProvider is a [Runner] factory function. The provided plugin must be installed in the returned [Runner] for
// callbacks taking [ExecutorContext] to work correctly.
type RunnerProvider func(ctx context.Context, reqCtx *a2asrv.RequestContext, plugin *plugin.Plugin) (RunnerConfig, Runner, error)

const (
	// OutputArtifactPerRun produces a single artifact per [runner.Runner.Run].
	OutputArtifactPerRun OutputMode = "artifact-per-run"
	// OutputArtifactPerEvent produces an artifact per non-partial [session.Event].
	// While agent is emitting events an artifact is build incrementally (parts are append to it).
	// The next partial event replaces accumulated contents and seals the artifact, meaning
	// the next event from this agent will create a new artifact.
	OutputArtifactPerEvent OutputMode = "artifact-per-event"
)

// RunnerConfig is part of the runner configuration executor code depends on.
// Custom [RunnerProvider] needs to return it back to callers.
type RunnerConfig struct {
	// AppName is the name of the application used in [session.Service] keys and A2A event metadata.
	AppName string
	// Agent is the root agent. It isued
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
	if config.RunnerProvider == nil {
		config.RunnerProvider = newDefaultRunnerProvider(config.RunnerConfig)
	}
	return &Executor{config: config}
}

func (e *Executor) Execute(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	msg := reqCtx.Message
	if msg == nil {
		return fmt.Errorf("message not provided")
	}
	content, err := toGenAIContent(ctx, msg, e.config.A2APartConverter)
	if err != nil {
		return fmt.Errorf("a2a message conversion failed: %w", err)
	}

	executorPlugin, err := newExecutorPlugin()
	if err != nil {
		return fmt.Errorf("failed to create a2a-executor plugin: %w", err)
	}

	cfg, r, err := e.config.RunnerProvider(ctx, reqCtx, executorPlugin.plugin)
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

	invocationMeta := toInvocationMeta(ctx, cfg, reqCtx)

	err = e.prepareSession(ctx, cfg, invocationMeta)
	if err != nil {
		event := toTaskFailedUpdateEvent(reqCtx, err, invocationMeta.eventMeta)
		execCtx := newExecutorContext(ctx, invocationMeta, executorPlugin, content)
		return e.writeFinalTaskStatus(execCtx, queue, nil, event, err)
	}

	event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateWorking, nil)
	event.Metadata = invocationMeta.eventMeta
	if err := queue.Write(ctx, event); err != nil {
		return err
	}

	var artifactTransform eventToArtifactTransform
	if e.config.OutputMode == OutputArtifactPerEvent {
		artifactTransform = newArtifactMaker(reqCtx)
	} else {
		artifactTransform = newLegacyArtifactMaker(reqCtx)
	}

	processor := newEventProcessor(reqCtx, invocationMeta, e.config.GenAIPartConverter, artifactTransform)
	executorContext := newExecutorContext(ctx, invocationMeta, executorPlugin, content)
	return e.process(executorContext, r, processor, queue)
}

func (e *Executor) Cancel(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCanceled, nil)
	event.Final = true
	return queue.Write(ctx, event)
}

func (e *Executor) Cleanup(ctx context.Context, reqCtx *a2asrv.RequestContext, result a2a.SendMessageResult, cause error) {
	cfg, err := e.createRunnerConfig(ctx, reqCtx)
	if err != nil {
		log.Error(ctx, "failed to create runner config", err)
		return
	}

	remoteSubagents := findRemoteSubagents(cfg.Agent)

	// If task was in input-required and got successfully cancelled - run the cleanup logic
	if reqCtx.StoredTask != nil && reqCtx.StoredTask.Status.State == a2a.TaskStateInputRequired {
		if task, ok := result.(*a2a.Task); ok && task.Status.State == a2a.TaskStateCanceled && reqCtx.Message == nil {
			if err := e.cancelChildInputRequiredTasks(ctx, reqCtx, reqCtx.StoredTask.Status, remoteSubagents); err != nil {
				log.Warn(ctx, "failed to cancel subagent tasks waiting for input", "cause", err)
			}
		}
	}

	if e.config.A2AExecutionCleanupCallback != nil {
		subAgentCards := make([]*a2a.AgentCard, len(remoteSubagents))
		for i, subagent := range remoteSubagents {
			subAgentCards[i] = subagent.config.AgentCard
		}
		e.config.A2AExecutionCleanupCallback(ctx, reqCtx, subAgentCards, result, cause)
	} else if cause != nil {
		if reqCtx.Message != nil {
			log.Warn(ctx, "execution failed", "error", cause)
		} else {
			log.Warn(ctx, "cancellation failed", "error", cause)
		}
	}
}

func (e *Executor) cancelChildInputRequiredTasks(ctx context.Context, reqCtx *a2asrv.RequestContext, status a2a.TaskStatus, subagents []remoteAgent) error {
	if len(subagents) == 0 {
		return nil
	}

	cfg, err := e.createRunnerConfig(ctx, reqCtx)
	if err != nil {
		return fmt.Errorf("failed to create runner config: %w", err)
	}

	meta := toInvocationMeta(ctx, cfg, reqCtx)
	getSessionResponse, err := cfg.SessionService.Get(ctx, &session.GetRequest{
		AppName:   cfg.AppName,
		UserID:    meta.userID,
		SessionID: meta.sessionID,
	})
	if err != nil {
		return fmt.Errorf("failed to get a session: %w", err)
	}

	tasksToCancel, err := getSubagentTasksToCancel(ctx, status, getSessionResponse.Session)
	if err != nil {
		return fmt.Errorf("subtask search failed: %w", err)
	}
	if len(tasksToCancel) == 0 {
		return nil
	}

	var failures []error
	clientCache := map[string]*a2aclient.Client{}
	for _, task := range tasksToCancel { // TODO(yarolegovich): run in parallel (how to limit?)
		remoteSubagentIdx := slices.IndexFunc(subagents, func(a remoteAgent) bool { return a.agent.Name() == task.agentName })
		if remoteSubagentIdx < 0 {
			continue
		}
		remoteSubagent := subagents[remoteSubagentIdx]
		client, ok := clientCache[task.agentName]
		if !ok {
			_, newClient, err := iremoteagent.CreateA2AClient(ctx, remoteSubagent.config)
			if err != nil {
				failures = append(failures, fmt.Errorf("failed to create A2A client: %w", err))
				continue
			}
			clientCache[task.agentName] = newClient
			client = newClient
		}
		_, err = client.CancelTask(ctx, &a2a.TaskIDParams{ID: task.taskID})
		if err != nil {
			failures = append(failures, fmt.Errorf("failed to cancel task: %w", err))
			continue
		}
	}
	for _, client := range clientCache {
		if err := client.Destroy(); err != nil {
			failures = append(failures, fmt.Errorf("client destroy failed: %w", err))
		}
	}
	return errors.Join(failures...)
}

// Processing failures should be delivered as Task failed events. An error is returned from this method if an event write fails.
func (e *Executor) process(ctx ExecutorContext, r Runner, processor *eventProcessor, q eventqueue.Queue) error {
	meta := processor.meta
	for adkEvent, adkErr := range r.Run(ctx, meta.userID, meta.sessionID, ctx.UserContent(), e.config.RunConfig) {
		if adkErr != nil {
			event := processor.makeTaskFailedEvent(fmt.Errorf("agent run failed: %w", adkErr), nil)
			return e.writeFinalTaskStatus(ctx, q, processor.makeFinalArtifactUpdate(), event, adkErr)
		}

		a2aEvent, pErr := processor.process(ctx, adkEvent)
		if pErr == nil && a2aEvent != nil && e.config.AfterEventCallback != nil {
			pErr = e.config.AfterEventCallback(ctx, adkEvent, a2aEvent)
		}

		if pErr != nil {
			event := processor.makeTaskFailedEvent(fmt.Errorf("processor failed: %w", pErr), adkEvent)
			return e.writeFinalTaskStatus(ctx, q, processor.makeFinalArtifactUpdate(), event, pErr)
		}

		if a2aEvent != nil {
			if err := q.Write(ctx, a2aEvent); err != nil {
				return fmt.Errorf("event write failed: %w", err)
			}
		}
	}

	finalStatus := processor.makeFinalStatusUpdate()
	return e.writeFinalTaskStatus(ctx, q, processor.makeFinalArtifactUpdate(), finalStatus, nil)
}

func (e *Executor) writeFinalTaskStatus(
	ctx ExecutorContext,
	queue eventqueue.Queue,
	partialReset *a2a.TaskArtifactUpdateEvent,
	status *a2a.TaskStatusUpdateEvent,
	err error,
) error {
	if e.config.AfterExecuteCallback != nil {
		if err = e.config.AfterExecuteCallback(ctx, status, err); err != nil {
			return fmt.Errorf("after execute: %w", err)
		}
	}
	if partialReset != nil {
		if err := queue.Write(ctx, partialReset); err != nil {
			return fmt.Errorf("partial artifact update write failed: %w", err)
		}
	}
	if err := queue.Write(ctx, status); err != nil {
		return fmt.Errorf("%q state update event write failed: %w", status.Status.State, err)
	}
	return nil
}

func (e *Executor) prepareSession(ctx context.Context, cfg RunnerConfig, meta invocationMeta) error {
	service := cfg.SessionService

	_, err := service.Get(ctx, &session.GetRequest{
		AppName:   cfg.AppName,
		UserID:    meta.userID,
		SessionID: meta.sessionID,
	})
	if err == nil {
		return nil
	}

	_, err = service.Create(ctx, &session.CreateRequest{
		AppName:   cfg.AppName,
		UserID:    meta.userID,
		SessionID: meta.sessionID,
		State:     make(map[string]any),
	})
	if err != nil {
		return fmt.Errorf("failed to create a session: %w", err)
	}

	return nil
}

func (e *Executor) createRunnerConfig(ctx context.Context, reqCtx *a2asrv.RequestContext) (RunnerConfig, error) {
	executorPlugin, err := newExecutorPlugin()
	if err != nil {
		return RunnerConfig{}, fmt.Errorf("failed to create a2a-plugin: %w", err)
	}
	cfg, _, err := e.config.RunnerProvider(ctx, reqCtx, executorPlugin.plugin)
	if err != nil {
		return RunnerConfig{}, fmt.Errorf("runner provider failed: %w", err)
	}
	return cfg, nil
}

func newDefaultRunnerProvider(baseConfig runner.Config) RunnerProvider {
	return func(ctx context.Context, reqCtx *a2asrv.RequestContext, plugin *plugin.Plugin) (RunnerConfig, Runner, error) {
		if baseConfig.Agent == nil {
			return RunnerConfig{}, nil, fmt.Errorf("runner.Config.Agent is not provided")
		}
		if baseConfig.Agent == nil {
			return RunnerConfig{}, nil, fmt.Errorf("runner.Config.SessionService is not provided")
		}

		cfg := baseConfig
		cfg.PluginConfig.Plugins = append(slices.Clone(cfg.PluginConfig.Plugins), plugin)
		r, err := runner.New(cfg)
		if err != nil {
			return RunnerConfig{}, nil, err
		}
		return toInternalRunnerConfig(cfg), &defaultRunner{runner: r}, nil
	}
}

type defaultRunner struct {
	runner *runner.Runner
}

func (r *defaultRunner) Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg agent.RunConfig) iter.Seq2[*session.Event, error] {
	return r.runner.Run(ctx, userID, sessionID, msg, cfg)
}

func toInternalRunnerConfig(cfg runner.Config) RunnerConfig {
	return RunnerConfig{Agent: cfg.Agent, AppName: cfg.AppName, SessionService: cfg.SessionService}
}
