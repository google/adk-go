// Copyright 2026 Google LLC
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

// Package acpagent runs a coding agent harness speaking the Agent Client
// Protocol (https://agentclientprotocol.com), such as `opencode acp`,
// `gemini --experimental-acp` or Zed's claude-agent-acp, as an ADK agent:
//
//	harness, closer, err := acpagent.New(ctx, acpagent.Config{
//		Name:        "coder",
//		Description: "Edits code in the current repository.",
//		Command:     []string{"opencode", "acp"},
//		AllowTools:  []string{"read", "search", "edit"},
//	})
//	if err != nil { ... }
//	defer closer.Close()
//
// The harness runs as a child process and ADK talks to it over its stdio. One
// process serves every ADK session; each ADK session maps to its own ACP
// session, so the harness keeps its own conversation history and the agent
// sends it only the messages it has not seen yet.
//
// The harness's reply streams into the session as the agent's events. Tool
// calls the harness makes on its own are not surfaced as ADK function calls.
//
// # Permissions
//
// Harnesses ask before running some tools. The agent answers those requests
// itself from [Config.AllowTools] and [Config.DenyTools]. Any other request,
// including a form the harness asks the user to fill in, ends the invocation
// with the question, and the user's next message to the agent answers it: an
// option such as "once" or "reject", "yes" or "no", or for a form a JSON
// object of its fields. A message which is not an answer interrupts the
// harness and becomes its next instruction. The question is held by the
// harness process, so it can be answered only while that process runs.
package acpagent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"slices"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aevent"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/a2aproject/a2a-go/v2/a2asrv/taskstore"
	"github.com/a2aproject/a2a-go/v2/integrations/executor/a2aacp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/remoteagent/v2"
	"google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"
)

// Config configures an ACP harness agent.
type Config struct {
	Name        string
	Description string

	// Command is the harness program and its arguments, for example
	// {"opencode", "acp"}. The program is looked up in PATH if it has no path
	// separators. Either Command or Start must be set.
	Command []string
	// Dir is the working directory of the harness and of its sessions.
	// Defaults to the current directory.
	Dir string
	// Env is added to the environment of the current process.
	Env []string
	// Stderr receives the harness's stderr. Discarded if nil.
	Stderr io.Writer
	// Start starts the harness instead of Command, for example in a sandbox,
	// and returns its stdio: reads come from the harness's output, writes go
	// to its input, and Close stops it. The harness must outlive ctx.
	Start func(ctx context.Context) (io.ReadWriteCloser, error)

	// AllowTools lists the ACP tool kinds the harness may run when it asks for
	// permission: read, edit, delete, move, search, execute, think, fetch or
	// other. In patterns, * matches any run of characters. DenyTools takes
	// precedence. Requests for any other tool are asked of the user.
	AllowTools []string
	DenyTools  []string

	// BeforeAgentCallbacks are called sequentially before the agent starts its
	// run. See [agent.BeforeAgentCallback] for how return values affect execution.
	BeforeAgentCallbacks []agent.BeforeAgentCallback
	// AfterAgentCallbacks are called sequentially after the agent completes its
	// run. See [agent.AfterAgentCallback] for when they are skipped and how return
	// values affect execution.
	AfterAgentCallbacks []agent.AfterAgentCallback
}

// New starts the harness and returns an agent backed by it. Close the returned
// closer to stop the harness once the agent is no longer in use; runs which are
// still in progress then fail.
func New(ctx context.Context, cfg Config) (agent.Agent, io.Closer, error) {
	var starter a2aacp.Agent
	switch {
	case cfg.Start != nil && len(cfg.Command) > 0:
		return nil, nil, errors.New("acpagent: set either Config.Command or Config.Start, not both")
	case cfg.Start != nil:
		starter = a2aacp.AgentFunc(func(ctx context.Context, _ string) (io.ReadWriteCloser, error) {
			return cfg.Start(ctx)
		})
	case len(cfg.Command) > 0:
		starter = a2aacp.Process{Argv: cfg.Command, Dir: cfg.Dir, Env: cfg.Env, Stderr: cfg.Stderr}
	default:
		return nil, nil, errors.New("acpagent: either Config.Command or Config.Start is required")
	}

	var sessionCfg func(context.Context, *a2asrv.ExecutorContext) (a2aacp.SessionParams, error)
	if cfg.Start != nil && cfg.Dir != "" {
		// For a Process, Dir already is the default session directory.
		sessionCfg = func(context.Context, *a2asrv.ExecutorContext) (a2aacp.SessionParams, error) {
			return a2aacp.SessionParams{Cwd: cfg.Dir}, nil
		}
	}

	exec, err := a2aacp.NewExecutor(ctx, starter, &a2aacp.Config{
		AutoAnswer: a2aacp.ToolRules{Allow: cfg.AllowTools, Deny: cfg.DenyTools},
		Session:    sessionCfg,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("acpagent: starting harness: %w", err)
	}

	// The harness is reached through remoteagent, which already maps ADK
	// sessions to A2A contexts, replays the history the peer has not seen and
	// converts the reply into events. The A2A request handler is called
	// directly, so no transport or server is involved.
	ag, err := remoteagent.NewA2A(remoteagent.A2AConfig{
		Name:        cfg.Name,
		Description: cfg.Description,
		AgentCard: &a2a.AgentCard{
			Name:         cfg.Name,
			Description:  cfg.Description,
			Capabilities: a2a.AgentCapabilities{Streaming: true},
		},
		ClientProvider: func(ctx context.Context, _ *a2a.AgentCard) (remoteagent.A2AClient, error) {
			return newHandlerClient(ctx, exec)
		},
		BeforeAgentCallbacks: cfg.BeforeAgentCallbacks,
		AfterAgentCallbacks:  cfg.AfterAgentCallbacks,
	})
	if err != nil {
		return nil, nil, errors.Join(err, exec.Close())
	}
	return ag, exec, nil
}

// responseKey is the custom metadata key under which remoteagent records the
// A2A event a session event was converted from.
var responseKey = adka2a.ToADKMetaKey("response")

// responseState returns the task state in the A2A event ev was converted from,
// or "" if ev records none.
func responseState(ev *session.Event) a2a.TaskState {
	resp, _ := ev.CustomMetadata[responseKey].(map[string]any)
	status, _ := resp["status"].(map[string]any)
	state, _ := status["state"].(string)
	return a2a.TaskState(state)
}

// pendingTask returns the task the agent's last turn in the invocation's
// isolation scope left waiting for an answer, or nil if there is none.
func pendingTask(ctx agent.InvocationContext) *a2a.Task {
	events := ctx.Session().Events()
	name := ctx.Agent().Name()
	inLastTurn := false
	for i := events.Len() - 1; i >= 0; i-- {
		ev := events.At(i)
		if ev.IsolationScope != ctx.IsolationScope() {
			continue
		}
		if ev.Author != name {
			if inLastTurn {
				break
			}
			continue
		}
		inLastTurn = true
		if responseState(ev) != a2a.TaskStateInputRequired {
			continue
		}
		taskID, contextID := adka2a.GetA2ATaskInfo(ev)
		if taskID == "" {
			return nil
		}
		return &a2a.Task{
			ID:        taskID,
			ContextID: contextID,
			Status:    a2a.TaskStatus{State: a2a.TaskStateInputRequired},
		}
	}
	return nil
}

// handlerClient sends requests straight to an in-process request handler.
type handlerClient struct {
	handler a2asrv.RequestHandler
	// pending is the ID of the task waiting for an answer when the invocation
	// started, or "".
	pending a2a.TaskID
}

// newHandlerClient returns a client with a handler and task store of its own.
// The store starts with the task waiting for an answer, if there is one.
//
// This is done here rather than in a request callback because a callback
// context does not expose the session.
func newHandlerClient(ctx context.Context, exec a2asrv.AgentExecutor) (*handlerClient, error) {
	c := &handlerClient{}
	store := taskstore.NewInMemory(nil)
	// The context is the invocation's, except when an adka2a server cancels
	// an abandoned task, which needs no pending task.
	if ic, ok := ctx.(agent.InvocationContext); ok {
		if t := pendingTask(ic); t != nil {
			if _, err := store.Create(ctx, t); err != nil {
				return nil, fmt.Errorf("restoring task %s: %w", t.ID, err)
			}
			c.pending = t.ID
		}
	}
	c.handler = a2asrv.NewHandler(exec,
		a2asrv.WithTaskStore(store),
		a2asrv.WithUpdateMaterializer(resumeMaterializer),
	)
	return c, nil
}

// route sends the user's message to the task waiting for an answer, if there
// is one. remoteagent continues a task only with a function response, while
// the harness takes a typed answer.
func (c *handlerClient) route(req *a2a.SendMessageRequest) *a2a.SendMessageRequest {
	if c.pending == "" || req.Message == nil || req.Message.TaskID != "" {
		return req
	}
	msg := *req.Message
	msg.TaskID = c.pending
	routed := *req
	routed.Message = &msg
	return &routed
}

func (c *handlerClient) SendMessage(ctx context.Context, req *a2a.SendMessageRequest) (a2a.SendMessageResult, error) {
	return c.handler.SendMessage(ctx, c.route(req))
}

func (c *handlerClient) SendStreamingMessage(ctx context.Context, req *a2a.SendMessageRequest) iter.Seq2[a2a.Event, error] {
	return c.handler.SendStreamingMessage(ctx, c.route(req))
}

func (c *handlerClient) CancelTask(ctx context.Context, req *a2a.CancelTaskRequest) (*a2a.Task, error) {
	return c.handler.CancelTask(ctx, req)
}

// Destroy is a no-op: the handler starts no work of its own, and the
// executor behind it is stopped by the closer [New] returns.
func (*handlerClient) Destroy() error { return nil }

// resumeMaterializer is the handler's default materializer, except that an
// append to an artifact the task lacks creates the artifact. The executor
// keeps streaming a resumed task's reply into the artifact it created before
// the question, which only the previous invocation's store had. The text
// streamed before the question was already delivered with that invocation.
var resumeMaterializer = taskstore.NewFullUpdateMaterializer(func(task *a2a.Task, ev a2a.Event) (*a2a.Task, error) {
	if au, ok := ev.(*a2a.TaskArtifactUpdateEvent); ok && au.Append && task != nil &&
		!slices.ContainsFunc(task.Artifacts, func(a *a2a.Artifact) bool { return a.ID == au.Artifact.ID }) {
		created := *au
		created.Append = false
		ev = &created
	}
	return a2aevent.ApplyShallowUpdate(task, ev)
})
