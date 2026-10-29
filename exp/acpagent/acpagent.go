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
// itself from [Config.AllowTools] and [Config.DenyTools] and never pauses the
// invocation: a tool which is not allowed is rejected, and form elicitations
// are declined.
package acpagent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/a2aproject/a2a-go/v2/integrations/executor/a2aacp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/remoteagent/v2"
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
	// precedence. Requests for any other tool are rejected.
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
		AutoAnswer: permissionPolicy{rules: a2aacp.ToolRules{Allow: cfg.AllowTools, Deny: cfg.DenyTools}},
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
		ClientProvider: func(context.Context, *a2a.AgentCard) (remoteagent.A2AClient, error) {
			return handlerClient{a2asrv.NewHandler(exec)}, nil
		},
		BeforeAgentCallbacks: cfg.BeforeAgentCallbacks,
		AfterAgentCallbacks:  cfg.AfterAgentCallbacks,
	})
	if err != nil {
		return nil, nil, errors.Join(err, exec.Close())
	}
	return ag, exec, nil
}

// handlerClient sends requests straight to an in-process request handler.
type handlerClient struct {
	a2asrv.RequestHandler
}

// Destroy is a no-op: the handler starts no work of its own, and the
// executor behind it is stopped by the closer [New] returns.
func (handlerClient) Destroy() error { return nil }

// permissionPolicy answers every question the harness asks, so that an
// invocation never waits on input-required: remoteagent only resumes such a
// task with a function response, which the harness would not understand.
type permissionPolicy struct {
	rules a2aacp.ToolRules
}

func (p permissionPolicy) AutoAnswer(ctx context.Context, execCtx *a2asrv.ExecutorContext, q a2aacp.Question) (a2aacp.Answer, bool) {
	if ans, ok := p.rules.AutoAnswer(ctx, execCtx, q); ok {
		return ans, true
	}
	for _, opt := range []string{a2aacp.OptionReject, a2aacp.OptionRejectAlways, a2aacp.OptionDecline} {
		if slices.Contains(q.Options, opt) {
			return a2aacp.Answer{Option: opt}, true
		}
	}
	return a2aacp.Answer{Option: a2aacp.OptionCancel}, true
}
