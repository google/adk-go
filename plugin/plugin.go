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

// Package plugin provides lifecycle hooks that observe and intercept an agent
// run, exposing Before*/After* callbacks for the run, agent, model, and tool
// stages.
package plugin

import (
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/session"
)

// Config configures a [Plugin], supplying its name and the lifecycle callbacks
// it registers for the run, agent, model, and tool stages.
type Config struct {
	Name string

	OnUserMessageCallback OnUserMessageCallback

	OnEventCallback OnEventCallback

	BeforeRunCallback BeforeRunCallback
	AfterRunCallback  AfterRunCallback

	BeforeAgentCallback  agent.BeforeAgentCallback
	AfterAgentCallback   agent.AfterAgentCallback
	OnAgentErrorCallback OnAgentErrorCallback

	BeforeModelCallback  llmagent.BeforeModelCallback
	AfterModelCallback   llmagent.AfterModelCallback
	OnModelErrorCallback llmagent.OnModelErrorCallback

	BeforeToolCallback  llmagent.BeforeToolCallback
	AfterToolCallback   llmagent.AfterToolCallback
	OnToolErrorCallback llmagent.OnToolErrorCallback

	CloseFunc func() error
}

// New creates a [Plugin] from cfg.
func New(cfg Config) (*Plugin, error) {
	p := &Plugin{
		name:                  cfg.Name,
		onUserMessageCallback: cfg.OnUserMessageCallback,
		onEventCallback:       cfg.OnEventCallback,
		beforeRunCallback:     cfg.BeforeRunCallback,
		afterRunCallback:      cfg.AfterRunCallback,
		beforeAgentCallback:   cfg.BeforeAgentCallback,
		afterAgentCallback:    cfg.AfterAgentCallback,
		onAgentErrorCallback:  cfg.OnAgentErrorCallback,
		beforeModelCallback:   cfg.BeforeModelCallback,
		afterModelCallback:    cfg.AfterModelCallback,
		onModelErrorCallback:  cfg.OnModelErrorCallback,
		beforeToolCallback:    cfg.BeforeToolCallback,
		afterToolCallback:     cfg.AfterToolCallback,
		onToolErrorCallback:   cfg.OnToolErrorCallback,
		closeFunc:             cfg.CloseFunc,
	}

	// Ensure closeFunc is never nil so p.Close() doesn't panic
	if p.closeFunc == nil {
		p.closeFunc = func() error {
			return nil
		}
	}

	return p, nil
}

// Plugin holds a set of lifecycle callbacks that observe and intercept an agent
// run. Create one with [New].
type Plugin struct {
	name string

	onUserMessageCallback OnUserMessageCallback
	onEventCallback       OnEventCallback

	beforeRunCallback BeforeRunCallback
	afterRunCallback  AfterRunCallback

	beforeAgentCallback agent.BeforeAgentCallback
	afterAgentCallback  agent.AfterAgentCallback

	onAgentErrorCallback OnAgentErrorCallback

	beforeModelCallback  llmagent.BeforeModelCallback
	afterModelCallback   llmagent.AfterModelCallback
	onModelErrorCallback llmagent.OnModelErrorCallback

	beforeToolCallback  llmagent.BeforeToolCallback
	afterToolCallback   llmagent.AfterToolCallback
	onToolErrorCallback llmagent.OnToolErrorCallback

	closeFunc func() error
}

// Name returns the name of the plugin.
func (p *Plugin) Name() string {
	return p.name
}

// Close safely calls the internal close function.
func (p *Plugin) Close() error {
	return p.closeFunc()
}

// --- Accessors ---

// OnUserMessageCallback returns the plugin's OnUserMessageCallback.
func (p *Plugin) OnUserMessageCallback() OnUserMessageCallback {
	return p.onUserMessageCallback
}

// OnEventCallback returns the plugin's OnEventCallback.
func (p *Plugin) OnEventCallback() OnEventCallback {
	return p.onEventCallback
}

// BeforeRunCallback returns the plugin's BeforeRunCallback.
func (p *Plugin) BeforeRunCallback() BeforeRunCallback {
	return p.beforeRunCallback
}

// AfterRunCallback returns the plugin's AfterRunCallback.
func (p *Plugin) AfterRunCallback() AfterRunCallback {
	return p.afterRunCallback
}

// BeforeAgentCallback returns the plugin's BeforeAgentCallback.
func (p *Plugin) BeforeAgentCallback() agent.BeforeAgentCallback {
	return p.beforeAgentCallback
}

// AfterAgentCallback returns the plugin's AfterAgentCallback.
func (p *Plugin) AfterAgentCallback() agent.AfterAgentCallback {
	return p.afterAgentCallback
}

// OnAgentErrorCallback returns the plugin's OnAgentErrorCallback.
func (p *Plugin) OnAgentErrorCallback() OnAgentErrorCallback {
	return p.onAgentErrorCallback
}

// BeforeModelCallback returns the plugin's BeforeModelCallback.
func (p *Plugin) BeforeModelCallback() llmagent.BeforeModelCallback {
	return p.beforeModelCallback
}

// AfterModelCallback returns the plugin's AfterModelCallback.
func (p *Plugin) AfterModelCallback() llmagent.AfterModelCallback {
	return p.afterModelCallback
}

// OnModelErrorCallback returns the plugin's OnModelErrorCallback.
func (p *Plugin) OnModelErrorCallback() llmagent.OnModelErrorCallback {
	return p.onModelErrorCallback
}

// BeforeToolCallback returns the plugin's BeforeToolCallback.
func (p *Plugin) BeforeToolCallback() llmagent.BeforeToolCallback {
	return p.beforeToolCallback
}

// AfterToolCallback returns the plugin's AfterToolCallback.
func (p *Plugin) AfterToolCallback() llmagent.AfterToolCallback {
	return p.afterToolCallback
}

// OnToolErrorCallback returns the plugin's OnToolErrorCallback.
func (p *Plugin) OnToolErrorCallback() llmagent.OnToolErrorCallback {
	return p.onToolErrorCallback
}

// OnUserMessageCallback is invoked when the user sends a message, and may
// return a replacement message.
type OnUserMessageCallback func(agent.InvocationContext, *genai.Content) (*genai.Content, error)

// BeforeRunCallback is invoked before a run starts, and may return content that
// short-circuits the run.
type BeforeRunCallback func(agent.InvocationContext) (*genai.Content, error)

// AfterRunCallback is invoked after a run completes.
type AfterRunCallback func(agent.InvocationContext)

// OnAgentErrorCallback is invoked when an error escapes an agent's run. It is
// notification-only: callbacks are called in registration order and the error
// is yielded to the caller unchanged.
//
// a is the agent that failed. It may be the embedded agent implementation
// rather than the value returned by a constructor, so use a.Name() to identify
// it. ctx is the callback context for the invocation; it does not expose the
// agent (ctx.Agent() returns nil). State changes made through ctx are not
// persisted. Artifact saves through ctx.Artifacts() are written to the artifact
// service, but no ArtifactDelta records them.
//
// A panicking callback is recovered so the remaining callbacks still run and
// the original error is still yielded, matching adk-python's
// on_agent_error_callback. A cancellation or deadline error is not reported
// when the invocation's own context is done, matching adk-python's handling of
// task cancellation; timeouts raised inside the run are reported. Callbacks may
// be invoked concurrently by parallel sub-agents and must be safe for
// concurrent use. An error from a sub-agent is reported once for each ancestor
// as it propagates, matching adk-python; callbacks that count failures should
// dedupe with errors.Is because ancestors wrap the child's error. RunLive
// bypasses the non-live run loop, so it does not invoke this callback.
type OnAgentErrorCallback func(ctx agent.Context, a agent.Agent, err error)

// OnEventCallback is invoked for each event, and may return a replacement event.
type OnEventCallback func(agent.InvocationContext, *session.Event) (*session.Event, error)
