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

// Package plugin provides.
package plugin

import (
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/session"
)

type Config struct {
	Name string

	OnUserMessageCallback OnUserMessageCallback

	OnEventCallback OnEventCallback

	BeforeRunCallback BeforeRunCallback
	AfterRunCallback  AfterRunCallback

	BeforeAgentCallback agent.BeforeAgentCallback
	AfterAgentCallback  agent.AfterAgentCallback

	BeforeModelCallback  llmagent.BeforeModelCallback
	AfterModelCallback   llmagent.AfterModelCallback
	OnModelErrorCallback llmagent.OnModelErrorCallback

	BeforeToolCallback  llmagent.BeforeToolCallback
	AfterToolCallback   llmagent.AfterToolCallback
	OnToolErrorCallback llmagent.OnToolErrorCallback

	CloseFunc func() error
}

func New(cfg Config) (*Plugin, error) {
	p := &Plugin{
		name:                  cfg.Name,
		onUserMessageCallback: cfg.OnUserMessageCallback,
		onEventCallback:       cfg.OnEventCallback,
		beforeRunCallback:     cfg.BeforeRunCallback,
		afterRunCallback:      cfg.AfterRunCallback,
		beforeAgentCallback:   cfg.BeforeAgentCallback,
		afterAgentCallback:    cfg.AfterAgentCallback,
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

type Plugin struct {
	name string

	onUserMessageCallback OnUserMessageCallback
	onEventCallback       OnEventCallback

	beforeRunCallback BeforeRunCallback
	afterRunCallback  AfterRunCallback

	beforeAgentCallback agent.BeforeAgentCallback
	afterAgentCallback  agent.AfterAgentCallback

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

func (p *Plugin) OnUserMessageCallback() OnUserMessageCallback {
	return p.onUserMessageCallback
}

func (p *Plugin) OnEventCallback() OnEventCallback {
	return p.onEventCallback
}

func (p *Plugin) BeforeRunCallback() BeforeRunCallback {
	return p.beforeRunCallback
}

func (p *Plugin) AfterRunCallback() AfterRunCallback {
	return p.afterRunCallback
}

func (p *Plugin) BeforeAgentCallback() agent.BeforeAgentCallback {
	return p.beforeAgentCallback
}

func (p *Plugin) AfterAgentCallback() agent.AfterAgentCallback {
	return p.afterAgentCallback
}

func (p *Plugin) BeforeModelCallback() llmagent.BeforeModelCallback {
	return p.beforeModelCallback
}

func (p *Plugin) AfterModelCallback() llmagent.AfterModelCallback {
	return p.afterModelCallback
}

func (p *Plugin) OnModelErrorCallback() llmagent.OnModelErrorCallback {
	return p.onModelErrorCallback
}

func (p *Plugin) BeforeToolCallback() llmagent.BeforeToolCallback {
	return p.beforeToolCallback
}

func (p *Plugin) AfterToolCallback() llmagent.AfterToolCallback {
	return p.afterToolCallback
}

func (p *Plugin) OnToolErrorCallback() llmagent.OnToolErrorCallback {
	return p.onToolErrorCallback
}

type OnUserMessageCallback func(agent.InvocationContext, *genai.Content) (*genai.Content, error)

type BeforeRunCallback func(agent.InvocationContext) (*genai.Content, error)

type AfterRunCallback func(agent.InvocationContext)

type OnEventCallback func(agent.InvocationContext, *session.Event) (*session.Event, error)
