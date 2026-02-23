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

package context

import (
	"context"
	"sync"

	"github.com/google/uuid"
	"github.com/rs/zerolog/log"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

type InvocationContextParams struct {
	Artifacts agent.Artifacts
	Memory    agent.Memory
	Session   session.Session

	Branch string
	Agent  agent.Agent

	UserContent        *genai.Content
	RunConfig          *agent.RunConfig
	EndInvocation      bool
	ResumabilityConfig *agent.ResumabilityConfig

	LiveRequestQueue            *agent.LiveRequestQueue
	LiveSessionResumptionHandle string

	InvocationID string
}

func NewInvocationContext(ctx context.Context, params InvocationContextParams) agent.InvocationContext {
	if params.InvocationID == "" {
		params.InvocationID = "e-" + uuid.NewString()
	}
	return &InvocationContext{
		Context: ctx,
		params:  params,
		state: &invocationState{
			endInvocation:               params.EndInvocation,
			liveSessionResumptionHandle: params.LiveSessionResumptionHandle,
			transcriptionCache:          make([]agent.TranscriptionEntry, 0),
			inputRealtimeCache:          make([]agent.RealtimeCacheEntry, 0),
			outputRealtimeCache:         make([]agent.RealtimeCacheEntry, 0),
		},
	}
}

type invocationState struct {
	mu                          sync.RWMutex
	endInvocation               bool
	liveSessionResumptionHandle string
	transcriptionCache          []agent.TranscriptionEntry
	inputRealtimeCache          []agent.RealtimeCacheEntry
	outputRealtimeCache         []agent.RealtimeCacheEntry
}

type InvocationContext struct {
	context.Context

	params InvocationContextParams
	state  *invocationState
}

func (c *InvocationContext) Artifacts() agent.Artifacts {
	return c.params.Artifacts
}

func (c *InvocationContext) Agent() agent.Agent {
	return c.params.Agent
}

func (c *InvocationContext) Branch() string {
	return c.params.Branch
}

func (c *InvocationContext) InvocationID() string {
	return c.params.InvocationID
}

func (c *InvocationContext) Memory() agent.Memory {
	return c.params.Memory
}

func (c *InvocationContext) Session() session.Session {
	return c.params.Session
}

func (c *InvocationContext) UserContent() *genai.Content {
	return c.params.UserContent
}

func (c *InvocationContext) RunConfig() *agent.RunConfig {
	return c.params.RunConfig
}

func (c *InvocationContext) EndInvocation() {
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	c.state.endInvocation = true
}

func (c *InvocationContext) Ended() bool {
	c.state.mu.RLock()
	defer c.state.mu.RUnlock()
	return c.state.endInvocation
}

func (c *InvocationContext) LiveRequestQueue() *agent.LiveRequestQueue {
	return c.params.LiveRequestQueue
}

func (c *InvocationContext) TranscriptionCache() []agent.TranscriptionEntry {
	c.state.mu.RLock()
	defer c.state.mu.RUnlock()
	return c.state.transcriptionCache
}

func (c *InvocationContext) LiveSessionResumptionHandle() string {
	c.state.mu.RLock()
	handle := c.state.liveSessionResumptionHandle
	c.state.mu.RUnlock()

	log.Info().Str("handle", handle).
		Str("func", "InvocationContext.LiveSessionResumptionHandle").
		Msg("Getting live session resumption handle")
	return handle
}

func (c *InvocationContext) InputRealtimeCache() []agent.RealtimeCacheEntry {
	c.state.mu.RLock()
	defer c.state.mu.RUnlock()
	return c.state.inputRealtimeCache
}

func (c *InvocationContext) OutputRealtimeCache() []agent.RealtimeCacheEntry {
	c.state.mu.RLock()
	defer c.state.mu.RUnlock()
	return c.state.outputRealtimeCache
}

func (c *InvocationContext) ResumabilityConfig() *agent.ResumabilityConfig {
	return c.params.ResumabilityConfig
}

func (c *InvocationContext) AppendInputRealtimeCache(entry agent.RealtimeCacheEntry) {
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	c.state.inputRealtimeCache = append(c.state.inputRealtimeCache, entry)
}

func (c *InvocationContext) AppendOutputRealtimeCache(entry agent.RealtimeCacheEntry) {
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	c.state.outputRealtimeCache = append(c.state.outputRealtimeCache, entry)
}

func (c *InvocationContext) ClearInputRealtimeCache() {
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	c.state.inputRealtimeCache = nil
}

func (c *InvocationContext) ClearOutputRealtimeCache() {
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	c.state.outputRealtimeCache = nil
}

func (c *InvocationContext) SetLiveSessionResumptionHandle(handle string) {
	log.Info().Str("handle", handle).
		Str("func", "InvocationContext.SetLiveSessionResumptionHandle").
		Msg("Setting live session resumption handle")

	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	c.state.liveSessionResumptionHandle = handle
}

func (c *InvocationContext) WithContext(ctx context.Context) agent.InvocationContext {
	newCtx := *c
	newCtx.Context = ctx
	return &newCtx
}

var _ agent.InvocationContext = (*InvocationContext)(nil)
