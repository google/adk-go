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

package agent

import (
	"context"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/artifact"
	"google.golang.org/adk/session"
)

// NewCallbackContext returns a CallbackContext that delegates reads
// to the given InvocationContext and accumulates state and artifact
// mutations into a fresh EventActions. Use NewCallbackContextWithDelta
// when the caller already owns the EventActions delta maps (typical
// for the per-event accumulators inside the LLM flow).
func NewCallbackContext(ctx InvocationContext) CallbackContext {
	return newCallbackContext(ctx, make(map[string]any), make(map[string]int64))
}

// NewCallbackContextWithDelta returns a CallbackContext that records
// every state Set into stateDelta and every Artifacts.Save into
// artifactDelta. Pass the caller-owned delta maps directly when the
// caller intends to inspect them after the callback returns (the
// scheduler does this to drive Event.Actions on each yielded event).
func NewCallbackContextWithDelta(ctx InvocationContext, stateDelta map[string]any, artifactDelta map[string]int64) CallbackContext {
	return newCallbackContext(ctx, stateDelta, artifactDelta)
}

func newCallbackContext(ctx InvocationContext, stateDelta map[string]any, artifactDelta map[string]int64) *callbackContextImpl {
	rCtx := NewReadonlyContext(ctx)
	eventActions := &session.EventActions{StateDelta: stateDelta, ArtifactDelta: artifactDelta}
	return &callbackContextImpl{
		ReadonlyContext: rCtx,
		invocationCtx:   ctx,
		eventActions:    eventActions,
		artifacts: &deltaTrackingArtifacts{
			Artifacts:    ctx.Artifacts(),
			eventActions: eventActions,
		},
	}
}

// callbackContextImpl is the canonical, in-process implementation of
// CallbackContext. It embeds a ReadonlyContext for the read surface
// and adds a writable State + a delta-tracking Artifacts wrapper.
type callbackContextImpl struct {
	ReadonlyContext
	artifacts     *deltaTrackingArtifacts
	invocationCtx InvocationContext
	eventActions  *session.EventActions
}

func (c *callbackContextImpl) Artifacts() Artifacts {
	return c.artifacts
}

func (c *callbackContextImpl) AgentName() string {
	return c.invocationCtx.Agent().Name()
}

func (c *callbackContextImpl) ReadonlyState() session.ReadonlyState {
	return c.invocationCtx.Session().State()
}

func (c *callbackContextImpl) State() session.State {
	return &deltaTrackingState{ctx: c}
}

func (c *callbackContextImpl) InvocationID() string {
	return c.invocationCtx.InvocationID()
}

func (c *callbackContextImpl) UserContent() *genai.Content {
	return c.invocationCtx.UserContent()
}

// deltaTrackingArtifacts wraps an Artifacts service and records every
// successful Save into the given EventActions.ArtifactDelta. Reads
// (Load, LoadVersion, List) pass through unchanged.
type deltaTrackingArtifacts struct {
	Artifacts
	eventActions *session.EventActions
}

func (a *deltaTrackingArtifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	resp, err := a.Artifacts.Save(ctx, name, data)
	if err != nil {
		return resp, err
	}
	if a.eventActions != nil {
		if a.eventActions.ArtifactDelta == nil {
			a.eventActions.ArtifactDelta = make(map[string]int64)
		}
		// TODO: RWLock, check the version stored is newer in case multiple tools save the same file.
		a.eventActions.ArtifactDelta[name] = resp.Version
	}
	return resp, nil
}

// deltaTrackingState wraps the session State so that every Set is
// also recorded into EventActions.StateDelta. Reads consult the
// delta first, then fall through to the underlying session state.
type deltaTrackingState struct {
	ctx *callbackContextImpl
}

func (s *deltaTrackingState) Get(key string) (any, error) {
	if s.ctx.eventActions != nil && s.ctx.eventActions.StateDelta != nil {
		if val, ok := s.ctx.eventActions.StateDelta[key]; ok {
			return val, nil
		}
	}
	return s.ctx.invocationCtx.Session().State().Get(key)
}

func (s *deltaTrackingState) Set(key string, val any) error {
	if s.ctx.eventActions != nil && s.ctx.eventActions.StateDelta != nil {
		s.ctx.eventActions.StateDelta[key] = val
	}
	return s.ctx.invocationCtx.Session().State().Set(key, val)
}

func (s *deltaTrackingState) All() iter.Seq2[string, any] {
	return s.ctx.invocationCtx.Session().State().All()
}

var _ CallbackContext = (*callbackContextImpl)(nil)
