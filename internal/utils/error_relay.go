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

package utils

import (
	"context"

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
)

// errRelayKey is the context key of the errRelay that the agent running with
// the context may report through.
type errRelayKey struct{}

// errRelay lets a ParallelAgent nested below another one report an error to
// the outer one as soon as it records it, before the agents in between have
// returned it. It is bound to one agent, compared by identity rather than by
// name, so another agent cannot report through it even under the same name. It
// is valid only in the invocation it was made for.
type errRelay struct {
	invocationID string
	// agent is the State of the only agent that may report through or pass
	// on this relay. A wrapper such as the one SequentialAgent returns shares
	// the State of the agent it wraps.
	agent  *agentinternal.State
	report func(error)
}

// agentState returns the State that identifies a, or nil for an agent that
// does not expose one, such as a type that embeds an agent.Agent.
func agentState(a agent.Agent) *agentinternal.State {
	if internalAgent, ok := a.(agentinternal.Agent); ok {
		return agentinternal.Reveal(internalAgent)
	}
	return nil
}

// boundTo returns the relay in ctx if it is bound to ctx.Agent() in this
// invocation.
func boundTo(ctx agent.InvocationContext) (errRelay, bool) {
	r, ok := ctx.Value(errRelayKey{}).(errRelay)
	if !ok || r.invocationID != ctx.InvocationID() || r.agent == nil || r.agent != agentState(ctx.Agent()) {
		return errRelay{}, false
	}
	return r, true
}

// WithErrRelay returns ctx with a relay that subAgent may report through in the
// invocation invocationID. report is called with each error reported, possibly
// from several goroutines at once, so it must be safe for concurrent use.
func WithErrRelay(ctx context.Context, invocationID string, subAgent agent.Agent, report func(error)) context.Context {
	return context.WithValue(ctx, errRelayKey{}, errRelay{invocationID: invocationID, agent: agentState(subAgent), report: report})
}

// PassErrRelay returns ctx with the relay bound to ctx.Agent() rebound to
// subAgent, or ctx itself when there is no such relay. Call it only for a
// sub-agent whose errors ctx.Agent() yields to its own caller at once and
// unchanged, and whose events it keeps reading until the sub-agent ends or
// fails, as SequentialAgent and LoopAgent do. An agent that may catch or
// retry an error does not call it, so a ParallelAgent it runs cannot report
// through the relay.
func PassErrRelay(ctx agent.InvocationContext, subAgent agent.Agent) agent.InvocationContext {
	r, ok := boundTo(ctx)
	if !ok {
		return ctx
	}
	r.agent = agentState(subAgent)
	return ctx.WithContext(context.WithValue(ctx, errRelayKey{}, r))
}

// ReportErr reports err through the relay in ctx if it is bound to
// ctx.Agent().
func ReportErr(ctx agent.InvocationContext, err error) {
	if r, ok := boundTo(ctx); ok {
		r.report(err)
	}
}
