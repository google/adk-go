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

package agent

import "context"

// An error relay lets a ParallelAgent nested below another one report an error
// to the outer one as soon as it records it, before the agents in between have
// returned it. The relay is bound to the one agent allowed to report through
// it. An agent that yields a sub-agent's errors to its own caller at once and
// unchanged, such as a SequentialAgent or a LoopAgent, rebinds the relay to
// each sub-agent it runs. Any other agent leaves it bound to itself, so a
// ParallelAgent it runs, which it may catch or retry, cannot report through it.

type errRelayKey struct{}

type errRelay struct {
	invocationID string
	agentName    string
	report       func(error)
}

// WithErrRelay returns ctx with a relay that the agent named agentName may
// report through in the invocation invocationID.
func WithErrRelay(ctx context.Context, invocationID, agentName string, report func(error)) context.Context {
	return context.WithValue(ctx, errRelayKey{}, errRelay{invocationID: invocationID, agentName: agentName, report: report})
}

// PassErrRelay rebinds the relay bound to parentName in ctx to childName. It
// reports false, and leaves ctx as it is, when ctx has no relay bound to
// parentName in this invocation. Call it only for a sub-agent whose errors the
// caller yields to its own caller at once and unchanged.
func PassErrRelay(ctx context.Context, invocationID, parentName, childName string) (context.Context, bool) {
	r, ok := ctx.Value(errRelayKey{}).(errRelay)
	if !ok || r.invocationID != invocationID || r.agentName != parentName {
		return ctx, false
	}
	r.agentName = childName
	return context.WithValue(ctx, errRelayKey{}, r), true
}

// ReportErr reports err through the relay in ctx if it is bound to the agent
// named agentName in the invocation invocationID.
func ReportErr(ctx context.Context, invocationID, agentName string, err error) {
	if r, ok := ctx.Value(errRelayKey{}).(errRelay); ok && r.invocationID == invocationID && r.agentName == agentName {
		r.report(err)
	}
}
