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

import "google.golang.org/adk/v2/internal/adkcontext"

// RecordEndOfAgent records that the agent ctx runs, ctx.Agent(), has ended in
// the current run, without yielding an event. A ParallelAgent in a resumable
// run writes its own end-of-agent event only after every sub-agent has ended,
// and a resumable ParallelAgent does not start a sub-agent that has ended.
// Agents built by llmagent, sequentialagent, loopagent and parallelagent record
// their end themselves, together with the end-of-agent event they yield; call
// RecordEndOfAgent from the Run function of an agent built with [New], when its
// Run is finished, so that a resumable ParallelAgent that contains it can end.
// Without the call such a ParallelAgent never writes its end-of-agent event.
// A graph workflow agent built by workflowagent.New records no end, and its
// Run cannot be given the call: a resumable ParallelAgent that contains one
// writes its end-of-agent event only after an escalation by a direct
// sub-agent, and resuming the invocation runs the workflow again, even when it
// finished in an earlier turn. The record lives in the run only: it adds
// nothing to the session, so a later resume of the invocation does not see
// this end and runs the agent again. Within the run the mark lasts until an
// enclosing LoopAgent starts its next pass, which clears it so that the agent
// runs again.
//
// It does nothing when the run is not resumable, when ctx does not belong to a
// run started by a runner, and when ctx.Agent() is nil. It is safe for
// concurrent use.
func RecordEndOfAgent(ctx InvocationContext) {
	a := ctx.Agent()
	if r, ok := ctx.Value(adkcontext.RunConfigKey).(agentEndRecorder); ok && a != nil {
		r.RecordAgentEnd(a.Name())
	}
}

// agentEndRecorder is the view package agent has of the run config stored
// under [adkcontext.RunConfigKey]: package agent cannot import the run config's
// package, so it reads the value through an interface the type satisfies.
type agentEndRecorder interface {
	RecordAgentEnd(name string)
}
