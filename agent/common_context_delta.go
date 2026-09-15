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

import (
	"context"
	"fmt"
	"log"
	"reflect"
	"strings"
	"sync"

	"google.golang.org/genai"
)

// CommonContextDelta holds all the changes which should be applied to a new child context based on agent.Context.
type CommonContextDelta struct {
	ResumeInputs           *map[string]any
	InvocationContextDelta *InvocationContextDelta
	Path                   *string
	RunID                  *string
	SubScheduler           *DynamicSubScheduler
	OutputForAncestors     *[]string
}

// InvocationContextDelta holds all the changes which should be applied to a new child context based on agent.InvocationContext
type InvocationContextDelta struct {
	Context        *context.Context
	UserContent    **genai.Content
	Agent          *Agent
	Branch         *string
	IsolationScope *string
}

// WithDelta returns a new CommmonContext with all the changes from d applied.
// If there are no changes, the original context is returned.
func (c *commonContext) WithDelta(d *CommonContextDelta) Context {
	if d == nil {
		return c
	}
	res := *c
	// true: the Context assignment below lands on the returned context whether or
	// not the invocation accepted the delta, so it is not lost here.
	res.invocationContext = withICDelta(res.invocationContext, d.InvocationContextDelta, true)

	if d.InvocationContextDelta != nil {
		if d.InvocationContextDelta.Context != nil {
			res.Context = *d.InvocationContextDelta.Context
		}
	}
	if d.ResumeInputs != nil {
		res.resumeInputs = *d.ResumeInputs
	}
	if d.Path != nil {
		res.path = *d.Path
	}
	if d.RunID != nil {
		res.runID = *d.RunID
	}
	if d.SubScheduler != nil {
		res.subScheduler = *d.SubScheduler
	}
	if d.OutputForAncestors != nil {
		res.outputForAncestors = *d.OutputForAncestors
	}

	return &res
}

// WithICDelta returns a new context (copying all the fields from the original one) with changes applied to the underlying InvocationContext
func (c *commonContext) WithICDelta(d *InvocationContextDelta) InvocationContext {
	if d == nil {
		return c
	}
	res := *c
	// false: this entry point never installs d.Context on the returned context,
	// so a discarded delta loses it.
	res.invocationContext = withICDelta(res.invocationContext, d, false)
	return &res
}

// withICDelta applies d to the invocation this context speaks for, and refuses a
// nil result.
//
// Nothing in this repository returns nil from WithICDelta. The guard is for
// implementations written outside it, which the exported interface allows and
// which cannot be enumerated — and which reach the shape easily, because a
// decorator that embeds InvocationContext has to override WithICDelta or lose
// itself on the first delta.
//
// Storing that nil leaves a commonContext whose invocation is gone, and most of
// its accessors — Agent, Branch, Session, UserID among them — dereference it.
// Where that panic surfaces depends on the node: a plain workflow node
// dereferences the invocation inside startNodeSpan, which runNode calls before
// installing its recover, so the process dies. A node that emits its own span
// gets a noop span and an untouched context, so the panic happens inside Run
// and is recovered as "node %q panicked".
//
// Keeping the previous invocation is the better of the two, but it is not free:
// the delta is gone, so the caller runs on with the previous Agent, Branch and
// IsolationScope. Nothing else distinguishes that from the delta having been
// applied, so it is reported. An agent running under the wrong parent is not a
// quiet kind of wrong.
// callerAppliesContext says whether the caller installs d.Context on the context
// it returns. The two entry points differ, and it decides whether a discarded
// delta costs the caller its Context or only the invocation's.
func withICDelta(ic InvocationContext, d *InvocationContextDelta, callerAppliesContext bool) InvocationContext {
	if ic == nil {
		return nil
	}
	if next := ic.WithICDelta(d); next != nil {
		return next
	}
	// d is forwarded above even when nil, because an implementation may treat a
	// nil delta as something other than "no change" — both wrappers in this
	// package delegate, and the inner commonContext answers a nil delta by
	// returning itself, which unwraps them. Only the report is skipped.
	if d != nil {
		reportDiscardedDelta(ic, d, callerAppliesContext)
	}
	return ic
}

// reportedNilICDelta remembers which implementations have already been reported.
// withICDelta runs once per derived context — per workflow node, per agent
// activation, per parallel item — and a nil return is a static property of the
// implementation rather than a transient, so reporting every occurrence buries
// the operator without telling them anything the first line did not.
var reportedNilICDelta sync.Map // reflect.Type -> struct{}

func reportDiscardedDelta(ic InvocationContext, d *InvocationContextDelta, callerAppliesContext bool) {
	// Only what the delta asked for is safe to render. An implementation that
	// has just returned nil is by definition partial, so calling its accessors
	// to enrich this message risks a second failure inside the error path.
	var lost []string
	if d.Agent != nil {
		lost = append(lost, "Agent")
	}
	if d.Branch != nil {
		lost = append(lost, fmt.Sprintf("Branch=%q", *d.Branch))
	}
	if d.IsolationScope != nil {
		lost = append(lost, fmt.Sprintf("IsolationScope=%q", *d.IsolationScope))
	}
	if d.UserContent != nil {
		lost = append(lost, "UserContent")
	}
	if d.Context != nil && !callerAppliesContext {
		lost = append(lost, "Context")
	}
	if len(lost) == 0 {
		return
	}
	// Claim the type only once there is something worth saying, so a delta that
	// asked for nothing cannot spend the one report a later, informative one needs.
	if _, dup := reportedNilICDelta.LoadOrStore(reflect.TypeOf(ic), struct{}{}); dup {
		return
	}
	log.Printf("agent: %T.WithICDelta returned nil, so the previous invocation is kept and "+
		"this delta is discarded (%s). Further occurrences from this type are not reported",
		ic, strings.Join(lost, ", "))
}
