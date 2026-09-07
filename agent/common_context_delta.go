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
	"log"

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
	res.invocationContext = withICDelta(res.invocationContext, d.InvocationContextDelta)

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
	res.invocationContext = withICDelta(res.invocationContext, d)
	return &res
}

// withICDelta applies d to the invocation this context speaks for, and refuses a
// nil result.
//
// An InvocationContext is free to return nil from WithICDelta, and a partial
// implementation does: ADK's own ContextMock returns nil from WithContext,
// WithBranch and WithAgentContext, so the shape is not exotic. Storing that nil
// leaves a commonContext whose invocation is gone, and most of its accessors —
// Agent, Branch, Session, UserID among them — dereference it. On the workflow
// path the scheduler catches the panic and reports it as "node %q panicked",
// naming the node rather than the cause.
//
// Keeping the previous invocation is the better of the two, but it is not free:
// the delta is gone, so the caller runs on with the previous Agent, Branch and
// IsolationScope. Nothing else distinguishes that from the delta having been
// applied, so it is logged. An agent running under the wrong parent is not a
// quiet kind of wrong.
func withICDelta(ic InvocationContext, d *InvocationContextDelta) InvocationContext {
	if ic == nil {
		return nil
	}
	if next := ic.WithICDelta(d); next != nil {
		return next
	}
	log.Printf("agent: %T.WithICDelta returned nil; keeping the previous invocation and "+
		"discarding the delta, so Agent, Branch and IsolationScope are unchanged", ic)
	return ic
}
