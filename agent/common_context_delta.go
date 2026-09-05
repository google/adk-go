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

	"google.golang.org/adk/v2/internal/adkcontext"
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

// withICDelta applies d to the invocation this context speaks for.
//
// It refuses only a nil result, keeping the original invocation instead. That
// guard protects the whole [Context] surface rather than the identity in
// particular: a nil invocation dereferences in most of the accessors on
// commonContext. The identity itself already fails closed for that shape —
// commonContext.identity reports nothing when it has no invocation, rather than
// asking its parent — so this is not what stands between a caller and the
// enclosing call's user.
//
// It does NOT try to detect the larger problem, which is that an
// InvocationContext written outside the module inherits WithICDelta by
// promotion, so the promoted method hands back the invocation it embeds and the
// decorator — with the session naming its own user — is dropped. Two attempts to
// catch that from here were measured and both were worse than the disease. Keying
// on "is the receiver one of ours" refuses the delta for an in-module test double
// that implements WithICDelta perfectly well. Keying on "did a value that could
// not answer for itself turn into one that can" misses a decorator whose parent
// is also from outside the module, and where it does fire it discards the whole
// delta — so agent.Run then reads Agent, Branch and IsolationScope from the
// enclosing invocation, or nil-panics when there is no enclosing agent.
//
// The defect is in the decorator contract, not here: a type that cannot override
// WithICDelta cannot survive a delta, and no amount of inspection at this call
// site reconstructs what it should have returned. It predates the identity key
// and is documented on IdentityFromContext instead.
func withICDelta(ic InvocationContext, d *InvocationContextDelta) InvocationContext {
	if ic == nil {
		return nil
	}
	// A delta with nothing to say about the invocation must not cost an
	// invocation from outside the module its identity. A promoted WithICDelta
	// hands back the invocation the decorator embeds whatever the delta says, so
	// merely asking is what drops the decorator — and entering a workflow does
	// exactly that, with a CommonContextDelta carrying no InvocationContextDelta
	// at all (workflow.go sets Path, RunID, OutputForAncestors and SubScheduler).
	//
	// "Nothing to say" is emptiness, not a nil pointer. A caller that allocates
	// the delta and then fills it conditionally hands over an empty one whenever
	// no condition fires, and keying on nil alone made that one-token neighbour
	// drop the decorator where the nil case did not.
	//
	// Ours are still asked, because for them an empty delta is not a no-op: the
	// tool and callback wrappers forward to the commonContext they hold, which
	// returns that inner context, and skipping the call would leave the wrapper
	// in place with the nil session it reports by design.
	if _, ours := ic.(adkcontext.Source); d.isZero() && !ours {
		return ic
	}
	if next := ic.WithICDelta(d); next != nil {
		return next
	}
	// Keeping the original beats propagating nil, which breaks most of the
	// accessors on commonContext. But the delta is gone, so the caller silently
	// runs with the previous Agent, Branch and IsolationScope — logged because
	// nothing else distinguishes that from the delta having been applied, and an
	// agent running under the wrong parent is not a quiet kind of wrong.
	log.Printf("agent: %T.WithICDelta returned nil; keeping the previous invocation and "+
		"discarding the delta, so Agent, Branch and IsolationScope are unchanged", ic)
	return ic
}

// isZero reports whether d asks for no change to the invocation. A nil delta and
// an allocated one with every field unset are the same request, and treating
// them differently is what let an empty delta drop an out-of-module invocation
// while a nil one preserved it.
func (d *InvocationContextDelta) isZero() bool {
	return d == nil || *d == InvocationContextDelta{}
}
