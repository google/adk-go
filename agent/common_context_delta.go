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
	"sync/atomic"

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
//
// It also hands every sibling derivation the same invocation object, where the
// working path gives each one a copy. That is the one cost a reader cannot
// recover by inspecting a value: EndInvocation on any child now ends the parent
// and its siblings, and the write races their Ended. #1135 makes that
// propagation deliberate and synchronises the flag, and until it lands this
// branch has the sharing without either.
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
	// in place with the nil session it reports by design. That is the same reason
	// the report below forwards a nil d rather than short-circuiting on it.
	//
	// What it costs, which is not nothing. An invocation from outside the module
	// that is ALSO a session-less forwarding view — the shape tool_context_wrapper
	// has, written out there instead of in here — is kept rather than unwrapped,
	// so the session it does not have is the one the caller gets, and UserID
	// nil-panics on it. That is a real regression against forwarding through, and
	// it is accepted because the two shapes cannot be told apart here: measured,
	// a forwarding view and a decorator whose own session is nil are both "not
	// ours, session unreadable" before the call, and diverge only after it, into
	// the right user and the enclosing one respectively. Preferring the call
	// rescues the first and hands the second a live user who made no such call.
	// So an out-of-module InvocationContext must carry its own session.
	//
	// It also hands back the invocation itself rather than whatever the call would
	// have produced, and that is true of BOTH shapes of zero delta — nil and
	// allocated-but-empty. Out-of-module invocations are the only population this
	// reaches, and for them both cases changed: without the shortcut each is passed
	// through, so a forwarding view is unwrapped and a decorator is dropped onto
	// what it embeds; with it, each is kept. Two dimensions move together, which
	// object comes back and whether the decorator survives, and the second is why
	// the trade is worth making.
	//
	// One consequence of keeping the object: EndInvocation on a context derived
	// this way reaches the invocation it was derived from, where an ADK-owned
	// invocation handed an empty non-nil delta gets a copy instead, because
	// agent.go's WithICDelta allocates once any delta is present.
	if _, ours := ic.(adkcontext.Source); d.isZero() && !ours {
		return ic
	}
	if next := ic.WithICDelta(d); next != nil {
		return next
	}
	// d is forwarded above even when nil, because an implementation may treat a
	// nil delta as something other than "no change" — both wrappers in this
	// package delegate, and the inner commonContext answers a nil delta by
	// returning itself, which unwraps them. Only the report is skipped.
	if d != nil {
		reportDiscardedDelta(ic, d)
	}
	return ic
}

// reportedNilICDelta remembers which losses have already been reported, as one
// bitmask of field sets per implementation type. A nil return is a static
// property of the implementation rather than a transient, so without this
// withICDelta reports once per derived context — per workflow node, per agent
// activation, per parallel item.
//
// The type alone would be the wrong key now that the message carries a field
// list: a later discard that loses different fields does say something the first
// line did not, and on the workflow path the first one is often the least
// informative — a Branch-only derivation from the scheduler, ahead of the Agent
// the run actually swapped. Field values do not enter the key, so a
// thousand-item fan-out deriving the same shape still reports once.
//
// The mask is a value rather than part of the key because a struct key would
// have to be boxed into an interface on every lookup, which allocates on exactly
// the repeat path this exists to make cheap.
var reportedNilICDelta sync.Map // reflect.Type -> *atomic.Uint32

const (
	lostAgent uint8 = 1 << iota
	lostBranch
	lostIsolationScope
	lostUserContent
	lostContext
)

func reportDiscardedDelta(ic InvocationContext, d *InvocationContextDelta) {
	// The mask is built first so a repeat pays neither the slice nor the
	// formatting below. Only an implementation that returns nil reaches here at
	// all, but it reaches here on every derivation for the life of the process.
	var fields uint8
	if d.Agent != nil {
		fields |= lostAgent
	}
	if d.Branch != nil {
		fields |= lostBranch
	}
	if d.IsolationScope != nil {
		fields |= lostIsolationScope
	}
	if d.UserContent != nil {
		fields |= lostUserContent
	}
	if d.Context != nil {
		fields |= lostContext
	}
	if fields == 0 {
		// The delta asked for nothing, so nothing was lost and no report is owed —
		// and claiming a bit here would spend one a real loss needs.
		return
	}
	typ := reflect.TypeOf(ic)
	seen, ok := reportedNilICDelta.Load(typ)
	if !ok {
		seen, _ = reportedNilICDelta.LoadOrStore(typ, new(atomic.Uint32))
	}
	if prev := seen.(*atomic.Uint32).Or(1 << fields); prev&(1<<fields) != 0 {
		return
	}

	// Only what the delta asked for is safe to render. An implementation that
	// has just returned nil is by definition partial, so calling its accessors
	// to enrich this message risks a second failure inside the error path.
	var lost []string
	if fields&lostAgent != 0 {
		lost = append(lost, "Agent")
	}
	if fields&lostBranch != 0 {
		lost = append(lost, fmt.Sprintf("Branch=%q", *d.Branch))
	}
	if fields&lostIsolationScope != 0 {
		lost = append(lost, fmt.Sprintf("IsolationScope=%q", *d.IsolationScope))
	}
	if fields&lostUserContent != 0 {
		lost = append(lost, "UserContent")
	}
	if fields&lostContext != 0 {
		lost = append(lost, "Context")
	}
	// The subject is the invocation, not the context the caller gets back.
	// WithDelta installs d.Context on the latter afterwards, so saying the delta
	// was "discarded" would be read as covering both. These fields did not reach
	// the invocation, which is true on either entry point.
	log.Printf("agent: %T.WithICDelta returned nil, so the previous invocation is kept and "+
		"these delta fields did not reach it: %s. Further occurrences of this loss from "+
		"this type are not reported", ic, strings.Join(lost, ", "))
}

// isZero reports whether d asks for no change to the invocation. A nil delta and
// an allocated one with every field unset are the same request, and treating
// them differently is what let an empty delta drop an out-of-module invocation
// while a nil one preserved it.
func (d *InvocationContextDelta) isZero() bool {
	return d == nil || *d == InvocationContextDelta{}
}
