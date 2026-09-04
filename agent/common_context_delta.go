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

	"google.golang.org/adk/v2/internal/adkcontext"

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

// withICDelta applies d to the invocation this context speaks for, and refuses to
// let that application change WHICH invocation it is.
//
// An InvocationContext written outside the module cannot override WithICDelta
// for a key it cannot name — but it inherits the method by promotion, and the
// promoted method hands back the invocation it EMBEDS. The decorator, and the
// session naming its own user, are dropped, and the context goes on to speak for
// the enclosing call. A per-user credential minted under it then belongs to
// someone who made no such call.
//
// The signature of that drop is precise: a value that could not answer for
// itself has turned into one that can. Anything else — an invocation of ours, or
// one from elsewhere that really did return a delta'd copy of itself — is left
// alone, so this costs nothing for a type that implements WithICDelta properly.
// A decorator that cannot have the delta applied to it correctly is better left
// un-delta'd than replaced by a different call's invocation.
func withICDelta(ic InvocationContext, d *InvocationContextDelta) InvocationContext {
	next := ic.WithICDelta(d)
	if next == nil {
		// It lost its invocation, and a context with none falls back to asking its
		// parent — the enclosing call again.
		return ic
	}
	if _, wasOurs := ic.(adkcontext.Source); !wasOurs {
		if _, isOurs := next.(adkcontext.Source); isOurs {
			return ic
		}
	}
	return next
}
