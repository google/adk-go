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

package workflow

import (
	"context"

	"google.golang.org/adk/agent"
)

// nodeBridge carries the per-node context plus the workflow-only state
// that cannot live on agent.Context (the agent package must not import
// workflow). It is stashed on the embedded Go context so that:
//   - tools running inside an LlmAgent node can recover the node's
//     agent.Context (and its NodeScheduler) via the value chain, and
//   - a nested dynamic node can recover its parent's
//     outputForAncestors chain.
//
// EXPERIMENTAL bridge introduced with context unification; it replaces
// the former workflow.NodeContext interface.
type nodeBridge struct {
	ctx agent.Context
	// outputForAncestors are the extra node paths this activation's
	// output counts for, when this activation is itself a
	// WithUseAsOutput child. This is workflow-only state that cannot
	// live on agent.Context (the agent package must not import
	// workflow). Resume inputs, by contrast, ride directly on
	// agent.Context (ctx.ResumeInputs()), so they are not duplicated
	// here.
	outputForAncestors []string
}

type nodeContextKey struct{}

// withNodeBridge returns a derived Go context carrying b under an opaque
// key, recoverable via nodeBridgeFromGoContext from any descendant
// context.Context. The scheduler calls this on every per-node activation.
func withNodeBridge(parent context.Context, b *nodeBridge) context.Context {
	if parent == nil || b == nil {
		return parent
	}
	return context.WithValue(parent, nodeContextKey{}, b)
}

// nodeBridgeFromGoContext returns the nodeBridge stashed by
// withNodeBridge, or (nil, false) if none is present on ctx.
func nodeBridgeFromGoContext(ctx context.Context) (*nodeBridge, bool) {
	if ctx == nil {
		return nil, false
	}
	b, ok := ctx.Value(nodeContextKey{}).(*nodeBridge)
	return b, ok
}

// NodeContextFromGoContext returns the workflow node's agent.Context
// stashed on ctx, or (nil, false) if ctx is not running inside a
// workflow node. Runnable tools use this to recover the surrounding
// node context (e.g. to call RunNode) without a dedicated parameter.
func NodeContextFromGoContext(ctx context.Context) (agent.Context, bool) {
	b, ok := nodeBridgeFromGoContext(ctx)
	if !ok {
		return nil, false
	}
	return b.ctx, true
}
