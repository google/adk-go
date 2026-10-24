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
	"slices"
	"sync"

	"google.golang.org/adk/v2/agent"
)

// Recovery evidence travels separately from the business error so retry
// predicates and callers still receive the node's original error object.
type failureInventory struct {
	known bool
	paths []string
}

func (i failureInventory) clone() failureInventory {
	i.paths = slices.Clone(i.paths)
	return i
}

type failureInventoryKey struct{}

// Direct public Run calls can inherit the enclosing context without being
// tracked child activations. Only the executor's actual producer may publish;
// repeated publications from a reused context are ambiguous, not last-wins.
// Matching the returned error cannot authenticate an opaque decorator: it
// might have handled a child failure before reusing the same sentinel.
type failureInventoryRecorder struct {
	producer  any
	mu        sync.Mutex
	published bool
	inventory failureInventory
}

func (r *failureInventoryRecorder) snapshot() failureInventory {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.inventory.clone()
}

func withFailureInventory(ctx agent.Context, node Node) (agent.Context, *failureInventoryRecorder) {
	var producer any = node
	if nested, ok := node.(*WorkflowNode); ok {
		producer = nested.subWorkflow.graph
	}
	recorder := &failureInventoryRecorder{producer: producer}
	base := context.WithValue(ctx, failureInventoryKey{}, recorder)
	return ctx.WithDelta(&agent.CommonContextDelta{
		InvocationContextDelta: &agent.InvocationContextDelta{Context: &base},
	}), recorder
}

func publishFailureInventory(ctx context.Context, producer any, inventory failureInventory) {
	if recorder, ok := ctx.Value(failureInventoryKey{}).(*failureInventoryRecorder); ok && recorder.producer == producer {
		recorder.mu.Lock()
		defer recorder.mu.Unlock()
		if recorder.published {
			recorder.inventory = failureInventory{}
		} else {
			recorder.inventory = inventory.clone()
		}
		recorder.published = true
	}
}
