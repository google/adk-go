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

package flowtool

import (
	"context"
	"errors"
	"fmt"

	"google.golang.org/adk/agent"
)

// validate enforces structural limits and resolves agent names against the
// catalog. Errors returned here are surfaced verbatim to the LLM so it can
// self-correct.
func (t *flowTool) validate(s *Spec) error {
	if t.maxNodes > 0 {
		if n := CountNodes(s); n > t.maxNodes {
			return fmt.Errorf("flowtool: spec has %d nodes, exceeds max_nodes=%d", n, t.maxNodes)
		}
	}
	if t.maxDepth > 0 {
		if d := Depth(s); d > t.maxDepth {
			return fmt.Errorf("flowtool: spec depth %d exceeds max_depth=%d", d, t.maxDepth)
		}
	}
	if t.maxParallelWidth > 0 {
		if w := MaxParallelWidth(s); w > t.maxParallelWidth {
			return fmt.Errorf("flowtool: parallel width %d exceeds max_parallel_width=%d", w, t.maxParallelWidth)
		}
	}

	var missing []string
	Walk(s, func(node *Spec) {
		if node.Type != KindAgent {
			return
		}
		if _, ok := t.catalog[node.Agent]; !ok {
			missing = append(missing, node.Agent)
		}
	})
	if len(missing) > 0 {
		return fmt.Errorf("flowtool: unknown agents in catalog: %v (available: %v)", missing, t.catalogNames())
	}
	return nil
}

func (t *flowTool) catalogNames() []string {
	names := make([]string, 0, len(t.catalog))
	for name := range t.catalog {
		names = append(names, name)
	}
	return names
}

// recursionKey is the context.Context key that tracks how many run_flow
// invocations are nested in the current call chain.
type recursionKey struct{}

// recursionDepth returns the current recursion depth pulled from ctx, or 0.
func recursionDepth(ctx context.Context) int {
	v := ctx.Value(recursionKey{})
	if v == nil {
		return 0
	}
	d, _ := v.(int)
	return d
}

// withRecursion returns a context whose recursion counter is one higher.
func withRecursion(ctx context.Context, d int) context.Context {
	return context.WithValue(ctx, recursionKey{}, d)
}

// errRecursionExceeded is returned when the recursion guard trips.
var errRecursionExceeded = errors.New("flowtool: recursion depth exceeded")

// resolveAgent returns the agent registered in the catalog for the given
// name, or an error.
func (t *flowTool) resolveAgent(name string) (agent.Agent, error) {
	a, ok := t.catalog[name]
	if !ok {
		return nil, fmt.Errorf("flowtool: agent %q not in catalog", name)
	}
	return a, nil
}
