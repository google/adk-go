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

import "fmt"

// RunNodeOption configures a single RunNode call. Today only WithRunID
// exists; the option set will grow to mirror adk-python's ctx.run_node
// kwargs (use_as_output for output delegation, override_isolation_scope,
// per-call timeout/retry overrides, etc.) as those features land.
type RunNodeOption func(*runNodeOptions)

type runNodeOptions struct {
	customRunID string
}

// WithRunID overrides the auto-generated counter with a stable
// user-supplied identifier — useful for reorderable lists keyed by
// e.g. an order id. id must be non-empty, contain at least one
// non-digit character (purely numeric ids collide with the
// auto-counter), and exclude the composite-path separators '/' and
// '@'. Violations surface as ErrInvalidRunID from RunNode.
//
// Mirrors adk-python's run_id kwarg
// (https://adk.dev/graphs/dynamic/#custom-execution-ids).
func WithRunID(id string) RunNodeOption {
	return func(o *runNodeOptions) { o.customRunID = id }
}

// RunNode schedules child as a sub-node of the currently-executing
// dynamic node and returns its typed output. ctx must be the
// NodeContext passed into the enclosing dynamic node's body.
//
// On failure:
//   - errors.Is(err, ErrNodeInterrupted): child paused for HITL.
//   - errors.Is(err, ErrNodeFailed): child errored after retries;
//     errors.As recovers *NodeRunError with diagnostics.
//   - ErrInvalidRunNodeContext, ErrInvalidRunID: misuse.
//   - ctx.Err(): parent cancellation.
func RunNode[OUT any](ctx NodeContext, child Node, input any, opts ...RunNodeOption) (OUT, error) {
	var zero OUT

	nc, ok := ctx.(*nodeContext)
	if !ok || nc.subScheduler == nil {
		return zero, ErrInvalidRunNodeContext
	}

	var o runNodeOptions
	for _, opt := range opts {
		opt(&o)
	}

	rawOut, err := nc.subScheduler.runNode(child, input, o.customRunID)
	if err != nil {
		return zero, err
	}
	if rawOut == nil {
		return zero, nil
	}
	typed, ok := rawOut.(OUT)
	if !ok {
		return zero, fmt.Errorf("workflow.RunNode: child %q output type %T does not satisfy expected %T",
			child.Name(), rawOut, zero)
	}
	return typed, nil
}
