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
	"errors"
	"fmt"
)

var (
	// ErrNodeFailed wraps a child node's runtime failure after retries.
	ErrNodeFailed = errors.New("workflow: dynamic child failed")

	// ErrNodeInterrupted wraps a child node's HITL pause request.
	ErrNodeInterrupted = errors.New("workflow: dynamic child interrupted")

	// ErrInvalidRunNodeContext is returned by RunNode when ctx is not
	// the NodeContext of a currently-executing dynamic node.
	ErrInvalidRunNodeContext = errors.New("workflow: RunNode called outside a dynamic node")

	// ErrInvalidRunID rejects a custom run id that would collide
	// with the auto-counter: purely numeric ids are reserved.
	ErrInvalidRunID = errors.New("workflow: invalid run id")

	// ErrParallelHITLUnsupported rejects two children of one
	// activation interrupting concurrently — at most one pending HITL
	// per activation.
	ErrParallelHITLUnsupported = errors.New("workflow: parallel HITL is not supported")

	// ErrInputValidation indicates that the node input did not pass validation.
	ErrInputValidation = errors.New("workflow: input validation failed")
	// ErrOutputAlreadyDelegated rejects a second WithUseAsOutput
	// child in the same parent activation.
	ErrOutputAlreadyDelegated = errors.New("workflow: parent already has a use_as_output child")
)

// NodeRunError wraps a sentinel with the failing child's identity.
// Use errors.As to recover the fields.
type NodeRunError struct {
	// ChildName is the child's Node.Name(), e.g. "fixer_agent".
	ChildName string
	// ChildPath is the composite path, e.g. "code_workflow/fixer_agent@2".
	ChildPath string
	// RunID is the per-invocation identifier, auto-counter or
	// user-supplied via WithRunID.
	RunID string
	Cause error
}

// Error formats as "workflow: dynamic child <path>: <cause>".
func (e *NodeRunError) Error() string {
	if e == nil {
		return "<nil>"
	}
	path := e.ChildPath
	if path == "" {
		path = e.ChildName
	}
	if e.Cause == nil {
		return fmt.Sprintf("workflow: dynamic child %s: <nil cause>", path)
	}
	return fmt.Sprintf("workflow: dynamic child %s: %v", path, e.Cause)
}

func (e *NodeRunError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}
