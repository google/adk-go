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
	"errors"
	"fmt"
	"time"
)

var (
	// ErrNodeFailed wraps a child node's runtime failure after retries.
	ErrNodeFailed = errors.New("workflow: dynamic child failed")

	// ErrNodeInterrupted wraps a child node's HITL pause request.
	ErrNodeInterrupted = errors.New("workflow: dynamic child interrupted")

	// ErrNodeWaitingForOutput is a pause with no interrupt ID: a
	// WaitForOutput child finished without output, so the parent parks
	// instead of completing. Wraps ErrNodeInterrupted so existing
	// errors.Is(ErrNodeInterrupted) checks still treat it as a pause.
	ErrNodeWaitingForOutput = fmt.Errorf("%w: waiting for output", ErrNodeInterrupted)

	// ErrInvalidRunNodeContext is returned by RunNode when ctx is the
	// context a workflow node that is not a dynamic node runs under, or
	// carries neither a session nor a function call, and by the scheduler
	// RunNode builds for a context without a sub-scheduler when the child
	// asks for input.
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
	ErrOutputAlreadyDelegated = errors.New("workflow: parent already has a use_as_output delegate")

	// ErrStateSchemaUnusable is returned, wrapped, for a state schema that ADK cannot check writes against at
	// run time: one whose references leave its own document (a remote $ref that its author resolved with a
	// loader). Match it with errors.Is; the wrapped cause is the schema library's text, which names the
	// reference, and the wrapping adds the node.
	ErrStateSchemaUnusable = errors.New("workflow: state schema cannot be validated at run time")
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

// NodeTimeoutError reports that a node with NodeConfig.Timeout set ran past
// that timeout. The workflow ends the node's attempt and reports this error as
// the attempt's failure: RetryConfig sees it, and once the retries are spent it
// is the run's error. Find it with errors.As, also through wrapping errors.
//
// The error names the node whose own timeout elapsed. When a node with a
// timeout sits inside another node with a timeout (a nested workflow, or an
// AgentNode over a workflow agent), the error names the inner node when its
// timeout elapses and it returns before the enclosing node's deadline passes;
// when the enclosing deadline has passed by the time the inner node returns,
// for example because the inner node takes long to stop, the error names the
// enclosing node. A deadline set by an enclosing node or by the caller of the
// workflow is not the inner node's timeout, and fails it with
// context.DeadlineExceeded. A node that runs its own code (not a
// nested workflow, a ParallelWorker or an AgentNode over a workflow agent)
// and returns an error matching context.DeadlineExceeded before its own
// timeout elapses also fails with a NodeTimeoutError, because the timeout it
// reports is taken to be its own.
//
// errors.Is(err, context.DeadlineExceeded) is true for every NodeTimeoutError,
// so code that checks for a deadline keeps working. Unwrap returns the error
// that signaled the timeout.
//
// The Error text is
//
//	node "slow" timed out after 0.05 seconds
//
// with the node's name quoted as by %q and the timeout in seconds as by %g.
// A NodeTimeoutError is immutable once built and safe for concurrent use.
type NodeTimeoutError struct {
	// NodeName is the Name of the node whose timeout elapsed.
	NodeName string
	// Timeout is that node's NodeConfig.Timeout.
	Timeout time.Duration

	cause error // the body's timeout error; nil when the node's own deadline passed
}

// Error formats as: node "SlowNode" timed out after 0.1 seconds. A nil
// receiver formats as "<nil>".
func (e *NodeTimeoutError) Error() string {
	if e == nil {
		return "<nil>"
	}
	return fmt.Sprintf("node %q timed out after %g seconds", e.NodeName, e.Timeout.Seconds())
}

// Unwrap returns the error that signaled the timeout: context.DeadlineExceeded
// when the node's own deadline passed, or the error the node's body returned
// when it reported a timeout of its own. It returns nil for a nil receiver.
func (e *NodeTimeoutError) Unwrap() error {
	if e == nil {
		return nil
	}
	if e.cause == nil {
		return context.DeadlineExceeded
	}
	return e.cause
}
