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

package runner

import (
	"errors"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

// runPath names the two ways a run executes its root: the loop in
// Runner.run, or Runner.runNode.
type runPath int

const (
	classicPath runPath = iota // the loop in Runner.run
	nodePath                   // Runner.runNode
)

// resolveMode says what happens to a function-response message that cannot
// be attributed to exactly one invocation.
type resolveMode int

const (
	resolveLenient        resolveMode = iota // an unattributable message has no invocation
	resolveRejectMultiple                    // responses of several invocations are an error; an unmatched response has no invocation
	resolveFirst                             // responses of several invocations resume the first answered call's invocation; an unmatched response has no invocation
	resolveStrict                            // an unattributable message is an error
)

// rootIsNode reports whether the root is one adk-python runs as a node and
// Go resolves strictly on any runner: a Config.Node root or a root graph
// workflow (adk-python's Workflow is a node).
func (r *Runner) rootIsNode() bool {
	return r.rootNode != nil || r.isInGraphWorkflow(r.rootAgent)
}

// runsAsNode reports whether adk-python runs the root of a run on path as a
// node: a root it sends to Runner.runNode (an LLM agent) or one that
// rootIsNode reports.
func (r *Runner) runsAsNode(path runPath) bool {
	return path == nodePath || r.rootIsNode()
}

// resolutionMode returns how a run on path, given callerID by
// WithInvocationID ("" for none), resolves a function-response message.
// A root that runsAsNode reports is strict when the runner is resumable or
// callerID is set; otherwise a root that rootIsNode reports still refuses
// responses of several invocations, and an LLM-agent root resumes the first
// answered call's invocation for them, where adk-python raises. Any other
// root is strict when the runner is resumable and lenient otherwise.
// adk-python is strict for every root it runs as a node; adk-go keeps the
// other cases lenient where existing callers rely on an unmatched response
// starting a new invocation.
func (r *Runner) resolutionMode(path runPath, callerID string) resolveMode {
	strict := r.resumable
	if r.runsAsNode(path) {
		strict = r.resumable || callerID != ""
	}
	switch {
	case strict:
		return resolveStrict
	case r.rootIsNode():
		return resolveRejectMultiple
	case path == nodePath:
		return resolveFirst
	default:
		return resolveLenient
	}
}

// runInvocationID returns the final invocation ID of a run on path over
// sess, or "" when the runner must mint one. msg is the normalized message
// (nil allowed) and callerID the WithInvocationID value ("" for none).
//
// A message without function responses keeps callerID. A message whose
// function responses answer calls of one stored invocation continues that
// invocation, whatever callerID is. When the responses cannot be attributed
// to exactly one invocation, a run keeps callerID or returns an error as
// its mode (see resolutionMode) says; a strict run also refuses a message
// whose first response has no ID, except where adk-python runs the root as a
// node and the caller gave no ID (adk-python Runner._resolve_invocation_id
// and workflow/_node_runner_utils.py). The errors carry no caller data.
func (r *Runner) runInvocationID(sess session.Session, msg *genai.Content, callerID string, path runPath) (string, error) {
	responses := utils.FunctionResponses(msg)
	if len(responses) == 0 {
		return callerID, nil
	}
	mode := r.resolutionMode(path, callerID)
	if mode == resolveStrict && responses[0].ID == "" && (callerID != "" || !r.runsAsNode(path)) {
		return "", errors.New("function response id is required to resume an invocation")
	}
	id, err := resolveInvocationID(sess, msg, mode)
	if err != nil {
		return "", err
	}
	if id == "" {
		return callerID, nil
	}
	return id, nil
}

// retriesStoredTurn reports whether msg retries the user turn of the
// invocation invocationID: msg has no function response and the live events
// of sess already hold that invocation's user message. The retry is
// recognized by the ID alone; the contents are not compared, so a message
// that a plugin rewrote, or a different text, still counts (adk-python
// Runner._handle_new_message). An empty invocationID never retries.
func retriesStoredTurn(sess session.Session, invocationID string, msg *genai.Content) bool {
	if invocationID == "" || len(utils.FunctionResponses(msg)) > 0 {
		return false
	}
	return utils.FindUserMessageForInvocation(sess.Events(), invocationID) != nil
}

// rejectFunctionCalls returns an error when a part of msg carries a function
// call: a user turn may answer calls but never make one.
func rejectFunctionCalls(msg *genai.Content) error {
	if len(utils.FunctionCalls(msg)) > 0 {
		return errUserFunctionCall
	}
	return nil
}
