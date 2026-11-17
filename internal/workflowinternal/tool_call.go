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

package workflowinternal

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"strconv"
	"strings"
	"sync"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// ToolBody is the body of one tool call, as generatortool wraps a Func.
//
// ctx is the call's own context: FunctionCallID, Actions, State, Artifacts,
// Path, Branch, Session and ResumedInput work on it, and it is canceled when
// the run is canceled or the call produces a second output. emit sends an
// event to the user; it is safe for concurrent use. A non-nil result is the
// call's output; nil means none.
type ToolBody func(ctx agent.Context, emit func(*session.Event) error) (result any, err error)

// RunToolCall runs body as a child node of toolCtx's node named after toolName,
// on the tool branch and under the call run id of toolCtx.FunctionCallID(), and
// returns its output.
//
// toolCtx must be a tool context made by agent.NewToolContext inside a
// dynamic node activation (it carries the sub-scheduler), and body must be
// non-nil. The child node's name is toolName with every character outside
// [a-zA-Z0-9_] replaced by '_' (and a leading '_' when that is empty or starts
// with a digit); its events carry that name as their author. Its branch is
// "<toolCtx.Branch()>.<toolName>@<call id>", or "<toolName>@<call id>" when
// toolCtx has no branch. Its run id is the call id when workflow.WithRunID
// accepts it, and an escaped form of it otherwise; a call whose id was
// already answered in this invocation, or whose earlier run finished without
// its response reaching the model, runs under a run id of its own, so its body
// runs again.
//
// On success RunToolCall returns the output (nil for none) after the child's
// events were emitted in order. The SkipSummarization flag the body sets on
// its context is copied to toolCtx.Actions() on every return, including a
// pause and a failure. Errors:
//   - no sub-scheduler on toolCtx: an error wrapping
//     workflow.ErrInvalidRunNodeContext; body does not run.
//   - a second output: a *toolinternal.FailRunError whose text is
//     "Output already set. A node can produce at most one output." and which
//     matches workflow.ErrMultipleOutputs.
//   - the body requested input and returned workflow.ErrNodeInterrupted: a
//     *toolinternal.PausedError that matches workflow.ErrNodeInterrupted.
//   - the body returned workflow.ErrNodeInterrupted without requesting
//     input: a plain error that does not match it, since nothing can resume
//     such a call.
//   - any other error of the body: that error, unchanged.
//   - cancellation or any other failure of the run: an error wrapping it.
//
// RunToolCall is safe for concurrent use; two calls share only the session.
func RunToolCall(toolCtx agent.Context, toolName string, body ToolBody) (any, error) {
	if toolCtx.SubScheduler() == nil {
		return nil, fmt.Errorf("generator tool %q: %w", toolName, workflow.ErrInvalidRunNodeContext)
	}
	nodeName := toolNodeName(toolName)
	callID := toolCtx.FunctionCallID()
	branch := toolName + "@" + callID
	if base := toolCtx.Branch(); base != "" {
		branch = base + "." + branch
	}
	runID := runIDFor(callID, callSlot(toolCtx, toolName, nodeName, callID))

	call := &toolCall{
		toolName: toolName,
		nodeName: nodeName,
		callID:   callID,
		actions:  &session.EventActions{StateDelta: map[string]any{}, ArtifactDelta: map[string]int64{}},
	}
	node := workflow.NewEmittingFunctionNode(nodeName,
		func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
			// The output goes out as an event of its own, so the node never
			// emits a terminal event (which would turn a string into text).
			return nil, call.run(ctx, emit, body)
		},
		workflow.NodeConfig{},
	)
	out, err := workflow.RunNode[any](toolCtx, node, nil, workflow.WithOverrideBranch(branch), workflow.WithRunID(runID))
	if call.actions.SkipSummarization {
		toolCtx.Actions().SkipSummarization = true
	}
	return call.result(out, err)
}

// toolCall is the state of one RunToolCall. mu guards every field below it.
type toolCall struct {
	toolName string
	nodeName string
	callID   string
	// actions is the body's own EventActions; its deltas move onto the
	// next non-partial event.
	actions *session.EventActions

	mu        sync.Mutex
	base      agent.Context
	cancel    context.CancelFunc
	emit      func(*session.Event) error
	outputs   int
	violation error
	bodyErr   error
}

// run is the node's body: it runs body with the call's context and emits
// the result as the call's output event.
func (c *toolCall) run(nodeCtx agent.Context, emit func(*session.Event) error, body ToolBody) error {
	base, cancel := nodeCtx.WithAgentCancel()
	defer cancel()
	c.mu.Lock()
	c.base, c.cancel, c.emit = base, cancel, emit
	c.mu.Unlock()

	answers := map[string]any{}
	for id := range interruptIDs(base) {
		if v, ok := base.ResumedInput(id); ok {
			answers[id] = v
		}
	}
	// WithDelta returns the unwrapped context, on which Session, Path and
	// ResumedInput work, unlike on the tool context wrapper.
	callCtx := agent.NewToolContext(base, c.callID, c.actions, nil).
		WithDelta(&agent.CommonContextDelta{ResumeInputs: &answers})

	result, err := body(callCtx, c.send)
	if err != nil {
		c.mu.Lock()
		c.bodyErr = err
		c.mu.Unlock()
		// The workflow's failure event carries no deltas, so the writes
		// made after the last emit go out on an event of their own.
		if c.hasPendingDeltas() {
			if sendErr := c.send(session.NewEvent(base, base.InvocationID())); sendErr != nil {
				return errors.Join(err, sendErr)
			}
		}
		return err
	}
	if result != nil {
		ev := session.NewEvent(base, base.InvocationID())
		ev.Output = result
		return c.send(ev)
	}
	if c.hasPendingDeltas() {
		return c.send(session.NewEvent(base, base.InvocationID()))
	}
	return nil
}

// send is the emit function the body receives.
func (c *toolCall) send(ev *session.Event) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.violation != nil {
		return c.violation
	}
	if ev == nil {
		return fmt.Errorf("generator tool %q: emit: event is nil", c.toolName)
	}
	if err := c.base.Err(); err != nil {
		return fmt.Errorf("generator tool %q: emit: %w", c.toolName, err)
	}
	if ev.Output != nil {
		if c.outputs > 0 {
			c.violation = multipleOutputsError{}
			c.cancel()
			return c.violation
		}
		c.outputs++
	}
	ev.Author = c.nodeName
	ev.InvocationID = c.base.InvocationID()
	if ev.Branch == "" {
		ev.Branch = c.base.Branch()
	}
	if !ev.Partial {
		c.flushDeltas(ev)
	}
	return c.emit(ev)
}

// flushDeltas moves the pending state and artifact deltas onto ev; a
// pending value wins over the event's own value for the same key. Called
// under mu.
func (c *toolCall) flushDeltas(ev *session.Event) {
	if len(c.actions.StateDelta) > 0 {
		if ev.Actions.StateDelta == nil {
			ev.Actions.StateDelta = map[string]any{}
		}
		maps.Copy(ev.Actions.StateDelta, c.actions.StateDelta)
		clear(c.actions.StateDelta)
	}
	if len(c.actions.ArtifactDelta) > 0 {
		if ev.Actions.ArtifactDelta == nil {
			ev.Actions.ArtifactDelta = map[string]int64{}
		}
		maps.Copy(ev.Actions.ArtifactDelta, c.actions.ArtifactDelta)
		clear(c.actions.ArtifactDelta)
	}
}

func (c *toolCall) hasPendingDeltas() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.actions.StateDelta) > 0 || len(c.actions.ArtifactDelta) > 0
}

// result translates the outcome of workflow.RunNode into RunToolCall's
// result, in the order of the error list of RunToolCall's doc.
func (c *toolCall) result(out any, err error) (any, error) {
	c.mu.Lock()
	violation, bodyErr := c.violation, c.bodyErr
	c.mu.Unlock()
	switch {
	case violation != nil:
		return nil, &toolinternal.FailRunError{Err: violation}
	case errors.Is(err, workflow.ErrNodeInterrupted):
		return nil, &toolinternal.PausedError{Err: err}
	case errors.Is(bodyErr, workflow.ErrNodeInterrupted):
		// FunctionNode swallows the sentinel when no request for input was
		// emitted, so the run looks successful; nothing could resume it.
		return nil, fmt.Errorf("generator tool %q: handler returned workflow.ErrNodeInterrupted without requesting input", c.toolName)
	case bodyErr != nil:
		return nil, bodyErr
	case err != nil:
		return nil, fmt.Errorf("generator tool %q: %w", c.toolName, err)
	}
	return out, nil
}

// interruptIDs returns every interrupt id raised in this invocation by the
// node of ctx or a node below it.
func interruptIDs(ctx agent.Context) map[string]bool {
	ids := map[string]bool{}
	path, invocationID := ctx.Path(), ctx.InvocationID()
	for ev := range ctx.Session().Events().All() {
		if ev == nil || ev.NodeInfo == nil || ev.InvocationID != invocationID {
			continue
		}
		if p := ev.NodeInfo.Path; p != path && !strings.HasPrefix(p, path+"/") {
			continue
		}
		for _, id := range ev.LongRunningToolIDs {
			ids[id] = true
		}
	}
	return ids
}

// multipleOutputsError is the error of a call that produced a second output.
// Its text is adk-python's; it matches workflow.ErrMultipleOutputs.
type multipleOutputsError struct{}

func (multipleOutputsError) Error() string {
	return "Output already set. A node can produce at most one output."
}

func (multipleOutputsError) Unwrap() error { return workflow.ErrMultipleOutputs }

// toolNodeName is the node name of a tool call: toolName with every
// character outside [a-zA-Z0-9_] replaced by '_', with a leading '_' when
// that is empty or starts with a digit (adk-python
// tools/function_tool.py:446-449).
func toolNodeName(toolName string) string {
	var b strings.Builder
	for _, r := range toolName {
		if r == '_' || 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z' || '0' <= r && r <= '9' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	name := b.String()
	if name == "" || '0' <= name[0] && name[0] <= '9' {
		name = "_" + name
	}
	return name
}

// runIDEscaper escapes the characters an escaped run id may not hold raw:
// '/' and '@' (refused by workflow.WithRunID), '~' (the escape marker and
// slot separator) and '%' (the escape character).
var runIDEscaper = strings.NewReplacer("%", "%25", "/", "%2F", "@", "%40", "~", "%7E")

// runIDFor is the run id of call id in slot k: the id itself, or an escaped form with k.
//
// The id itself is used for slot 0 when workflow.WithRunID accepts it and it
// does not start with '~'. Otherwise the run id is "~" + the escaped id,
// followed by "~<k>" when k > 0. A plain run id never starts with '~' and an
// escaped one holds no '~' but the first and the separator, so two different
// (id, k) pairs never share a run id.
func runIDFor(id string, k int) string {
	if k == 0 && plainRunID(id) {
		return id
	}
	runID := "~" + runIDEscaper.Replace(id)
	if k > 0 {
		runID += "~" + strconv.Itoa(k)
	}
	return runID
}

// plainRunID reports whether id can be the run id as it is: workflow.WithRunID
// accepts it (non-empty, not all ASCII digits, no '/' or '@') and it does not
// start with '~'.
func plainRunID(id string) bool {
	if id == "" || strings.HasPrefix(id, "~") || strings.ContainsAny(id, "/@") {
		return false
	}
	return strings.ContainsFunc(id, func(r rune) bool { return r < '0' || r > '9' })
}

// callSlot is the slot of this call: the smallest k that is at least the
// number of earlier answered calls of id and whose node path the
// sub-scheduler would not serve from its cache.
//
// A served path would return an earlier run's output without running the
// body, where adk-python runs the body of every call that has no response
// (it builds a fresh scheduler per call, workflow/_dynamic_node_scheduler.py
// 722-735). Slot 0 is the path of a first call, of the replay of a paused
// call (its path ends with its interrupt, so it holds the call's answers)
// and of a confirmed call. A call whose id the model already got an answer
// for gets a higher slot, and so does a call whose run finished in a model
// step that a sibling call paused (the flow sends no responses for that
// step), so its body runs again on the replay.
func callSlot(toolCtx agent.Context, toolName, nodeName, id string) int {
	// The unwrapped context: Session and InvocationID work on it.
	base := toolCtx.WithDelta(&agent.CommonContextDelta{})
	prefix := toolCtx.SubScheduler().ParentPath() + "/" + nodeName + "@"
	served := servedPaths(base, prefix)
	k := answeredCalls(base, toolName, id)
	for served[prefix+runIDFor(id, k)] {
		k++
	}
	return k
}

// answeredCalls counts the function responses of this invocation for call
// id of tool toolName. The response event of a call that waits for
// confirmation does not count: the confirmed call runs its body for the
// first time. The name keeps out a user's answer to a request for input
// whose interrupt id equals the call id.
func answeredCalls(base agent.Context, toolName, id string) int {
	n := 0
	for ev := range base.Session().Events().All() {
		if ev == nil || ev.InvocationID != base.InvocationID() {
			continue
		}
		if _, placeholder := ev.Actions.RequestedToolConfirmations[id]; placeholder {
			continue
		}
		for _, fr := range ev.FunctionResponses() {
			if fr.ID == id && fr.Name == toolName {
				n++
			}
		}
	}
	return n
}

// workflowNodeOutcomeKey is the custom metadata key under which the workflow
// engine records a node run's outcome. It copies the unexported constant of
// workflow/persistence.go.
const workflowNodeOutcomeKey = "adk.workflow.node_outcome"

// servedPaths returns the paths under prefix that the sub-scheduler would serve from its cache.
//
// It follows the dynamic sub-scheduler's rehydration rule for one exact
// path: among the non-partial events of this invocation on that path, the
// last one that carries an output, an interrupt or a failure outcome
// decides, and the path is served when that event carries an output. The
// scheduler also drops cached paths for failures recorded on other paths;
// this function does not, which can only give a call a higher slot than it
// needs.
func servedPaths(base agent.Context, prefix string) map[string]bool {
	served := map[string]bool{}
	for ev := range base.Session().Events().All() {
		if ev == nil || ev.Partial || ev.NodeInfo == nil || ev.InvocationID != base.InvocationID() {
			continue
		}
		path := ev.NodeInfo.Path
		if !strings.HasPrefix(path, prefix) {
			continue
		}
		switch outcome, _ := ev.CustomMetadata[workflowNodeOutcomeKey].(string); {
		case len(ev.LongRunningToolIDs) > 0 || ev.RequestedInput != nil:
			delete(served, path)
		case ev.Output != nil:
			served[path] = true
		case ev.ErrorCode != "" || outcome == "failed" || outcome == "cancelled":
			delete(served, path)
		}
	}
	return served
}
