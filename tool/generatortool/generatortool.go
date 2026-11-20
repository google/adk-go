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

// Package generatortool provides a function tool whose handler can send events
// to the user while it runs and then produce one result for the model.
//
// A generator tool is the Go form of an adk-python tool written as a generator
// function. Its handler, a [Func], receives an emit function besides its
// arguments. Every event it emits is persisted in order and shown to the user
// before the function response, on the call's own branch
// "<caller branch>.<tool name>@<function call id>", so the model's later
// prompts leave those events out. The handler's return value, or the Output of
// one emitted event, is the function response the model sees. A call has at
// most one output: a second one fails the run with the error
// "Output already set. A node can produce at most one output.", which matches
// [workflow.ErrMultipleOutputs].
//
// A generator tool works in non-live runs only. Each call runs as a child
// workflow node of the calling agent. Its events are shown and it can pause
// for input when the agent runs on the runner's node path, in a workflow's
// agent node, in that node run as a dynamic child, or under
// llmagent.RunLLMAgentAsNode in a workflow node. In a workflow's agent node,
// in that node run as a dynamic child, and for a single_turn agent under
// llmagent.RunLLMAgentAsNode, the user's answer does not run the call again.
// Under a root agent that is neither an LLM agent nor a workflow, the handler
// runs and its result is the function response, and its state and artifact
// changes go out with the function response, but the events it emits are not
// shown to the user and it cannot pause for input.
//
// A handler pauses for input by calling [workflow.ResumeOrRequestInput] with
// its context and emit function and returning that function's error. The run
// then ends without an error, waiting for the user's answer. When the user
// answers, the call runs again from the top of the handler, and this time
// workflow.ResumeOrRequestInput returns the answer, so everything a handler
// does before a pause must be safe to repeat.
//
// Argument validation, the confirmation gate, the tool callbacks and the
// declaration work as for a tool made by [functiontool.New].
package generatortool

import (
	"fmt"
	"reflect"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/workflowinternal"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// Func is the handler of a generator tool.
//
// The caller (the tool) provides ctx, a tool call context for this one call:
// Actions, State, FunctionCallID, ResumedInput, Session, Path and Branch work.
// args are the arguments after the validation and conversion functiontool does
// for any function tool. emit sends an event to the user.
//
// ctx.Done is closed when the run is canceled and when the call has produced a
// second output. emit is safe for concurrent use. It stamps Author (the tool's
// name), InvocationID and Branch (the call's branch) and, on a non-partial
// event, moves pending ctx.State and artifact changes onto the event. An event
// whose Output is non-nil is the call's output. An event the handler builds
// leaves NodeInfo nil: the scheduler stamps it.
//
// emit returns an error and sends nothing when ctx is canceled, when ev is nil,
// and when ev would be a second output. After a second output, every later emit
// returns that same error and sends nothing, and ctx is canceled, so a handler
// that ignores the error cannot add events to a call that has already failed;
// the handler must return.
//
// The result is the call's output. A nil interface, pointer, map or slice means
// no output; any other value, including the zero value of a struct, is an
// output. A non-nil result after an emitted output fails the call.
//
// To pause for input, call workflow.ResumeOrRequestInput with ctx and emit and
// return its error. The call restarts from the top of the handler once the
// user answers, so everything before the pause must be repeatable; so does a
// call that finished in the same model response as a call that paused, and its
// events are sent again. Returning workflow.ErrNodeInterrupted without having
// emitted a request for input is a handler error: nothing could answer it.
//
// An error other than a pause is the call's error; callbacks and the flow treat
// it as they treat any function tool's error.
type Func[TArgs, TResult any] func(ctx agent.Context, args TArgs, emit func(*session.Event) error) (TResult, error)

// New returns a tool that runs handler as a node of the calling agent's node.
//
// cfg is functiontool.Config with the same meaning as for functiontool.New:
// InputSchema, OutputSchema, IsLongRunning, RequireConfirmation,
// RequireConfirmationProvider, ResponseScheduling, Sequential and
// CustomMetadata apply to the tool unchanged. With Sequential, a call's
// callbacks and handler start only after every earlier call of its model
// turn has returned (see tool.SequentialRunner). cfg.Name must be non-empty.
// The declaration's response schema is the one functiontool.New gives for
// TResult (or cfg.OutputSchema): for any it is the schema true, which accepts
// every value. TArgs must be a struct, a map, or a pointer to one, as for
// functiontool.New.
//
// New returns an error wrapping functiontool.ErrInvalidArgument for an empty
// cfg.Name, a bad TArgs or a nil handler, and the errors functiontool.New
// returns for a bad schema. The result is non-nil exactly when the error is
// nil.
//
// The tool is safe for concurrent use if handler is. Where the call's events
// are not shown (see the package doc), handler runs on every call with no
// resumed input. Its emit keeps of each non-partial event the Output, as the
// call's output, and the state and artifact deltas, which move onto the
// function response event; it drops the rest. A request for input it does
// not send: emit returns an error wrapping workflow.ErrInvalidRunNodeContext.
func New[TArgs, TResult any](cfg functiontool.Config, handler Func[TArgs, TResult]) (tool.Tool, error) {
	if cfg.Name == "" {
		return nil, fmt.Errorf("generatortool: Config.Name is empty: %w", functiontool.ErrInvalidArgument)
	}
	if handler == nil {
		return nil, fmt.Errorf("generatortool: handler is nil: %w", functiontool.ErrInvalidArgument)
	}
	run := func(ctx agent.Context, args TArgs) (TResult, error) {
		out, err := workflowinternal.RunToolCall(ctx, cfg.Name, func(c agent.Context, emit func(*session.Event) error) (any, error) {
			res, err := handler(c, args, emit)
			if err != nil {
				return nil, err
			}
			if isNil(res) {
				return nil, nil
			}
			return res, nil
		})
		if err != nil {
			var zero TResult
			return zero, err
		}
		return resultOf[TResult](out)
	}
	return functiontool.New[TArgs, TResult](cfg, run)
}

// resultOf is the call's output as a TResult: the zero value for no output.
// RunToolCall returns the live value of this call's own run, so any other type
// is a broken invariant and fails the call.
func resultOf[TResult any](out any) (TResult, error) {
	var zero TResult
	if out == nil {
		return zero, nil
	}
	v, ok := out.(TResult)
	if !ok {
		return zero, fmt.Errorf("generatortool: output is %T, want %v", out, reflect.TypeFor[TResult]())
	}
	return v, nil
}

// isNil reports whether v is nil, or a nil pointer, map, slice or interface.
func isNil(v any) bool {
	rv := reflect.ValueOf(v)
	if !rv.IsValid() {
		return true
	}
	switch rv.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Interface:
		return rv.IsNil()
	}
	return false
}
