// Copyright 2025 Google LLC
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

// Package tool defines the interfaces for tools that can be called by an agent.
// A tool is a piece of code that performs a specific task. You can either define
// your own custom tools or use built-in ones, for example, GoogleSearch.
package tool

import (
	"errors"
	"fmt"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/toolutils"
)

// ErrConfirmationRequired indicates that the tool requires confirmation.
var ErrConfirmationRequired = errors.New("requires confirmation, please approve or reject")

// ErrConfirmationRejected indicated that the tool call confirmation rejected.
var ErrConfirmationRejected = errors.New("call is rejected")

// Tool defines the interface for a callable tool.
type Tool interface {
	// Name returns the name of the tool.
	Name() string
	// Description returns a description of the tool.
	Description() string
	// IsLongRunning indicates whether the tool is a long-running operation,
	// which typically returns a resource id first and finishes the operation later.
	IsLongRunning() bool
}

// ResponseScheduler is implemented by a tool that asks the model to treat its
// function responses in a particular way. The LLM flow copies the value onto
// the Scheduling field of every genai.FunctionResponse it builds for the tool,
// whatever produced the result: the tool, a before-tool, after-tool or
// on-tool-error callback, or the confirmation gate. An empty value means unset:
// the field stays empty and is omitted from the response's JSON.
//
// Only models that support asynchronous function calling read the value (the
// Gemini Live API); others ignore it. In a Live run a set value also makes the
// tool non-blocking, which the Live flow decides, not this interface.
//
// tool.WithConfirmation and tool.WithToolNamePrefix keep the value of the tools
// they wrap.
type ResponseScheduler interface {
	// ResponseScheduling returns the tool's scheduling, or "" for none. It must
	// return the same value on every call and be safe for concurrent use.
	ResponseScheduling() genai.FunctionResponseScheduling
}

// SequentialRunner is implemented by a tool that wants each of its calls to
// start only after the calls before it in the same model turn have returned.
// ADK runs the calls of one model turn concurrently, and State has no
// ordering; a tool that reads and then writes state that its sibling calls
// also write, with no lock of its own, declares itself sequential to get
// the writes of the earlier calls.
//
// Before a sequential call's callbacks and the tool run, the flow waits until
// every earlier call of the turn has returned, whether those calls are
// sequential or not. If the call's context is done first, the call is answered
// with the error "wait for earlier function calls: context deadline exceeded"
// (or "... context canceled"), and neither its callbacks nor the tool run. The
// wait happens before any code of the tool, so a lock the tool takes cannot be
// held across it.
//
// A sequential tool must not wait for something that a later call of the
// same turn does: the later call is waiting for it. Only calls of a tool run
// through ADK's function-call path are sequenced; the calls of a streaming
// tool are not.
//
// tool.WithConfirmation and tool.WithToolNamePrefix keep the value of the
// tools they wrap.
type SequentialRunner interface {
	// RunsSequentially reports whether the tool's calls are sequential. It must
	// return the same value on every call and be safe for concurrent use.
	RunsSequentially() bool
}

// Toolset is an interface for a collection of tools. It allows grouping
// related tools together and providing them to an agent.
//
// A toolset that holds a resource, such as a network connection, implements
// [io.Closer]. When the application calls Close on the runner that runs the
// toolset's agent (Runner.Close in package runner), the runner calls Close on
// every such toolset reachable from the runner's root, once, one toolset
// after another, and gives each ten seconds. Close must release the resource
// and return, may be called more than once, and may be called while Tools
// runs. A toolset that wraps another toolset passes Close through to it, as
// [FilterToolset] and [WithConfirmation] do, and has an Unwrap() Toolset
// method that returns the wrapped toolset, so that a toolset shared by
// several wrappers is closed once.
type Toolset interface {
	// Name returns the name of the toolset.
	Name() string
	// Tools returns a list of tools in the toolset. The provided
	// ReadonlyContext can be used to dynamically determine which tools
	// to return based on the current invocation state.
	Tools(ctx agent.ReadonlyContext) ([]Tool, error)
}

// Predicate is a function which decides whether a tool should be exposed to LLM.
type Predicate func(ctx agent.ReadonlyContext, tool Tool) bool

// StringPredicate is a helper that creates a Predicate from a string slice.
// Deprecated: use AllowedToolsPredicate instead.
func StringPredicate(allowedTools []string) Predicate {
	return AllowedToolsPredicate(allowedTools)
}

// AllowedToolsPredicate returns a Predicate that allows only the tools with the given names.
func AllowedToolsPredicate(allowedTools []string) Predicate {
	m := make(map[string]bool)
	for _, t := range allowedTools {
		m[t] = true
	}

	return func(ctx agent.ReadonlyContext, tool Tool) bool {
		return m[tool.Name()]
	}
}

// FilterToolset returns a Toolset that filters the tools in the given Toolset
// using the given predicate. Close on the returned toolset closes toolset when
// that implements [io.Closer]. When toolset has a request hook, a
// ProcessRequest(agent.Context, *model.LLMRequest) error method, the returned
// toolset has one that calls it, so filtering keeps the toolset's own request
// processing.
func FilterToolset(toolset Toolset, predicate Predicate) Toolset {
	if toolset == nil {
		panic("toolset must not be nil")
	}
	if predicate == nil {
		panic("predicate must not be nil")
	}

	return &filteredToolset{
		toolset:   toolset,
		predicate: predicate,
	}
}

type filteredToolset struct {
	toolset   Toolset
	predicate Predicate
}

func (f *filteredToolset) Name() string {
	return f.toolset.Name()
}

func (f *filteredToolset) Tools(ctx agent.ReadonlyContext) ([]Tool, error) {
	tools, err := f.toolset.Tools(ctx)
	if err != nil {
		return nil, err
	}
	var filtered []Tool
	for _, tool := range tools {
		if f.predicate(ctx, tool) {
			filtered = append(filtered, tool)
		}
	}
	return filtered, nil
}

func (f *filteredToolset) Close() error { return utils.CloseIfCloser(f.toolset) }

func (f *filteredToolset) Unwrap() Toolset { return f.toolset }

func (f *filteredToolset) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return processToolsetRequest(f.toolset, ctx, req)
}

// processToolsetRequest runs the request hook of ts when ts has one, and does
// nothing otherwise. A toolset wrapper forwards its own ProcessRequest here, so
// the flow, which looks for the hook on the outermost toolset, still reaches
// the wrapped toolset's hook.
func processToolsetRequest(ts Toolset, ctx agent.Context, req *model.LLMRequest) error {
	rp, ok := ts.(requestProcessor)
	if !ok {
		return nil
	}
	return rp.ProcessRequest(ctx, req)
}

// defersResponse reports what t's DefersResponse answers, and false when t has
// no such method. A tool wrapper forwards the method through it, because the
// flow looks for it on the tool it dispatches to.
func defersResponse(t Tool) bool {
	d, ok := t.(interface{ DefersResponse() bool })
	return ok && d.DefersResponse()
}

// displaysResultOnSkipSummarization reports what t's
// DisplayResultOnSkipSummarization answers, and false when t has no such method.
func displaysResultOnSkipSummarization(t Tool) bool {
	d, ok := t.(interface{ DisplayResultOnSkipSummarization() bool })
	return ok && d.DisplayResultOnSkipSummarization()
}

// runsSequentially reports what t's RunsSequentially answers, and false when t
// is not a SequentialRunner.
func runsSequentially(t Tool) bool {
	s, ok := t.(SequentialRunner)
	return ok && s.RunsSequentially()
}

// responseScheduling reports what t's ResponseScheduling answers, and "" when
// t is not a ResponseScheduler.
func responseScheduling(t Tool) genai.FunctionResponseScheduling {
	if s, ok := t.(ResponseScheduler); ok {
		return s.ResponseScheduling()
	}
	return ""
}

// rawResultRunner mirrors adk/internal/toolinternal.RawResultTool. It is
// declared here rather than imported because toolinternal imports this
// package.
type rawResultRunner interface {
	RunRaw(ctx agent.Context, args any) (any, error)
}

// runRaw runs t through its RunRaw method when it has one, and through Run
// otherwise. A tool wrapper forwards RunRaw through it, because the flow calls
// RunRaw on the tool it dispatches to so that a function tool's *genai.Part
// and nil results are not converted.
func runRaw(t runnableTool, ctx agent.Context, args any) (any, error) {
	if r, ok := t.(rawResultRunner); ok {
		return r.RunRaw(ctx, args)
	}
	return t.Run(ctx, args)
}

// ConfirmationProvider defines a function that dynamically determines whether
// a specific tool execution requires user confirmation.
//
// It accepts the tool name and the input parameters as arguments.
// Returning true signals that the system must wait for Human-in-the-Loop (HITL)
// approval before proceeding with the execution.
//
// EXPERIMENTAL: ConfirmationProvider is experimental and not currently in scope for the v1.0 API.
type ConfirmationProvider func(toolName string, toolInput any) bool

// WithConfirmation wraps a toolset to inject confirmation logic in each tool.
//
// Every tool the ADK flow can execute is wrapped: tools that provide a
// FunctionDeclaration and a Run method, and streaming tools that provide a
// FunctionDeclaration and a RunStream method. Tools the flow never executes
// itself, such as the model-side built-ins in tool/geminitool, cannot be
// confirmed and are included in the returned Toolset without modification.
//
// Close on the returned toolset closes ts when that implements [io.Closer].
// The returned toolset's request hook calls the request hook of ts, when ts has
// one. A wrapped tool keeps the DefersResponse,
// DisplayResultOnSkipSummarization, ResponseScheduling and RunsSequentially
// answers of the tool it wraps.
//
// EXPERIMENTAL: WithConfirmation is experimental and not currently in scope for the v1.0 API.
func WithConfirmation(ts Toolset, requireConfirmation bool, requireConfirmationProvider ConfirmationProvider) Toolset {
	return &confirmationToolset{
		toolset:                     ts,
		requireConfirmation:         requireConfirmation,
		requireConfirmationProvider: requireConfirmationProvider,
	}
}

type confirmationToolset struct {
	toolset                     Toolset
	requireConfirmation         bool
	requireConfirmationProvider ConfirmationProvider
}

func (c *confirmationToolset) Name() string { return c.toolset.Name() }

func (c *confirmationToolset) Close() error { return utils.CloseIfCloser(c.toolset) }

func (c *confirmationToolset) Unwrap() Toolset { return c.toolset }

func (c *confirmationToolset) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return processToolsetRequest(c.toolset, ctx, req)
}

func (c *confirmationToolset) Tools(ctx agent.ReadonlyContext) ([]Tool, error) {
	tools, err := c.toolset.Tools(ctx)
	if err != nil {
		return nil, err
	}
	wrappedTools := make([]Tool, 0, len(tools))
	for _, t := range tools {
		// Streaming is checked first because the flow dispatches on
		// StreamingFunctionTool before FunctionTool. A tool that satisfies both
		// must be wrapped the way it will be called.
		switch rt := t.(type) {
		case streamingRunnableTool:
			wrappedTools = append(wrappedTools, &confirmationStreamingTool{
				streamingRunnableTool: rt,
				requireConfirmation:   c.requireConfirmation,
				provider:              c.requireConfirmationProvider,
			})
		case runnableTool:
			wrappedTools = append(wrappedTools, &confirmationTool{
				runnableTool:        rt,
				requireConfirmation: c.requireConfirmation,
				provider:            c.requireConfirmationProvider,
			})
		default:
			// The flow cannot execute this tool, so there is nothing to confirm.
			wrappedTools = append(wrappedTools, t)
		}
	}

	return wrappedTools, nil
}

// confirmCall applies the human-in-the-loop gate before a tool call runs.
//
// It returns nil to proceed, and a non-nil error when the call was rejected,
// when a confirmation request was raised and the caller must stop and wait, or
// when raising that request failed.
func confirmCall(ctx agent.Context, t Tool, args any, requireConfirmation bool, provider ConfirmationProvider) error {
	if confirmation := ctx.ToolConfirmation(); confirmation != nil {
		if !confirmation.Confirmed {
			return fmt.Errorf("error tool %q %w", t.Name(), ErrConfirmationRejected)
		}
		return nil
	}

	if provider != nil {
		requireConfirmation = provider(t.Name(), args)
	}
	if !requireConfirmation {
		return nil
	}

	if err := ctx.RequestConfirmation(
		fmt.Sprintf("Please approve or reject the tool call %s() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
			t.Name()), nil); err != nil {
		return err
	}
	ctx.Actions().SkipSummarization = true
	return fmt.Errorf("error tool %q %w", t.Name(), ErrConfirmationRequired)
}

// confirmationTool is a wrapper around a tool that adds confirmation logic.
// It implements tool.Tool and adk/internal/toolinternal.FunctionTool and adk/internal/toolinternal.RequestProcessor.
type confirmationTool struct {
	runnableTool
	requireConfirmation bool
	provider            ConfirmationProvider
}

type runnableTool interface {
	Tool
	Declaration() *genai.FunctionDeclaration
	Run(ctx agent.Context, args any) (result map[string]any, err error)
}

// streamingRunnableTool mirrors adk/internal/toolinternal.StreamingFunctionTool.
// It is declared here rather than imported because toolinternal imports this
// package.
type streamingRunnableTool interface {
	Tool
	Declaration() *genai.FunctionDeclaration
	RunStream(ctx agent.Context, args any) iter.Seq2[string, error]
}

func (t *confirmationTool) Declaration() *genai.FunctionDeclaration {
	return t.runnableTool.Declaration()
}

func (t *confirmationTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if rp, ok := t.runnableTool.(interface {
		ProcessRequest(ctx agent.Context, req *model.LLMRequest) error
	}); ok {
		_, existedBefore := req.Tools[t.Name()]
		if err := rp.ProcessRequest(ctx, req); err != nil {
			return err
		}
		// If the inner tool packed itself into req.Tools during ProcessRequest,
		// replace it with the confirmation wrapper so confirmationTool.Run is invoked.
		if !existedBefore && req.Tools != nil && req.Tools[t.Name()] != nil {
			req.Tools[t.Name()] = t
			return nil
		}
	}
	return toolutils.PackTool(req, t)
}

func (t *confirmationTool) DefersResponse() bool { return defersResponse(t.runnableTool) }

func (t *confirmationTool) DisplayResultOnSkipSummarization() bool {
	return displaysResultOnSkipSummarization(t.runnableTool)
}

func (t *confirmationTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	// Check for Human-in-the-Loop confirmation.
	if err := confirmCall(ctx, t, args, t.requireConfirmation, t.provider); err != nil {
		return nil, err
	}
	return t.runnableTool.Run(ctx, args)
}

// RunRaw applies the confirmation gate, then runs the wrapped tool's RunRaw,
// or its Run when it has none.
func (t *confirmationTool) RunRaw(ctx agent.Context, args any) (any, error) {
	if err := confirmCall(ctx, t, args, t.requireConfirmation, t.provider); err != nil {
		return nil, err
	}
	return runRaw(t.runnableTool, ctx, args)
}

func (t *confirmationTool) ResponseScheduling() genai.FunctionResponseScheduling {
	return responseScheduling(t.runnableTool)
}

func (t *confirmationTool) RunsSequentially() bool { return runsSequentially(t.runnableTool) }

// confirmationStreamingTool is the streaming counterpart of confirmationTool.
// It embeds the streamingRunnableTool interface rather than the concrete tool so
// that only the streaming surface is promoted: a tool that also has a Run method
// must not reach the flow through this wrapper unguarded.
type confirmationStreamingTool struct {
	streamingRunnableTool
	requireConfirmation bool
	provider            ConfirmationProvider
}

func (t *confirmationStreamingTool) Declaration() *genai.FunctionDeclaration {
	return t.streamingRunnableTool.Declaration()
}

func (t *confirmationStreamingTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if rp, ok := t.streamingRunnableTool.(interface {
		ProcessRequest(ctx agent.Context, req *model.LLMRequest) error
	}); ok {
		_, existedBefore := req.Tools[t.Name()]
		if err := rp.ProcessRequest(ctx, req); err != nil {
			return err
		}
		// If the inner tool packed itself into req.Tools during ProcessRequest,
		// replace it with the confirmation wrapper so this wrapper's RunStream
		// is invoked.
		if !existedBefore && req.Tools != nil && req.Tools[t.Name()] != nil {
			req.Tools[t.Name()] = t
			return nil
		}
	}
	return toolutils.PackTool(req, t)
}

func (t *confirmationStreamingTool) DefersResponse() bool {
	return defersResponse(t.streamingRunnableTool)
}

func (t *confirmationStreamingTool) DisplayResultOnSkipSummarization() bool {
	return displaysResultOnSkipSummarization(t.streamingRunnableTool)
}

func (t *confirmationStreamingTool) ResponseScheduling() genai.FunctionResponseScheduling {
	return responseScheduling(t.streamingRunnableTool)
}

func (t *confirmationStreamingTool) RunStream(ctx agent.Context, args any) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		if err := confirmCall(ctx, t, args, t.requireConfirmation, t.provider); err != nil {
			yield("", err)
			return
		}
		for chunk, err := range t.streamingRunnableTool.RunStream(ctx, args) {
			if !yield(chunk, err) {
				return
			}
		}
	}
}
