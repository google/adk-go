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

// Package flow exposes the LLM-flow runtime as a public API. Mirrors
// adk-python's google.adk.flows.llm_flows subsystem.
//
// A Flow is a pipeline of request processors → model call → response
// processors → tool dispatch, looping until the model produces a final
// response. Tools, callbacks, and processors are configured per-flow.
//
// Phase 7 lifts what was previously inside internal/llminternal into a
// stable, public surface via type aliases. Existing callers that import
// internal/llminternal continue to compile because the underlying types
// are unchanged — they live in internal and are re-exported here.
//
// The processor pipeline is the public extension point. Custom request
// or response processors can be inserted into Flow.RequestProcessors or
// Flow.ResponseProcessors.
package flow

import (
	"iter"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// Flow is the LLM-flow runtime. A Flow runs a configured pipeline of
// request processors, then calls the model, then runs response
// processors and tool dispatch — looping until a final response is
// produced. Mirrors adk-python's BaseLlmFlow.
type Flow = llminternal.Flow

// BaseFlow is an alias for Flow that mirrors Python's naming. New code
// should prefer Flow; BaseFlow is provided for symmetry with the Python
// surface and for callers porting from Python.
type BaseFlow = Flow

// RequestProcessor is the function signature for processors that mutate
// the LLMRequest before the model call. Mirrors Python's
// BaseLlmRequestProcessor.run_async (yields Events as a side effect).
type RequestProcessor = func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]

// ResponseProcessor is the function signature for processors that
// inspect or mutate the LLMResponse after the model call returns.
type ResponseProcessor = func(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) error

// BeforeModelCallback runs before each model call. Returning a non-nil
// LLMResponse short-circuits the call.
type BeforeModelCallback = llminternal.BeforeModelCallback

// AfterModelCallback runs after each model call. Returning a non-nil
// LLMResponse replaces the model's response.
type AfterModelCallback = llminternal.AfterModelCallback

// OnModelErrorCallback runs when the model returns an error. Returning a
// non-nil LLMResponse turns the error into a successful response.
type OnModelErrorCallback = llminternal.OnModelErrorCallback

// BeforeToolCallback runs before a tool's Run method. Returning a
// non-nil result short-circuits the tool execution.
type BeforeToolCallback = llminternal.BeforeToolCallback

// AfterToolCallback runs after a tool's Run method, with the tool's
// result. Returning a non-nil map replaces the result.
type AfterToolCallback = llminternal.AfterToolCallback

// OnToolErrorCallback runs when a tool returns an error.
type OnToolErrorCallback = llminternal.OnToolErrorCallback

// DefaultRequestProcessors returns a copy of the default request-side
// pipeline used by AutoFlow. Callers can prepend / append custom
// processors before assigning to Flow.RequestProcessors.
func DefaultRequestProcessors() []RequestProcessor {
	src := llminternal.DefaultRequestProcessors
	out := make([]RequestProcessor, len(src))
	copy(out, src)
	return out
}

// DefaultResponseProcessors returns a copy of the default response-side
// pipeline used by AutoFlow.
func DefaultResponseProcessors() []ResponseProcessor {
	src := llminternal.DefaultResponseProcessors
	out := make([]ResponseProcessor, len(src))
	copy(out, src)
	return out
}

// AutoFlow returns a Flow preconfigured with the full default processor
// pipeline (instructions, contents, tools, agent transfer, output schema,
// confirmation, NL planning, code execution, etc.). This is the
// configuration LlmAgent uses by default.
//
// The returned Flow is mutable — callers can prepend / append processors
// or callbacks before invoking Run.
func AutoFlow(m model.LLM, tools ...tool.Tool) *Flow {
	return &Flow{
		Model:              m,
		Tools:              append([]tool.Tool(nil), tools...),
		RequestProcessors:  DefaultRequestProcessors(),
		ResponseProcessors: DefaultResponseProcessors(),
	}
}

// SingleFlow returns a Flow with the request-side processors but no
// looping — the LlmAgent runtime decides when to stop. In practice
// SingleFlow is the same shape as AutoFlow but documented as the
// "single-call, no automatic tool loop" preset for callers that prefer
// to drive iteration externally.
//
// Mirrors Python's SingleFlow naming. The looping behavior is enforced
// at the caller layer in adk-go because Flow.Run already terminates on a
// final response.
func SingleFlow(m model.LLM, tools ...tool.Tool) *Flow {
	return AutoFlow(m, tools...)
}
