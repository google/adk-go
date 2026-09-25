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

package tool

import (
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
)

// FunctionTool is a tool executed locally in response to a model function call.
// Run receives the function call's arguments as a map[string]any. Its result is
// sent to the model as the function response; errors are handled by the agent's
// tool callbacks before being returned as an error response.
//
// Declaration describes the function to the model. A nil declaration keeps the
// tool callable without advertising it to the model. ProcessRequest registers the
// tool and prepares its declaration for each model request; see [RequestProcessor].
type FunctionTool interface {
	Tool
	RequestProcessor
	Declaration() *genai.FunctionDeclaration
	Run(ctx agent.Context, args any) (result map[string]any, err error)
}

// StreamingFunctionTool is a tool that yields text chunks in response to a
// model function call. RunStream receives the arguments as a map[string]any.
// In live sessions, chunks are sent as they arrive; otherwise, they are joined
// into the function response's "result" field. Streaming execution takes
// precedence when a tool implements both StreamingFunctionTool and [FunctionTool].
//
// Declaration and request processing follow [FunctionTool].
type StreamingFunctionTool interface {
	Tool
	RequestProcessor
	Declaration() *genai.FunctionDeclaration
	RunStream(ctx agent.Context, args any) iter.Seq2[string, error]
}

// RequestProcessor customizes a model request before it is sent. Both tools and
// toolsets may implement it. A callable tool must register itself and its
// declaration through ProcessRequest, typically using
// [google.golang.org/adk/v2/tool/toolutils.PackTool]. A processor may omit
// registration to keep the tool unavailable for that request.
type RequestProcessor interface {
	ProcessRequest(ctx agent.Context, req *model.LLMRequest) error
}

// ResponseDeferrer allows a tool to defer its function response to an external
// producer. When DefersResponse returns true and the tool returns a nil result,
// the framework does not emit an immediate function response.
type ResponseDeferrer interface {
	DefersResponse() bool
}

// SkipSummarizationResultDisplayer opts a tool into displaying its result as
// text when the tool sets SkipSummarization on its context's actions. When
// DisplayResultOnSkipSummarization returns true, the framework adds displayable
// result text to the final function response event. Tools whose results are
// internal acknowledgements should leave this capability unimplemented or
// return false.
type SkipSummarizationResultDisplayer interface {
	DisplayResultOnSkipSummarization() bool
}

// Wrapper decorates another tool while preserving capabilities it does not
// implement itself. The framework uses [As] to discover capabilities through
// wrappers, including capabilities added in later releases.
//
// A wrapper must preserve the wrapped tool's Name so request registration and
// function declarations continue to refer to the same tool. Unwrap must produce
// a finite chain ending in a non-wrapper or nil; cycles are not supported.
//
// To override a capability, implement its complete interface. For example, a
// wrapper overriding Run can embed [FunctionTool], implement Run, and return the
// embedded tool from Unwrap. Embedding only [Tool] and adding Run does not
// implement FunctionTool because Declaration and ProcessRequest are also required.
type Wrapper interface {
	Tool
	Unwrap() Tool
}

// As returns the first value in t's [Wrapper] chain that implements T, starting
// with t itself. It returns the zero value of T and false if none does. T is
// usually a capability interface, such as [FunctionTool] or [RequestProcessor].
// Callers that inspect tool capabilities should use As so wrappers preserve
// optional capabilities without forwarding each method.
//
// A capability implemented by an outer wrapper takes precedence over the same
// capability on an inner tool. As resolves complete interfaces; it does not
// combine methods from different values in the chain. A nil t has no
// capabilities. As does not detect cycles or typed nil pointers.
func As[T any](t Tool) (T, bool) {
	for t != nil {
		if capability, ok := t.(T); ok {
			return capability, true
		}
		wrapper, ok := t.(Wrapper)
		if !ok {
			break
		}
		t = wrapper.Unwrap()
	}
	var zero T
	return zero, false
}
