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

// Package tool defines internal-only interfaces and logic for tools.
package toolinternal

import (
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

type FunctionTool interface {
	tool.Tool
	Declaration() *genai.FunctionDeclaration
	Run(ctx agent.Context, args any) (result map[string]any, err error)
}

type StreamingFunctionTool interface {
	tool.Tool
	Declaration() *genai.FunctionDeclaration
	RunStream(ctx agent.Context, args any) iter.Seq2[string, error]
}

// ConfirmationGatedStreamingTool is a StreamingFunctionTool whose
// human-in-the-loop confirmation gate can be applied apart from its stream.
//
// In live mode the flow answers a streaming call with a pending status and runs
// the stream in a background goroutine. It calls CheckConfirmation on the
// calling goroutine first, so that a call the gate stops is answered with an
// error rather than a pending status, and so that the confirmation request the
// gate records on the tool context is there before the function response event
// is built from it.
type ConfirmationGatedStreamingTool interface {
	StreamingFunctionTool
	// CheckConfirmation applies the confirmation gate to one call. It returns
	// nil when the call may run, and otherwise the error RunStream would yield
	// for the call: one wrapping tool.ErrConfirmationRequired after recording a
	// confirmation request on ctx, one wrapping tool.ErrConfirmationRejected, or
	// the error that stopped the check.
	CheckConfirmation(ctx agent.Context, args any) error
	// RunStreamUngated runs the call's stream without applying the gate again.
	// Call it only after CheckConfirmation has returned nil for the same call.
	RunStreamUngated(ctx agent.Context, args any) iter.Seq2[string, error]
}

type RequestProcessor interface {
	ProcessRequest(ctx agent.Context, req *model.LLMRequest) error
}

// ResponseDeferrer allows to skip generation of the FR by the tool.
// Used in the cases when FR is generated externally (e.g. TaskAgentTool)
type ResponseDeferrer interface {
	DefersResponse() bool
}

// SkipSummarizationResultDisplayer is implemented by tools whose result
// should still be shown to the user as text when SkipSummarization causes
// the agent loop to end on their function response event.
//
// SkipSummarization is set for two opposite reasons: agenttool sets it
// because the sub-agent already produced the final answer, which is meant to
// be seen; other tools (UI/widget tools, pending-confirmation flows) set it
// to suppress an internal acknowledgement that was never meant to be shown.
// Only tools that implement this interface, with DisplayResultOnSkipSummarization
// returning true, get their result surfaced as a visible text part.
type SkipSummarizationResultDisplayer interface {
	DisplayResultOnSkipSummarization() bool
}
