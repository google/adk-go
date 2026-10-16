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

// Package toolinternal retains compatibility interfaces for tool capabilities.
// Dispatch must use tool.As to preserve capabilities through wrapper chains,
// including when introducing new capabilities.
package toolinternal

import (
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
)

// Keep the original method set and defined type: loadmemorytool.New exposes it
// in its return type, and request processors may register execution-only adapters.
type FunctionTool interface {
	tool.Tool
	Declaration() *genai.FunctionDeclaration
	Run(ctx agent.Context, args any) (map[string]any, error)
}

type StreamingFunctionTool interface {
	tool.Tool
	Declaration() *genai.FunctionDeclaration
	RunStream(ctx agent.Context, args any) iter.Seq2[string, error]
}

type RequestProcessor interface {
	tool.RequestProcessor
}

type ResponseDeferrer interface {
	tool.ResponseDeferrer
}

type SkipSummarizationResultDisplayer interface {
	tool.SkipSummarizationResultDisplayer
}
