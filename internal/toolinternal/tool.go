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
	"errors"
	"fmt"
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

// PanicError is a panic recovered from a tool's handler. Its message includes
// the panic value and the stack, so callbacks and plugins can record them, but
// the flow must not show it to the model.
type PanicError struct {
	msg string
}

// NewPanicError returns the error for the value a tool named toolName panicked
// with, recovered at the stack given.
func NewPanicError(toolName string, value any, stack []byte) *PanicError {
	// Format the message now, so that every reader gets the same text even if
	// value changes later.
	return &PanicError{msg: fmt.Sprintf("panic in tool %q: %v\nstack: %s", toolName, value, stack)}
}

func (e *PanicError) Error() string { return e.msg }

// ModelErrorText is the text a failed call of the tool named toolName may show
// the model, and whether err turned out to be a recovered panic. A recovered
// panic becomes a generic message, because its own text carries the panic
// value and the goroutine stack; that holds however deeply it is wrapped,
// since an error on its way out of a tool is wrapped more than once. Any other
// error keeps its text, because returning an error is how a Go tool tells the
// model what went wrong.
//
// Call this wherever a tool error becomes a function response. err must not be
// nil.
func ModelErrorText(toolName string, err error) (text string, panicked bool) {
	if _, ok := errors.AsType[*PanicError](err); ok {
		return fmt.Sprintf("Invoking `%s()` failed with an internal error.", toolName), true
	}
	return err.Error(), false
}
