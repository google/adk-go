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

package codeexecutor

import (
	"context"

	"google.golang.org/adk/v2/agent"
)

// BuiltIn identifies the executors that delegate code execution to the
// model's own tool. llmagent adds that tool to every request of an agent
// whose CodeExecutor is a BuiltIn. Only *BuiltInExecutor and types that embed
// it implement BuiltIn: the unexported method keeps other packages from
// claiming the built-in behavior by accident.
type BuiltIn interface {
	CodeExecutor
	builtIn()
}

// BuiltInExecutor is the executor that lets a Gemini model run its code in
// the model's built-in code-execution tool. It runs nothing itself.
//
// The zero value is ready to use and the type has no state, so every method
// is safe for concurrent use, also on a nil *BuiltInExecutor. Its Settings are
// always DefaultSettings. To change one, for example the delimiters the flow
// uses for history, embed *BuiltInExecutor in a struct and define Settings on
// the struct; the struct still implements BuiltIn.
type BuiltInExecutor struct{}

// NewBuiltInExecutor returns a non-nil *BuiltInExecutor. It cannot fail.
func NewBuiltInExecutor() *BuiltInExecutor {
	return &BuiltInExecutor{}
}

// Settings returns DefaultSettings(), a new value (with a new
// CodeBlockDelimiters slice) on every call.
func (*BuiltInExecutor) Settings() Settings {
	return DefaultSettings()
}

// ExecuteCode ignores all of its arguments, so any of them may be nil, and
// returns a nil Result and a nil error: the model runs the code, not this
// executor. A caller reads the nil Result as "no result".
//
// It returns (nil, nil) even when ctx is done, as adk-python's
// BuiltInCodeExecutor.execute_code returns None without raising
// (code_executors/built_in_code_executor.py).
func (*BuiltInExecutor) ExecuteCode(ctx context.Context, invocationCtx agent.InvocationContext, in *Input) (*Result, error) {
	return nil, nil
}

func (*BuiltInExecutor) builtIn() {}
