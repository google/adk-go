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
	"slices"

	"google.golang.org/adk/v2/agent"
)

// CodeExecutor runs code that a model wrote and reports what the code
// printed. Set one on llmagent.Config.CodeExecutor.
//
// An implementation must be safe for concurrent use: one agent shares its
// executor across every invocation and every clone of the agent.
type CodeExecutor interface {
	// Settings returns the executor's configuration. Readers call it each
	// time they need a value and do not cache it, so an implementation
	// returns the same values on every call. Readers call WithDefaults()
	// on the result to apply defaults to unset fields. Readers do not modify
	// the slices in the returned value.
	Settings() Settings

	// ExecuteCode runs in.Code and returns its output.
	//
	// The caller provides a non-nil ctx, a non-nil invocationCtx and a
	// non-nil in. ctx alone governs cancellation and deadlines;
	// invocationCtx only gives access to the agent, session and artifacts.
	// An implementation must not use invocationCtx for cancellation. When
	// ctx is done, an executor that runs code returns promptly with a nil
	// Result and an error that wraps ctx.Err(). *BuiltInExecutor runs
	// nothing and returns a nil Result and a nil error even then.
	//
	// A non-nil error means the code could not be run at all (cancellation,
	// an unreachable sandbox). Code that ran and failed is reported in
	// Result.Stderr, and in Result.ExitCode when the executor knows it, with
	// a nil error.
	//
	// An executor that runs code returns a non-nil Result when the error is
	// nil. A caller must still handle a nil Result with a nil error, which
	// *BuiltInExecutor returns, and read it as "no result".
	//
	// The flow calls ExecuteCode with the code, the input files it extracted
	// and, for a stateful executor, an execution ID. Skills calls it with
	// only the code, passing the tool's agent.Context, which embeds
	// agent.InvocationContext, as invocationCtx.
	ExecuteCode(ctx context.Context, invocationCtx agent.InvocationContext, in *Input) (*Result, error)
}

// Input holds the parameters of one code execution. The zero value is an
// empty input: no code, no files, no execution ID.
type Input struct {
	// Code is the source to run.
	Code string
	// InputFiles are the files the code can read. A nil or empty slice means
	// none; no element is nil.
	InputFiles []*File
	// ExecutionID names a stateful execution. The flow sets it, to the
	// session ID, only when Settings().Stateful is true. Empty means the
	// execution is not stateful (Python's None).
	ExecutionID string
}

// Result holds the output of one code execution. The zero value is a result
// with empty output, no files and no exit code.
type Result struct {
	// Stdout is what the code wrote to standard output.
	Stdout string
	// Stderr is what the code wrote to standard error. The flow treats a
	// non-empty Stderr as a failed execution.
	Stderr string
	// OutputFiles are the files the code produced. A nil or empty slice means
	// none; no element is nil.
	OutputFiles []*File
	// ExitCode is the status the executed process exited with. nil means the
	// executor did not report one, which is different from a pointer to 0.
	// Skills reads it, the flow does not. The Result owns the pointed-to int.
	ExitCode *int
}

// File is a named blob passed to or returned from an executor. A File is
// treated as immutable: once built, neither the File nor its Content is
// modified. The zero value has no name, no content and an empty MimeType.
type File struct {
	// Name is the file name with its extension, for example "data.csv".
	Name string
	// Content is the raw bytes of the file. It is never base64 text.
	Content []byte
	// MimeType is the MIME type, for example "image/png". The empty string
	// means "text/plain", Python's default.
	MimeType string
}

// Settings is an executor's configuration. The zero value is valid but not
// the default; readers call WithDefaults() on it to get the defaults.
type Settings struct {
	// OptimizeDataFile asks the flow to extract data files (CSV) from the
	// conversation and pass them to the executor. Default false. Only the
	// flow's non-built-in pre-processor reads it, so it has no effect on
	// *BuiltInExecutor.
	OptimizeDataFile bool

	// Stateful makes the flow give each execution the session ID as
	// Input.ExecutionID, so a stateful executor can keep variables between
	// calls. Default false: ExecutionID stays empty.
	Stateful bool

	// ErrorRetryAttempts is the number of consecutive failed executions
	// (non-empty Stderr) after which the flow stops running code for the
	// invocation. The flow runs code only while its consecutive error count
	// is below this value, so a pointer to 0 or to a negative value means
	// code never runs. nil means the default of 2.
	ErrorRetryAttempts *int

	// CodeBlockDelimiters are the (opening, closing) strings that fence code
	// in model text, for example {"```python\n", "\n```"}. Order matters: the
	// flow writes code into conversation history with the first pair and
	// searches model text for every pair. nil means the default of the two
	// pairs {"```tool_code\n", "\n```"} and {"```python\n", "\n```"}, in that
	// order. A non-nil empty slice means no pair.
	CodeBlockDelimiters [][2]string

	// ExecutionResultDelimiters are the (opening, closing) strings the flow
	// puts around an execution result when it writes the result into
	// conversation history as text. nil means the default of
	// {"```tool_output\n", "\n```"}. The zero value wraps with empty strings.
	ExecutionResultDelimiters *[2]string

	// TimeoutSeconds is the number of seconds an executor that enforces a
	// timeout allows one execution. nil means the settings set none (the
	// default). Neither the flow nor Skills reads it; only a concrete
	// executor does, and this package does not validate the value.
	TimeoutSeconds *int
}

// DefaultSettings returns the settings Python's BaseCodeExecutor gives every
// executor. It returns the same values as Settings{}.WithDefaults(), with a
// new slice on every call.
func DefaultSettings() Settings {
	return Settings{}.WithDefaults()
}

// WithDefaults returns a copy of s in which every unset field holds Python's
// default (adk-python code_executors/base_code_executor.py). Its
// ErrorRetryAttempts and ExecutionResultDelimiters are non-nil, and its
// CodeBlockDelimiters is a new slice, so a caller may modify the result's
// delimiters without changing s. TimeoutSeconds stays nil when s leaves it
// unset, because Python's default is no timeout. Values that s sets, including
// a pointer to 0 and a non-nil empty CodeBlockDelimiters, are kept.
func (s Settings) WithDefaults() Settings {
	if s.ErrorRetryAttempts == nil {
		attempts := 2
		s.ErrorRetryAttempts = &attempts
	}
	if s.CodeBlockDelimiters == nil {
		s.CodeBlockDelimiters = [][2]string{
			{"```tool_code\n", "\n```"},
			{"```python\n", "\n```"},
		}
	} else {
		s.CodeBlockDelimiters = slices.Clone(s.CodeBlockDelimiters)
	}
	if s.ExecutionResultDelimiters == nil {
		s.ExecutionResultDelimiters = &[2]string{"```tool_output\n", "\n```"}
	}
	return s
}
