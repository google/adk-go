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

package codeexecution

import (
	"google.golang.org/adk/agent"
)

// CodeExecutor implements Code Execution
//
// The code executor allows the agent to execute code blocks from model responses
// and incorporate the execution results into the final response.
type CodeExecutor interface {
	// Executes code and return the code execution result
	ExecuteCode(agent.InvocationContext, *CodeExecutionInput) (*CodeExecutionResult, error)
	// Access to standard code execution configuration details.
	Config() *CodeExecutorConfig
}

// CodeExecutionInput contains the input of code execution.
type CodeExecutionInput struct {
	// The code to execute.
	Code string
	// The input files available to the code.
	InputFiles []*File
	//  The execution ID for the stateful code execution.
	ExecutionId string
}

// CodeExecutionResult contains the result of code execution.
type CodeExecutionResult struct {
	// The standard output of the code execution.
	Stdout string
	// The standard error of the code execution.
	Stderr string
	// The output files from the code execution.
	OutputFiles []*File
}

// File contains a file name and its content.
type File struct {
	// The name of the file with file extension (e.g., "file.csv").
	Name string
	//  The base64-encoded bytes of the file content or the original bytes of the file content.
	Content []byte
	// The mime type of the file (e.g., "image/png").
	MimeType string
}

// Config is the configuration for a CodeExecutor.
type CodeExecutorConfig struct {
	// If true, extract and process data files from the model request
	// and attach them to the code executor.
	OptimizeDataFile bool
	// Whether the code executor is stateful.
	Stateful bool
	// The number of attempts to retry on consecutive code execution errors.
	ErrorRetryAttempts int32
	// The list of the enclosing delimiters to identify the code blocks.
	//
	// For example, the delimiter ('```python\\n', '\\n```') can be
	// used to identify code blocks with the following format:
	//   ```python
	//   print("hello")
	//   ```
	CodeBlockDelimiters []Delimiter
	// The delimiters to format the code execution result.
	ExecutionResultDelimiters []Delimiter
	// The timeout in seconds for the code execution.
	TimeoutSeconds int32
}

// Delimiter is used to identity code execution and result blocks.
type Delimiter struct {
	// Code-leading delimiter
	Leading string
	// Code-trailing delimiter
	Trailing string
}

// ConfigOption is a function that configures the CodeExecutor.
type ConfigOption func(*CodeExecutorConfig)

// WithCodeBlockDelimiters sets the delimiters on the CodeExecutor.
func WithCodeBlockDelimiters(delimiters []Delimiter) ConfigOption {
	return func(c *CodeExecutorConfig) {
		c.CodeBlockDelimiters = delimiters
	}
}

// WithStateful sets the statefulness of the CodeExecutor.
func WithStateful() ConfigOption {
	return func(c *CodeExecutorConfig) {
		c.Stateful = true
	}
}

// WithStateful sets the code execution retry attemps count of the CodeExecutor.
func WithErrorRetryAttempts(count int32) ConfigOption {
	return func(c *CodeExecutorConfig) {
		c.ErrorRetryAttempts = count
	}
}

// WithTimeoutSeconds sets the execution timeout of the CodeExecutor.
func WithTimeoutSeconds(seconds int32) ConfigOption {
	return func(c *CodeExecutorConfig) {
		c.TimeoutSeconds = seconds
	}
}

// NewCodeExecutionConfig configures the CodeExecutorConfig.
//
// Also sets default values.
func newCodeExecutionConfig(opts ...ConfigOption) *CodeExecutorConfig {
	cfg := &CodeExecutorConfig{
		OptimizeDataFile:   false,
		Stateful:           false,
		ErrorRetryAttempts: 2,
		CodeBlockDelimiters: []Delimiter{
			{
				Leading:  "```python\n",
				Trailing: "\n```",
			},
		},
		ExecutionResultDelimiters: []Delimiter{
			{
				Leading:  "```python\n",
				Trailing: "\n```",
			},
		},
		TimeoutSeconds: 60,
	}
	for _, opt := range opts {
		opt(cfg)
	}
	return cfg
}
