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

package code_executors

import "google.golang.org/adk/agent"

type CodeExecutor interface {
	ExecuteCode(ctx agent.InvocationContext, input CodeExecutionInput) (CodeExecutionResult, error)
}

// BaseCodeExecutor Abstract base class for all code executors.
//
//The code executor allows the agent to execute code blocks from model responses
//and incorporate the execution results into the final response.
//
//Attributes:
// 	optimize_data_file: If true, extract and process data files from the model request and attach them to the code executor. Supported data file MimeTypes are [text/csv]. Default to False.
//	stateful: Whether the code executor is stateful. Default to False.
//	error_retry_attempts: The number of attempts to retry on consecutive code execution errors. Default to 2.
//	code_block_delimiters: The list of the enclosing delimiters to identify the code blocks.
//	execution_result_delimiters: The delimiters to format the code execution result.

type Delimiter struct {
	Start string
	End   string
}
type BaseCodeExecutor struct {
	OptimizeDataFile          bool
	Stateful                  bool
	ErrorRetryAttempts        int
	CodeBlockDelimiters       []Delimiter
	ExecutionResultDelimiters [2]string
}

func (b *BaseCodeExecutor) ExecuteCode(ctx agent.InvocationContext, input CodeExecutionInput) (CodeExecutionResult, error) {
	return CodeExecutionResult{}, nil
}

func DefaultBaseCodeExecutor() *BaseCodeExecutor {
	return &BaseCodeExecutor{
		OptimizeDataFile:   false,
		Stateful:           false,
		ErrorRetryAttempts: 2,
		CodeBlockDelimiters: []Delimiter{
			{Start: "```tool_code\n", End: "\n```"},
			{Start: "```python\n", End: "\n```"},
		},
		ExecutionResultDelimiters: [2]string{
			"```tool_output\n", "\n```",
		},
	}
}
