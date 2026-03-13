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

type File struct {
	Name     string `json:"name"`                //  A structure that contains a file name and its content.
	Content  []byte `json:"content"`             // The base64-encoded bytes of the file content or the original bytes of the file content.
	MimeType string `json:"mime_type,omitempty"` //  The mime type of the file (e.g., "image/png",'text/plain').
}

type CodeExecutionInput struct {
	Args        any    `json:"args"`
	ScriptPath  string `json:"script_path"`
	InputFiles  []File `json:"input_files,omitempty"`  //  The input files available to the code.
	ExecutionID string `json:"execution_id,omitempty"` //  The execution ID for the stateful code execution.
}

type CodeExecutionResult struct {
	StdOut      string `json:"stdout"`                 // The standard output of the code execution.
	StdErr      string `json:"stderr,omitempty"`       // The standard error of the code execution.
	OutputFiles []File `json:"output_files,omitempty"` // The output files from the code execution.
}
