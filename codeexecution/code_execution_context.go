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
	"time"

	"google.golang.org/adk/session"
)

const (
	contextKey            = "_code_execution_context"
	sessionIdKey          = "_execution_session_id"
	processedFileNamesKey = "_processed_input_files"
	inputFileKey          = "_code_executor_input_files"
	errorCountKey         = "_code_executor_error_counts"
	executionResultsKey   = "_code_execution_results"
)

// CodeExecutorContext manages the persistent context used to configure the code executor.
type CodeExecutorContext struct {
	contextState map[string]any
	sessionState session.State
}

type executionResult struct {
	Code      string
	Stdout    string
	Stderr    string
	Timestamp time.Time
}

// GetStateDelta() gets the state delta to update in the persistent session state.
func (c *CodeExecutorContext) GetStateDelta() map[string]any {
	return map[string]any{
		contextKey: c.contextState,
	}
}

// GetExecutionId gets the session ID for the code executor
func (c *CodeExecutorContext) GetExecutionId() string {
	if id, ok := c.contextState[sessionIdKey].(string); ok {
		return id
	}
	return ""
}

// SetExecutionId sets the session ID for the code executor
func (c *CodeExecutorContext) SetExecutionId(sessionId string) {
	c.contextState[sessionIdKey] = sessionId
}

// GetProcessedFileNames gets the processed file names from the session state.
func (c *CodeExecutorContext) GetProcessedFileNames() []string {
	if fileNames, ok := c.contextState[processedFileNamesKey].([]string); ok {
		return fileNames
	}
	return nil
}

// AddProcessedFileNames adds the processed file name to the context state
func (c *CodeExecutorContext) AddProcessedFileNames(fileNames []string) {
	currentFiles, _ := c.contextState[processedFileNamesKey].([]string)
	c.contextState[processedFileNamesKey] = append(currentFiles, fileNames...)
}

// GetInputFiles gets the code executor input file names from the session state
func (c *CodeExecutorContext) GetInputFiles() []*File {
	inputFiles, _ := c.sessionState.Get(inputFileKey)
	files, _ := inputFiles.([]*File)
	return files
}

// AddInputFiles adds the input files to the code executor context.
func (c *CodeExecutorContext) AddInputFiles(inputFiles []*File) error {
	files, _ := c.sessionState.Get(inputFileKey)
	currentFiles, _ := files.([]*File)
	return c.sessionState.Set(inputFileKey, append(currentFiles, inputFiles...))
}

// ClearInputFiles removes the input files and processed file names to the code executor context.
func (c *CodeExecutorContext) ClearInputFiles() error {
	if err := c.sessionState.Set(inputFileKey, nil); err != nil {
		return err
	}
	if c.contextState != nil {
		delete(c.contextState, inputFileKey)
	}
	return nil
}

// GetErrorCount gets the error count from the session state.
func (c *CodeExecutorContext) GetErrorCount(invocationId string) int32 {
	errCounts, _ := c.sessionState.Get(errorCountKey)
	errCountMap, _ := errCounts.(map[string]int32)
	return errCountMap[invocationId]
}

// ResetErrorCount resets the error count from the session state.
func (c *CodeExecutorContext) ResetErrorCount(invocationId string) error {
	errCount, _ := c.sessionState.Get(errorCountKey)
	errCountMap, _ := errCount.(map[string]int32)
	if errCountMap == nil {
		return nil
	}
	delete(errCountMap, invocationId)
	return c.sessionState.Set(errorCountKey, errCountMap)
}

// IncrementErrorCount increments the error count for a given invocation ID
func (c *CodeExecutorContext) IncrementErrorCount(invocationId string) error {
	errCount, _ := c.sessionState.Get(errorCountKey)
	errCountMap, _ := errCount.(map[string]int32)
	if errCountMap == nil {
		errCountMap = make(map[string]int32)
	}
	errCountMap[invocationId]++
	return c.sessionState.Set(errorCountKey, errCountMap)
}

// UpdateCodeExecutionResult updates the code execution result.
func (c *CodeExecutorContext) UpdateCodeExecutionResult(invocationId, code, stdout, stderr string) error {
	val, _ := c.sessionState.Get(executionResultsKey)
	results, _ := val.(map[string][]executionResult)
	if results == nil {
		results = make(map[string][]executionResult)
	}
	results[invocationId] = append(results[invocationId], executionResult{
		Code:      code,
		Stdout:    stdout,
		Stderr:    stderr,
		Timestamp: time.Now().UTC(),
	})
	return c.sessionState.Set(executionResultsKey, results)
}

// getCodeExecutorContext gets the code executor context from the session state.
func getCodeExecutorContext(state session.State) map[string]any {
	val, _ := state.Get(contextKey)
	contextMap, _ := val.(map[string]any)
	if contextMap == nil {
		return make(map[string]any)
	}
	return contextMap
}

func NewCodeExecutorContext(state session.State) *CodeExecutorContext {
	return &CodeExecutorContext{
		contextState: getCodeExecutorContext(state),
		sessionState: state,
	}
}
