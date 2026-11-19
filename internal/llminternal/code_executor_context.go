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

package llminternal

import (
	"bytes"
	"errors"
	"fmt"
	"maps"
	"slices"
	"sync"
	"time"

	"google.golang.org/adk/v2/codeexecutor"
	"google.golang.org/adk/v2/internal/sessionutils"
	"google.golang.org/adk/v2/session"
)

// The session state keys of the code executor's bookkeeping. The strings are
// adk-python's (code_executors/code_executor_context.py).
const (
	// codeExecutionContextKey holds a map[string]any; it is the only key that
	// travels in an event's state delta.
	codeExecutionContextKey = "_code_execution_context"
	// executionSessionIDKey is the entry of the code execution context that
	// holds the execution ID of a stateful executor.
	executionSessionIDKey = "execution_session_id"
	// processedInputFilesKey is the entry of the code execution context that
	// lists the names of the input files already explored.
	processedInputFilesKey = "processed_input_files"
	// codeExecutorInputFilesKey holds a []*codeexecutor.File with raw bytes.
	codeExecutorInputFilesKey = "_code_executor_input_files"
	// codeExecutorErrorCountsKey holds a map[string]int by invocation ID.
	codeExecutorErrorCountsKey = "_code_executor_error_counts"
	// codeExecutionResultsKey holds a map[string][]map[string]any by
	// invocation ID.
	codeExecutionResultsKey = "_code_execution_results"
)

// codeExecutorContext reads and writes the code executor's bookkeeping in a
// session's State. It holds no data besides the State, so copies are cheap and
// every method sees what the others wrote.
//
// Only the code execution context travels in a state delta (stateDelta); the
// input files, error counts and execution records are written straight into
// the State and no session service stores them, as in adk-python. A stored map
// is never edited after it was stored: every write stores an edited copy, so a
// reader of the State never sees a map change under it.
type codeExecutorContext struct{ state session.State }

// codeExecutorStateMu serializes the bookkeeping updates of every invocation in
// the process, whatever session they belong to. It guards no data of its own;
// it makes each read-modify-write of a bookkeeping key atomic, as Python's is.
// A method holds it for one state read and write only, never across ExecuteCode,
// an artifact save or a yield. A session.State whose Set does I/O runs that I/O
// under it.
var codeExecutorStateMu sync.Mutex

// newCodeExecutorContext returns the bookkeeping view of state, a non-nil
// session State. It stores an empty code execution context when state has
// none, as adk-python's CodeExecutorContext does on construction. It fails
// when state cannot be read or written, or when the context key holds a value
// that is not a map[string]any.
func newCodeExecutorContext(state session.State) (*codeExecutorContext, error) {
	c := &codeExecutorContext{state: state}
	codeExecutorStateMu.Lock()
	defer codeExecutorStateMu.Unlock()
	_, ok, err := c.executionContext()
	if err != nil {
		return nil, err
	}
	if ok {
		return c, nil
	}
	if err := c.state.Set(codeExecutionContextKey, map[string]any{}); err != nil {
		return nil, fmt.Errorf("set %s: %w", codeExecutionContextKey, err)
	}
	return c, nil
}

// stateDelta returns the state delta of a code execution result event: the
// code execution context under its key, as a deep copy that shares nothing
// with the State.
func (c *codeExecutorContext) stateDelta() (map[string]any, error) {
	ctxMap, _, err := c.executionContext()
	if err != nil {
		return nil, err
	}
	if ctxMap == nil {
		ctxMap = map[string]any{}
	}
	return map[string]any{codeExecutionContextKey: sessionutils.CloneValue(ctxMap)}, nil
}

// getOrSetExecutionID returns the execution ID of a stateful executor: the one
// stored in the code execution context, or sessionID, which it then stores.
// It returns "" when stateful is false.
func (c *codeExecutorContext) getOrSetExecutionID(sessionID string, stateful bool) (string, error) {
	if !stateful {
		return "", nil
	}
	codeExecutorStateMu.Lock()
	defer codeExecutorStateMu.Unlock()
	ctxMap, _, err := c.executionContext()
	if err != nil {
		return "", err
	}
	switch id := ctxMap[executionSessionIDKey].(type) {
	case nil:
	case string:
		if id != "" {
			return id, nil
		}
	default:
		return "", fmt.Errorf("%s.%s holds %T, want string", codeExecutionContextKey, executionSessionIDKey, id)
	}
	updated := maps.Clone(ctxMap)
	if updated == nil {
		updated = map[string]any{}
	}
	updated[executionSessionIDKey] = sessionID
	if err := c.state.Set(codeExecutionContextKey, updated); err != nil {
		return "", fmt.Errorf("set %s: %w", codeExecutionContextKey, err)
	}
	return sessionID, nil
}

// inputFiles returns a copy of the stored list of input files, or nil when
// none are stored. The *codeexecutor.File elements are shared; a File is
// immutable.
func (c *codeExecutorContext) inputFiles() ([]*codeexecutor.File, error) {
	v, ok, err := c.get(codeExecutorInputFilesKey)
	if err != nil || !ok {
		return nil, err
	}
	files, ok := v.([]*codeexecutor.File)
	if !ok {
		return nil, fmt.Errorf("%s holds %T, want []*codeexecutor.File", codeExecutorInputFilesKey, v)
	}
	return slices.Clone(files), nil
}

// storeInputFile stores f, a non-nil file, in the list of input files. A
// stored file with f's name, MIME type and bytes is left as it is. A stored
// file with f's name and other content is replaced in place, and its name is
// removed from the processed input files, so that it is explored again.
func (c *codeExecutorContext) storeInputFile(f *codeexecutor.File) error {
	codeExecutorStateMu.Lock()
	defer codeExecutorStateMu.Unlock()
	files, err := c.inputFiles()
	if err != nil {
		return err
	}
	i := slices.IndexFunc(files, func(stored *codeexecutor.File) bool { return stored != nil && stored.Name == f.Name })
	if i < 0 {
		return c.set(codeExecutorInputFilesKey, append(files, f))
	}
	if files[i].MimeType == f.MimeType && bytes.Equal(files[i].Content, f.Content) {
		return nil
	}
	files[i] = f
	if err := c.set(codeExecutorInputFilesKey, files); err != nil {
		return err
	}
	names, err := c.processedFileNames()
	if err != nil {
		return err
	}
	if !slices.Contains(names, f.Name) {
		return nil
	}
	return c.setProcessedFileNames(slices.DeleteFunc(names, func(n string) bool { return n == f.Name }))
}

// processedFileNames returns a copy of the names of the input files already
// explored, nil when none is listed. It accepts the list as a []string or, as
// a persistent session service reads it back from JSON, a []any of strings.
func (c *codeExecutorContext) processedFileNames() ([]string, error) {
	ctxMap, _, err := c.executionContext()
	if err != nil {
		return nil, err
	}
	switch v := ctxMap[processedInputFilesKey].(type) {
	case nil:
		return nil, nil
	case []string:
		return slices.Clone(v), nil
	case []any:
		names := make([]string, len(v))
		for i, e := range v {
			name, ok := e.(string)
			if !ok {
				return nil, fmt.Errorf("%s.%s holds an element of type %T, want string", codeExecutionContextKey, processedInputFilesKey, e)
			}
			names[i] = name
		}
		return names, nil
	default:
		return nil, fmt.Errorf("%s.%s holds %T, want a list of strings", codeExecutionContextKey, processedInputFilesKey, v)
	}
}

// addProcessedFileName lists name as an explored input file. It writes
// nothing when name is already listed.
func (c *codeExecutorContext) addProcessedFileName(name string) error {
	codeExecutorStateMu.Lock()
	defer codeExecutorStateMu.Unlock()
	names, err := c.processedFileNames()
	if err != nil {
		return err
	}
	if slices.Contains(names, name) {
		return nil
	}
	return c.setProcessedFileNames(append(names, name))
}

// setProcessedFileNames stores names as the processed input files in a copy
// of the code execution context. The caller holds codeExecutorStateMu.
func (c *codeExecutorContext) setProcessedFileNames(names []string) error {
	ctxMap, _, err := c.executionContext()
	if err != nil {
		return err
	}
	updated := maps.Clone(ctxMap)
	if updated == nil {
		updated = map[string]any{}
	}
	updated[processedInputFilesKey] = names
	return c.set(codeExecutionContextKey, updated)
}

// errorCount returns the number of consecutive failed executions of the
// invocation, 0 when none is stored.
func (c *codeExecutorContext) errorCount(invocationID string) (int, error) {
	counts, err := c.errorCounts()
	if err != nil {
		return 0, err
	}
	return counts[invocationID], nil
}

// incrementErrorCount adds one to the error count of the invocation.
func (c *codeExecutorContext) incrementErrorCount(invocationID string) error {
	codeExecutorStateMu.Lock()
	defer codeExecutorStateMu.Unlock()
	counts, err := c.errorCounts()
	if err != nil {
		return err
	}
	updated := maps.Clone(counts)
	if updated == nil {
		updated = map[string]int{}
	}
	updated[invocationID]++
	return c.set(codeExecutorErrorCountsKey, updated)
}

// resetErrorCount removes the error count of the invocation. It writes nothing
// when no error counts are stored.
func (c *codeExecutorContext) resetErrorCount(invocationID string) error {
	codeExecutorStateMu.Lock()
	defer codeExecutorStateMu.Unlock()
	counts, err := c.errorCounts()
	if err != nil {
		return err
	}
	if counts == nil {
		return nil
	}
	updated := maps.Clone(counts)
	delete(updated, invocationID)
	return c.set(codeExecutorErrorCountsKey, updated)
}

// recordExecution appends one execution record to the invocation's list:
// code, result_stdout, result_stderr and timestamp, which is at in Unix
// seconds.
func (c *codeExecutorContext) recordExecution(invocationID, code, stdout, stderr string, at time.Time) error {
	codeExecutorStateMu.Lock()
	defer codeExecutorStateMu.Unlock()
	var results map[string][]map[string]any
	v, ok, err := c.get(codeExecutionResultsKey)
	if err != nil {
		return err
	}
	if ok {
		if results, ok = v.(map[string][]map[string]any); !ok {
			return fmt.Errorf("%s holds %T, want map[string][]map[string]any", codeExecutionResultsKey, v)
		}
	}
	updated := maps.Clone(results)
	if updated == nil {
		updated = map[string][]map[string]any{}
	}
	updated[invocationID] = append(slices.Clone(results[invocationID]), map[string]any{
		"code":          code,
		"result_stdout": stdout,
		"result_stderr": stderr,
		"timestamp":     at.Unix(),
	})
	return c.set(codeExecutionResultsKey, updated)
}

// executionContext returns the stored code execution context and whether the
// key is present.
func (c *codeExecutorContext) executionContext() (map[string]any, bool, error) {
	v, ok, err := c.get(codeExecutionContextKey)
	if err != nil || !ok {
		return nil, ok, err
	}
	m, isMap := v.(map[string]any)
	if !isMap {
		return nil, true, fmt.Errorf("%s holds %T, want map[string]any", codeExecutionContextKey, v)
	}
	return m, true, nil
}

// errorCounts returns the stored error counts, nil when none are stored.
func (c *codeExecutorContext) errorCounts() (map[string]int, error) {
	v, ok, err := c.get(codeExecutorErrorCountsKey)
	if err != nil || !ok {
		return nil, err
	}
	counts, ok := v.(map[string]int)
	if !ok {
		return nil, fmt.Errorf("%s holds %T, want map[string]int", codeExecutorErrorCountsKey, v)
	}
	return counts, nil
}

// get reads key. A missing key and a nil value both give ok == false, as
// adk-python's session_state.get(key) returns None for both.
func (c *codeExecutorContext) get(key string) (any, bool, error) {
	v, err := c.state.Get(key)
	if errors.Is(err, session.ErrStateKeyNotExist) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("get %s: %w", key, err)
	}
	return v, v != nil, nil
}

func (c *codeExecutorContext) set(key string, value any) error {
	if err := c.state.Set(key, value); err != nil {
		return fmt.Errorf("set %s: %w", key, err)
	}
	return nil
}
