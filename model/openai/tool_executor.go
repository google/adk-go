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

package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"sort"
	"sync"
	"time"

	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// ToolExecutor manages execution of tools for OpenAI adapter.
// It handles both parallel and sequential execution modes.
type ToolExecutor struct {
	tools             map[string]tool.Tool
	parallelExecution bool
	timeout           time.Duration
	maxRetries        int
	logger            *log.Logger
}

// ToolExecutorConfig configures the tool executor.
type ToolExecutorConfig struct {
	// ParallelExecution enables concurrent tool execution
	ParallelExecution bool
	// Timeout for individual tool execution
	Timeout time.Duration
	// MaxRetries for failed tool calls
	MaxRetries int
	// Logger for tool execution logging (optional)
	Logger *log.Logger
}

// NewToolExecutor creates a new tool executor.
func NewToolExecutor(tools map[string]any, cfg *ToolExecutorConfig) *ToolExecutor {
	if cfg == nil {
		cfg = &ToolExecutorConfig{
			ParallelExecution: true,
			Timeout:           30 * time.Second,
			MaxRetries:        2,
		}
	}

	// Convert any tools to tool.Tool interface
	toolMap := make(map[string]tool.Tool)
	for name, t := range tools {
		if toolImpl, ok := t.(tool.Tool); ok {
			toolMap[name] = toolImpl
		}
	}

	return &ToolExecutor{
		tools:             toolMap,
		parallelExecution: cfg.ParallelExecution,
		timeout:           cfg.Timeout,
		maxRetries:        cfg.MaxRetries,
		logger:            cfg.Logger,
	}
}

// ToolCallResult represents the result of a tool call execution.
type ToolCallResult struct {
	ToolCallID string
	Name       string
	Response   map[string]any
	Error      error
	Duration   time.Duration
	Order      int // Original order for sorting
}

// ExecuteToolCalls executes multiple tool calls and returns their responses.
// It handles:
// - Parallel or sequential execution based on configuration
// - Error handling and retries
// - Timeout enforcement
// - Result ordering by tool call ID for predictability
func (te *ToolExecutor) ExecuteToolCalls(ctx context.Context, toolCalls []ToolCall, toolCtx tool.Context) ([]*ToolCallResult, error) {
	if len(toolCalls) == 0 {
		return nil, nil
	}

	te.logf("Executing %d tool calls (parallel=%v)", len(toolCalls), te.parallelExecution)

	// Analyze dependencies (optional future enhancement)
	executionOrder := te.analyzeDependencies(toolCalls)

	var results []*ToolCallResult

	if te.parallelExecution && len(executionOrder) == 1 {
		// All calls are independent, execute in parallel
		results = te.executeParallel(ctx, toolCalls, toolCtx)
	} else {
		// Execute sequentially (either forced or has dependencies)
		results = te.executeSequential(ctx, toolCalls, executionOrder, toolCtx)
	}

	// Sort by order for predictable output
	sort.Slice(results, func(i, j int) bool {
		return results[i].Order < results[j].Order
	})

	te.logf("Completed %d tool calls", len(results))
	return results, nil
}

// executeParallel executes tool calls concurrently.
func (te *ToolExecutor) executeParallel(ctx context.Context, toolCalls []ToolCall, toolCtx tool.Context) []*ToolCallResult {
	var wg sync.WaitGroup
	results := make([]*ToolCallResult, len(toolCalls))

	for i, tc := range toolCalls {
		wg.Add(1)
		go func(index int, call ToolCall) {
			defer wg.Done()
			results[index] = te.executeOne(ctx, call, toolCtx, index)
		}(i, tc)
	}

	wg.Wait()
	return results
}

// executeSequential executes tool calls one by one in specified order.
func (te *ToolExecutor) executeSequential(ctx context.Context, toolCalls []ToolCall, executionOrder [][]int, toolCtx tool.Context) []*ToolCallResult {
	results := make([]*ToolCallResult, len(toolCalls))
	resultMap := make(map[string]map[string]any) // Store results by tool call ID
	var resultMapMu sync.Mutex                   // Protect concurrent access to resultMap

	for _, batch := range executionOrder {
		if len(batch) == 1 {
			// Single call in this batch
			idx := batch[0]
			result := te.executeOne(ctx, toolCalls[idx], toolCtx, idx)
			results[idx] = result
			resultMapMu.Lock()
			resultMap[result.ToolCallID] = result.Response
			resultMapMu.Unlock()
		} else {
			// Multiple independent calls in this batch - execute in parallel
			var wg sync.WaitGroup
			for _, idx := range batch {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()
					result := te.executeOne(ctx, toolCalls[index], toolCtx, index)
					results[index] = result
					resultMapMu.Lock()
					resultMap[result.ToolCallID] = result.Response
					resultMapMu.Unlock()
				}(idx)
			}
			wg.Wait()
		}
	}

	return results
}

// executeOne executes a single tool call with retry logic.
func (te *ToolExecutor) executeOne(ctx context.Context, tc ToolCall, toolCtx tool.Context, order int) *ToolCallResult {
	start := time.Now()

	result := &ToolCallResult{
		ToolCallID: tc.ID,
		Name:       tc.Function.Name,
		Order:      order,
	}

	te.logf("Executing tool: %s (id=%s)", tc.Function.Name, tc.ID)

	// Find the tool
	toolImpl, ok := te.tools[tc.Function.Name]
	if !ok {
		result.Error = fmt.Errorf("tool not found: %s", tc.Function.Name)
		result.Response = map[string]any{"error": result.Error.Error()}
		result.Duration = time.Since(start)
		te.logf("Tool %s not found", tc.Function.Name)
		return result
	}

	// Parse arguments
	args, err := te.parseArguments(tc.Function.Arguments)
	if err != nil {
		result.Error = fmt.Errorf("failed to parse arguments: %w", err)
		result.Response = map[string]any{"error": result.Error.Error()}
		result.Duration = time.Since(start)
		te.logf("Failed to parse args for %s: %v", tc.Function.Name, err)
		return result
	}

	// Execute with retries
	var lastErr error
	for attempt := 0; attempt <= te.maxRetries; attempt++ {
		if attempt > 0 {
			te.logf("Retrying tool %s (attempt %d/%d)", tc.Function.Name, attempt, te.maxRetries)
		}

		// Create timeout context
		execCtx, cancel := context.WithTimeout(ctx, te.timeout)

		// Execute the tool
		response, err := te.executeTool(execCtx, toolImpl, args, toolCtx)
		cancel()

		if err == nil {
			result.Response = response
			result.Duration = time.Since(start)
			te.logf("Tool %s completed successfully (duration=%v)", tc.Function.Name, result.Duration)
			return result
		}

		lastErr = err
		te.logf("Tool %s failed (attempt %d): %v", tc.Function.Name, attempt+1, err)

		// Exponential backoff for retries
		if attempt < te.maxRetries {
			backoff := time.Duration(attempt+1) * 100 * time.Millisecond
			time.Sleep(backoff)
		}
	}

	// All retries failed
	result.Error = fmt.Errorf("tool execution failed after %d attempts: %w", te.maxRetries+1, lastErr)
	result.Response = map[string]any{"error": result.Error.Error()}
	result.Duration = time.Since(start)
	te.logf("Tool %s failed permanently: %v", tc.Function.Name, result.Error)

	return result
}

// executeTool executes a single tool with proper context and error handling.
func (te *ToolExecutor) executeTool(ctx context.Context, t tool.Tool, args map[string]any, toolCtx tool.Context) (map[string]any, error) {
	// Check if tool implements the function tool interface
	type functionTool interface {
		Run(ctx tool.Context, args map[string]any) (map[string]any, error)
	}

	ft, ok := t.(functionTool)
	if !ok {
		return nil, fmt.Errorf("tool %s does not implement Run method", t.Name())
	}

	// Execute with context cancellation support
	resultCh := make(chan map[string]any, 1)
	errCh := make(chan error, 1)

	go func() {
		result, err := ft.Run(toolCtx, args)
		if err != nil {
			errCh <- err
			return
		}
		resultCh <- result
	}()

	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("tool execution timeout: %w", ctx.Err())
	case err := <-errCh:
		return nil, err
	case result := <-resultCh:
		return result, nil
	}
}

// parseArguments parses JSON string arguments into a map.
func (te *ToolExecutor) parseArguments(argsJSON string) (map[string]any, error) {
	if argsJSON == "" {
		return make(map[string]any), nil
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return nil, fmt.Errorf("invalid JSON arguments: %w", err)
	}

	return args, nil
}

// analyzeDependencies analyzes tool call dependencies and returns execution order.
// Returns batches of indices where calls in the same batch can run in parallel.
// Detects dependencies by analyzing if tool arguments reference other tool call IDs.
func (te *ToolExecutor) analyzeDependencies(toolCalls []ToolCall) [][]int {
	// If parallel execution is disabled, each tool gets its own batch (sequential)
	if !te.parallelExecution {
		batches := make([][]int, len(toolCalls))
		for i := range toolCalls {
			batches[i] = []int{i}
		}
		return batches
	}

	// Build dependency graph
	dependencies := make(map[int][]int) // dependencies[i] = list of indices that i depends on
	hasDependency := false

	for i, tc := range toolCalls {
		// Check if this tool's arguments reference other tool call IDs
		deps := te.findDependencies(tc, toolCalls)
		if len(deps) > 0 {
			dependencies[i] = deps
			hasDependency = true
		}
	}

	// If no dependencies found, execute all in parallel
	if !hasDependency {
		batch := make([]int, len(toolCalls))
		for i := range toolCalls {
			batch[i] = i
		}
		return [][]int{batch}
	}

	// Build execution batches using topological sort
	return te.buildExecutionBatches(toolCalls, dependencies)
}

// findDependencies checks if a tool call's arguments reference other tool calls.
// Returns indices of tool calls that this call depends on.
func (te *ToolExecutor) findDependencies(tc ToolCall, allCalls []ToolCall) []int {
	deps := []int{}
	args := tc.Function.Arguments

	// Check if arguments contain references to other tool call IDs
	// Format: "${call_id}" or similar patterns
	for i, otherCall := range allCalls {
		if tc.ID == otherCall.ID {
			continue // Skip self
		}

		// Simple dependency detection: check if other call's ID appears in our arguments
		// This catches patterns like: {"input": "${call_1}"} or {"data": "result_from_call_1"}
		if containsReference(args, otherCall.ID) {
			deps = append(deps, i)
		}
	}

	return deps
}

// containsReference checks if a string contains a reference to a tool call ID.
// Uses word boundaries to avoid false positives (e.g., "call_abc" matching "call_abc123").
func containsReference(args string, callID string) bool {
	if len(args) == 0 || len(callID) == 0 {
		return false
	}

	// Using word boundaries `\b` is more robust than a simple substring check to avoid false positives.
	// regexp.QuoteMeta ensures special regex characters in callID are escaped properly.
	pattern := `\b` + regexp.QuoteMeta(callID) + `\b`

	// An error from regexp.MatchString with a quoted meta string is highly unlikely,
	// but we handle it gracefully by returning false (no match).
	matched, err := regexp.MatchString(pattern, args)
	if err != nil {
		return false
	}
	return matched
}

// buildExecutionBatches creates batches of tool calls that can be executed in order.
// Uses a simple level-based approach where each batch contains calls with no dependencies
// on calls in the same or later batches.
func (te *ToolExecutor) buildExecutionBatches(toolCalls []ToolCall, dependencies map[int][]int) [][]int {
	n := len(toolCalls)
	processed := make([]bool, n)
	batches := [][]int{}

	for {
		// Find all tools that can run in this batch
		// (either no dependencies or all dependencies already processed)
		currentBatch := []int{}

		for i := 0; i < n; i++ {
			if processed[i] {
				continue
			}

			// Check if all dependencies are processed
			canRun := true
			for _, depIdx := range dependencies[i] {
				if !processed[depIdx] {
					canRun = false
					break
				}
			}

			if canRun {
				currentBatch = append(currentBatch, i)
			}
		}

		// If no tools can run, we're done (or have circular dependency)
		if len(currentBatch) == 0 {
			break
		}

		// Mark these as processed
		for _, idx := range currentBatch {
			processed[idx] = true
		}

		batches = append(batches, currentBatch)
	}

	// Check if all tools were processed (no circular dependencies)
	allProcessed := true
	for i := 0; i < n; i++ {
		if !processed[i] {
			allProcessed = false
			te.logf("WARNING: Tool call %d (%s) has unresolved dependencies - executing anyway",
				i, toolCalls[i].ID)
			// Add remaining tools to final batch to ensure they run
			remainingBatch := []int{}
			for j := i; j < n; j++ {
				if !processed[j] {
					remainingBatch = append(remainingBatch, j)
				}
			}
			if len(remainingBatch) > 0 {
				batches = append(batches, remainingBatch)
			}
			break
		}
	}

	if allProcessed && len(batches) > 1 {
		te.logf("Detected dependencies: executing in %d batches (sequential)", len(batches))
	}

	return batches
}

// ConvertToFunctionResponses converts tool call results to genai.FunctionResponse parts.
func (te *ToolExecutor) ConvertToFunctionResponses(results []*ToolCallResult) []*genai.Part {
	parts := make([]*genai.Part, len(results))

	for i, result := range results {
		response := result.Response
		if result.Error != nil && response == nil {
			response = map[string]any{"error": result.Error.Error()}
		}

		parts[i] = genai.NewPartFromFunctionResponse(result.Name, response)
	}

	return parts
}

// logf logs a message if logger is configured.
func (te *ToolExecutor) logf(format string, args ...any) {
	if te.logger != nil {
		te.logger.Printf("[ToolExecutor] "+format, args...)
	}
}
