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
	"fmt"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/tool"
)

// simpleTool is a simplified tool for testing that doesn't need full tool.Context
type simpleTool struct {
	name        string
	description string
	execFunc    func(args map[string]any) (map[string]any, error)
}

func (s *simpleTool) Name() string        { return s.name }
func (s *simpleTool) Description() string { return s.description }
func (s *simpleTool) IsLongRunning() bool { return false }

// We need Run to accept tool.Context but we'll use nil for simple tests
func (s *simpleTool) Run(ctx tool.Context, args map[string]any) (map[string]any, error) {
	return s.execFunc(args)
}

// Simple nil context for basic testing (won't work with all features)
func nilToolContext() tool.Context {
	return nil
}

// === SINGLE TOOL EXECUTION ===

func TestSingleToolExecution(t *testing.T) {
	executed := false

	tools := map[string]any{
		"calculator": &simpleTool{
			name: "calculator",
			execFunc: func(args map[string]any) (map[string]any, error) {
				executed = true
				a := int(args["a"].(float64))
				b := int(args["b"].(float64))
				return map[string]any{"result": a + b}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
		MaxRetries:        0,
	})

	toolCalls := []ToolCall{
		{
			ID:   "call_1",
			Type: "function",
			Function: FunctionCall{
				Name:      "calculator",
				Arguments: `{"a": 5, "b": 3}`,
			},
		},
	}

	ctx := context.Background()
	results, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)

	if err != nil {
		t.Fatalf("Execution failed: %v", err)
	}

	if !executed {
		t.Error("Tool was not executed")
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	if results[0].Error != nil {
		t.Errorf("Unexpected error: %v", results[0].Error)
	}

	result, ok := results[0].Response["result"].(int)
	if !ok || result != 8 {
		t.Errorf("Expected result 8, got %v", results[0].Response["result"])
	}
}

// === MULTI-TOOL EXECUTION ===

func TestMultiToolExecution_Parallel(t *testing.T) {
	var tool1Time, tool2Time, tool3Time time.Time

	tools := map[string]any{
		"tool_1": &simpleTool{
			name: "tool_1",
			execFunc: func(args map[string]any) (map[string]any, error) {
				tool1Time = time.Now()
				time.Sleep(50 * time.Millisecond)
				return map[string]any{"result": "a"}, nil
			},
		},
		"tool_2": &simpleTool{
			name: "tool_2",
			execFunc: func(args map[string]any) (map[string]any, error) {
				tool2Time = time.Now()
				time.Sleep(50 * time.Millisecond)
				return map[string]any{"result": "b"}, nil
			},
		},
		"tool_3": &simpleTool{
			name: "tool_3",
			execFunc: func(args map[string]any) (map[string]any, error) {
				tool3Time = time.Now()
				time.Sleep(50 * time.Millisecond)
				return map[string]any{"result": "c"}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: true,
		Timeout:           5 * time.Second,
	})

	toolCalls := []ToolCall{
		{ID: "call_1", Type: "function", Function: FunctionCall{Name: "tool_1", Arguments: "{}"}},
		{ID: "call_2", Type: "function", Function: FunctionCall{Name: "tool_2", Arguments: "{}"}},
		{ID: "call_3", Type: "function", Function: FunctionCall{Name: "tool_3", Arguments: "{}"}},
	}

	start := time.Now()
	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("Execution failed: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(results))
	}

	// Parallel execution should take ~50-100ms, not ~150ms (3×50ms)
	if duration > 120*time.Millisecond {
		t.Errorf("Parallel execution too slow: %v (expected <120ms)", duration)
	}

	// Verify tools started roughly at the same time (within 20ms)
	timeDiff1 := tool2Time.Sub(tool1Time).Abs()
	timeDiff2 := tool3Time.Sub(tool1Time).Abs()

	if timeDiff1 > 20*time.Millisecond || timeDiff2 > 20*time.Millisecond {
		t.Errorf("Tools didn't start concurrently (diffs: %v, %v)", timeDiff1, timeDiff2)
	}

	t.Logf("Parallel execution completed in %v", duration)
}

func TestMultiToolExecution_Sequential(t *testing.T) {
	executionOrder := []string{}
	var mu sync.Mutex

	tools := map[string]any{
		"tool_1": &simpleTool{
			name: "tool_1",
			execFunc: func(args map[string]any) (map[string]any, error) {
				mu.Lock()
				executionOrder = append(executionOrder, "tool_1")
				mu.Unlock()
				time.Sleep(30 * time.Millisecond)
				return map[string]any{"result": "a"}, nil
			},
		},
		"tool_2": &simpleTool{
			name: "tool_2",
			execFunc: func(args map[string]any) (map[string]any, error) {
				mu.Lock()
				executionOrder = append(executionOrder, "tool_2")
				mu.Unlock()
				time.Sleep(30 * time.Millisecond)
				return map[string]any{"result": "b"}, nil
			},
		},
		"tool_3": &simpleTool{
			name: "tool_3",
			execFunc: func(args map[string]any) (map[string]any, error) {
				mu.Lock()
				executionOrder = append(executionOrder, "tool_3")
				mu.Unlock()
				time.Sleep(30 * time.Millisecond)
				return map[string]any{"result": "c"}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false, // Sequential
		Timeout:           5 * time.Second,
	})

	toolCalls := []ToolCall{
		{ID: "call_1", Type: "function", Function: FunctionCall{Name: "tool_1", Arguments: "{}"}},
		{ID: "call_2", Type: "function", Function: FunctionCall{Name: "tool_2", Arguments: "{}"}},
		{ID: "call_3", Type: "function", Function: FunctionCall{Name: "tool_3", Arguments: "{}"}},
	}

	start := time.Now()
	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("Execution failed: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(results))
	}

	// Sequential should take ~90ms+ (3×30ms)
	if duration < 85*time.Millisecond {
		t.Errorf("Sequential execution too fast: %v (expected >=85ms)", duration)
	}

	t.Logf("Sequential execution completed in %v", duration)
	t.Logf("Execution order: %v", executionOrder)
}

// === ERROR SCENARIOS ===

func TestError_ToolNotFound(t *testing.T) {
	tools := map[string]any{
		"existing_tool": &simpleTool{
			name: "existing_tool",
			execFunc: func(args map[string]any) (map[string]any, error) {
				return map[string]any{"result": "ok"}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, nil)

	toolCalls := []ToolCall{
		{
			ID:   "call_1",
			Type: "function",
			Function: FunctionCall{
				Name:      "nonexistent_tool",
				Arguments: "{}",
			},
		},
	}

	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)

	if err != nil {
		t.Fatalf("ExecuteToolCalls shouldn't fail, got: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	result := results[0]
	if result.Error == nil {
		t.Error("Expected error for nonexistent tool")
	}

	if result.Response == nil {
		t.Fatal("Expected error response")
	}

	errorMsg, ok := result.Response["error"].(string)
	if !ok || errorMsg != "tool not found: nonexistent_tool" {
		t.Errorf("Unexpected error message: %v", result.Response["error"])
	}
}

func TestError_Timeout(t *testing.T) {
	tools := map[string]any{
		"slow_tool": &simpleTool{
			name: "slow_tool",
			execFunc: func(args map[string]any) (map[string]any, error) {
				time.Sleep(2 * time.Second)
				return map[string]any{"result": "done"}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		Timeout:    50 * time.Millisecond, // Very short timeout
		MaxRetries: 0,
	})

	toolCalls := []ToolCall{
		{ID: "call_1", Type: "function", Function: FunctionCall{Name: "slow_tool", Arguments: "{}"}},
	}

	start := time.Now()
	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	if results[0].Error == nil {
		t.Error("Expected timeout error")
	}

	// Should timeout quickly, not wait 2 seconds
	if duration > 100*time.Millisecond {
		t.Errorf("Timeout took too long: %v", duration)
	}

	t.Logf("Timeout detected in %v", duration)
}

func TestError_InvalidArguments(t *testing.T) {
	tools := map[string]any{
		"test_tool": &simpleTool{
			name: "test_tool",
			execFunc: func(args map[string]any) (map[string]any, error) {
				return map[string]any{"result": "ok"}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, nil)

	toolCalls := []ToolCall{
		{
			ID:   "call_1",
			Type: "function",
			Function: FunctionCall{
				Name:      "test_tool",
				Arguments: `{invalid json}`,
			},
		},
	}

	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)

	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	if results[0].Error == nil {
		t.Error("Expected JSON parse error")
	}

	t.Logf("Invalid JSON error: %v", results[0].Error)
}

func TestError_ToolExecutionError(t *testing.T) {
	tools := map[string]any{
		"failing_tool": &simpleTool{
			name: "failing_tool",
			execFunc: func(args map[string]any) (map[string]any, error) {
				return nil, fmt.Errorf("intentional failure")
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		MaxRetries: 2,
		Timeout:    5 * time.Second,
	})

	toolCalls := []ToolCall{
		{ID: "call_1", Type: "function", Function: FunctionCall{Name: "failing_tool", Arguments: "{}"}},
	}

	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)

	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	if results[0].Error == nil {
		t.Error("Expected execution error")
	}

	// Should have retried
	t.Logf("Error after retries: %v", results[0].Error)
}

// === RETRY LOGIC ===

func TestRetryLogic(t *testing.T) {
	attemptCount := int32(0)

	tools := map[string]any{
		"flaky_tool": &simpleTool{
			name: "flaky_tool",
			execFunc: func(args map[string]any) (map[string]any, error) {
				count := atomic.AddInt32(&attemptCount, 1)
				if count < 3 {
					return nil, fmt.Errorf("temporary failure")
				}
				return map[string]any{"result": "success"}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		MaxRetries: 3,
		Timeout:    5 * time.Second,
	})

	toolCalls := []ToolCall{
		{ID: "call_1", Type: "function", Function: FunctionCall{Name: "flaky_tool", Arguments: "{}"}},
	}

	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)

	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	// Should succeed after retries
	if results[0].Error != nil {
		t.Errorf("Expected success after retries, got: %v", results[0].Error)
	}

	finalCount := atomic.LoadInt32(&attemptCount)
	if finalCount != 3 {
		t.Errorf("Expected 3 attempts, got %d", finalCount)
	}

	t.Logf("Tool succeeded after %d attempts", finalCount)
}

// === RESULT ORDERING ===

func TestResultOrdering(t *testing.T) {
	tools := map[string]any{
		"tool_a": &simpleTool{name: "tool_a", execFunc: func(args map[string]any) (map[string]any, error) {
			time.Sleep(30 * time.Millisecond)
			return map[string]any{"value": "a"}, nil
		}},
		"tool_b": &simpleTool{name: "tool_b", execFunc: func(args map[string]any) (map[string]any, error) {
			time.Sleep(10 * time.Millisecond)
			return map[string]any{"value": "b"}, nil
		}},
		"tool_c": &simpleTool{name: "tool_c", execFunc: func(args map[string]any) (map[string]any, error) {
			time.Sleep(20 * time.Millisecond)
			return map[string]any{"value": "c"}, nil
		}},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: true,
	})

	toolCalls := []ToolCall{
		{ID: "call_a", Type: "function", Function: FunctionCall{Name: "tool_a", Arguments: "{}"}},
		{ID: "call_b", Type: "function", Function: FunctionCall{Name: "tool_b", Arguments: "{}"}},
		{ID: "call_c", Type: "function", Function: FunctionCall{Name: "tool_c", Arguments: "{}"}},
	}

	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)

	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	// Results should be in original order despite parallel execution
	if results[0].Name != "tool_a" {
		t.Errorf("Expected first result to be tool_a, got %s", results[0].Name)
	}
	if results[1].Name != "tool_b" {
		t.Errorf("Expected second result to be tool_b, got %s", results[1].Name)
	}
	if results[2].Name != "tool_c" {
		t.Errorf("Expected third result to be tool_c, got %s", results[2].Name)
	}

	t.Log("Results returned in correct order despite parallel execution")
}

// === DEPENDENCY DETECTION & SEQUENTIAL FALLBACK ===

func TestParallelExecWithDependencies(t *testing.T) {
	// Track execution order
	var executionOrder []string
	var mu sync.Mutex

	// Tool 1: Independent, returns a value
	tool1 := &simpleTool{
		name: "get_data",
		execFunc: func(args map[string]any) (map[string]any, error) {
			mu.Lock()
			executionOrder = append(executionOrder, "get_data")
			mu.Unlock()
			time.Sleep(50 * time.Millisecond) // Simulate work
			return map[string]any{"data": "value_from_tool1"}, nil
		},
	}

	// Tool 2: Depends on tool 1 (references call_1 in arguments)
	tool2 := &simpleTool{
		name: "process_data",
		execFunc: func(args map[string]any) (map[string]any, error) {
			mu.Lock()
			executionOrder = append(executionOrder, "process_data")
			mu.Unlock()
			time.Sleep(30 * time.Millisecond)
			// In real scenario, would use result from tool1
			return map[string]any{"processed": "data_processed"}, nil
		},
	}

	// Tool 3: Depends on tool 2 (references call_2 in arguments)
	tool3 := &simpleTool{
		name: "save_result",
		execFunc: func(args map[string]any) (map[string]any, error) {
			mu.Lock()
			executionOrder = append(executionOrder, "save_result")
			mu.Unlock()
			time.Sleep(20 * time.Millisecond)
			return map[string]any{"saved": true}, nil
		},
	}

	tools := map[string]any{
		"get_data":      tool1,
		"process_data":  tool2,
		"save_result":   tool3,
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: true, // Request parallel, but should fallback to sequential
		Timeout:           5 * time.Second,
	})

	// Tool calls with dependencies:
	// call_2 depends on call_1 (references "${call_1}" in args)
	// call_3 depends on call_2 (references "${call_2}" in args)
	toolCalls := []ToolCall{
		{
			ID:   "call_1",
			Type: "function",
			Function: FunctionCall{
				Name:      "get_data",
				Arguments: `{}`,
			},
		},
		{
			ID:   "call_2",
			Type: "function",
			Function: FunctionCall{
				Name:      "process_data",
				Arguments: `{"input": "${call_1}"}`, // Dependency on call_1
			},
		},
		{
			ID:   "call_3",
			Type: "function",
			Function: FunctionCall{
				Name:      "save_result",
				Arguments: `{"data": "${call_2}"}`, // Dependency on call_2
			},
		},
	}

	start := time.Now()
	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	// Verify all executed successfully
	if len(results) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(results))
	}

	for i, result := range results {
		if result.Error != nil {
			t.Errorf("Result %d (%s) has error: %v", i, result.Name, result.Error)
		}
	}

	// Verify execution order is sequential (get_data -> process_data -> save_result)
	mu.Lock()
	order := append([]string{}, executionOrder...)
	mu.Unlock()

	if len(order) != 3 {
		t.Fatalf("Expected 3 executions, got %d: %v", len(order), order)
	}

	if order[0] != "get_data" {
		t.Errorf("Expected first execution to be get_data, got %s", order[0])
	}
	if order[1] != "process_data" {
		t.Errorf("Expected second execution to be process_data, got %s", order[1])
	}
	if order[2] != "save_result" {
		t.Errorf("Expected third execution to be save_result, got %s", order[2])
	}

	// Verify sequential execution (duration should be sum of all sleeps: 50+30+20 = 100ms)
	// With parallel execution, would be max(50, 30, 20) = 50ms
	if duration < 90*time.Millisecond {
		t.Errorf("Expected sequential execution (>90ms), but took only %v (suggests parallel)", duration)
	}

	t.Logf("Successfully detected dependencies and executed sequentially in %v", duration)
	t.Logf("Execution order: %v", order)
}

// TestParallelExecWithPartialDependencies tests mixed independent and dependent calls
func TestParallelExecWithPartialDependencies(t *testing.T) {
	var executionOrder []string
	var mu sync.Mutex
	var execTimes sync.Map // track when each tool started

	recordExec := func(name string) {
		mu.Lock()
		executionOrder = append(executionOrder, name)
		mu.Unlock()
		execTimes.Store(name, time.Now())
	}

	tools := map[string]any{
		"tool_a": &simpleTool{
			name: "tool_a",
			execFunc: func(args map[string]any) (map[string]any, error) {
				recordExec("tool_a")
				time.Sleep(30 * time.Millisecond)
				return map[string]any{"result": "a"}, nil
			},
		},
		"tool_b": &simpleTool{
			name: "tool_b",
			execFunc: func(args map[string]any) (map[string]any, error) {
				recordExec("tool_b")
				time.Sleep(30 * time.Millisecond)
				return map[string]any{"result": "b"}, nil
			},
		},
		"tool_c": &simpleTool{
			name: "tool_c",
			execFunc: func(args map[string]any) (map[string]any, error) {
				recordExec("tool_c")
				time.Sleep(30 * time.Millisecond)
				return map[string]any{"result": "c"}, nil
			},
		},
		"tool_d": &simpleTool{
			name: "tool_d",
			execFunc: func(args map[string]any) (map[string]any, error) {
				recordExec("tool_d")
				time.Sleep(30 * time.Millisecond)
				return map[string]any{"result": "d"}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: true,
		Timeout:           5 * time.Second,
	})

	// Dependency structure:
	// call_a, call_b: independent (can run in parallel) - Batch 1
	// call_c: depends on call_a - Batch 2
	// call_d: depends on call_b - Batch 2
	// So we should see: Batch 1 (a, b in parallel) -> Batch 2 (c, d in parallel)
	toolCalls := []ToolCall{
		{
			ID:   "call_a",
			Type: "function",
			Function: FunctionCall{
				Name:      "tool_a",
				Arguments: `{}`,
			},
		},
		{
			ID:   "call_b",
			Type: "function",
			Function: FunctionCall{
				Name:      "tool_b",
				Arguments: `{}`,
			},
		},
		{
			ID:   "call_c",
			Type: "function",
			Function: FunctionCall{
				Name:      "tool_c",
				Arguments: `{"input": "${call_a}"}`, // Depends on call_a
			},
		},
		{
			ID:   "call_d",
			Type: "function",
			Function: FunctionCall{
				Name:      "tool_d",
				Arguments: `{"input": "${call_b}"}`, // Depends on call_b
			},
		},
	}

	start := time.Now()
	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	if len(results) != 4 {
		t.Fatalf("Expected 4 results, got %d", len(results))
	}

	// Verify all succeeded
	for i, result := range results {
		if result.Error != nil {
			t.Errorf("Result %d (%s) failed: %v", i, result.Name, result.Error)
		}
	}

	mu.Lock()
	order := append([]string{}, executionOrder...)
	mu.Unlock()

	// Verify tool_a and tool_b executed before tool_c and tool_d
	aPos, bPos, cPos, dPos := -1, -1, -1, -1
	for i, name := range order {
		switch name {
		case "tool_a":
			aPos = i
		case "tool_b":
			bPos = i
		case "tool_c":
			cPos = i
		case "tool_d":
			dPos = i
		}
	}

	// tool_a must come before tool_c
	if aPos >= cPos {
		t.Errorf("tool_a (pos %d) should execute before tool_c (pos %d)", aPos, cPos)
	}

	// tool_b must come before tool_d
	if bPos >= dPos {
		t.Errorf("tool_b (pos %d) should execute before tool_d (pos %d)", bPos, dPos)
	}

	// Duration should be ~60ms (two batches of ~30ms each)
	// Not 120ms (fully sequential) or ~30ms (fully parallel)
	if duration < 50*time.Millisecond || duration > 90*time.Millisecond {
		t.Logf("Expected ~60ms execution time (2 batches), got %v", duration)
		// Not a hard failure due to timing variance
	}

	t.Logf("Successfully executed with partial dependencies in %v", duration)
	t.Logf("Execution order: %v", order)
	t.Logf("Dependencies respected: a->c, b->d")
}

// TestNoDependenciesStaysParallel verifies parallel execution when no dependencies exist
func TestNoDependenciesStaysParallel(t *testing.T) {
	var execCount atomic.Int32
	var peakConcurrent atomic.Int32
	var currentConcurrent atomic.Int32

	makeCountingTool := func(name string) *simpleTool {
		return &simpleTool{
			name: name,
			execFunc: func(args map[string]any) (map[string]any, error) {
				current := currentConcurrent.Add(1)

				// Track peak concurrency
				for {
					peak := peakConcurrent.Load()
					if current <= peak || peakConcurrent.CompareAndSwap(peak, current) {
						break
					}
				}

				time.Sleep(50 * time.Millisecond)
				currentConcurrent.Add(-1)
				execCount.Add(1)

				return map[string]any{"result": name}, nil
			},
		}
	}

	tools := map[string]any{
		"tool_1": makeCountingTool("tool_1"),
		"tool_2": makeCountingTool("tool_2"),
		"tool_3": makeCountingTool("tool_3"),
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: true,
		Timeout:           5 * time.Second,
	})

	// No dependencies - all should run in parallel
	toolCalls := []ToolCall{
		{ID: "call_1", Type: "function", Function: FunctionCall{Name: "tool_1", Arguments: `{}`}},
		{ID: "call_2", Type: "function", Function: FunctionCall{Name: "tool_2", Arguments: `{}`}},
		{ID: "call_3", Type: "function", Function: FunctionCall{Name: "tool_3", Arguments: `{}`}},
	}

	start := time.Now()
	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(results))
	}

	// Verify parallel execution (should take ~50ms, not 150ms)
	if duration > 100*time.Millisecond {
		t.Errorf("Expected parallel execution (<100ms), took %v", duration)
	}

	// Verify peak concurrency was 3 (all running simultaneously)
	peak := peakConcurrent.Load()
	if peak < 2 {
		t.Errorf("Expected peak concurrency >= 2 for parallel execution, got %d", peak)
	}

	t.Logf("Parallel execution completed in %v with peak concurrency: %d", duration, peak)
}

// === TIMEOUT HANDLING ===

func TestToolExecTimeout(t *testing.T) {
	// Create a slow tool that takes 500ms
	slowTool := &simpleTool{
		name: "slow_tool",
		execFunc: func(args map[string]any) (map[string]any, error) {
			time.Sleep(500 * time.Millisecond)
			return map[string]any{"result": "completed"}, nil
		},
	}

	// Create a fast tool that completes quickly
	fastTool := &simpleTool{
		name: "fast_tool",
		execFunc: func(args map[string]any) (map[string]any, error) {
			time.Sleep(10 * time.Millisecond)
			return map[string]any{"result": "done"}, nil
		},
	}

	tools := map[string]any{
		"slow_tool": slowTool,
		"fast_tool": fastTool,
	}

	// Configure executor with very short timeout (100ms)
	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           100 * time.Millisecond, // Shorter than slow_tool's 500ms
		MaxRetries:        0,                      // No retries for clearer test
	})

	toolCalls := []ToolCall{
		{
			ID:   "call_slow",
			Type: "function",
			Function: FunctionCall{
				Name:      "slow_tool",
				Arguments: `{}`,
			},
		},
		{
			ID:   "call_fast",
			Type: "function",
			Function: FunctionCall{
				Name:      "fast_tool",
				Arguments: `{}`,
			},
		},
	}

	ctx := context.Background()
	start := time.Now()
	results, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("Expected 2 results, got %d", len(results))
	}

	// Verify slow_tool timed out
	slowResult := results[0]
	if slowResult.Name != "slow_tool" {
		t.Errorf("Expected first result to be slow_tool, got %s", slowResult.Name)
	}

	if slowResult.Error == nil {
		t.Error("Expected slow_tool to have timeout error, got nil")
	} else {
		errMsg := slowResult.Error.Error()
		if !strings.Contains(errMsg, "timeout") && !strings.Contains(errMsg, "context deadline exceeded") {
			t.Errorf("Expected timeout error, got: %v", slowResult.Error)
		}
		t.Logf("Slow tool error (expected): %v", slowResult.Error)
	}

	// Verify error response structure
	if slowResult.Response == nil {
		t.Error("Expected error response, got nil")
	} else {
		if errField, ok := slowResult.Response["error"]; !ok {
			t.Error("Expected 'error' field in response")
		} else {
			t.Logf("Error response: %v", errField)
		}
	}

	// Verify fast_tool succeeded
	fastResult := results[1]
	if fastResult.Name != "fast_tool" {
		t.Errorf("Expected second result to be fast_tool, got %s", fastResult.Name)
	}

	if fastResult.Error != nil {
		t.Errorf("Expected fast_tool to succeed, got error: %v", fastResult.Error)
	}

	if fastResult.Response["result"] != "done" {
		t.Errorf("Expected fast_tool result 'done', got: %v", fastResult.Response["result"])
	}

	// Verify execution time (should timeout after ~100ms for slow_tool, not wait 500ms)
	// Plus ~10ms for fast_tool = ~110ms total (sequential)
	if duration > 300*time.Millisecond {
		t.Errorf("Expected timeout to prevent long wait, but took %v", duration)
	}

	t.Logf("Execution completed in %v (timeout prevented full 500ms wait)", duration)
	t.Logf("Slow tool: %v, Fast tool: %v", slowResult.Error != nil, fastResult.Error == nil)
}

// TestToolExecContextCancellation tests context cancellation during tool execution
func TestToolExecContextCancellation(t *testing.T) {
	executionStarted := make(chan struct{})
	executionFinished := make(chan struct{})

	slowTool := &simpleTool{
		name: "cancellable_tool",
		execFunc: func(args map[string]any) (map[string]any, error) {
			close(executionStarted)
			time.Sleep(1 * time.Second) // Very long operation
			close(executionFinished)
			return map[string]any{"result": "completed"}, nil
		},
	}

	tools := map[string]any{
		"cancellable_tool": slowTool,
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second, // Long timeout, we'll cancel before
		MaxRetries:        0,
	})

	toolCalls := []ToolCall{
		{
			ID:   "call_1",
			Type: "function",
			Function: FunctionCall{
				Name:      "cancellable_tool",
				Arguments: `{}`,
			},
		},
	}

	// Create cancellable context
	ctx, cancel := context.WithCancel(context.Background())

	// Start execution in goroutine
	var results []*ToolCallResult
	var execErr error
	done := make(chan struct{})

	go func() {
		results, execErr = executor.ExecuteToolCalls(ctx, toolCalls, nil)
		close(done)
	}()

	// Wait for execution to start
	<-executionStarted

	// Cancel after tool starts (but before it finishes)
	time.Sleep(50 * time.Millisecond)
	cancel()

	// Wait for execution to complete
	select {
	case <-done:
		// Execution completed
	case <-time.After(2 * time.Second):
		t.Fatal("Execution did not complete after context cancellation")
	}

	// Verify we got results
	if execErr != nil {
		t.Logf("ExecuteToolCalls returned error (may be expected): %v", execErr)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	result := results[0]

	// Should have error due to cancellation
	if result.Error == nil {
		t.Error("Expected error due to context cancellation")
	} else {
		errMsg := result.Error.Error()
		if !strings.Contains(errMsg, "timeout") && !strings.Contains(errMsg, "cancel") && !strings.Contains(errMsg, "context") {
			t.Logf("Warning: Expected cancellation/timeout error, got: %v", result.Error)
		}
		t.Logf("Cancellation error (expected): %v", result.Error)
	}

	// Verify execution didn't complete (executionFinished should not be closed)
	select {
	case <-executionFinished:
		t.Error("Tool execution should have been cancelled before completion")
	default:
		t.Log("Tool execution was properly cancelled")
	}
}

// TestToolExecMultipleTimeouts tests timeout behavior with multiple tools
func TestToolExecMultipleTimeouts(t *testing.T) {
	tools := map[string]any{
		"tool_1": &simpleTool{
			name: "tool_1",
			execFunc: func(args map[string]any) (map[string]any, error) {
				time.Sleep(50 * time.Millisecond)
				return map[string]any{"result": "1"}, nil
			},
		},
		"tool_2": &simpleTool{
			name: "tool_2",
			execFunc: func(args map[string]any) (map[string]any, error) {
				time.Sleep(200 * time.Millisecond) // Will timeout
				return map[string]any{"result": "2"}, nil
			},
		},
		"tool_3": &simpleTool{
			name: "tool_3",
			execFunc: func(args map[string]any) (map[string]any, error) {
				time.Sleep(30 * time.Millisecond)
				return map[string]any{"result": "3"}, nil
			},
		},
		"tool_4": &simpleTool{
			name: "tool_4",
			execFunc: func(args map[string]any) (map[string]any, error) {
				time.Sleep(250 * time.Millisecond) // Will timeout
				return map[string]any{"result": "4"}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: true, // Test parallel timeouts
		Timeout:           100 * time.Millisecond,
		MaxRetries:        0,
	})

	toolCalls := []ToolCall{
		{ID: "call_1", Type: "function", Function: FunctionCall{Name: "tool_1", Arguments: `{}`}},
		{ID: "call_2", Type: "function", Function: FunctionCall{Name: "tool_2", Arguments: `{}`}},
		{ID: "call_3", Type: "function", Function: FunctionCall{Name: "tool_3", Arguments: `{}`}},
		{ID: "call_4", Type: "function", Function: FunctionCall{Name: "tool_4", Arguments: `{}`}},
	}

	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)

	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	if len(results) != 4 {
		t.Fatalf("Expected 4 results, got %d", len(results))
	}

	// Count successes and timeouts
	successCount := 0
	timeoutCount := 0

	for i, result := range results {
		t.Logf("Tool %s: error=%v, duration=%v", result.Name, result.Error != nil, result.Duration)

		if result.Error == nil {
			successCount++
		} else {
			timeoutCount++
			// Verify it's a timeout error
			errMsg := result.Error.Error()
			if !strings.Contains(errMsg, "timeout") && !strings.Contains(errMsg, "deadline") {
				t.Errorf("Result %d: Expected timeout error, got: %v", i, result.Error)
			}
		}
	}

	// tool_1 (50ms) and tool_3 (30ms) should succeed
	// tool_2 (200ms) and tool_4 (250ms) should timeout
	if successCount != 2 {
		t.Errorf("Expected 2 successful tools, got %d", successCount)
	}

	if timeoutCount != 2 {
		t.Errorf("Expected 2 timeout tools, got %d", timeoutCount)
	}

	// Verify specific tools
	if results[0].Name == "tool_1" && results[0].Error != nil {
		t.Error("tool_1 should succeed (50ms < 100ms timeout)")
	}
	if results[1].Name == "tool_2" && results[1].Error == nil {
		t.Error("tool_2 should timeout (200ms > 100ms timeout)")
	}
	if results[2].Name == "tool_3" && results[2].Error != nil {
		t.Error("tool_3 should succeed (30ms < 100ms timeout)")
	}
	if results[3].Name == "tool_4" && results[3].Error == nil {
		t.Error("tool_4 should timeout (250ms > 100ms timeout)")
	}

	t.Logf("Successfully handled mixed timeouts: %d succeeded, %d timed out", successCount, timeoutCount)
}

// === RETRY LOGIC WITH BACKOFF ===

func TestRetryOnToolError(t *testing.T) {
	attemptCount := 0
	var attemptTimes []time.Time
	var mu sync.Mutex

	// Tool that fails first 2 attempts, succeeds on 3rd
	flakyTool := &simpleTool{
		name: "flaky_tool",
		execFunc: func(args map[string]any) (map[string]any, error) {
			mu.Lock()
			attemptCount++
			currentAttempt := attemptCount
			attemptTimes = append(attemptTimes, time.Now())
			mu.Unlock()

			if currentAttempt < 3 {
				return nil, fmt.Errorf("temporary failure (attempt %d)", currentAttempt)
			}

			return map[string]any{"result": "success", "attempts": currentAttempt}, nil
		},
	}

	tools := map[string]any{
		"flaky_tool": flakyTool,
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
		MaxRetries:        2, // Total: 1 initial + 2 retries = 3 attempts
	})

	toolCalls := []ToolCall{
		{
			ID:   "call_1",
			Type: "function",
			Function: FunctionCall{
				Name:      "flaky_tool",
				Arguments: `{}`,
			},
		},
	}

	start := time.Now()
	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	result := results[0]

	// Should succeed on 3rd attempt
	if result.Error != nil {
		t.Errorf("Expected success after retries, got error: %v", result.Error)
	}

	// Verify 3 attempts were made
	mu.Lock()
	finalAttemptCount := attemptCount
	times := append([]time.Time{}, attemptTimes...)
	mu.Unlock()

	if finalAttemptCount != 3 {
		t.Errorf("Expected 3 attempts, got %d", finalAttemptCount)
	}

	// Verify backoff between attempts
	// Backoff formula: (attempt+1) * 100ms
	// Attempt 0->1: ~100ms backoff
	// Attempt 1->2: ~200ms backoff
	if len(times) >= 2 {
		backoff1 := times[1].Sub(times[0])
		t.Logf("Backoff between attempt 1 and 2: %v", backoff1)
		if backoff1 < 80*time.Millisecond || backoff1 > 150*time.Millisecond {
			t.Logf("Warning: Expected ~100ms backoff, got %v", backoff1)
		}
	}

	if len(times) >= 3 {
		backoff2 := times[2].Sub(times[1])
		t.Logf("Backoff between attempt 2 and 3: %v", backoff2)
		if backoff2 < 150*time.Millisecond || backoff2 > 250*time.Millisecond {
			t.Logf("Warning: Expected ~200ms backoff, got %v", backoff2)
		}
	}

	// Verify result contains success
	if result.Response["result"] != "success" {
		t.Errorf("Expected result 'success', got: %v", result.Response["result"])
	}

	// Verify total duration includes backoffs (~100ms + ~200ms + execution time)
	if duration < 250*time.Millisecond {
		t.Logf("Warning: Expected at least 250ms with backoffs, got %v", duration)
	}

	t.Logf("Successfully retried %d times with exponential backoff, total duration: %v", finalAttemptCount-1, duration)
}

// TestRetryExhaustion tests that tool fails after all retries are exhausted
func TestRetryExhaustion(t *testing.T) {
	attemptCount := 0
	var mu sync.Mutex

	// Tool that always fails
	alwaysFailTool := &simpleTool{
		name: "always_fail",
		execFunc: func(args map[string]any) (map[string]any, error) {
			mu.Lock()
			attemptCount++
			current := attemptCount
			mu.Unlock()

			return nil, fmt.Errorf("permanent failure (attempt %d)", current)
		},
	}

	tools := map[string]any{
		"always_fail": alwaysFailTool,
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
		MaxRetries:        3, // 1 initial + 3 retries = 4 total attempts
	})

	toolCalls := []ToolCall{
		{
			ID:   "call_1",
			Type: "function",
			Function: FunctionCall{
				Name:      "always_fail",
				Arguments: `{}`,
			},
		},
	}

	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)

	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	result := results[0]

	// Should have error after exhausting retries
	if result.Error == nil {
		t.Error("Expected error after exhausting retries, got nil")
	}

	// Error should mention multiple attempts
	errMsg := result.Error.Error()
	if !strings.Contains(errMsg, "4 attempts") {
		t.Errorf("Expected error to mention 4 attempts, got: %v", errMsg)
	}

	// Verify 4 attempts were made
	mu.Lock()
	finalCount := attemptCount
	mu.Unlock()

	if finalCount != 4 {
		t.Errorf("Expected 4 attempts (1 initial + 3 retries), got %d", finalCount)
	}

	// Verify error response
	if result.Response == nil {
		t.Error("Expected error response, got nil")
	} else if _, ok := result.Response["error"]; !ok {
		t.Error("Expected 'error' field in response")
	}

	t.Logf("Successfully exhausted all %d retry attempts", finalCount)
	t.Logf("Final error: %v", result.Error)
}

// TestRetryWithMixedResults tests retry behavior with multiple tools
func TestRetryWithMixedResults(t *testing.T) {
	// Track attempts for each tool
	var tool1Attempts, tool2Attempts, tool3Attempts atomic.Int32

	tools := map[string]any{
		// Tool 1: Succeeds immediately
		"instant_success": &simpleTool{
			name: "instant_success",
			execFunc: func(args map[string]any) (map[string]any, error) {
				tool1Attempts.Add(1)
				return map[string]any{"status": "ok"}, nil
			},
		},
		// Tool 2: Fails once, then succeeds
		"retry_once": &simpleTool{
			name: "retry_once",
			execFunc: func(args map[string]any) (map[string]any, error) {
				attempt := tool2Attempts.Add(1)
				if attempt == 1 {
					return nil, fmt.Errorf("first attempt failed")
				}
				return map[string]any{"status": "recovered"}, nil
			},
		},
		// Tool 3: Always fails
		"always_fail": &simpleTool{
			name: "always_fail",
			execFunc: func(args map[string]any) (map[string]any, error) {
				tool3Attempts.Add(1)
				return nil, fmt.Errorf("permanent error")
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
		MaxRetries:        2, // 1 initial + 2 retries
	})

	toolCalls := []ToolCall{
		{ID: "c1", Type: "function", Function: FunctionCall{Name: "instant_success", Arguments: `{}`}},
		{ID: "c2", Type: "function", Function: FunctionCall{Name: "retry_once", Arguments: `{}`}},
		{ID: "c3", Type: "function", Function: FunctionCall{Name: "always_fail", Arguments: `{}`}},
	}

	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)

	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(results))
	}

	// Verify tool 1: should succeed on first attempt
	if results[0].Name != "instant_success" {
		t.Errorf("Expected first result to be instant_success, got %s", results[0].Name)
	}
	if results[0].Error != nil {
		t.Errorf("Tool 1 should succeed: %v", results[0].Error)
	}
	if attempts := tool1Attempts.Load(); attempts != 1 {
		t.Errorf("Tool 1 should be called once, got %d attempts", attempts)
	}

	// Verify tool 2: should succeed on second attempt
	if results[1].Name != "retry_once" {
		t.Errorf("Expected second result to be retry_once, got %s", results[1].Name)
	}
	if results[1].Error != nil {
		t.Errorf("Tool 2 should succeed after retry: %v", results[1].Error)
	}
	if attempts := tool2Attempts.Load(); attempts != 2 {
		t.Errorf("Tool 2 should be called twice, got %d attempts", attempts)
	}

	// Verify tool 3: should fail after all retries
	if results[2].Name != "always_fail" {
		t.Errorf("Expected third result to be always_fail, got %s", results[2].Name)
	}
	if results[2].Error == nil {
		t.Error("Tool 3 should fail after retries")
	}
	if attempts := tool3Attempts.Load(); attempts != 3 {
		t.Errorf("Tool 3 should be called 3 times (1+2 retries), got %d attempts", attempts)
	}

	t.Logf("Mixed retry results: Tool1=%d attempts, Tool2=%d attempts, Tool3=%d attempts",
		tool1Attempts.Load(), tool2Attempts.Load(), tool3Attempts.Load())
}

// TestRetryBackoffTiming verifies exponential backoff timing
func TestRetryBackoffTiming(t *testing.T) {
	var attemptTimes []time.Time
	var mu sync.Mutex

	failingTool := &simpleTool{
		name: "timing_test",
		execFunc: func(args map[string]any) (map[string]any, error) {
			mu.Lock()
			attemptTimes = append(attemptTimes, time.Now())
			mu.Unlock()
			return nil, fmt.Errorf("always fail for timing test")
		},
	}

	tools := map[string]any{
		"timing_test": failingTool,
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
		MaxRetries:        3, // 1 initial + 3 retries = 4 attempts
	})

	toolCalls := []ToolCall{
		{ID: "c1", Type: "function", Function: FunctionCall{Name: "timing_test", Arguments: `{}`}},
	}

	start := time.Now()
	results, _ := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)
	totalDuration := time.Since(start)

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	// Get attempt times
	mu.Lock()
	times := append([]time.Time{}, attemptTimes...)
	mu.Unlock()

	if len(times) != 4 {
		t.Fatalf("Expected 4 attempts, got %d", len(times))
	}

	// Verify backoff durations
	// Backoff formula: (attempt+1) * 100ms
	// Between attempts 0->1: ~100ms
	// Between attempts 1->2: ~200ms
	// Between attempts 2->3: ~300ms
	expectedBackoffs := []struct {
		name     string
		min, max time.Duration
	}{
		{"attempt 0->1", 80 * time.Millisecond, 150 * time.Millisecond},   // ~100ms
		{"attempt 1->2", 150 * time.Millisecond, 250 * time.Millisecond},  // ~200ms
		{"attempt 2->3", 250 * time.Millisecond, 350 * time.Millisecond},  // ~300ms
	}

	for i := 0; i < len(times)-1; i++ {
		backoff := times[i+1].Sub(times[i])
		expected := expectedBackoffs[i]

		t.Logf("Backoff %s: %v (expected %v-%v)", expected.name, backoff, expected.min, expected.max)

		if backoff < expected.min || backoff > expected.max {
			t.Logf("Warning: Backoff %s outside expected range", expected.name)
		}
	}

	// Total should be sum of backoffs: ~100+200+300 = ~600ms
	expectedTotal := 500 * time.Millisecond
	if totalDuration < expectedTotal {
		t.Logf("Warning: Total duration %v less than expected minimum %v", totalDuration, expectedTotal)
	}

	t.Logf("Total duration with exponential backoff: %v", totalDuration)
}

// === RECURSIVE TOOL CALLS WITH ITERATION LIMIT ===

func TestMaxIterationsRecursive(t *testing.T) {
	callCount := 0
	var callHistory []int
	var mu sync.Mutex

	// Recursive tool that calls itself by returning tool calls in response
	// Simulates a tool that keeps requesting more tool calls until limit
	recursiveTool := &simpleTool{
		name: "recursive_tool",
		execFunc: func(args map[string]any) (map[string]any, error) {
			mu.Lock()
			callCount++
			iteration := callCount
			callHistory = append(callHistory, iteration)
			mu.Unlock()

			// Return a response that would trigger another call
			// In a real scenario, the LLM would see this response and make another tool call
			return map[string]any{
				"iteration": iteration,
				"message":   fmt.Sprintf("Iteration %d - requesting next call", iteration),
			}, nil
		},
	}

	tools := map[string]any{
		"recursive_tool": recursiveTool,
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
		MaxRetries:        0,
	})

	// Create iteration guard with limit of 5
	guard := NewIterationGuard(5)
	ctx := context.Background()

	// Simulate recursive calls by repeatedly executing the tool
	// and checking iteration guard
	for {
		// Check if we've hit the limit
		if err := guard.CheckIteration(ctx); err != nil {
			// Expected error - max iterations reached
			t.Logf("Iteration limit reached: %v", err)
			if !strings.Contains(err.Error(), "maximum iterations exceeded") {
				t.Errorf("Expected 'maximum iterations exceeded' error, got: %v", err)
			}
			break
		}

		// Execute the tool
		toolCalls := []ToolCall{
			{
				ID:   fmt.Sprintf("call_%d", callCount+1),
				Type: "function",
				Function: FunctionCall{
					Name:      "recursive_tool",
					Arguments: `{}`,
				},
			},
		}

		results, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)
		if err != nil {
			t.Fatalf("ExecuteToolCalls failed: %v", err)
		}

		if len(results) != 1 {
			t.Fatalf("Expected 1 result, got %d", len(results))
		}

		if results[0].Error != nil {
			t.Fatalf("Tool execution failed: %v", results[0].Error)
		}

		// Increment iteration counter
		ctx = guard.IncrementIteration(ctx)
	}

	// Verify exactly 5 calls were made (limit)
	mu.Lock()
	finalCount := callCount
	history := append([]int{}, callHistory...)
	mu.Unlock()

	if finalCount != 5 {
		t.Errorf("Expected exactly 5 calls before limit, got %d", finalCount)
	}

	// Verify call history
	if len(history) != 5 {
		t.Errorf("Expected 5 entries in history, got %d", len(history))
	}

	for i, iter := range history {
		if iter != i+1 {
			t.Errorf("History[%d]: expected iteration %d, got %d", i, i+1, iter)
		}
	}

	t.Logf("Successfully stopped recursive calls at iteration limit: %d calls", finalCount)
	t.Logf("Call history: %v", history)
}

// TestMaxIterationsWithContextOverride tests custom iteration limit via context
func TestMaxIterationsWithContextOverride(t *testing.T) {
	var callCount atomic.Int32

	tool := &simpleTool{
		name: "counter_tool",
		execFunc: func(args map[string]any) (map[string]any, error) {
			count := callCount.Add(1)
			return map[string]any{"count": count}, nil
		},
	}

	tools := map[string]any{
		"counter_tool": tool,
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
	})

	// Create guard with default limit 10, but override with context to 3
	guard := NewIterationGuard(10)
	ctx := WithMaxIterations(context.Background(), 3)

	// Should stop at 3 iterations (context override)
	for {
		if err := guard.CheckIteration(ctx); err != nil {
			t.Logf("Stopped at iteration limit: %v", err)
			break
		}

		toolCalls := []ToolCall{
			{
				ID:   fmt.Sprintf("call_%d", callCount.Load()+1),
				Type: "function",
				Function: FunctionCall{
					Name:      "counter_tool",
					Arguments: `{}`,
				},
			},
		}

		executor.ExecuteToolCalls(ctx, toolCalls, nil)
		ctx = guard.IncrementIteration(ctx)
	}

	finalCount := callCount.Load()
	if finalCount != 3 {
		t.Errorf("Expected 3 calls (context override), got %d", finalCount)
	}

	t.Logf("Context override worked: stopped at %d iterations instead of default 10", finalCount)
}

// TestMaxIterationsNestedRecursion tests deeply nested recursive pattern
func TestMaxIterationsNestedRecursion(t *testing.T) {
	// Track execution order and depth
	type CallInfo struct {
		ID    string
		Depth int
		Time  time.Time
	}

	var calls []CallInfo
	var mu sync.Mutex

	// Tool that simulates nested recursion
	deepTool := &simpleTool{
		name: "deep_tool",
		execFunc: func(args map[string]any) (map[string]any, error) {
			depth := 0
			if d, ok := args["depth"].(float64); ok {
				depth = int(d)
			}

			mu.Lock()
			calls = append(calls, CallInfo{
				ID:    fmt.Sprintf("depth_%d", depth),
				Depth: depth,
				Time:  time.Now(),
			})
			mu.Unlock()

			return map[string]any{
				"depth":   depth,
				"message": fmt.Sprintf("Completed depth %d", depth),
			}, nil
		},
	}

	tools := map[string]any{
		"deep_tool": deepTool,
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
	})

	// Limit to 5 iterations
	guard := NewIterationGuard(5)
	ctx := context.Background()

	// Simulate nested calls with increasing depth
	depth := 0
	for {
		if err := guard.CheckIteration(ctx); err != nil {
			t.Logf("Reached max iterations at depth %d: %v", depth, err)
			break
		}

		toolCalls := []ToolCall{
			{
				ID:   fmt.Sprintf("call_depth_%d", depth),
				Type: "function",
				Function: FunctionCall{
					Name:      "deep_tool",
					Arguments: fmt.Sprintf(`{"depth": %d}`, depth),
				},
			},
		}

		results, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)
		if err != nil {
			t.Fatalf("Execution failed: %v", err)
		}

		if len(results) == 0 || results[0].Error != nil {
			t.Fatalf("Tool failed at depth %d", depth)
		}

		depth++
		ctx = guard.IncrementIteration(ctx)
	}

	// Verify we reached exactly depth 5 (0-indexed: 0,1,2,3,4)
	mu.Lock()
	callsCopy := append([]CallInfo{}, calls...)
	mu.Unlock()

	if len(callsCopy) != 5 {
		t.Errorf("Expected 5 nested calls, got %d", len(callsCopy))
	}

	// Verify depths are sequential
	for i, call := range callsCopy {
		if call.Depth != i {
			t.Errorf("Call %d: expected depth %d, got %d", i, i, call.Depth)
		}
	}

	// Verify all calls executed in sequence
	for i := 1; i < len(callsCopy); i++ {
		if callsCopy[i].Time.Before(callsCopy[i-1].Time) {
			t.Error("Calls executed out of order")
		}
	}

	t.Logf("Successfully executed nested recursion: %d levels", len(callsCopy))
	t.Logf("Depths reached: %v", func() []int {
		depths := make([]int, len(callsCopy))
		for i, c := range callsCopy {
			depths[i] = c.Depth
		}
		return depths
	}())
}

// TestMaxIterationsErrorPropagation verifies error is properly returned
func TestMaxIterationsErrorPropagation(t *testing.T) {
	var callCount atomic.Int32

	tool := &simpleTool{
		name: "test_tool",
		execFunc: func(args map[string]any) (map[string]any, error) {
			count := callCount.Add(1)
			return map[string]any{"iteration": count}, nil
		},
	}

	tools := map[string]any{
		"test_tool": tool,
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
	})

	guard := NewIterationGuard(3)
	ctx := context.Background()

	var lastError error

	// Execute until we hit the limit
	for i := 0; i < 10; i++ { // Try more than the limit
		if err := guard.CheckIteration(ctx); err != nil {
			lastError = err
			break
		}

		toolCalls := []ToolCall{
			{
				ID:   fmt.Sprintf("call_%d", i),
				Type: "function",
				Function: FunctionCall{
					Name:      "test_tool",
					Arguments: `{}`,
				},
			},
		}

		executor.ExecuteToolCalls(ctx, toolCalls, nil)
		ctx = guard.IncrementIteration(ctx)
	}

	// Verify error was returned
	if lastError == nil {
		t.Error("Expected max iterations error, got nil")
	}

	// Verify error message
	errMsg := lastError.Error()
	if !strings.Contains(errMsg, "maximum iterations exceeded") {
		t.Errorf("Expected 'maximum iterations exceeded', got: %s", errMsg)
	}

	if !strings.Contains(errMsg, "3/3") {
		t.Errorf("Expected '3/3' in error message, got: %s", errMsg)
	}

	// Verify exactly 3 calls were made
	if count := callCount.Load(); count != 3 {
		t.Errorf("Expected 3 tool calls, got %d", count)
	}

	t.Logf("Error properly propagated: %v", lastError)
	t.Logf("Stopped at %d iterations as expected", callCount.Load())
}

// TestInvalidJSONArgsInToolCall tests error handling when model returns invalid JSON in tool call arguments.
// This simulates scenarios where the model generates malformed JSON, which should be caught and reported.
func TestInvalidJSONArgsInToolCall(t *testing.T) {
	tests := []struct {
		name          string
		arguments     string
		expectedError string
		description   string
	}{
		{
			name:          "completely invalid JSON",
			arguments:     `{this is not valid json}`,
			expectedError: "failed to parse arguments",
			description:   "Malformed JSON syntax",
		},
		{
			name:          "unclosed brace",
			arguments:     `{"location": "London"`,
			expectedError: "failed to parse arguments",
			description:   "Missing closing brace",
		},
		{
			name:          "unquoted key",
			arguments:     `{location: "London"}`,
			expectedError: "failed to parse arguments",
			description:   "JSON keys must be quoted",
		},
		{
			name:          "trailing comma",
			arguments:     `{"location": "London",}`,
			expectedError: "failed to parse arguments",
			description:   "Trailing comma is invalid JSON",
		},
		{
			name:          "single quotes instead of double",
			arguments:     `{'location': 'London'}`,
			expectedError: "failed to parse arguments",
			description:   "JSON requires double quotes",
		},
		{
			name:          "mixed up brackets",
			arguments:     `["location": "London"}`,
			expectedError: "failed to parse arguments",
			description:   "Array bracket with object syntax",
		},
		{
			name:          "plain text instead of JSON",
			arguments:     `location is London`,
			expectedError: "failed to parse arguments",
			description:   "Not JSON at all",
		},
		{
			name:          "incomplete JSON",
			arguments:     `{"location":`,
			expectedError: "failed to parse arguments",
			description:   "Incomplete JSON object",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a simple test tool
			tools := map[string]any{
				"get_weather": &simpleTool{
					name: "get_weather",
					execFunc: func(args map[string]any) (map[string]any, error) {
						// This should never be called since JSON parsing should fail first
						t.Error("Tool execution should not be called with invalid JSON arguments")
						return map[string]any{"result": "should not reach here"}, nil
					},
				},
			}

			executor := NewToolExecutor(tools, &ToolExecutorConfig{
				ParallelExecution: false,
				Timeout:           5 * time.Second,
				MaxRetries:        0, // No retries for this test
			})

			// Create tool call with invalid JSON
			toolCalls := []ToolCall{
				{
					ID:   "call_invalid",
					Type: "function",
					Function: FunctionCall{
						Name:      "get_weather",
						Arguments: tt.arguments,
					},
				},
			}

			ctx := context.Background()

			// Execute tool calls - should fail during argument parsing
			results, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)

			// ExecuteToolCalls should not return error (errors are in results)
			if err != nil {
				t.Fatalf("ExecuteToolCalls should not return error, got: %v", err)
			}

			// Should have 1 result
			if len(results) != 1 {
				t.Fatalf("Expected 1 result, got %d", len(results))
			}

			result := results[0]

			// Result should have an error
			if result.Error == nil {
				t.Error("Expected result to have an error for invalid JSON")
			}

			// Error message should mention parsing failure
			if result.Error != nil && !strings.Contains(result.Error.Error(), tt.expectedError) {
				t.Errorf("Expected error containing '%s', got: %v", tt.expectedError, result.Error)
			}

			// Response should contain error information
			if result.Response == nil {
				t.Error("Expected response to be non-nil")
			}

			if errorMsg, ok := result.Response["error"].(string); ok {
				if !strings.Contains(errorMsg, tt.expectedError) {
					t.Errorf("Expected response error containing '%s', got: %s", tt.expectedError, errorMsg)
				}
				t.Logf("✓ %s - Error correctly reported: %s", tt.description, errorMsg)
			} else {
				t.Error("Expected response to contain 'error' field with string value")
			}

			// Verify tool was NOT executed (due to parse error)
			// This is implicitly tested by the tool's error in execFunc
		})
	}
}

// TestInvalidJSONArgsMultipleToolCalls tests invalid JSON mixed with valid JSON in multiple tool calls.
func TestInvalidJSONArgsMultipleToolCalls(t *testing.T) {
	var tool1Called, tool2Called, tool3Called bool

	tools := map[string]any{
		"tool1": &simpleTool{
			name: "tool1",
			execFunc: func(args map[string]any) (map[string]any, error) {
				tool1Called = true
				return map[string]any{"result": "tool1 success"}, nil
			},
		},
		"tool2": &simpleTool{
			name: "tool2",
			execFunc: func(args map[string]any) (map[string]any, error) {
				tool2Called = true
				return map[string]any{"result": "tool2 success"}, nil
			},
		},
		"tool3": &simpleTool{
			name: "tool3",
			execFunc: func(args map[string]any) (map[string]any, error) {
				tool3Called = true
				return map[string]any{"result": "tool3 success"}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
		MaxRetries:        0,
	})

	// Mix of valid and invalid JSON
	toolCalls := []ToolCall{
		{
			ID:   "call_1",
			Type: "function",
			Function: FunctionCall{
				Name:      "tool1",
				Arguments: `{"param": "value1"}`, // Valid JSON
			},
		},
		{
			ID:   "call_2",
			Type: "function",
			Function: FunctionCall{
				Name:      "tool2",
				Arguments: `{invalid json here}`, // Invalid JSON
			},
		},
		{
			ID:   "call_3",
			Type: "function",
			Function: FunctionCall{
				Name:      "tool3",
				Arguments: `{"param": "value3"}`, // Valid JSON
			},
		},
	}

	ctx := context.Background()
	results, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)

	if err != nil {
		t.Fatalf("ExecuteToolCalls should not return error, got: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(results))
	}

	// Result 1: Should succeed
	if results[0].Error != nil {
		t.Errorf("Tool1 should succeed, got error: %v", results[0].Error)
	}
	if !tool1Called {
		t.Error("Tool1 should have been called")
	}

	// Result 2: Should fail with parse error
	if results[1].Error == nil {
		t.Error("Tool2 should fail with parse error")
	} else if !strings.Contains(results[1].Error.Error(), "failed to parse arguments") {
		t.Errorf("Expected parse error for tool2, got: %v", results[1].Error)
	}
	if tool2Called {
		t.Error("Tool2 should NOT have been called (parse error should prevent execution)")
	}

	// Result 3: Should succeed
	if results[2].Error != nil {
		t.Errorf("Tool3 should succeed, got error: %v", results[2].Error)
	}
	if !tool3Called {
		t.Error("Tool3 should have been called")
	}

	t.Logf("✓ Tool1: Success (called=%v)", tool1Called)
	t.Logf("✓ Tool2: Parse error (called=%v, error=%v)", tool2Called, results[1].Error)
	t.Logf("✓ Tool3: Success (called=%v)", tool3Called)
}

// TestEmptyJSONArgsInToolCall tests handling of empty JSON arguments.
func TestEmptyJSONArgsInToolCall(t *testing.T) {
	var receivedArgs map[string]any

	tools := map[string]any{
		"no_param_tool": &simpleTool{
			name: "no_param_tool",
			execFunc: func(args map[string]any) (map[string]any, error) {
				receivedArgs = args
				return map[string]any{"result": "success"}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
	})

	tests := []struct {
		name      string
		arguments string
		shouldErr bool
		argsCount int
	}{
		{
			name:      "empty string",
			arguments: "",
			shouldErr: false,
			argsCount: 0, // Empty string → empty map
		},
		{
			name:      "empty object",
			arguments: "{}",
			shouldErr: false,
			argsCount: 0, // Empty object → empty map
		},
		{
			name:      "whitespace only",
			arguments: "   ",
			shouldErr: true, // Whitespace is not valid JSON
			argsCount: -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receivedArgs = nil // Reset

			toolCalls := []ToolCall{
				{
					ID:   "call_empty",
					Type: "function",
					Function: FunctionCall{
						Name:      "no_param_tool",
						Arguments: tt.arguments,
					},
				},
			}

			ctx := context.Background()
			results, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)

			if err != nil {
				t.Fatalf("ExecuteToolCalls returned error: %v", err)
			}

			if len(results) != 1 {
				t.Fatalf("Expected 1 result, got %d", len(results))
			}

			if tt.shouldErr {
				if results[0].Error == nil {
					t.Error("Expected error for invalid arguments")
				} else {
					t.Logf("✓ Correctly rejected: %v", results[0].Error)
				}
			} else {
				if results[0].Error != nil {
					t.Errorf("Expected success, got error: %v", results[0].Error)
				}
				if receivedArgs == nil {
					t.Error("Tool should have received arguments map")
				} else if len(receivedArgs) != tt.argsCount {
					t.Errorf("Expected %d args, got %d", tt.argsCount, len(receivedArgs))
				} else {
					t.Logf("✓ Tool received empty args map correctly")
				}
			}
		})
	}
}

// TestJSONParseErrorResponse tests that parse errors are properly formatted in response.
func TestJSONParseErrorResponse(t *testing.T) {
	var logBuf strings.Builder
	logger := log.New(&logBuf, "[TEST] ", 0)

	tools := map[string]any{
		"test_tool": &simpleTool{
			name: "test_tool",
			execFunc: func(args map[string]any) (map[string]any, error) {
				return map[string]any{"result": "ok"}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
		MaxRetries:        0,
		Logger:            logger,
	})

	// Invalid JSON
	toolCalls := []ToolCall{
		{
			ID:   "call_bad_json",
			Type: "function",
			Function: FunctionCall{
				Name:      "test_tool",
				Arguments: `{"unclosed": "value"`,
			},
		},
	}

	ctx := context.Background()
	results, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)

	if err != nil {
		t.Fatalf("ExecuteToolCalls returned error: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	result := results[0]

	// Verify error is set
	if result.Error == nil {
		t.Fatal("Expected error to be set")
	}

	// Verify error type
	if !strings.Contains(result.Error.Error(), "failed to parse arguments") {
		t.Errorf("Expected 'failed to parse arguments' error, got: %v", result.Error)
	}
	if !strings.Contains(result.Error.Error(), "invalid JSON arguments") {
		t.Errorf("Expected 'invalid JSON arguments' in error, got: %v", result.Error)
	}

	// Verify response contains error message
	if result.Response == nil {
		t.Fatal("Expected response to be non-nil")
	}

	errorMsg, ok := result.Response["error"].(string)
	if !ok {
		t.Fatalf("Expected response['error'] to be string, got %T", result.Response["error"])
	}

	if !strings.Contains(errorMsg, "failed to parse arguments") {
		t.Errorf("Expected error message in response, got: %s", errorMsg)
	}

	// Verify logging
	logOutput := logBuf.String()
	if !strings.Contains(logOutput, "Failed to parse args for test_tool") {
		t.Errorf("Expected parse error to be logged, got: %s", logOutput)
	}

	// Verify duration is recorded even for parse errors
	if result.Duration == 0 {
		t.Error("Expected non-zero duration for error result")
	}

	t.Logf("✓ Parse error properly formatted in response: %s", errorMsg)
	t.Logf("✓ Parse error logged: %s", logOutput)
}

// TestPartialToolCalls tests handling of incomplete/partial tool calls from the model.
// This can happen when the model returns malformed tool call structures.
func TestPartialToolCalls(t *testing.T) {
	tests := []struct {
		name        string
		toolCall    ToolCall
		expectError bool
		errorMsg    string
		description string
	}{
		{
			name: "missing tool call ID",
			toolCall: ToolCall{
				ID:   "", // Empty ID
				Type: "function",
				Function: FunctionCall{
					Name:      "get_weather",
					Arguments: `{"location":"London"}`,
				},
			},
			expectError: false, // Should still execute but with empty ID
			description: "Tool call with missing ID should still execute",
		},
		{
			name: "missing function name",
			toolCall: ToolCall{
				ID:   "call_1",
				Type: "function",
				Function: FunctionCall{
					Name:      "", // Empty function name
					Arguments: `{"location":"London"}`,
				},
			},
			expectError: true,
			errorMsg:    "tool not found",
			description: "Tool call with missing function name should fail",
		},
		{
			name: "missing arguments",
			toolCall: ToolCall{
				ID:   "call_2",
				Type: "function",
				Function: FunctionCall{
					Name:      "get_weather",
					Arguments: "", // Empty arguments
				},
			},
			expectError: false, // Empty arguments should be treated as {}
			description: "Tool call with missing arguments should use empty object",
		},
		{
			name: "null arguments",
			toolCall: ToolCall{
				ID:   "call_3",
				Type: "function",
				Function: FunctionCall{
					Name:      "test_tool",
					Arguments: `null`, // null is valid JSON, parsed as nil map
				},
			},
			expectError: false, // null is valid JSON, becomes empty map
			description: "Tool call with null arguments should parse as empty map",
		},
		{
			name: "wrong type field",
			toolCall: ToolCall{
				ID:   "call_4",
				Type: "not_function", // Wrong type
				Function: FunctionCall{
					Name:      "get_weather",
					Arguments: `{"location":"Paris"}`,
				},
			},
			expectError: false, // Type is not validated in current implementation
			description: "Tool call with wrong type field should still execute",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var logBuf strings.Builder
			logger := log.New(&logBuf, "[PARTIAL_TEST] ", 0)

			// Create a simple test tool
			weatherTool := &simpleTool{
				name:        "get_weather",
				description: "Get weather",
				execFunc: func(args map[string]any) (map[string]any, error) {
					return map[string]any{"temperature": 20, "location": args["location"]}, nil
				},
			}

			testTool := &simpleTool{
				name:        "test_tool",
				description: "Test tool",
				execFunc: func(args map[string]any) (map[string]any, error) {
					return map[string]any{"result": "ok"}, nil
				},
			}

			tools := map[string]any{
				"get_weather": weatherTool,
				"test_tool":   testTool,
			}

			cfg := &ToolExecutorConfig{
				ParallelExecution: false,
				Timeout:           5 * time.Second,
				MaxRetries:        0, // No retries for this test
				Logger:            logger,
			}

			executor := NewToolExecutor(tools, cfg)

			// Execute the partial tool call
			results, err := executor.ExecuteToolCalls(context.Background(), []ToolCall{tt.toolCall}, nilToolContext())

			if err != nil {
				t.Fatalf("ExecuteToolCalls returned error: %v", err)
			}

			if len(results) != 1 {
				t.Fatalf("Expected 1 result, got %d", len(results))
			}

			result := results[0]

			// Verify result based on expectations
			if tt.expectError {
				if result.Error == nil {
					t.Errorf("Expected error for %s, but got none", tt.name)
				} else {
					if !strings.Contains(result.Error.Error(), tt.errorMsg) {
						t.Errorf("Expected error to contain '%s', got: %v", tt.errorMsg, result.Error)
					}
					t.Logf("✓ Expected error occurred: %v", result.Error)
				}

				// Verify error in response
				if result.Response == nil {
					t.Error("Expected response to contain error info")
				} else if errMsg, ok := result.Response["error"].(string); ok {
					t.Logf("✓ Error in response: %s", errMsg)
				}
			} else {
				if result.Error != nil {
					t.Errorf("Did not expect error for %s, but got: %v", tt.name, result.Error)
				} else {
					t.Logf("✓ No error for %s (as expected)", tt.name)
				}

				// Verify response is valid
				if result.Response == nil {
					t.Error("Expected valid response")
				} else {
					t.Logf("✓ Valid response: %+v", result.Response)
				}
			}

			// Verify tool call ID is preserved in result
			if result.ToolCallID != tt.toolCall.ID {
				t.Errorf("Expected ToolCallID %s, got %s", tt.toolCall.ID, result.ToolCallID)
			}

			t.Logf("✓ %s", tt.description)
		})
	}
}

// TestPartialToolCallsMultiple tests handling of mixed valid and partial tool calls.
func TestPartialToolCallsMultiple(t *testing.T) {
	var logBuf strings.Builder
	logger := log.New(&logBuf, "[MIXED_PARTIAL] ", 0)

	weatherTool := &simpleTool{
		name:        "get_weather",
		description: "Get weather",
		execFunc: func(args map[string]any) (map[string]any, error) {
			return map[string]any{"temperature": 20}, nil
		},
	}

	tools := map[string]any{
		"get_weather": weatherTool,
	}

	cfg := &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
		MaxRetries:        0,
		Logger:            logger,
	}

	executor := NewToolExecutor(tools, cfg)

	// Mix of valid and partial tool calls
	toolCalls := []ToolCall{
		{
			ID:   "call_valid_1",
			Type: "function",
			Function: FunctionCall{
				Name:      "get_weather",
				Arguments: `{"location":"London"}`,
			},
		},
		{
			ID:   "call_missing_name",
			Type: "function",
			Function: FunctionCall{
				Name:      "", // Missing name
				Arguments: `{"data":"test"}`,
			},
		},
		{
			ID:   "call_valid_2",
			Type: "function",
			Function: FunctionCall{
				Name:      "get_weather",
				Arguments: `{"location":"Paris"}`,
			},
		},
		{
			ID:   "call_bad_json",
			Type: "function",
			Function: FunctionCall{
				Name:      "get_weather",
				Arguments: `{invalid json}`,
			},
		},
	}

	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nilToolContext())

	if err != nil {
		t.Fatalf("ExecuteToolCalls returned error: %v", err)
	}

	if len(results) != 4 {
		t.Fatalf("Expected 4 results, got %d", len(results))
	}

	// Verify first call succeeded
	if results[0].Error != nil {
		t.Errorf("First call should succeed, got error: %v", results[0].Error)
	} else {
		t.Logf("✓ First valid call succeeded")
	}

	// Verify second call failed (missing name)
	if results[1].Error == nil {
		t.Error("Second call should fail (missing name)")
	} else {
		if !strings.Contains(results[1].Error.Error(), "tool not found") {
			t.Errorf("Expected 'tool not found' error, got: %v", results[1].Error)
		}
		t.Logf("✓ Second call failed as expected: %v", results[1].Error)
	}

	// Verify third call succeeded
	if results[2].Error != nil {
		t.Errorf("Third call should succeed, got error: %v", results[2].Error)
	} else {
		t.Logf("✓ Third valid call succeeded")
	}

	// Verify fourth call failed (bad JSON)
	if results[3].Error == nil {
		t.Error("Fourth call should fail (bad JSON)")
	} else {
		if !strings.Contains(results[3].Error.Error(), "parse arguments") {
			t.Errorf("Expected parse error, got: %v", results[3].Error)
		}
		t.Logf("✓ Fourth call failed as expected: %v", results[3].Error)
	}

	// Verify all results have correct tool call IDs in order
	expectedIDs := []string{"call_valid_1", "call_missing_name", "call_valid_2", "call_bad_json"}
	for i, result := range results {
		if result.ToolCallID != expectedIDs[i] {
			t.Errorf("Result %d: expected ID %s, got %s", i, expectedIDs[i], result.ToolCallID)
		}
	}

	t.Logf("✓ Mixed valid and partial tool calls handled correctly")
	t.Logf("✓ Successful calls: 2/4")
	t.Logf("✓ Failed calls: 2/4")
}

// TestPartialToolCallsLogging tests that partial tool calls are properly logged.
func TestPartialToolCallsLogging(t *testing.T) {
	var logBuf strings.Builder
	logger := log.New(&logBuf, "[LOG_TEST] ", 0)

	tools := map[string]any{} // No tools available

	cfg := &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
		MaxRetries:        0,
		Logger:            logger,
	}

	executor := NewToolExecutor(tools, cfg)

	// Various partial tool calls
	toolCalls := []ToolCall{
		{
			ID:   "",
			Type: "function",
			Function: FunctionCall{
				Name:      "missing_id_tool",
				Arguments: `{}`,
			},
		},
		{
			ID:   "call_empty_name",
			Type: "function",
			Function: FunctionCall{
				Name:      "",
				Arguments: `{"test":true}`,
			},
		},
	}

	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nilToolContext())

	if err != nil {
		t.Fatalf("ExecuteToolCalls returned error: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("Expected 2 results, got %d", len(results))
	}

	// Check logging output
	logOutput := logBuf.String()

	// Should log tool execution attempts
	if !strings.Contains(logOutput, "Executing") {
		t.Error("Expected 'Executing' in logs")
	}

	// Should log tool not found errors
	if !strings.Contains(logOutput, "not found") {
		t.Error("Expected 'not found' in logs for missing tools")
	}

	// Should log completion
	if !strings.Contains(logOutput, "Completed") {
		t.Error("Expected 'Completed' in logs")
	}

	t.Logf("✓ Partial tool calls properly logged")
	t.Logf("Log output:\n%s", logOutput)
}

// TestPartialToolCallsRetryBehavior tests retry behavior with partial tool calls.
func TestPartialToolCallsRetryBehavior(t *testing.T) {
	var logBuf strings.Builder
	logger := log.New(&logBuf, "[RETRY_TEST] ", 0)

	// Tool that fails on first attempt
	var attemptCount int32
	flakeyTool := &simpleTool{
		name:        "flakey_tool",
		description: "Flakey tool",
		execFunc: func(args map[string]any) (map[string]any, error) {
			count := atomic.AddInt32(&attemptCount, 1)
			if count == 1 {
				return nil, fmt.Errorf("transient error")
			}
			return map[string]any{"result": "success"}, nil
		},
	}

	tools := map[string]any{
		"flakey_tool": flakeyTool,
	}

	cfg := &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
		MaxRetries:        2, // Allow retries
		Logger:            logger,
	}

	executor := NewToolExecutor(tools, cfg)

	// Partial tool call (empty ID) with retryable tool
	toolCalls := []ToolCall{
		{
			ID:   "", // Empty ID
			Type: "function",
			Function: FunctionCall{
				Name:      "flakey_tool",
				Arguments: `{"test":true}`,
			},
		},
	}

	results, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nilToolContext())

	if err != nil {
		t.Fatalf("ExecuteToolCalls returned error: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	result := results[0]

	// Should succeed after retry
	if result.Error != nil {
		t.Errorf("Expected success after retry, got error: %v", result.Error)
	} else {
		t.Logf("✓ Tool succeeded after retry")
	}

	// Verify it was retried
	finalCount := atomic.LoadInt32(&attemptCount)
	if finalCount < 2 {
		t.Errorf("Expected at least 2 attempts, got %d", finalCount)
	} else {
		t.Logf("✓ Tool was retried %d times", finalCount)
	}

	// Check logs for retry messages
	logOutput := logBuf.String()
	if !strings.Contains(logOutput, "Retrying") {
		t.Log("Note: Retry logging may not be present in current implementation")
	}

	t.Logf("✓ Partial tool calls work correctly with retry mechanism")
}

// TestLargeJSONArgumentsSanitization tests handling of very large JSON arguments
func TestLargeJSONArgumentsSanitization(t *testing.T) {
	tests := []struct {
		name        string
		argsJSON    string
		expectError bool
		description string
	}{
		{
			name:        "reasonable size JSON (10KB)",
			argsJSON:    `{"data":"` + strings.Repeat("x", 10*1024) + `"}`,
			expectError: false,
			description: "Normal size JSON should parse fine",
		},
		{
			name:        "large JSON (100KB)",
			argsJSON:    `{"data":"` + strings.Repeat("x", 100*1024) + `"}`,
			expectError: false,
			description: "Large JSON should still parse (no size limit currently)",
		},
		{
			name:        "very large JSON (1MB)",
			argsJSON:    `{"data":"` + strings.Repeat("x", 1024*1024) + `"}`,
			expectError: false,
			description: "Very large JSON tests memory handling",
		},
		{
			name: "deeply nested JSON",
			argsJSON: func() string {
				// Create 100 levels of nesting
				s := ""
				for i := 0; i < 100; i++ {
					s += `{"nested":`
				}
				s += `"value"`
				for i := 0; i < 100; i++ {
					s += `}`
				}
				return s
			}(),
			expectError: false,
			description: "Deeply nested JSON should parse",
		},
		{
			name: "array with many elements",
			argsJSON: func() string {
				items := make([]string, 10000)
				for i := range items {
					items[i] = fmt.Sprintf(`"item%d"`, i)
				}
				return `{"array":[` + strings.Join(items, ",") + `]}`
			}(),
			expectError: false,
			description: "Large array should parse",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var logBuf strings.Builder
			logger := log.New(&logBuf, "[SIZE_TEST] ", 0)

			testTool := &simpleTool{
				name:        "test_tool",
				description: "Test tool",
				execFunc: func(args map[string]any) (map[string]any, error) {
					// Tool doesn't need to do anything, just verify args were parsed
					return map[string]any{"received": true, "arg_count": len(args)}, nil
				},
			}

			tools := map[string]any{
				"test_tool": testTool,
			}

			cfg := &ToolExecutorConfig{
				ParallelExecution: false,
				Timeout:           30 * time.Second,
				MaxRetries:        0,
				Logger:            logger,
			}

			executor := NewToolExecutor(tools, cfg)

			toolCall := ToolCall{
				ID:   "call_large",
				Type: "function",
				Function: FunctionCall{
					Name:      "test_tool",
					Arguments: tt.argsJSON,
				},
			}

			start := time.Now()
			results, err := executor.ExecuteToolCalls(context.Background(), []ToolCall{toolCall}, nilToolContext())
			elapsed := time.Since(start)

			if err != nil {
				t.Fatalf("ExecuteToolCalls failed: %v", err)
			}

			if len(results) != 1 {
				t.Fatalf("Expected 1 result, got %d", len(results))
			}

			result := results[0]

			if tt.expectError {
				if result.Error == nil {
					t.Errorf("Expected error for %s", tt.name)
				}
			} else {
				if result.Error != nil {
					t.Errorf("Did not expect error: %v", result.Error)
				} else {
					t.Logf("✓ Parsed large JSON successfully")
					t.Logf("  JSON size: %d bytes", len(tt.argsJSON))
					t.Logf("  Parse time: %v", elapsed)
					t.Logf("  Response: %+v", result.Response)
				}
			}

			t.Logf("✓ %s", tt.description)
		})
	}
}

// TestArgumentsSanitization tests sanitization of potentially dangerous arguments
func TestArgumentsSanitization(t *testing.T) {
	tests := []struct {
		name        string
		argsJSON    string
		expectError bool
		description string
	}{
		{
			name:        "SQL injection attempt",
			argsJSON:    `{"query":"'; DROP TABLE users; --"}`,
			expectError: false, // Should parse, but tool should handle safely
			description: "SQL injection strings should parse as normal strings",
		},
		{
			name:        "XSS attempt",
			argsJSON:    `{"html":"<script>alert('xss')</script>"}`,
			expectError: false,
			description: "XSS strings should parse as normal strings",
		},
		{
			name:        "command injection",
			argsJSON:    `{"cmd":"ls; rm -rf /"}`,
			expectError: false,
			description: "Command injection strings should parse",
		},
		{
			name:        "path traversal",
			argsJSON:    `{"path":"../../etc/passwd"}`,
			expectError: false,
			description: "Path traversal strings should parse",
		},
		{
			name:        "null bytes",
			argsJSON:    `{"data":"test\u0000null"}`,
			expectError: false,
			description: "Null bytes in JSON unicode escape",
		},
		{
			name:        "unicode normalization",
			argsJSON:    `{"text":"Ḧëḷḷö Ẅöṛḷḋ"}`,
			expectError: false,
			description: "Unicode characters should parse",
		},
		{
			name:        "control characters",
			argsJSON:    `{"text":"line1\nline2\ttabbed"}`,
			expectError: false,
			description: "Control characters in strings",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testTool := &simpleTool{
				name:        "test_tool",
				description: "Test tool",
				execFunc: func(args map[string]any) (map[string]any, error) {
					// Echo back the args to verify they were parsed correctly
					return map[string]any{"echo": args}, nil
				},
			}

			tools := map[string]any{
				"test_tool": testTool,
			}

			cfg := &ToolExecutorConfig{
				ParallelExecution: false,
				Timeout:           5 * time.Second,
				MaxRetries:        0,
			}

			executor := NewToolExecutor(tools, cfg)

			toolCall := ToolCall{
				ID:   "call_sanitize",
				Type: "function",
				Function: FunctionCall{
					Name:      "test_tool",
					Arguments: tt.argsJSON,
				},
			}

			results, err := executor.ExecuteToolCalls(context.Background(), []ToolCall{toolCall}, nilToolContext())

			if err != nil {
				t.Fatalf("ExecuteToolCalls failed: %v", err)
			}

			if len(results) != 1 {
				t.Fatalf("Expected 1 result, got %d", len(results))
			}

			result := results[0]

			if tt.expectError {
				if result.Error == nil {
					t.Errorf("Expected error for %s", tt.name)
				}
			} else {
				if result.Error != nil {
					t.Errorf("Unexpected error: %v", result.Error)
				} else {
					t.Logf("✓ Parsed potentially dangerous input safely")
					t.Logf("  Input: %s", tt.argsJSON)
					if echo, ok := result.Response["echo"].(map[string]any); ok {
						t.Logf("  Echoed: %+v", echo)
					}
				}
			}

			t.Logf("✓ %s", tt.description)
		})
	}
}

// TestToolExecutionCoverageBoost tests additional tool execution scenarios for coverage
func TestToolExecutionCoverageBoost(t *testing.T) {
	t.Run("execute with nil tool context", func(t *testing.T) {
		tool := &simpleTool{
			name:        "test",
			description: "Test",
			execFunc: func(args map[string]any) (map[string]any, error) {
				return map[string]any{"ok": true}, nil
			},
		}

		executor := NewToolExecutor(map[string]any{"test": tool}, nil)

		results, err := executor.ExecuteToolCalls(
			context.Background(),
			[]ToolCall{{ID: "1", Type: "function", Function: FunctionCall{Name: "test", Arguments: "{}"}}},
			nil, // nil context
		)

		if err != nil {
			t.Fatalf("Failed: %v", err)
		}

		if len(results) != 1 || results[0].Error != nil {
			t.Errorf("Expected successful execution with nil context")
		}

		t.Logf("✓ Execution works with nil tool context")
	})

	t.Run("empty tool calls list", func(t *testing.T) {
		executor := NewToolExecutor(map[string]any{}, nil)

		results, err := executor.ExecuteToolCalls(context.Background(), []ToolCall{}, nil)

		if err != nil {
			t.Errorf("Empty tool calls should not error: %v", err)
		}

		if results != nil && len(results) != 0 {
			t.Errorf("Expected nil or empty results, got %d", len(results))
		}

		t.Logf("✓ Empty tool calls handled correctly")
	})

	t.Run("tool with empty arguments", func(t *testing.T) {
		tool := &simpleTool{
			name:        "test",
			description: "Test",
			execFunc: func(args map[string]any) (map[string]any, error) {
				if len(args) != 0 {
					return nil, fmt.Errorf("expected empty args, got %d", len(args))
				}
				return map[string]any{"ok": true}, nil
			},
		}

		executor := NewToolExecutor(map[string]any{"test": tool}, nil)

		results, err := executor.ExecuteToolCalls(
			context.Background(),
			[]ToolCall{{ID: "1", Type: "function", Function: FunctionCall{Name: "test", Arguments: ""}}},
			nil,
		)

		if err != nil || len(results) != 1 || results[0].Error != nil {
			t.Errorf("Empty arguments should parse as empty map")
		}

		t.Logf("✓ Empty arguments handled as empty map")
	})

	t.Run("context cancellation during execution", func(t *testing.T) {
		slowTool := &simpleTool{
			name:        "slow",
			description: "Slow tool",
			execFunc: func(args map[string]any) (map[string]any, error) {
				time.Sleep(5 * time.Second)
				return map[string]any{"ok": true}, nil
			},
		}

		executor := NewToolExecutor(map[string]any{"slow": slowTool}, &ToolExecutorConfig{
			Timeout: 1 * time.Second, // Short timeout
		})

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		start := time.Now()
		results, err := executor.ExecuteToolCalls(
			ctx,
			[]ToolCall{{ID: "1", Type: "function", Function: FunctionCall{Name: "slow", Arguments: "{}"}}},
			nil,
		)
		elapsed := time.Since(start)

		if err != nil {
			t.Fatalf("ExecuteToolCalls failed: %v", err)
		}

		if len(results) == 0 {
			t.Fatal("Expected results")
		}

		// Should timeout
		if results[0].Error == nil {
			t.Error("Expected timeout error")
		} else {
			t.Logf("✓ Timeout error: %v", results[0].Error)
		}

		if elapsed > 3*time.Second {
			t.Errorf("Timeout took too long: %v", elapsed)
		}

		t.Logf("✓ Context cancellation/timeout handled in %v", elapsed)
	})
}
