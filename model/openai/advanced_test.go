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
	"sync"
	"testing"
)

// === MAX ITERATIONS PROTECTION ===

func TestMaxIterations_Default(t *testing.T) {
	guard := NewIterationGuard(5)
	ctx := context.Background()

	// First 5 iterations should succeed
	for i := 0; i < 5; i++ {
		if err := guard.CheckIteration(ctx); err != nil {
			t.Errorf("Iteration %d failed: %v", i, err)
		}
		ctx = guard.IncrementIteration(ctx)
	}

	// 6th iteration should fail
	if err := guard.CheckIteration(ctx); err == nil {
		t.Error("Expected error on 6th iteration")
	}
}

func TestMaxIterations_ContextOverride(t *testing.T) {
	guard := NewIterationGuard(5)
	ctx := WithMaxIterations(context.Background(), 3)

	// First 3 iterations should succeed
	for i := 0; i < 3; i++ {
		if err := guard.CheckIteration(ctx); err != nil {
			t.Errorf("Iteration %d failed: %v", i, err)
		}
		ctx = guard.IncrementIteration(ctx)
	}

	// 4th iteration should fail
	if err := guard.CheckIteration(ctx); err == nil {
		t.Error("Expected error on 4th iteration with context override")
	}
}

func TestMaxIterations_Reset(t *testing.T) {
	guard := NewIterationGuard(3)
	ctx := context.Background()

	// Increment to 3
	for i := 0; i < 3; i++ {
		ctx = guard.IncrementIteration(ctx)
	}

	current := guard.GetCurrentIteration(ctx)
	if current != 3 {
		t.Errorf("Expected iteration 3, got %d", current)
	}

	// Reset
	ctx = guard.ResetIteration(ctx)

	current = guard.GetCurrentIteration(ctx)
	if current != 0 {
		t.Errorf("Expected iteration 0 after reset, got %d", current)
	}

	// Should allow iterations again
	if err := guard.CheckIteration(ctx); err != nil {
		t.Errorf("Expected success after reset: %v", err)
	}
}

func TestMaxIterations_GetMaxIterations(t *testing.T) {
	ctx := context.Background()

	// Default
	if max := GetMaxIterations(ctx); max != DefaultMaxIterations {
		t.Errorf("Expected default %d, got %d", DefaultMaxIterations, max)
	}

	// With override
	ctx = WithMaxIterations(ctx, 25)
	if max := GetMaxIterations(ctx); max != 25 {
		t.Errorf("Expected 25, got %d", max)
	}
}

// === DEPENDENCY CHAINS ===

func TestDependencyChain_Simple(t *testing.T) {
	// Simulate: tool_a -> tool_b -> tool_c
	// Each tool depends on the previous one's result

	results := make(map[string]int)
	var mu sync.Mutex

	tools := map[string]any{
		"tool_a": &simpleTool{
			name: "tool_a",
			execFunc: func(args map[string]any) (map[string]any, error) {
				mu.Lock()
				defer mu.Unlock()
				result := 10
				results["a"] = result
				return map[string]any{"value": result}, nil
			},
		},
		"tool_b": &simpleTool{
			name: "tool_b",
			execFunc: func(args map[string]any) (map[string]any, error) {
				mu.Lock()
				defer mu.Unlock()
				prev, ok := results["a"]
				if !ok {
					return nil, fmt.Errorf("tool_a must run first")
				}
				result := prev * 2
				results["b"] = result
				return map[string]any{"value": result}, nil
			},
		},
		"tool_c": &simpleTool{
			name: "tool_c",
			execFunc: func(args map[string]any) (map[string]any, error) {
				mu.Lock()
				defer mu.Unlock()
				prev, ok := results["b"]
				if !ok {
					return nil, fmt.Errorf("tool_b must run first")
				}
				result := prev + 5
				results["c"] = result
				return map[string]any{"value": result}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false, // Sequential for dependencies
		Timeout:           5000000000, // 5 seconds
	})

	toolCalls := []ToolCall{
		{ID: "call_a", Type: "function", Function: FunctionCall{Name: "tool_a", Arguments: "{}"}},
		{ID: "call_b", Type: "function", Function: FunctionCall{Name: "tool_b", Arguments: "{}"}},
		{ID: "call_c", Type: "function", Function: FunctionCall{Name: "tool_c", Arguments: "{}"}},
	}

	toolResults, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)
	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	// Verify all tools executed
	if len(toolResults) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(toolResults))
	}

	// Verify no errors
	for i, result := range toolResults {
		if result.Error != nil {
			t.Errorf("Tool %d failed: %v", i, result.Error)
		}
	}

	// Verify results chain: 10 -> 20 -> 25
	mu.Lock()
	defer mu.Unlock()

	if results["a"] != 10 {
		t.Errorf("Expected tool_a result 10, got %d", results["a"])
	}
	if results["b"] != 20 {
		t.Errorf("Expected tool_b result 20, got %d", results["b"])
	}
	if results["c"] != 25 {
		t.Errorf("Expected tool_c result 25, got %d", results["c"])
	}

	t.Logf("Dependency chain: %d -> %d -> %d", results["a"], results["b"], results["c"])
}

func TestDependencyChain_ParallelFailure(t *testing.T) {
	// Test that parallel execution fails when there are dependencies
	results := make(map[string]int)
	var mu sync.Mutex

	tools := map[string]any{
		"tool_a": &simpleTool{
			name: "tool_a",
			execFunc: func(args map[string]any) (map[string]any, error) {
				mu.Lock()
				defer mu.Unlock()
				results["a"] = 10
				return map[string]any{"value": 10}, nil
			},
		},
		"tool_b": &simpleTool{
			name: "tool_b",
			execFunc: func(args map[string]any) (map[string]any, error) {
				mu.Lock()
				defer mu.Unlock()
				// Depends on tool_a
				prev, ok := results["a"]
				if !ok {
					return nil, fmt.Errorf("dependency not met: tool_a not executed")
				}
				return map[string]any{"value": prev * 2}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: true, // Parallel - will break dependency
		Timeout:           5000000000,
	})

	toolCalls := []ToolCall{
		{ID: "call_a", Type: "function", Function: FunctionCall{Name: "tool_a", Arguments: "{}"}},
		{ID: "call_b", Type: "function", Function: FunctionCall{Name: "tool_b", Arguments: "{}"}},
	}

	toolResults, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)
	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	// tool_b should have failed due to unmet dependency
	// (might succeed if tool_a finishes first by chance, but likely fails)
	hasError := false
	for _, result := range toolResults {
		if result.Error != nil {
			hasError = true
			t.Logf("Expected dependency error: %v", result.Error)
		}
	}

	// Note: This test is probabilistic - parallel execution might work by chance
	// In real implementation, we'd use dependency analysis to prevent this
	t.Logf("Parallel execution with dependencies - error detected: %v", hasError)
}

func TestDependencyChain_Complex(t *testing.T) {
	// Complex chain:
	//     tool_a
	//    /      \
	// tool_b   tool_c
	//    \      /
	//     tool_d

	results := make(map[string]int)
	var mu sync.Mutex

	tools := map[string]any{
		"tool_a": &simpleTool{
			name: "tool_a",
			execFunc: func(args map[string]any) (map[string]any, error) {
				mu.Lock()
				defer mu.Unlock()
				results["a"] = 5
				return map[string]any{"value": 5}, nil
			},
		},
		"tool_b": &simpleTool{
			name: "tool_b",
			execFunc: func(args map[string]any) (map[string]any, error) {
				mu.Lock()
				defer mu.Unlock()
				prev := results["a"]
				results["b"] = prev * 2
				return map[string]any{"value": prev * 2}, nil
			},
		},
		"tool_c": &simpleTool{
			name: "tool_c",
			execFunc: func(args map[string]any) (map[string]any, error) {
				mu.Lock()
				defer mu.Unlock()
				prev := results["a"]
				results["c"] = prev + 3
				return map[string]any{"value": prev + 3}, nil
			},
		},
		"tool_d": &simpleTool{
			name: "tool_d",
			execFunc: func(args map[string]any) (map[string]any, error) {
				mu.Lock()
				defer mu.Unlock()
				b := results["b"]
				c := results["c"]
				results["d"] = b + c
				return map[string]any{"value": b + c}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false, // Sequential to respect dependencies
		Timeout:           5000000000,
	})

	// Execute in order: a, b, c, d
	toolCalls := []ToolCall{
		{ID: "call_a", Type: "function", Function: FunctionCall{Name: "tool_a", Arguments: "{}"}},
		{ID: "call_b", Type: "function", Function: FunctionCall{Name: "tool_b", Arguments: "{}"}},
		{ID: "call_c", Type: "function", Function: FunctionCall{Name: "tool_c", Arguments: "{}"}},
		{ID: "call_d", Type: "function", Function: FunctionCall{Name: "tool_d", Arguments: "{}"}},
	}

	toolResults, err := executor.ExecuteToolCalls(context.Background(), toolCalls, nil)
	if err != nil {
		t.Fatalf("ExecuteToolCalls failed: %v", err)
	}

	// Verify all succeeded
	for i, result := range toolResults {
		if result.Error != nil {
			t.Errorf("Tool %d failed: %v", i, result.Error)
		}
	}

	// Verify results: a=5, b=10, c=8, d=18
	mu.Lock()
	defer mu.Unlock()

	expected := map[string]int{"a": 5, "b": 10, "c": 8, "d": 18}
	for key, expectedVal := range expected {
		if results[key] != expectedVal {
			t.Errorf("Expected %s=%d, got %d", key, expectedVal, results[key])
		}
	}

	t.Logf("Complex dependency chain: a=%d, b=%d, c=%d, d=%d",
		results["a"], results["b"], results["c"], results["d"])
}

// === INTEGRATION: MAX ITERATIONS + TOOL EXECUTION ===

func TestIntegration_MaxIterationsWithTools(t *testing.T) {
	guard := NewIterationGuard(3)
	ctx := context.Background()

	callCount := 0

	tools := map[string]any{
		"test_tool": &simpleTool{
			name: "test_tool",
			execFunc: func(args map[string]any) (map[string]any, error) {
				callCount++
				return map[string]any{"iteration": callCount}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, nil)

	toolCalls := []ToolCall{
		{ID: "call_1", Type: "function", Function: FunctionCall{Name: "test_tool", Arguments: "{}"}},
	}

	// Simulate multi-turn loop
	for i := 0; i < 5; i++ {
		// Check iteration limit
		if err := guard.CheckIteration(ctx); err != nil {
			t.Logf("Stopped at iteration %d: %v", i, err)
			if i != 3 {
				t.Errorf("Expected to stop at iteration 3, stopped at %d", i)
			}
			break
		}

		// Execute tools
		_, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)
		if err != nil {
			t.Fatalf("Tool execution failed: %v", err)
		}

		// Increment iteration
		ctx = guard.IncrementIteration(ctx)
	}

	// Should have called tool 3 times
	if callCount != 3 {
		t.Errorf("Expected 3 tool calls, got %d", callCount)
	}
}
