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
	"testing"
	"time"
)

// Benchmark single tool execution
func BenchmarkSingleTool(b *testing.B) {
	tools := map[string]any{
		"calculator": &simpleTool{
			name: "calculator",
			execFunc: func(args map[string]any) (map[string]any, error) {
				a := int(args["a"].(float64))
				b := int(args["b"].(float64))
				return map[string]any{"result": a + b}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           5 * time.Second,
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

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark parallel execution with varying number of tools
func BenchmarkParallelExecution_3Tools(b *testing.B) {
	benchmarkParallelTools(b, 3)
}

func BenchmarkParallelExecution_5Tools(b *testing.B) {
	benchmarkParallelTools(b, 5)
}

func BenchmarkParallelExecution_10Tools(b *testing.B) {
	benchmarkParallelTools(b, 10)
}

func BenchmarkParallelExecution_20Tools(b *testing.B) {
	benchmarkParallelTools(b, 20)
}

func benchmarkParallelTools(b *testing.B, numTools int) {
	tools := make(map[string]any)
	toolCalls := make([]ToolCall, numTools)

	for i := 0; i < numTools; i++ {
		toolName := fmt.Sprintf("tool_%d", i)
		tools[toolName] = &simpleTool{
			name: toolName,
			execFunc: func(args map[string]any) (map[string]any, error) {
				time.Sleep(5 * time.Millisecond) // Simulate work
				return map[string]any{"result": "done"}, nil
			},
		}
		toolCalls[i] = ToolCall{
			ID:   fmt.Sprintf("call_%d", i),
			Type: "function",
			Function: FunctionCall{
				Name:      toolName,
				Arguments: "{}",
			},
		}
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: true,
		Timeout:           10 * time.Second,
	})

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark sequential execution
func BenchmarkSequentialExecution_3Tools(b *testing.B) {
	benchmarkSequentialTools(b, 3)
}

func BenchmarkSequentialExecution_5Tools(b *testing.B) {
	benchmarkSequentialTools(b, 5)
}

func BenchmarkSequentialExecution_10Tools(b *testing.B) {
	benchmarkSequentialTools(b, 10)
}

func benchmarkSequentialTools(b *testing.B, numTools int) {
	tools := make(map[string]any)
	toolCalls := make([]ToolCall, numTools)

	for i := 0; i < numTools; i++ {
		toolName := fmt.Sprintf("tool_%d", i)
		tools[toolName] = &simpleTool{
			name: toolName,
			execFunc: func(args map[string]any) (map[string]any, error) {
				time.Sleep(5 * time.Millisecond)
				return map[string]any{"result": "done"}, nil
			},
		}
		toolCalls[i] = ToolCall{
			ID:   fmt.Sprintf("call_%d", i),
			Type: "function",
			Function: FunctionCall{
				Name:      toolName,
				Arguments: "{}",
			},
		}
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false, // Sequential
		Timeout:           10 * time.Second,
	})

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark tool chain with dependencies
func BenchmarkToolChain_Sequential(b *testing.B) {
	// Simulate a dependency chain: tool_a -> tool_b -> tool_c
	resultStore := make(map[string]any)

	tools := map[string]any{
		"tool_a": &simpleTool{
			name: "tool_a",
			execFunc: func(args map[string]any) (map[string]any, error) {
				time.Sleep(3 * time.Millisecond)
				result := 10
				resultStore["a"] = result
				return map[string]any{"value": result}, nil
			},
		},
		"tool_b": &simpleTool{
			name: "tool_b",
			execFunc: func(args map[string]any) (map[string]any, error) {
				time.Sleep(3 * time.Millisecond)
				prev := resultStore["a"].(int)
				result := prev * 2
				resultStore["b"] = result
				return map[string]any{"value": result}, nil
			},
		},
		"tool_c": &simpleTool{
			name: "tool_c",
			execFunc: func(args map[string]any) (map[string]any, error) {
				time.Sleep(3 * time.Millisecond)
				prev := resultStore["b"].(int)
				result := prev + 5
				return map[string]any{"value": result}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, &ToolExecutorConfig{
		ParallelExecution: false,
		Timeout:           10 * time.Second,
	})

	toolCalls := []ToolCall{
		{ID: "call_a", Type: "function", Function: FunctionCall{Name: "tool_a", Arguments: "{}"}},
		{ID: "call_b", Type: "function", Function: FunctionCall{Name: "tool_b", Arguments: "{}"}},
		{ID: "call_c", Type: "function", Function: FunctionCall{Name: "tool_c", Arguments: "{}"}},
	}

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resultStore = make(map[string]any) // Reset for each iteration
		_, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark retry overhead
func BenchmarkRetryOverhead(b *testing.B) {
	attemptCount := 0

	tools := map[string]any{
		"flaky_tool": &simpleTool{
			name: "flaky_tool",
			execFunc: func(args map[string]any) (map[string]any, error) {
				attemptCount++
				if attemptCount%3 != 0 {
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

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		attemptCount = 0
		_, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark JSON parsing overhead
func BenchmarkJSONParsing(b *testing.B) {
	tools := map[string]any{
		"json_tool": &simpleTool{
			name: "json_tool",
			execFunc: func(args map[string]any) (map[string]any, error) {
				return map[string]any{"result": "ok"}, nil
			},
		},
	}

	executor := NewToolExecutor(tools, nil)

	// Complex JSON arguments
	complexJSON := `{
		"location": "New York",
		"date": "2025-01-16",
		"options": {
			"units": "metric",
			"details": ["temperature", "humidity", "wind"],
			"forecast_days": 7
		}
	}`

	toolCalls := []ToolCall{
		{
			ID:   "call_1",
			Type: "function",
			Function: FunctionCall{
				Name:      "json_tool",
				Arguments: complexJSON,
			},
		},
	}

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark comparison: Parallel vs Sequential
func BenchmarkComparison_ParallelVsSequential(b *testing.B) {
	numTools := 5

	tools := make(map[string]any)
	toolCalls := make([]ToolCall, numTools)

	for i := 0; i < numTools; i++ {
		toolName := fmt.Sprintf("tool_%d", i)
		tools[toolName] = &simpleTool{
			name: toolName,
			execFunc: func(args map[string]any) (map[string]any, error) {
				time.Sleep(5 * time.Millisecond)
				return map[string]any{"result": "done"}, nil
			},
		}
		toolCalls[i] = ToolCall{
			ID:   fmt.Sprintf("call_%d", i),
			Type: "function",
			Function: FunctionCall{
				Name:      toolName,
				Arguments: "{}",
			},
		}
	}

	b.Run("Parallel", func(b *testing.B) {
		executor := NewToolExecutor(tools, &ToolExecutorConfig{
			ParallelExecution: true,
			Timeout:           10 * time.Second,
		})
		ctx := context.Background()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			executor.ExecuteToolCalls(ctx, toolCalls, nil)
		}
	})

	b.Run("Sequential", func(b *testing.B) {
		executor := NewToolExecutor(tools, &ToolExecutorConfig{
			ParallelExecution: false,
			Timeout:           10 * time.Second,
		})
		ctx := context.Background()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			executor.ExecuteToolCalls(ctx, toolCalls, nil)
		}
	})
}

// Benchmark parallel vs sequential with varying tool counts
func BenchmarkParallelVsSequential(b *testing.B) {
	toolCounts := []int{1, 5, 10}
	workDurations := []time.Duration{
		5 * time.Millisecond,  // Light work
		10 * time.Millisecond, // Medium work
		20 * time.Millisecond, // Heavy work
	}

	for _, count := range toolCounts {
		for _, workDuration := range workDurations {
			name := fmt.Sprintf("Tools=%d/Work=%dms", count, workDuration.Milliseconds())

			b.Run(name, func(b *testing.B) {
				// Setup tools
				tools := make(map[string]any)
				toolCalls := make([]ToolCall, count)

				for i := 0; i < count; i++ {
					toolName := fmt.Sprintf("tool_%d", i)
					duration := workDuration // Capture for closure

					tools[toolName] = &simpleTool{
						name: toolName,
						execFunc: func(args map[string]any) (map[string]any, error) {
							time.Sleep(duration)
							return map[string]any{"result": "done"}, nil
						},
					}
					toolCalls[i] = ToolCall{
						ID:   fmt.Sprintf("call_%d", i),
						Type: "function",
						Function: FunctionCall{
							Name:      toolName,
							Arguments: "{}",
						},
					}
				}

				ctx := context.Background()

				// Parallel execution
				b.Run("Parallel", func(b *testing.B) {
					executor := NewToolExecutor(tools, &ToolExecutorConfig{
						ParallelExecution: true,
						Timeout:           30 * time.Second,
					})

					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						_, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)
						if err != nil {
							b.Fatal(err)
						}
					}
				})

				// Sequential execution
				b.Run("Sequential", func(b *testing.B) {
					executor := NewToolExecutor(tools, &ToolExecutorConfig{
						ParallelExecution: false,
						Timeout:           30 * time.Second,
					})

					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						_, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)
						if err != nil {
							b.Fatal(err)
						}
					}
				})
			})
		}
	}
}

// Benchmark CPU-bound vs I/O-bound tools
func BenchmarkWorkloadTypes(b *testing.B) {
	toolCount := 5

	b.Run("CPUBound", func(b *testing.B) {
		// CPU-intensive work (computation)
		tools := make(map[string]any)
		toolCalls := make([]ToolCall, toolCount)

		for i := 0; i < toolCount; i++ {
			toolName := fmt.Sprintf("cpu_tool_%d", i)
			tools[toolName] = &simpleTool{
				name: toolName,
				execFunc: func(args map[string]any) (map[string]any, error) {
					// Simulate CPU work
					sum := 0
					for j := 0; j < 100000; j++ {
						sum += j
					}
					return map[string]any{"result": sum}, nil
				},
			}
			toolCalls[i] = ToolCall{
				ID:   fmt.Sprintf("call_%d", i),
				Type: "function",
				Function: FunctionCall{
					Name:      toolName,
					Arguments: "{}",
				},
			}
		}

		ctx := context.Background()

		b.Run("Parallel", func(b *testing.B) {
			executor := NewToolExecutor(tools, &ToolExecutorConfig{
				ParallelExecution: true,
				Timeout:           10 * time.Second,
			})

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				executor.ExecuteToolCalls(ctx, toolCalls, nil)
			}
		})

		b.Run("Sequential", func(b *testing.B) {
			executor := NewToolExecutor(tools, &ToolExecutorConfig{
				ParallelExecution: false,
				Timeout:           10 * time.Second,
			})

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				executor.ExecuteToolCalls(ctx, toolCalls, nil)
			}
		})
	})

	b.Run("IOBound", func(b *testing.B) {
		// I/O-bound work (simulated with sleep)
		tools := make(map[string]any)
		toolCalls := make([]ToolCall, toolCount)

		for i := 0; i < toolCount; i++ {
			toolName := fmt.Sprintf("io_tool_%d", i)
			tools[toolName] = &simpleTool{
				name: toolName,
				execFunc: func(args map[string]any) (map[string]any, error) {
					// Simulate I/O wait
					time.Sleep(10 * time.Millisecond)
					return map[string]any{"result": "done"}, nil
				},
			}
			toolCalls[i] = ToolCall{
				ID:   fmt.Sprintf("call_%d", i),
				Type: "function",
				Function: FunctionCall{
					Name:      toolName,
					Arguments: "{}",
				},
			}
		}

		ctx := context.Background()

		b.Run("Parallel", func(b *testing.B) {
			executor := NewToolExecutor(tools, &ToolExecutorConfig{
				ParallelExecution: true,
				Timeout:           10 * time.Second,
			})

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				executor.ExecuteToolCalls(ctx, toolCalls, nil)
			}
		})

		b.Run("Sequential", func(b *testing.B) {
			executor := NewToolExecutor(tools, &ToolExecutorConfig{
				ParallelExecution: false,
				Timeout:           10 * time.Second,
			})

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				executor.ExecuteToolCalls(ctx, toolCalls, nil)
			}
		})
	})
}

// Benchmark scalability: measure overhead as tool count increases
func BenchmarkScalability(b *testing.B) {
	toolCounts := []int{1, 2, 5, 10, 20, 50}

	for _, count := range toolCounts {
		b.Run(fmt.Sprintf("Tools=%d", count), func(b *testing.B) {
			tools := make(map[string]any)
			toolCalls := make([]ToolCall, count)

			for i := 0; i < count; i++ {
				toolName := fmt.Sprintf("tool_%d", i)
				tools[toolName] = &simpleTool{
					name: toolName,
					execFunc: func(args map[string]any) (map[string]any, error) {
						// Very light work to measure overhead
						time.Sleep(1 * time.Millisecond)
						return map[string]any{"result": "ok"}, nil
					},
				}
				toolCalls[i] = ToolCall{
					ID:   fmt.Sprintf("call_%d", i),
					Type: "function",
					Function: FunctionCall{
						Name:      toolName,
						Arguments: "{}",
					},
				}
			}

			executor := NewToolExecutor(tools, &ToolExecutorConfig{
				ParallelExecution: true,
				Timeout:           30 * time.Second,
			})

			ctx := context.Background()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := executor.ExecuteToolCalls(ctx, toolCalls, nil)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
