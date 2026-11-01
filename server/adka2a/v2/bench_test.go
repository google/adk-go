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

package adka2a

import (
	"fmt"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func BenchmarkEventProcessorStream(b *testing.B) {
	ctx := b.Context()
	agnt, err := agent.New(agent.Config{Name: "bench_agent"})
	if err != nil {
		b.Fatalf("agent.New failed: %v", err)
	}
	reqCtx := &a2asrv.ExecutorContext{
		TaskID:    a2a.TaskID("task-bench-1"),
		ContextID: "ctx-bench-1",
	}
	invMeta := toInvocationMeta(ctx, RunnerConfig{
		AppName: "bench_app",
		Agent:   agnt,
	}, reqCtx)

	for _, mode := range []struct {
		name      string
		makeMaker func(*a2asrv.ExecutorContext) eventToArtifactTransform
	}{
		{name: "PerRun", makeMaker: func(rc *a2asrv.ExecutorContext) eventToArtifactTransform { return newLegacyArtifactMaker(rc) }},
		{name: "PerEvent", makeMaker: func(rc *a2asrv.ExecutorContext) eventToArtifactTransform { return newArtifactMaker(rc) }},
	} {
		b.Run(mode.name, func(b *testing.B) {
			for _, chunks := range []int{200, 2000} {
				b.Run(fmt.Sprintf("chunks=%d", chunks), func(b *testing.B) {
					events := makeBenchmarkStreamEvents(chunks)
					b.ReportAllocs()
					b.ResetTimer()
					for range b.N {
						proc := newEventProcessor(reqCtx, invMeta, nil, mode.makeMaker(reqCtx))
						for _, ev := range events {
							out, err := proc.process(ctx, ev)
							if err != nil {
								b.Fatalf("proc.process failed: %v", err)
							}
							if out == nil {
								b.Fatal("proc.process returned nil")
							}
						}
					}
				})
			}
		})
	}
}

func BenchmarkToEventMetaStreamChunk(b *testing.B) {
	ctx := b.Context()
	agnt, err := agent.New(agent.Config{Name: "bench_agent"})
	if err != nil {
		b.Fatalf("agent.New failed: %v", err)
	}
	reqCtx := &a2asrv.ExecutorContext{
		TaskID:    a2a.TaskID("task-bench-1"),
		ContextID: "ctx-bench-1",
	}
	invMeta := toInvocationMeta(ctx, RunnerConfig{
		AppName: "bench_app",
		Agent:   agnt,
	}, reqCtx)
	ev := &session.Event{
		InvocationID: "inv-bench-1",
		Author:       "bench_agent",
		Branch:       "root.bench_agent",
		LLMResponse: model.LLMResponse{
			Partial: true,
			Content: genai.NewContentFromText("streamed chunk delta", genai.RoleModel),
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		meta, err := toEventMeta(invMeta, ev)
		if err != nil {
			b.Fatalf("toEventMeta failed: %v", err)
		}
		if len(meta) == 0 {
			b.Fatal("empty meta")
		}
	}
}

func BenchmarkToSessionEventArtifactUpdate(b *testing.B) {
	agnt, err := agent.New(agent.Config{Name: "bench_agent"})
	if err != nil {
		b.Fatalf("agent.New failed: %v", err)
	}
	ictx := icontext.NewInvocationContext(b.Context(), icontext.InvocationContextParams{
		Branch: "root.bench_agent",
		Agent:  agnt,
	})
	task := &a2a.Task{ID: a2a.TaskID("task-bench-1"), ContextID: "ctx-bench-1"}
	update := a2a.NewArtifactEvent(task, a2a.NewTextPart("streamed chunk delta"))
	update.Metadata = map[string]any{
		metadataPartialKey:     true,
		ToA2AMetaKey("author"): "bench_agent",
		ToA2AMetaKey("branch"): "root.bench_agent",
	}

	for _, chunks := range []int{200, 2000} {
		b.Run(fmt.Sprintf("chunks=%d", chunks), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				for range chunks {
					ev, err := ToSessionEventWithParts(ictx, update, nil)
					if err != nil {
						b.Fatalf("ToSessionEventWithParts failed: %v", err)
					}
					if ev == nil {
						b.Fatal("ToSessionEventWithParts returned nil")
					}
				}
			}
		})
	}
}

func BenchmarkToA2APartsCodeExecution(b *testing.B) {
	parts := benchmark5KBCodeExecutionParts()

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		out, err := ToA2AParts(parts, nil)
		if err != nil {
			b.Fatalf("ToA2AParts failed: %v", err)
		}
		if len(out) != 2 {
			b.Fatalf("len(out) = %d, want 2", len(out))
		}
	}
}

func BenchmarkToGenAIPartsCodeExecution(b *testing.B) {
	a2aParts, err := ToA2AParts(benchmark5KBCodeExecutionParts(), nil)
	if err != nil {
		b.Fatalf("ToA2AParts setup failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		out, err := ToGenAIParts(a2aParts)
		if err != nil {
			b.Fatalf("ToGenAIParts failed: %v", err)
		}
		if len(out) != 2 {
			b.Fatalf("len(out) = %d, want 2", len(out))
		}
	}
}

func makeBenchmarkStreamEvents(chunks int) []*session.Event {
	events := make([]*session.Event, chunks)
	for i := range chunks {
		events[i] = &session.Event{
			InvocationID: "inv-bench-1",
			Author:       "bench_agent",
			Branch:       "root.bench_agent",
			LLMResponse: model.LLMResponse{
				Partial: i < chunks-1,
				Content: genai.NewContentFromText("streamed chunk delta ", genai.RoleModel),
			},
		}
	}
	return events
}

func benchmark5KBCodeExecutionParts() []*genai.Part {
	code := strings.Repeat("print('computing row result')\n", 32)
	stdout := strings.Repeat("row-01: value=42.125 status=OK\n", 160)
	return []*genai.Part{
		{
			ExecutableCode: &genai.ExecutableCode{
				ID:       "exec-bench-1",
				Language: genai.LanguagePython,
				Code:     code,
			},
		},
		{
			CodeExecutionResult: &genai.CodeExecutionResult{
				ID:      "exec-bench-1",
				Outcome: genai.OutcomeOK,
				Output:  stdout,
			},
		},
	}
}
