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

package remoteagent

import (
	"context"
	"fmt"
	"iter"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"
)

func BenchmarkRemoteAgentStream(b *testing.B) {
	card := &a2a.AgentCard{
		Name: "bench_remote",
		SupportedInterfaces: []*a2a.AgentInterface{
			a2a.NewAgentInterface("http://127.0.0.1:0", a2a.TransportProtocolJSONRPC),
		},
		Capabilities: a2a.AgentCapabilities{Streaming: true},
	}

	directConverter := func(ctx agent.InvocationContext, _ *a2a.SendMessageRequest, ev a2a.Event, err error) (*session.Event, error) {
		if err != nil {
			return nil, err
		}
		return adka2a.ToSessionEvent(ctx, ev)
	}

	for _, variant := range []struct {
		name      string
		converter A2AEventConverter
	}{
		{name: "DefaultConverter", converter: nil},
		{name: "DirectToSessionEvent", converter: directConverter},
	} {
		b.Run(variant.name, func(b *testing.B) {
			for _, chunks := range []int{200, 2000} {
				b.Run(fmt.Sprintf("chunks=%d", chunks), func(b *testing.B) {
					streamEvents := makeBenchmarkA2AStream(chunks)
					client := &benchmarkStreamClient{events: streamEvents}
					agnt, err := NewA2A(A2AConfig{
						Name:      "bench_remote",
						AgentCard: card,
						Converter: variant.converter,
						ClientProvider: func(context.Context, *a2a.AgentCard) (A2AClient, error) {
							return client, nil
						},
					})
					if err != nil {
						b.Fatalf("NewA2A failed: %v", err)
					}

					ctx := b.Context()
					svc := session.InMemoryService()
					created, err := svc.Create(ctx, &session.CreateRequest{AppName: "bench", UserID: "user"})
					if err != nil {
						b.Fatalf("svc.Create failed: %v", err)
					}
					if err := svc.AppendEvent(ctx, created.Session, newUserHello()); err != nil {
						b.Fatalf("svc.AppendEvent failed: %v", err)
					}
					ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
						Session:   created.Session,
						Agent:     agnt,
						RunConfig: &agent.RunConfig{StreamingMode: agent.StreamingModeSSE},
					})

					b.ReportAllocs()
					b.ResetTimer()
					for range b.N {
						count := 0
						for ev, err := range agnt.Run(ic) {
							if err != nil {
								b.Fatalf("agnt.Run failed: %v", err)
							}
							if ev != nil {
								count++
							}
						}
						if count != chunks+1 {
							b.Fatalf("emitted %d events, want %d", count, chunks+1)
						}
					}
				})
			}
		})
	}
}

type benchmarkStreamClient struct {
	A2AClient
	events []a2a.Event
}

func (c *benchmarkStreamClient) SendStreamingMessage(_ context.Context, _ *a2a.SendMessageRequest) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		for _, ev := range c.events {
			if !yield(ev, nil) {
				return
			}
		}
	}
}

func (c *benchmarkStreamClient) Destroy() error { return nil }

func makeBenchmarkA2AStream(chunks int) []a2a.Event {
	task := &a2a.Task{ID: a2a.TaskID("task-bench-1"), ContextID: "ctx-bench-1"}
	partialArtifactID := a2a.ArtifactID("art-partial-1")
	finalArtifactID := a2a.ArtifactID("art-final-1")

	events := make([]a2a.Event, 0, chunks+1)
	for range chunks - 1 {
		part := a2a.NewTextPart("streamed chunk delta ")
		part.SetMeta(adka2a.ToA2AMetaKey("partial"), true)
		ev := a2a.NewArtifactUpdateEvent(task, partialArtifactID, part)
		ev.Metadata = map[string]any{adka2a.ToA2AMetaKey("partial"): true}
		events = append(events, ev)
	}

	finalArt := a2a.NewArtifactUpdateEvent(task, finalArtifactID, a2a.NewTextPart("final answer text"))
	finalArt.Metadata = map[string]any{adka2a.ToA2AMetaKey("partial"): false}
	events = append(events, finalArt)

	status := a2a.NewStatusUpdateEvent(task, a2a.TaskStateCompleted, nil)
	events = append(events, status)
	return events
}
