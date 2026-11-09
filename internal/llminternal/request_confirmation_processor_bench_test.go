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

package llminternal_test

import (
	"context"
	"iter"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

type benchEvents struct {
	events []*session.Event
}

func (b *benchEvents) Len() int {
	return len(b.events)
}

func (b *benchEvents) At(i int) *session.Event {
	return b.events[i]
}

func (b *benchEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, e := range b.events {
			if !yield(e) {
				return
			}
		}
	}
}

type benchSession struct {
	evs *benchEvents
}

func (s *benchSession) ID() string                { return "bench-session" }
func (s *benchSession) UserID() string            { return "bench-user" }
func (s *benchSession) AppName() string           { return "bench-app" }
func (s *benchSession) LastUpdateTime() time.Time { return time.Time{} }
func (s *benchSession) State() session.State      { return nil }
func (s *benchSession) Events() session.Events    { return s.evs }

var (
	_ session.Session = (*benchSession)(nil)
	_ session.Events  = (*benchEvents)(nil)
)

func BenchmarkRequestConfirmationRequestProcessor_StandardTurn(b *testing.B) {
	// 20 session events in conversation history, ending with a standard user message (no confirmation responses)
	var events []*session.Event
	for i := 0; i < 20; i++ {
		author := "model"
		if i%2 == 0 {
			author = "user"
		}
		events = append(events, &session.Event{
			Author: author,
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: "Hello, this is message content"}},
					Role:  author,
				},
			},
		})
	}

	testModel := &testModel{}
	tools := []tool.Tool{
		&mockTool{name: "mock_tool_1"},
		&mockTool{name: "mock_tool_2"},
		&mockTool{name: "mock_tool_3"},
	}
	agnt, err := llmagent.New(llmagent.Config{
		Name:  "testAgent",
		Model: testModel,
		Tools: tools,
	})
	if err != nil {
		b.Fatalf("failed to create agent: %v", err)
	}

	sess := &benchSession{evs: &benchEvents{events: events}}
	ctx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{
		Agent:   agnt,
		Session: sess,
	})
	flow := &llminternal.Flow{Tools: tools}
	req := &model.LLMRequest{}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, err := range llminternal.RequestConfirmationRequestProcessor(ctx, req, flow) {
			if err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkRequestConfirmationRequestProcessor_ActiveConfirmation(b *testing.B) {
	callID := "call-to-confirm"
	events := []*session.Event{
		{Author: "user", LLMResponse: model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "run tool"}}, Role: "user"}}},
		{Author: "testAgent", LLMResponse: model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{Name: toolconfirmation.FunctionCallName, ID: "conf-1", Args: map[string]any{
				"original_function_call": map[string]any{"name": mockToolName, "id": callID, "args": map[string]any{}},
			}}},
		}, Role: "model"}}},
		{Author: "user", LLMResponse: model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{Name: toolconfirmation.FunctionCallName, ID: "conf-1", Response: map[string]any{"confirmed": true}}},
		}, Role: "user"}}},
	}

	testModel := &testModel{}
	tools := []tool.Tool{
		&mockTool{name: mockToolName},
	}
	agnt, err := llmagent.New(llmagent.Config{
		Name:  "testAgent",
		Model: testModel,
		Tools: tools,
	})
	if err != nil {
		b.Fatalf("failed to create agent: %v", err)
	}

	sess := &benchSession{evs: &benchEvents{events: events}}
	ctx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{
		Agent:   agnt,
		Session: sess,
	})
	flow := &llminternal.Flow{Tools: tools}
	req := &model.LLMRequest{}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, err := range llminternal.RequestConfirmationRequestProcessor(ctx, req, flow) {
			if err != nil {
				b.Fatal(err)
			}
		}
	}
}
