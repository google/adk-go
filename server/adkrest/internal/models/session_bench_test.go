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

package models

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func BenchmarkFromSession(b *testing.B) {
	for _, n := range []int{100, 1000} {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			sess := seedBenchSession(b, n)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				got, err := FromSession(sess)
				if err != nil {
					b.Fatal(err)
				}
				if len(got.Events) != n {
					b.Fatalf("len(Events) = %d, want %d", len(got.Events), n)
				}
			}
		})
	}
}

func BenchmarkSessionIDFromHTTPParameters(b *testing.B) {
	vars := map[string]string{
		"app_name":   "my-app",
		"user_id":    "user-123",
		"session_id": "sess-456",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, err := SessionIDFromHTTPParameters(vars)
		if err != nil {
			b.Fatal(err)
		}
		if got.ID != "sess-456" {
			b.Fatalf("got.ID = %q", got.ID)
		}
	}
}

var benchText = strings.Repeat("The quick brown fox jumps over the lazy dog. ", 7)

func seedBenchSession(b *testing.B, n int) session.Session {
	b.Helper()
	ctx := b.Context()
	svc := session.InMemoryService()
	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "sess",
		State:     map[string]any{"k1": "v1", "k2": 42},
	})
	if err != nil {
		b.Fatal(err)
	}
	start := time.Unix(1_700_000_000, 0)
	for i := 0; i < n; i++ {
		ev := &session.Event{
			ID:           fmt.Sprintf("ev-%06d", i),
			InvocationID: fmt.Sprintf("inv-%06d", i/2),
			Timestamp:    start.Add(time.Duration(i) * time.Second),
		}
		switch {
		case i%2 == 0:
			ev.Author = "user"
			ev.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText(benchText, genai.RoleUser)}
		case i%8 == 1:
			ev.Author = "agent"
			ev.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{ID: fmt.Sprintf("call-%06d", i), Name: "lookup", Args: map[string]any{"q": "weather"}},
			}}}}
		case i%8 == 3:
			ev.Author = "agent"
			ev.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{ID: fmt.Sprintf("call-%06d", i-2), Name: "lookup", Response: map[string]any{"result": benchText}},
			}}}}
		default:
			ev.Author = "agent"
			ev.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText(benchText, genai.RoleModel)}
		}
		if i%10 == 9 {
			ev.Actions.StateDelta = map[string]any{"turn": i}
		}
		if err := svc.AppendEvent(ctx, created.Session, ev); err != nil {
			b.Fatal(err)
		}
	}
	resp, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "sess"})
	if err != nil {
		b.Fatal(err)
	}
	return resp.Session
}
