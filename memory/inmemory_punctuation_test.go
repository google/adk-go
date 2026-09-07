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

package memory_test

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestInMemoryServiceSearchMemoryNormalizesPunctuationAndWhitespace(t *testing.T) {
	s := memory.InMemoryService()
	content := genai.NewContentFromText("Error: connection\ntimeout! Please\tretry.", genai.RoleModel)
	if err := s.AddSessionToMemory(t.Context(), makeSession(t, "app1", "user1", "sess1", []*session.Event{
		{
			ID: "event1",
			LLMResponse: model.LLMResponse{
				Content: content,
			},
		},
	})); err != nil {
		t.Fatalf("AddSessionToMemory() error = %v", err)
	}

	for _, query := range []string{"error", "timeout", "retry"} {
		got, err := s.SearchMemory(t.Context(), &memory.SearchRequest{
			AppName: "app1",
			UserID:  "user1",
			Query:   query,
		})
		if err != nil {
			t.Fatalf("SearchMemory(%q) error = %v", query, err)
		}
		if len(got.Memories) != 1 || got.Memories[0].ID != "event1" {
			t.Fatalf("SearchMemory(%q) = %+v, want event1", query, got.Memories)
		}
	}
}
