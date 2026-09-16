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

package session_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
)

// TestNewEventDefaults covers the deprecated NewEvent, which must keep its
// original signature and use the wall clock and a random UUID.
func TestNewEventDefaults(t *testing.T) {
	before := time.Now()
	ev := session.NewEvent(t.Context(), "inv-1")
	after := time.Now()

	if ev.InvocationID != "inv-1" {
		t.Errorf("InvocationID = %q, want %q", ev.InvocationID, "inv-1")
	}
	if _, err := uuid.Parse(ev.ID); err != nil {
		t.Errorf("ID = %q, not a valid UUID: %v", ev.ID, err)
	}
	if ev.Timestamp.Before(before) || ev.Timestamp.After(after) {
		t.Errorf("Timestamp = %v, want within [%v, %v]", ev.Timestamp, before, after)
	}
}

func TestNewEventUsesProviders(t *testing.T) {
	fixedTime := time.Date(2024, time.January, 2, 3, 4, 5, 0, time.UTC)
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return fixedTime })
	ctx = platform.WithUUIDProvider(ctx, func() string { return "fixed-event-id" })

	ev := session.NewEvent(ctx, "inv-1")

	if ev.ID != "fixed-event-id" {
		t.Errorf("ID = %q, want %q", ev.ID, "fixed-event-id")
	}
	if !ev.Timestamp.Equal(fixedTime) {
		t.Errorf("Timestamp = %v, want %v", ev.Timestamp, fixedTime)
	}
	if ev.InvocationID != "inv-1" {
		t.Errorf("InvocationID = %q, want %q", ev.InvocationID, "inv-1")
	}
}

// TestNewEventDeterministicReplay verifies that replaying the same sequence of
// provider values produces identical events. This is the property a workflow
// engine relies on to make event creation replay-safe.
func TestNewEventDeterministicReplay(t *testing.T) {
	newCtx := func() context.Context {
		var ids int
		ctx := platform.WithUUIDProvider(t.Context(), func() string {
			ids++
			return "event-" + string(rune('0'+ids))
		})
		var times int
		return platform.WithTimeProvider(ctx, func() time.Time {
			times++
			return time.Date(2024, time.January, 1, 0, 0, times, 0, time.UTC)
		})
	}

	run := func(ctx context.Context) []*session.Event {
		return []*session.Event{
			session.NewEvent(ctx, "inv"),
			session.NewEvent(ctx, "inv"),
		}
	}

	first := run(newCtx())
	second := run(newCtx())

	for i := range first {
		if first[i].ID != second[i].ID {
			t.Errorf("event %d ID: first run %q, second run %q", i, first[i].ID, second[i].ID)
		}
		if !first[i].Timestamp.Equal(second[i].Timestamp) {
			t.Errorf("event %d Timestamp: first run %v, second run %v", i, first[i].Timestamp, second[i].Timestamp)
		}
	}
}

// TestIsFinalResponseSkipsNilParts covers events whose Content.Parts hold a nil
// *genai.Part. Parts is a []*genai.Part, so a nil element is representable in
// content decoded from JSON or produced outside the library; the helpers
// behind IsFinalResponse must skip it rather than panic.
func TestIsFinalResponseSkipsNilParts(t *testing.T) {
	text := &genai.Part{Text: "hi"}
	call := &genai.Part{FunctionCall: &genai.FunctionCall{Name: "f"}}
	response := &genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "f"}}
	codeResult := &genai.Part{CodeExecutionResult: &genai.CodeExecutionResult{}}

	for _, tc := range []struct {
		name  string
		parts []*genai.Part
		want  bool
	}{
		{name: "only nil", parts: []*genai.Part{nil}, want: true},
		{name: "nil before text", parts: []*genai.Part{nil, text}, want: true},
		{name: "nil after text", parts: []*genai.Part{text, nil}, want: true},
		{name: "nil before function call", parts: []*genai.Part{nil, call}, want: false},
		{name: "nil between function responses", parts: []*genai.Part{response, nil, response}, want: false},
		{name: "nil before code execution result", parts: []*genai.Part{nil, codeResult}, want: false},
		{name: "nil after code execution result", parts: []*genai.Part{codeResult, nil}, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			event := &session.Event{
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{Role: genai.RoleModel, Parts: tc.parts},
				},
			}
			if got := event.IsFinalResponse(); got != tc.want {
				t.Errorf("IsFinalResponse() = %v, want %v", got, tc.want)
			}
		})
	}

	t.Run("decoded from JSON", func(t *testing.T) {
		var event session.Event
		if err := json.Unmarshal([]byte(`{"content":{"role":"model","parts":[null,{"text":"hi"}]}}`), &event); err != nil {
			t.Fatalf("json.Unmarshal failed: %v", err)
		}
		if got := event.IsFinalResponse(); got != true {
			t.Errorf("IsFinalResponse() = %v, want true", got)
		}
	})
}
