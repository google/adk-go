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

package helper

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/server/agentengine/internal/models"
	"google.golang.org/adk/v2/session"
)

// benchText is the body of the benchmark's text event, ~300 bytes.
var benchText = strings.Repeat("The quick brown fox jumps over the lazy dog. ", 7)

// benchTextEvent is a complete model turn: text, finish reason, usage.
func benchTextEvent() session.Event {
	return session.Event{
		ID:           "ev-000042",
		InvocationID: "e-5d2c1b8e-0c1c-4a4e-9a9e-1f1c3d5e7f90",
		Author:       "assistant",
		Timestamp:    time.Unix(1_700_000_000, 0),
		LLMResponse: model.LLMResponse{
			Content:      genai.NewContentFromText(benchText, genai.RoleModel),
			FinishReason: genai.FinishReasonStop,
			TurnComplete: true,
			ModelVersion: "gemini-2.5-flash",
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     1234,
				CandidatesTokenCount: 87,
				TotalTokenCount:      1321,
			},
		},
		Actions: session.EventActions{
			StateDelta:    map[string]any{"turn": 21},
			ArtifactDelta: map[string]int64{},
		},
	}
}

// benchFunctionCallEvent is a model turn that calls a tool with a few arguments.
func benchFunctionCallEvent() session.Event {
	ev := benchTextEvent()
	ev.ID = "ev-000043"
	ev.LLMResponse.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{
		FunctionCall: &genai.FunctionCall{
			ID:   "call-000043",
			Name: "lookup_weather",
			Args: map[string]any{"city": "Warsaw", "unit": "celsius", "days": 3.0, "detailed": true},
		},
	}}}
	ev.LLMResponse.FinishReason = ""
	ev.LLMResponse.TurnComplete = false
	ev.LongRunningToolIDs = []string{"call-000043"}
	return ev
}

// benchEnvelope wraps an event the way streaming_agent_run_with_events emits it.
func benchEnvelope(ev session.Event) models.StreamingAgentRunWithEventsResponse {
	return models.StreamingAgentRunWithEventsResponse{Events: []*session.Event{&ev}, SessionID: "s-1"}
}

// discardResponseWriter is an http.ResponseWriter that drops its body, so the
// benchmark measures encoding rather than buffering.
type discardResponseWriter struct{ h http.Header }

func (d *discardResponseWriter) Header() http.Header         { return d.h }
func (d *discardResponseWriter) Write(p []byte) (int, error) { return len(p), nil }
func (d *discardResponseWriter) WriteHeader(int)             {}
func (d *discardResponseWriter) Flush()                      {}

// BenchmarkConvertSnake is the snake_case conversion alone, on the two event
// shapes an Agent Engine stream carries most, each as the bare event
// (stream_query) and inside the streaming_agent_run_with_events envelope.
func BenchmarkConvertSnake(b *testing.B) {
	for _, tc := range []struct {
		name  string
		value any
	}{
		{"text", benchTextEvent()},
		{"function-call", benchFunctionCallEvent()},
		{"text/envelope", benchEnvelope(benchTextEvent())},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				ConvertSnake(tc.value)
			}
		})
	}
}

// BenchmarkEmitJSON is what the handlers pay per emitted event: the
// conversion plus json encoding of its result.
func BenchmarkEmitJSON(b *testing.B) {
	for _, tc := range []struct {
		name  string
		value any
	}{
		{"text", benchTextEvent()},
		{"function-call", benchFunctionCallEvent()},
	} {
		b.Run(tc.name, func(b *testing.B) {
			rw := &discardResponseWriter{h: http.Header{}}
			b.ReportAllocs()
			for b.Loop() {
				if err := EmitJSON(rw, tc.value); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkJSONMarshal is the reference point: encoding/json on the same
// events, camelCase, no conversion.
func BenchmarkJSONMarshal(b *testing.B) {
	for _, tc := range []struct {
		name  string
		value any
	}{
		{"text", benchTextEvent()},
		{"function-call", benchFunctionCallEvent()},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := json.Marshal(tc.value); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
