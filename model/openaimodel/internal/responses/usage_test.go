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

package responses

import (
	"fmt"
	"math"
	"net/http"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/openai/openai-go/v3"
	oairesponses "github.com/openai/openai-go/v3/responses"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/openaimodel/internal/shared"
)

// TestConvertUsage_InvalidCountDropsUsage sets each count, keyed by the name
// the Responses API gives it, to a value no request can have. Usage must be
// dropped, and InvalidUsage must name that field, since the name is what the
// span event reports.
func TestConvertUsage_InvalidCountDropsUsage(t *testing.T) {
	setters := map[string]func(*oairesponses.ResponseUsage, int64){
		"input_tokens":  func(u *oairesponses.ResponseUsage, v int64) { u.InputTokens = v },
		"output_tokens": func(u *oairesponses.ResponseUsage, v int64) { u.OutputTokens = v },
		"total_tokens":  func(u *oairesponses.ResponseUsage, v int64) { u.TotalTokens = v },
		"input_tokens_details.cached_tokens": func(u *oairesponses.ResponseUsage, v int64) {
			u.InputTokensDetails.CachedTokens = v
		},
		"output_tokens_details.reasoning_tokens": func(u *oairesponses.ResponseUsage, v int64) {
			u.OutputTokensDetails.ReasoningTokens = v
		},
	}
	// Each wraps or saturates to a plausible int32 if narrowed blindly:
	// MinInt32-1 to MaxInt32, -2^32+1 to 1, MaxInt32+1 to MaxInt32, 2^32 to 0.
	invalid := []int64{-1, math.MinInt32, math.MinInt32 - 1, -(1 << 32) + 1, math.MaxInt32 + 1, 1 << 32, math.MaxInt64}
	for field, set := range setters {
		for _, v := range invalid {
			t.Run(fmt.Sprintf("%s=%d", field, v), func(t *testing.T) {
				u := oairesponses.ResponseUsage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15}
				set(&u, v)
				if got := convertUsage(u); got != nil {
					t.Errorf("convertUsage() = %+v, want nil for an impossible count", got)
				}
				if name, _ := shared.InvalidUsage(usageCounts(u)...); name != field {
					t.Errorf("InvalidUsage(usageCounts()) name = %q, want %q", name, field)
				}
			})
		}
	}
}

func TestConvertUsage_ValidBounds(t *testing.T) {
	for _, v := range []int64{0, math.MaxInt32} {
		t.Run(fmt.Sprint(v), func(t *testing.T) {
			u := oairesponses.ResponseUsage{InputTokens: v, OutputTokens: v, TotalTokens: v}
			u.InputTokensDetails.CachedTokens = v
			u.OutputTokensDetails.ReasoningTokens = v
			n := int32(v)
			want := &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:        n,
				CandidatesTokenCount:    n,
				TotalTokenCount:         n,
				CachedContentTokenCount: n,
				ThoughtsTokenCount:      n,
				PromptTokensDetails:     []*genai.ModalityTokenCount{{Modality: genai.MediaModalityText, TokenCount: n}},
				CandidatesTokensDetails: []*genai.ModalityTokenCount{{Modality: genai.MediaModalityText, TokenCount: n}},
			}
			if diff := cmp.Diff(want, convertUsage(u)); diff != "" {
				t.Errorf("convertUsage() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConvertUsage_MapsEachCount(t *testing.T) {
	u := oairesponses.ResponseUsage{InputTokens: 11, OutputTokens: 22, TotalTokens: 33}
	u.InputTokensDetails.CachedTokens = 4
	u.OutputTokensDetails.ReasoningTokens = 5
	want := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        11,
		CandidatesTokenCount:    22,
		TotalTokenCount:         33,
		CachedContentTokenCount: 4,
		ThoughtsTokenCount:      5,
		PromptTokensDetails:     []*genai.ModalityTokenCount{{Modality: genai.MediaModalityText, TokenCount: 11}},
		CandidatesTokensDetails: []*genai.ModalityTokenCount{{Modality: genai.MediaModalityText, TokenCount: 22}},
	}
	if diff := cmp.Diff(want, convertUsage(u)); diff != "" {
		t.Errorf("convertUsage() mismatch (-want +got):\n%s", diff)
	}
}

// TestModel_InvalidUsageIsUnknown drives both paths with a server whose usage
// wraps when narrowed. The turn must survive with its answer intact, report
// usage as unknown, and say why on the model call's span.
func TestModel_InvalidUsageIsUnknown(t *testing.T) {
	body := func(usage string) string {
		return `{"id":"resp_1","model":"stream-model","status":"completed","usage":` + usage + `,` +
			`"output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]}]}`
	}
	event := func(field, reason string) []attribute.KeyValue {
		return []attribute.KeyValue{
			attribute.String("openai.usage.field", field),
			attribute.String("openai.usage.reason", reason),
		}
	}
	for _, tc := range []struct {
		name      string
		body      string
		wantEvent []attribute.KeyValue // nil: usage is valid and kept
	}{
		{
			name:      "negative",
			body:      body(`{"input_tokens":-2147483649,"output_tokens":4,"total_tokens":7}`),
			wantEvent: event("input_tokens", "negative"),
		},
		{
			name:      "exceeds int32",
			body:      body(`{"input_tokens":3,"output_tokens":2147483648,"total_tokens":7}`),
			wantEvent: event("output_tokens", "exceeds_int32"),
		},
		{name: "valid", body: body(`{"input_tokens":3,"output_tokens":4,"total_tokens":7}`)},
	} {
		wantUsage := tc.wantEvent == nil
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", tc.name, stream), func(t *testing.T) {
				server := newLoopbackServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					if stream {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = fmt.Fprintf(w, "data: %s\n\ndata: %s\n\ndata: [DONE]\n\n",
							`{"type":"response.output_text.delta","delta":"hello"}`,
							`{"type":"response.completed","response":`+tc.body+`}`)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					_, _ = fmt.Fprint(w, tc.body)
				}))
				recorder := tracetest.NewSpanRecorder()
				provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
				t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
				ctx, span := provider.Tracer("test").Start(t.Context(), "generate_content")

				llm := newTestModel(openai.ChatModelGPT4oMini, &testClientConfig{
					APIKey: "test", BaseURL: server.URL + "/v1", HTTPClient: server.Client(),
				})
				req := &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)}}
				var final *model.LLMResponse
				for resp, err := range llm.GenerateContent(ctx, req, stream) {
					if err != nil {
						t.Fatalf("GenerateContent() err = %v, want the turn to survive", err)
					}
					if !resp.Partial {
						final = resp
					}
				}
				span.End()

				if final == nil {
					t.Fatal("no final response")
				}
				wantContent := &genai.Content{Role: string(genai.RoleModel), Parts: []*genai.Part{{Text: "hello"}}}
				if diff := cmp.Diff(wantContent, final.Content); diff != "" {
					t.Errorf("content changed (-want +got):\n%s", diff)
				}
				if gotUsage := final.UsageMetadata != nil; gotUsage != wantUsage {
					t.Errorf("usage = %+v, want present=%t", final.UsageMetadata, wantUsage)
				}
				spans := recorder.Ended()
				if len(spans) != 1 {
					t.Fatalf("recorded %d spans, want 1", len(spans))
				}
				var events [][]attribute.KeyValue
				for _, ev := range spans[0].Events() {
					if ev.Name == "openai.usage_invalid" {
						events = append(events, ev.Attributes)
					}
				}
				switch {
				case wantUsage && len(events) != 0:
					t.Errorf("recorded %d usage_invalid events, want none", len(events))
				case !wantUsage && (len(events) != 1 || !slices.Equal(events[0], tc.wantEvent)):
					t.Errorf("usage_invalid events = %v, want exactly one with %v", events, tc.wantEvent)
				}
			})
		}
	}
}
