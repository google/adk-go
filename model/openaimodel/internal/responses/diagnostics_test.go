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
	"errors"
	"io"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/openaimodel/internal/shared"
)

func TestModel_GenerateStream_TerminalConversionDiagnostic(t *testing.T) {
	const (
		message            = `{"type":"message","content":[{"type":"output_text","text":"hello"}]}`
		unsupported        = `{"type":"private-output-type"}`
		unsupportedContent = `{"type":"message","content":[{"type":"private-content-type"}]}`
		added              = `{"type":"response.output_item.added","item":{"type":"function_call","id":"item_1","call_id":"private-call-id","name":"private-tool-name"}}`
		argsDone           = `{"type":"response.function_call_arguments.done","item_id":"item_1","arguments":"{\"value\":\"private-argument\"}"}`
		badCall            = `{"type":"function_call","name":"private-tool-name","call_id":"private-call-id","arguments":"private-invalid-arguments"}`
	)
	completed := func(output string) string {
		return `{"type":"response.completed","response":{"id":"private-response-id","model":"private-model-name","status":"completed","output":` + output + `}}`
	}
	tests := []struct {
		name       string
		events     []string
		wantType   string
		wantErr    error
		wantText   string
		wantCall   bool
		wantFinish genai.FinishReason
		stopEarly  bool
	}{
		{
			name:     "unsupported output retains streamed text",
			events:   []string{evDelta1, completed("[" + message + "," + unsupported + "]")},
			wantType: "unsupported_output_item", wantText: "hel",
		},
		{
			name:     "server tool output prevents terminal conversion",
			events:   []string{evDelta1, completed("[" + message + `,{"type":"web_search_call","id":"search_1","status":"completed"}]`)},
			wantType: "unsupported_output_item", wantText: "hel",
		},
		{
			name:     "unsupported message content retains streamed text",
			events:   []string{evDelta1, completed("[" + message + "," + unsupportedContent + "]")},
			wantType: "unsupported_message_content", wantText: "hel",
		},
		{
			name:     "invalid arguments recover the streamed call",
			events:   []string{evDelta1, added, argsDone, completed("[" + message + "," + badCall + "]")},
			wantType: "invalid_function_call_arguments", wantText: "hel", wantCall: true,
		},
		{
			name:     "later terminal does not duplicate diagnostic",
			events:   []string{evDelta1, completed("[" + unsupported + "]"), evCompleted},
			wantType: "unsupported_output_item", wantText: "hel",
		},
		{
			name:   "successful terminal conversion",
			events: []string{evDelta1, evCompleted}, wantText: "hello",
		},
		{
			name:     "terminal omits streamed text",
			events:   []string{evDelta1, completed(`[{"type":"message","content":[{"type":"output_text","text":"unrelated"}]}]`)},
			wantText: "hel",
		},
		{
			name:   "empty output",
			events: []string{evDelta1, completed("[]")}, wantText: "hel",
		},
		{
			name:   "empty message",
			events: []string{evDelta1, completed(`[{"type":"message","content":[]}]`)}, wantText: "hel",
		},
		{
			name:   "no aggregate returns conversion error",
			events: []string{completed("[" + unsupported + "]")}, wantErr: shared.ErrUnsupportedOutputItemType,
		},
		{
			name:   "invalid arguments without a streamed call return an error",
			events: []string{evDelta1, completed("[" + badCall + "]")}, wantErr: shared.ErrFunctionCallArgs,
		},
		{
			name:   "later call failure suppresses recovery diagnostic",
			events: []string{evDelta1, completed("[" + unsupported + "," + badCall + "]")}, wantErr: shared.ErrFunctionCallArgs,
		},
		{
			name:   "server failure is returned",
			events: []string{evDelta1, evFailed}, wantErr: shared.ErrResponseFailed,
		},
		{
			name:     "incomplete terminal does not attempt recovery",
			events:   []string{evDelta1, evMaxTokens, completed("[" + unsupported + "]")},
			wantText: "hel", wantFinish: genai.FinishReasonMaxTokens,
		},
		{
			name: "incomplete terminal recovers invalid arguments without diagnostic",
			events: []string{
				evDelta1, added, argsDone,
				`{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[` + badCall + `]}}`,
			},
			wantText: "hel", wantCall: true, wantFinish: genai.FinishReasonMaxTokens,
		},
		{
			name:   "consumer stops before terminal",
			events: []string{evDelta1, completed("[" + unsupported + "]")}, stopEarly: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for _, recording := range []bool{true, false} {
				name := "recording"
				sampler := sdktrace.AlwaysSample()
				if !recording {
					name = "not recording"
					sampler = sdktrace.NeverSample()
				}
				t.Run(name, func(t *testing.T) {
					recorder := tracetest.NewSpanRecorder()
					provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder), sdktrace.WithSampler(sampler))
					t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
					ctx, span := provider.Tracer("test").Start(t.Context(), "generate_content")
					defer span.End()

					payload := "data: " + strings.Join(tc.events, "\n\ndata: ") + "\n\ndata: [DONE]\n\n"
					llm := newTestModel("test-model", &testClientConfig{
						APIKey: "test", BaseURL: "https://example.invalid/v1",
						HTTPClient: &http.Client{Transport: terminalDiagnosticTransport{payload: payload}},
					})
					req := &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("private-prompt", genai.RoleUser)}}
					var final *model.LLMResponse
					var gotErr error
					finals, partials := 0, 0
					for resp, err := range llm.GenerateContent(ctx, req, true) {
						if err != nil {
							gotErr = err
							break
						}
						if resp.Partial {
							partials++
						} else {
							// InstrumentGenerateContent ends its span before
							// yielding the final response to the caller.
							span.End()
							final = resp
							finals++
						}
						if tc.stopEarly {
							break
						}
					}
					span.End()
					if !errors.Is(gotErr, tc.wantErr) {
						t.Fatal("stream error does not match expected sentinel")
					}
					if tc.stopEarly || tc.wantErr != nil {
						if finals != 0 {
							t.Errorf("stream emitted %d final responses, want none", finals)
						}
					} else {
						if finals != 1 || partials == 0 {
							t.Fatalf("stream emitted %d finals and %d partials, want one final after partial content", finals, partials)
						}
						wantParts := []*genai.Part{{Text: tc.wantText}}
						if tc.wantCall {
							wantParts = append(wantParts, &genai.Part{FunctionCall: &genai.FunctionCall{
								Name: "private-tool-name", ID: "private-call-id", Args: map[string]any{"value": "private-argument"},
							}})
						}
						if !cmp.Equal(final.Content, &genai.Content{Role: string(genai.RoleModel), Parts: wantParts}) {
							t.Error("final content changed during fallback")
						}
						finish := tc.wantFinish
						if finish == "" {
							finish = genai.FinishReasonStop
						}
						if final.FinishReason != finish || !final.TurnComplete || final.ErrorCode != "" || final.ErrorMessage != "" {
							t.Error("fallback changed the turn completion or error fields")
						}
					}
					spans := recorder.Ended()
					if !recording {
						if len(spans) != 0 {
							t.Error("unsampled call exported a span")
						}
						return
					}
					if len(spans) != 1 {
						t.Fatalf("recorded %d spans, want 1", len(spans))
					}
					if spans[0].Status().Code != codes.Unset {
						t.Error("fallback changed the span status")
					}
					events := spans[0].Events()
					if tc.wantType == "" {
						if len(events) != 0 {
							t.Errorf("recorded %d diagnostic events, want none", len(events))
						}
						return
					}
					if len(events) != 1 {
						t.Fatalf("recorded %d diagnostic events, want 1", len(events))
					}
					if events[0].Name != "openai.terminal_response_conversion_failed" {
						t.Error("unexpected diagnostic event name")
					}
					// Exact attributes reject response data and wrapped error text.
					wantAttrs := []attribute.KeyValue{attribute.String("error.type", tc.wantType)}
					if !slices.Equal(events[0].Attributes, wantAttrs) {
						t.Error("diagnostic must contain only the expected error category")
					}
				})
			}
		})
	}
}

type terminalDiagnosticTransport struct{ payload string }

func (tr terminalDiagnosticTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(tr.payload)),
		Request:    req,
	}, nil
}
