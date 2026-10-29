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

package telemetry

import (
	"context"
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/attribute"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// A model may edit the request it is given, as gemini does when it appends a
// turn-alternation placeholder: the span and the event describe the request as
// the model left it.
func TestInstrumentGenerateContent_RecordsRequestAsTheModelLeftIt(t *testing.T) {
	setEnvForTesting(t, map[string]string{adkTelemetrySchemaVersionOptIn: "otel_semconv_1_44", captureMessageContentEnvVar: "SPAN_AND_EVENT"})
	spans := setupTestTracer(t)
	logs := &inMemoryExporter{}
	OverrideLoggerForTesting(t, sdklog.NewLoggerProvider(sdklog.WithProcessor(sdklog.NewSimpleProcessor(logs))))
	req := &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("original", genai.RoleUser)}}
	call := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
		return func(yield func(*model.LLMResponse, error) bool) {
			req.Contents = append(req.Contents, genai.NewContentFromText("appended by the model", genai.RoleUser))
			yield(&model.LLMResponse{Content: genai.NewContentFromText("ok", genai.RoleModel), FinishReason: genai.FinishReasonStop}, nil)
		}
	}
	params := GenerateContentParams{ModelName: "m", Request: req}
	for range InstrumentGenerateContent(t.Context(), params, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "" }, call) {
	}

	message := func(text string) any {
		return map[string]any{"role": "user", "parts": []any{map[string]any{"type": "text", "content": text}}}
	}
	want := []goKeyValue{{Key: string(genAIInputMessages), Value: []any{message("original"), message("appended by the model")}}}
	only := func(kvs []attribute.KeyValue) []goKeyValue {
		var out []goKeyValue
		for _, kv := range toGoKeyValues(kvs) {
			if kv.Key == string(genAIInputMessages) {
				out = append(out, kv)
			}
		}
		return out
	}
	if got := spans.GetSpans(); len(got) != 1 {
		t.Fatalf("got %d spans, want 1", len(got))
	} else if diff := cmp.Diff(want, only(got[0].Attributes)); diff != "" {
		t.Errorf("span messages (-want +got):\n%s", diff)
	}
	if len(logs.records) != 1 {
		t.Fatalf("got %d log records, want 1", len(logs.records))
	}
	var recorded []attribute.KeyValue
	logs.records[0].WalkAttributes(func(kv attribute.KeyValue) bool {
		recorded = append(recorded, kv)
		return true
	})
	if diff := cmp.Diff(want, only(recorded)); diff != "" {
		t.Errorf("event messages (-want +got):\n%s", diff)
	}
}
