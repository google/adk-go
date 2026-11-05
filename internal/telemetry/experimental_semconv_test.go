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
	"testing"

	"github.com/google/go-cmp/cmp"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

func TestOperationDetailsNoteLLMRequest(t *testing.T) {
	setEnvForTesting(t, map[string]string{adkTelemetrySchemaVersionOptIn: "otel_semconv_1_44", captureMessageContentEnvVar: "SPAN_AND_EVENT"})
	setupTestTracer(t)
	_, span := tracer().Start(t.Context(), "generate_content")
	defer span.End()
	req := &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("original", genai.RoleUser)}}
	d := operationDetails{span: span}
	d.noteLLMRequest(req)
	first := d.eventRequest

	d.noteLLMRequest(req)
	if &d.eventRequest[0] != &first[0] {
		t.Error("noting an unchanged request encoded its messages again")
	}

	req.Contents[0].Parts[0].Text = "edited by the model"
	d.noteLLMRequest(req)
	want := []goKeyValue{{Key: "gen_ai.input.messages", Value: []any{map[string]any{
		"role":  "user",
		"parts": []any{map[string]any{"type": "text", "content": "edited by the model"}},
	}}}}
	if diff := cmp.Diff(want, toGoKeyValues(d.eventRequest)); diff != "" {
		t.Errorf("event messages after an in-place edit (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(want, toGoKeyValues(span.(sdktrace.ReadOnlySpan).Attributes())); diff != "" {
		t.Errorf("span messages after an in-place edit (-want +got):\n%s", diff)
	}
}
