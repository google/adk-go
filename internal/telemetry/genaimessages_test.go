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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// cloudTraceAttributeValueLimit is the largest attribute value Cloud Trace
// accepts; anything larger is dropped without warning. It is stated here as
// the external limit the implementation has to respect, deliberately not
// derived from maxContentAttributeBytes, so this bound still means something
// if that constant changes.
const cloudTraceAttributeValueLimit = 64 * 1024

// captureContent turns on the message content opt-in for one test.
func captureContent(t *testing.T) {
	t.Helper()
	// Spans have to be asked for by name. A truthy value means log records
	// only, which is what the variable meant before spans could carry content.
	t.Setenv(captureMessageContentEnvVar, "SPAN_AND_EVENT")
	ApplyEnv()
}

// attrJSON returns the named attribute, and whether it was set. A structured
// value is returned encoded as JSON, so assertions hold for both encodings.
func attrJSON(attrs []attribute.KeyValue, key attribute.Key) (string, bool) {
	for _, kv := range attrs {
		if kv.Key != key {
			continue
		}
		if kv.Value.Type() == attribute.STRING {
			return kv.Value.AsString(), true
		}
		b, err := json.Marshal(FromLogValue(kv.Value))
		return string(b), err == nil
	}
	return "", false
}

// jsonDiff compares two JSON documents by value, not by field order.
func jsonDiff(t *testing.T, want, got string) string {
	t.Helper()
	var w, g any
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("want is not JSON: %v", err)
	}
	if err := json.Unmarshal([]byte(got), &g); err != nil {
		t.Fatalf("got is not JSON: %v", err)
	}
	return cmp.Diff(w, g)
}

// --- Schema conformance ------------------------------------------------

// requiredPartFields mirrors the "required" list of each part definition in
// the OpenTelemetry GenAI input and output message schemas. A part type that
// is absent here is a GenericPart, which requires only "type".
var requiredPartFields = map[string][]string{
	"text":                      {"type", "content"},
	"reasoning":                 {"type", "content"},
	"tool_call":                 {"type", "name"},
	"tool_call_response":        {"type", "response"},
	"server_tool_call":          {"type", "name", "server_tool_call"},
	"server_tool_call_response": {"type", "server_tool_call_response"},
	"blob":                      {"type", "modality", "content"},
	"file":                      {"type", "modality", "file_id"},
	"uri":                       {"type", "modality", "uri"},
	"compaction":                {"type"},
}

var (
	schemaRoles         = map[string]bool{"system": true, "user": true, "assistant": true, "tool": true}
	schemaFinishReasons = map[string]bool{
		"stop": true, "length": true, "content_filter": true,
		"tool_call": true, "compaction": true, "error": true,
	}
	schemaModalities = map[string]bool{"image": true, "video": true, "audio": true, "document": true}
)

// enumFields are the fields whose values ADK derives from genai enums. They
// must never carry a protobuf wire name such as GOOGLE_SEARCH_WEB.
var enumFields = map[string]bool{
	"type": true, "role": true, "name": true, "modality": true,
	"outcome": true, "language": true, "finish_reason": true,
}

// validateMessages checks a gen_ai.input.messages or gen_ai.output.messages
// value against the schema. wantFinishReason asks for the output message
// rules, where finish_reason is required.
//
// It is stricter than the published schema in two places, both deliberate:
// role and finish_reason are anyOf[enum, string], but ADK derives them from
// genai values it knows, so anything outside the enum means a mapping was
// missed.
func validateMessages(t *testing.T, encoded string, wantFinishReason bool) []map[string]any {
	t.Helper()
	var msgs []map[string]any
	if err := json.Unmarshal([]byte(encoded), &msgs); err != nil {
		t.Fatalf("messages are not a JSON array of objects: %v\n%s", err, encoded)
	}
	for i, m := range msgs {
		role, ok := m["role"].(string)
		if !ok {
			t.Errorf("message %d: missing string role: %v", i, m)
		} else if !schemaRoles[role] {
			t.Errorf("message %d: role %q is not in the schema enum", i, role)
		}
		parts, ok := m["parts"].([]any)
		if !ok {
			t.Errorf("message %d: missing parts array: %v", i, m)
		}
		if wantFinishReason {
			fr, ok := m["finish_reason"].(string)
			if !ok || fr == "" {
				t.Errorf("message %d: finish_reason is required on output messages: %v", i, m)
			} else if !schemaFinishReasons[fr] {
				t.Errorf("message %d: finish_reason %q is not in the schema enum", i, fr)
			}
		} else if _, present := m["finish_reason"]; present {
			t.Errorf("message %d: finish_reason must not appear on input messages", i)
		}
		for j, p := range parts {
			validatePart(t, fmt.Sprintf("message %d part %d", i, j), p)
		}
		checkEnumFields(t, fmt.Sprintf("message %d", i), m)
	}
	return msgs
}

// validateParts checks a gen_ai.system_instructions value, whose items are
// TextParts or GenericParts.
func validateParts(t *testing.T, encoded string) []any {
	t.Helper()
	var parts []any
	if err := json.Unmarshal([]byte(encoded), &parts); err != nil {
		t.Fatalf("system instructions are not a JSON array: %v\n%s", err, encoded)
	}
	for i, p := range parts {
		validatePart(t, fmt.Sprintf("instruction part %d", i), p)
	}
	return parts
}

func validatePart(t *testing.T, where string, p any) {
	t.Helper()
	obj, ok := p.(map[string]any)
	if !ok {
		t.Errorf("%s: part is not an object: %v", where, p)
		return
	}
	typ, ok := obj["type"].(string)
	if !ok || typ == "" {
		t.Errorf("%s: every part requires a non-empty string type: %v", where, obj)
		return
	}
	for _, field := range requiredPartFields[typ] {
		if _, present := obj[field]; !present {
			t.Errorf("%s: part type %q requires field %q: %v", where, typ, field, obj)
		}
	}
	if m, present := obj["modality"]; present {
		if s, ok := m.(string); !ok || !schemaModalities[s] {
			t.Errorf("%s: modality %v is not in the schema enum", where, m)
		}
	}
	// Nested server tool bodies are themselves objects requiring a type.
	for _, key := range []string{"server_tool_call", "server_tool_call_response"} {
		if nested, present := obj[key]; present {
			n, ok := nested.(map[string]any)
			if !ok {
				t.Errorf("%s: %s is not an object: %v", where, key, nested)
				continue
			}
			if s, ok := n["type"].(string); !ok || s == "" {
				t.Errorf("%s: %s requires a non-empty string type: %v", where, key, n)
			}
			checkEnumFields(t, where+" "+key, n)
		}
	}
	checkEnumFields(t, where, obj)
}

// checkEnumFields fails if an enum-derived field still carries a protobuf
// wire name, which is SCREAMING_SNAKE_CASE.
func checkEnumFields(t *testing.T, where string, obj map[string]any) {
	t.Helper()
	for k, v := range obj {
		if !enumFields[k] {
			continue
		}
		s, ok := v.(string)
		if !ok || s == "" {
			continue
		}
		if s == strings.ToUpper(s) && strings.ContainsAny(s, "ABCDEFGHIJKLMNOPQRSTUVWXYZ") {
			t.Errorf("%s: field %q leaks the protobuf wire name %q into telemetry", where, k, s)
		}
	}
}

// --- Part mapping ------------------------------------------------------
func TestRequestContentAttributes_TextPartShape(t *testing.T) {
	captureContent(t)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "Weather in Paris?"}}},
			{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "Rainy."}}},
		},
	}
	got, ok := attrJSON(spanContentAttributes(requestContent(req)), genAIInputMessages)
	if !ok {
		t.Fatal("gen_ai.input.messages was not set")
	}
	want := `[{"role":"user","parts":[{"type":"text","content":"Weather in Paris?"}]},` +
		`{"role":"assistant","parts":[{"type":"text","content":"Rainy."}]}]`
	if diff := jsonDiff(t, want, got); diff != "" {
		t.Errorf("gen_ai.input.messages mismatch (-want +got):\n%s", diff)
	}
}

func TestSchemaRole(t *testing.T) {
	tests := []struct {
		name    string
		content *genai.Content
		want    string
	}{
		{"user", &genai.Content{Role: genai.RoleUser}, "user"},
		{"model becomes assistant", &genai.Content{Role: genai.RoleModel}, "assistant"},
		{"empty defaults to user", &genai.Content{}, "user"},
		{
			name: "function response turn is a tool message",
			content: &genai.Content{
				Role:  genai.RoleUser,
				Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Name: "t"}}},
			},
			want: "tool",
		},
		{
			name: "server tool response turn is a tool message",
			content: &genai.Content{
				Role:  genai.RoleModel,
				Parts: []*genai.Part{{ToolResponse: &genai.ToolResponse{ID: "1"}}},
			},
			want: "tool",
		},
		{
			name: "mixed turn with a tool result is a tool message",
			content: &genai.Content{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "here you go"},
					{FunctionResponse: &genai.FunctionResponse{Name: "t"}},
				},
			},
			want: "tool",
		},
		{"unknown role passes through", &genai.Content{Role: "narrator"}, "narrator"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := schemaRole(tc.content); got != tc.want {
				t.Errorf("schemaRole() = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestRequestContentAttributes_ToolTurnRole checks the role a full tool
// round trip produces, since genai labels a tool result as "user".
func TestRequestContentAttributes_ToolTurnRole(t *testing.T) {
	captureContent(t)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "weather?"}}},
			{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "get_weather"}}}},
			{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				Name: "get_weather", Response: map[string]any{"temp": 12},
			}}}},
		},
	}
	encoded, ok := attrJSON(spanContentAttributes(requestContent(req)), genAIInputMessages)
	if !ok {
		t.Fatal("gen_ai.input.messages was not set")
	}
	msgs := validateMessages(t, encoded, false)
	var gotRoles []string
	for _, m := range msgs {
		gotRoles = append(gotRoles, m["role"].(string))
	}
	if diff := cmp.Diff([]string{"user", "assistant", "tool"}, gotRoles); diff != "" {
		t.Errorf("roles (-want +got):\n%s", diff)
	}
}

func TestRequestContentAttributes_SystemInstructions(t *testing.T) {
	captureContent(t)

	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{Parts: []*genai.Part{
				{Text: "You are a translator."},
				{Text: "Translate English to French."},
			}},
		},
	}
	attrs := spanContentAttributes(requestContent(req))
	got, ok := attrJSON(attrs, genAISystemInstructions)
	if !ok {
		t.Fatalf("gen_ai.system_instructions was not set; got %v", attrs)
	}
	validateParts(t, got)
	want := `[{"type":"text","content":"You are a translator."},` +
		`{"type":"text","content":"Translate English to French."}]`
	if diff := jsonDiff(t, want, got); diff != "" {
		t.Errorf("gen_ai.system_instructions mismatch (-want +got):\n%s", diff)
	}
	if _, present := attrJSON(attrs, genAIInputMessages); present {
		t.Error("gen_ai.input.messages must not be set when there are no contents")
	}
}

// --- Output messages ---------------------------------------------------

func TestResponseContentAttributes(t *testing.T) {
	captureContent(t)

	resp := &model.LLMResponse{
		Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "It is rainy."}}},
		FinishReason: genai.FinishReasonStop,
	}
	got, ok := attrJSON(spanContentAttributes(responseContent(resp, nil)), genAIOutputMessages)
	if !ok {
		t.Fatal("gen_ai.output.messages was not set")
	}
	validateMessages(t, got, true)
	want := `[{"role":"assistant","parts":[{"type":"text","content":"It is rainy."}],"finish_reason":"stop"}]`
	if diff := jsonDiff(t, want, got); diff != "" {
		t.Errorf("gen_ai.output.messages mismatch (-want +got):\n%s", diff)
	}
}

// TestResponseContentAttributes_SkipsStreamingChunks checks that a partial
// chunk records nothing. Recording one would overwrite the attribute with a
// fragment of the turn.
func TestResponseContentAttributes_SkipsStreamingChunks(t *testing.T) {
	captureContent(t)

	partial := &model.LLMResponse{
		Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "It is ra"}}},
		Partial: true,
	}
	if attrs := spanContentAttributes(responseContent(partial, nil)); len(attrs) != 0 {
		t.Errorf("attributes for a partial chunk = %v, want none", attrs)
	}

	settled := &model.LLMResponse{
		Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "It is rainy."}}},
		FinishReason: genai.FinishReasonStop,
	}
	got, ok := attrJSON(spanContentAttributes(responseContent(settled, nil)), genAIOutputMessages)
	if !ok {
		t.Fatal("gen_ai.output.messages was not set for the settled response")
	}
	if !strings.Contains(got, "It is rainy.") {
		t.Errorf("settled response was not recorded: %s", got)
	}
}

// TestResponseContentAttributes_EmptyContent covers a response blocked before
// any content was produced: finish_reason is required, so a message is still
// emitted.
func TestResponseContentAttributes_EmptyContent(t *testing.T) {
	captureContent(t)

	resp := &model.LLMResponse{FinishReason: genai.FinishReasonSafety}
	got, ok := attrJSON(spanContentAttributes(responseContent(resp, nil)), genAIOutputMessages)
	if !ok {
		t.Fatal("gen_ai.output.messages was not set")
	}
	validateMessages(t, got, true)
	want := `[{"role":"assistant","parts":[],"finish_reason":"content_filter"}]`
	if diff := jsonDiff(t, want, got); diff != "" {
		t.Errorf("gen_ai.output.messages mismatch (-want +got):\n%s", diff)
	}
}

func TestSchemaFinishReason(t *testing.T) {
	tests := []struct {
		name string
		resp *model.LLMResponse
		want string
		err  error
	}{
		{name: "stop", resp: &model.LLMResponse{FinishReason: genai.FinishReasonStop}, want: "stop"},
		{name: "stop with a tool call", resp: &model.LLMResponse{
			Content:      &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{}}}},
			FinishReason: genai.FinishReasonStop,
		}, want: "tool_call"},
		{name: "unset", resp: &model.LLMResponse{}, want: "stop"},
		{name: "unset with a tool call", resp: &model.LLMResponse{
			Content: &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{}}}},
		}, want: "tool_call"},
		{name: "unspecified", resp: &model.LLMResponse{FinishReason: genai.FinishReasonUnspecified}, want: "stop"},
		{name: "max tokens", resp: &model.LLMResponse{FinishReason: genai.FinishReasonMaxTokens}, want: "length"},
		{name: "safety", resp: &model.LLMResponse{FinishReason: genai.FinishReasonSafety}, want: "content_filter"},
		{name: "recitation", resp: &model.LLMResponse{FinishReason: genai.FinishReasonRecitation}, want: "content_filter"},
		{name: "blocklist", resp: &model.LLMResponse{FinishReason: genai.FinishReasonBlocklist}, want: "content_filter"},
		{name: "spii", resp: &model.LLMResponse{FinishReason: genai.FinishReasonSPII}, want: "content_filter"},
		{name: "malformed function call", resp: &model.LLMResponse{FinishReason: genai.FinishReasonMalformedFunctionCall}, want: "error"},
		{name: "too many tool calls", resp: &model.LLMResponse{FinishReason: genai.FinishReasonTooManyToolCalls}, want: "error"},
		{name: "other", resp: &model.LLMResponse{FinishReason: genai.FinishReasonOther}, want: "error"},
		{
			name: "error code beats a successful finish reason",
			resp: &model.LLMResponse{FinishReason: genai.FinishReasonStop, ErrorCode: "429"},
			want: "error",
		},
		{
			name: "interruption beats a successful finish reason",
			resp: &model.LLMResponse{FinishReason: genai.FinishReasonStop, Interrupted: true},
			want: "error",
		},
		{
			name: "error code beats a tool call",
			resp: &model.LLMResponse{
				Content:   &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{}}}},
				ErrorCode: "500",
			},
			want: "error",
		},
		{
			name: "error beats a successful finish reason",
			resp: &model.LLMResponse{FinishReason: genai.FinishReasonStop},
			err:  errTest,
			want: "error",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := schemaFinishReason(tc.resp, tc.err); got != tc.want {
				t.Errorf("schemaFinishReason() = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestSchemaFinishReason_EveryGenaiValue checks that no known genai finish
// reason ever reaches telemetry as a protobuf wire name, and that the result
// is always a value the schema defines.
func TestSchemaFinishReason_EveryGenaiValue(t *testing.T) {
	all := []genai.FinishReason{
		genai.FinishReasonUnspecified, genai.FinishReasonStop, genai.FinishReasonMaxTokens,
		genai.FinishReasonSafety, genai.FinishReasonRecitation, genai.FinishReasonLanguage,
		genai.FinishReasonOther, genai.FinishReasonBlocklist, genai.FinishReasonProhibitedContent,
		genai.FinishReasonSPII, genai.FinishReasonMalformedFunctionCall, genai.FinishReasonImageSafety,
		genai.FinishReasonUnexpectedToolCall, genai.FinishReasonTooManyToolCalls,
		genai.FinishReasonImageProhibitedContent, genai.FinishReasonNoImage,
		genai.FinishReasonImageRecitation, genai.FinishReasonImageOther,
	}
	for _, fr := range all {
		got := schemaFinishReason(&model.LLMResponse{FinishReason: fr}, nil)
		if !schemaFinishReasons[got] {
			t.Errorf("genai %s maps to %q, which the schema does not define", fr, got)
		}
	}
	// An unknown future value is lowercased rather than passed through.
	if got := schemaFinishReason(&model.LLMResponse{FinishReason: "BRAND_NEW_REASON"}, nil); got != "brand_new_reason" {
		t.Errorf("unknown finish reason = %q, want brand_new_reason", got)
	}
}

// --- End to end through the span ---------------------------------------

// TestGenerateContentSpan_ContentAttributes drives the real span helpers and
// asserts the exact JSON that lands on the span.
func TestGenerateContentSpan_ContentAttributes(t *testing.T) {
	captureContent(t)
	exporter := setupTestTracer(t)

	req := &model.LLMRequest{
		Contents: []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "hi"}}}},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{Parts: []*genai.Part{{Text: "be brief"}}},
		},
	}
	_, span := startGenerateContentSpan(t.Context(), GenerateContentParams{
		ModelName: "test-model", InvocationID: "inv-1", Request: req,
	})
	traceGenerateContentResult(span, generateContentResult{
		Response: &model.LLMResponse{
			Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "ok"}}},
			FinishReason: genai.FinishReasonStop,
		},
	})
	span.End()

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	want := map[attribute.Key]string{
		genAISystemInstructions: `[{"type":"text","content":"be brief"}]`,
		genAIInputMessages:      `[{"role":"user","parts":[{"type":"text","content":"hi"}]}]`,
		genAIOutputMessages:     `[{"role":"assistant","parts":[{"type":"text","content":"ok"}],"finish_reason":"stop"}]`,
	}
	for k, v := range want {
		got, ok := attrJSON(spans[0].Attributes, k)
		if !ok {
			t.Errorf("attribute %s was not set", k)
			continue
		}
		if diff := jsonDiff(t, v, got); diff != "" {
			t.Errorf("attribute %s mismatch (-want +got):\n%s", k, diff)
		}
	}
}

// TestGenerateContentSpan_ErrorStillCarriesThePrompt checks that the input is
// on the span even when the call fails, which is when it is most wanted.
func TestGenerateContentSpan_ErrorStillCarriesThePrompt(t *testing.T) {
	captureContent(t)
	exporter := setupTestTracer(t)

	req := &model.LLMRequest{
		Contents: []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "hi"}}}},
	}
	_, span := startGenerateContentSpan(t.Context(), GenerateContentParams{
		ModelName: "test-model", Request: req,
	})
	traceGenerateContentResult(span, generateContentResult{Error: errTest})
	span.End()

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	got := attributesToMap(spans[0].Attributes)
	if _, ok := got[genAIInputMessages]; !ok {
		t.Error("gen_ai.input.messages is missing from a failed generate_content span")
	}
	if _, ok := got[genAIOutputMessages]; ok {
		t.Error("gen_ai.output.messages was set for a call that produced no response")
	}
}

func TestGenerateContentSpan_NilRequest(t *testing.T) {
	captureContent(t)
	exporter := setupTestTracer(t)

	_, span := startGenerateContentSpan(t.Context(), GenerateContentParams{ModelName: "test-model"})
	span.End()

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	got := attributesToMap(spans[0].Attributes)
	if _, ok := got[genAIInputMessages]; ok {
		t.Error("gen_ai.input.messages was set for a nil request")
	}
}

// startAttrRecorder captures the attributes a span carries at creation, before
// anything is added to it afterwards.
type startAttrRecorder struct {
	sdktrace.SpanProcessor
	atStart []attribute.KeyValue
}

func (r *startAttrRecorder) OnStart(_ context.Context, s sdktrace.ReadWriteSpan) {
	r.atStart = s.Attributes()
}
func (r *startAttrRecorder) OnEnd(sdktrace.ReadOnlySpan)      {}
func (r *startAttrRecorder) Shutdown(context.Context) error   { return nil }
func (r *startAttrRecorder) ForceFlush(context.Context) error { return nil }

// TestStartGenerateContentSpan_ConvertsAfterTheSamplingDecision pins the order
// of two things: the sampler runs first, the conversation is converted second.
// Converting a whole conversation costs milliseconds and buys nothing on a span
// the sampler is about to drop, and the conventions do not list content among
// the attributes wanted at creation time for sampling.
//
// Observed through a span processor, since what a span carries at OnStart is
// exactly what was passed to tracer.Start.
func TestStartGenerateContentSpan_ConvertsAfterTheSamplingDecision(t *testing.T) {
	captureContent(t)
	rec := &startAttrRecorder{}
	OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec)))

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("a question worth capturing", genai.RoleUser)},
	}
	_, span := startGenerateContentSpan(t.Context(), GenerateContentParams{
		ModelName: "mock", InvocationID: "inv-1", Request: req,
	})
	span.End()

	if _, ok := attrJSON(rec.atStart, genAIInputMessages); ok {
		t.Error("content was converted before the sampler could decide")
	}
	// Sanity: the span-creation attributes are there, so the recorder works.
	if _, ok := attrJSON(rec.atStart, semconv.GenAIRequestModelKey); !ok {
		t.Fatalf("recorder saw no creation attributes at all: %v", rec.atStart)
	}
}

// TestStartExecuteToolSpan_ArgumentsAfterTheSamplingDecision is the same order
// for the tool arguments.
func TestStartExecuteToolSpan_ArgumentsAfterTheSamplingDecision(t *testing.T) {
	captureContent(t)
	t.Setenv(adkTelemetrySchemaVersionOptIn, "otel_semconv_1_44")
	rec := &startAttrRecorder{}
	OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec)))

	_, span := StartExecuteToolSpan(t.Context(), StartExecuteToolSpanParams{ToolName: "t", Args: map[string]any{"a": 1}})
	span.End()

	if _, ok := attrJSON(rec.atStart, genAIToolCallArguments); ok {
		t.Error("tool arguments were converted before the sampler could decide")
	}
	if _, ok := attrJSON(rec.atStart, semconv.GenAIToolNameKey); !ok {
		t.Fatalf("recorder saw no creation attributes at all: %v", rec.atStart)
	}
}

// TestOversizedAttributeIsDropped covers the span size guard, under both
// schemas. A request carries the whole conversation and is rebuilt on every
// model call, so the attribute grows with the session. telemetry.googleapis.com
// rejects the whole export request carrying one over its limit, so this drops
// the attribute rather than lose the batch.
func TestOversizedAttributeIsDropped(t *testing.T) {
	captureContent(t)
	var contents []*genai.Content
	for range 200 {
		contents = append(contents, &genai.Content{
			Role:  genai.RoleUser,
			Parts: []*genai.Part{{Text: strings.Repeat("x", 1000)}},
		})
	}
	for _, schema := range []string{otelSemconv136, "otel_semconv_1_44"} {
		t.Run(schema, func(t *testing.T) {
			t.Setenv(adkTelemetrySchemaVersionOptIn, schema)

			attrs := spanContentAttributes(requestContent(&model.LLMRequest{Contents: contents}))
			if got, ok := attrJSON(attrs, genAIInputMessages); ok {
				t.Errorf("an attribute of %d bytes was recorded, want it dropped", len(got))
			}

			// An ordinary conversation is unaffected, and comfortably inside
			// the limit the implementation exists to respect.
			attrs = spanContentAttributes(requestContent(&model.LLMRequest{
				Contents: []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "hello"}}}},
			}))
			got, ok := attrJSON(attrs, genAIInputMessages)
			if !ok {
				t.Fatal("an ordinary conversation was dropped")
			}
			if len(got) > cloudTraceAttributeValueLimit {
				t.Errorf("attribute is %d bytes, over the %d the backend accepts", len(got), cloudTraceAttributeValueLimit)
			}
		})
	}
}

// TestMediaParts covers inline data and file references, which the reporter of
// #608 was getting in v0.4.0 and would otherwise lose.
func TestMediaParts(t *testing.T) {
	captureContent(t)

	small := []byte("pretend this is a png")
	req := &model.LLMRequest{Contents: []*genai.Content{{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: small}},
			{FileData: &genai.FileData{MIMEType: "VIDEO/MP4", FileURI: "gs://bucket/clip.mp4"}},
		},
	}}}

	got, ok := attrJSON(spanContentAttributes(requestContent(req)), genAIInputMessages)
	if !ok {
		t.Fatal("gen_ai.input.messages was not set")
	}
	want := `[{"role":"user","parts":[` +
		`{"type":"blob","mime_type":"image/png","modality":"image","content":"` +
		base64.StdEncoding.EncodeToString(small) + `"},` +
		`{"type":"uri","mime_type":"VIDEO/MP4","modality":"video","uri":"gs://bucket/clip.mp4"}]}]`
	if diff := jsonDiff(t, want, got); diff != "" {
		t.Errorf("gen_ai.input.messages mismatch (-want +got):\n%s", diff)
	}
}

// TestInlineDataIsCappedPerPayload keeps one image from consuming the whole
// attribute and leaving the conversation around it unrecorded.
func TestInlineDataIsCappedPerPayload(t *testing.T) {
	captureContent(t)

	big := make([]byte, maxInlineDataBytes*4)
	req := &model.LLMRequest{Contents: []*genai.Content{{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			{InlineData: &genai.Blob{MIMEType: "audio/wav", Data: big}},
			{Text: "what is in this recording?"},
		},
	}}}

	got, ok := attrJSON(spanContentAttributes(requestContent(req)), genAIInputMessages)
	if !ok {
		t.Fatal("a turn with one large payload dropped the whole attribute")
	}
	if full := base64.StdEncoding.EncodeToString(big); strings.Contains(got, full) {
		t.Error("the whole payload was recorded")
	}
	if !strings.Contains(got, `"truncated":true`) {
		t.Errorf("truncation not marked: %.200s", got)
	}
	// The text alongside it still has to survive; that is the point of the cap.
	if !strings.Contains(got, "what is in this recording?") {
		t.Errorf("the rest of the turn was lost: %.200s", got)
	}
}

// TestModalityIsCaseInsensitive covers RFC 2045 section 5.1 — a MIME type has
// no canonical case, and modality is required on blob and uri parts.
func TestModalityIsCaseInsensitive(t *testing.T) {
	for _, mt := range []string{"image/png", "IMAGE/PNG", "Image/Png"} {
		if got := modalityOf(mt); got != "image" {
			t.Errorf("modalityOf(%q) = %q, want image", mt, got)
		}
	}
	if got := modalityOf("application/pdf"); got != "document" {
		t.Errorf("modalityOf(pdf) = %q, want document", got)
	}
}

// TestContentCaptureModes pins which signals each value of the environment
// variable turns on.
//
// The back-compatible case is the one that matters: this variable was a boolean
// before spans could carry content, so a truthy value has to keep meaning log
// records only. Logs and traces go to different backends with different
// readers, and an upgrade must not move prompts into a new one on a
// configuration nobody changed. adk-python reads a truthy value the same way.
func TestContentCaptureModes(t *testing.T) {
	for _, tc := range []struct {
		env    string
		inLogs bool
		onSpan bool
	}{
		{"", false, false},
		{"false", false, false},
		{"0", false, false},
		{"nonsense", false, false},
		{"true", true, false},
		{"1", true, false},
		{" TRUE ", true, false},
		{"EVENT_ONLY", true, false},
		{"SPAN_ONLY", false, true},
		{"SPAN_AND_EVENT", true, true},
		{"span_and_event", true, true},
	} {
		t.Run(tc.env, func(t *testing.T) {
			t.Setenv(captureMessageContentEnvVar, tc.env)
			ApplyEnv()
			if got := getGenAICaptureMessageContent(); got != tc.inLogs {
				t.Errorf("content in log records = %v, want %v", got, tc.inLogs)
			}
			if got := captureContentOnSpans(); got != tc.onSpan {
				t.Errorf("content on spans = %v, want %v", got, tc.onSpan)
			}
		})
	}
}

// TestTruthyValueDoesNotPutContentOnSpans is the regression this whole mode
// parse exists for, stated on its own so it cannot be lost in a table: every
// agent deployed through the CLI has this variable set to "true", and merging
// span capture behind that value would have started shipping full conversations
// to a tracing backend nobody opted into.
func TestTruthyValueDoesNotPutContentOnSpans(t *testing.T) {
	setEnvForTesting(t, map[string]string{captureMessageContentEnvVar: "true", adkTelemetrySchemaVersionOptIn: "otel_semconv_1_44"})
	exporter := setupTestTracer(t)

	req := &model.LLMRequest{
		Contents: []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "canary"}}}},
	}
	_, span := startGenerateContentSpan(t.Context(), GenerateContentParams{ModelName: "m", Request: req})
	traceGenerateContentResult(span, generateContentResult{
		Response: &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "canary"}}}},
	})
	span.End()
	_, span = StartExecuteToolSpan(t.Context(), StartExecuteToolSpanParams{ToolName: "t", Args: map[string]any{"a": "canary"}})
	TraceToolResult(span, TraceToolResultParams{ResponseEvent: &session.Event{LLMResponse: model.LLMResponse{
		Content: &genai.Content{Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Response: map[string]any{"r": "canary"}}}}},
	}}})
	span.End()

	for _, s := range exporter.GetSpans() {
		for _, attr := range s.Attributes {
			if strings.Contains(attr.Value.Emit(), "canary") {
				t.Errorf("a truthy value put %s on span %q", attr.Key, s.Name)
			}
		}
	}
}

// TestEventContentAttributes pins the two ways event content differs from a
// legacy span's JSON string: it is not held to the Cloud Trace size cap, and
// tool argument numbers keep their type.
func TestEventContentAttributes(t *testing.T) {
	long := strings.Repeat("x", cloudTraceAttributeValueLimit)
	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: genai.RoleUser, Parts: []*genai.Part{{Text: long}}},
		{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "f", Args: map[string]any{"n": 3, "f": 1.5}}}}},
	}}

	attrs := eventContentAttributes(requestContent(req))

	if len(attrs) != 1 || attrs[0].Key != genAIInputMessages {
		t.Fatalf("attributes = %v, want only %s", attrs, genAIInputMessages)
	}
	msgs := attrs[0].Value.AsSlice()
	text := FromLogValue(msgs[0]).(map[string]any)["parts"].([]any)[0].(map[string]any)["content"]
	if text != long {
		t.Errorf("the long message was not recorded in full")
	}
	args := FromLogValue(msgs[1]).(map[string]any)["parts"].([]any)[0].(map[string]any)["arguments"]
	if diff := cmp.Diff(map[string]any{"n": int64(3), "f": 1.5}, args); diff != "" {
		t.Errorf("tool arguments mismatch (-want +got):\n%s", diff)
	}
}
