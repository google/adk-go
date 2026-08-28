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
	"math"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
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
	t.Setenv(captureMessageContentEnvVar, "true")
	ApplyEnv()
}

// attrString returns the value of the named attribute, and whether it was set.
func attrString(attrs []attribute.KeyValue, key attribute.Key) (string, bool) {
	for _, kv := range attrs {
		if kv.Key == key {
			return kv.Value.AsString(), true
		}
	}
	return "", false
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

// TestRequestContentAttributes_AllPartVariants feeds one of every
// genai.Part variant through the converter and checks the result against the
// schema's per-type required fields.
func TestRequestContentAttributes_AllPartVariants(t *testing.T) {
	captureContent(t)

	req := &model.LLMRequest{
		Contents: []*genai.Content{{
			Role: genai.RoleModel,
			Parts: []*genai.Part{
				{Text: "plain text"},
				{Text: "let me think", Thought: true},
				{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "get_weather", Args: map[string]any{"city": "Paris"}}},
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1, 2, 3}}},
				{FileData: &genai.FileData{MIMEType: "video/mp4", FileURI: "gs://bucket/clip.mp4"}},
				{ExecutableCode: &genai.ExecutableCode{ID: "ec-1", Code: "print(1)", Language: genai.LanguagePython}},
				{CodeExecutionResult: &genai.CodeExecutionResult{ID: "ec-1", Outcome: genai.OutcomeOK, Output: "1"}},
				{ToolCall: &genai.ToolCall{ID: "st-1", ToolType: genai.ToolTypeGoogleSearchWeb, Args: map[string]any{"q": "rain"}}},
				{AudioTranscription: &genai.Transcription{Text: "hello there", LanguageCode: "en-US"}},
			},
		}},
	}

	attrs := requestContentAttributes(req)
	encoded, ok := attrString(attrs, genAIInputMessages)
	if !ok {
		t.Fatalf("gen_ai.input.messages was not set; got %v", attrs)
	}
	msgs := validateMessages(t, encoded, false)
	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1", len(msgs))
	}

	var gotTypes []string
	for _, p := range msgs[0]["parts"].([]any) {
		gotTypes = append(gotTypes, p.(map[string]any)["type"].(string))
	}
	wantTypes := []string{
		"text", "reasoning", "tool_call", "blob", "uri",
		"server_tool_call", "server_tool_call_response", "server_tool_call",
		"audio_transcription",
	}
	if diff := cmp.Diff(wantTypes, gotTypes); diff != "" {
		t.Errorf("part types (-want +got):\n%s", diff)
	}

	// Spot-check the values that the schema and the genai types disagree on.
	parts := msgs[0]["parts"].([]any)
	blob := parts[3].(map[string]any)
	if got, want := blob["content"], base64.StdEncoding.EncodeToString([]byte{1, 2, 3}); got != want {
		t.Errorf("blob content = %v, want %v", got, want)
	}
	if got := blob["modality"]; got != "image" {
		t.Errorf("blob modality = %v, want image", got)
	}
	if got := parts[4].(map[string]any)["modality"]; got != "video" {
		t.Errorf("uri modality = %v, want video", got)
	}
	code := parts[5].(map[string]any)["server_tool_call"].(map[string]any)
	if got := code["language"]; got != "python" {
		t.Errorf("code language = %v, want python", got)
	}
	result := parts[6].(map[string]any)["server_tool_call_response"].(map[string]any)
	if got := result["outcome"]; got != "ok" {
		t.Errorf("code outcome = %v, want ok", got)
	}
	if got := parts[7].(map[string]any)["name"]; got != "google_search_web" {
		t.Errorf("server tool name = %v, want google_search_web", got)
	}
}

// TestRequestContentAttributes_TextPartShape pins the exact JSON of the
// simplest case so a change of field names or nesting is visible.
func TestRequestContentAttributes_TextPartShape(t *testing.T) {
	captureContent(t)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "Weather in Paris?"}}},
			{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "Rainy."}}},
		},
	}
	got, ok := attrString(requestContentAttributes(req), genAIInputMessages)
	if !ok {
		t.Fatal("gen_ai.input.messages was not set")
	}
	want := `[{"role":"user","parts":[{"type":"text","content":"Weather in Paris?"}]},` +
		`{"role":"assistant","parts":[{"type":"text","content":"Rainy."}]}]`
	if got != want {
		t.Errorf("gen_ai.input.messages =\n%s\nwant\n%s", got, want)
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
	encoded, ok := attrString(requestContentAttributes(req), genAIInputMessages)
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
	attrs := requestContentAttributes(req)
	got, ok := attrString(attrs, genAISystemInstructions)
	if !ok {
		t.Fatalf("gen_ai.system_instructions was not set; got %v", attrs)
	}
	validateParts(t, got)
	want := `[{"type":"text","content":"You are a translator."},` +
		`{"type":"text","content":"Translate English to French."}]`
	if got != want {
		t.Errorf("gen_ai.system_instructions =\n%s\nwant\n%s", got, want)
	}
	if _, present := attrString(attrs, genAIInputMessages); present {
		t.Error("gen_ai.input.messages must not be set when there are no contents")
	}
}

// --- Opt-in ------------------------------------------------------------

func TestContentAttributes_OptInIsOffByDefault(t *testing.T) {
	t.Setenv(captureMessageContentEnvVar, "")
	ApplyEnv()

	req := &model.LLMRequest{
		Contents: []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "secret prompt"}}}},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{Parts: []*genai.Part{{Text: "secret instruction"}}},
		},
	}
	if attrs := requestContentAttributes(req); len(attrs) != 0 {
		t.Errorf("request attributes with capture off = %v, want none", attrs)
	}
	resp := &model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "secret answer"}}}}
	if attrs := responseContentAttributes(resp); len(attrs) != 0 {
		t.Errorf("response attributes with capture off = %v, want none", attrs)
	}
}

// --- Output messages ---------------------------------------------------

func TestResponseContentAttributes(t *testing.T) {
	captureContent(t)

	resp := &model.LLMResponse{
		Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "It is rainy."}}},
		FinishReason: genai.FinishReasonStop,
	}
	got, ok := attrString(responseContentAttributes(resp), genAIOutputMessages)
	if !ok {
		t.Fatal("gen_ai.output.messages was not set")
	}
	validateMessages(t, got, true)
	want := `[{"role":"assistant","parts":[{"type":"text","content":"It is rainy."}],"finish_reason":"stop"}]`
	if got != want {
		t.Errorf("gen_ai.output.messages =\n%s\nwant\n%s", got, want)
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
	if attrs := responseContentAttributes(partial); len(attrs) != 0 {
		t.Errorf("attributes for a partial chunk = %v, want none", attrs)
	}

	settled := &model.LLMResponse{
		Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "It is rainy."}}},
		FinishReason: genai.FinishReasonStop,
	}
	got, ok := attrString(responseContentAttributes(settled), genAIOutputMessages)
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
	got, ok := attrString(responseContentAttributes(resp), genAIOutputMessages)
	if !ok {
		t.Fatal("gen_ai.output.messages was not set")
	}
	validateMessages(t, got, true)
	want := `[{"role":"assistant","parts":[],"finish_reason":"content_filter"}]`
	if got != want {
		t.Errorf("gen_ai.output.messages =\n%s\nwant\n%s", got, want)
	}
}

func TestSchemaFinishReason(t *testing.T) {
	tests := []struct {
		name     string
		resp     *model.LLMResponse
		toolCall bool
		want     string
	}{
		{"stop", &model.LLMResponse{FinishReason: genai.FinishReasonStop}, false, "stop"},
		{"stop with a tool call", &model.LLMResponse{FinishReason: genai.FinishReasonStop}, true, "tool_call"},
		{"unset", &model.LLMResponse{}, false, "stop"},
		{"unset with a tool call", &model.LLMResponse{}, true, "tool_call"},
		{"unspecified", &model.LLMResponse{FinishReason: genai.FinishReasonUnspecified}, false, "stop"},
		{"max tokens", &model.LLMResponse{FinishReason: genai.FinishReasonMaxTokens}, false, "length"},
		{"safety", &model.LLMResponse{FinishReason: genai.FinishReasonSafety}, false, "content_filter"},
		{"recitation", &model.LLMResponse{FinishReason: genai.FinishReasonRecitation}, false, "content_filter"},
		{"blocklist", &model.LLMResponse{FinishReason: genai.FinishReasonBlocklist}, false, "content_filter"},
		{"spii", &model.LLMResponse{FinishReason: genai.FinishReasonSPII}, false, "content_filter"},
		{"malformed function call", &model.LLMResponse{FinishReason: genai.FinishReasonMalformedFunctionCall}, false, "error"},
		{"too many tool calls", &model.LLMResponse{FinishReason: genai.FinishReasonTooManyToolCalls}, false, "error"},
		{"other", &model.LLMResponse{FinishReason: genai.FinishReasonOther}, false, "error"},
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
			name:     "error code beats a tool call",
			resp:     &model.LLMResponse{ErrorCode: "500"},
			toolCall: true,
			want:     "error",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := schemaFinishReason(tc.resp, tc.toolCall); got != tc.want {
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
		got := schemaFinishReason(&model.LLMResponse{FinishReason: fr}, false)
		if !schemaFinishReasons[got] {
			t.Errorf("genai %s maps to %q, which the schema does not define", fr, got)
		}
	}
	// An unknown future value is lowercased rather than passed through.
	if got := schemaFinishReason(&model.LLMResponse{FinishReason: "BRAND_NEW_REASON"}, false); got != "brand_new_reason" {
		t.Errorf("unknown finish reason = %q, want brand_new_reason", got)
	}
}

// --- Hostile tool payloads ---------------------------------------------

// TestValue_UnmarshalableToolPayloads covers arguments and responses filled
// in by application code, which encoding/json cannot always serialize. The
// attribute must stay valid JSON matching the schema.
func TestValue_UnmarshalableToolPayloads(t *testing.T) {
	captureContent(t)

	cyclic := map[string]any{}
	cyclic["self"] = cyclic
	cyclic["also"] = cyclic

	tests := []struct {
		name string
		args map[string]any
	}{
		{"NaN", map[string]any{"v": math.NaN()}},
		{"positive infinity", map[string]any{"v": math.Inf(1)}},
		{"function", map[string]any{"v": func() {}}},
		{"channel", map[string]any{"v": make(chan int)}},
		{"reference cycle", cyclic},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := &model.LLMRequest{Contents: []*genai.Content{{
				Role:  genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "t", Args: tc.args}}},
			}}}

			// A self-referential map fans out exponentially under a naive
			// recursive walk, so bound the work in wall-clock time as well
			// as asserting the result.
			done := make(chan []attribute.KeyValue, 1)
			go func() { done <- requestContentAttributes(req) }()
			var attrs []attribute.KeyValue
			select {
			case attrs = <-done:
			case <-time.After(30 * time.Second):
				t.Fatal("converting the tool arguments did not finish within 30s")
			}

			encoded, ok := attrString(attrs, genAIInputMessages)
			if !ok {
				t.Fatalf("gen_ai.input.messages was not set; got %v", attrs)
			}
			msgs := validateMessages(t, encoded, false)
			part := msgs[0]["parts"].([]any)[0].(map[string]any)
			if got := part["arguments"]; got != "<not serializable>" {
				t.Errorf("arguments = %v, want the placeholder <not serializable>", got)
			}
			if got := part["truncated"]; got != true {
				t.Errorf("truncated = %v, want true", got)
			}
		})
	}
}

// --- Size bounds -------------------------------------------------------

// TestEncodedAttributesStayUnderTheCloudTraceLimit drives the sizes that
// actually occur: a long conversation history, text that inflates six-fold
// under JSON escaping, and a blob that inflates by a third under base64.
func TestEncodedAttributesStayUnderTheCloudTraceLimit(t *testing.T) {
	captureContent(t)

	tests := []struct {
		name string
		req  *model.LLMRequest
	}{
		{
			name: "long history",
			req: func() *model.LLMRequest {
				var contents []*genai.Content
				for i := range 500 {
					contents = append(contents, &genai.Content{
						Role:  genai.RoleUser,
						Parts: []*genai.Part{{Text: fmt.Sprintf("turn %d: %s", i, strings.Repeat("a", 2000))}},
					})
				}
				return &model.LLMRequest{Contents: contents}
			}(),
		},
		{
			name: "text that escaping inflates six-fold",
			req: &model.LLMRequest{Contents: []*genai.Content{{
				Role: genai.RoleUser,
				// 200 KiB raw, 1.2 MiB once every "<" becomes \u003c.
				Parts: []*genai.Part{{Text: strings.Repeat("<", 200*1024)}},
			}}},
		},
		{
			name: "one enormous text part",
			req: &model.LLMRequest{Contents: []*genai.Content{{
				Role:  genai.RoleUser,
				Parts: []*genai.Part{{Text: strings.Repeat("x", 5*1024*1024)}},
			}}},
		},
		{
			name: "many parts in one message",
			req: func() *model.LLMRequest {
				var parts []*genai.Part
				for i := range 20000 {
					parts = append(parts, &genai.Part{Text: fmt.Sprintf("part %d", i)})
				}
				return &model.LLMRequest{Contents: []*genai.Content{{Role: genai.RoleUser, Parts: parts}}}
			}(),
		},
		{
			name: "blob that base64 inflates",
			req: &model.LLMRequest{Contents: []*genai.Content{{
				Role: genai.RoleUser,
				Parts: []*genai.Part{{InlineData: &genai.Blob{
					MIMEType: "image/png",
					Data:     make([]byte, 3*1024*1024),
				}}},
			}}},
		},
		{
			name: "oversized tool arguments",
			req: &model.LLMRequest{Contents: []*genai.Content{{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
					Name: "t",
					Args: map[string]any{"blob": strings.Repeat("y", 2*1024*1024)},
				}}},
			}}},
		},
		{
			name: "enormous system instruction",
			req: &model.LLMRequest{Config: &genai.GenerateContentConfig{
				SystemInstruction: &genai.Content{Parts: []*genai.Part{
					{Text: strings.Repeat("z", 1024*1024)},
					{Text: strings.Repeat("w", 1024*1024)},
				}},
			}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			attrs := requestContentAttributes(tc.req)
			if len(attrs) == 0 {
				t.Fatal("no content attributes were recorded")
			}
			for _, kv := range attrs {
				v := kv.Value.AsString()
				if len(v) > cloudTraceAttributeValueLimit {
					t.Errorf("%s is %d bytes, over the %d byte Cloud Trace limit",
						kv.Key, len(v), cloudTraceAttributeValueLimit)
				}
				// Oversized content must still be the JSON array the schema
				// declares, not a marker string.
				switch kv.Key {
				case genAIInputMessages:
					validateMessages(t, v, false)
				case genAISystemInstructions:
					validateParts(t, v)
				}
			}
		})
	}
}

// TestEncodeMessages_KeepsTheNewestTurns checks which end of an oversized
// history survives, and that the loss is recorded rather than silent.
func TestEncodeMessages_KeepsTheNewestTurns(t *testing.T) {
	captureContent(t)

	var contents []*genai.Content
	for i := range 200 {
		contents = append(contents, &genai.Content{
			Role:  genai.RoleUser,
			Parts: []*genai.Part{{Text: fmt.Sprintf("turn-%03d %s", i, strings.Repeat("a", 1000))}},
		})
	}
	encoded, ok := attrString(requestContentAttributes(&model.LLMRequest{Contents: contents}), genAIInputMessages)
	if !ok {
		t.Fatal("gen_ai.input.messages was not set")
	}
	msgs := validateMessages(t, encoded, false)
	if len(msgs) >= 200 {
		t.Fatalf("got %d messages, expected the history to be trimmed", len(msgs))
	}
	if !strings.Contains(encoded, "turn-199") {
		t.Error("the newest turn was dropped; the trimming keeps the wrong end")
	}
	if strings.Contains(encoded, "turn-000") {
		t.Error("the oldest turn survived; the trimming keeps the wrong end")
	}
	// Messages that survive keep their order.
	firstText := msgs[0]["parts"].([]any)[0].(map[string]any)["content"].(string)
	lastText := msgs[len(msgs)-1]["parts"].([]any)[0].(map[string]any)["content"].(string)
	if firstText >= lastText {
		t.Errorf("messages are out of order: first %.11q, last %.11q", firstText, lastText)
	}
	dropped, ok := msgs[0]["dropped_preceding_messages"].(float64)
	if !ok {
		t.Fatalf("the oldest surviving message does not record the dropped turns: %v", msgs[0])
	}
	if want := float64(200 - len(msgs)); dropped != want {
		t.Errorf("dropped_preceding_messages = %v, want %v", dropped, want)
	}
}

// TestBase64_TruncatesBeforeEncoding checks that an oversized blob is not
// expanded in full just to be thrown away, and that what is kept decodes.
func TestBase64_TruncatesBeforeEncoding(t *testing.T) {
	b := newContentBuilder()
	data := make([]byte, 4*1024*1024)
	for i := range data {
		data[i] = byte(i)
	}
	got, cut := b.base64(data)
	if !cut {
		t.Fatal("an oversized blob was not marked as truncated")
	}
	if len(got) > maxPartContentBytes {
		t.Errorf("encoded blob is %d bytes, over the per-part budget of %d", len(got), maxPartContentBytes)
	}
	decoded, err := base64.StdEncoding.DecodeString(got)
	if err != nil {
		t.Fatalf("the truncated blob is not valid base64: %v", err)
	}
	if len(decoded) == 0 || !strings.HasPrefix(string(data), string(decoded)) {
		t.Error("the truncated blob is not a prefix of the original data")
	}

	small := []byte{9, 8, 7}
	if got, cut := b.base64(small); cut || got != base64.StdEncoding.EncodeToString(small) {
		t.Errorf("base64(small) = %q, %v; want the full encoding and no truncation", got, cut)
	}
}

// --- JSON escaping -----------------------------------------------------

// TestEscapedRuneLen_MatchesEncodingJSON pins the escaped-length arithmetic
// to what encoding/json actually produces. Truncation is measured in encoded
// bytes, so an error here would let an attribute exceed its bound.
func TestEscapedRuneLen_MatchesEncodingJSON(t *testing.T) {
	var corpus []string
	for b := range 256 {
		corpus = append(corpus, string([]byte{byte(b)}))
	}
	corpus = append(corpus,
		"", "plain", `"quoted"`, `back\slash`, "tab\there", "nl\nhere", "cr\rhere",
		"<html>&amp;", "\u2028\u2029", "héllo wörld", "日本語のテキスト", "emoji 😀 here",
		"\xff\xfe invalid", "mixed <\x00\xff> 日本",
	)
	for _, s := range corpus {
		want, err := json.Marshal(s)
		if err != nil {
			t.Fatalf("json.Marshal(%q): %v", s, err)
		}
		got := 0
		for i := 0; i < len(s); {
			r, size := utf8.DecodeRuneInString(s[i:])
			got += escapedRuneLen(r, size)
			i += size
		}
		// want includes the two enclosing quotes.
		if got != len(want)-2 {
			t.Errorf("escaped length of %q = %d, want %d (encoding/json produced %s)",
				s, got, len(want)-2, want)
		}
	}
}

func TestTruncateJSONString(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		limit   int
		want    string
		wantCut bool
	}{
		{"fits", "hello", 10, "hello", false},
		{"exact fit", "hello", 5, "hello", false},
		{"plain cut", "hello", 4, "hell", true},
		{"escaped characters cost six bytes each", "<<<<<", 12, "<<", true},
		{"never splits a rune", "日本語", 7, "日本", true},
		{"a rune that does not fit at all", "日本語", 2, "", true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, cut := truncateJSONString(tc.in, tc.limit)
			if got != tc.want || cut != tc.wantCut {
				t.Errorf("truncateJSONString(%q, %d) = %q, %v; want %q, %v",
					tc.in, tc.limit, got, cut, tc.want, tc.wantCut)
			}
			if !utf8.ValidString(got) && utf8.ValidString(tc.in) {
				t.Errorf("truncation split a rune: %q", got)
			}
			encoded, err := json.Marshal(got)
			if err != nil {
				t.Fatalf("json.Marshal(%q): %v", got, err)
			}
			if len(encoded)-2 > tc.limit {
				t.Errorf("%q encodes to %d bytes, over the limit of %d", got, len(encoded)-2, tc.limit)
			}
		})
	}
}

// --- Span integration --------------------------------------------------

// TestGenerateContentSpan_ContentAttributes exercises the attributes through
// the exported span helpers, the way base_flow uses them.
func TestGenerateContentSpan_ContentAttributes(t *testing.T) {
	captureContent(t)
	exporter := setupTestTracer(t)

	req := &model.LLMRequest{
		Contents: []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "hi"}}}},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{Parts: []*genai.Part{{Text: "be brief"}}},
		},
	}
	_, span := StartGenerateContentSpan(t.Context(), StartGenerateContentSpanParams{
		ModelName: "test-model", InvocationID: "inv-1", Request: req,
	})
	TraceGenerateContentResult(span, TraceGenerateContentResultParams{
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
	got := attributesToMap(spans[0].Attributes)
	for k, v := range want {
		if got[k] != v {
			t.Errorf("attribute %s =\n%s\nwant\n%s", k, got[k], v)
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
	_, span := StartGenerateContentSpan(t.Context(), StartGenerateContentSpanParams{
		ModelName: "test-model", Request: req,
	})
	TraceGenerateContentResult(span, TraceGenerateContentResultParams{Error: errTest})
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

	_, span := StartGenerateContentSpan(t.Context(), StartGenerateContentSpanParams{ModelName: "test-model"})
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

// TestRequestContentAttributes_TruncationFlagStaysWithItsOwner covers a flag
// that belongs to the system instruction leaking onto the first message. The
// two are converted by the same builder, and gen_ai.system_instructions is a
// bare part list with nowhere to record truncation, so the flag has to be
// cleared rather than left for whatever is built next.
func TestRequestContentAttributes_TruncationFlagStaysWithItsOwner(t *testing.T) {
	captureContent(t)

	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText(strings.Repeat("s", 200_000), genai.RoleUser),
		},
		Contents: []*genai.Content{
			genai.NewContentFromText("hi", genai.RoleUser),
		},
	}

	got, ok := attrString(requestContentAttributes(req), genAIInputMessages)
	if !ok {
		t.Fatal("gen_ai.input.messages was not set")
	}
	var msgs []map[string]any
	if err := json.Unmarshal([]byte(got), &msgs); err != nil {
		t.Fatalf("not a JSON array of messages: %v\n%s", err, got)
	}
	if len(msgs) != 1 {
		t.Fatalf("want one message, got %d: %s", len(msgs), got)
	}
	if _, flagged := msgs[0]["truncated"]; flagged {
		t.Errorf("a two-character prompt is reported as truncated: %s", got)
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
	_, span := StartGenerateContentSpan(t.Context(), StartGenerateContentSpanParams{
		ModelName: "mock", InvocationID: "inv-1", Request: req,
	})
	span.End()

	if _, ok := attrString(rec.atStart, genAIInputMessages); ok {
		t.Error("content was converted before the sampler could decide")
	}
	// Sanity: the span-creation attributes are there, so the recorder works.
	if _, ok := attrString(rec.atStart, semconv.GenAIRequestModelKey); !ok {
		t.Fatalf("recorder saw no creation attributes at all: %v", rec.atStart)
	}
}
