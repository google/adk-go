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
	"go.opentelemetry.io/otel/log"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	tracenoop "go.opentelemetry.io/otel/trace/noop"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// countingMarshaler counts its encodings. Converting content that holds it as
// a tool payload encodes it.
type countingMarshaler struct{ n *int }

func (m countingMarshaler) MarshalJSON() ([]byte, error) {
	*m.n++
	return []byte(`"x"`), nil
}

// enabledRecorder is a log processor that takes the event names in enable,
// and records what it was asked and given.
type enabledRecorder struct {
	enable  map[string]bool
	asked   []question
	emitted []string
}

type question struct {
	EventName string
	Severity  log.Severity
}

func (p *enabledRecorder) Enabled(_ context.Context, param sdklog.EnabledParameters) bool {
	p.asked = append(p.asked, question{param.EventName, param.Severity})
	return p.enable[param.EventName]
}

func (p *enabledRecorder) OnEmit(_ context.Context, r *sdklog.Record) error {
	p.emitted = append(p.emitted, r.EventName())
	return nil
}

func (p *enabledRecorder) Shutdown(context.Context) error   { return nil }
func (p *enabledRecorder) ForceFlush(context.Context) error { return nil }

// driveTelemetry runs a model call, a tool call and a merged tool call whose
// request, response and tool result each carry m.
func driveTelemetry(t *testing.T, m countingMarshaler) {
	payload := map[string]any{"v": m}
	call := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "f", Args: payload}}}}
	req := &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser), call}}
	resp := &model.LLMResponse{Content: call, FinishReason: genai.FinishReasonStop}
	params := GenerateContentParams{ModelName: "m", Request: req}
	gen := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
		return func(yield func(*model.LLMResponse, error) bool) { yield(resp, nil) }
	}
	for _, err := range InstrumentGenerateContent(t.Context(), params, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "" }, gen) {
		if err != nil {
			t.Fatal(err)
		}
	}
	ev := &session.Event{ID: "e", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{FunctionResponse: &genai.FunctionResponse{ID: "c", Name: "f", Response: payload}},
	}}}}
	_, span := StartExecuteToolSpan(t.Context(), StartExecuteToolSpanParams{ToolName: "f"})
	TraceToolResult(span, TraceToolResultParams{ResponseEvent: ev})
	span.End()
	_, merged := StartTrace(t.Context(), "execute_tool (merged)")
	TraceMergedToolCallsResult(merged, ev, nil)
	merged.End()
}

// TestContentConvertedOnlyForAListener checks that with content capture on, no
// content is converted unless a logger takes the records or a span records.
func TestContentConvertedOnlyForAListener(t *testing.T) {
	allEvents := map[string]bool{inferenceOperationDetailsEventName: true, systemMessageEventName: true, userMessageEventName: true, choiceEventName: true}
	for _, schema := range []string{otelSemconv136, "otel_semconv_1_44"} {
		for _, tc := range []struct {
			name      string
			tracer    trace.TracerProvider
			logger    log.LoggerProvider
			converted bool
		}{
			{"no-op providers", tracenoop.NewTracerProvider(), defaultLoggerProvider, false},
			{"sampled out, logger not enabled", sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.NeverSample())), sdklog.NewLoggerProvider(sdklog.WithProcessor(&enabledRecorder{})), false},
			{"recording span", sdktrace.NewTracerProvider(), defaultLoggerProvider, true},
			{"enabled logger", tracenoop.NewTracerProvider(), sdklog.NewLoggerProvider(sdklog.WithProcessor(&enabledRecorder{enable: allEvents})), true},
		} {
			t.Run(schema+"/"+tc.name, func(t *testing.T) {
				setEnvForTesting(t, map[string]string{adkTelemetrySchemaVersionOptIn: schema, captureMessageContentEnvVar: "SPAN_AND_EVENT"})
				OverrideTracerForTesting(t, tc.tracer)
				OverrideLoggerForTesting(t, tc.logger)
				var n int
				driveTelemetry(t, countingMarshaler{&n})
				if got := n > 0; got != tc.converted {
					t.Errorf("content encoded %d times, want converted = %t", n, tc.converted)
				}
			})
		}
	}
}

// TestLoggerAskedForEachEvent checks that a logger installed but not enabled
// for an event gets no record of it, and that the inference event is asked
// about at the severity it is emitted with.
func TestLoggerAskedForEachEvent(t *testing.T) {
	inference := []question{{inferenceOperationDetailsEventName, log.SeverityDebug}}
	legacy := []question{{EventName: systemMessageEventName}, {EventName: userMessageEventName}, {EventName: choiceEventName}}
	for _, tc := range []struct {
		name   string
		schema string
		enable map[string]bool
		asked  []question
		want   []string
	}{
		{"experimental, not enabled", "otel_semconv_1_44", nil, inference, nil},
		{"experimental, enabled", "otel_semconv_1_44", map[string]bool{inferenceOperationDetailsEventName: true}, inference, []string{inferenceOperationDetailsEventName}},
		{"legacy, not enabled", otelSemconv136, nil, legacy, nil},
		{"legacy, user messages only", otelSemconv136, map[string]bool{userMessageEventName: true}, legacy, []string{userMessageEventName, userMessageEventName}},
		{"legacy, system and choice", otelSemconv136, map[string]bool{systemMessageEventName: true, choiceEventName: true}, legacy, []string{systemMessageEventName, choiceEventName}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			setEnvForTesting(t, map[string]string{adkTelemetrySchemaVersionOptIn: tc.schema, captureMessageContentEnvVar: "true"})
			OverrideTracerForTesting(t, tracenoop.NewTracerProvider())
			p := &enabledRecorder{enable: tc.enable}
			OverrideLoggerForTesting(t, sdklog.NewLoggerProvider(sdklog.WithProcessor(p)))
			req := &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("a", genai.RoleUser), genai.NewContentFromText("b", genai.RoleModel)}}
			gen := func(context.Context) iter.Seq2[*model.LLMResponse, error] {
				return func(yield func(*model.LLMResponse, error) bool) {
					yield(&model.LLMResponse{Content: genai.NewContentFromText("ok", genai.RoleModel), FinishReason: genai.FinishReasonStop}, nil)
				}
			}
			for _, err := range InstrumentGenerateContent(t.Context(), GenerateContentParams{ModelName: "m", Request: req}, func(r *model.LLMResponse) (*model.LLMResponse, string) { return r, "" }, gen) {
				if err != nil {
					t.Fatal(err)
				}
			}
			if diff := cmp.Diff(tc.asked, p.asked); diff != "" {
				t.Errorf("Enabled questions (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.want, p.emitted); diff != "" {
				t.Errorf("emitted records (-want +got):\n%s", diff)
			}
		})
	}
}

func msgs(ms ...chatMessage) []contentAttr {
	return []contentAttr{{genAIInputMessages, messageList(ms)}}
}

func parts(ps ...contentValue) []contentAttr {
	return msgs(chatMessage{Role: roleUser, Parts: partList(ps)})
}

func TestContentAttrsEqual(t *testing.T) {
	text := textPart{partTypeText, "a"}
	shared := []contentAttr{{genAIInputMessages, messageList{{Role: roleUser, Parts: partList{text}}}}}
	for _, tc := range []struct {
		name string
		a, b []contentAttr
		want bool
	}{
		{"nil and nil", nil, nil, true},
		{"nil and empty", nil, []contentAttr{}, false},
		{"lengths differ", shared, append(shared[:1:1], shared[0]), false},
		{"same backing array", shared, shared, true},
		{"keys differ", []contentAttr{{genAIInputMessages, messageList{}}}, []contentAttr{{genAIOutputMessages, messageList{}}}, false},
		{"message list and part list", []contentAttr{{genAIInputMessages, messageList{}}}, []contentAttr{{genAIInputMessages, partList{}}}, false},
		{"nil and empty message list", []contentAttr{{genAIInputMessages, messageList(nil)}}, []contentAttr{{genAIInputMessages, messageList{}}}, false},
		{"system instructions equal", []contentAttr{{genAISystemInstructions, partList{text}}}, []contentAttr{{genAISystemInstructions, partList{text}}}, true},
		{"system instructions differ", []contentAttr{{genAISystemInstructions, partList{text}}}, []contentAttr{{genAISystemInstructions, partList{textPart{partTypeText, "b"}}}}, false},
		{"roles differ", msgs(chatMessage{Role: roleUser, Parts: partList{}}), msgs(chatMessage{Role: roleTool, Parts: partList{}}), false},
		{"nil and empty parts", msgs(chatMessage{Role: roleUser}), msgs(chatMessage{Role: roleUser, Parts: partList{}}), false},
		{"finish reasons equal at two addresses", msgs(chatMessage{FinishReason: new(finishStop)}), msgs(chatMessage{FinishReason: new(finishStop)}), true},
		{"finish reason nil and set", msgs(chatMessage{}), msgs(chatMessage{FinishReason: new("")}), false},
		{"finish reasons differ", msgs(chatMessage{FinishReason: new(finishStop)}), msgs(chatMessage{FinishReason: new(finishLength)}), false},
		{"parts differ in number", parts(text), parts(text, text), false},
		{"text equal", parts(text), parts(textPart{partTypeText, "a"}), true},
		{"text differs", parts(text), parts(textPart{partTypeText, "b"}), false},
		{"text and reasoning", parts(text), parts(textPart{partTypeReasoning, "a"}), false},
		{"same fields, other part type", parts(textPart{"t", "c"}), parts(uriPart{Type: "t", URI: "c"}), false},
		{"blob equal", parts(blobPart{Content: "AA=="}), parts(blobPart{Content: "AA=="}), true},
		{"blob truncated differs", parts(blobPart{Content: "AA=="}), parts(blobPart{Content: "AA==", Truncated: true}), false},
		{"uri differs", parts(uriPart{URI: "a"}), parts(uriPart{URI: "b"}), false},
		{"tool call equal", parts(toolCallPart{Type: "t", ID: "1", Name: "f", Arguments: rawJSON(`{}`)}), parts(toolCallPart{Type: "t", ID: "1", Name: "f", Arguments: rawJSON(`{}`)}), true},
		{"tool call type differs", parts(toolCallPart{Type: "t"}), parts(toolCallPart{Type: "u"}), false},
		{"tool call ID differs", parts(toolCallPart{ID: "1"}), parts(toolCallPart{ID: "2"}), false},
		{"tool call name differs", parts(toolCallPart{Name: "f"}), parts(toolCallPart{Name: "g"}), false},
		{"tool call arguments differ", parts(toolCallPart{Arguments: rawJSON(`{"q":1}`)}), parts(toolCallPart{Arguments: rawJSON(`{"q":2}`)}), false},
		{"tool call arguments nil and empty", parts(toolCallPart{}), parts(toolCallPart{Arguments: rawJSON{}}), false},
		{"tool response equal", parts(toolResponsePart{Type: "t", ID: "1", Name: "f", Response: rawJSON(`1`)}), parts(toolResponsePart{Type: "t", ID: "1", Name: "f", Response: rawJSON(`1`)}), true},
		{"tool response type differs", parts(toolResponsePart{Type: "t"}), parts(toolResponsePart{Type: "u"}), false},
		{"tool response ID differs", parts(toolResponsePart{ID: "1"}), parts(toolResponsePart{ID: "2"}), false},
		{"tool response name differs", parts(toolResponsePart{Name: "f"}), parts(toolResponsePart{Name: "g"}), false},
		{"tool response differs", parts(toolResponsePart{Response: rawJSON(`1`)}), parts(toolResponsePart{Response: rawJSON(`2`)}), false},
		{"tool response nil and empty", parts(toolResponsePart{}), parts(toolResponsePart{Response: rawJSON{}}), false},
		{"other type equal", parts(&textPart{"t", "c"}), parts(&textPart{"t", "c"}), true},
		{"other type differs", parts(&textPart{"t", "c"}), parts(&textPart{"t", "d"}), false},
	} {
		for _, swap := range []bool{false, true} {
			a, b := tc.a, tc.b
			if swap {
				a, b = b, a
			}
			if got := contentAttrsEqual(a, b); got != tc.want {
				t.Errorf("%s (swapped %t): contentAttrsEqual = %t, want %t", tc.name, swap, got, tc.want)
			}
		}
	}
}
