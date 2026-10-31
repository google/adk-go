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
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/log"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// refLogRequest, refLogResponse, refLogSystemMessage and refLogUserMessage are
// the legacy request and response logging as it was before it asked the
// logger whether it is enabled: every record is built and emitted, each
// through a logger resolved for it. The tests below hold the current
// functions to them.

func refLogRequest(ctx context.Context, req *model.LLMRequest, backend genai.Backend) {
	if !useLegacySchema() {
		return
	}
	genAISystem := variantToGenAISystem(backend)
	refLogSystemMessage(ctx, req, genAISystem)
	for _, content := range req.Contents {
		refLogUserMessage(ctx, content, genAISystem)
	}
}

func refLogResponse(ctx context.Context, resp *model.LLMResponse, backend genai.Backend) {
	if !useLegacySchema() {
		return
	}
	record := log.Record{}
	record.SetEventName("gen_ai.choice")

	var finishReason string
	var content *genai.Content
	if resp != nil {
		finishReason = string(resp.FinishReason)
		if resp.Content != nil {
			content = resp.Content
		}
	}

	kvs := []attribute.KeyValue{
		attribute.Int("index", 0),
		{Key: "content", Value: contentToLogValue(content)},
	}

	if finishReason != "" {
		kvs = append(kvs, attribute.String("finish_reason", finishReason))
	}
	record.SetBody(attribute.MapValue(kvs...))

	genAISystem := variantToGenAISystem(backend)
	if genAISystem != nil {
		record.AddAttributes(*genAISystem)
	}

	otelLogger().Emit(ctx, record)
}

func refLogSystemMessage(ctx context.Context, req *model.LLMRequest, genAISystem *attribute.KeyValue) {
	record := log.Record{}
	record.SetEventName("gen_ai.system.message")
	record.SetBody(attribute.MapValue(
		attribute.KeyValue{Key: "content", Value: extractSystemMessage(req)},
	))
	if genAISystem != nil {
		record.AddAttributes(*genAISystem)
	}
	otelLogger().Emit(ctx, record)
}

func refLogUserMessage(ctx context.Context, content *genai.Content, genAISystem *attribute.KeyValue) {
	record := log.Record{}
	record.SetEventName("gen_ai.user.message")
	record.SetBody(attribute.MapValue(
		attribute.KeyValue{Key: "content", Value: toLogValue(contentToJSONLikeValue(content))},
	))
	if genAISystem != nil {
		record.AddAttributes(*genAISystem)
	}

	otelLogger().Emit(ctx, record)
}

// equivalenceParts covers every part kind the legacy records encode, with
// unicode, thought and non-thought text, and nested tool payloads.
func equivalenceParts() []*genai.Part {
	return []*genai.Part{
		{Text: "Zażółć gęślą jaźń — 東京 🚀 \u2028 <b>&amp;</b>"},
		{Text: "Let me think.", Thought: true, ThoughtSignature: []byte{0, 1, 2, 255}},
		{Text: ""},
		{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "lookup", Args: map[string]any{
			"q": "weather", "n": 3, "f": 0.1, "big": int64(1) << 60, "ok": true, "none": nil,
			"nested": map[string]any{"list": []any{1, "two", map[string]any{"三": 3.5}}},
		}}},
		{FunctionResponse: &genai.FunctionResponse{ID: "call-1", Name: "lookup", Response: map[string]any{
			"result": []any{map[string]any{"city": "Kraków", "temp": -2.5}}, "error": nil,
		}}},
		{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("\x89PNG\r\n\x1a\n")}},
		{FileData: &genai.FileData{MIMEType: "application/pdf", FileURI: "gs://bucket/doc.pdf"}},
	}
}

func legacyLogRequests() map[string]*model.LLMRequest {
	large := strings.Repeat("ö", 100_000)
	return map[string]*model.LLMRequest{
		"empty":                   {},
		"empty contents":          {Contents: []*genai.Content{}},
		"system instruction only": {Config: &genai.GenerateContentConfig{SystemInstruction: &genai.Content{Parts: []*genai.Part{{Text: "Be brief."}, {Text: ""}, {Text: "Ünïcode ✓", Thought: true}}}}},
		"config without system":   {Config: &genai.GenerateContentConfig{}, Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)}},
		"nil content":             {Contents: []*genai.Content{nil, genai.NewContentFromText("after nil", genai.RoleUser)}},
		"no parts":                {Contents: []*genai.Content{{Role: genai.RoleUser}, {Role: genai.RoleModel, Parts: []*genai.Part{}}}},
		"every part kind": {
			Config:   &genai.GenerateContentConfig{SystemInstruction: genai.NewContentFromText("You are terse.", genai.RoleUser)},
			Contents: []*genai.Content{{Role: genai.RoleUser, Parts: equivalenceParts()}, {Role: genai.RoleModel, Parts: equivalenceParts()}},
		},
		"large":            {Contents: []*genai.Content{genai.NewContentFromText(large, genai.RoleUser)}},
		"not serializable": {Contents: []*genai.Content{{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "f", Args: map[string]any{"x": math.NaN()}}}}}}},
		"history":          benchHistory(16),
	}
}

func legacyLogResponses() map[string]*model.LLMResponse {
	return map[string]*model.LLMResponse{
		"nil":                nil,
		"empty":              {},
		"text":               {Content: genai.NewContentFromText("Done — ✓", genai.RoleModel), FinishReason: genai.FinishReasonStop},
		"every part kind":    {Content: &genai.Content{Role: genai.RoleModel, Parts: equivalenceParts()}, FinishReason: genai.FinishReasonMaxTokens},
		"no content":         {FinishReason: genai.FinishReasonSafety},
		"large":              {Content: genai.NewContentFromText(strings.Repeat("ä", 100_000), genai.RoleModel)},
		"not serializable":   {Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "f", Args: map[string]any{"x": math.Inf(1)}}}}}},
		"partial with error": {Partial: true, ErrorCode: "500", ErrorMessage: "boom"},
	}
}

// recordSnapshot is what an exporter receives of a record, less the observed
// timestamp, which the SDK sets to the time of the call. Values keep their
// types and their order, so snapshots compare equal only for identical records.
type recordSnapshot struct {
	EventName         string
	Severity          log.Severity
	SeverityText      string
	Timestamp         time.Time
	Body              any
	Attributes        []any
	DroppedAttributes int
	TraceID           trace.TraceID
	SpanID            trace.SpanID
	TraceFlags        trace.TraceFlags
	Scope             [3]string
}

func snapshotRecords(records []sdklog.Record) []recordSnapshot {
	var out []recordSnapshot
	for _, r := range records {
		var attrs []any
		r.WalkAttributes(func(kv attribute.KeyValue) bool {
			attrs = append(attrs, typedKeyValue(kv))
			return true
		})
		scope := r.InstrumentationScope()
		out = append(out, recordSnapshot{
			EventName:         r.EventName(),
			Severity:          r.Severity(),
			SeverityText:      r.SeverityText(),
			Timestamp:         r.Timestamp(),
			Body:              typedValue(r.Body()),
			Attributes:        attrs,
			DroppedAttributes: r.DroppedAttributes(),
			TraceID:           r.TraceID(),
			SpanID:            r.SpanID(),
			TraceFlags:        r.TraceFlags(),
			Scope:             [3]string{scope.Name, scope.Version, scope.SchemaURL},
		})
	}
	return out
}

// typedValue renders v with its type at every level and map entries in their
// stored order.
func typedValue(v attribute.Value) any {
	switch v.Type() {
	case attribute.SLICE:
		s := []any{}
		for _, e := range v.AsSlice() {
			s = append(s, typedValue(e))
		}
		return s
	case attribute.MAP:
		m := []any{}
		for _, kv := range v.AsMap() {
			m = append(m, typedKeyValue(kv))
		}
		return m
	default:
		return fmt.Sprintf("%s(%s)", v.Type(), v.Emit())
	}
}

func typedKeyValue(kv attribute.KeyValue) any {
	return [2]any{string(kv.Key), typedValue(kv.Value)}
}

// assertSameRecords fails unless got are exactly the records want are, field
// by field, observed timestamp aside.
func assertSameRecords(t *testing.T, what string, want, got []sdklog.Record) {
	t.Helper()
	if len(want) == 0 {
		t.Fatalf("%s: the reference emitted no records, so there is nothing to compare", what)
	}
	wantSnap, gotSnap := snapshotRecords(want), snapshotRecords(got)
	if !reflect.DeepEqual(wantSnap, gotSnap) {
		t.Errorf("%s: records differ (-reference +current):\n%s", what, cmp.Diff(wantSnap, gotSnap))
		return
	}
	for i := range want {
		w, g := want[i].Clone(), got[i].Clone()
		w.SetObservedTimestamp(time.Time{})
		g.SetObservedTimestamp(time.Time{})
		if !reflect.DeepEqual(w, g) {
			t.Errorf("%s: record %d differs from the reference", what, i)
		}
	}
}

// TestLegacyLogsMatchReference holds logRequest and logResponse to the
// reference above for a logger that is enabled, inside a recording span, for
// every capture setting and backend.
func TestLegacyLogsMatchReference(t *testing.T) {
	for _, capture := range []string{"", "true", "SPAN_ONLY", "SPAN_AND_EVENT"} {
		for _, backend := range []genai.Backend{genai.BackendUnspecified, genai.BackendGeminiAPI, genai.BackendVertexAI} {
			t.Run(fmt.Sprintf("capture=%q/%s", capture, backend), func(t *testing.T) {
				setEnvForTesting(t, map[string]string{adkTelemetrySchemaVersionOptIn: otelSemconv136, captureMessageContentEnvVar: capture})
				setupTestTracer(t)
				exporter := &inMemoryExporter{}
				OverrideLoggerForTesting(t, sdklog.NewLoggerProvider(sdklog.WithProcessor(sdklog.NewSimpleProcessor(exporter))))
				ctx, span := tracer().Start(t.Context(), "generate_content")
				defer span.End()
				if !span.IsRecording() {
					t.Fatal("span is not recording")
				}
				emitted := func(f func()) []sdklog.Record {
					exporter.records = nil
					f()
					return exporter.records
				}

				for name, req := range legacyLogRequests() {
					want := emitted(func() { refLogRequest(ctx, req, backend) })
					got := emitted(func() { logRequest(ctx, req, backend) })
					assertSameRecords(t, "request "+name, want, got)
				}
				for name, resp := range legacyLogResponses() {
					want := emitted(func() { refLogResponse(ctx, resp, backend) })
					got := emitted(func() { logResponse(ctx, resp, backend) })
					assertSameRecords(t, "response "+name, want, got)
				}
			})
		}
	}
}

// enabledRecorder is a log processor enabled for the event names in enable. It
// counts what it is given and remembers what it was asked.
type enabledRecorder struct {
	enable  map[string]bool
	asked   []sdklog.EnabledParameters
	emitted []string
}

func (p *enabledRecorder) Enabled(_ context.Context, param sdklog.EnabledParameters) bool {
	p.asked = append(p.asked, param)
	return p.enable[param.EventName]
}

func (p *enabledRecorder) OnEmit(_ context.Context, r *sdklog.Record) error {
	p.emitted = append(p.emitted, r.EventName())
	return nil
}

func (p *enabledRecorder) Shutdown(context.Context) error   { return nil }
func (p *enabledRecorder) ForceFlush(context.Context) error { return nil }

// TestLegacyLogsOnlyForAnEnabledLogger checks that each kind of legacy record
// is built only when the logger is enabled for its event name, asked without a
// severity, since the records carry none.
func TestLegacyLogsOnlyForAnEnabledLogger(t *testing.T) {
	for _, tc := range []struct {
		name   string
		enable map[string]bool
		want   []string
	}{
		{"none", nil, nil},
		{"user messages", map[string]bool{userMessageEventName: true}, []string{userMessageEventName, userMessageEventName, userMessageEventName}},
		{"system and choice", map[string]bool{systemMessageEventName: true, choiceEventName: true}, []string{systemMessageEventName, choiceEventName}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			setEnvForTesting(t, map[string]string{adkTelemetrySchemaVersionOptIn: otelSemconv136, captureMessageContentEnvVar: "true"})
			p := &enabledRecorder{enable: tc.enable}
			OverrideLoggerForTesting(t, sdklog.NewLoggerProvider(sdklog.WithProcessor(p)))

			logRequest(t.Context(), benchHistory(3), genai.BackendVertexAI)
			logResponse(t.Context(), &model.LLMResponse{}, genai.BackendVertexAI)

			if diff := cmp.Diff(tc.want, p.emitted); diff != "" {
				t.Errorf("emitted records mismatch (-want +got):\n%s", diff)
			}
			var asked []string
			for _, a := range p.asked {
				if a.Severity != log.SeverityUndefined {
					t.Errorf("Enabled asked with severity %v for %s, want none", a.Severity, a.EventName)
				}
				asked = append(asked, a.EventName)
			}
			if diff := cmp.Diff([]string{systemMessageEventName, userMessageEventName, choiceEventName}, asked); diff != "" {
				t.Errorf("Enabled questions mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
