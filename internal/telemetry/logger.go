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
	"encoding/json"
	"os"
	"strings"
	"sync"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/log"
	"go.opentelemetry.io/otel/log/global"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/version"
	"google.golang.org/adk/v2/model"
)

// captureMessageContentEnvVar is the OpenTelemetry-spec env var that says where
// ADK may record full message content. Defined by
// https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/.
const captureMessageContentEnvVar = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"

// contentCaptureMode says which signals may carry message content.
//
// The variable was a boolean here before spans could carry content, and
// everyone who set it agreed to log records. Logs and traces routinely go to
// different backends, with different retention and different readers, so a
// truthy value keeps meaning log records only and spans have to be named. This
// matches adk-python, which reads a truthy value as EVENT_ONLY for the same
// reason.
type contentCaptureMode int

const (
	captureNone contentCaptureMode = iota
	captureEventOnly
	captureSpanOnly
	captureSpanAndEvent
)

var (
	captureMode contentCaptureMode = captureNone
	once        sync.Once
)

func parseContentCaptureMode(s string) contentCaptureMode {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "EVENT_ONLY":
		return captureEventOnly
	case "SPAN_ONLY":
		return captureSpanOnly
	case "SPAN_AND_EVENT":
		return captureSpanAndEvent
	}
	if evalsToTrue(s) {
		return captureEventOnly
	}
	return captureNone
}

func contentCapture() contentCaptureMode {
	once.Do(func() {
		ApplyEnv()
	})
	return captureMode
}

// getGenAICaptureMessageContent reports whether message content should be
// captured in log records.
func getGenAICaptureMessageContent() bool {
	mode := contentCapture()
	return mode == captureEventOnly || mode == captureSpanAndEvent
}

// captureContentOnSpans reports whether message content should be recorded as
// span attributes. Deliberately not satisfied by a truthy value: see
// [contentCaptureMode].
func captureContentOnSpans() bool {
	mode := contentCapture()
	return mode == captureSpanOnly || mode == captureSpanAndEvent
}

const elidedContent = "<elided>"

var otelLogger = global.GetLoggerProvider().Logger(
	systemName,
	log.WithSchemaURL(semconv.SchemaURL),
	log.WithInstrumentationVersion(version.Version),
)

// OverrideLoggerForTesting replaces the package-level otelLogger
// with one derived from lp for the duration of the calling test.
// The original logger is restored via t.Cleanup.
func OverrideLoggerForTesting(t interface{ Cleanup(func()) }, lp log.LoggerProvider) {
	original := otelLogger
	otelLogger = lp.Logger(
		systemName,
		log.WithSchemaURL(semconv.SchemaURL),
		log.WithInstrumentationVersion(version.Version),
	)
	t.Cleanup(func() { otelLogger = original })
}

// LogRequest logs the request to the model - the system message and user messages.
// It iterates over the request contents and logs each as a separate event.
// Check [logSystemMessage] and [logUserMessage] for emitted event details.
func LogRequest(ctx context.Context, req *model.LLMRequest, backend genai.Backend) {
	genAISystem := variantToGenAISystem(backend)
	logSystemMessage(ctx, req, genAISystem)
	for _, content := range req.Contents {
		logUserMessage(ctx, content, genAISystem)
	}
}

// LogResponse logs the inference result.
// Semconv reference: https://github.com/open-telemetry/semantic-conventions/blob/v1.36.0/docs/gen-ai/gen-ai-events.md#event-gen_aichoice.
// NOTE: The current implementation doesn't fully follow the spec, but aims for consistency with ADK Python. The differences are:
// * The spec embeds the "content" field to be under the "message" key, but it's added directly in body.
// * The "tool_calls" field is required if available in the spec, but it's omitted.
func LogResponse(ctx context.Context, resp *model.LLMResponse, backend genai.Backend) {
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
		// ADK internal data model only supports single candidate, even though the implementations can return multiple candidates. Hardcoding index to 0.
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

	otelLogger.Emit(ctx, record)
}

// logSystemMessage logs the system message from the request.
// Semconv reference: https://github.com/open-telemetry/semantic-conventions/blob/v1.36.0/docs/gen-ai/gen-ai-events.md#event-gen_aisystemmessage.
// NOTE: The current implementation doesn't fully follow the spec, but aims for consistency with ADK Python. The differences are:
// * The spec requires a "role" body field, but it's ommited.
func logSystemMessage(ctx context.Context, req *model.LLMRequest, genAISystem *attribute.KeyValue) {
	record := log.Record{}
	record.SetEventName("gen_ai.system.message")
	record.SetBody(attribute.MapValue(
		attribute.KeyValue{Key: "content", Value: extractSystemMessage(req)},
	))
	if genAISystem != nil {
		record.AddAttributes(*genAISystem)
	}
	otelLogger.Emit(ctx, record)
}

// logUserMessage logs the user message from the request.
// Semconv reference: https://github.com/open-telemetry/semantic-conventions/blob/v1.36.0/docs/gen-ai/gen-ai-events.md#event-gen_aiusermessage.
// NOTE: The current implementation doesn't fully follow the spec, but aims for consistency with ADK Python. The differences are:
// * The spec requires a "role" body field, but it's ommited. If the role is set in [genai.Content], then it will be available in body.content.role.
func logUserMessage(ctx context.Context, content *genai.Content, genAISystem *attribute.KeyValue) {
	record := log.Record{}
	record.SetEventName("gen_ai.user.message")
	record.SetBody(attribute.MapValue(
		attribute.KeyValue{Key: "content", Value: toLogValue(contentToJSONLikeValue(content))},
	))
	if genAISystem != nil {
		record.AddAttributes(*genAISystem)
	}

	otelLogger.Emit(ctx, record)
}

// GenAISystemAttr returns the gen_ai.system attribute for a backend, and
// whether this repo can name one.
//
// The single definition, so telemetry that reports a provider agrees with
// itself. It reports false for a provider it cannot identify, since naming the
// wrong one is worse than saying nothing.
//
// Ref: https://github.com/open-telemetry/semantic-conventions/blob/v1.36.0/docs/registry/attributes/gen-ai.md#gen-ai-system well-known values.
// GenAISystemAttr reports the semconv gen_ai.system attribute for a Google
// backend, and whether one is known at all.
//
// Split out so the compaction spans can set the same attribute the rest of the
// telemetry does, rather than deriving it a second way.
func GenAISystemAttr(variant genai.Backend) (attribute.KeyValue, bool) {
	switch variant {
	case genai.BackendVertexAI:
		return semconv.GenAISystemGCPVertexAI, true
	case genai.BackendGeminiAPI:
		return semconv.GenAISystemGCPGemini, true
	}
	return attribute.KeyValue{}, false
}

func variantToGenAISystem(variant genai.Backend) *attribute.KeyValue {
	attr, ok := GenAISystemAttr(variant)
	if !ok {
		return nil
	}
	return &attr
}

// extractSystemMessage extracts the system message from the request config and concatenates it into a single string.
// If the content is elided, it returns the elided content string.
func extractSystemMessage(req *model.LLMRequest) attribute.Value {
	if !getGenAICaptureMessageContent() {
		return attribute.StringValue(elidedContent)
	}
	if req == nil || req.Config == nil || req.Config.SystemInstruction == nil {
		return attribute.Value{}
	}
	var text []string
	for _, p := range req.Config.SystemInstruction.Parts {
		if p.Text != "" {
			text = append(text, p.Text)
		}
	}
	content := strings.Join(text, "\n")
	return attribute.StringValue(content)
}

func contentToLogValue(c *genai.Content) attribute.Value {
	return toLogValue(contentToJSONLikeValue(c))
}

// contentToJSONLikeValue converts a genai.Content to a JSON, which is then converted to an attribute.Value.
func contentToJSONLikeValue(c *genai.Content) any {
	if !getGenAICaptureMessageContent() {
		return elidedContent
	}
	if c == nil {
		return nil
	}

	// Marshall to JSON first to preserve the json key names, omit null fields, etc.
	b, err := json.Marshal(c)
	if err != nil {
		return "<not_serializable>"
	}

	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return "<not_serializable>"
	}
	return m
}

// ApplyEnv reads OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT. It accepts
// EVENT_ONLY, SPAN_ONLY and SPAN_AND_EVENT, and treats "1" or "true" as
// EVENT_ONLY for back-compatibility. Anything else records no content.
func ApplyEnv() {
	captureMode = parseContentCaptureMode(os.Getenv(captureMessageContentEnvVar))
}

func evalsToTrue(s string) bool {
	u := strings.ToLower(strings.TrimSpace(s))
	return u == "1" || u == "true"
}
