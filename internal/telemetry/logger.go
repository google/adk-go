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
	"strings"
	"sync/atomic"

	"go.opentelemetry.io/otel/log"
	"go.opentelemetry.io/otel/log/global"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/version"
	"google.golang.org/adk/model"
)

// genAICaptureMessageContent is true if message content should be elided. False by default.
var genAICaptureMessageContent atomic.Bool

// SetGenAICaptureMessageContent sets whether message content should be elided.
func SetGenAICaptureMessageContent(capture bool) {
	genAICaptureMessageContent.Store(capture)
}

// getGenAICaptureMessageContent returns whether message content should be elided.
func getGenAICaptureMessageContent() bool {
	return genAICaptureMessageContent.Load()
}

const elidedContent = "<elided>"

var otelLogger = global.GetLoggerProvider().Logger(
	systemName,
	log.WithSchemaURL(semconv.SchemaURL),
	log.WithInstrumentationVersion(version.Version),
)

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

	kvs := []log.KeyValue{
		// ADK internal data model only supports single candidate, even though the implementations can return multiple candidates. Hardcoding index to 0.
		log.Int("index", 0),
		{Key: "content", Value: contentToLogValue(content)},
	}

	if finishReason != "" {
		kvs = append(kvs, log.String("finish_reason", finishReason))
	}
	record.SetBody(log.MapValue(kvs...))

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
func logSystemMessage(ctx context.Context, req *model.LLMRequest, genAISystem *log.KeyValue) {
	record := log.Record{}
	record.SetEventName("gen_ai.system.message")
	record.SetBody(log.MapValue(
		log.KeyValue{Key: "content", Value: extractSystemMessage(req)},
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
func logUserMessage(ctx context.Context, content *genai.Content, genAISystem *log.KeyValue) {
	record := log.Record{}
	record.SetEventName("gen_ai.user.message")
	record.SetBody(log.MapValue(
		log.KeyValue{Key: "content", Value: toLogValue(contentToJSONLikeValue(content))},
	))
	if genAISystem != nil {
		record.AddAttributes(*genAISystem)
	}

	otelLogger.Emit(ctx, record)
}

// Ref: https://github.com/open-telemetry/semantic-conventions/blob/v1.36.0/docs/registry/attributes/gen-ai.md#gen-ai-system well-known values.
func variantToGenAISystem(variant genai.Backend) *log.KeyValue {
	if variant == genai.BackendVertexAI {
		val := log.KeyValueFromAttribute(semconv.GenAISystemGCPVertexAI)
		return &val
	}
	if variant == genai.BackendGeminiAPI {
		val := log.KeyValueFromAttribute(semconv.GenAISystemGCPGemini)
		return &val
	}
	return nil
}

// extractSystemMessage extracts the system message from the request config and concatenates it into a single string.
// If the content is elided, it returns the elided content string.
func extractSystemMessage(req *model.LLMRequest) log.Value {
	if !getGenAICaptureMessageContent() {
		return log.StringValue(elidedContent)
	}
	if req == nil || req.Config == nil || req.Config.SystemInstruction == nil {
		return log.Value{}
	}
	var text []string
	for _, p := range req.Config.SystemInstruction.Parts {
		if p.Text != "" {
			text = append(text, p.Text)
		}
	}
	content := strings.Join(text, "\n")
	return log.StringValue(content)
}

func contentToLogValue(c *genai.Content) log.Value {
	return toLogValue(contentToJSONLikeValue(c))
}

// contentToJSONLikeValue converts a genai.Content to a JSON, which is then converted to a log.Value.
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
