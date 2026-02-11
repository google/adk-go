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
	"fmt"
	"os"
	"strings"

	"go.opentelemetry.io/otel/log"
	"go.opentelemetry.io/otel/log/global"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/version"
	"google.golang.org/adk/model"
)

// Message content is not logged by default. Set the following env variable to enable logging of prompt/response content.
// OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
var elideMessageContent = !isEnvVarTrue("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")

const elidedContent = "<elided>"

// guessedGenAISystem is the AI system we are using.
var guessedGenAISystem = guessAISystem()

var logger = global.GetLoggerProvider().Logger(
	systemName,
	log.WithSchemaURL(semconv.SchemaURL),
	log.WithInstrumentationVersion(version.Version),
)

// LogRequest logs the request to the model - the system message and user messages.
// It iterates over the request contents and logs each as a separate event.
func LogRequest(ctx context.Context, req *model.LLMRequest) {
	logSystemMessage(ctx, req)
	for _, content := range req.Contents {
		logUserMessage(ctx, content)
	}
}

// LogResponse logs the inference result.
// It follows the format defined in https://github.com/open-telemetry/semantic-conventions/blob/v1.36.0/docs/gen-ai/gen-ai-events.md#event-gen_aichoice.
func LogResponse(ctx context.Context, resp *model.LLMResponse, err error) {
	record := log.Record{}
	record.SetEventName("gen_ai.choice")

	var messageKvs []log.KeyValue
	var finishReason string

	var content *genai.Content
	var toolCalls []log.Value
	var role string
	if resp != nil {
		finishReason = string(resp.FinishReason)
		if resp.Content != nil {
			content = resp.Content
			role = resp.Content.Role
			toolCalls = extractToolCalls(content)
		}
	}
	messageKvs = append(messageKvs,
		log.KeyValue{Key: "content", Value: contentToLogValue(content)},
	)
	if len(toolCalls) > 0 {
		messageKvs = append(messageKvs, log.Slice("tool_calls", toolCalls...))
	}
	if role != "" {
		messageKvs = append(messageKvs, log.String("role", role))
	}

	kvs := []log.KeyValue{
		// Gemini only returns one choice, hardcoding to 0.
		log.Int("index", 0),
		{Key: "message", Value: log.MapValue(messageKvs...)},
	}

	if finishReason != "" {
		kvs = append(kvs, log.String("finish_reason", finishReason))
	}
	record.SetBody(log.MapValue(kvs...))

	logger.Emit(ctx, record)
}

// logSystemMessage logs the system message from the request.
// It follows the format defined in https://github.com/open-telemetry/semantic-conventions/blob/v1.36.0/docs/gen-ai/gen-ai-events.md#event-gen_aisystemmessage.
func logSystemMessage(ctx context.Context, req *model.LLMRequest) {
	record := log.Record{}
	record.SetEventName("gen_ai.system.message")
	record.SetBody(log.MapValue(
		log.String("role", "system"),
		log.KeyValue{Key: "content", Value: extractSystemMessage(req)},
	))
	record.AddAttributes(
		aiSystemAttribute(),
	)
	logger.Emit(ctx, record)
}

// logUserMessage logs the user message from the request.
// It follows the format defined in https://github.com/open-telemetry/semantic-conventions/blob/v1.36.0/docs/gen-ai/gen-ai-events.md#event-gen_aiusermessage.
func logUserMessage(ctx context.Context, content *genai.Content) {
	record := log.Record{}
	record.SetEventName("gen_ai.user.message")

	var kvs []log.KeyValue
	var role string
	if content != nil {
		role = content.Role
	}
	if role != "" {
		kvs = append(kvs, log.String("role", role))
	}
	v := contentToJSONValue(content)
	kvs = append(kvs, log.KeyValue{Key: "content", Value: mapToLogValue(v)})
	record.SetBody(log.MapValue(kvs...))
	record.AddAttributes(
		aiSystemAttribute(),
	)

	logger.Emit(ctx, record)
}

func isEnvVarTrue(name string) bool {
	val, ok := os.LookupEnv(name)
	if !ok {
		return false
	}
	val = strings.ToLower(val)
	return val == "true" || val == "1"
}

func guessAISystem() string {
	if isEnvVarTrue("GOOGLE_GENAI_USE_VERTEXAI") {
		return semconv.GenAISystemGCPVertexAI.Value.AsString()
	}
	return semconv.GenAISystemGCPGenAI.Value.AsString()
}

func aiSystemAttribute() log.KeyValue {
	return log.String(string(semconv.GenAISystemKey), guessedGenAISystem)
}

// extractSystemMessage extracts the system message from the request config and concatenates it into a single string.
// If the content is elided, it returns the elided content string.
func extractSystemMessage(req *model.LLMRequest) log.Value {
	if elideMessageContent {
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

func extractToolCalls(content *genai.Content) []log.Value {
	if content == nil {
		return nil
	}
	var toolCalls []log.Value
	for _, part := range content.Parts {
		if part.FunctionCall != nil {
			argsStr := safeSerialize(part.FunctionCall.Args)
			toolCalls = append(toolCalls, log.MapValue(
				log.String("id", part.FunctionCall.ID),
				log.String("type", "function"),
				log.KeyValue{Key: "function", Value: log.MapValue(
					log.String("name", part.FunctionCall.Name),
					log.String("arguments", argsStr),
				)},
			))
		}
	}
	return toolCalls
}

func contentToLogValue(c *genai.Content) log.Value {
	return mapToLogValue(contentToJSONValue(c))
}

func contentToJSONValue(c *genai.Content) any {
	if elideMessageContent {
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

// mapToLogValue converts a JSON value to a log.Value.
// From [encoding/json.Unmarshal] documentation:
// To unmarshal JSON into an interface value,
// Unmarshal stores one of these in the interface value:
//
//   - bool, for JSON booleans
//   - float64, for JSON numbers
//   - string, for JSON strings
//   - []any, for JSON arrays
//   - map[string]any, for JSON objects
//   - nil for JSON null
func mapToLogValue(v any) log.Value {
	switch val := v.(type) {
	case nil:
		return log.Value{}
	case string:
		return log.StringValue(val)
	case bool:
		return log.BoolValue(val)
	case float64:
		return log.Float64Value(val)
	case int:
		return log.IntValue(val)
	case []any:
		values := make([]log.Value, 0, len(val))
		for _, item := range val {
			values = append(values, mapToLogValue(item))
		}
		return log.SliceValue(values...)
	case map[string]any:
		kvs := make([]log.KeyValue, 0, len(val))
		for k, v := range val {
			kvs = append(kvs, log.KeyValue{Key: k, Value: mapToLogValue(v)})
		}
		return log.MapValue(kvs...)
	default:
		// Fallback for other types
		return log.StringValue(fmt.Sprintf("%v", val))
	}
}
