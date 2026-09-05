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
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// Opt-in content attributes from the GenAI semantic conventions.
// https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md#inference
var (
	genAIInputMessages      = attribute.Key("gen_ai.input.messages")
	genAIOutputMessages     = attribute.Key("gen_ai.output.messages")
	genAISystemInstructions = attribute.Key("gen_ai.system_instructions")
	genAIToolDefinitions    = attribute.Key("gen_ai.tool.definitions")
)

var genaiTypeNames = map[string]string{
	string(genai.TypeString):  "string",
	string(genai.TypeNumber):  "number",
	string(genai.TypeInteger): "integer",
	string(genai.TypeBoolean): "boolean",
	string(genai.TypeArray):   "array",
	string(genai.TypeObject):  "object",
	string(genai.TypeNULL):    "null",
}

// Part type discriminators and roles defined by the message JSON schemas.
const (
	partTypeText             = "text"
	partTypeReasoning        = "reasoning"
	partTypeToolCall         = "tool_call"
	partTypeToolCallResponse = "tool_call_response"
	partTypeBlob             = "blob"
	partTypeURI              = "uri"

	roleUser      = "user"
	roleAssistant = "assistant"
	roleTool      = "tool"

	// Members of the schema's FinishReason enum.
	finishStop          = "stop"
	finishLength        = "length"
	finishContentFilter = "content_filter"
	finishToolCall      = "tool_call"
	finishError         = "error"
)

// maxContentAttributeBytes bounds each content attribute.
//
// A request carries the whole conversation, rebuilt on every model call, so an
// attribute grows with the session and would eventually exceed what a backend
// accepts — Cloud Trace discards an attribute value over 64 KiB in full, and
// silently. Rather than trim, an attribute that does not fit is left unset:
// losing one span's content is recoverable, and a partial value that claims to
// be the whole conversation is not. Trimming to fit is worth adding later.
const maxContentAttributeBytes = 60 << 10

// unserializablePlaceholder stands in for a tool payload encoding/json rejects.
// Tool arguments and responses are filled in by application code and can hold a
// NaN, an Inf, a func, a chan or a reference cycle.
const unserializablePlaceholder = `"<unserializable>"`

// unserializableSchemaPlaceholder is a valid JSON Schema used when a tool
// parameter schema cannot be encoded. Unlike unserializablePlaceholder, this
// value is used where the semantic convention requires a schema object.
const unserializableSchemaPlaceholder = `{"type":"object","properties":{"serialization_error":{"type":"string"}}}`

// maxInlineDataBytes bounds one inline payload before base64, which inflates it
// by a third. A single image would otherwise consume the whole attribute and
// leave the conversation around it unrecorded.
const maxInlineDataBytes = 16 << 10

// chatMessage is one turn. FinishReason is required on an output message and
// absent from an input one, which is the only difference between the two
// message schemas.
type chatMessage struct {
	Role         string `json:"role"`
	Parts        []any  `json:"parts"`
	FinishReason string `json:"finish_reason,omitempty"`
}

type textPart struct {
	Type    string `json:"type"`
	Content string `json:"content"`
}

type toolCallPart struct {
	Type      string          `json:"type"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

type blobPart struct {
	Type      string `json:"type"`
	MIMEType  string `json:"mime_type,omitempty"`
	Modality  string `json:"modality"`
	Content   string `json:"content"`
	Truncated bool   `json:"truncated,omitempty"`
}

type uriPart struct {
	Type     string `json:"type"`
	MIMEType string `json:"mime_type,omitempty"`
	Modality string `json:"modality"`
	URI      string `json:"uri"`
}

type toolResponsePart struct {
	Type     string          `json:"type"`
	ID       string          `json:"id,omitempty"`
	Name     string          `json:"name,omitempty"`
	Response json.RawMessage `json:"response"`
}

// functionToolDefinition is the representation used by the GenAI semantic
// conventions for a function declaration. Description is always emitted, while
// parameters are included only when explicitly enabled.
type functionToolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Type        string          `json:"type"`
}

// toolDefinitions converts the function declarations present in the actual
// model request into the gen_ai.tool.definitions representation. Reading the
// request is important because a Toolset may produce a different set of tools
// for each model call.
func toolDefinitions(config *genai.GenerateContentConfig) []functionToolDefinition {
	if config == nil {
		return nil
	}
	includeParameters := captureToolDefinitionParametersOnSpans()
	var definitions []functionToolDefinition
	for _, tool := range config.Tools {
		if tool == nil {
			continue
		}
		for _, declaration := range tool.FunctionDeclarations {
			if declaration == nil {
				continue
			}
			definition := functionToolDefinition{
				Name:        declaration.Name,
				Description: declaration.Description,
				Type:        "function",
			}
			if includeParameters {
				definition.Parameters = toolDefinitionParameters(declaration)
			}
			definitions = append(definitions, definition)
		}
	}
	return definitions
}

// toolDefinitionParameters returns a valid JSON value for a declaration's
// parameters. Each declaration is encoded independently so one malformed
// application-provided schema cannot erase every tool definition on the span.
func toolDefinitionParameters(declaration *genai.FunctionDeclaration) json.RawMessage {
	if declaration.Parameters == nil && declaration.ParametersJsonSchema == nil {
		return json.RawMessage("null")
	}

	value := any(declaration.Parameters)
	if declaration.Parameters == nil {
		value = declaration.ParametersJsonSchema
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return json.RawMessage(unserializableSchemaPlaceholder)
	}

	var normalized any
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	if err := decoder.Decode(&normalized); err != nil {
		return json.RawMessage(unserializableSchemaPlaceholder)
	}
	switch normalized.(type) {
	case map[string]any, bool:
		// JSON Schema draft-07 permits schemas to be objects or booleans.
	default:
		return json.RawMessage(unserializableSchemaPlaceholder)
	}
	normalizeSchemaTypes(normalized)
	encoded, err = json.Marshal(normalized)
	if err != nil {
		return json.RawMessage(unserializableSchemaPlaceholder)
	}
	return json.RawMessage(encoded)
}

// normalizeSchemaTypes converts only genai.Schema's protobuf enum spellings
// to the lowercase type values required by JSON Schema. It deliberately leaves
// arbitrary strings in data-bearing fields such as default and examples alone.
func normalizeSchemaTypes(value any) {
	normalizeSchemaNode(value)
}

// normalizeSchemaNode walks only the JSON Schema locations that contain nested
// schemas. Walking every nested map would also rewrite ordinary instance data
// under keywords such as default, examples, and dependencies.
func normalizeSchemaNode(value any) {
	obj, ok := value.(map[string]any)
	if !ok {
		return
	}
	if typeValue, ok := obj["type"]; ok {
		if normalized, keep := normalizeSchemaTypeValue(typeValue); keep {
			obj["type"] = normalized
		} else {
			delete(obj, "type")
		}
	}
	for key, child := range obj {
		switch key {
		case "properties", "patternProperties", "dependentSchemas", "definitions", "$defs":
			if schemas, ok := child.(map[string]any); ok {
				for _, schema := range schemas {
					normalizeSchemaNode(schema)
				}
			}
		case "dependencies":
			// In draft-07 a dependency is either a property-name array or a
			// schema. Normalize only schema-valued dependencies.
			if dependencies, ok := child.(map[string]any); ok {
				for _, dependency := range dependencies {
					if schema, ok := dependency.(map[string]any); ok {
						normalizeSchemaNode(schema)
					}
				}
			}
		case "additionalProperties", "additionalItems", "items", "contains", "not", "if", "then", "else", "propertyNames", "unevaluatedProperties", "unevaluatedItems":
			normalizeSchemaValue(child)
		case "allOf", "anyOf", "oneOf", "prefixItems":
			if schemas, ok := child.([]any); ok {
				for _, schema := range schemas {
					normalizeSchemaNode(schema)
				}
			}
		}
	}
}

func normalizeSchemaValue(value any) {
	switch value := value.(type) {
	case map[string]any:
		normalizeSchemaNode(value)
	case []any:
		for _, schema := range value {
			normalizeSchemaNode(schema)
		}
	}
}

func normalizeSchemaTypeValue(value any) (any, bool) {
	switch value := value.(type) {
	case string:
		if normalized, ok := genaiTypeNames[value]; ok {
			return normalized, true
		}
		if value == string(genai.TypeUnspecified) {
			return nil, false
		}
		return value, true
	case []any:
		normalized := make([]any, 0, len(value))
		for _, item := range value {
			item, keep := normalizeSchemaTypeValue(item)
			if keep {
				normalized = append(normalized, item)
			}
		}
		if len(normalized) == 0 {
			return nil, false
		}
		return normalized, true
	default:
		return value, true
	}
}

// requestContentAttributes returns the gen_ai.system_instructions,
// gen_ai.tool.definitions, and gen_ai.input.messages attributes for req, or nil when content capture is off
// or req carries nothing to record.
//
// Content is sensitive and often large, so the semantic conventions require
// instrumentations not to capture it by default. This is gated on
// OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, the same flag that
// governs content in log records.
func requestContentAttributes(req *model.LLMRequest) []attribute.KeyValue {
	if req == nil || !captureContentOnSpans() {
		return nil
	}
	var attrs []attribute.KeyValue
	if req.Config != nil && req.Config.SystemInstruction != nil {
		if parts := semconvParts(req.Config.SystemInstruction.Parts); len(parts) > 0 {
			attrs = appendJSON(attrs, genAISystemInstructions, parts)
		}
	}
	if definitions := toolDefinitions(req.Config); len(definitions) > 0 {
		attrs = appendJSON(attrs, genAIToolDefinitions, definitions)
	}
	if len(req.Contents) > 0 {
		// Messages MUST be recorded in the order they were sent. A turn whose
		// parts are all unrepresentable is kept with an empty parts list rather
		// than dropped, so a consumer sees a turn it cannot render instead of a
		// gap in the conversation.
		msgs := make([]chatMessage, 0, len(req.Contents))
		for _, c := range req.Contents {
			if c == nil {
				continue
			}
			msgs = append(msgs, chatMessage{Role: schemaRole(c), Parts: semconvParts(c.Parts)})
		}
		attrs = appendJSON(attrs, genAIInputMessages, msgs)
	}
	return attrs
}

// responseContentAttributes returns the gen_ai.output.messages attribute for
// resp, gated exactly as [requestContentAttributes] is.
//
// Partial responses are skipped. Each streamed chunk would otherwise overwrite
// the attribute, leaving the span holding a fragment rather than the answer.
func responseContentAttributes(resp *model.LLMResponse) []attribute.KeyValue {
	if resp == nil || resp.Partial || !captureContentOnSpans() {
		return nil
	}
	// A candidate suppressed by a safety filter has a finish reason and no
	// parts, and is the case an operator most wants on the span, so the message
	// is recorded with an empty parts list. ADK carries a single candidate, so
	// there is always exactly one output message.
	var parts []any
	if resp.Content != nil {
		parts = semconvParts(resp.Content.Parts)
	} else {
		parts = []any{}
	}
	return appendJSON(nil, genAIOutputMessages, []chatMessage{{
		Role:         roleAssistant,
		Parts:        parts,
		FinishReason: schemaFinishReason(resp, hasToolCall(parts)),
	}})
}

// semconvParts converts genai parts, skipping those with no mapping. Never
// returns nil, so a message always has the parts array the schema requires.
func semconvParts(ps []*genai.Part) []any {
	out := make([]any, 0, len(ps))
	for _, p := range ps {
		if converted := semconvPart(p); converted != nil {
			out = append(out, converted)
		}
	}
	return out
}

// semconvPart converts one genai part.
//
// Code execution and provider-side tool parts have schema equivalents and are
// dropped for now rather than recorded in a shape that would have to be
// corrected later.
//
// Structured variants are matched before Text. genai documents that exactly one
// field of a part should be set, but if that is ever violated, losing a tool
// call to an accompanying string is the worse outcome.
func semconvPart(p *genai.Part) any {
	switch {
	case p == nil:
		return nil

	case p.FunctionCall != nil:
		return toolCallPart{
			Type:      partTypeToolCall,
			ID:        p.FunctionCall.ID,
			Name:      p.FunctionCall.Name,
			Arguments: toolPayload(p.FunctionCall.Args),
		}

	case p.FunctionResponse != nil:
		return toolResponsePart{
			Type:     partTypeToolCallResponse,
			ID:       p.FunctionResponse.ID,
			Name:     p.FunctionResponse.Name,
			Response: toolPayload(p.FunctionResponse.Response),
		}

	case p.InlineData != nil:
		content, cut := encodeInlineData(p.InlineData.Data)
		return blobPart{
			Type:      partTypeBlob,
			MIMEType:  p.InlineData.MIMEType,
			Modality:  modalityOf(p.InlineData.MIMEType),
			Content:   content,
			Truncated: cut,
		}

	case p.FileData != nil:
		return uriPart{
			Type:     partTypeURI,
			MIMEType: p.FileData.MIMEType,
			Modality: modalityOf(p.FileData.MIMEType),
			URI:      p.FileData.FileURI,
		}

	case p.Text != "":
		// A thought part is the model's reasoning rather than its answer, and
		// the schema gives it a part type of its own.
		if p.Thought {
			return textPart{Type: partTypeReasoning, Content: p.Text}
		}
		return textPart{Type: partTypeText, Content: p.Text}

	default:
		return nil
	}
}

// toolPayload encodes an application-supplied tool value.
//
// json.Marshal is both the encoding and the check: it rejects NaN, Inf, funcs
// and chans, and detects reference cycles. Walking the value to sanitise it
// would be worse than useless — a map holding two references to itself fans out
// exponentially and never returns.
func toolPayload(v map[string]any) json.RawMessage {
	if v == nil {
		return json.RawMessage("null")
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return json.RawMessage(unserializablePlaceholder)
	}
	return raw
}

// encodeInlineData base64-encodes an inline payload, cutting it to
// maxInlineDataBytes first and reporting whether it did. content is required on
// a blob part, so a payload of no bytes still records the empty string rather
// than being omitted.
func encodeInlineData(data []byte) (string, bool) {
	if len(data) <= maxInlineDataBytes {
		return base64.StdEncoding.EncodeToString(data), false
	}
	return base64.StdEncoding.EncodeToString(data[:maxInlineDataBytes]), true
}

// modalityOf maps an IANA MIME type onto the schema's modality enum, which is
// required on blob and uri parts and has no "other" member. MIME types are
// case-insensitive, per RFC 2045 section 5.1.
func modalityOf(mimeType string) string {
	switch t := strings.ToLower(mimeType); {
	case strings.HasPrefix(t, "image/"):
		return "image"
	case strings.HasPrefix(t, "video/"):
		return "video"
	case strings.HasPrefix(t, "audio/"):
		return "audio"
	default:
		return "document"
	}
}

// hasToolCall reports whether any schema part is a tool call.
func hasToolCall(parts []any) bool {
	for _, p := range parts {
		switch p.(type) {
		case toolCallPart:
			return true
		}
	}
	return false
}

// schemaRole maps a genai role onto the schema's role enum. genai uses "user"
// and "model"; the schema uses system, user, assistant and tool. A turn
// carrying a tool result is a "tool" message even though genai labels it
// "user". Any other role is passed through, which the schema allows.
func schemaRole(c *genai.Content) string {
	if c == nil {
		return "user"
	}
	for _, p := range c.Parts {
		if p != nil && (p.FunctionResponse != nil || p.ToolResponse != nil) {
			return "tool"
		}
	}
	switch c.Role {
	case "", genai.RoleUser:
		return "user"
	case genai.RoleModel:
		return "assistant"
	default:
		return c.Role
	}
}

// schemaFinishReason maps a response onto the schema's finish_reason, which
// is required on every output message.
//
// A response carrying an error code, or one cut short by an interruption, did
// not complete whatever its finish reason claims, so it reports "error".
// Gemini reports STOP for a turn that is a tool call, which the schema
// distinguishes from a plain stop, so a turn containing tool call parts
// reports "tool_call". genai enum values are protobuf wire names in
// SCREAMING_SNAKE_CASE and are never emitted verbatim; a value this mapping
// does not know is lowercased, which the schema allows since finish_reason
// accepts any string.
func schemaFinishReason(resp *model.LLMResponse, toolCall bool) string {
	if resp == nil {
		return finishError
	}
	if resp.ErrorCode != "" || resp.Interrupted {
		return finishError
	}
	switch resp.FinishReason {
	case "", genai.FinishReasonUnspecified, genai.FinishReasonStop:
		if toolCall {
			return finishToolCall
		}
		return finishStop
	case genai.FinishReasonMaxTokens:
		return finishLength
	case genai.FinishReasonSafety,
		genai.FinishReasonRecitation,
		genai.FinishReasonLanguage,
		genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent,
		genai.FinishReasonSPII,
		genai.FinishReasonImageSafety,
		genai.FinishReasonImageProhibitedContent,
		genai.FinishReasonImageRecitation:
		return finishContentFilter
	case genai.FinishReasonOther,
		genai.FinishReasonMalformedFunctionCall,
		genai.FinishReasonUnexpectedToolCall,
		genai.FinishReasonTooManyToolCalls,
		genai.FinishReasonNoImage,
		genai.FinishReasonImageOther:
		return finishError
	default:
		return strings.ToLower(string(resp.FinishReason))
	}
}

// appendJSON encodes value and appends it to attrs under key, unless it does
// not fit. See [maxContentAttributeBytes] for why an oversized attribute is
// dropped rather than trimmed. Values used for tool payloads are encoded
// independently before they reach this function so one bad payload cannot
// discard its siblings.
func appendJSON(attrs []attribute.KeyValue, key attribute.Key, value any) []attribute.KeyValue {
	encoded, err := json.Marshal(value)
	if err != nil {
		// Callers should pre-encode values whose individual failures must be
		// isolated, such as tool definition parameters.
		return attrs
	}
	if len(encoded) > maxContentAttributeBytes {
		return attrs
	}
	return append(attrs, key.String(string(encoded)))
}
