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
	"log"
	"strings"
	"unicode/utf8"

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
)

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

// maxContentAttributeBytes stays under Cloud Trace's 64 KiB attribute limit,
// past which it rejects the whole export request.
const maxContentAttributeBytes = 60 << 10

// unserializablePlaceholder stands in for a tool payload encoding/json rejects.
// Tool arguments and responses are filled in by application code and can hold a
// NaN, an Inf, a func, a chan or a reference cycle.
const unserializablePlaceholder = `"<unserializable>"`

// maxInlineDataBytes bounds one inline payload before base64, which inflates it
// by a third. A single image would otherwise consume the whole attribute and
// leave the conversation around it unrecorded.
const maxInlineDataBytes = 16 << 10

// contentValue is recorded as its JSON encoding in the legacy schema, and as value() otherwise.
type contentValue interface {
	value() attribute.Value
}

type contentAttr struct {
	key   attribute.Key
	value contentValue
}

type messageList []chatMessage

func (l messageList) value() attribute.Value {
	values := make([]attribute.Value, len(l))
	for i, m := range l {
		values[i] = m.value()
	}
	return attribute.SliceValue(values...)
}

// chatMessage is one turn. FinishReason is required on an output message and
// absent from an input one, which is the only difference between the two
// message schemas.
type chatMessage struct {
	Role         string   `json:"role"`
	Parts        partList `json:"parts"`
	FinishReason string   `json:"finish_reason,omitempty"`
}

func (m chatMessage) value() attribute.Value {
	kvs := []attribute.KeyValue{
		str("role", m.Role),
		{Key: "parts", Value: m.Parts.value()},
	}
	if m.FinishReason != "" {
		kvs = append(kvs, str("finish_reason", m.FinishReason))
	}
	return attribute.MapValue(kvs...)
}

type partList []contentValue

func (l partList) value() attribute.Value {
	values := make([]attribute.Value, len(l))
	for i, p := range l {
		values[i] = p.value()
	}
	return attribute.SliceValue(values...)
}

type textPart struct {
	Type    string `json:"type"`
	Content string `json:"content"`
}

func (p textPart) value() attribute.Value {
	return attribute.MapValue(str("type", p.Type), str("content", p.Content))
}

type toolCallPart struct {
	Type      string  `json:"type"`
	ID        string  `json:"id,omitempty"`
	Name      string  `json:"name"`
	Arguments rawJSON `json:"arguments,omitempty"`
}

func (p toolCallPart) value() attribute.Value {
	kvs := []attribute.KeyValue{str("type", p.Type)}
	if p.ID != "" {
		kvs = append(kvs, str("id", p.ID))
	}
	kvs = append(kvs, str("name", p.Name))
	if len(p.Arguments) > 0 {
		kvs = append(kvs, attribute.KeyValue{Key: "arguments", Value: p.Arguments.value()})
	}
	return attribute.MapValue(kvs...)
}

type blobPart struct {
	Type      string `json:"type"`
	MIMEType  string `json:"mime_type,omitempty"`
	Modality  string `json:"modality"`
	Content   string `json:"content"`
	Truncated bool   `json:"truncated,omitempty"`
}

func (p blobPart) value() attribute.Value {
	kvs := []attribute.KeyValue{str("type", p.Type)}
	if p.MIMEType != "" {
		kvs = append(kvs, str("mime_type", p.MIMEType))
	}
	kvs = append(kvs, str("modality", p.Modality), str("content", p.Content))
	if p.Truncated {
		kvs = append(kvs, attribute.Bool("truncated", true))
	}
	return attribute.MapValue(kvs...)
}

type uriPart struct {
	Type     string `json:"type"`
	MIMEType string `json:"mime_type,omitempty"`
	Modality string `json:"modality"`
	URI      string `json:"uri"`
}

func (p uriPart) value() attribute.Value {
	kvs := []attribute.KeyValue{str("type", p.Type)}
	if p.MIMEType != "" {
		kvs = append(kvs, str("mime_type", p.MIMEType))
	}
	kvs = append(kvs, str("modality", p.Modality), str("uri", p.URI))
	return attribute.MapValue(kvs...)
}

type toolResponsePart struct {
	Type     string  `json:"type"`
	ID       string  `json:"id,omitempty"`
	Name     string  `json:"name,omitempty"`
	Response rawJSON `json:"response"`
}

func (p toolResponsePart) value() attribute.Value {
	kvs := []attribute.KeyValue{str("type", p.Type)}
	if p.ID != "" {
		kvs = append(kvs, str("id", p.ID))
	}
	if p.Name != "" {
		kvs = append(kvs, str("name", p.Name))
	}
	kvs = append(kvs, attribute.KeyValue{Key: "response", Value: p.Response.value()})
	return attribute.MapValue(kvs...)
}

// str replaces invalid UTF-8 as encoding/json does, since OTLP exporters reject it.
func str(k, v string) attribute.KeyValue {
	if !utf8.ValidString(v) {
		var b strings.Builder
		for _, r := range v {
			b.WriteRune(r)
		}
		v = b.String()
	}
	return attribute.String(k, v)
}

// rawJSON is an encoded tool payload.
type rawJSON []byte

func (r rawJSON) MarshalJSON() ([]byte, error) {
	if r == nil {
		return []byte("null"), nil
	}
	return r, nil
}

// value decodes r, as JSON is the safe way to walk arbitrary application values.
func (r rawJSON) value() attribute.Value {
	// Keeps integers integers.
	dec := json.NewDecoder(bytes.NewReader(r))
	dec.UseNumber()
	var decoded any
	if err := dec.Decode(&decoded); err != nil {
		return attribute.Value{}
	}
	return toLogValue(decoded)
}

// requestContent is user content, so callers gate it on content capture.
func requestContent(req *model.LLMRequest) []contentAttr {
	if req == nil {
		return nil
	}
	var content []contentAttr
	if req.Config != nil && req.Config.SystemInstruction != nil {
		if parts := semconvParts(req.Config.SystemInstruction.Parts); len(parts) > 0 {
			content = append(content, contentAttr{genAISystemInstructions, parts})
		}
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
		content = append(content, contentAttr{genAIInputMessages, messageList(msgs)})
	}
	return content
}

// responseContent skips partial responses, so a stream records its final answer.
func responseContent(resp *model.LLMResponse, err error) []contentAttr {
	if resp == nil || resp.Partial {
		return nil
	}
	// A candidate suppressed by a safety filter has a finish reason and no
	// parts, and is the case an operator most wants on the span, so the message
	// is recorded with an empty parts list. ADK carries a single candidate, so
	// there is always exactly one output message.
	parts := partList{}
	if resp.Content != nil {
		parts = semconvParts(resp.Content.Parts)
	}
	return []contentAttr{{genAIOutputMessages, messageList{{
		Role:         roleAssistant,
		Parts:        parts,
		FinishReason: schemaFinishReason(resp, err),
	}}}}
}

// spanContentAttributes drops, with a warning, attributes too large to export.
func spanContentAttributes(content []contentAttr) []attribute.KeyValue {
	legacy := useLegacySchema()
	var attrs []attribute.KeyValue
	for _, c := range content {
		encoded, err := json.Marshal(c.value)
		if err != nil {
			// Unreachable.
			continue
		}
		if len(encoded) > maxContentAttributeBytes {
			log.Printf("adk: telemetry: leaving %s off the span: its JSON encoding is %d bytes, over the %d-byte limit", c.key, len(encoded), maxContentAttributeBytes)
			continue
		}
		if legacy {
			attrs = append(attrs, c.key.String(string(encoded)))
		} else {
			attrs = append(attrs, attribute.KeyValue{Key: c.key, Value: c.value.value()})
		}
	}
	return attrs
}

func eventContentAttributes(content []contentAttr) []attribute.KeyValue {
	attrs := make([]attribute.KeyValue, len(content))
	for i, c := range content {
		attrs[i] = attribute.KeyValue{Key: c.key, Value: c.value.value()}
	}
	return attrs
}

// semconvParts converts genai parts, skipping those with no mapping. Never
// returns nil, so a message always has the parts array the schema requires.
func semconvParts(ps []*genai.Part) partList {
	out := make(partList, 0, len(ps))
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
func semconvPart(p *genai.Part) contentValue {
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
// and chans, and detects reference cycles. Walking the value to sanitize it
// would be worse than useless — a map holding two references to itself fans out
// exponentially and never returns.
func toolPayload(v map[string]any) rawJSON {
	if v == nil {
		return rawJSON("null")
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return rawJSON(unserializablePlaceholder)
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

// hasFunctionCall reports whether content carries a tool call. It reads the
// genai parts directly rather than converting them: conversion base64-encodes
// inline data and marshals tool arguments, and this runs on every span whether
// content capture is on or off.
func hasFunctionCall(c *genai.Content) bool {
	if c == nil {
		return false
	}
	for _, p := range c.Parts {
		if p != nil && p.FunctionCall != nil {
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
// A non-nil error, a response carrying an error code, or one cut short by an
// interruption did not complete whatever its finish reason claims, so it
// reports "error".
// Gemini reports STOP for a turn that is a tool call, which the schema
// distinguishes from a plain stop, so a turn containing tool call parts
// reports "tool_call". genai enum values are protobuf wire names in
// SCREAMING_SNAKE_CASE and are never emitted verbatim; a value this mapping
// does not know is lowercased, which the schema allows since finish_reason
// accepts any string.
func schemaFinishReason(resp *model.LLMResponse, err error) string {
	if resp == nil || err != nil || resp.ErrorCode != "" || resp.Interrupted {
		return finishError
	}
	switch resp.FinishReason {
	case "", genai.FinishReasonUnspecified, genai.FinishReasonStop:
		if hasFunctionCall(resp.Content) {
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
