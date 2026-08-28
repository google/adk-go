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

// This file maps ADK's genai request and response types onto the OpenTelemetry
// GenAI content model and records them as span attributes.
//
// The attributes and their JSON schemas are defined by
// https://github.com/open-telemetry/semantic-conventions-genai:
//
//   - model/gen-ai/gen-ai-input-messages.json
//   - model/gen-ai/gen-ai-output-messages.json
//   - model/gen-ai/gen-ai-system-instructions.json
//
// The Go OpenTelemetry attribute API has no complex value type, so each
// attribute holds the JSON serialization of the corresponding schema, which
// the specification permits for spans.
//
// Content is sensitive and often large, so it is captured only when the user
// opts in via OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT. See
// [getGenAICaptureMessageContent].

import (
	"encoding/base64"
	"encoding/json"
	"sort"
	"strings"
	"unicode/utf8"

	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

var (
	genAIInputMessages      = attribute.Key("gen_ai.input.messages")
	genAIOutputMessages     = attribute.Key("gen_ai.output.messages")
	genAISystemInstructions = attribute.Key("gen_ai.system_instructions")
)

const (
	// maxContentAttributeBytes bounds the JSON written to each content
	// attribute. Cloud Trace silently discards any attribute value larger
	// than 64 KiB, so stay comfortably below that. The bound is on the
	// encoded bytes, not the input: JSON escaping expands "<" to "\u003c"
	// and base64 expands blobs by 4/3, so bounding the input would not
	// bound the attribute.
	maxContentAttributeBytes = 60 << 10

	// maxPartContentBytes bounds a single text, blob or tool payload so one
	// oversized part cannot crowd out every other part of the conversation.
	maxPartContentBytes = 8 << 10

	// droppedAnnotationBytes reserves room for the
	// "dropped_preceding_messages" property added to the oldest retained
	// message. The property is at most
	// len(`,"dropped_preceding_messages":`) + 20 digits = 50 bytes.
	droppedAnnotationBytes = 64
)

// Placeholders substituted for tool payloads that cannot be recorded
// verbatim. Both are valid JSON values, so the surrounding document still
// validates against the schema.
var (
	notSerializableJSON = json.RawMessage(`"<not serializable>"`)
	truncatedJSON       = json.RawMessage(`"<truncated>"`)
)

// Part type discriminators defined by the schemas.
const (
	partTypeText                   = "text"
	partTypeReasoning              = "reasoning"
	partTypeToolCall               = "tool_call"
	partTypeToolCallResponse       = "tool_call_response"
	partTypeServerToolCall         = "server_tool_call"
	partTypeServerToolCallResponse = "server_tool_call_response"
	partTypeBlob                   = "blob"
	partTypeURI                    = "uri"
	// partTypeAudioTranscription has no dedicated schema type; it is
	// recorded as a GenericPart, which only requires "type".
	partTypeAudioTranscription = "audio_transcription"
)

// chatMessage is a ChatMessage (input) or an OutputMessage (output). The two
// differ only in finish_reason, which is required on output messages and
// absent on input messages.
//
// Truncated and DroppedPrecedingMessages are extra properties, which the
// schema permits, that tell a reader the recorded content is incomplete.
type chatMessage struct {
	Role                     string `json:"role"`
	Parts                    []any  `json:"parts"`
	FinishReason             string `json:"finish_reason,omitempty"`
	Truncated                bool   `json:"truncated,omitempty"`
	DroppedPrecedingMessages int    `json:"dropped_preceding_messages,omitempty"`
}

// textPart is a TextPart or a ReasoningPart; both carry only a string.
type textPart struct {
	Type      string `json:"type"`
	Content   string `json:"content"`
	Truncated bool   `json:"truncated,omitempty"`
}

// toolCallPart is a ToolCallRequestPart.
type toolCallPart struct {
	Type      string          `json:"type"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
	Truncated bool            `json:"truncated,omitempty"`
}

// toolResponsePart is a ToolCallResponsePart. Response is required, so it is
// written even when empty, where a nil json.RawMessage encodes as null.
type toolResponsePart struct {
	Type      string          `json:"type"`
	ID        string          `json:"id,omitempty"`
	Response  json.RawMessage `json:"response"`
	Truncated bool            `json:"truncated,omitempty"`
}

// blobPart is a BlobPart: binary data sent inline, base64 encoded.
type blobPart struct {
	Type      string `json:"type"`
	MIMEType  string `json:"mime_type,omitempty"`
	Modality  string `json:"modality"`
	Content   string `json:"content"`
	Truncated bool   `json:"truncated,omitempty"`
}

// uriPart is a UriPart: data referenced by URI rather than sent inline.
type uriPart struct {
	Type     string `json:"type"`
	MIMEType string `json:"mime_type,omitempty"`
	Modality string `json:"modality"`
	URI      string `json:"uri"`
}

// serverToolCallPart is a ServerToolCallPart: a tool the provider runs itself,
// such as code execution or search.
type serverToolCallPart struct {
	Type           string `json:"type"`
	ID             string `json:"id,omitempty"`
	Name           string `json:"name"`
	ServerToolCall any    `json:"server_tool_call"`
}

// serverToolCallResponsePart is a ServerToolCallResponsePart.
type serverToolCallResponsePart struct {
	Type                   string `json:"type"`
	ID                     string `json:"id,omitempty"`
	ServerToolCallResponse any    `json:"server_tool_call_response"`
}

// genericPart is a GenericPart, the schema's extension point for content that
// has no dedicated type. Only "type" is required.
type genericPart struct {
	Type    string `json:"type"`
	Content string `json:"content,omitempty"`
	// Language is the BCP-47 tag of an audio transcription, when known.
	Language  string `json:"language,omitempty"`
	Truncated bool   `json:"truncated,omitempty"`
}

// codeExecutionCall and codeExecutionResultDetails are GenericServerToolCall
// and GenericServerToolCallResponse bodies for the provider-run code
// execution tool.
type codeExecutionCall struct {
	Type      string `json:"type"`
	Code      string `json:"code"`
	Language  string `json:"language,omitempty"`
	Truncated bool   `json:"truncated,omitempty"`
}

type codeExecutionResultDetails struct {
	Type      string `json:"type"`
	Outcome   string `json:"outcome,omitempty"`
	Output    string `json:"output,omitempty"`
	Truncated bool   `json:"truncated,omitempty"`
}

type genericServerToolCall struct {
	Type      string          `json:"type"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

type genericServerToolResponse struct {
	Type     string          `json:"type"`
	Response json.RawMessage `json:"response,omitempty"`
}

const (
	serverToolCodeExecution = "code_execution"
	// serverToolUnnamed is used when the provider did not say which
	// server-side tool ran; ServerToolCallPart requires a name.
	serverToolUnnamed = "server_tool"
)

// requestContentAttributes returns the gen_ai.system_instructions and
// gen_ai.input.messages attributes for req, or nil when content capture is
// off or req carries nothing to record.
func requestContentAttributes(req *model.LLMRequest) []attribute.KeyValue {
	if req == nil || !getGenAICaptureMessageContent() {
		return nil
	}
	var attrs []attribute.KeyValue
	b := newContentBuilder()
	if req.Config != nil && req.Config.SystemInstruction != nil {
		if parts := b.parts(req.Config.SystemInstruction.Parts); len(parts) > 0 {
			if s, ok := encodeParts(parts, maxContentAttributeBytes); ok {
				attrs = append(attrs, genAISystemInstructions.String(s))
			}
		}
		// Clear the flag the instruction set. It belongs to the instruction,
		// whose attribute is a bare part list with nowhere to record it. Left
		// set, the next message built claims it, and a prompt recorded in full
		// is reported as truncated.
		b.takeTruncated()
	}
	msgs := make([]*chatMessage, 0, len(req.Contents))
	for _, c := range req.Contents {
		if c == nil {
			continue
		}
		msgs = append(msgs, &chatMessage{
			Role:      schemaRole(c),
			Parts:     b.parts(c.Parts),
			Truncated: b.takeTruncated(),
		})
	}
	if len(msgs) > 0 {
		if s, ok := encodeMessages(msgs, maxContentAttributeBytes); ok {
			attrs = append(attrs, genAIInputMessages.String(s))
		}
	}
	return attrs
}

// responseContentAttributes returns the gen_ai.output.messages attribute for
// resp, or nil when content capture is off.
//
// Streaming chunks are skipped: each chunk holds a fragment of the turn, and
// recording one would overwrite the attribute with that fragment. Only the
// settled response is recorded, matching [LogResponse].
func responseContentAttributes(resp *model.LLMResponse) []attribute.KeyValue {
	if resp == nil || resp.Partial || !getGenAICaptureMessageContent() {
		return nil
	}
	b := newContentBuilder()
	// parts is required and must be an array, so it is never left nil: a
	// response blocked before it produced anything still reports its
	// finish reason.
	parts := []any{}
	role := "assistant"
	if resp.Content != nil {
		parts = b.parts(resp.Content.Parts)
		role = schemaRole(resp.Content)
	}
	msg := &chatMessage{
		Role:         role,
		Parts:        parts,
		FinishReason: schemaFinishReason(resp, hasToolCall(parts)),
		Truncated:    b.takeTruncated(),
	}
	s, ok := encodeMessages([]*chatMessage{msg}, maxContentAttributeBytes)
	if !ok {
		return nil
	}
	return []attribute.KeyValue{genAIOutputMessages.String(s)}
}

// contentBuilder converts genai parts into schema parts, shortening any
// payload that exceeds the per-part budget. One builder serves one span, and
// spans are built from a single goroutine, so it needs no locking.
type contentBuilder struct {
	stringLimit int
	valueLimit  int
	// truncated records whether anything was shortened since the flag was
	// last taken.
	truncated bool
}

func newContentBuilder() *contentBuilder {
	return &contentBuilder{stringLimit: maxPartContentBytes, valueLimit: maxPartContentBytes}
}

// takeTruncated reports and clears the truncation flag.
func (b *contentBuilder) takeTruncated() bool {
	t := b.truncated
	b.truncated = false
	return t
}

// text shortens s to the per-part budget, measured in encoded bytes.
func (b *contentBuilder) text(s string) (string, bool) {
	out, cut := truncateJSONString(s, b.stringLimit)
	b.truncated = b.truncated || cut
	return out, cut
}

// value serializes an arbitrary application-supplied payload, such as tool
// call arguments.
//
// The payload comes from application code and may hold NaN, channels,
// functions or reference cycles. encoding/json rejects all of those, and
// detects cycles itself, so a single Marshal is both the serialization and
// the validity check. Walking the value here to sanitize it would be a trap:
// a map holding two references to itself fans out exponentially.
func (b *contentBuilder) value(v map[string]any) (json.RawMessage, bool) {
	if len(v) == 0 {
		return nil, false
	}
	raw, err := json.Marshal(v)
	if err != nil {
		b.truncated = true
		return notSerializableJSON, true
	}
	if len(raw) > b.valueLimit {
		b.truncated = true
		return truncatedJSON, true
	}
	return raw, false
}

// parts converts genai parts into schema parts, preserving order.
func (b *contentBuilder) parts(ps []*genai.Part) []any {
	out := make([]any, 0, len(ps))
	for _, p := range ps {
		out = append(out, b.part(p)...)
	}
	return out
}

// part converts one genai part. genai.Part is a union whose variants are all
// optional fields, so a part may in principle set more than one; each set
// variant becomes its own schema part rather than one silently winning.
// Fields that carry no content (video metadata, media resolution, thought
// signatures, part metadata) are not recorded.
func (b *contentBuilder) part(p *genai.Part) []any {
	if p == nil {
		return nil
	}
	var out []any
	if p.Text != "" {
		typ := partTypeText
		if p.Thought {
			typ = partTypeReasoning
		}
		content, cut := b.text(p.Text)
		out = append(out, textPart{Type: typ, Content: content, Truncated: cut})
	}
	if fc := p.FunctionCall; fc != nil {
		args, cut := b.value(fc.Args)
		out = append(out, toolCallPart{
			Type:      partTypeToolCall,
			ID:        fc.ID,
			Name:      fc.Name,
			Arguments: args,
			Truncated: cut,
		})
	}
	if fr := p.FunctionResponse; fr != nil {
		resp, cut := b.value(fr.Response)
		out = append(out, toolResponsePart{
			Type:      partTypeToolCallResponse,
			ID:        fr.ID,
			Response:  resp,
			Truncated: cut,
		})
	}
	if blob := p.InlineData; blob != nil {
		content, cut := b.base64(blob.Data)
		out = append(out, blobPart{
			Type:      partTypeBlob,
			MIMEType:  blob.MIMEType,
			Modality:  modalityOf(blob.MIMEType),
			Content:   content,
			Truncated: cut,
		})
	}
	if fd := p.FileData; fd != nil {
		uri, _ := b.text(fd.FileURI)
		out = append(out, uriPart{
			Type:     partTypeURI,
			MIMEType: fd.MIMEType,
			Modality: modalityOf(fd.MIMEType),
			URI:      uri,
		})
	}
	if ec := p.ExecutableCode; ec != nil {
		code, cut := b.text(ec.Code)
		out = append(out, serverToolCallPart{
			Type: partTypeServerToolCall,
			ID:   ec.ID,
			Name: serverToolCodeExecution,
			ServerToolCall: codeExecutionCall{
				Type:      serverToolCodeExecution,
				Code:      code,
				Language:  codeLanguage(ec.Language),
				Truncated: cut,
			},
		})
	}
	if cr := p.CodeExecutionResult; cr != nil {
		output, cut := b.text(cr.Output)
		out = append(out, serverToolCallResponsePart{
			Type: partTypeServerToolCallResponse,
			ID:   cr.ID,
			ServerToolCallResponse: codeExecutionResultDetails{
				Type:      serverToolCodeExecution,
				Outcome:   codeExecutionOutcome(cr.Outcome),
				Output:    output,
				Truncated: cut,
			},
		})
	}
	if tc := p.ToolCall; tc != nil {
		args, _ := b.value(tc.Args)
		name := serverToolName(tc.ToolType)
		out = append(out, serverToolCallPart{
			Type:           partTypeServerToolCall,
			ID:             tc.ID,
			Name:           name,
			ServerToolCall: genericServerToolCall{Type: name, Arguments: args},
		})
	}
	if tr := p.ToolResponse; tr != nil {
		resp, _ := b.value(tr.Response)
		out = append(out, serverToolCallResponsePart{
			Type:                   partTypeServerToolCallResponse,
			ID:                     tr.ID,
			ServerToolCallResponse: genericServerToolResponse{Type: serverToolName(tr.ToolType), Response: resp},
		})
	}
	if at := p.AudioTranscription; at != nil && at.Text != "" {
		content, cut := b.text(at.Text)
		out = append(out, genericPart{
			Type:      partTypeAudioTranscription,
			Content:   content,
			Language:  at.LanguageCode,
			Truncated: cut,
		})
	}
	return out
}

// base64 encodes data for a BlobPart, encoding only as many input bytes as
// fit the per-part budget so an oversized blob is never expanded in full.
func (b *contentBuilder) base64(data []byte) (string, bool) {
	if len(data) == 0 {
		return "", false
	}
	max := base64.StdEncoding.DecodedLen(b.stringLimit)
	if len(data) <= max {
		return base64.StdEncoding.EncodeToString(data), false
	}
	b.truncated = true
	return base64.StdEncoding.EncodeToString(data[:max]), true
}

// hasToolCall reports whether any schema part is a tool call, client or
// server side.
func hasToolCall(parts []any) bool {
	for _, p := range parts {
		switch p.(type) {
		case toolCallPart, serverToolCallPart:
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

// Finish reasons defined by the output messages schema.
const (
	finishStop          = "stop"
	finishLength        = "length"
	finishContentFilter = "content_filter"
	finishToolCall      = "tool_call"
	finishError         = "error"
)

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

// modalityOf maps an IANA MIME type onto the schema's modality enum, which is
// required on blob and uri parts. Anything that is not image, video or audio
// is a document.
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

// codeLanguage renders a genai Language wire name, such as PYTHON, in the
// lower case form used in telemetry.
func codeLanguage(l genai.Language) string {
	if l == "" || l == genai.LanguageUnspecified {
		return ""
	}
	return strings.ToLower(string(l))
}

// codeExecutionOutcome renders a genai Outcome wire name, such as OUTCOME_OK,
// without its enum prefix.
func codeExecutionOutcome(o genai.Outcome) string {
	switch o {
	case genai.OutcomeOK:
		return "ok"
	case genai.OutcomeFailed:
		return "failed"
	case genai.OutcomeDeadlineExceeded:
		return "deadline_exceeded"
	default:
		return ""
	}
}

// serverToolName renders a genai ToolType wire name, such as
// GOOGLE_SEARCH_WEB, in the lower case form used in telemetry.
// ServerToolCallPart requires a name, so an unspecified tool type falls back
// to a placeholder.
func serverToolName(t genai.ToolType) string {
	if t == "" || t == genai.ToolTypeUnspecified {
		return serverToolUnnamed
	}
	return strings.ToLower(string(t))
}

// encodeMessages serializes msgs as the JSON array the input and output
// message schemas define, bounded to limit bytes.
//
// The conversation history grows with every turn, so the array will
// eventually exceed the bound. Messages are dropped oldest first, which keeps
// the most recent context and preserves the order of what remains, as the
// specification requires; the oldest surviving message records how many were
// dropped. If even the newest message is too large on its own, its trailing
// parts are dropped instead. The result is always the JSON array the schema
// declares, never a marker string.
//
// It reports false if nothing could be encoded within the bound, in which
// case the caller records no attribute at all.
func encodeMessages(msgs []*chatMessage, limit int) (string, bool) {
	if len(msgs) == 0 {
		return "", false
	}
	raws := make([]json.RawMessage, len(msgs))
	sizes := make([]int, len(msgs))
	for i, m := range msgs {
		// Every field is a string, bool, int or pre-serialized
		// json.RawMessage by this point, so marshalling cannot fail.
		raw, err := json.Marshal(m)
		if err != nil {
			return "", false
		}
		raws[i] = raw
		sizes[i] = len(raw)
	}

	start := suffixStart(sizes, limit-droppedAnnotationBytes)
	if start == len(msgs) {
		// Not even the newest message fits by itself.
		last := msgs[len(msgs)-1]
		raw, ok := shrinkMessage(last, len(msgs)-1, limit)
		if !ok {
			return "", false
		}
		return joinJSONArray([]json.RawMessage{raw}, limit)
	}
	if start > 0 {
		head := msgs[start]
		head.DroppedPrecedingMessages = start
		raw, err := json.Marshal(head)
		if err != nil {
			return "", false
		}
		raws[start] = raw
	}
	return joinJSONArray(raws[start:], limit)
}

// encodeParts serializes parts as the JSON array the system instructions
// schema defines, bounded to limit bytes. Instructions are ordered by
// importance rather than recency, so the leading parts are kept.
func encodeParts(parts []any, limit int) (string, bool) {
	raws := make([]json.RawMessage, 0, len(parts))
	total := 2
	for _, p := range parts {
		raw, err := json.Marshal(p)
		if err != nil {
			return "", false
		}
		grow := len(raw)
		if len(raws) > 0 {
			grow++
		}
		if total+grow > limit {
			break
		}
		total += grow
		raws = append(raws, raw)
	}
	if len(raws) == 0 {
		return "", false
	}
	return joinJSONArray(raws, limit)
}

// suffixStart returns the index of the first element of the longest suffix of
// sizes whose JSON array encoding fits in limit bytes, or len(sizes) if no
// suffix fits.
func suffixStart(sizes []int, limit int) int {
	total := 2 // the enclosing brackets
	start := len(sizes)
	for i := len(sizes) - 1; i >= 0; i-- {
		grow := sizes[i]
		if start < len(sizes) {
			grow++ // the separating comma
		}
		if total+grow > limit {
			break
		}
		total += grow
		start = i
	}
	return start
}

// shrinkMessage drops trailing parts from m until its encoding fits in limit
// bytes, leaving as many leading parts as possible. dropped is the number of
// older messages already discarded.
func shrinkMessage(m *chatMessage, dropped, limit int) (json.RawMessage, bool) {
	parts := m.Parts
	m.Truncated = true
	m.DroppedPrecedingMessages = dropped
	// The encoded size grows with the number of parts kept, so binary search
	// for the largest number that still fits. Reserve two bytes for the
	// enclosing array brackets.
	tooBig := func(n int) bool {
		m.Parts = parts[:n]
		raw, err := json.Marshal(m)
		if err != nil {
			return true
		}
		return len(raw)+2 > limit
	}
	// sort.Search returns the smallest part count that overflows, so the
	// largest that fits is one less. Zero means even a message with no parts
	// is too large, which the caller reports as no attribute.
	first := sort.Search(len(parts)+1, tooBig)
	if first == 0 {
		return nil, false
	}
	m.Parts = parts[:first-1]
	raw, err := json.Marshal(m)
	if err != nil {
		return nil, false
	}
	return raw, true
}

// joinJSONArray assembles already-encoded JSON values into an array,
// re-checking the bound. Marshalling a json.RawMessage preserves it verbatim,
// so the assembled length is exactly the sum of the parts.
func joinJSONArray(raws []json.RawMessage, limit int) (string, bool) {
	out, err := json.Marshal(raws)
	if err != nil || len(out) > limit {
		return "", false
	}
	return string(out), true
}

// truncateJSONString shortens s so that its JSON encoding, excluding the
// enclosing quotes, is at most limit bytes, cutting only at a rune boundary.
// It reports whether anything was removed.
//
// The bound is on encoded bytes because encoding/json escapes "<", ">" and
// "&" as six-byte \uXXXX sequences: a string of 1000 "<" occupies 6000 bytes
// on the span, so a bound on raw bytes would not bound the attribute.
func truncateJSONString(s string, limit int) (string, bool) {
	n := 0
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		c := escapedRuneLen(r, size)
		if n+c > limit {
			return s[:i], true
		}
		n += c
		i += size
	}
	return s, false
}

// escapedRuneLen returns the number of bytes r occupies inside a JSON string,
// where size is the width of its UTF-8 encoding. It mirrors the escaping
// encoding/json applies, including HTML escaping, which is on by default.
func escapedRuneLen(r rune, size int) int {
	switch {
	case r == utf8.RuneError && size == 1:
		return 6 // an invalid byte is written as the escape \ufffd
	case r == '"' || r == '\\' || r == '\b' || r == '\f' || r == '\n' || r == '\r' || r == '\t':
		return 2 // written as a two-character escape such as \n
	case r < 0x20 || r == '<' || r == '>' || r == '&' || r == '\u2028' || r == '\u2029':
		return 6 // \uXXXX
	default:
		return size
	}
}
