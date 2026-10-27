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

package agentanalytics

import (
	"bytes"
	"cmp"
	"encoding"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"strconv"
	"strings"
	"unicode/utf8"

	"google.golang.org/genai"
)

// Values written in place of data that cannot be logged as is. Apart from
// circularRefPlaceholder, which stands in for a back-reference rather than
// for data, writing one of them marks the row as truncated.
const (
	truncationSuffix          = "...[TRUNCATED]"
	inlineDataText            = "[BINARY DATA]"
	binaryDataPlaceholder     = "[BINARY_DATA]"
	circularRefPlaceholder    = "[CIRCULAR_REFERENCE]"
	maxDepthPlaceholder       = "[MAX_DEPTH_EXCEEDED]"
	sanitizeBudgetPlaceholder = "[SANITIZE_BUDGET_EXCEEDED]"
	unsupportedKeyPlaceholder = "[UNSUPPORTED_KEY]"
	unsupportedObjPlaceholder = "[UNSUPPORTED_OBJECT]"
)

// maxSanitizeDepth bounds how deeply nested values are converted; deeper
// values are replaced with maxDepthPlaceholder. maxSanitizeNodes bounds the
// total number of nodes visited in a single value; values exceeding it are
// replaced with sanitizeBudgetPlaceholder. Both match adk-python.
const (
	maxSanitizeDepth  = 50
	maxSanitizeNodes  = 100000
	maxContainerAlloc = 1024
)

type nodeBudget struct {
	remaining int
}

func (b *nodeBudget) take() bool {
	if b.remaining <= 0 {
		return false
	}
	b.remaining--
	return true
}

var (
	jsonMarshalerType = reflect.TypeFor[json.Marshaler]()
	textMarshalerType = reflect.TypeFor[encoding.TextMarshaler]()
	jsonNumberType    = reflect.TypeFor[json.Number]()
)

// Values of the storage_mode column of content_parts.
const (
	storageModeInline       = "INLINE"
	storageModeExternalURI  = "EXTERNAL_URI"
	storageModeGCSReference = "GCS_REFERENCE"
)

// defaultBinaryMIMEType is the MIME type that an offloaded inline blob without
// one is recorded with and stored as. Without offloading, a missing MIME type
// is written as it is.
const defaultBinaryMIMEType = "application/octet-stream"

// FormatContentParts formats Content parts into a map array for BigQuery logging.
func FormatContentParts(content *genai.Content, maxLength int) []map[string]any {
	parts, _ := formatContentParts(content, maxLength, true)
	return parts
}

// formatContentParts is FormatContentParts that also reports whether any
// part lost content to truncation or URI redaction.
func formatContentParts(content *genai.Content, maxLength int, redact bool) ([]map[string]any, bool) {
	c := partsCollector{opts: partsOptions{maxLen: maxLength, redact: redact}}
	c.addContent(content, 0)
	return c.parts, c.truncated
}

// partsOptions configures a partsCollector.
type partsOptions struct {
	// maxLen is the longest text, in runes, that is written inline; a
	// negative value means no limit.
	maxLen int
	// redact replaces the values of sensitive keys in part_attributes.
	redact bool
	// offload turns on Cloud Storage offloading: inline binary data, and text
	// longer than inlineLimit bytes or maxLen runes, are queued as
	// pendingUploads, and gs:// files get an object_ref.
	offload bool
	// inlineLimit is the length in bytes above which text is offloaded. It
	// must be positive when offload is set.
	inlineLimit int
	// connectionID is the authorizer of the object references.
	connectionID string
	// reserveBytes, if set, reserves n bytes of in-flight upload budget and
	// reports whether the payload fits within the cap.
	reserveBytes func(n int) bool
}

// partsCollector builds the content_parts of one row from one or more
// messages. Everything it keeps is detached from the messages: strings are
// immutable, and the payloads queued for upload are copies, so the row can be
// handed to another goroutine while the caller goes on modifying the messages.
type partsCollector struct {
	opts partsOptions
	// parts are the content_parts entries in order.
	parts []map[string]any
	// pending are the payloads to upload, each naming its entry in parts.
	pending []pendingUpload
	// truncated reports that a part lost content to truncation or URI
	// redaction.
	truncated bool
	// nonUploadTruncated reports truncation from parts other than queued text
	// uploads, so resolveUploads can recompute is_truncated after uploads run.
	nonUploadTruncated bool
}

func (c *partsCollector) reserve(n int) (int64, bool) {
	if c.opts.reserveBytes == nil {
		return 0, true
	}
	if !c.opts.reserveBytes(n) {
		return 0, false
	}
	return int64(n), true
}

// hasPendingText reports whether any queued upload is a text part.
func (c *partsCollector) hasPendingText() bool {
	for _, u := range c.pending {
		if u.isText {
			return true
		}
	}
	return false
}

// addContent appends the parts of content. part_index counts the parts of one
// message, so contentOrd, the position of the message in its request, is what
// tells apart the parts of different messages in object names.
func (c *partsCollector) addContent(content *genai.Content, contentOrd int) {
	if content == nil {
		return
	}
	for i, part := range content.Parts {
		entry := map[string]any{
			"part_index":   i,
			"storage_mode": storageModeInline,
		}
		if part != nil {
			c.addPart(entry, part, contentOrd, i)
		}
		c.parts = append(c.parts, entry)
	}
}

// addPart fills entry, which is about to become the next element of c.parts,
// from part.
func (c *partsCollector) addPart(entry map[string]any, part *genai.Part, contentOrd, index int) {
	switch {
	case part.Text != "":
		c.addText(entry, part.Text, contentOrd, index)
	case part.InlineData != nil:
		c.addInlineData(entry, part.InlineData, contentOrd, index)
	case part.FileData != nil:
		c.addFileData(entry, part.FileData)
	case part.FunctionCall != nil:
		name, nameTrunc := sanitizeAndTruncateText(orUnknown(part.FunctionCall.Name), c.opts.maxLen, c.opts.redact)
		attrs, attrTrunc := formatPartAttributes(map[string]any{
			"function_name": part.FunctionCall.Name,
			"function_call": part.FunctionCall,
		}, c.opts.maxLen, c.opts.redact)
		entry["mime_type"] = "application/json"
		entry["text"] = "Function: " + name
		entry["part_attributes"] = attrs
		c.truncated = c.truncated || nameTrunc || attrTrunc
		c.nonUploadTruncated = c.nonUploadTruncated || nameTrunc || attrTrunc
	case part.FunctionResponse != nil:
		name, nameTrunc := sanitizeAndTruncateText(orUnknown(part.FunctionResponse.Name), c.opts.maxLen, c.opts.redact)
		attrs, attrTrunc := formatPartAttributes(map[string]any{
			"function_response": normalizeFunctionResponse(part.FunctionResponse),
		}, c.opts.maxLen, c.opts.redact)
		entry["mime_type"] = "application/json"
		entry["text"] = "Function response: " + name
		entry["part_attributes"] = attrs
		c.truncated = c.truncated || nameTrunc || attrTrunc
		c.nonUploadTruncated = c.nonUploadTruncated || nameTrunc || attrTrunc
	case part.ExecutableCode != nil:
		c.addStructured(entry, "text/plain", part.ExecutableCode.Code,
			"executable_code", part.ExecutableCode)
	case part.CodeExecutionResult != nil:
		c.addStructured(entry, "text/plain", part.CodeExecutionResult.Output,
			"code_execution_result", part.CodeExecutionResult)
	default:
		entry["mime_type"] = "text/plain"
		entry["part_attributes"] = "{}"
	}
}

// normalizeFunctionResponse converts the SDK-owned fields of r and its nested
// media parts to the snake_case names that adk-python writes, while leaving
// caller-defined keys inside r.Response untouched. The bytes of inline media
// are left as []byte, so the sanitizer redacts and truncates UTF-8 data as
// text and replaces binary data with a placeholder.
func normalizeFunctionResponse(r *genai.FunctionResponse) map[string]any {
	if r == nil {
		return nil
	}
	out := map[string]any{}
	if r.ID != "" {
		out["id"] = r.ID
	}
	if r.Name != "" {
		out["name"] = r.Name
	}
	if len(r.Response) > 0 {
		out["response"] = r.Response
	}
	if r.WillContinue != nil {
		out["will_continue"] = *r.WillContinue
	}
	if r.Scheduling != "" {
		out["scheduling"] = r.Scheduling
	}
	if len(r.Parts) > 0 {
		parts := make([]map[string]any, 0, len(r.Parts))
		for _, p := range r.Parts {
			if p == nil {
				continue
			}
			partMap := map[string]any{}
			if b := p.InlineData; b != nil {
				blobMap := map[string]any{}
				if b.MIMEType != "" {
					blobMap["mime_type"] = b.MIMEType
				}
				if len(b.Data) > 0 {
					// Not base64: that would hide the data from the sanitizer.
					blobMap["data"] = b.Data
				}
				if b.DisplayName != "" {
					blobMap["display_name"] = b.DisplayName
				}
				partMap["inline_data"] = blobMap
			}
			if f := p.FileData; f != nil {
				fileMap := map[string]any{}
				if f.FileURI != "" {
					fileMap["file_uri"] = f.FileURI
				}
				if f.MIMEType != "" {
					fileMap["mime_type"] = f.MIMEType
				}
				if f.DisplayName != "" {
					fileMap["display_name"] = f.DisplayName
				}
				partMap["file_data"] = fileMap
			}
			parts = append(parts, partMap)
		}
		if len(parts) > 0 {
			out["parts"] = parts
		}
	}
	return out
}

// queue records a payload for upload to the entry that is about to be
// appended to c.parts.
func (c *partsCollector) queue(u pendingUpload) {
	u.slot = len(c.parts)
	c.pending = append(c.pending, u)
}

// addText writes a text part. If it is too long to keep inline and an upload
// is possible, the entry holds the truncated text until the upload replaces
// it with a preview and a reference.
func (c *partsCollector) addText(entry map[string]any, text string, contentOrd, index int) {
	var contentLost bool
	if c.opts.redact {
		text, contentLost = sanitizeRawText(text)
	}
	shown, lengthTruncated := truncateString(text, c.opts.maxLen)
	truncated := contentLost || lengthTruncated
	entry["mime_type"] = "text/plain"
	entry["text"] = shown
	if c.opts.offload && (lengthTruncated || len(text) > c.opts.inlineLimit) {
		if reserved, ok := c.reserve(len(text)); ok {
			c.queue(pendingUpload{
				contentOrd:        contentOrd,
				partIdx:           index,
				mimeType:          "text/plain",
				isText:            true,
				text:              text,
				fallbackTruncated: truncated,
				reserved:          reserved,
			})
			c.truncated = c.truncated || contentLost
			c.nonUploadTruncated = c.nonUploadTruncated || contentLost
			return
		}
	}
	c.truncated = c.truncated || truncated
	c.nonUploadTruncated = c.nonUploadTruncated || truncated
}

// addInlineData writes a binary part. The bytes are copied if they are to be
// uploaded; otherwise they are left out and a placeholder stands in for them.
// The MIME type is written as the blob has it, except that offloading, which
// needs a content type for the object, records defaultBinaryMIMEType for a
// blob that has none.
func (c *partsCollector) addInlineData(entry map[string]any, blob *genai.Blob, contentOrd, index int) {
	entry["mime_type"] = blob.MIMEType
	entry["text"] = inlineDataText
	if c.opts.offload {
		mimeType := cmp.Or(blob.MIMEType, defaultBinaryMIMEType)
		entry["mime_type"] = mimeType
		if reserved, ok := c.reserve(len(blob.Data)); ok {
			c.queue(pendingUpload{
				contentOrd: contentOrd,
				partIdx:    index,
				mimeType:   mimeType,
				data:       bytes.Clone(blob.Data),
				reserved:   reserved,
			})
		} else {
			entry["text"] = uploadFailedText
		}
	}
}

// addFileData writes a part that already refers to external data. When
// offloading is enabled, Cloud Storage gs:// files also receive an object_ref
// (with details "{}" and ConnectionID as authorizer) so BigQuery ObjectRef
// functions can inspect existing GCS URIs alongside offloaded ones; note that
// adk-python leaves object_ref NULL on EXTERNAL_URI parts. A URI that had to
// be redacted names no real object, so it gets none.
func (c *partsCollector) addFileData(entry map[string]any, file *genai.FileData) {
	uri := file.FileURI
	var changed bool
	if c.opts.redact {
		uri, changed = sanitizeURI(uri)
	}
	entry["mime_type"] = file.MIMEType
	entry["uri"] = uri
	entry["storage_mode"] = storageModeExternalURI
	c.truncated = c.truncated || changed
	c.nonUploadTruncated = c.nonUploadTruncated || changed
	if c.opts.offload && !changed && strings.HasPrefix(uri, "gs://") {
		entry["object_ref"] = objectRef(uri, "{}", c.opts.connectionID)
	}
}

// addStructured writes a part that stands for a structured object: a short
// text, and the object itself under key in part_attributes.
func (c *partsCollector) addStructured(entry map[string]any, mimeType, text, key string, value any) {
	shown, truncated := sanitizeAndTruncateText(text, c.opts.maxLen, c.opts.redact)
	entry["mime_type"] = mimeType
	entry["text"] = shown
	c.truncated = c.truncated || truncated
	attrs, attrsTruncated := formatPartAttributes(map[string]any{key: value}, c.opts.maxLen, c.opts.redact)
	entry["part_attributes"] = attrs
	c.truncated = c.truncated || attrsTruncated
	c.nonUploadTruncated = c.nonUploadTruncated || truncated || attrsTruncated
}

// summaryPiece is one part's contribution to a message summary. Text parts
// whose Cloud Storage upload succeeds are omitted when the summary is rebuilt.
type summaryPiece struct {
	partIdx   int
	isText    bool
	text      string
	truncated bool
	omitted   bool
}

// messageSummary holds the per-part summary pieces of one genai.Content at
// contentOrd so its summary string can be rendered before or after uploads.
type messageSummary struct {
	contentOrd int
	role       string
	hasRole    bool
	isSystem   bool
	isResponse bool
	pieces     []summaryPiece
}

func newMessageSummary(content *genai.Content, contentOrd, maxLength int, redact bool) *messageSummary {
	m := &messageSummary{contentOrd: contentOrd}
	if content == nil {
		return m
	}
	for i, part := range content.Parts {
		switch {
		case part == nil, part.FileData != nil, part.InlineData != nil, part.FunctionCall != nil:
		case part.Text != "":
			shown, t := sanitizeAndTruncateText(part.Text, maxLength, redact)
			m.pieces = append(m.pieces, summaryPiece{partIdx: i, isText: true, text: shown, truncated: t})
		case part.FunctionResponse != nil:
			name, t := sanitizeAndTruncateText(orUnknown(part.FunctionResponse.Name), maxLength, redact)
			m.pieces = append(m.pieces, summaryPiece{partIdx: i, text: "Function response: " + name, truncated: t})
		case part.ExecutableCode != nil:
			code, t := sanitizeAndTruncateText(part.ExecutableCode.Code, maxLength, redact)
			m.pieces = append(m.pieces, summaryPiece{
				partIdx: i,
				text: fmt.Sprintf("Executable code (%s): %s",
					orUnknown(string(part.ExecutableCode.Language)), code),
				truncated: t,
			})
		case part.CodeExecutionResult != nil:
			output, t := sanitizeAndTruncateText(part.CodeExecutionResult.Output, maxLength, redact)
			m.pieces = append(m.pieces, summaryPiece{
				partIdx: i,
				text: fmt.Sprintf("Code execution result (%s): %s",
					orUnknown(string(part.CodeExecutionResult.Outcome)), output),
				truncated: t,
			})
		}
	}
	return m
}

func newResponseMessageSummary(content *genai.Content, contentOrd, maxLength int, redact bool) *messageSummary {
	m := &messageSummary{contentOrd: contentOrd, isResponse: true}
	if content == nil {
		return m
	}
	for i, part := range content.Parts {
		var t bool
		piece := "other"
		isText := false
		switch {
		case part == nil:
		case part.Text != "":
			var text string
			text, t = sanitizeAndTruncateText(part.Text, maxLength, redact)
			piece = "text: '" + text + "'"
			isText = true
		case part.FunctionCall != nil:
			piece = "call: " + part.FunctionCall.Name
		case part.FunctionResponse != nil:
			piece = "resp: " + part.FunctionResponse.Name
		case part.ExecutableCode != nil:
			var code string
			code, t = sanitizeAndTruncateText(part.ExecutableCode.Code, maxLength, redact)
			piece = fmt.Sprintf("Executable code (%s): %s", orUnknown(string(part.ExecutableCode.Language)), code)
		case part.CodeExecutionResult != nil:
			var output string
			output, t = sanitizeAndTruncateText(part.CodeExecutionResult.Output, maxLength, redact)
			piece = fmt.Sprintf("Code execution result (%s): %s", orUnknown(string(part.CodeExecutionResult.Outcome)), output)
		}
		m.pieces = append(m.pieces, summaryPiece{partIdx: i, isText: isText, text: piece, truncated: t})
	}
	return m
}

func (m *messageSummary) render(maxLength int) (string, bool) {
	if m == nil {
		return "", false
	}
	if len(m.pieces) == 0 {
		if m.isResponse {
			return "None", false
		}
		return "", false
	}
	active := make([]string, 0, len(m.pieces))
	truncated := false
	for _, p := range m.pieces {
		if p.omitted {
			continue
		}
		active = append(active, p.text)
		truncated = truncated || p.truncated
	}
	joined := strings.Join(active, " | ")
	if m.isResponse {
		return joined, truncated
	}
	summary, t := truncateString(joined, maxLength)
	return summary, truncated || t
}

// contentSummaryBuilder retains the message/part ordinals of a row that has
// pending text uploads so resolveUploads can rebuild the content column if any
// text upload fails.
type contentSummaryBuilder struct {
	isLLMRequest   bool
	isLLMResponse  bool
	contentDenied  bool
	maxLen         int
	redact         bool
	roleTruncated  bool
	otherTruncated bool
	needsRebuild   bool
	usage          map[string]any
	messages       []*messageSummary
}

func (b *contentSummaryBuilder) omitTextFromSummary(contentOrd, partIdx int) {
	for _, m := range b.messages {
		if m.contentOrd != contentOrd {
			continue
		}
		for i := range m.pieces {
			if m.pieces[i].partIdx == partIdx && m.pieces[i].isText {
				if !m.pieces[i].omitted || m.pieces[i].truncated {
					m.pieces[i].omitted = true
					m.pieces[i].truncated = false
					b.needsRebuild = true
				}
			}
		}
	}
}

func (b *contentSummaryBuilder) retainInlineSummary(contentOrd, partIdx int, fallbackText string, fallbackTruncated bool) {
	for _, m := range b.messages {
		if m.contentOrd != contentOrd {
			continue
		}
		for i := range m.pieces {
			if m.pieces[i].partIdx == partIdx && m.pieces[i].isText {
				b.needsRebuild = true
				m.pieces[i].omitted = false
				if m.isResponse {
					m.pieces[i].text = "text: '" + fallbackText + "'"
				} else {
					m.pieces[i].text = fallbackText
				}
				m.pieces[i].truncated = fallbackTruncated
			}
		}
	}
}

func (b *contentSummaryBuilder) hasFailedUploadTruncated() bool {
	for _, m := range b.messages {
		for _, p := range m.pieces {
			if !p.omitted && p.truncated {
				return true
			}
		}
	}
	return false
}

func (b *contentSummaryBuilder) buildContent() (map[string]any, bool) {
	if b.isLLMResponse {
		content := map[string]any{}
		truncated := false
		if len(b.messages) > 0 {
			content["response"], truncated = b.messages[0].render(b.maxLen)
		}
		if b.usage != nil {
			content["usage"] = b.usage
		}
		return content, truncated
	}
	if !b.isLLMRequest {
		if len(b.messages) == 0 {
			return map[string]any{"text_summary": ""}, false
		}
		summary, t := b.messages[0].render(b.maxLen)
		return map[string]any{"text_summary": summary}, t
	}
	content := map[string]any{}
	truncated := b.roleTruncated
	var prompt []map[string]any
	for _, m := range b.messages {
		summary, t := m.render(b.maxLen)
		truncated = truncated || t
		if m.isSystem {
			content["system_prompt"] = summary
		} else if m.hasRole {
			prompt = append(prompt, map[string]any{"role": m.role, "content": summary})
		}
	}
	if len(prompt) > 0 {
		content["prompt"] = prompt
	}
	return content, truncated
}

func (b *contentSummaryBuilder) applyToRow(row map[string]any) {
	if !b.needsRebuild {
		return
	}
	summaryTruncated := false
	jsonTruncated := false
	if !b.contentDenied {
		contentVal, st := b.buildContent()
		summaryTruncated = st
		encoded, err := json.Marshal(contentVal)
		if err != nil {
			row[columnContent] = strconv.Quote(unsupportedObjPlaceholder)
			jsonTruncated = true
		} else {
			row[columnContent] = string(encoded)
		}
	}
	row["is_truncated"] = b.otherTruncated || summaryTruncated || jsonTruncated || b.hasFailedUploadTruncated()
}

func formatPartAttributes(v any, maxLength int, redact bool) (string, bool) {
	b, truncated, err := smartTruncate(v, maxLength, redact)
	if err != nil {
		return strconv.Quote(unsupportedObjPlaceholder), true
	}
	return string(b), truncated
}

func sanitizeAndTruncateText(raw string, maxLength int, redact bool) (string, bool) {
	var contentLost bool
	if redact {
		raw, contentLost = sanitizeRawText(raw)
	}
	truncatedStr, lengthTruncated := truncateString(raw, maxLength)
	return truncatedStr, contentLost || lengthTruncated
}

// contentSummary returns the summary that adk-python writes to
// content.text_summary: the text parts, each truncated, and descriptions of
// function responses and code parts, joined by " | " and truncated again.
// Function calls and inline or file data are left out.
func contentSummary(content *genai.Content, maxLength int, redact bool) (string, bool) {
	return newMessageSummary(content, 0, maxLength, redact).render(maxLength)
}

// formatResponseContent renders model output the way adk-python writes
// content.response: one entry per part joined by " | ", or "None" if there
// are no parts.
func formatResponseContent(content *genai.Content, maxLength int, redact bool) (string, bool) {
	return newResponseMessageSummary(content, 0, maxLength, redact).render(maxLength)
}

func orUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}

// SmartTruncate converts obj to JSON. Strings longer than maxLength runes are
// truncated (maxLength < 0 means no limit) and the values of sensitive keys,
// such as api_key or password, are replaced with "[REDACTED]". Cyclic,
// overly deep and unsupported values are replaced with placeholders. The
// bool result reports whether any content was truncated or dropped.
func SmartTruncate(obj any, maxLength int) ([]byte, bool, error) {
	return smartTruncate(obj, maxLength, true)
}

// smartTruncate is SmartTruncate with sensitive-key redaction optional.
func smartTruncate(obj any, maxLength int, redact bool) ([]byte, bool, error) {
	if obj == nil {
		return []byte("null"), false, nil
	}
	v, truncated := sanitizeValue(obj, maxLength, redact)
	b, err := json.Marshal(v)
	return b, truncated, err
}

// sanitizeValue converts obj into a tree of JSON-native values (nil, bool,
// string, json.Number, integers, finite floats, map[string]any and []any)
// that json.Marshal always accepts, applying the rules of SmartTruncate.
func sanitizeValue(obj any, maxLength int, redact bool) (any, bool) {
	s := sanitizer{
		maxLength: maxLength,
		redact:    redact,
		budget:    &nodeBudget{remaining: maxSanitizeNodes},
		visiting:  make(map[visitKey]struct{}),
	}
	return s.walk(reflect.ValueOf(obj), 0)
}

type sanitizer struct {
	maxLength int
	redact    bool
	budget    *nodeBudget
	// visiting holds the references on the path from the root to the value
	// being converted; meeting one again means the value contains itself.
	visiting map[visitKey]struct{}
}

type visitKey struct {
	ptr uintptr
	typ reflect.Type
	len int
}

func (s *sanitizer) walk(v reflect.Value, depth int) (any, bool) {
	for v.IsValid() && v.Kind() == reflect.Interface {
		if v.IsNil() {
			if !s.budget.take() {
				return sanitizeBudgetPlaceholder, true
			}
			return nil, false
		}
		v = v.Elem()
	}
	if !v.IsValid() || (v.Kind() == reflect.Pointer && v.IsNil()) {
		if !s.budget.take() {
			return sanitizeBudgetPlaceholder, true
		}
		return nil, false
	}
	if v.Kind() == reflect.Pointer {
		if depth >= maxSanitizeDepth {
			return maxDepthPlaceholder, true
		}
		key := visitKey{ptr: v.Pointer(), typ: v.Type()}
		if !s.enter(key) {
			return circularRefPlaceholder, false
		}
		defer s.leave(key)
		return s.walk(v.Elem(), depth)
	}
	if !s.budget.take() {
		return sanitizeBudgetPlaceholder, true
	}
	if depth >= maxSanitizeDepth {
		return maxDepthPlaceholder, true
	}
	if v.Type() == jsonNumberType {
		return s.number(v.String())
	}
	if out, truncated, ok := s.marshaled(v, depth); ok {
		return out, truncated
	}

	switch v.Kind() {
	case reflect.String:
		return s.stringValue(v.String(), depth)
	case reflect.Bool:
		return v.Bool(), false
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int(), false
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint(), false
	case reflect.Float32, reflect.Float64:
		return floatValue(v), false
	case reflect.Map:
		if v.IsNil() {
			return nil, false
		}
		key := visitKey{ptr: v.Pointer(), typ: v.Type()}
		if !s.enter(key) {
			return circularRefPlaceholder, false
		}
		defer s.leave(key)
		return s.walkMap(v, depth)
	case reflect.Slice:
		if v.IsNil() {
			return nil, false
		}
		if v.Type().Elem().Kind() == reflect.Uint8 {
			return s.bytes(v.Bytes(), depth)
		}
		if v.Len() > 0 {
			key := visitKey{ptr: v.Pointer(), typ: v.Type(), len: v.Len()}
			if !s.enter(key) {
				return circularRefPlaceholder, false
			}
			defer s.leave(key)
		}
		return s.walkList(v, depth)
	case reflect.Array:
		return s.walkList(v, depth)
	case reflect.Struct:
		return s.walkStruct(v, depth)
	default:
		// Complex numbers, channels, functions and unsafe pointers have no
		// JSON form.
		return unsupportedObjPlaceholder, true
	}
}

func (s *sanitizer) enter(key visitKey) bool {
	if _, ok := s.visiting[key]; ok {
		return false
	}
	s.visiting[key] = struct{}{}
	return true
}

func (s *sanitizer) leave(key visitKey) {
	delete(s.visiting, key)
}

// number passes a valid JSON number through unchanged, so that integers
// beyond 2^53 keep their exact value, and writes anything else as a string.
func (s *sanitizer) number(n string) (any, bool) {
	if isJSONNumber(n) {
		return json.Number(n), false
	}
	return truncateString(n, s.maxLength)
}

func isJSONNumber(n string) bool {
	if n == "" || strings.TrimSpace(n) != n {
		return false
	}
	if c := n[0]; c != '-' && (c < '0' || c > '9') {
		return false
	}
	return json.Valid([]byte(n))
}

// floatValue returns v as a JSON-encodable value. JSON has no NaN or
// infinities, so those are written as strings.
func floatValue(v reflect.Value) any {
	f := v.Float()
	switch {
	case math.IsNaN(f):
		return "NaN"
	case math.IsInf(f, 1):
		return "+Inf"
	case math.IsInf(f, -1):
		return "-Inf"
	case v.Kind() == reflect.Float32:
		return float32(f)
	default:
		return f
	}
}

// stringValue redacts sensitive keys inside JSON-encoded strings and scrubs
// credential patterns when redact is set, then truncates the result to
// maxLength runes.
func (s *sanitizer) stringValue(str string, depth int) (string, bool) {
	var contentLost bool
	if s.redact {
		str, _, contentLost = redactJSONTextWithBudget(str, depth, s.budget)
		scrubbed := scrubErrorText(str)
		if scrubbed != str {
			str = scrubbed
			contentLost = true
		}
	}
	truncatedStr, lengthTruncated := truncateString(str, s.maxLength)
	return truncatedStr, contentLost || lengthTruncated
}

// bytes writes UTF-8 data as a string, so that key redaction and truncation
// apply to it, and replaces binary data with a placeholder.
func (s *sanitizer) bytes(b []byte, depth int) (any, bool) {
	if !utf8.Valid(b) {
		return binaryDataPlaceholder, true
	}
	return s.stringValue(string(b), depth)
}

// marshaled converts a value that implements json.Marshaler or
// encoding.TextMarshaler through that method, as encoding/json would, and
// then sanitizes the result. ok is false if v implements neither.
func (s *sanitizer) marshaled(v reflect.Value, depth int) (out any, truncated, ok bool) {
	if m, found := implementer(v, jsonMarshalerType); found {
		b, err := safeCall(m.(json.Marshaler).MarshalJSON)
		if err != nil {
			return unsupportedObjPlaceholder, true, true
		}
		var decoded any
		if err := unmarshalJSONPreserveNumbers(b, &decoded); err != nil {
			return unsupportedObjPlaceholder, true, true
		}
		out, truncated = s.walk(reflect.ValueOf(decoded), depth)
		return out, truncated, true
	}
	if m, found := implementer(v, textMarshalerType); found {
		b, err := safeCall(m.(encoding.TextMarshaler).MarshalText)
		if err != nil {
			return unsupportedObjPlaceholder, true, true
		}
		out, truncated = s.stringValue(string(b), depth)
		return out, truncated, true
	}
	return nil, false, false
}

// implementer returns v, or its address when only the pointer type has the
// method, as an interface value implementing iface.
func implementer(v reflect.Value, iface reflect.Type) (any, bool) {
	if v.Type().Implements(iface) && v.CanInterface() {
		return v.Interface(), true
	}
	if v.Kind() != reflect.Pointer && v.CanAddr() && reflect.PointerTo(v.Type()).Implements(iface) {
		if a := v.Addr(); a.CanInterface() {
			return a.Interface(), true
		}
	}
	return nil, false
}

// safeCall runs a marshaling method, turning a panic into an error.
func safeCall(f func() ([]byte, error)) (b []byte, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = errors.New("marshal method panicked")
		}
	}()
	return f()
}

// unmarshalJSONPreserveNumbers decodes b into v, keeping numbers as
// json.Number so that integers beyond 2^53 are not rounded to float64.
func unmarshalJSONPreserveNumbers(b []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	if err := dec.Decode(v); err != nil {
		return err
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return errors.New("unexpected data after JSON value")
	}
	return nil
}

func (s *sanitizer) walkMap(v reflect.Value, depth int) (any, bool) {
	out := make(map[string]any, min(v.Len(), maxContainerAlloc))
	truncated := false
	unsupportedKeys := 0
	iter := v.MapRange()
	for iter.Next() {
		if s.budget.remaining <= 0 {
			out[sanitizeBudgetPlaceholder] = sanitizeBudgetPlaceholder
			truncated = true
			break
		}
		k, ok := mapKeyString(iter.Key())
		if !ok {
			unsupportedKeys++
			k = unsupportedKeyPlaceholder
			if unsupportedKeys > 1 {
				k = fmt.Sprintf("[UNSUPPORTED_KEY_%d]", unsupportedKeys)
			}
			truncated = true
		} else if s.redact && isSensitiveKey(k) {
			if !s.budget.take() {
				out[sanitizeBudgetPlaceholder] = sanitizeBudgetPlaceholder
				truncated = true
				break
			}
			out[k] = redactedValue
			continue
		}
		val, t := s.walk(iter.Value(), depth+1)
		if str, ok := val.(string); ok && str == sanitizeBudgetPlaceholder && s.budget.remaining <= 0 {
			out[sanitizeBudgetPlaceholder] = sanitizeBudgetPlaceholder
			truncated = true
			break
		}
		out[k] = val
		truncated = truncated || t
	}
	return out, truncated
}

// mapKeyString returns the JSON object key for a map key: strings, text
// marshalers and integers as encoding/json writes them, plus booleans and
// floats in their JSON form. ok is false for any other key.
func mapKeyString(k reflect.Value) (string, bool) {
	for k.Kind() == reflect.Interface {
		if k.IsNil() {
			return "null", true
		}
		k = k.Elem()
	}
	if k.Kind() == reflect.String {
		return k.String(), true
	}
	if m, found := implementer(k, textMarshalerType); found {
		if k.Kind() == reflect.Pointer && k.IsNil() {
			return "", true
		}
		b, err := safeCall(m.(encoding.TextMarshaler).MarshalText)
		return string(b), err == nil
	}
	switch k.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.FormatInt(k.Int(), 10), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return strconv.FormatUint(k.Uint(), 10), true
	case reflect.Bool:
		return strconv.FormatBool(k.Bool()), true
	case reflect.Float32, reflect.Float64:
		return strconv.FormatFloat(k.Float(), 'g', -1, k.Type().Bits()), true
	default:
		return "", false
	}
}

func (s *sanitizer) walkList(v reflect.Value, depth int) (any, bool) {
	n := v.Len()
	out := make([]any, 0, min(n, maxContainerAlloc))
	truncated := false
	for i := range n {
		if s.budget.remaining <= 0 {
			out = append(out, sanitizeBudgetPlaceholder)
			truncated = true
			break
		}
		val, t := s.walk(v.Index(i), depth+1)
		out = append(out, val)
		truncated = truncated || t
		if str, ok := val.(string); ok && str == sanitizeBudgetPlaceholder && s.budget.remaining <= 0 {
			break
		}
	}
	return out, truncated
}

// walkStruct converts a struct to a map the way encoding/json encodes it:
// exported fields under their json tag names, "-" and omitempty honored, and
// the fields of untagged embedded structs promoted, with shallower fields
// winning name conflicts.
func (s *sanitizer) walkStruct(v reflect.Value, depth int) (any, bool) {
	out := make(map[string]any)
	truncated := false
	level := []reflect.Value{v}
	seen := map[uintptr]bool{}
	seenNames := map[string]bool{}
	budgetStopped := false
	for embedDepth := 0; len(level) > 0 && embedDepth < maxSanitizeDepth && !budgetStopped; embedDepth++ {
		var next []reflect.Value
		for _, sv := range level {
			if budgetStopped {
				break
			}
			t := sv.Type()
			for i := range t.NumField() {
				if s.budget.remaining <= 0 {
					out[sanitizeBudgetPlaceholder] = sanitizeBudgetPlaceholder
					truncated = true
					budgetStopped = true
					break
				}
				f := t.Field(i)
				name, opts, skip := parseJSONTag(f)
				if skip {
					continue
				}
				fv := sv.Field(i)
				if f.Anonymous && name == "" && isStructOrStructPointer(f.Type) {
					if embedded, ok := embeddedStruct(f, fv, seen); ok {
						next = append(next, embedded)
					}
					continue
				}
				if !f.IsExported() {
					continue
				}
				if name == "" {
					name = f.Name
				}
				if seenNames[name] {
					continue
				}
				seenNames[name] = true
				if (opts.omitEmpty && isEmptyValue(fv)) || (opts.omitZero && fv.IsZero()) {
					continue
				}
				if s.redact && isSensitiveKey(name) {
					if !s.budget.take() {
						out[sanitizeBudgetPlaceholder] = sanitizeBudgetPlaceholder
						truncated = true
						budgetStopped = true
						break
					}
					out[name] = redactedValue
					continue
				}
				val, tr := s.walk(fv, depth+1)
				if str, ok := val.(string); ok && str == sanitizeBudgetPlaceholder && s.budget.remaining <= 0 {
					out[sanitizeBudgetPlaceholder] = sanitizeBudgetPlaceholder
					truncated = true
					budgetStopped = true
					break
				}
				out[name] = val
				truncated = truncated || tr
			}
		}
		level = next
	}
	return out, truncated
}

func isStructOrStructPointer(t reflect.Type) bool {
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	return t.Kind() == reflect.Struct
}

// embeddedStruct returns the value of an untagged embedded struct field,
// whose fields are promoted. Embedded pointers that are nil or already
// promoted contribute no fields. As in encoding/json, an embedded pointer to
// an unexported struct type still promotes its exported fields.
func embeddedStruct(f reflect.StructField, fv reflect.Value, seen map[uintptr]bool) (reflect.Value, bool) {
	if f.Type.Kind() != reflect.Pointer {
		return fv, true
	}
	if fv.IsNil() || seen[fv.Pointer()] {
		return reflect.Value{}, false
	}
	seen[fv.Pointer()] = true
	return fv.Elem(), true
}

type tagOptions struct {
	omitEmpty bool
	omitZero  bool
}

// parseJSONTag returns the name and options of a field's json tag; skip
// reports a field tagged "-".
func parseJSONTag(f reflect.StructField) (name string, opts tagOptions, skip bool) {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "", opts, true
	}
	name, rest, _ := strings.Cut(tag, ",")
	for rest != "" {
		var opt string
		opt, rest, _ = strings.Cut(rest, ",")
		switch opt {
		case "omitempty":
			opts.omitEmpty = true
		case "omitzero":
			opts.omitZero = true
		}
	}
	return name, opts, false
}

// isEmptyValue reports whether v is empty in the sense of encoding/json's
// omitempty option.
func isEmptyValue(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64,
		reflect.Interface, reflect.Pointer:
		return v.IsZero()
	default:
		return false
	}
}

// truncateString shortens s to maxLength runes followed by truncationSuffix.
// A negative maxLength means no limit.
func truncateString(s string, maxLength int) (string, bool) {
	if maxLength < 0 {
		return s, false
	}
	r := []rune(s)
	if len(r) <= maxLength {
		return s, false
	}
	return string(r[:maxLength]) + truncationSuffix, true
}
