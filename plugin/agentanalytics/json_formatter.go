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

// FormatContentParts formats Content parts into a map array for BigQuery logging.
func FormatContentParts(content *genai.Content, maxLength int) []map[string]any {
	parts, _ := formatContentParts(content, maxLength, true)
	return parts
}

// formatContentParts is FormatContentParts that also reports whether any
// part lost content to truncation or URI redaction.
func formatContentParts(content *genai.Content, maxLength int, redact bool) ([]map[string]any, bool) {
	if content == nil || len(content.Parts) == 0 {
		return nil, false
	}
	parts := make([]map[string]any, 0, len(content.Parts))
	truncated := false
	for i, part := range content.Parts {
		partObj := map[string]any{
			"part_index":   i,
			"storage_mode": "INLINE",
		}
		switch {
		case part == nil:
		case part.Text != "":
			text, t := sanitizeAndTruncateText(part.Text, maxLength, redact)
			partObj["mime_type"] = "text/plain"
			partObj["text"] = text
			truncated = truncated || t
		case part.InlineData != nil:
			partObj["mime_type"] = part.InlineData.MIMEType
			partObj["text"] = inlineDataText
		case part.FileData != nil:
			uri := part.FileData.FileURI
			var changed bool
			if redact {
				uri, changed = sanitizeURI(uri)
			}
			partObj["mime_type"] = part.FileData.MIMEType
			partObj["uri"] = uri
			partObj["storage_mode"] = "EXTERNAL_URI"
			truncated = truncated || changed
		case part.FunctionCall != nil:
			name, nameTrunc := sanitizeAndTruncateText(orUnknown(part.FunctionCall.Name), maxLength, redact)
			attrs, attrTrunc := formatPartAttributes(map[string]any{
				"function_name": part.FunctionCall.Name,
				"function_call": part.FunctionCall,
			}, maxLength, redact)
			partObj["mime_type"] = "application/json"
			partObj["text"] = "Function: " + name
			partObj["part_attributes"] = attrs
			truncated = truncated || nameTrunc || attrTrunc
		case part.FunctionResponse != nil:
			name, nameTrunc := sanitizeAndTruncateText(orUnknown(part.FunctionResponse.Name), maxLength, redact)
			attrs, attrTrunc := formatPartAttributes(map[string]any{"function_response": part.FunctionResponse}, maxLength, redact)
			partObj["mime_type"] = "application/json"
			partObj["text"] = "Function response: " + name
			partObj["part_attributes"] = attrs
			truncated = truncated || nameTrunc || attrTrunc
		case part.ExecutableCode != nil:
			code, codeTrunc := sanitizeAndTruncateText(part.ExecutableCode.Code, maxLength, redact)
			attrs, attrTrunc := formatPartAttributes(map[string]any{"executable_code": part.ExecutableCode}, maxLength, redact)
			partObj["mime_type"] = "text/plain"
			partObj["text"] = code
			partObj["part_attributes"] = attrs
			truncated = truncated || codeTrunc || attrTrunc
		case part.CodeExecutionResult != nil:
			output, outTrunc := sanitizeAndTruncateText(part.CodeExecutionResult.Output, maxLength, redact)
			attrs, attrTrunc := formatPartAttributes(map[string]any{"code_execution_result": part.CodeExecutionResult}, maxLength, redact)
			partObj["mime_type"] = "text/plain"
			partObj["text"] = output
			partObj["part_attributes"] = attrs
			truncated = truncated || outTrunc || attrTrunc
		default:
			partObj["mime_type"] = "text/plain"
			partObj["part_attributes"] = "{}"
		}
		parts = append(parts, partObj)
	}
	return parts, truncated
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
	if content == nil {
		return "", false
	}
	var pieces []string
	truncated := false
	for _, part := range content.Parts {
		switch {
		case part == nil, part.FileData != nil, part.InlineData != nil:
		case part.Text != "":
			text, t := sanitizeAndTruncateText(part.Text, maxLength, redact)
			pieces = append(pieces, text)
			truncated = truncated || t
		case part.FunctionCall != nil:
		case part.FunctionResponse != nil:
			name, t := sanitizeAndTruncateText(orUnknown(part.FunctionResponse.Name), maxLength, redact)
			pieces = append(pieces, "Function response: "+name)
			truncated = truncated || t
		case part.ExecutableCode != nil:
			code, t := sanitizeAndTruncateText(part.ExecutableCode.Code, maxLength, redact)
			pieces = append(pieces, fmt.Sprintf("Executable code (%s): %s",
				orUnknown(string(part.ExecutableCode.Language)), code))
			truncated = truncated || t
		case part.CodeExecutionResult != nil:
			output, t := sanitizeAndTruncateText(part.CodeExecutionResult.Output, maxLength, redact)
			pieces = append(pieces, fmt.Sprintf("Code execution result (%s): %s",
				orUnknown(string(part.CodeExecutionResult.Outcome)), output))
			truncated = truncated || t
		}
	}
	summary, t := truncateString(strings.Join(pieces, " | "), maxLength)
	return summary, truncated || t
}

// formatResponseContent renders model output the way adk-python writes
// content.response: one entry per part joined by " | ", or "None" if there
// are no parts.
func formatResponseContent(content *genai.Content, maxLength int, redact bool) (string, bool) {
	if content == nil || len(content.Parts) == 0 {
		return "None", false
	}
	pieces := make([]string, 0, len(content.Parts))
	truncated := false
	for _, part := range content.Parts {
		var t bool
		piece := "other"
		switch {
		case part == nil:
		case part.Text != "":
			var text string
			text, t = sanitizeAndTruncateText(part.Text, maxLength, redact)
			piece = "text: '" + text + "'"
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
		pieces = append(pieces, piece)
		truncated = truncated || t
	}
	return strings.Join(pieces, " | "), truncated
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
// whose fields are promoted. Embedded pointers that are nil, already
// promoted or of an unexported type contribute no fields.
func embeddedStruct(f reflect.StructField, fv reflect.Value, seen map[uintptr]bool) (reflect.Value, bool) {
	if f.Type.Kind() != reflect.Pointer {
		return fv, true
	}
	if !f.IsExported() || fv.IsNil() || seen[fv.Pointer()] {
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
