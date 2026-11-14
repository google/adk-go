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

package toolinternal

import (
	"cmp"
	"reflect"
	"slices"

	"google.golang.org/genai"
)

// maxMediaContainerDepth is the deepest container whose entries are searched
// for media: the value a tool returns, and one container inside it, as
// adk-python's _MAX_MEDIA_CONTAINER_DEPTH (flows/llm_flows/tools/_caller.py:63).
const maxMediaContainerDepth = 1

// MediaPart converts v to a function response part when v is a media part: a
// non-nil *genai.Part or a genai.Part whose InlineData has non-nil Data and a
// MIME type, or whose FileData has a URI and a MIME type. Inline data is
// checked first. DisplayName is not copied, as adk-python's
// _as_function_response_part does not copy it (_caller.py:174-193). It
// returns nil for any other value, including a part without a MIME type.
func MediaPart(v any) *genai.FunctionResponsePart {
	var p *genai.Part
	switch t := v.(type) {
	case *genai.Part:
		p = t
	case genai.Part:
		p = &t
	}
	if p == nil {
		return nil
	}
	if b := p.InlineData; b != nil && b.Data != nil && b.MIMEType != "" {
		return genai.NewFunctionResponsePartFromBytes(b.Data, b.MIMEType)
	}
	if f := p.FileData; f != nil && f.FileURI != "" && f.MIMEType != "" {
		return genai.NewFunctionResponsePartFromURI(f.FileURI, f.MIMEType)
	}
	return nil
}

// ExtractMedia moves the media parts (see [MediaPart]) of a tool result into
// function response parts, as adk-python's _extract_multimodal_parts does
// (_caller.py:196-268). It searches result itself and the containers directly
// inside it, never deeper. A container is a map with string keys, visited in
// sorted key order because a Go map has no insertion order, or a slice or
// array whose element kind is not uint8, visited in index order; []byte and
// json.RawMessage are values, not containers.
//
// It returns what is left of result and the extracted parts. A media part is
// removed from its container, and an inner container left empty is removed
// too. The remaining map is a new map[string]any and the remaining list a new
// []any; result is not modified. When result is itself a media part, or
// nothing remains of it, remaining is an empty map[string]any. When result
// carries no media, ExtractMedia returns result unchanged and nil parts.
func ExtractMedia(result any) (remaining any, parts []*genai.FunctionResponsePart) {
	return extractMedia(result, 0)
}

func extractMedia(result any, depth int) (any, []*genai.FunctionResponsePart) {
	if p := MediaPart(result); p != nil {
		return map[string]any{}, []*genai.FunctionResponsePart{p}
	}
	v := reflect.ValueOf(result)
	var parts []*genai.FunctionResponsePart
	var remaining any
	empty := false
	switch {
	case isStringMap(v):
		kept := map[string]any{}
		keys := v.MapKeys()
		slices.SortFunc(keys, func(a, b reflect.Value) int {
			return cmp.Compare(a.String(), b.String())
		})
		for _, k := range keys {
			if keep, val := extractFromEntry(v.MapIndex(k).Interface(), &parts, depth); keep {
				kept[k.String()] = val
			}
		}
		remaining, empty = kept, len(kept) == 0
	case isList(v):
		kept := []any{}
		for i := range v.Len() {
			if keep, val := extractFromEntry(v.Index(i).Interface(), &parts, depth); keep {
				kept = append(kept, val)
			}
		}
		remaining, empty = kept, len(kept) == 0
	default:
		return result, nil
	}
	if len(parts) == 0 {
		return result, nil
	}
	if empty {
		return map[string]any{}, parts
	}
	return remaining, parts
}

// extractFromEntry is _extract_media_from_entry (_caller.py:196-224): it
// appends the media of one container entry to parts and reports whether the
// entry is kept and what is left of it.
func extractFromEntry(value any, parts *[]*genai.FunctionResponsePart, depth int) (bool, any) {
	if p := MediaPart(value); p != nil {
		*parts = append(*parts, p)
		return false, nil
	}
	v := reflect.ValueOf(value)
	if depth >= maxMediaContainerDepth || !(isStringMap(v) || isList(v)) {
		return true, value
	}
	rest, nested := extractMedia(value, depth+1)
	if len(nested) == 0 {
		return true, value
	}
	*parts = append(*parts, nested...)
	if isEmptyContainer(rest) {
		return false, nil
	}
	return true, rest
}

// CarriesPart reports whether v is a non-nil *genai.Part or a genai.Part, media
// or not, or a container (as [ExtractMedia] defines it) that holds one at any
// depth of nested containers. A container already visited is skipped, so a
// cyclic value returns false rather than overflowing the stack.
func CarriesPart(v any) bool {
	return carriesPart(reflect.ValueOf(v), map[visitKey]bool{})
}

// visitKey identifies a map, or a slice by its first element and length, so
// that a shorter slice of the same array does not hide the longer one.
type visitKey struct {
	ptr uintptr
	len int
}

var partType = reflect.TypeFor[genai.Part]()

func carriesPart(v reflect.Value, seen map[visitKey]bool) bool {
	for v.Kind() == reflect.Interface {
		v = v.Elem()
	}
	if !v.IsValid() {
		return false
	}
	switch {
	case v.Type() == partType:
		return true
	case v.Kind() == reflect.Pointer && v.Type().Elem() == partType:
		return !v.IsNil()
	case isStringMap(v):
		key := visitKey{ptr: uintptr(v.UnsafePointer()), len: -1}
		if seen[key] {
			return false
		}
		seen[key] = true
		for it := v.MapRange(); it.Next(); {
			if carriesPart(it.Value(), seen) {
				return true
			}
		}
	case isList(v):
		if v.Kind() == reflect.Slice {
			key := visitKey{ptr: uintptr(v.UnsafePointer()), len: v.Len()}
			if seen[key] {
				return false
			}
			seen[key] = true
		}
		for i := range v.Len() {
			if carriesPart(v.Index(i), seen) {
				return true
			}
		}
	}
	return false
}

func isStringMap(v reflect.Value) bool {
	return v.Kind() == reflect.Map && v.Type().Key().Kind() == reflect.String
}

func isList(v reflect.Value) bool {
	return (v.Kind() == reflect.Slice || v.Kind() == reflect.Array) && v.Type().Elem().Kind() != reflect.Uint8
}

func isEmptyContainer(v any) bool {
	switch t := v.(type) {
	case map[string]any:
		return len(t) == 0
	case []any:
		return len(t) == 0
	}
	return false
}
