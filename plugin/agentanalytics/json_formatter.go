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
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"google.golang.org/genai"
)

// FormatContentParts formats Content parts into a map array for BigQuery logging.
func FormatContentParts(content *genai.Content, maxLength int) []map[string]any {
	var parts []map[string]any
	if content == nil || content.Parts == nil {
		return parts
	}

	for i, part := range content.Parts {
		partObj := make(map[string]any)
		partObj["part_index"] = i
		partObj["storage_mode"] = "INLINE"

		if part.Text != "" {
			partObj["mime_type"] = "text/plain"
			text, _ := truncateString(part.Text, maxLength)
			partObj["text"] = text
		} else if part.InlineData != nil {
			partObj["mime_type"] = part.InlineData.MIMEType
			partObj["text"] = "[BINARY DATA]"
		} else if part.FileData != nil {
			partObj["mime_type"] = part.FileData.MIMEType
			partObj["uri"] = part.FileData.FileURI
			partObj["storage_mode"] = "EXTERNAL_URI"
		}
		parts = append(parts, partObj)
	}

	return parts
}

// SmartTruncate recursively truncates long strings inside a map or slice and returns JSON bytes.
func SmartTruncate(obj any, maxLength int) ([]byte, bool, error) {
	if obj == nil {
		return []byte("null"), false, nil
	}

	truncatedObj, truncated, err := recursiveSmartTruncate(obj, maxLength)
	if err != nil {
		truncatedObj = obj
		truncated = false
	}

	j, err := json.Marshal(truncatedObj)
	return j, truncated, err
}

func recursiveSmartTruncate(obj any, maxLength int) (any, bool, error) {
	if obj == nil {
		return nil, false, nil
	}
	truncated := false

	switch v := obj.(type) {
	case string:
		s, t := truncateString(v, maxLength)
		return s, t, nil
	case map[string]any:
		newMap := make(map[string]any)
		for k, val := range v {
			tVal, t, _ := recursiveSmartTruncate(val, maxLength)
			newMap[k] = tVal
			truncated = truncated || t
		}
		return newMap, truncated, nil
	case []any:
		newArr := make([]any, len(v))
		for i, val := range v {
			tVal, t, _ := recursiveSmartTruncate(val, maxLength)
			newArr[i] = tVal
			truncated = truncated || t
		}
		return newArr, truncated, nil
	default:
		// Traverse into other complex types via reflection
		val := reflect.ValueOf(obj)

		// Unpack interfaces and pointers
		for val.Kind() == reflect.Ptr || val.Kind() == reflect.Interface {
			if val.IsNil() {
				return nil, false, nil
			}
			val = val.Elem()
		}

		switch val.Kind() {
		case reflect.Struct:
			newMap := make(map[string]any)
			typ := val.Type()
			for i := 0; i < val.NumField(); i++ {
				field := typ.Field(i)
				if !field.IsExported() {
					continue
				}

				name := field.Name
				tag := field.Tag.Get("json")
				if tag == "-" {
					continue
				}
				if tag != "" {
					parts := strings.Split(tag, ",")
					if parts[0] != "" {
						name = parts[0]
					}
				}

				tVal, t, _ := recursiveSmartTruncate(val.Field(i).Interface(), maxLength)
				newMap[name] = tVal
				truncated = truncated || t
			}
			return newMap, truncated, nil

		case reflect.Slice, reflect.Array:
			newArr := make([]any, val.Len())
			for i := 0; i < val.Len(); i++ {
				tVal, t, _ := recursiveSmartTruncate(val.Index(i).Interface(), maxLength)
				newArr[i] = tVal
				truncated = truncated || t
			}
			return newArr, truncated, nil

		case reflect.Map:
			newMap := make(map[string]any)
			for _, key := range val.MapKeys() {
				// Best effort for map keys
				kStr := fmt.Sprintf("%v", key.Interface())
				tVal, t, _ := recursiveSmartTruncate(val.MapIndex(key).Interface(), maxLength)
				newMap[kStr] = tVal
				truncated = truncated || t
			}
			return newMap, truncated, nil

		case reflect.String:
			s, t := truncateString(val.String(), maxLength)
			return s, t, nil

		default:
			// numbers, bools, etc.
			return obj, false, nil
		}
	}
}

func truncateString(s string, maxLength int) (string, bool) {
	if maxLength < 0 {
		return s, false
	}
	r := []rune(s)
	if len(r) <= maxLength {
		return s, false
	}
	return string(r[:maxLength]) + "...[truncated]", true
}
