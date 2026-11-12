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

package setmodelresponsetool

import (
	"maps"
	"math"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/sessionutils"
)

// The string forms pydantic's lax mode accepts for an int and a float field.
var (
	intPattern   = regexp.MustCompile(`^[+-]?[0-9]+(_[0-9]+)*(\.0+)?$`)
	floatPattern = regexp.MustCompile(`^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$`)
)

// normalize checks v against s as pydantic's lax mode does and returns the
// answer. It appends one issue per failure and never stops at the first. A nil
// s is untyped and accepts anything. normalize never removes a nil; the object
// rule decides which keys stay.
func normalize(v any, s *genai.Schema, loc []string, issues *[]issue) any {
	if s == nil || (v == nil && isNullable(s)) {
		return v
	}
	var out any
	var kind string
	switch upperType(s) {
	case genai.TypeString:
		if len(s.Enum) > 0 {
			return normalizeEnum(v, s, loc, issues)
		}
		out, kind = v, ""
		if _, ok := v.(string); !ok {
			kind = "string_type"
		}
	case genai.TypeInteger:
		out, kind = coerceInteger(v)
	case genai.TypeNumber:
		out, kind = coerceNumber(v)
	case genai.TypeBoolean:
		out, kind = coerceBool(v)
	case genai.TypeArray:
		return normalizeArray(v, s, loc, issues)
	case genai.TypeObject:
		if len(s.Properties) > 0 {
			return normalizeRecord(v, s, loc, issues)
		}
		out, kind = v, ""
		if _, ok := v.(map[string]any); !ok {
			kind = "dict_type"
		}
	default:
		return v
	}
	if kind != "" {
		addIssue(issues, loc, kind, message(kind))
	}
	return out
}

// normalizeRecord applies the object rule: required keys must be present,
// absent keys get their default, a null for a nullable key is kept only when
// the key is required, and keys the schema does not list are dropped.
func normalizeRecord(v any, s *genai.Schema, loc []string, issues *[]issue) any {
	m, ok := v.(map[string]any)
	if !ok {
		addIssue(issues, loc, "model_type", "Input should be a valid dictionary or instance of "+titleOf(s))
		return v
	}
	out := make(map[string]any, len(s.Properties))
	for _, name := range propertyOrder(s) {
		prop := s.Properties[name]
		val, present := m[name]
		required := slices.Contains(s.Required, name)
		switch {
		case !present && required:
			addIssue(issues, child(loc, name), "missing", "Field required")
		case !present && prop != nil && prop.Default != nil:
			out[name] = sessionutils.CloneValue(prop.Default)
		case !present:
		case prop == nil:
			// A nil property schema is untyped, like a nil Items.
			out[name] = val
		case val == nil && isNullable(prop):
			// Python drops every None (model_dump(exclude_none=True)); a
			// required key keeps its null because workflow output schemas
			// mark *T fields required and nullable and expect the key.
			if required {
				out[name] = nil
			}
		default:
			out[name] = normalize(val, prop, child(loc, name), issues)
		}
	}
	return out
}

// propertyOrder lists the properties in PropertyOrdering order, then the rest
// by name, so issues come out in a stable order as pydantic's field order.
func propertyOrder(s *genai.Schema) []string {
	names := make([]string, 0, len(s.Properties))
	for _, name := range s.PropertyOrdering {
		if _, ok := s.Properties[name]; ok && !slices.Contains(names, name) {
			names = append(names, name)
		}
	}
	for _, name := range slices.Sorted(maps.Keys(s.Properties)) {
		if !slices.Contains(s.PropertyOrdering, name) {
			names = append(names, name)
		}
	}
	return names
}

func normalizeArray(v any, s *genai.Schema, loc []string, issues *[]issue) any {
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Slice && rv.Kind() != reflect.Array {
		addIssue(issues, loc, "list_type", message("list_type"))
		return v
	}
	out := make([]any, rv.Len())
	for i := range out {
		out[i] = normalize(rv.Index(i).Interface(), s.Items, child(loc, strconv.Itoa(i)), issues)
	}
	return out
}

func normalizeEnum(v any, s *genai.Schema, loc []string, issues *[]issue) any {
	if str, ok := v.(string); ok && slices.Contains(s.Enum, str) {
		return v
	}
	quoted := make([]string, len(s.Enum))
	for i, e := range s.Enum {
		quoted[i] = "'" + e + "'"
	}
	msg := "Input should be " + quoted[len(quoted)-1]
	if len(quoted) > 1 {
		msg = "Input should be " + strings.Join(quoted[:len(quoted)-1], ", ") + " or " + quoted[len(quoted)-1]
	}
	addIssue(issues, loc, "enum", msg)
	return v
}

// coerceInteger returns the value for an INTEGER field and the issue kind, or
// "" when v is valid. Numbers keep their Go type; a string or a bool becomes a
// float64, the type of a JSON-decoded number.
func coerceInteger(v any) (any, string) {
	if _, _, ok := integer(v); ok {
		return v, ""
	}
	if f, ok := float(v); ok {
		switch {
		case math.IsNaN(f) || math.IsInf(f, 0):
			return v, "finite_number"
		case f != math.Trunc(f):
			return v, "int_from_float"
		case math.Abs(f) >= 1<<63:
			return v, "int_parsing_size"
		}
		return v, ""
	}
	switch x := v.(type) {
	case bool:
		return boolNumber(x), ""
	case string:
		t := strings.TrimSpace(x)
		if !intPattern.MatchString(t) {
			return v, "int_parsing"
		}
		f, err := strconv.ParseFloat(t, 64)
		if err != nil || math.IsInf(f, 0) {
			return v, "int_parsing_size"
		}
		return f, ""
	}
	return v, "int_type"
}

// coerceNumber returns the value for a NUMBER field and the issue kind, or ""
// when v is valid. NaN and infinity are refused, as numbers and as strings,
// because the answer must marshal as JSON.
func coerceNumber(v any) (any, string) {
	if _, _, ok := integer(v); ok {
		return v, ""
	}
	if f, ok := float(v); ok {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return v, "finite_number"
		}
		return v, ""
	}
	switch x := v.(type) {
	case bool:
		return boolNumber(x), ""
	case string:
		t := strings.TrimSpace(x)
		if !floatPattern.MatchString(t) {
			return v, "float_parsing"
		}
		f, err := strconv.ParseFloat(t, 64)
		if err != nil || math.IsInf(f, 0) {
			return v, "float_parsing"
		}
		return f, ""
	}
	return v, "float_type"
}

// coerceBool returns the value for a BOOLEAN field and the issue kind, or ""
// when v is valid.
func coerceBool(v any) (any, string) {
	if n, fits, ok := integer(v); ok {
		if !fits {
			return v, "bool_type"
		}
		return intBool(v, n)
	}
	if f, ok := float(v); ok {
		if math.IsNaN(f) || math.IsInf(f, 0) || f != math.Trunc(f) || f < -(1<<63) || f >= 1<<63 {
			return v, "bool_type"
		}
		return intBool(v, int64(f))
	}
	switch x := v.(type) {
	case bool:
		return x, ""
	case string:
		switch strings.ToLower(x) {
		case "true", "t", "yes", "y", "on", "1":
			return true, ""
		case "false", "f", "no", "n", "off", "0":
			return false, ""
		}
		return v, "bool_parsing"
	}
	return v, "bool_type"
}

func intBool(v any, n int64) (any, string) {
	switch n {
	case 0:
		return false, ""
	case 1:
		return true, ""
	}
	return v, "bool_parsing"
}

// integer reports whether v holds a Go integer, and its value as an int64.
// fits is false only for an unsigned value beyond math.MaxInt64.
func integer(v any) (n int64, fits, ok bool) {
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rv.Int(), true, true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		u := rv.Uint()
		if u > math.MaxInt64 {
			return 0, false, true
		}
		return int64(u), true, true
	}
	return 0, false, false
}

// float reports whether v holds a Go floating-point number, and its value.
func float(v any) (float64, bool) {
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Float32, reflect.Float64:
		return rv.Float(), true
	}
	return 0, false
}

func boolNumber(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func isNullable(s *genai.Schema) bool {
	return s.Nullable != nil && *s.Nullable
}

// child returns loc extended by name, in a new backing array so that sibling
// locations never share one.
func child(loc []string, name string) []string {
	return append(slices.Clip(loc), name)
}

func addIssue(issues *[]issue, loc []string, kind, msg string) {
	*issues = append(*issues, issue{loc: strings.Join(loc, "."), msg: msg, kind: kind})
}

// message is pydantic's message for an issue kind whose message is fixed.
func message(kind string) string {
	switch kind {
	case "string_type":
		return "Input should be a valid string"
	case "int_type":
		return "Input should be a valid integer"
	case "int_parsing":
		return "Input should be a valid integer, unable to parse string as an integer"
	case "int_from_float":
		return "Input should be a valid integer, got a number with a fractional part"
	case "int_parsing_size":
		return "Unable to parse input string as an integer, exceeded maximum size"
	case "finite_number":
		return "Input should be a finite number"
	case "float_type":
		return "Input should be a valid number"
	case "float_parsing":
		return "Input should be a valid number, unable to parse string as a number"
	case "bool_type":
		return "Input should be a valid boolean"
	case "bool_parsing":
		return "Input should be a valid boolean, unable to interpret input"
	case "list_type":
		return "Input should be a valid list"
	case "dict_type":
		return "Input should be a valid dictionary"
	}
	return kind
}
