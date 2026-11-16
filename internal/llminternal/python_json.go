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

package llminternal

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
)

// encodePythonJSON returns v as JSON text in the form adk-python's
// json.dumps(v, ensure_ascii=False) writes: ", " between items, ": " after
// keys, non-ASCII characters (U+2028 and U+2029 included) and "<", ">", "&"
// written literally, floats in Python's repr form (1e-05, 1e+16, -0.0), and
// NaN and the infinities as NaN, Infinity and -Infinity.
//
// v may be any value json.Marshal accepts, and may also hold NaN or infinite
// float64 and float32 values in the map[string]any and []any values genai
// decodes arguments to; typed slices, maps and structs are written like their
// generic JSON form. Object keys are sorted, where Python keeps insertion
// order, which a Go map does not have. A number json.Marshal writes without a
// fraction or exponent stays an integer, since genai decodes every JSON number
// to float64 and cannot tell the model's 3 from 3.0; at 1e16 and above, where
// Python's repr switches to exponent form, a value float64 holds exactly is
// written as that float.
//
// It returns json.Marshal's error for a value it cannot encode, such as a
// channel or a complex number.
func encodePythonJSON(v any) (string, error) {
	generic, err := toGenericJSON(v)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	writePythonJSON(&b, generic)
	return b.String(), nil
}

// nonFiniteFloat is a NaN or infinite float, which encoding/json refuses and
// json.dumps writes by name.
type nonFiniteFloat float64

// String returns f as json.dumps writes it.
func (f nonFiniteFloat) String() string {
	switch {
	case math.IsNaN(float64(f)):
		return "NaN"
	case f > 0:
		return "Infinity"
	default:
		return "-Infinity"
	}
}

// toGenericJSON returns v as encoding/json decodes its JSON text with
// UseNumber, except that a NaN or infinite float, at the top or inside
// map[string]any and []any values, becomes a nonFiniteFloat.
func toGenericJSON(v any) (any, error) {
	switch v := v.(type) {
	case map[string]any:
		if v != nil {
			return genericMap(v)
		}
	case []any:
		if v != nil {
			return genericList(v)
		}
	case float64:
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nonFiniteFloat(v), nil
		}
	case float32:
		if f := float64(v); math.IsNaN(f) || math.IsInf(f, 0) {
			return nonFiniteFloat(f), nil
		}
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var generic any
	if err := dec.Decode(&generic); err != nil {
		return nil, fmt.Errorf("decode marshaled value: %w", err)
	}
	return generic, nil
}

func genericMap(m map[string]any) (map[string]any, error) {
	out := make(map[string]any, len(m))
	for key, item := range m {
		g, err := toGenericJSON(item)
		if err != nil {
			return nil, err
		}
		out[key] = g
	}
	return out, nil
}

func genericList(l []any) ([]any, error) {
	out := make([]any, len(l))
	for i, item := range l {
		g, err := toGenericJSON(item)
		if err != nil {
			return nil, err
		}
		out[i] = g
	}
	return out, nil
}

// writePythonJSON writes v, a value toGenericJSON returns, to b in json.dumps
// form.
func writePythonJSON(b *strings.Builder, v any) {
	switch v := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		b.WriteString(strconv.FormatBool(v))
	case string:
		writePythonJSONString(b, v)
	case json.Number:
		b.WriteString(pythonJSONNumber(v))
	case nonFiniteFloat:
		b.WriteString(v.String())
	case []any:
		b.WriteByte('[')
		for i, item := range v {
			if i > 0 {
				b.WriteString(", ")
			}
			writePythonJSON(b, item)
		}
		b.WriteByte(']')
	case map[string]any:
		b.WriteByte('{')
		for i, key := range slices.Sorted(maps.Keys(v)) {
			if i > 0 {
				b.WriteString(", ")
			}
			writePythonJSONString(b, key)
			b.WriteString(": ")
			writePythonJSON(b, v[key])
		}
		b.WriteByte('}')
	}
}

// writePythonJSONString writes s quoted as json.dumps(s, ensure_ascii=False)
// does: only the quote, the backslash and control characters are escaped.
func writePythonJSONString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

// pythonJSONNumber returns n, as json.Marshal wrote it, in the form Python's
// repr gives the same float: exponent form when the decimal exponent is below
// -4 or at least 16, positional form otherwise. A negative zero is a float
// in Python, which has no negative integer zero, so it is written -0.0.
func pythonJSONNumber(n json.Number) string {
	s := n.String()
	f, err := n.Float64()
	if err != nil {
		return s
	}
	if f == 0 && math.Signbit(f) {
		return "-0.0"
	}
	if !strings.ContainsAny(s, ".eE") {
		// An integer below 1e16, or one float64 cannot hold exactly, is an
		// integer in Python too.
		if f > -1e16 && f < 1e16 || strconv.FormatFloat(f, 'f', -1, 64) != strings.TrimPrefix(s, "+") {
			return s
		}
	}
	e := strconv.FormatFloat(f, 'e', -1, 64)
	exp, err := strconv.Atoi(e[strings.IndexByte(e, 'e')+1:])
	if err != nil || exp < -4 || exp >= 16 {
		return e
	}
	return strconv.FormatFloat(f, 'f', -1, 64)
}
