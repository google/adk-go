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

package toolresultlimit

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestNew(t *testing.T) {
	for _, limit := range []int{-1, 0, 1, minResultBytes - 1, minResultBytes, 4096} {
		t.Run(fmt.Sprint(limit), func(t *testing.T) {
			p, err := New(Config{MaxResultBytes: limit})
			if limit < minResultBytes {
				if err == nil || p != nil {
					t.Fatalf("New(%d) = (%v, %v), want nil plugin and error", limit, p, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if p.AfterToolCallback() == nil {
				t.Fatal("missing AfterToolCallback")
			}
		})
	}
}

func TestAfterToolUnchanged(t *testing.T) {
	exact := map[string]any{"value": strings.Repeat("x", 400)}
	limit := len(encode(t, exact))
	tests := []struct {
		name   string
		result map[string]any
		err    error
	}{
		{name: "nil"},
		{name: "empty", result: map[string]any{}},
		{name: "small typed values", result: map[string]any{"value": int64(9007199254740993)}},
		{name: "exact limit", result: exact},
		{name: "tool error with oversized result", result: map[string]any{"value": strings.Repeat("x", 4096)}, err: errors.New("tool failed")},
		{name: "tool error with unencodable result", result: map[string]any{"value": make(chan int)}, err: errors.New("tool failed")},
	}
	p, err := New(Config{MaxResultBytes: limit})
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := maps.Clone(tt.result)
			got, err := p.AfterToolCallback()(nil, nil, nil, tt.result, tt.err)
			if got != nil || err != nil {
				t.Fatal("callback replaced the result or returned an error")
			}
			if !reflect.DeepEqual(tt.result, original) {
				t.Fatal("callback modified the original result")
			}
		})
	}
}

func TestAfterToolTruncates(t *testing.T) {
	large := strings.Repeat("x", 4096)
	manyKeys := make(map[string]any)
	for i := range 300 {
		manyKeys[fmt.Sprint(i)] = i
	}
	tests := []struct {
		name   string
		result map[string]any
	}{
		{name: "string", result: map[string]any{"value": large}},
		{name: "nested typed slice", result: map[string]any{"data": map[string]any{"items": []string{large}}}},
		{name: "large key", result: map[string]any{large: true}},
		{name: "many small fields", result: manyKeys},
		{name: "unicode", result: map[string]any{"value": strings.Repeat("中文🙂", 500)}},
		{name: "escapes", result: map[string]any{"value": strings.Repeat("\"\\\n\t<>&\u2028", 300)}},
		{name: "error payload", result: map[string]any{"error": large}},
		{name: "structured error", result: map[string]any{"error": map[string]any{"message": large}}},
		{name: "null error", result: map[string]any{"error": nil, "data": large}},
		{name: "existing truncation fields", result: map[string]any{"truncated": false, "output": large}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := encode(t, tt.result)
			for limit := minResultBytes; limit < minResultBytes+64; limit++ {
				p, err := New(Config{MaxResultBytes: limit})
				if err != nil {
					t.Fatal(err)
				}
				got, err := p.AfterToolCallback()(nil, nil, nil, tt.result, nil)
				if err != nil {
					t.Fatal(err)
				}
				checkTruncated(t, got, original, limit, tt.result["error"] != nil)
			}
			if !bytes.Equal(original, encode(t, tt.result)) {
				t.Fatal("original result was mutated")
			}
		})
	}
}

func TestOneByteOverLimit(t *testing.T) {
	result := map[string]any{"value": strings.Repeat("x", 400)}
	original := encode(t, result)
	limit := len(original) - 1
	p, err := New(Config{MaxResultBytes: limit})
	if err != nil {
		t.Fatal(err)
	}
	got, err := p.AfterToolCallback()(nil, nil, nil, result, nil)
	if err != nil {
		t.Fatal(err)
	}
	checkTruncated(t, got, original, limit, false)
}

func TestEncodingError(t *testing.T) {
	cycle := map[string]any{}
	cycle["self"] = cycle
	for name, result := range map[string]map[string]any{
		"unsupported type": {"value": make(chan int)},
		"NaN":              {"value": math.NaN()},
		"cycle":            cycle,
	} {
		t.Run(name, func(t *testing.T) {
			p, err := New(Config{MaxResultBytes: 1024})
			if err != nil {
				t.Fatal(err)
			}
			got, err := p.AfterToolCallback()(nil, nil, nil, result, nil)
			if got != nil {
				t.Fatal("callback returned a result for unencodable input")
			}
			if err == nil {
				t.Fatal("callback returned no encoding error")
			}
			var typeErr *json.UnsupportedTypeError
			var valueErr *json.UnsupportedValueError
			if !errors.As(err, &typeErr) && !errors.As(err, &valueErr) {
				t.Fatalf("JSON encoding error is not in the error chain: %T", err)
			}
		})
	}
}

type failingMarshaler struct {
	err error
}

func (m failingMarshaler) MarshalJSON() ([]byte, error) { return nil, m.err }

func TestEncodingErrorDoesNotExposeResult(t *testing.T) {
	const payload = "private-tool-content"
	marshalErr := errors.New(strings.Repeat(payload, 100))
	for name, value := range map[string]any{
		"invalid number":   json.Number(payload),
		"custom marshaler": failingMarshaler{err: marshalErr},
	} {
		t.Run(name, func(t *testing.T) {
			p, err := New(Config{MaxResultBytes: 256})
			if err != nil {
				t.Fatal(err)
			}
			result, err := p.AfterToolCallback()(nil, nil, nil, map[string]any{"value": value}, nil)
			if result != nil || err == nil {
				t.Fatal("expected an encoding error without a replacement result")
			}
			if strings.Contains(err.Error(), payload) {
				t.Fatal("encoding error contains tool data")
			}
			if len(encode(t, map[string]any{"error": err.Error()})) > 256 {
				t.Fatal("encoding error response exceeds the limit")
			}
			if errors.Unwrap(err) == nil {
				t.Fatal("encoding error lost its cause")
			}
			if name == "custom marshaler" && !errors.Is(err, marshalErr) {
				t.Fatal("custom marshaler error is not in the error chain")
			}
		})
	}
}

func FuzzResultLimit(f *testing.F) {
	f.Add(strings.Repeat("中文🙂\"\\\n<>&", 300), uint16(0), false)
	f.Add(strings.Repeat("failure", 300), uint16(13), true)
	f.Add("\xff\xfe", uint16(256), false)
	f.Fuzz(func(t *testing.T, value string, extra uint16, failure bool) {
		limit := minResultBytes + int(extra)
		result := map[string]any{"value": value}
		if failure {
			result["error"] = "tool failed"
		}
		original := encode(t, result)
		p, err := New(Config{MaxResultBytes: limit})
		if err != nil {
			t.Fatal(err)
		}
		got, err := p.AfterToolCallback()(nil, nil, nil, result, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(original) <= limit {
			if got != nil {
				t.Fatal("result within limit was replaced")
			}
			return
		}
		checkTruncated(t, got, original, limit, failure)
	})
}

func encode(t *testing.T, value any) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("json.Marshal failed: %T", err)
	}
	return data
}

func checkTruncated(t *testing.T, result map[string]any, original []byte, limit int, failure bool) {
	t.Helper()
	if size := len(encode(t, result)); size > limit {
		t.Fatalf("encoded result size = %d, exceeds %d", size, limit)
	}
	output, ok := result["output"].(map[string]any)
	if !ok || output["truncated"] != true || output["original_size_bytes"] != len(original) {
		t.Fatal("missing or incorrect truncation metadata")
	}
	if message, ok := output["message"].(string); !ok || message == "" {
		t.Fatal("missing guidance for the model")
	}
	preview, ok := output["preview"].(string)
	if !ok || preview == "" || !utf8.ValidString(preview) || !bytes.HasPrefix(original, []byte(preview)) {
		t.Fatal("preview is empty, invalid UTF-8, or not a prefix of the original JSON")
	}
	if (result["error"] != nil) != failure {
		t.Errorf("error marker present = %t, want %t", result["error"] != nil, failure)
	}
	_, next := utf8.DecodeRune(original[len(preview):])
	longerOutput := maps.Clone(output)
	longerOutput["preview"] = string(original[:len(preview)+next])
	longerResult := maps.Clone(result)
	longerResult["output"] = longerOutput
	if len(encode(t, longerResult)) <= limit {
		t.Errorf("preview could retain another rune within %d bytes", limit)
	}
}
