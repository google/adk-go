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
	"encoding/base64"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

func TestFormatContentParts(t *testing.T) {
	tests := []struct {
		name      string
		content   *genai.Content
		maxLength int
		want      []map[string]any
	}{
		{
			name:      "nil content",
			content:   nil,
			maxLength: 10,
			want:      []map[string]any{},
		},
		{
			name: "empty parts",
			content: &genai.Content{
				Parts: nil,
			},
			maxLength: 10,
			want:      []map[string]any{},
		},
		{
			name: "various parts",
			content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "this is a very long text"},
					{InlineData: &genai.Blob{MIMEType: "image/png"}},
					{FileData: &genai.FileData{MIMEType: "video/mp4", FileURI: "gs://bucket/file.mp4"}},
					{Text: "short"},
				},
			},
			maxLength: 10,
			want: []map[string]any{
				{
					"part_index":   0,
					"storage_mode": "INLINE",
					"mime_type":    "text/plain",
					"text":         "this is a ...[TRUNCATED]",
				},
				{
					"part_index":   1,
					"storage_mode": "INLINE",
					"mime_type":    "image/png",
					"text":         "[BINARY DATA]",
				},
				{
					"part_index":   2,
					"storage_mode": "EXTERNAL_URI",
					"mime_type":    "video/mp4",
					"uri":          "gs://bucket/file.mp4",
				},
				{
					"part_index":   3,
					"storage_mode": "INLINE",
					"mime_type":    "text/plain",
					"text":         "short",
				},
			},
		},
		{
			name: "file URI with credentials",
			content: &genai.Content{
				Parts: []*genai.Part{
					{FileData: &genai.FileData{MIMEType: "image/png", FileURI: "https://example.com/i.png?key=abc&alt=media"}},
				},
			},
			maxLength: 10,
			want: []map[string]any{
				{
					"part_index":   0,
					"storage_mode": "EXTERNAL_URI",
					"mime_type":    "image/png",
					"uri":          "https://example.com/i.png?key=%5BREDACTED%5D&alt=media",
				},
			},
		},
		{
			name: "text part with JSON credentials sanitized before truncation",
			content: &genai.Content{
				Parts: []*genai.Part{
					{Text: `{"api_key":"secret","note":"0123456789"}`},
				},
			},
			maxLength: 25,
			want: []map[string]any{
				{
					"part_index":   0,
					"storage_mode": "INLINE",
					"mime_type":    "text/plain",
					"text":         `{"api_key":"[REDACTED]","...[TRUNCATED]`,
				},
			},
		},
		{
			name: "structured function and code execution parts",
			content: &genai.Content{
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "lookup", Args: map[string]any{"api_key": "secret", "q": "weather"}}},
					{FunctionResponse: &genai.FunctionResponse{ID: "call-1", Name: "lookup", Response: map[string]any{"token": "tok", "answer": "sunny"}}},
					{ExecutableCode: &genai.ExecutableCode{Language: genai.LanguagePython, Code: "print(1)"}},
					{CodeExecutionResult: &genai.CodeExecutionResult{Outcome: genai.OutcomeOK, Output: "1\n"}},
					{},
				},
			},
			maxLength: 200,
			want: []map[string]any{
				{
					"part_index":      0,
					"storage_mode":    "INLINE",
					"mime_type":       "application/json",
					"text":            "Function: lookup",
					"part_attributes": `{"function_call":{"args":{"api_key":"[REDACTED]","q":"weather"},"id":"call-1","name":"lookup"},"function_name":"lookup"}`,
				},
				{
					"part_index":      1,
					"storage_mode":    "INLINE",
					"mime_type":       "application/json",
					"text":            "Function response: lookup",
					"part_attributes": `{"function_response":{"id":"call-1","name":"lookup","response":{"answer":"sunny","token":"[REDACTED]"}}}`,
				},
				{
					"part_index":      2,
					"storage_mode":    "INLINE",
					"mime_type":       "text/plain",
					"text":            "print(1)",
					"part_attributes": `{"executable_code":{"code":"print(1)","language":"PYTHON"}}`,
				},
				{
					"part_index":      3,
					"storage_mode":    "INLINE",
					"mime_type":       "text/plain",
					"text":            "1\n",
					"part_attributes": `{"code_execution_result":{"outcome":"OUTCOME_OK","output":"1\n"}}`,
				},
				{
					"part_index":      4,
					"storage_mode":    "INLINE",
					"mime_type":       "text/plain",
					"part_attributes": `{}`,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FormatContentParts(tt.content, tt.maxLength)
			if len(got) != len(tt.want) {
				t.Fatalf("FormatContentParts() len = %v, want %v", len(got), len(tt.want))
			}
			for i := range got {
				if !reflect.DeepEqual(got[i], tt.want[i]) {
					t.Errorf("FormatContentParts() got[%d] = %v, want %v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

type testStruct struct {
	ExportedString      string `json:"exported_string"`
	IgnoredField        string `json:"-"`
	UnexportedField     string
	ExportedInt         int
	Nested              *nestedStruct `json:"nested,omitempty"`
	ExportedStringNoTag string
}

type nestedStruct struct {
	Value string
}

func TestSmartTruncate(t *testing.T) {
	tests := []struct {
		name      string
		obj       any
		maxLength int
		wantStr   string
		wantTrunc bool
	}{
		{
			name:      "nil",
			obj:       nil,
			maxLength: 10,
			wantStr:   `null`,
			wantTrunc: false,
		},
		{
			name:      "short string",
			obj:       "short",
			maxLength: 10,
			wantStr:   `"short"`,
			wantTrunc: false,
		},
		{
			name:      "simple string",
			obj:       "this is a very long string",
			maxLength: 10,
			wantStr:   `"this is a ...[TRUNCATED]"`,
			wantTrunc: true,
		},
		{
			name: "map with strings",
			obj: map[string]any{
				"key1": "long long long string",
				"key2": "short",
			},
			maxLength: 10,
			wantStr:   `{"key1":"long long ...[TRUNCATED]","key2":"short"}`,
			wantTrunc: true,
		},
		{
			name:      "slice of strings",
			obj:       []any{"long long long string", "short"},
			maxLength: 10,
			wantStr:   `["long long ...[TRUNCATED]","short"]`,
			wantTrunc: true,
		},
		{
			name: "complex struct",
			obj: &testStruct{
				ExportedString:      "long string here",
				IgnoredField:        "should not appear",
				UnexportedField:     "should not appear",
				ExportedInt:         42,
				ExportedStringNoTag: "another long string",
				Nested: &nestedStruct{
					Value: "nested long string",
				},
			},
			maxLength: 10,
			wantStr:   `{"ExportedInt":42,"ExportedStringNoTag":"another lo...[TRUNCATED]","UnexportedField":"should not...[TRUNCATED]","exported_string":"long strin...[TRUNCATED]","nested":{"Value":"nested lon...[TRUNCATED]"}}`,
			wantTrunc: true,
		},
		{
			name: "typed slice",
			obj: []string{
				"very long string 1",
				"short",
			},
			maxLength: 10,
			wantStr:   `["very long ...[TRUNCATED]","short"]`,
			wantTrunc: true,
		},
		{
			name: "map with non-string keys",
			obj: map[int]string{
				1: "very long string 1",
			},
			maxLength: 10,
			wantStr:   `{"1":"very long ...[TRUNCATED]"}`,
			wantTrunc: true,
		},
		{
			name: "array",
			obj: [1]string{
				"very long string",
			},
			maxLength: 10,
			wantStr:   `["very long ...[TRUNCATED]"]`,
			wantTrunc: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotBytes, truncated, err := SmartTruncate(tt.obj, tt.maxLength)
			if err != nil {
				t.Fatalf("SmartTruncate() error = %v", err)
			}
			if truncated != tt.wantTrunc {
				t.Errorf("SmartTruncate() truncated = %v, want %v", truncated, tt.wantTrunc)
			}

			// We use json.Unmarshal to canonicalize the JSON before comparison because map key order is non-deterministic
			var gotObj, wantObj any
			if string(gotBytes) != tt.wantStr {
				// To handle non-deterministic map ordering, unmarshal then compare
				if err := json.Unmarshal(gotBytes, &gotObj); err != nil {
					t.Fatalf("Failed to unmarshal gotBytes (%s): %v", string(gotBytes), err)
				}
				if err := json.Unmarshal([]byte(tt.wantStr), &wantObj); err != nil {
					t.Fatalf("Failed to unmarshal wantStr (%s): %v", tt.wantStr, err)
				}

				if gotObj == nil || wantObj == nil || !reflect.DeepEqual(gotObj, wantObj) {
					t.Errorf("SmartTruncate() = %s, want %s", string(gotBytes), tt.wantStr)
				}
			}
		})
	}
}

type cyclicNode struct {
	Name string
	Next *cyclicNode
}

// panicMarshaler panics when it is marshaled.
type panicMarshaler struct{}

func (panicMarshaler) MarshalJSON() ([]byte, error) { panic("marshal failure with payload data") }

// errorMarshaler fails when it is marshaled.
type errorMarshaler struct{}

func (errorMarshaler) MarshalJSON() ([]byte, error) { return nil, errors.New("cannot marshal") }

type credentials struct {
	User   string `json:"user"`
	APIKey string `json:"api_key"`
	Token  string
	Unset  string `json:"password,omitempty"`
}

// nested returns depth maps nested in each other around the string "leaf".
func nested(depth int) any {
	var v any = "leaf"
	for range depth {
		v = map[string]any{"n": v}
	}
	return v
}

func TestSmartTruncate_Guards(t *testing.T) {
	cyclicMap := map[string]any{"a": 1}
	cyclicMap["self"] = cyclicMap
	cyclicSlice := make([]any, 1)
	cyclicSlice[0] = cyclicSlice
	node := &cyclicNode{Name: "a"}
	node.Next = node
	shared := map[string]any{"v": 1}

	tests := []struct {
		name      string
		obj       any
		want      string
		wantTrunc bool
	}{
		{name: "map cycle", obj: cyclicMap, want: `{"a":1,"self":"[CIRCULAR_REFERENCE]"}`},
		{name: "slice cycle", obj: cyclicSlice, want: `["[CIRCULAR_REFERENCE]"]`},
		{name: "pointer cycle", obj: node, want: `{"Name":"a","Next":"[CIRCULAR_REFERENCE]"}`},
		{name: "shared value is not a cycle", obj: map[string]any{"x": shared, "y": shared}, want: `{"x":{"v":1},"y":{"v":1}}`},
		{name: "int64 beyond 2^53", obj: int64(9007199254740993), want: `9007199254740993`},
		{name: "uint64 maximum", obj: uint64(math.MaxUint64), want: `18446744073709551615`},
		{name: "json.Number", obj: json.Number("12345678901234567890"), want: `12345678901234567890`},
		{name: "json.RawMessage", obj: json.RawMessage(`{"n":9007199254740993}`), want: `{"n":9007199254740993}`},
		{name: "NaN and infinities", obj: []any{math.NaN(), math.Inf(1), math.Inf(-1)}, want: `["NaN","+Inf","-Inf"]`},
		{name: "panicking MarshalJSON", obj: panicMarshaler{}, want: `"[UNSUPPORTED_OBJECT]"`, wantTrunc: true},
		{name: "failing MarshalJSON", obj: errorMarshaler{}, want: `"[UNSUPPORTED_OBJECT]"`, wantTrunc: true},
		{name: "function", obj: func() {}, want: `"[UNSUPPORTED_OBJECT]"`, wantTrunc: true},
		{name: "channel", obj: make(chan int), want: `"[UNSUPPORTED_OBJECT]"`, wantTrunc: true},
		{name: "binary bytes", obj: []byte{0xff, 0xfe}, want: `"[BINARY_DATA]"`, wantTrunc: true},
		{name: "UTF-8 bytes", obj: []byte("hello"), want: `"hello"`},
		{name: "unsupported map key", obj: map[[2]int]string{{1, 2}: "x"}, want: `{"[UNSUPPORTED_KEY]":"x"}`, wantTrunc: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, truncated, err := SmartTruncate(tt.obj, 100)
			if err != nil {
				t.Fatalf("SmartTruncate() error: %v", err)
			}
			if string(got) != tt.want {
				t.Errorf("SmartTruncate() = %s, want %s", got, tt.want)
			}
			if truncated != tt.wantTrunc {
				t.Errorf("SmartTruncate() truncated = %v, want %v", truncated, tt.wantTrunc)
			}
		})
	}
}

func TestSmartTruncate_DepthLimit(t *testing.T) {
	tests := []struct {
		name     string
		depth    int
		wantLeaf bool
	}{
		{name: "within the limit", depth: 10, wantLeaf: true},
		{name: "beyond the limit", depth: 60, wantLeaf: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, truncated, err := SmartTruncate(nested(tt.depth), 100)
			if err != nil {
				t.Fatalf("SmartTruncate() error: %v", err)
			}
			if hasLeaf := bytes.Contains(got, []byte(`"leaf"`)); hasLeaf != tt.wantLeaf {
				t.Errorf("SmartTruncate() kept the leaf = %v, want %v", hasLeaf, tt.wantLeaf)
			}
			if hasMarker := bytes.Contains(got, []byte(maxDepthPlaceholder)); hasMarker == tt.wantLeaf {
				t.Errorf("SmartTruncate() wrote %s = %v, want %v", maxDepthPlaceholder, hasMarker, !tt.wantLeaf)
			}
			if truncated == tt.wantLeaf {
				t.Errorf("SmartTruncate() truncated = %v, want %v", truncated, !tt.wantLeaf)
			}
		})
	}
}

func TestSmartTruncate_RedactsSensitiveKeys(t *testing.T) {
	tests := []struct {
		name     string
		obj      any
		noRedact bool
		want     string
	}{
		{
			name: "credential keys at any depth",
			obj: map[string]any{
				"api_key":  "k",
				"PassWord": "p",
				"query":    "weather",
				"headers":  map[string]any{"Authorization": "Bearer t", "Accept": "json"},
				"items":    []any{map[string]any{"x-goog-signature": "s", "id": 1}},
			},
			want: `{"PassWord":"[REDACTED]","api_key":"[REDACTED]",` +
				`"headers":{"Accept":"json","Authorization":"[REDACTED]"},` +
				`"items":[{"id":1,"x-goog-signature":"[REDACTED]"}],"query":"weather"}`,
		},
		{
			name: "session temp keys",
			obj:  map[string]any{"temp:draft": "d", "user:name": "n"},
			want: `{"temp:draft":"[REDACTED]","user:name":"n"}`,
		},
		{
			name: "struct fields by JSON name",
			obj:  credentials{User: "u", APIKey: "k", Token: "t"},
			want: `{"Token":"[REDACTED]","api_key":"[REDACTED]","user":"u"}`,
		},
		{
			name:     "redaction off keeps map keys",
			obj:      map[string]any{"api_key": "k", "temp:draft": "d"},
			noRedact: true,
			want:     `{"api_key":"k","temp:draft":"d"}`,
		},
		{
			name:     "redaction off keeps struct fields",
			obj:      credentials{User: "u", APIKey: "k", Token: "t"},
			noRedact: true,
			want:     `{"Token":"t","api_key":"k","user":"u"}`,
		},
		{
			name: "JSON-encoded object string has sensitive keys redacted",
			obj: map[string]any{
				"payload": `{"api_key":"secret","nested":{"token":"tok"},"keep":"ok"}`,
			},
			want: `{"payload":"{\"api_key\":\"[REDACTED]\",\"keep\":\"ok\",\"nested\":{\"token\":\"[REDACTED]\"}}"}`,
		},
		{
			name: "JSON-encoded array string and UTF-8 bytes have sensitive keys redacted",
			obj: map[string]any{
				"arr":   `[{"password":"pw","id":1}]`,
				"bytes": []byte(`{"client_secret":"sec","v":2}`),
			},
			want: `{"arr":"[{\"id\":1,\"password\":\"[REDACTED]\"}]","bytes":"{\"client_secret\":\"[REDACTED]\",\"v\":2}"}`,
		},
		{
			name: "double-encoded JSON string has sensitive keys redacted",
			obj: map[string]any{
				"payload": `"{\"api_key\":\"secret\"}"`,
			},
			want: `{"payload":"\"{\\\"api_key\\\":\\\"[REDACTED]\\\"}\""}`,
		},
		{
			name: "duplicate keys in JSON string are re-serialized without leaking earlier key",
			obj: map[string]any{
				"payload": `{"data":{"api_key":"secret"},"data":"ok"}`,
			},
			want: `{"payload":"{\"data\":\"ok\"}"}`,
		},
		{
			name: "multiple leading BOMs before JSON object string have sensitive keys redacted",
			obj: map[string]any{
				"payload": "\ufeff\ufeff{\"api_key\":\"secret\"}",
			},
			want: `{"payload":"{\"api_key\":\"[REDACTED]\"}"}`,
		},
		{
			name:     "redaction off keeps JSON-encoded string credentials",
			obj:      map[string]any{"payload": `{"api_key":"secret"}`},
			noRedact: true,
			want:     `{"payload":"{\"api_key\":\"secret\"}"}`,
		},
		{
			name: "camelCase and unseparated credential keys",
			obj: map[string]any{
				"accessToken":    "a",
				"APIKey":         "k",
				"clientSecret":   "c",
				"XGoogAPIKey":    "g",
				"x_goog_api_key": "x",
				"apikey":         "p",
				"tokenCount":     3,
				"api_key_id":     "id",
				"secretName":     "projects/p/secrets/s",
			},
			want: `{"APIKey":"[REDACTED]","XGoogAPIKey":"[REDACTED]","accessToken":"[REDACTED]",` +
				`"api_key_id":"id","apikey":"[REDACTED]","clientSecret":"[REDACTED]",` +
				`"secretName":"projects/p/secrets/s","tokenCount":3,"x_goog_api_key":"[REDACTED]"}`,
		},
		{
			name: "struct fields by Go name",
			obj: struct {
				APIKey       string
				ClientSecret string
				Name         string
			}{APIKey: "k", ClientSecret: "c", Name: "n"},
			want: `{"APIKey":"[REDACTED]","ClientSecret":"[REDACTED]","Name":"n"}`,
		},
		{
			name: "session user and app keys",
			obj: map[string]any{
				"user:api_key":     "k",
				"app:clientSecret": "c",
				"USER:Token":       "t",
				"user:name":        "n",
				"app:theme":        "dark",
			},
			want: `{"USER:Token":"[REDACTED]","app:clientSecret":"[REDACTED]","app:theme":"dark",` +
				`"user:api_key":"[REDACTED]","user:name":"n"}`,
		},
		{
			name: "JSON-encoded string with camelCase keys has sensitive keys redacted",
			obj:  map[string]any{"payload": `{"accessToken":"tok","keep":"ok"}`},
			want: `{"payload":"{\"accessToken\":\"[REDACTED]\",\"keep\":\"ok\"}"}`,
		},
		{
			name:     "redaction off keeps camelCase and prefixed keys",
			obj:      map[string]any{"accessToken": "a", "user:api_key": "k"},
			noRedact: true,
			want:     `{"accessToken":"a","user:api_key":"k"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, truncated, err := smartTruncate(tt.obj, 100, !tt.noRedact)
			if err != nil {
				t.Fatalf("smartTruncate() error: %v", err)
			}
			if string(got) != tt.want {
				t.Errorf("smartTruncate() = %s, want %s", got, tt.want)
			}
			if truncated {
				t.Error("smartTruncate() truncated = true, want false")
			}
		})
	}
}

func TestRedactJSONText_FailClosed(t *testing.T) {
	deepJSON := `{"api_key":"secret"}`
	for range maxSanitizeDepth + 2 {
		deepJSON = `{"nested":` + deepJSON + `}`
	}
	tests := []struct {
		name string
		obj  any
		want string
	}{
		{
			name: "malformed JSON object string",
			obj:  map[string]any{"payload": `{"api_key":"secret"`},
			want: `{"payload":"[UNPARSEABLE_JSON_BLOB]"}`,
		},
		{
			name: "JSON object with trailing suffix",
			obj:  map[string]any{"payload": `{"api_key":"secret"} trailing`},
			want: `{"payload":"[UNPARSEABLE_JSON_BLOB]"}`,
		},
		{
			name: "JSON object followed by second JSON object",
			obj:  map[string]any{"payload": `{"ok":1} {"api_key":"secret"}`},
			want: `{"payload":"[UNPARSEABLE_JSON_BLOB]"}`,
		},
		{
			name: "double-encoded malformed JSON string",
			obj:  map[string]any{"payload": `"{\"api_key\":\"secret\""`},
			want: `{"payload":"[UNPARSEABLE_JSON_BLOB]"}`,
		},
		{
			name: "quoted string with trailing JSON container",
			obj:  map[string]any{"payload": `"note" {"api_key":"secret"}`},
			want: `{"payload":"[UNPARSEABLE_JSON_BLOB]"}`,
		},
		{
			name: "quoted string followed by unicode-escaped JSON credential string",
			obj:  map[string]any{"payload": `"note" then "\u007b\"api_key\":\"SECRET\"\u007d"`},
			want: `{"payload":"[UNPARSEABLE_JSON_BLOB]"}`,
		},
		{
			name: "multiple leading BOMs before malformed JSON object string",
			obj:  map[string]any{"payload": "\ufeff\ufeff{\"api_key\":\"secret\""},
			want: `{"payload":"[UNPARSEABLE_JSON_BLOB]"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, truncated, err := SmartTruncate(tt.obj, 200)
			if err != nil {
				t.Fatalf("SmartTruncate() error: %v", err)
			}
			if string(got) != tt.want {
				t.Errorf("SmartTruncate() = %s, want %s", got, tt.want)
			}
			if !truncated {
				t.Error("SmartTruncate() truncated = false, want true when failing closed")
			}
		})
	}

	t.Run("deeply nested JSON string fails closed", func(t *testing.T) {
		got, truncated, err := SmartTruncate(map[string]any{"payload": deepJSON}, 10000)
		if err != nil {
			t.Fatalf("SmartTruncate() error: %v", err)
		}
		if bytes.Contains(got, []byte("secret")) {
			t.Errorf("SmartTruncate() leaked secret in deep JSON: %s", got)
		}
		if !truncated {
			t.Error("SmartTruncate() truncated = false, want true for deep JSON")
		}
	})
}

func TestWalkStruct_OmitEmptyShadowsEmbeddedField(t *testing.T) {
	type Embedded struct {
		X string `json:"x"`
		Y string `json:"y"`
	}
	type Outer struct {
		Embedded
		X string `json:"x,omitempty"`
	}
	v := Outer{Embedded: Embedded{X: "inner-x", Y: "inner-y"}, X: ""}
	got, truncated, err := SmartTruncate(v, 100)
	if err != nil {
		t.Fatalf("SmartTruncate() error: %v", err)
	}
	if truncated {
		t.Error("SmartTruncate() truncated = true, want false")
	}
	wantBytes, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal() error: %v", err)
	}
	if string(got) != string(wantBytes) {
		t.Errorf("SmartTruncate() = %s, want %s (matching encoding/json)", got, wantBytes)
	}
}

func TestWalkStruct_PromotesEmbeddedPointerToUnexportedStruct(t *testing.T) {
	type promotedInner struct {
		Name  string `json:"name"`
		Token string `json:"token,omitempty"`
	}
	type promotedPtrOuter struct {
		*promotedInner
		Extra string `json:"extra"`
	}

	tests := []struct {
		name string
		v    promotedPtrOuter
	}{
		{
			name: "non-nil embedded pointer",
			v:    promotedPtrOuter{promotedInner: &promotedInner{Name: "inner-name"}, Extra: "outer-extra"},
		},
		{
			name: "nil embedded pointer",
			v:    promotedPtrOuter{Extra: "outer-extra"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name+" matches encoding/json", func(t *testing.T) {
			got, truncated, err := SmartTruncate(tt.v, 1000)
			if err != nil {
				t.Fatalf("SmartTruncate() error: %v", err)
			}
			if truncated {
				t.Error("SmartTruncate() truncated = true, want false")
			}
			wantBytes, err := json.Marshal(tt.v)
			if err != nil {
				t.Fatalf("json.Marshal() error: %v", err)
			}
			var gotV, wantV any
			if err := json.Unmarshal(got, &gotV); err != nil {
				t.Fatalf("SmartTruncate() output %s is not JSON: %v", got, err)
			}
			if err := json.Unmarshal(wantBytes, &wantV); err != nil {
				t.Fatalf("json.Unmarshal(%s) error: %v", wantBytes, err)
			}
			if !reflect.DeepEqual(gotV, wantV) {
				t.Errorf("SmartTruncate() = %s, want %s (matching encoding/json)", got, wantBytes)
			}
		})
	}

	t.Run("sensitive promoted field is redacted rather than dropped", func(t *testing.T) {
		v := promotedPtrOuter{
			promotedInner: &promotedInner{Name: "inner-name", Token: "r14-promoted-token-secret"},
			Extra:         "outer-extra",
		}
		got, _, err := SmartTruncate(v, 1000)
		if err != nil {
			t.Fatalf("SmartTruncate() error: %v", err)
		}
		if bytes.Contains(got, []byte("r14-promoted-token-secret")) {
			t.Errorf("SmartTruncate() leaked the promoted token: %s", got)
		}
		var m map[string]any
		if err := json.Unmarshal(got, &m); err != nil {
			t.Fatalf("SmartTruncate() output %s is not a JSON object: %v", got, err)
		}
		want := map[string]any{"name": "inner-name", "token": redactedValue, "extra": "outer-extra"}
		if !reflect.DeepEqual(m, want) {
			t.Errorf("SmartTruncate() = %v, want %v", m, want)
		}
	})
}

func TestSanitizeURI(t *testing.T) {
	tests := []struct {
		name        string
		uri         string
		want        string
		wantChanged bool
	}{
		{name: "empty", uri: "", want: ""},
		{name: "no query", uri: "gs://bucket/file.mp4", want: "gs://bucket/file.mp4"},
		{name: "harmless query", uri: "https://example.com/a?alt=media", want: "https://example.com/a?alt=media"},
		{name: "harmless fragment", uri: "https://example.com/a#section-1", want: "https://example.com/a#section-1"},
		{
			name:        "API key",
			uri:         "https://example.com/a?key=abc&alt=media",
			want:        "https://example.com/a?key=%5BREDACTED%5D&alt=media",
			wantChanged: true,
		},
		{
			name:        "signed URL",
			uri:         "https://storage.googleapis.com/b/o?X-Goog-Signature=abc&X-Goog-Credential=def",
			want:        "https://storage.googleapis.com/b/o?X-Goog-Signature=%5BREDACTED%5D&X-Goog-Credential=%5BREDACTED%5D",
			wantChanged: true,
		},
		{
			name:        "access token",
			uri:         "https://example.com/a?access_token=abc",
			want:        "https://example.com/a?access_token=%5BREDACTED%5D",
			wantChanged: true,
		},
		{
			name:        "fragment access token",
			uri:         "https://example.com/callback#access_token=secret&state=ok",
			want:        "https://example.com/callback#access_token=%5BREDACTED%5D&state=ok",
			wantChanged: true,
		},
		{
			name:        "path segment api_key pair",
			uri:         "https://example.com/v1/api_key/secret/items",
			want:        "https://example.com/v1/%5BREDACTED%5D/%5BREDACTED%5D/items",
			wantChanged: true,
		},
		{
			name:        "path segment token pair",
			uri:         "https://example.com/token/abc",
			want:        "https://example.com/%5BREDACTED%5D/%5BREDACTED%5D",
			wantChanged: true,
		},
		{
			name:        "path segment key=value",
			uri:         "https://example.com/v1/api_key=secret/items",
			want:        "https://example.com/v1/api_key=%5BREDACTED%5D/items",
			wantChanged: true,
		},
		{
			name:        "double-escaped path segment access_token pair",
			uri:         "https://example.test/access%255Ftoken/SECRET",
			want:        "https://example.test/%5BREDACTED%5D/%5BREDACTED%5D",
			wantChanged: true,
		},
		{
			name:        "double-escaped path segment api_key=value",
			uri:         "https://example.test/api%255Fkey=SECRET",
			want:        "https://example.test/api%255Fkey=%5BREDACTED%5D",
			wantChanged: true,
		},
		{
			name:        "access token in a fragment route",
			uri:         "https://example.com/app#/cb?access_token=SECRET123",
			want:        "https://example.com/app#/cb?access_token=%5BREDACTED%5D",
			wantChanged: true,
		},
		{
			name:        "access token in a URL nested in a query value",
			uri:         "https://example.com/login?next=https://api.example.com/cb?access_token=SECRET123",
			want:        "https://example.com/login?next=https%3A%2F%2Fapi.example.com%2Fcb%3Faccess_token%3D%5BREDACTED%5D",
			wantChanged: true,
		},
		{
			name:        "path segment token:value",
			uri:         "https://example.com/token:SECRET123/file.png",
			want:        "https://example.com/token:%5BREDACTED%5D/file.png",
			wantChanged: true,
		},
		{
			name:        "access token after a semicolon in a query value",
			uri:         "https://example.com/a?a=1;access_token=SECRET123",
			want:        "https://example.com/a?a=1%3Baccess_token%3D%5BREDACTED%5D",
			wantChanged: true,
		},
		{
			name:        "camelCase and unseparated credential parameters",
			uri:         "https://example.com/a?apiKey=abc&accessToken=def&apikey=ghi&tokenCount=5",
			want:        "https://example.com/a?apiKey=%5BREDACTED%5D&accessToken=%5BREDACTED%5D&apikey=%5BREDACTED%5D&tokenCount=5",
			wantChanged: true,
		},
		{
			name: "query value mentioning basic",
			uri:  "https://example.com/search?q=basic%20recipes",
			want: "https://example.com/search?q=basic%20recipes",
		},
		{
			name: "query value mentioning bearer of",
			uri:  "https://example.com/search?q=bearer+of+bad+news#the-bearer-of-bad-news",
			want: "https://example.com/search?q=bearer+of+bad+news#the-bearer-of-bad-news",
		},
		{name: "user information", uri: "https://user:secret@example.com/a", want: redactedURIPlaceholder, wantChanged: true},
		{name: "unparsable", uri: "://missing-scheme", want: redactedURIPlaceholder, wantChanged: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, changed := sanitizeURI(tt.uri)
			if got != tt.want || changed != tt.wantChanged {
				t.Errorf("sanitizeURI(%q) = (%q, %v), want (%q, %v)", tt.uri, got, changed, tt.want, tt.wantChanged)
			}
		})
	}
}

func TestRedactErrorMessage(t *testing.T) {
	tests := []struct {
		name string
		msg  string
		want string
	}{
		{
			name: "JSON error with sensitive key and Bearer token in message field",
			msg:  `{"api_key":"k","message":"Bearer secret"}`,
			want: `{"api_key":"[REDACTED]","message":"Bearer [REDACTED]"}`,
		},
		{
			name: "JSON error with Bearer token in non-sensitive field and sensitive key",
			msg:  `{"error":"Bearer S","token":"x"}`,
			want: `{"error":"Bearer [REDACTED]","token":"[REDACTED]"}`,
		},
		{
			name: "JSON error with sensitive key and signed URL in detail field",
			msg:  `{"api_key":"k","detail":"https://example.com/v1?key=url-secret"}`,
			want: `{"api_key":"[REDACTED]","detail":"https://example.com/v1?key=%5BREDACTED%5D"}`,
		},
		{
			name: "Bearer token separated by tab",
			msg:  "auth failed: Bearer\tsecret",
			want: "auth failed: Bearer [REDACTED]",
		},
		{
			name: "Bearer token separated by newline",
			msg:  "auth failed: Bearer\nsecret",
			want: "auth failed: Bearer [REDACTED]",
		},
		{
			name: "Basic token separated by tab",
			msg:  "auth failed: Basic\tdXNlcjpwYXNz",
			want: "auth failed: Basic [REDACTED]",
		},
		{
			name: "Authorization token header",
			msg:  "request failed: Authorization: token ghp_S",
			want: "request failed: Authorization: [REDACTED]",
		},
		{
			name: "Authorization Digest header",
			msg:  `Authorization: Digest username="Mufasa", realm="test", response="6629fae49393a05397450978507c4ef1"`,
			want: "Authorization: [REDACTED]",
		},
		{
			name: "Proxy-Authorization header",
			msg:  "Proxy-Authorization: Basic dXNlcjpwYXNz",
			want: "Proxy-Authorization: [REDACTED]",
		},
		{
			name: "X-API-Key header",
			msg:  "X-API-Key: sk-secret-header",
			want: "X-API-Key: [REDACTED]",
		},
		{
			name: "key=value followed by suffix text is idempotent",
			msg:  "api_key=S failed",
			want: "api_key=[REDACTED] failed",
		},
		{
			name: "percent-encoded sensitive key fails closed",
			msg:  "request failed: api%5fkey=SECRET",
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "unicode-escaped sensitive key fails closed",
			msg:  `request failed: api\u005fkey=SECRET`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "unicode-escaped space after Bearer fails closed",
			msg:  `request failed: Bearer\u0020SECRET`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "truncated JSON error body without credentials is preserved",
			msg:  `{"error": {"code": 429, "message": "quota exceeded...`,
			want: `{"error": {"code": 429, "message": "quota exceeded...`,
		},
		{
			name: "brace-prefixed diagnostic text without credentials is preserved",
			msg:  `{bad request} retry later`,
			want: `{bad request} retry later`,
		},
		{
			name: "truncated JSON error body with access_token fails closed",
			msg:  `{"error": {"code": 401, "access_token": "secret", "message": "unauth`,
			want: unparseableJSONPlaceholder,
		},
		{
			name: "malformed JSON error body with password fails closed",
			msg:  `{"password":"correct SECRET`,
			want: unparseableJSONPlaceholder,
		},
		{
			name: "key=value with sentinel suffix preserves sentinel and strips secret",
			msg:  `access_token=[REDACTED]SECRET`,
			want: `access_token=[REDACTED]`,
		},
		{
			name: "Bearer with sentinel suffix preserves sentinel and strips secret",
			msg:  `Bearer [REDACTED]SECRET`,
			want: `Bearer [REDACTED]`,
		},
		{
			name: "already-redacted key=value sentinel alone is preserved",
			msg:  `access_token=[REDACTED]`,
			want: `access_token=[REDACTED]`,
		},
		{
			name: "already-redacted Bearer sentinel alone is preserved",
			msg:  `Bearer [REDACTED]`,
			want: `Bearer [REDACTED]`,
		},
		{
			name: "percent-encoded sensitive key with unrelated percent prose fails closed",
			msg:  `request failed at 100%: api%5fkey=SECRET`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "nested 5-layer percent-encoded sensitive key fails closed",
			msg:  `request failed: api%252525255fkey=SECRET`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "URI query parameter after secret is preserved",
			msg:  "API error: GET https://api.exchangerate.host/convert?api_key=SECRET_12345&to=USD",
			want: "API error: GET https://api.exchangerate.host/convert?api_key=%5BREDACTED%5D&to=USD",
		},
		{
			name: "plain token key-value followed by ampersand parameter is preserved",
			msg:  "error: token=SECRET_VAL&format=json",
			want: "error: token=[REDACTED]&format=json",
		},
		{
			name: "quoted JSON error with password and Bearer token in message",
			msg:  `{"password":"FIRST_SECRET","message":"Bearer SYNTHETIC_SECRET"}`,
			want: `{"message":"Bearer [REDACTED]","password":"[REDACTED]"}`,
		},
		{
			name: "error message with percent-encoded path and sensitive word is preserved",
			msg:  "GET https://example.com/v1/projects%2Fp/items failed: token expired",
			want: "GET https://example.com/v1/projects%2Fp/items failed: token expired",
		},
		{
			name: "bracket-led error message with token expired is preserved",
			msg:  "[ERROR] token expired, please sign in again",
			want: "[ERROR] token expired, please sign in again",
		},
		{
			name: "malformed JSON document with padded sensitive key fails closed",
			msg:  `{" api_key ":"SYNTHETIC_SECRET"`,
			want: unparseableJSONPlaceholder,
		},
		{
			name: "malformed JSON document with truncated temp key fails closed",
			msg:  `{"temp:cache/value":"SYNTHETIC_SECRET"`,
			want: unparseableJSONPlaceholder,
		},
		{
			name: "malformed JSON document with Unicode-padded key fails closed",
			msg:  "{\"\u00a0api_key\u00a0\":\"SYNTHETIC_SECRET\"",
			want: unparseableJSONPlaceholder,
		},
		{
			name: "malformed JSON document with escaped Unicode-padded key fails closed",
			msg:  `{"\u00a0api_key\u00a0":"SYNTHETIC_SECRET"`,
			want: unparseableJSONPlaceholder,
		},
		{
			name: "malformed JSON document with Unicode-padded temp key fails closed",
			msg:  "{\" \u00a0temp:path/to/key\u00a0 \":\"secret\"",
			want: unparseableJSONPlaceholder,
		},
		{
			name: "quoted diagnostic with escaped password returns rejection sentinel",
			msg:  `"note" then "{\"password\":\"SYNTHETIC_SECRET\"}"`,
			want: unparseableJSONPlaceholder,
		},
		{
			name: "bare password with hash suffix consumes punctuation",
			msg:  "failed: password=prefix#SYNTHETIC_SECRET",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "bare password with question mark suffix consumes punctuation",
			msg:  "failed: password=prefix?SYNTHETIC_SECRET",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "bare temp key with padded value redacts the whole value",
			msg:  "failed: temp:cache=SYNTHETIC_SECRET==",
			want: "failed: temp:cache=[REDACTED]",
		},
		{
			name: "bare temp key with colon delimiter and padded value redacts the whole value",
			msg:  "failed: temp:cache:SYNTHETIC_SECRET==",
			want: "failed: temp:cache:[REDACTED]",
		},
		{
			name: "bare temp key with spaced colon delimiter and padded value redacts the whole value",
			msg:  "failed: temp:cache: SYNTHETIC_SECRET==",
			want: "failed: temp:cache:[REDACTED]",
		},
		{
			name: "bare temp key with colon-bearing value redacts the whole value",
			msg:  "failed: temp:cache=urn:uuid:12345",
			want: "failed: temp:cache=[REDACTED]",
		},
		{
			name: "bare temp key with colon delimiter and colon-bearing value redacts the whole value",
			msg:  "failed: temp:cache:urn:uuid:12345",
			want: "failed: temp:cache:[REDACTED]",
		},
		{
			name: "bare temp key with spaced colon delimiter and colon-bearing value redacts the whole value",
			msg:  "failed: temp:cache: urn:uuid:12345",
			want: "failed: temp:cache:[REDACTED]",
		},
		{
			name: "bare temp key with slash and padded value redacts the whole value",
			msg:  "failed: temp:cache/val=SYNTHETIC_SECRET==",
			want: "failed: temp:cache/val=[REDACTED]",
		},
		{
			name: "JSON document with equals sign in temp key redacts the value",
			msg:  `{"temp:cache=foo": "bar"}`,
			want: `{"temp:cache=foo":"[REDACTED]"}`,
		},
		{
			name: "JSON document with colon in temp key redacts the value",
			msg:  `{"temp:cache:foo": "bar"}`,
			want: `{"temp:cache:foo":"[REDACTED]"}`,
		},
		{
			name: "embedded JSON with equals sign in quoted temp key redacts the value",
			msg:  `failed: {"temp:cache=foo": "SYNTHETIC_SECRET"}`,
			// As in adk-python, the key=value pass reads "temp:cache" as the
			// key and redacts the rest of the text up to "}" as its value.
			want: `failed: {"temp:cache=[REDACTED]}`,
		},
		{
			name: "bare password with escaped double quote in quoted value redacts the whole value",
			msg:  `failed: password="prefix\"suffix"`,
			want: `failed: password="[REDACTED]"`,
		},
		{
			name: "bare password with escaped single quote in quoted value redacts the whole value",
			msg:  `failed: password='prefix\'suffix'`,
			want: `failed: password='[REDACTED]'`,
		},
		{
			name: "bare temp key with escaped double quote in quoted value redacts the whole value",
			msg:  `failed: temp:cache="prefix\"suffix"`,
			want: `failed: temp:cache="[REDACTED]"`,
		},
		{
			name: "bare temp key with colon and escaped double quote in quoted value redacts the whole value",
			msg:  `failed: temp:cache: "prefix\"suffix"`,
			want: `failed: temp:cache:"[REDACTED]"`,
		},
		{
			name: "Bearer token with escaped double quote in quoted value redacts the whole value",
			msg:  `failed: Bearer "prefix\"suffix"`,
			want: `failed: Bearer [REDACTED]`,
		},
		{
			name: "JSON quoted with %q redacts the escaped value",
			msg:  `token exchange failed: "{\"access_token\": \"SECRET123\"}"`,
			want: `token exchange failed: "{\"access_token\":\"[REDACTED]\"}"`,
		},
		{
			name: "value quoted with escaped double quotes redacts the whole value",
			msg:  `failed: password=\"SECRET123\"`,
			want: `failed: password=\"[REDACTED]\"`,
		},
		{
			name: "value quoted with escaped single quotes redacts the whole value",
			msg:  `failed: password=\'SECRET123\'`,
			want: `failed: password=\'[REDACTED]\'`,
		},
		{
			name: "single-quoted key redacts the value",
			msg:  `bad config: {'api_key': 'SECRET123'}`,
			want: `bad config: {'api_key':'[REDACTED]'}`,
		},
		{
			name: "credentials envelope is redacted whole",
			msg:  `auth failed: {"credentials": {"access_token": "SECRET123", "expires_in": 3600}}`,
			want: `auth failed: {"credentials":"[REDACTED]"}`,
		},
		{
			name: "AWS STS credentials envelope is redacted whole",
			msg:  `assume role failed: {"Credentials": {"AccessKeyId": "EXAMPLEKEYID", "SecretAccessKey": "SECRET123", "SessionToken": "SECRET123", "Expiration": "2026-10-14T00:00:00Z"}}`,
			want: `assume role failed: {"Credentials":"[REDACTED]"}`,
		},
		{
			name: "JSON document with a credentials envelope is redacted whole",
			msg:  `{"Credentials":{"AccessKeyId":"EXAMPLEKEYID","SecretAccessKey":"SECRET123","SessionToken":"SECRET123"},"Message":"ok"}`,
			want: `{"Credentials":"[REDACTED]","Message":"ok"}`,
		},
		{
			name: "AWS secret access key and session token in key=value text are redacted",
			msg:  "failed: SecretAccessKey=SECRET123 SessionToken=SECRET456 secret_access_key=SECRET789 session-token=SECRET000",
			want: "failed: SecretAccessKey=[REDACTED] SessionToken=[REDACTED] secret_access_key=[REDACTED] session-token=[REDACTED]",
		},
		{
			name: "AWS environment variables for a secret access key and a session token are redacted",
			msg:  "AWS_SECRET_ACCESS_KEY=SECRET123 AWS_SESSION_TOKEN=SECRET123",
			want: "AWS_SECRET_ACCESS_KEY=[REDACTED] AWS_SESSION_TOKEN=[REDACTED]",
		},
		{
			name: "AWS shared credentials file entries are redacted",
			msg:  "bad profile: [default]\naws_access_key_id = EXAMPLEKEYID\naws_secret_access_key = SECRET123\naws_session_token = SECRET456",
			want: "bad profile: [default]\naws_access_key_id = EXAMPLEKEYID\naws_secret_access_key=[REDACTED]\naws_session_token=[REDACTED]",
		},
		{
			name: "credentials block in YAML keeps its nested keys and redacts their values",
			msg:  "config:\ncredentials:\n  token: SECRET123\n  region: us",
			want: "config:\ncredentials:\n  token:[REDACTED]\n  region: us",
		},
		{
			name: "credentials block in YAML with a nested password",
			msg:  "config:\ncredentials:\n  password: SECRET123",
			want: "config:\ncredentials:\n  password:[REDACTED]",
		},
		{
			name: "indented credentials block in YAML with a nested client secret",
			msg:  "config:\n  credentials:\n    client_secret: SECRET123",
			want: "config:\n  credentials:\n    client_secret:[REDACTED]",
		},
		{
			name: "credentials block in YAML with a nested hyphenated header",
			msg:  "credentials:\n  X-Auth-Token: SECRET123",
			want: "credentials:\n  X-Auth-Token:[REDACTED]",
		},
		{
			name: "credentials block in YAML with a quoted nested key and value",
			msg:  "credentials:\n  'token': 'SECRET123'",
			want: "credentials:\n  'token':'[REDACTED]'",
		},
		{
			name: "credentials block in YAML with a nested camelCase key",
			msg:  "Credentials:\n  SessionToken: SECRET123",
			want: "Credentials:\n  SessionToken:[REDACTED]",
		},
		{
			name: "credentials block in YAML with a nested dotted key",
			msg:  "credentials:\n  spring.datasource.password: SECRET123",
			want: "credentials:\n  spring.datasource.password:[REDACTED]",
		},
		{
			name: "credentials block in YAML whose nested value is on the next line",
			msg:  "credentials:\n  token:\n    SECRET123",
			want: "credentials:\n  token:[REDACTED]",
		},
		{
			name: "credentials followed by a key: value pair in text",
			msg:  "loading credentials: token: SECRET123",
			want: "loading credentials: token:[REDACTED]",
		},
		{
			name: "credentials followed by a key = value pair in text",
			msg:  "credentials: token = SECRET123",
			want: "credentials: token=[REDACTED]",
		},
		{
			name: "credentials followed by a key: value pair after an unterminated quote",
			msg:  `bad config: credentials: "token: SECRET123`,
			want: `bad config: credentials: "token:[REDACTED]`,
		},
		{
			name: "credentials followed by a key: value pair after an unterminated escaped quote",
			msg:  `bad config: credentials: \"token: SECRET123`,
			want: `bad config: credentials: \"token:[REDACTED]`,
		},
		{
			name: "quoted value that starts with a key: value pair is redacted whole",
			msg:  `password: 'token: SECRETA SECRETB'`,
			want: `password:'[REDACTED]'`,
		},
		{
			name: "quoted value that starts with a key=value pair is redacted whole",
			msg:  `token: "password=SECRETA SECRETB"`,
			want: `token:"[REDACTED]"`,
		},
		{
			name: "quoted value that starts with a key:value pair is redacted whole",
			msg:  `password: "token:SECRETA SECRETB"`,
			want: `password:"[REDACTED]"`,
		},
		{
			name: "Python dict value that starts with a key=value pair is redacted whole",
			msg:  `bad config: {'api_key': 'sig=SECRETA SECRETB'}`,
			want: `bad config: {'api_key':'[REDACTED]'}`,
		},
		{
			name: "bare value that ends with a key:value pair is redacted whole",
			msg:  "password=SECRETP-token:SECRETB",
			want: "password=[REDACTED]",
		},
		{
			name: "bare value that ends with a dotted key=value pair is redacted whole",
			msg:  "password=SECRETP.secret=SECRETB",
			want: "password=[REDACTED]",
		},
		{
			name: "bare value that ends with a dotted sig=value pair is redacted whole",
			msg:  "api_key=SECRETP.sig=SECRETB",
			want: "api_key=[REDACTED]",
		},
		{
			name: "credentials in a Python dict whose value starts with a key=value pair is redacted whole",
			msg:  `bad config: {'credentials': 'secret=correct horse SECRETB staple'}`,
			want: `bad config: {'credentials':'[REDACTED]'}`,
		},
		{
			name: "credential in a Python dict whose value starts with a key: value pair is redacted whole",
			msg:  `bad config: {'credential': 'token: abc SECRETB'}`,
			want: `bad config: {'credential':'[REDACTED]'}`,
		},
		{
			name: "quoted credentials value that starts with a key=value pair is redacted whole",
			msg:  `credentials="token=abc SECRETB"`,
			want: `credentials="[REDACTED]"`,
		},
		{
			name: "upper-case quoted credentials value that starts with a key=value pair is redacted whole",
			msg:  `CREDENTIALS="secret=correct horse SECRETB"`,
			want: `CREDENTIALS="[REDACTED]"`,
		},
		{
			name: "credentials bare value that ends with a key:value pair is redacted whole",
			msg:  "credentials: SECRETP-token:SECRETB",
			want: "credentials:[REDACTED]",
		},
		{
			name: "single-quoted credentials value that starts with a key=value pair is redacted whole",
			msg:  "credentials: 'secret=correct horse SECRETB'",
			want: "credentials:'[REDACTED]'",
		},
		{
			name: "quoted value that is a key name with no delimiter after it is redacted",
			msg:  `failed: password: "password"`,
			want: `failed: password:"[REDACTED]"`,
		},
		{
			name: "quoted value followed by a delimiter is redacted when it is not a key name",
			msg:  `failed: password: "SECRET123": invalid`,
			want: `failed: password:"[REDACTED]": invalid`,
		},
		{
			name: "nested key that ends with = is kept and its value redacted",
			msg:  "credentials: token= SECRET123",
			want: "credentials: token=[REDACTED]",
		},
		{
			name: "value followed by a spaced delimiter is redacted when it is not a key name",
			msg:  "failed: password: SECRET123 : retry",
			want: "failed: password:[REDACTED] : retry",
		},
		{
			name: "value that ends with a redacted value is redacted when its head is not a key name",
			msg:  "failed: password=SECRET123:[REDACTED]",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "nested key with a colon in its name is redacted idempotently",
			msg:  "credentials:\n  app:client_secret: SECRET123",
			want: "credentials:\n  app:client_secret:[REDACTED]",
		},
		{
			name: "value that ends with a key name joined to other text is redacted",
			msg:  "failed: password: SECRET123token: retry",
			want: "failed: password:[REDACTED] retry",
		},
		{
			name: "value that only ends in a key name before a wrapped error is redacted with the pair it starts",
			msg:  "connect user=bob password=dev-SECRET123-password: dial tcp 10.0.0.1:5432: connection refused",
			want: "connect user=bob password=[REDACTED] tcp 10.0.0.1:5432: connection refused",
		},
		{
			name: "quoted value that only ends in a key name is redacted with the pair it starts",
			msg:  `login failed: password="SECRET123-secret": invalid`,
			want: `login failed: password="[REDACTED]"`,
		},
		{
			name: "single-quoted value that only ends in a key name is redacted with the pair it starts",
			msg:  `login failed: password='SECRET123-secret': invalid`,
			want: `login failed: password='[REDACTED]'`,
		},
		{
			name: "value quoted with escaped quotes that only ends in a key name is redacted with the pair it starts",
			msg:  `login failed: password=\"SECRET123-secret\": invalid`,
			want: `login failed: password=\"[REDACTED]\"`,
		},
		{
			name: "quoted value that only ends in a key name before a spaced delimiter is redacted with the pair it starts",
			msg:  `login failed: password="SECRET123-secret" : invalid`,
			want: `login failed: password="[REDACTED]"`,
		},
		{
			name: "value that only ends in a hyphenated key name is redacted with the pair it starts",
			msg:  "api_key=prodSECRET123-api-key: permission denied",
			want: "api_key=[REDACTED]",
		},
		{
			name: "token that only ends in a key name in prose is redacted with the pair it starts",
			msg:  "my login is token=abc-SECRET123-token: please check",
			want: "my login is token=[REDACTED] check",
		},
		{
			name: "value that only ends in a dotted key name is redacted with the pair it starts",
			msg:  "password=svc.SECRET123.password: dial tcp",
			want: "password=[REDACTED] tcp",
		},
		{
			name: "upper-case value that only ends in a key name is redacted with the pair it starts",
			msg:  "PASSWORD=DEV-SECRET123-PASSWORD: DIAL TCP",
			want: "PASSWORD=[REDACTED] TCP",
		},
		{
			name: "credentials value that only ends in a key name is redacted with the pair it starts",
			msg:  "credentials=dev-SECRET123-password: dial tcp",
			want: "credentials=[REDACTED] tcp",
		},
		{
			name: "value that only ends in a key name before a spaced delimiter is redacted with the pair it starts",
			msg:  "failed: password: SECRET123-token = SECRET456 retry",
			want: "failed: password:[REDACTED] retry",
		},
		{
			name: "compound key name on the line of credentials is redacted with its value",
			msg:  "credentials: X-Amz-Security-Token: SECRET123",
			want: "credentials:[REDACTED]",
		},
		{
			name: "temp: value ending in a key name before a wrapped error is redacted with the pair it starts",
			msg:  "password=temp:SECRET123-token: dial tcp",
			want: "password=[REDACTED] tcp",
		},
		{
			name: "temp: value before a wrapped error is redacted with the pair it starts",
			msg:  "password=temp:SECRET123: dial tcp",
			want: "password=[REDACTED] tcp",
		},
		{
			name: "user:temp: value before a wrapped error is redacted with the pair it starts",
			msg:  "password=user:temp:SECRET123: dial tcp",
			want: "password=[REDACTED] tcp",
		},
		{
			name: "app:temp: value before a wrapped error is redacted with the pair it starts",
			msg:  "password=app:temp:SECRET123: dial tcp",
			want: "password=[REDACTED] tcp",
		},
		{
			name: "quoted temp: value before a wrapped error is redacted with the pair it starts",
			msg:  `password="temp:SECRET123": dial tcp`,
			want: `password="[REDACTED]" tcp`,
		},
		{
			name: "temp: key on the line of credentials is redacted with its value",
			msg:  "credentials: temp:foo: SECRET123",
			want: "credentials:[REDACTED]",
		},
		{
			name: "temp: key on an indented line under credentials is kept and its value redacted",
			msg:  "credentials:\n  temp:foo: SECRET123",
			want: "credentials:\n  temp:foo:[REDACTED]",
		},
		{
			name: "user:temp: key on an indented line under credentials is kept and its value redacted",
			msg:  "credentials:\n  user:temp:foo: SECRET123",
			want: "credentials:\n  user:temp:foo:[REDACTED]",
		},
		{
			name: "prefixed JSON object with a credential after an escaped line break fails closed",
			msg:  `POST /v1/login: 400: {"msg":"retry\npassword=SECRET123 vault: secret: kv/data/db-password: \"OTHER pass\""}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "prefix with bracket tag before JSON object with a credential after an escaped line break fails closed",
			msg:  `POST /v1/login [400]: {"msg":"retry\npassword=SECRET123 vault: secret: kv/data/db-password: \"OTHER pass\""}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "bracket-led log line before JSON object with only an escaped-newline credential fails closed",
			msg:  `[ERROR] POST /v1/login: 400: {"msg":"retry\npassword=SECRET123"}`,
			want: unparseableJSONPlaceholder,
		},
		{
			name: "bracket-prefixed JSON object with a credential after an escaped line break and a raw marker fails closed",
			msg:  `[ERROR] POST /v1/login: 400: {"msg":"retry\npassword=SECRET123 vault: secret: kv/data/db-password: \"OTHER pass\""}`,
			want: unparseableJSONPlaceholder,
		},
		{
			name: "JSON object followed by trailing status text with a credential after an escaped line break fails closed",
			msg:  `POST /v1/login: {"msg":"retry\npassword=SECRET123"} (status 400)`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "prefixed JSON array with a credential after an escaped line break fails closed",
			msg:  `POST /v1/login: 400: ["retry\npassword=dev-SECRET123-password"]`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "JSON object with a credential after an escaped line break and a bare pair in the last field redacts each pair",
			msg:  `{"msg":"retry\npassword=SECRET123 vault: secret: kv/data/db-password: \"OTHER pass\"","code":"token=abc"}`,
			want: `{"code":"token=[REDACTED]","msg":"retry\npassword=[REDACTED] vault: secret:[REDACTED]"}`,
		},
		{
			name: "JSON object whose string ends with a value ending in a key name before a word redacts the pair",
			msg:  `{"msg":"retry\npassword=dev-SECRET123-password: dial"}`,
			want: `{"msg":"retry\npassword=[REDACTED]"}`,
		},
		{
			name: "JSON object with a password after an escaped line break and a final value ending in a key name redacts both",
			msg:  `{"msg":"retry\npassword=hunter2val token=abc-SECRET123-token: dial"}`,
			want: `{"msg":"retry\npassword=[REDACTED] token=[REDACTED]"}`,
		},
		{
			name: "JSON array whose string ends with a value ending in a key name before a word redacts the pair",
			msg:  `["retry\npassword=dev-SECRET123-password: dial"]`,
			want: `["retry\npassword=[REDACTED]"]`,
		},
		{
			name: "JSON object whose string ends with a value ending in a key name redacts the pair",
			msg:  `{"msg":"retry\npassword=dev-SECRET123-password"}`,
			want: `{"msg":"retry\npassword=[REDACTED]"}`,
		},
		{
			name: "JSON array whose string ends with a value ending in a key name redacts the pair",
			msg:  `["retry\npassword=dev-SECRET123-password"]`,
			want: `["retry\npassword=[REDACTED]"]`,
		},
		{
			name: "stray quote before JSON object with an escaped-newline credential fails closed",
			msg:  `bad token " in body: {"msg":"retry\npassword=SECRET123"}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "escaped quote before JSON object with an escaped-newline credential fails closed",
			msg:  `flag \"foo: {"msg":"retry\npassword=SECRET123"}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "non-JSON backslash before escaped quote and escaped-newline credential fails closed",
			msg:  `err: "dir C:\work\"retry\npassword=SECRET123"`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "truncated unclosed JSON error body with an escaped-newline credential fails closed",
			msg:  `upstream error: {"error":"login failed\npassword=SECRET123`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "single-quoted Python dict with an escaped-newline credential fails closed",
			msg:  `{'error': 'retry\npassword=SECRET123'}`,
			want: unparseableJSONPlaceholder,
		},
		{
			name: "prefixed single-quoted Python dict with an escaped-newline credential fails closed",
			msg:  `bad config: {'error': 'retry\npassword=SECRET123'}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "unquoted escaped log line with an escaped-newline credential fails closed",
			msg:  `err=retry\npassword=SECRET123`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "escaped backspace before a credential fails closed",
			msg:  `retry\bpassword=SECRET123`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "escaped form feed before a credential fails closed",
			msg:  `retry\fpassword=SECRET123`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "escaped carriage return before a credential fails closed",
			msg:  `retry\rpassword=SECRET123`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "escaped tab before a credential fails closed",
			msg:  `retry\tpassword=SECRET123`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "escaped slash in a URL with userinfo credentials fails closed",
			msg:  `https:\/\/user:SECRET123@example.com`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "deeply backslash-escaped double quote before a key word with no value is kept",
			msg:  strings.Repeat(`\`, 256) + `"token`,
			want: strings.Repeat(`\`, 256) + `"token`,
		},
		{
			name: "deeply backslash-escaped single quote before a key word with no value is kept",
			msg:  strings.Repeat(`\`, 256) + `'token`,
			want: strings.Repeat(`\`, 256) + `'token`,
		},
		{
			name: "escaped backslash before a unicode-escaped space in a bearer token fails closed",
			msg:  `Bearer\\u0020SECRET123`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "Windows path without credentials is preserved",
			msg:  `C:\new\file.txt`,
			want: `C:\new\file.txt`,
		},
		{
			name: "value that only ends in a key name glued to a quoted value is redacted with it",
			msg:  `failed: password: dev-SECRET123-token="a SECRET123 b" retry`,
			want: "failed: password:[REDACTED] retry",
		},
		{
			name: "value that only ends in a key name before a nested key is redacted with the pair that key starts",
			msg:  "password=a-SECRET123-token: token: SECRET456 retry",
			want: "password=[REDACTED] retry",
		},
		{
			name: "values that only end in key names one after another are redacted together",
			msg:  "password=a-SECRET123-token: b-SECRET123-token: c-SECRET123-token: SECRET456 retry",
			want: "password=[REDACTED] retry",
		},
		{
			name: "Go struct field whose value only ends in a key name is redacted with the pair it starts",
			msg:  "failed: {Password:a-SECRET123-token: SECRET456} retry",
			want: "failed: {Password:[REDACTED]} retry",
		},
		{
			name: "Go map entry whose value only ends in a key name is redacted with the pair it starts",
			msg:  "failed: map[password:a-SECRET123-token: SECRET456 user:bob]",
			want: "failed: map[password:[REDACTED] user:bob]",
		},
		{
			name: "value that only ends in a key name before a redacted value is redacted",
			msg:  "failed: token=abc-SECRET123-token:[REDACTED] retry",
			want: "failed: token=[REDACTED] retry",
		},
		{
			name: "credentials block in YAML with a nested hyphenated compound key",
			msg:  "credentials:\n  X-Amz-Security-Token: SECRET123",
			want: "credentials:\n  X-Amz-Security-Token:[REDACTED]",
		},
		{
			name: "hyphenated header name after credentials is kept and its value redacted",
			msg:  "credentials: X-Auth-Token: SECRET123",
			want: "credentials: X-Auth-Token:[REDACTED]",
		},
		{
			name: "key-shaped bare value with no value after its colon is redacted whole",
			msg:  "password=SECRETP-token:",
			want: "password=[REDACTED]",
		},
		{
			name: "key-shaped bare value with no value after its equals sign is redacted whole",
			msg:  "password=SECRETP/token=",
			want: "password=[REDACTED]",
		},
		{
			name: "base64 value that ends with a key name and one pad is redacted whole",
			msg:  "api_key=SECRETAbC+xYz/Sig=",
			want: "api_key=[REDACTED]",
		},
		{
			name: "base64 value that ends with a key name, one pad and a newline is redacted whole",
			msg:  "api_key=SECRETAbC+xYz/Sig=\n",
			want: "api_key=[REDACTED]\n",
		},
		{
			name: "base64 value that ends with a key name, one pad and a space is redacted whole",
			msg:  "api_key=SECRETAbC+xYz/Sig= ",
			want: "api_key=[REDACTED] ",
		},
		{
			name: "key-shaped bare value followed by a comma is redacted whole",
			msg:  "password=SECRETP-token:,next",
			want: "password=[REDACTED],next",
		},
		{
			name: "key-shaped bare value followed by a semicolon is redacted whole",
			msg:  "password=SECRETP-token:;next",
			want: "password=[REDACTED];next",
		},
		{
			name: "key-shaped bare value followed by a space and a comma is redacted whole",
			msg:  "password=SECRETP-token: ,next",
			want: "password=[REDACTED] ,next",
		},
		{
			name: "key-shaped bare value followed by an ampersand is redacted whole",
			msg:  "password=SECRETP-token:&next=1",
			want: "password=[REDACTED]&next=1",
		},
		{
			name: "key-shaped bare value followed by a tab and a semicolon is redacted whole",
			msg:  "password=SECRETP-token:\t;x",
			want: "password=[REDACTED]\t;x",
		},
		{
			name: "key-shaped bare value at the end of a CRLF line is redacted whole",
			msg:  "password=SECRETP-token:\r\n",
			want: "password=[REDACTED]\r\n",
		},
		{
			name: "key-shaped bare value followed by a form feed is redacted whole",
			msg:  "password=SECRETP-token:\f",
			want: "password=[REDACTED]\f",
		},
		{
			name: "key-shaped bare value followed by a closing brace is redacted whole",
			msg:  "failed: password=SECRETP-token: }",
			want: "failed: password=[REDACTED] }",
		},
		{
			name: "key-shaped value of a Go struct field followed by the closing brace is redacted whole",
			msg:  "failed: {Password:abc-token: }",
			want: "failed: {Password:[REDACTED] }",
		},
		{
			name: "base64 value that ends with a key name, one pad and a closing brace is redacted whole",
			msg:  "api_key=SECRETAbC+xYz/Sig= }",
			want: "api_key=[REDACTED] }",
		},
		{
			name: "key-shaped bare value followed by a closing bracket is redacted whole",
			msg:  "password=SECRETP-token: ]",
			want: "password=[REDACTED] ]",
		},
		{
			name: "key-shaped value of a Go struct field followed by the closing brace and another struct is redacted whole",
			msg:  "failed: [{Password:abc-token: } {User:bob}]",
			want: "failed: [{Password:[REDACTED] } {User:bob}]",
		},
		{
			name: "key-shaped bare value followed by only punctuation is redacted whole",
			msg:  "password=abc-token: ...",
			want: "password=[REDACTED] ...",
		},
		{
			name: "nested key whose value starts with a closing bracket is kept and its value redacted",
			msg:  "credentials: token = ]SECRETZ",
			want: "credentials: token=[REDACTED]",
		},
		{
			name: "nested key in YAML whose value starts with a closing brace is kept and its value redacted",
			msg:  "credentials:\n  password: }SECRETZ",
			want: "credentials:\n  password:[REDACTED]",
		},
		{
			name: "nested key in a Go struct whose value starts with a closing bracket is kept and its value redacted",
			msg:  "failed: {Credentials:token: ]SECRETZ}",
			want: "failed: {Credentials:token:[REDACTED]}",
		},
		{
			name: "nested key whose value is quoted with leading whitespace is kept and its value redacted",
			msg:  `credentials: token: " SECRETX"`,
			want: `credentials: token:"[REDACTED]"`,
		},
		{
			name: "nested key whose single-quoted value has leading whitespace is kept and its value redacted",
			msg:  `credentials: token: ' SECRETX'`,
			want: `credentials: token:'[REDACTED]'`,
		},
		{
			name: "nested key whose value is quoted with escaped quotes and has leading whitespace is kept and its value redacted",
			msg:  `credentials: token: \" SECRETX\"`,
			want: `credentials: token:\"[REDACTED]\"`,
		},
		{
			name: "nested key in YAML whose quoted value has leading whitespace is kept and its value redacted",
			msg:  "credentials:\n  token: \" SECRETX\"\n  region: us",
			want: "credentials:\n  token:\"[REDACTED]\"\n  region: us",
		},
		{
			name: "key-shaped bare value followed by the quote that closes the string around it is redacted whole",
			msg:  `failed: "password=SECRETP-token: "`,
			want: `failed: "password=[REDACTED] "`,
		},
		{
			name: "key-shaped bare value followed by the closing single quote of the string around it is redacted whole",
			msg:  `failed: 'password=SECRETP-token: '`,
			want: `failed: 'password=[REDACTED] '`,
		},
		{
			name: "key-shaped bare value followed by the closing escaped quote of the string around it is redacted whole",
			msg:  `failed: \"password=SECRETP-token: \"`,
			want: `failed: \"password=[REDACTED] \"`,
		},
		{
			name: "key-shaped bare value in a quoted string followed by a colon is redacted whole",
			msg:  `parsing "password=SECRETP-token: ": invalid syntax`,
			want: `parsing "password=[REDACTED] ": invalid syntax`,
		},
		{
			name: "key-shaped bare value in a quoted string followed by another quoted string is redacted whole",
			msg:  `failed: "password=SECRETP-token: " for user "bob"`,
			want: `failed: "password=[REDACTED] " for user "bob"`,
		},
		{
			name: "key-shaped bare value in a quoted string followed by a quoted string that starts with a non-ASCII letter is redacted whole",
			msg:  `failed: "password=SECRETP-token: " for user "Émile"`,
			want: `failed: "password=[REDACTED] " for user "Émile"`,
		},
		{
			name: "key-shaped bare value in a quoted string followed by an empty quoted string is redacted whole",
			msg:  `got "password=SECRETP-token: ", want ""`,
			want: `got "password=[REDACTED] ", want ""`,
		},
		{
			name: "key-shaped bare value in an escaped quoted string followed by an empty escaped quoted string is redacted whole",
			msg:  `got \"password=SECRETP-token: \", want \"\"`,
			want: `got \"password=[REDACTED] \", want \"\"`,
		},
		{
			name: "nested key whose escaped quoted value is followed by an escaped newline fails closed",
			msg:  `credentials: token: \" SECRET123\"\nretry`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "nested key glued to a quoted value with spaces is kept and its value redacted",
			msg:  `credentials: password="correct SECRET123 battery"`,
			want: `credentials: password="[REDACTED]"`,
		},
		{
			name: "nested key glued to a quoted value with leading whitespace is kept and its value redacted",
			msg:  `credentials: token=" SECRET123"`,
			want: `credentials: token="[REDACTED]"`,
		},
		{
			name: "nested key glued by a colon to a quoted bearer value is kept and its value redacted",
			msg:  `credentials: token:"Bearer SECRET123"`,
			want: `credentials: token:"[REDACTED]"`,
		},
		{
			name: "nested key glued to a single-quoted value with a comma is kept and its value redacted",
			msg:  `credentials: token='a,SECRET123'`,
			want: `credentials: token='[REDACTED]'`,
		},
		{
			name: "nested key glued to a quoted value with a semicolon is kept and its value redacted",
			msg:  `credentials: token="a;SECRET123" more`,
			want: `credentials: token="[REDACTED]" more`,
		},
		{
			name: "nested key glued to an escaped quoted value with spaces is kept and its value redacted",
			msg:  `credentials: token=\"a SECRET123\"`,
			want: `credentials: token=\"[REDACTED]\"`,
		},
		{
			name: "nested key under password glued to a quoted value with spaces is kept and its value redacted",
			msg:  `password: token:"a SECRET123"`,
			want: `password: token:"[REDACTED]"`,
		},
		{
			name: "nested key in a Go struct glued to a quoted value with leading whitespace is kept and its value redacted",
			msg:  `failed: {Credentials:token=" SECRET123"}`,
			want: `failed: {Credentials:token="[REDACTED]"}`,
		},
		{
			name: "nested key in a Go map glued to a quoted value with spaces is kept and its value redacted",
			msg:  `failed: map[credentials:token="a SECRET123"]`,
			want: `failed: map[credentials:token="[REDACTED]"]`,
		},
		{
			name: "nested key in YAML glued to a quoted value with spaces is kept and its value redacted",
			msg:  "credentials:\n  token=\"a SECRET123\"\n  region: us",
			want: "credentials:\n  token=\"[REDACTED]\"\n  region: us",
		},
		{
			name: "nested pair glued to a bare value is redacted whole",
			msg:  `credentials: token=SECRET123`,
			want: `credentials:[REDACTED]`,
		},
		{
			name: "nested pair glued to a quoted value without spaces is redacted whole",
			msg:  `credentials: token="SECRET123"`,
			want: `credentials:[REDACTED]`,
		},
		{
			name: "key-shaped bare value glued to the closing quote of a string followed by another quoted string is redacted whole",
			msg:  `got "password=SECRETP-token=", want "abc"`,
			want: `got "password=[REDACTED], want "abc"`,
		},
		{
			name: "name that is not a key name glued to a quoted value with spaces is redacted",
			msg:  `credentials: user="SECRET123 token=SECRET456"`,
			want: `credentials:[REDACTED] token=[REDACTED]`,
		},
		{
			name: "key name glued to a quoted value with spaces by a character other than a delimiter is redacted",
			msg:  `credentials: token("SECRET123 token=SECRET456")`,
			want: `credentials:[REDACTED] token=[REDACTED]`,
		},
		{
			name: "nested password in YAML whose quoted value has leading whitespace is kept and its value redacted",
			msg:  "credentials:\n  password: \" SECRET123\"\n  region: us",
			want: "credentials:\n  password:\"[REDACTED]\"\n  region: us",
		},
		{
			name: "quoted nested key whose single-quoted value has leading whitespace is kept and its value redacted",
			msg:  `credentials: 'token': ' SECRET123'`,
			want: `credentials: 'token':'[REDACTED]'`,
		},
		{
			name: "nested key whose quoted value starts with punctuation is kept and its value redacted",
			msg:  `credentials: token = ". SECRET123"`,
			want: `credentials: token="[REDACTED]"`,
		},
		{
			name: "nested key whose quoted value starts with a closing brace is kept and its value redacted",
			msg:  `credentials: token: "} SECRET123"`,
			want: `credentials: token:"[REDACTED]"`,
		},
		{
			name: "nested key in a Go struct whose quoted value has leading whitespace is kept and its value redacted",
			msg:  `failed: {Credentials:token: " SECRET123"}`,
			want: `failed: {Credentials:token:"[REDACTED]"}`,
		},
		{
			name: "nested key in a slice of Go structs whose quoted value has leading whitespace is kept and its value redacted",
			msg:  `failed: [{Credentials:token: " SECRET123"}]`,
			want: `failed: [{Credentials:token:"[REDACTED]"}]`,
		},
		{
			name: "key-shaped value of a Go struct field followed by an unclosed quote is redacted whole",
			msg:  `failed: {Password:SECRET123-token: "}`,
			want: `failed: {Password:[REDACTED] "}`,
		},
		{
			name: "key-shaped value of a Go struct field followed by an unclosed single quote is redacted whole",
			msg:  `failed: {Password:SECRET123-token: '}`,
			want: `failed: {Password:[REDACTED] '}`,
		},
		{
			name: "key-shaped value of a Go struct field followed by empty quotes is redacted whole",
			msg:  `failed: {Password:SECRET123-token: ""}`,
			want: `failed: {Password:[REDACTED] ""}`,
		},
		{
			name: "key-shaped value of a Go map entry followed by empty quotes is redacted whole",
			msg:  `failed: map[Password:SECRET123-token: ""]`,
			want: `failed: map[Password:[REDACTED] ""]`,
		},
		{
			name: "key-shaped bare value followed by a lone backslash is redacted whole",
			msg:  `password=SECRET123-token: \`,
			want: `password=[REDACTED] \`,
		},
		{
			name: "quoted key-shaped value followed by a delimiter with no value is redacted whole",
			msg:  `password="SECRETQ-token" :`,
			want: `password="[REDACTED]" :`,
		},
		{
			name: "key-shaped bare value followed by a spaced delimiter with no value is redacted whole",
			msg:  "password=SECRETQ-token :",
			want: "password=[REDACTED] :",
		},
		{
			name: "key-shaped bare value followed by a tab and a delimiter with no value is redacted whole",
			msg:  "password=SECRETP-token\t:",
			want: "password=[REDACTED]\t:",
		},
		{
			name: "key-shaped value after a spaced delimiter with no value after it is redacted whole",
			msg:  "password = SECRETP-token:",
			want: "password=[REDACTED]",
		},
		{
			name: "temp: key-shaped value with no value after it is redacted whole",
			msg:  "password=temp:SECRETT:",
			want: "password=[REDACTED]",
		},
		{
			name: "value that is only a key name and a colon is redacted",
			msg:  "password=secret:",
			want: "password=[REDACTED]",
		},
		{
			name: "value that is only a capitalized key name and a colon is redacted",
			msg:  "password=Secret:",
			want: "password=[REDACTED]",
		},
		{
			name: "dotted key-shaped value with no value after it is redacted whole",
			msg:  "secret: SECRETR.password=",
			want: "secret:[REDACTED]",
		},
		{
			name: "nested key followed by a newline and a delimiter is kept and its value redacted",
			msg:  "credentials: token\n= SECRETA",
			want: "credentials: token=[REDACTED]",
		},
		{
			name: "nested key followed by a tab and a delimiter is kept and its value redacted",
			msg:  "credentials: token\t= SECRETA",
			want: "credentials: token=[REDACTED]",
		},
		{
			name: "nested key followed by a spaced colon is kept and its value redacted",
			msg:  "credentials: token : SECRETA",
			want: "credentials: token:[REDACTED]",
		},
		{
			name: "credentials object in text is redacted whole",
			msg:  "credentials: {token: SECRET123, refresh: SECRET456}",
			want: "credentials:[REDACTED]",
		},
		{
			name: "credentials list in text is redacted whole",
			msg:  "credentials: [token: SECRET123, refresh: SECRET456]",
			want: "credentials:[REDACTED]",
		},
		{
			name: "credentials envelope printed with %+v is redacted whole",
			msg:  "assume role failed: {Credentials:{AccessKeyID:EXAMPLEKEYID SecretAccessKey:SECRET123 SessionToken:SECRET456}}",
			want: "assume role failed: {Credentials:[REDACTED]}",
		},
		{
			name: "object under a sensitive key is redacted whole",
			msg:  `auth failed: {"token": {"value": "SECRET123"}}`,
			want: `auth failed: {"token":"[REDACTED]"}`,
		},
		{
			name: "token object under a sensitive key is redacted whole",
			msg:  `auth failed: {"token": {"access_token": "SECRET123", "expires_in": 3600}} retrying`,
			want: `auth failed: {"token":"[REDACTED]"} retrying`,
		},
		{
			name: "bare value with a parenthesis is redacted whole",
			msg:  "failed: password=abc)SECRET123",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "bare value starting with a parenthesis is redacted whole",
			msg:  "failed: password=)SECRET123",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "bare value starting with slashes is redacted whole",
			msg:  "failed: password: //SECRET123",
			want: "failed: password:[REDACTED]",
		},
		{
			name: "bare value with a bracket is redacted whole",
			msg:  "failed: password=abc]SECRET123",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "bare value with a quote is redacted whole",
			msg:  `failed: password=abc"SECRET123`,
			want: "failed: password=[REDACTED]",
		},
		{
			name: "bare value with a closing brace is redacted whole",
			msg:  "failed: password=ab}SECRET123",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "bare value ending with a closing brace is redacted whole",
			msg:  "failed: password=SECRET123}",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "bare value starting with a closing brace is redacted whole",
			msg:  "failed: password=}SECRET123",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "basic credential that decodes to user:password is redacted",
			msg:  "auth failed: basic dXNlcjpwYXNz",
			want: "auth failed: basic [REDACTED]",
		},
		{
			name: "padded basic credential is redacted",
			msg:  "auth failed: Basic dXNlcg==",
			want: "auth failed: Basic [REDACTED]",
		},
		{
			name: "unpadded basic value without a colon is kept",
			msg:  "auth failed: basic YWJj",
			want: "auth failed: basic YWJj",
		},
		{
			name: "basic followed by a word is kept",
			msg:  "basic auth not configured",
			want: "basic auth not configured",
		},
		{
			name: "basic in prose is kept",
			msg:  "Write a basic HTTP server in Go",
			want: "Write a basic HTTP server in Go",
		},
		{
			name: "bearer of in prose is kept",
			msg:  "I'm the bearer of bad news",
			want: "I'm the bearer of bad news",
		},
		{
			name: "bearer token starting with of is redacted",
			msg:  "auth failed: Bearer of-SECRET123",
			want: "auth failed: Bearer [REDACTED]",
		},
		{
			name: "markdown link mentioning basic is kept",
			msg:  "[Basic usage](https://example.com/docs)",
			want: "[Basic usage](https://example.com/docs)",
		},
		{
			name: "quoted sensitive word in prose is kept",
			msg:  `[Q] What does "token" mean?`,
			want: `[Q] What does "token" mean?`,
		},
		{
			name: "URL with basic in a query value is kept",
			msg:  "GET https://example.com/search?q=basic%20recipes failed",
			want: "GET https://example.com/search?q=basic%20recipes failed",
		},
		{
			name: "key with a Kelvin sign, which case-insensitive matching folds to k, is redacted",
			msg:  "failed: TO\u212aEN=SECRET123",
			want: "failed: TO\u212aEN=[REDACTED]",
		},
		{
			name: "key with a long s, which case-insensitive matching folds to s, is redacted",
			msg:  "failed: pa\u017f\u017fword=SECRET123",
			want: "failed: pa\u017f\u017fword=[REDACTED]",
		},
		{
			name: "camelCase keys in key=value and key: value text are redacted",
			msg:  "auth failed: accessToken=SECRET123 apiKey: SECRET456",
			want: "auth failed: accessToken=[REDACTED] apiKey:[REDACTED]",
		},
		{
			name: "x_goog_api_key in key=value text is redacted",
			msg:  "request failed: x_goog_api_key=SECRET123",
			want: "request failed: x_goog_api_key=[REDACTED]",
		},
		{
			name: "x-goog-api-key header is redacted",
			msg:  "request failed: x-goog-api-key: SECRET123",
			want: "request failed: x-goog-api-key: [REDACTED]",
		},
		{
			name: "Go struct printed with %+v has credential fields redacted",
			msg:  "bad config: {APIKey:SECRET123 ClientSecret:SECRET456 Region:us}",
			want: "bad config: {APIKey:[REDACTED] ClientSecret:[REDACTED] Region:us}",
		},
		{
			name: "Go struct printed with %+v ending with credential fields keeps its closing brace",
			msg:  "bad config: {APIKey:SECRET123 ClientSecret:SECRET456}",
			want: "bad config: {APIKey:[REDACTED] ClientSecret:[REDACTED]}",
		},
		{
			name: "Go struct printed with %+v whose credential field holds a closing brace",
			msg:  "bad config: {Password:ab}SECRET123}",
			want: "bad config: {Password:[REDACTED]}",
		},
		{
			name: "Go struct printed with %+v whose credential field holds a brace pair",
			msg:  "bad config: {Password:ab{SECRET123} Region:us}",
			want: "bad config: {Password:[REDACTED] Region:us}",
		},
		{
			name: "nested Go structs printed with %+v keep their closing braces",
			msg:  "bad config: {Auth:{Token:SECRET123}} and {Auth:{Token:SECRET456} Region:us}",
			want: "bad config: {Auth:{Token:[REDACTED]}} and {Auth:{Token:[REDACTED]} Region:us}",
		},
		{
			name: "slice of Go structs and Go map printed with %+v keep their closing brackets",
			msg:  "bad config: [{APIKey:SECRET123}] map[password:SECRET456]",
			want: "bad config: [{APIKey:[REDACTED]}] map[password:[REDACTED]]",
		},
		{
			name: "Go struct printed with %+v and quoted with %q keeps its closing brace and quote",
			msg:  `load failed: "{Region:us APIKey:SECRET123}": invalid`,
			want: `load failed: "{Region:us APIKey:[REDACTED]}": invalid`,
		},
		{
			name: "quoted credentials ending with a closing bracket keep their quotes",
			msg:  `bad config: {'password': 'SECRET123}'} [{'token': 'SECRET456]'}] config.Auth{Password:"SECRET789}"}`,
			want: `bad config: {'password':'[REDACTED]'} [{'token':'[REDACTED]'}] config.Auth{Password:"[REDACTED]"}`,
		},
		{
			name: "JSON quoted with %q whose credential ends with a closing brace keeps its quotes",
			msg:  `load failed: "{\"password\":\"SECRET123}\"}"`,
			want: `load failed: "{\"password\":\"[REDACTED]\"}"`,
		},
		{
			name: "Go struct printed with %+v whose credential field holds a closing brace before another field",
			msg:  "bad config: {Password:ab}cd Token:SECRET123}",
			want: "bad config: {Password:[REDACTED] Token:[REDACTED]}",
		},
		{
			name: "Go struct printed with %+v in parentheses keeps its closing brace and parenthesis",
			msg:  "({APIKey:SECRET123})",
			want: "({APIKey:[REDACTED]})",
		},
		{
			name: "Go struct printed with %+v in single quotes ending a sentence keeps its closing brace, quote and period",
			msg:  "'{APIKey:SECRET123}'.",
			want: "'{APIKey:[REDACTED]}'.",
		},
		{
			name: "Go struct printed with %+v in JSON quoted with %q keeps its closing brace and escaped quote",
			msg:  `bad config: "{\"msg\":\"{APIKey:SECRET123}\"}"`,
			want: `bad config: "{\"msg\":\"{APIKey:[REDACTED]}\"}"`,
		},
		{
			name: "closing brace that closes no bracket is redacted with the value",
			msg:  "] password=SECRET123}",
			want: "] password=[REDACTED]",
		},
		{
			name: "closing bracket that closes no bracket does not hide a later open brace",
			msg:  "unexpected ] in {Token:SECRET123}",
			want: "unexpected ] in {Token:[REDACTED]}",
		},
		{
			name: "AWS credentials printed with %+v are redacted",
			msg:  "assume role failed: {AccessKeyID:EXAMPLEKEYID SecretAccessKey:SECRET123 SessionToken:SECRET456}",
			want: "assume role failed: {AccessKeyID:EXAMPLEKEYID SecretAccessKey:[REDACTED] SessionToken:[REDACTED]}",
		},
		{
			name: "empty credential field in a Go struct printed with %+v is kept",
			msg:  "bad config: {Password:}",
			want: "bad config: {Password:}",
		},
		{
			name: "JSON error with a camelCase sensitive key",
			msg:  `{"accessToken":"SECRET123","message":"denied"}`,
			want: `{"accessToken":"[REDACTED]","message":"denied"}`,
		},
		{
			name: "embedded JSON with a camelCase sensitive key",
			msg:  `token exchange failed: {"accessToken": "SECRET123", "expiresIn": 3600}`,
			want: `token exchange failed: {"accessToken":"[REDACTED]", "expiresIn": 3600}`,
		},
		{
			name: "session user and app keys in key=value text are redacted",
			msg:  "failed: user:api_key=SECRET123 app:clientSecret=SECRET456",
			want: "failed: user:api_key=[REDACTED] app:clientSecret=[REDACTED]",
		},
		{
			name: "keys that only contain a sensitive key name are kept",
			msg:  "failed: tokenCount=5 api_key_id=abc secretName=projects/p/secrets/s",
			want: "failed: tokenCount=5 api_key_id=abc secretName=projects/p/secrets/s",
		},
		{
			name: "hyphenated sensitive keys are redacted",
			msg:  "failed: x-api-key=SECRET123 refresh-token=SECRET456",
			want: "failed: x-api-key=[REDACTED] refresh-token=[REDACTED]",
		},
		{
			name: "command-line flags are redacted",
			msg:  "mysql -u root --password=SECRET123 failed",
			want: "mysql -u root --password=[REDACTED] failed",
		},
		{
			name: "command-line flag with a hyphenated key is redacted",
			msg:  "--api-key=SECRET123",
			want: "--api-key=[REDACTED]",
		},
		{
			name: "command-line flags with hyphenated token keys are redacted",
			msg:  "exec failed: tool --token=SECRET123 --access-token=SECRET456 github-token=SECRET789",
			want: "exec failed: tool --token=[REDACTED] --access-token=[REDACTED] github-token=[REDACTED]",
		},
		{
			name: "hyphenated header ending in token is redacted",
			msg:  "X-Auth-Token: SECRET123",
			want: "X-Auth-Token:[REDACTED]",
		},
		{
			name: "upper-case hyphenated header ending in token is redacted",
			msg:  "PRIVATE-TOKEN: SECRET123",
			want: "PRIVATE-TOKEN:[REDACTED]",
		},
		{
			name: "credential in a nested URL in text is redacted in place",
			msg:  "see https://example.com/login?next=https://api.example.com/cb?access_token=SECRET123 for details",
			want: "see https://example.com/login?next=https%3A%2F%2Fapi.example.com%2Fcb%3Faccess_token%3D%5BREDACTED%5D for details",
		},
		{
			name: "credential in a nested URL that ends a sentence is redacted in place",
			msg:  "redirected to https://example.com/login?next=https://api.example.com/cb?access_token=SECRET123.",
			want: "redirected to https://example.com/login?next=https%3A%2F%2Fapi.example.com%2Fcb%3Faccess_token%3D%5BREDACTED%5D.",
		},
		{
			name: "credential in a nested URL in parentheses is redacted in place",
			msg:  "see (https://example.com/login?next=https://api.example.com/cb?access_token=SECRET123) for details",
			want: "see (https://example.com/login?next=https%3A%2F%2Fapi.example.com%2Fcb%3Faccess_token%3D%5BREDACTED%5D) for details",
		},
		{
			name: "URL with a credential query value and another escape is redacted in place",
			msg:  "GET https://example.com/cb?access_token=SECRET123&q=a%20b failed",
			want: "GET https://example.com/cb?access_token=%5BREDACTED%5D&q=a%20b failed",
		},
		{
			name: "URL with a credential path segment and another escape is redacted in place",
			msg:  "GET https://example.com/v1/access_token=SECRET123 failed at 100%25",
			want: "GET https://example.com/v1/access_token=%5BREDACTED%5D failed at 100%25",
		},
		{
			name: "object quoted with %q whose string field holds a closing brace is redacted whole",
			msg:  `auth failed: "{\"token\": {\"label\": \"a},\", \"val\": \"SECRET123\"}}"`,
			want: `auth failed: "{\"token\":[REDACTED]}"`,
		},
		{
			name: "array quoted with %q whose string field holds a closing bracket is redacted whole",
			msg:  `auth failed: "{\"token\": [\"a],\", \"SECRET123\"], \"expires_in\": 3600}"`,
			want: `auth failed: "{\"token\":[REDACTED], \"expires_in\": 3600}"`,
		},
		{
			name: "quote-led prose with brackets is kept",
			msg:  `"Hello," she said [laughing]. "Bye."`,
			want: `"Hello," she said [laughing]. "Bye."`,
		},
		{
			name: "quoted JSON string whose escape hides a bearer token fails closed",
			msg:  `"Bearer\tSECRET123"`,
			want: unparseableJSONPlaceholder,
		},
		{
			name: "JSON object where earlier token value ends with escaped newline before value ending in key name",
			msg:  `{"msg":"token=abc\npassword: dev-SECRET123-password: dial"}`,
			want: `{"msg":"token=[REDACTED]\npassword:[REDACTED]"}`,
		},
		{
			name: "prefixed JSON object where earlier token swallows escaped newline before value ending in key name",
			msg:  `POST /v1/login: 400: {"msg":"token=abc\npassword: dev-SECRET123-password: dial"}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "JSON object where earlier compound key value ends with escaped newline before password",
			msg:  `{"msg":"secret: kv/data/db-password: other\npassword: SECRET123"}`,
			want: `{"msg":"secret:[REDACTED]\npassword:[REDACTED]"}`,
		},
		{
			name: "prefixed JSON object where earlier compound key value swallows escaped newline before password",
			msg:  `POST /v1/login: 400: {"msg":"secret: kv/data/db-password: other\npassword: SECRET123"}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "JSON object where password colon is followed by escaped newline and space",
			msg:  `{"msg":"password:\n SECRET123"}`,
			want: `{"msg":"password:[REDACTED]"}`,
		},
		{
			name: "prefixed JSON object where password colon is followed by escaped newline and space",
			msg:  `POST /v1/login: 400: {"msg":"password:\n SECRET123"}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "prefixed single-quoted Python dict where earlier token swallows escaped newline before password",
			msg:  `bad config: {'error': 'token=abc\npassword: SECRET123'}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "truncated JSON error body where earlier token swallows escaped newline before password",
			msg:  `upstream error: {"error":"token=abc\npassword: SECRET123`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "bare token value ending with escaped quote before password key",
			msg:  `token=abc\"password\": SECRET123`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "prefixed quoted string where earlier token swallows escaped newline before password",
			msg:  `upstream failed: "token=abc\npassword: SECRET123"`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "prefixed JSON object where earlier token swallows percent-encoded newline before password",
			msg:  `POST /v1/login: 400: {"msg":"token=abc%0Apassword: SECRET123"}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "prefixed JSON object where earlier token swallows double-escaped unicode newline before password",
			msg:  `POST /v1/login: 400: {"msg":"token=abc\\u000apassword: SECRET123"}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "prefixed JSON object where earlier token swallows escaped newline and unicode colon before password",
			msg:  `POST /v1/login: 400: {"msg":"token=abc\npassword\u003a SECRET123"}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "JSON object where earlier token swallows unicode-escaped backslash and n before password",
			msg:  `{"msg":"token=abc\u005cnpassword: SECRET123"}`,
			want: `{"msg":"[REDACTED_SENSITIVE_TEXT]"}`,
		},
		{
			name: "prefixed JSON object where earlier token swallows unicode-escaped backslash and n before password",
			msg:  `POST /v1/login: 400: {"msg":"token=abc\u005cnpassword: SECRET123"}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "JSON object where earlier token swallows percent-escaped backslash and n before password",
			msg:  `{"msg":"token=abc%5Cnpassword: SECRET123"}`,
			want: `{"msg":"[REDACTED_SENSITIVE_TEXT]"}`,
		},
		{
			name: "prefixed JSON object where earlier token swallows hex-escaped backslash and n before password",
			msg:  `POST /v1/login: 400: {"msg":"token=abc\x5cnpassword: SECRET123"}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "prefixed JSON object where earlier token swallows unicode-escaped backslash and unicode n before password",
			msg:  `POST /v1/login: 400: {"msg":"token=abc\u005c\u006epassword: SECRET123"}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "JSON object where password colon is followed by escaped backspace and space",
			msg:  `{"msg":"password:\b SECRET123"}`,
			want: `{"msg":"password:[REDACTED]"}`,
		},
		{
			name: "prefixed JSON object where password colon is followed by escaped backspace and space",
			msg:  `POST /v1/login: 400: {"msg":"password:\b SECRET123"}`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "plain text where password colon is followed by escaped backspace and space",
			msg:  `password:\b SECRET123`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "JSON object where password colon is followed by unicode backspace and space",
			msg:  `{"msg":"password:\u0008 SECRET123"}`,
			want: `{"msg":"password:[REDACTED]"}`,
		},
		{
			name: "JSON object where password colon is followed by unicode vertical tab and space",
			msg:  `{"msg":"password:\u000b SECRET123"}`,
			want: `{"msg":"password:[REDACTED]"}`,
		},
		{
			name: "JSON object where password colon is followed by unicode NUL and space",
			msg:  `{"msg":"password:\u0000 SECRET123"}`,
			want: `{"msg":"password:[REDACTED]"}`,
		},
		{
			name: "Go struct field ending in key name followed by space, closing brace, backspace byte and secret",
			msg:  "failed: {Password:abc-token: }\bSECRET123",
			want: "failed: {Password:[REDACTED]",
		},
		{
			name: "Go struct field ending in key name followed by space, backspace byte and closing brace",
			msg:  "failed: {Password:abc-token: \b}",
			want: "failed: {Password:[REDACTED] \b}",
		},
		{
			name: "caller-supplied redaction placeholders splitting secret after escaped newline fail closed",
			msg:  `password:\n [REDACTED]a[REDACTED]u[REDACTED]t[REDACTED]h[REDACTED] authorization=1 authorization=2`,
			want: redactedSensitiveTextPlaceholder,
		},
		{
			name: "caller-supplied redaction placeholders with nonce prefix splitting secret after escaped newline fail closed",
			msg:  `password:\n [REDACTED]zzRedactedLiteral[REDACTED]a[REDACTED]u[REDACTED]t[REDACTED]h[REDACTED] authorization=1 authorization=2`,
			want: redactedSensitiveTextPlaceholder,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := redactErrorMessage(tt.msg)
			if got != tt.want {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", tt.msg, got, tt.want)
			}
			if twice := redactErrorMessage(got); twice != got {
				t.Errorf("redactErrorMessage is not idempotent: second pass(%q) = %q", got, twice)
			}
		})
	}
}

func TestSanitizeRawText(t *testing.T) {
	t.Run("deeply quoted credential text remains unparseable blob", func(t *testing.T) {
		payload := `{"\u0061pi_key": "secret_credential_value"`
		for range 8 {
			b, err := json.Marshal(payload)
			if err != nil {
				t.Fatalf("json.Marshal: %v", err)
			}
			payload = string(b)
		}
		got, lost := sanitizeRawText(payload)
		if got != unparseableJSONPlaceholder {
			t.Errorf("sanitizeRawText with 8 quoted layers got %q, want %s", got, unparseableJSONPlaceholder)
		}
		if !lost {
			t.Errorf("sanitizeRawText with 8 quoted layers reported lost = false, want true")
		}
	})

	t.Run("raw content with truncated temp key and Unicode-padded sensitive keys fails closed", func(t *testing.T) {
		samples := []string{
			`{"temp:cache/value":"SYNTHETIC_SECRET"`,
			"{\"\u00a0api_key\u00a0\":\"SYNTHETIC_SECRET\"",
			`{"\u00a0api_key\u00a0":"SYNTHETIC_SECRET"`,
			"{\" \u00a0temp:path/to/key\u00a0 \":\"secret\"",
		}
		for _, s := range samples {
			got, lost := sanitizeRawText(s)
			if got != unparseableJSONPlaceholder || !lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", s, got, lost, unparseableJSONPlaceholder)
			}
		}
	})

	t.Run("bare temp assignment with padded or colon-bearing value redacts the whole value", func(t *testing.T) {
		tests := []struct {
			raw  string
			want string
		}{
			{"failed: temp:cache=SYNTHETIC_SECRET==", "failed: temp:cache=[REDACTED]"},
			{"failed: temp:cache:SYNTHETIC_SECRET==", "failed: temp:cache:[REDACTED]"},
			{"failed: temp:cache=urn:uuid:12345", "failed: temp:cache=[REDACTED]"},
			{"failed: temp:cache:urn:uuid:12345", "failed: temp:cache:[REDACTED]"},
			{"failed: temp:cache/val=SYNTHETIC_SECRET==", "failed: temp:cache/val=[REDACTED]"},
		}
		for _, tt := range tests {
			got, lost := sanitizeRawText(tt.raw)
			if got != tt.want || !lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", tt.raw, got, lost, tt.want)
			}
		}
	})

	t.Run("harmless bracket-led prose without quotes or escapes is preserved", func(t *testing.T) {
		prose := "[docs](https://cloud.google.com/bigquery)"
		got, lost := sanitizeRawText(prose)
		if got != prose || lost {
			t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, false)", prose, got, lost, prose)
		}
	})

	t.Run("prose mentioning sensitive words without credential patterns is preserved", func(t *testing.T) {
		proseSamples := []string{
			"[Help] I forgot my password",
			"[INFO] refreshed the access token",
			"[Docs] see the credentials page",
			"[Note] your password was reset",
		}
		for _, prose := range proseSamples {
			got, lost := sanitizeRawText(prose)
			if got != prose || lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, false)", prose, got, lost, prose)
			}
		}
	})

	t.Run("bracket-led prose with quotes and Windows paths is preserved", func(t *testing.T) {
		proseSamples := []string{
			`[INFO] user clicked "Submit"`,
			`[1] Smith, "On Analytics", 2020`,
			`{name} said "hello" to the team`,
			`[path] C:\Users\me\report.txt was saved`,
			`[regex] use \d+ to match digits`,
			`My reset link https://example.com/reset?email=bob%40example.com says the token expired`,
			`I typed 100%25 of my password correctly`,
			`Fetched https://example.com/items?q=red%20shoes; the token expired after 3 retries`,
		}
		for _, prose := range proseSamples {
			got, lost := sanitizeRawText(prose)
			if got != prose || lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, false)", prose, got, lost, prose)
			}
		}
	})

	t.Run("large JSON documents over 64 KiB are inspected and preserved", func(t *testing.T) {
		largeObj := map[string]any{
			"document": strings.Repeat("A", 100*1024),
			"api_key":  "secret_key_val",
		}
		got, truncated, err := SmartTruncate(largeObj, 500*1024)
		if err != nil {
			t.Fatalf("SmartTruncate: %v", err)
		}
		if truncated {
			t.Errorf("expected truncated=false for large JSON under max limit")
		}
		if !strings.Contains(string(got), redactedValue) {
			t.Errorf("expected api_key to be redacted in large JSON")
		}
		if !strings.Contains(string(got), "AAAA") {
			t.Errorf("expected document content to be preserved")
		}
	})

	t.Run("text without credentials is unchanged", func(t *testing.T) {
		proseSamples := []string{
			"Write a basic HTTP server in Go",
			"I'm the bearer of bad news",
			"basic auth not configured",
			"[Basic usage](https://example.com/docs)",
			`[Q] What does "token" mean?`,
			"See https://example.com/search?q=basic%20recipes for ideas",
			`"Hello," she said [laughing]. "Bye."`,
			`"Hello," she said {quietly}. "Bye."`,
			`"What does [this] mean?"`,
		}
		for _, prose := range proseSamples {
			got, lost := sanitizeRawText(prose)
			if got != prose || lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, false)", prose, got, lost, prose)
			}
		}
	})

	t.Run("quote-led text with a JSON string that decodes to a credential fails closed", func(t *testing.T) {
		// The checks on the raw text do not decode \t, so these are only
		// caught by reading the quoted strings as JSON.
		samples := []string{
			`"Bearer\tSECRET123"`,
			`"Bearer\tSECRET123" was rejected`,
			`"Hi" then "Bearer\tSECRET123"`,
		}
		for _, s := range samples {
			got, lost := sanitizeRawText(s)
			if got != unparseableJSONPlaceholder || !lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", s, got, lost, unparseableJSONPlaceholder)
			}
		}
	})

	t.Run("credentials in quoted or nested values are redacted", func(t *testing.T) {
		tests := []struct {
			raw  string
			want string
		}{
			{`token exchange failed: "{\"access_token\": \"SECRET123\"}"`, `token exchange failed: "{\"access_token\":\"[REDACTED]\"}"`},
			{`failed: password=\"SECRET123\"`, `failed: password=\"[REDACTED]\"`},
			{`bad config: {'api_key': 'SECRET123'}`, `bad config: {'api_key':'[REDACTED]'}`},
			{
				`auth failed: {"credentials": {"access_token": "SECRET123", "expires_in": 3600}}`,
				`auth failed: {"credentials":"[REDACTED]"}`,
			},
			{
				`assume role failed: {"Credentials": {"AccessKeyId": "EXAMPLEKEYID", "SecretAccessKey": "SECRET123", "SessionToken": "SECRET123"}}`,
				`assume role failed: {"Credentials":"[REDACTED]"}`,
			},
			{`auth failed: {"token": {"value": "SECRET123"}}`, `auth failed: {"token":"[REDACTED]"}`},
			{"failed: password=abc)SECRET123", "failed: password=[REDACTED]"},
			{"failed: password=ab}SECRET123", "failed: password=[REDACTED]"},
			{`bad config: {'password': 'SECRET123}'}`, `bad config: {'password':'[REDACTED]'}`},
			{"bad config: {APIKey:SECRET123 ClientSecret:SECRET123}", "bad config: {APIKey:[REDACTED] ClientSecret:[REDACTED]}"},
			{"failed: password: //SECRET123", "failed: password:[REDACTED]"},
			{
				`auth failed: "{\"token\": {\"label\": \"a},\", \"val\": \"SECRET123\"}}"`,
				`auth failed: "{\"token\":[REDACTED]}"`,
			},
			{
				"see https://example.com/login?next=https://api.example.com/cb?access_token=SECRET123 for details",
				"see https://example.com/login?next=https%3A%2F%2Fapi.example.com%2Fcb%3Faccess_token%3D%5BREDACTED%5D for details",
			},
			{
				"GET https://example.com/cb?access_token=SECRET123&q=a%20b failed",
				"GET https://example.com/cb?access_token=%5BREDACTED%5D&q=a%20b failed",
			},
			{"config:\ncredentials:\n  token: SECRET123\n  region: us", "config:\ncredentials:\n  token:[REDACTED]\n  region: us"},
			{"config:\ncredentials:\n  password: SECRET123", "config:\ncredentials:\n  password:[REDACTED]"},
			{"config:\n  credentials:\n    client_secret: SECRET123", "config:\n  credentials:\n    client_secret:[REDACTED]"},
			{"credentials:\n  X-Auth-Token: SECRET123", "credentials:\n  X-Auth-Token:[REDACTED]"},
			{"credentials:\n  'token': 'SECRET123'", "credentials:\n  'token':'[REDACTED]'"},
			{"loading credentials: token: SECRET123", "loading credentials: token:[REDACTED]"},
			{"credentials: token = SECRET123", "credentials: token=[REDACTED]"},
			{"Credentials:\n  SessionToken: SECRET123", "Credentials:\n  SessionToken:[REDACTED]"},
			{"credentials:\n  token:\n" + strings.Repeat(" ", 100) + "SECRET123", "credentials:\n  token:[REDACTED]"},
			{"credentials:\n  db/password: SECRET123", "credentials:\n  db/password:[REDACTED]"},
			{"credentials: db/password = SECRET123", "credentials:[REDACTED]"},
			{"connect user=bob password=dev-SECRET123-password: dial tcp 10.0.0.1:5432: connection refused", "connect user=bob password=[REDACTED] tcp 10.0.0.1:5432: connection refused"},
			{`login failed: password="SECRET123-secret": invalid`, `login failed: password="[REDACTED]"`},
			{"api_key=prodSECRET123-api-key: permission denied", "api_key=[REDACTED]"},
			{"my login is token=abc-SECRET123-token: please check", "my login is token=[REDACTED] check"},
			{"credentials: X-Amz-Security-Token: SECRET123", "credentials:[REDACTED]"},
			{"password=a-SECRET123-token: token: SECRET456 retry", "password=[REDACTED] retry"},
			{"credentials: {token: SECRET123, refresh: SECRET456}", "credentials:[REDACTED]"},
			{`password: 'token: SECRETA SECRETB'`, `password:'[REDACTED]'`},
			{`token: "password=SECRETA SECRETB"`, `token:"[REDACTED]"`},
			{`password: "token:SECRETA SECRETB"`, `password:"[REDACTED]"`},
			{`bad config: {'api_key': 'sig=SECRETA SECRETB'}`, `bad config: {'api_key':'[REDACTED]'}`},
			{"password=SECRETP-token:SECRETB", "password=[REDACTED]"},
			{"password=SECRETP.secret=SECRETB", "password=[REDACTED]"},
			{"api_key=SECRETP.sig=SECRETB", "api_key=[REDACTED]"},
			{`bad config: {'credentials': 'secret=correct horse SECRETB staple'}`, `bad config: {'credentials':'[REDACTED]'}`},
			{`bad config: {'credential': 'token: abc SECRETB'}`, `bad config: {'credential':'[REDACTED]'}`},
			{`credentials="token=abc SECRETB"`, `credentials="[REDACTED]"`},
			{`CREDENTIALS="secret=correct horse SECRETB"`, `CREDENTIALS="[REDACTED]"`},
			{"credentials: SECRETP-token:SECRETB", "credentials:[REDACTED]"},
			{"credentials: 'secret=correct horse SECRETB'", "credentials:'[REDACTED]'"},
			{`failed: password: "password"`, `failed: password:"[REDACTED]"`},
			{`failed: password: "SECRET123": invalid`, `failed: password:"[REDACTED]": invalid`},
			{"credentials: token= SECRET123", "credentials: token=[REDACTED]"},
			{"failed: password: SECRET123 : retry", "failed: password:[REDACTED] : retry"},
			{"failed: password=SECRET123:[REDACTED]", "failed: password=[REDACTED]"},
			{"credentials:\n  app:client_secret: SECRET123", "credentials:\n  app:client_secret:[REDACTED]"},
			{"failed: password: SECRET123token: retry", "failed: password:[REDACTED] retry"},
			{"password=SECRETP-token:", "password=[REDACTED]"},
			{"password=SECRETP/token=", "password=[REDACTED]"},
			{"api_key=SECRETAbC+xYz/Sig=", "api_key=[REDACTED]"},
			{"api_key=SECRETAbC+xYz/Sig=\n", "api_key=[REDACTED]\n"},
			{"api_key=SECRETAbC+xYz/Sig= ", "api_key=[REDACTED] "},
			{"password=SECRETP-token:,next", "password=[REDACTED],next"},
			{"password=SECRETP-token:;next", "password=[REDACTED];next"},
			{"password=SECRETP-token: ,next", "password=[REDACTED] ,next"},
			{"password=SECRETP-token:&next=1", "password=[REDACTED]&next=1"},
			{"password=SECRETP-token:\t;x", "password=[REDACTED]\t;x"},
			{"password=SECRETP-token:\r\n", "password=[REDACTED]\r\n"},
			{"password=SECRETP-token:\f", "password=[REDACTED]\f"},
			{"failed: password=SECRETP-token: }", "failed: password=[REDACTED] }"},
			{"failed: {Password:abc-token: }", "failed: {Password:[REDACTED] }"},
			{"api_key=SECRETAbC+xYz/Sig= }", "api_key=[REDACTED] }"},
			{"password=SECRETP-token: ]", "password=[REDACTED] ]"},
			{"failed: [{Password:abc-token: } {User:bob}]", "failed: [{Password:[REDACTED] } {User:bob}]"},
			{"password=abc-token: ...", "password=[REDACTED] ..."},
			{"credentials: token = ]SECRETZ", "credentials: token=[REDACTED]"},
			{"credentials:\n  password: }SECRETZ", "credentials:\n  password:[REDACTED]"},
			{"failed: {Credentials:token: ]SECRETZ}", "failed: {Credentials:token:[REDACTED]}"},
			{`credentials: token: " SECRETX"`, `credentials: token:"[REDACTED]"`},
			{`credentials: token: ' SECRETX'`, `credentials: token:'[REDACTED]'`},
			{`credentials: token: \" SECRETX\"`, `credentials: token:\"[REDACTED]\"`},
			{"credentials:\n  token: \" SECRETX\"\n  region: us", "credentials:\n  token:\"[REDACTED]\"\n  region: us"},
			{`failed: "password=SECRETP-token: "`, `failed: "password=[REDACTED] "`},
			{`failed: 'password=SECRETP-token: '`, `failed: 'password=[REDACTED] '`},
			{`failed: \"password=SECRETP-token: \"`, `failed: \"password=[REDACTED] \"`},
			{`parsing "password=SECRETP-token: ": invalid syntax`, `parsing "password=[REDACTED] ": invalid syntax`},
			{`failed: "password=SECRETP-token: " for user "bob"`, `failed: "password=[REDACTED] " for user "bob"`},
			{`failed: "password=SECRETP-token: " for user "Émile"`, `failed: "password=[REDACTED] " for user "Émile"`},
			{`got "password=SECRETP-token: ", want ""`, `got "password=[REDACTED] ", want ""`},
			{`got "password=SECRETP-token: ", want "2"`, `got "password=[REDACTED] ", want "2"`},
			{`got "password=SECRETP-token: ", want "_id"`, `got "password=[REDACTED] ", want "_id"`},
			{`got \"password=SECRETP-token: \", want \"\"`, `got \"password=[REDACTED] \", want \"\"`},
			{"credentials:\n  password: \" SECRET123\"\n  region: us", "credentials:\n  password:\"[REDACTED]\"\n  region: us"},
			{`credentials: 'token': ' SECRET123'`, `credentials: 'token':'[REDACTED]'`},
			{`credentials: token = ". SECRET123"`, `credentials: token="[REDACTED]"`},
			{`credentials: token: "} SECRET123"`, `credentials: token:"[REDACTED]"`},
			{`failed: {Credentials:token: " SECRET123"}`, `failed: {Credentials:token:"[REDACTED]"}`},
			{`failed: [{Credentials:token: " SECRET123"}]`, `failed: [{Credentials:token:"[REDACTED]"}]`},
			{`failed: {Password:SECRET123-token: "}`, `failed: {Password:[REDACTED] "}`},
			{`failed: {Password:SECRET123-token: '}`, `failed: {Password:[REDACTED] '}`},
			{`failed: {Password:SECRET123-token: ""}`, `failed: {Password:[REDACTED] ""}`},
			{`failed: map[Password:SECRET123-token: ""]`, `failed: map[Password:[REDACTED] ""]`},
			{`password=SECRET123-token: \`, `password=[REDACTED] \`},
			{`credentials: password="correct SECRET123 battery"`, `credentials: password="[REDACTED]"`},
			{`credentials: token=" SECRET123"`, `credentials: token="[REDACTED]"`},
			{`credentials: token:"Bearer SECRET123"`, `credentials: token:"[REDACTED]"`},
			{`credentials: token='a,SECRET123'`, `credentials: token='[REDACTED]'`},
			{`credentials: token="a;SECRET123" more`, `credentials: token="[REDACTED]" more`},
			{`credentials: token=\"a SECRET123\"`, `credentials: token=\"[REDACTED]\"`},
			{`password: token:"a SECRET123"`, `password: token:"[REDACTED]"`},
			{`failed: {Credentials:token=" SECRET123"}`, `failed: {Credentials:token="[REDACTED]"}`},
			{`failed: map[credentials:token="a SECRET123"]`, `failed: map[credentials:token="[REDACTED]"]`},
			{"credentials:\n  token=\"a SECRET123\"\n  region: us", "credentials:\n  token=\"[REDACTED]\"\n  region: us"},
			{`credentials: token=SECRET123`, `credentials:[REDACTED]`},
			{`credentials: token="SECRET123"`, `credentials:[REDACTED]`},
			{`got "password=SECRETP-token=", want "abc"`, `got "password=[REDACTED], want "abc"`},
			{`credentials: user="SECRET123 token=SECRET456"`, `credentials:[REDACTED] token=[REDACTED]`},
			{`credentials: token("SECRET123 token=SECRET456")`, `credentials:[REDACTED] token=[REDACTED]`},
			{`password="SECRETQ-token" :`, `password="[REDACTED]" :`},
			{"password=SECRETQ-token :", "password=[REDACTED] :"},
			{"password=SECRETP-token\t:", "password=[REDACTED]\t:"},
			{"password = SECRETP-token:", "password=[REDACTED]"},
			{"password=temp:SECRETT:", "password=[REDACTED]"},
			{"password=secret:", "password=[REDACTED]"},
			{"password=Secret:", "password=[REDACTED]"},
			{"secret: SECRETR.password=", "secret:[REDACTED]"},
			{"credentials: token\n= SECRETA", "credentials: token=[REDACTED]"},
			{"credentials: token\t= SECRETA", "credentials: token=[REDACTED]"},
			{"credentials: token : SECRETA", "credentials: token:[REDACTED]"},
			{"password=temp:SECRET123-token: dial tcp", "password=[REDACTED] tcp"},
			{"password=temp:SECRET123: dial tcp", "password=[REDACTED] tcp"},
			{"password=user:temp:SECRET123: dial tcp", "password=[REDACTED] tcp"},
			{"password=app:temp:SECRET123: dial tcp", "password=[REDACTED] tcp"},
			{`password="temp:SECRET123": dial tcp`, `password="[REDACTED]" tcp`},
			{"credentials: temp:foo: SECRET123", "credentials:[REDACTED]"},
			{"credentials:\n  temp:foo: SECRET123", "credentials:\n  temp:foo:[REDACTED]"},
			{"credentials:\n  user:temp:foo: SECRET123", "credentials:\n  user:temp:foo:[REDACTED]"},
			{`{"msg":"retry\npassword=SECRET123 vault: secret: kv/data/db-password: \"OTHER pass\"","code":"token=abc"}`, redactedSensitiveTextPlaceholder},
			{`POST /v1/login: 400: {"msg":"retry\npassword=SECRET123 vault: secret: kv/data/db-password: \"OTHER pass\""}`, redactedSensitiveTextPlaceholder},
			{`POST /v1/login [400]: {"msg":"retry\npassword=SECRET123 vault: secret: kv/data/db-password: \"OTHER pass\""}`, redactedSensitiveTextPlaceholder},
			{`[ERROR] POST /v1/login: 400: {"msg":"retry\npassword=SECRET123"}`, unparseableJSONPlaceholder},
			{`[ERROR] POST /v1/login: 400: {"msg":"retry\npassword=SECRET123 vault: secret: kv/data/db-password: \"OTHER pass\""}`, unparseableJSONPlaceholder},
			{`POST /v1/login: {"msg":"retry\npassword=SECRET123"} (status 400)`, redactedSensitiveTextPlaceholder},
			{`{"msg":"password=SECRET123"}`, `{"msg":"password=[REDACTED]}`},
			{`{"msg":"retry\npassword=dev-SECRET123-password: dial"}`, redactedSensitiveTextPlaceholder},
			{`{"msg":"retry\npassword=hunter2val token=abc-SECRET123-token: dial"}`, redactedSensitiveTextPlaceholder},
			{`["retry\npassword=dev-SECRET123-password: dial"]`, redactedSensitiveTextPlaceholder},
			{`{"msg":"retry\npassword=dev-SECRET123-password"}`, redactedSensitiveTextPlaceholder},
			{`["retry\npassword=dev-SECRET123-password"]`, redactedSensitiveTextPlaceholder},
			{`bad token " in body: {"msg":"retry\npassword=SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`flag \"foo: {"msg":"retry\npassword=SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`err: "dir C:\work\"retry\npassword=SECRET123"`, redactedSensitiveTextPlaceholder},
			{`upstream error: {"error":"login failed\npassword=SECRET123`, redactedSensitiveTextPlaceholder},
			{`{'error': 'retry\npassword=SECRET123'}`, unparseableJSONPlaceholder},
			{`bad config: {'error': 'retry\npassword=SECRET123'}`, redactedSensitiveTextPlaceholder},
			{`err=retry\npassword=SECRET123`, redactedSensitiveTextPlaceholder},
			{`retry\bpassword=SECRET123`, redactedSensitiveTextPlaceholder},
			{`retry\fpassword=SECRET123`, redactedSensitiveTextPlaceholder},
			{`retry\npassword=SECRET123`, redactedSensitiveTextPlaceholder},
			{`retry\rpassword=SECRET123`, redactedSensitiveTextPlaceholder},
			{`retry\tpassword=SECRET123`, redactedSensitiveTextPlaceholder},
			{`https:\/\/user:SECRET123@example.com`, redactedSensitiveTextPlaceholder},
			{"Basic%20" + base64.StdEncoding.EncodeToString([]byte(strings.Repeat("a", 300)+":SECRET123:"+strings.Repeat("b", 285))), redactedSensitiveTextPlaceholder},
			{`Basic\t` + base64.StdEncoding.EncodeToString([]byte(strings.Repeat("a", 300)+":SECRET123:"+strings.Repeat("b", 285))), redactedSensitiveTextPlaceholder},
			{"https%3A//maps.googleapis.com/maps/api/geocode/json?address=" + strings.Repeat("a", 300) + "&key=SECRET123&sensor=" + strings.Repeat("b", 300), redactedSensitiveTextPlaceholder},
			{`https:\/\/maps.googleapis.com/maps/api/geocode/json?address=` + strings.Repeat("a", 300) + "&key=SECRET123&sensor=" + strings.Repeat("b", 300), redactedSensitiveTextPlaceholder},
			{"https%3A//" + strings.Repeat("a", 260) + ":SECRET123@example.com/" + strings.Repeat("b", 260), redactedSensitiveTextPlaceholder},
			{strings.Repeat(`\`, 256) + `npassword=SECRET123`, redactedSensitiveTextPlaceholder},
			{strings.Repeat(`\`, 256) + `tpassword=SECRET123`, redactedSensitiveTextPlaceholder},
			{"password" + strings.Repeat(`\`, 256) + "u003dSECRET123", redactedSensitiveTextPlaceholder},
			{strings.Repeat(`\`, 600) + `"password": "SECRET123"`, strings.Repeat(`\`, 600) + `"password":"[REDACTED]"`},
			{`Bearer\\u0020SECRET123`, redactedSensitiveTextPlaceholder},
			{`{"msg":"token=abc\npassword: dev-SECRET123-password: dial"}`, redactedSensitiveTextPlaceholder},
			{`POST /v1/login: 400: {"msg":"token=abc\npassword: dev-SECRET123-password: dial"}`, redactedSensitiveTextPlaceholder},
			{`{"msg":"secret: kv/data/db-password: other\npassword: SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`POST /v1/login: 400: {"msg":"secret: kv/data/db-password: other\npassword: SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`{"msg":"password:\n SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`POST /v1/login: 400: {"msg":"password:\n SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`bad config: {'error': 'token=abc\npassword: SECRET123'}`, redactedSensitiveTextPlaceholder},
			{`upstream error: {"error":"token=abc\npassword: SECRET123`, redactedSensitiveTextPlaceholder},
			{`token=abc\"password\": SECRET123`, redactedSensitiveTextPlaceholder},
			{`upstream failed: "token=abc\npassword: SECRET123"`, redactedSensitiveTextPlaceholder},
			{`POST /v1/login: 400: {"msg":"token=abc%0Apassword: SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`POST /v1/login: 400: {"msg":"token=abc\\u000apassword: SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`POST /v1/login: 400: {"msg":"token=abc\npassword\u003a SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`{"msg":"token=abc\u005cnpassword: SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`POST /v1/login: 400: {"msg":"token=abc\u005cnpassword: SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`{"msg":"token=abc%5Cnpassword: SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`POST /v1/login: 400: {"msg":"token=abc\x5cnpassword: SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`POST /v1/login: 400: {"msg":"token=abc\u005c\u006epassword: SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`{"msg":"password:\b SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`POST /v1/login: 400: {"msg":"password:\b SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`password:\b SECRET123`, redactedSensitiveTextPlaceholder},
			{`{"msg":"password:\u0008 SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`{"msg":"password:\u000b SECRET123"}`, redactedSensitiveTextPlaceholder},
			{`{"msg":"password:\u0000 SECRET123"}`, redactedSensitiveTextPlaceholder},
			{"failed: {Password:abc-token: }\bSECRET123", "failed: {Password:[REDACTED]"},
			{"failed: {Password:abc-token: \b}", "failed: {Password:[REDACTED] \b}"},
			{"credentials:\n  password=SECRET123-token: OTHER456", "credentials:\n  password=[REDACTED]"},
			{`password=abc-token: " SECRET123"`, "password=[REDACTED]"},
			{`password=abc-token: ' SECRET123'`, "password=[REDACTED]"},
			{`password=abc-token: \" SECRET123\"`, `password=[REDACTED]`},
			{`password:\n [REDACTED]a[REDACTED]u[REDACTED]t[REDACTED]h[REDACTED] authorization=1 authorization=2`, redactedSensitiveTextPlaceholder},
			{`GET https://example.com?token=SECRET123 hello\nworld`, `GET https://example.com?token=%5BREDACTED%5D hello\nworld`},
		}
		for _, tt := range tests {
			got, lost := sanitizeRawText(tt.raw)
			if got != tt.want || !lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", tt.raw, got, lost, tt.want)
			}
			truncJSON, truncated, err := SmartTruncate(tt.raw, 4096)
			var trunc string
			if err != nil || json.Unmarshal(truncJSON, &trunc) != nil || trunc != tt.want || !truncated {
				t.Errorf("SmartTruncate(%q) = (%s, %v, %v), want (%q, true, nil)", tt.raw, truncJSON, truncated, err, tt.want)
			}
		}
		for _, depth := range []int{5, 9} {
			nested := `"retry\npassword=SECRET123"`
			for range depth {
				b, err := json.Marshal(nested)
				if err != nil {
					t.Fatalf("json.Marshal: %v", err)
				}
				nested = string(b)
			}
			prefixed := "upstream: " + nested
			if got := redactErrorMessage(prefixed); got != redactedSensitiveTextPlaceholder {
				t.Errorf("redactErrorMessage(%d-deep) = %q, want %q", depth, got, redactedSensitiveTextPlaceholder)
			}
			if got, lost := sanitizeRawText(prefixed); got != redactedSensitiveTextPlaceholder || !lost {
				t.Errorf("sanitizeRawText(%d-deep) = (%q, %v), want (%q, true)", depth, got, lost, redactedSensitiveTextPlaceholder)
			}
		}
		if got, want := scrubErrorText(`{"msg":"password=SECRET123"}`), `{"msg":"password=[REDACTED]}`; got != want {
			t.Errorf("scrubErrorText unescaped JSON pair = %q, want %q", got, want)
		}
		for _, harmless := range []string{
			`C:\new\file.txt`,
			`{"msg":"talk to Ken\ntomorrow"}`,
			`GET [REDACTED_SENSITIVE_URI] [REDACTED] %5BREDACTED%5D hello\nworld`,
			`{"sep":"` + strings.Repeat(`\\`, 128) + `"}`,
			`{"sep":"` + strings.Repeat(`\\`, 300) + `"}`,
			`"` + strings.Repeat(`\`, 300) + `"`,
			`hello: "` + strings.Repeat(`\`, 256) + `"`,
			"design review notes; " + strings.Repeat(`\`, 256) + "next",
			strings.Repeat(`\`, 256) + `"token`,
			strings.Repeat(`\`, 256) + `'token`,
			"path " + strings.Repeat(`\`, 256) + "/end",
			"mytoken=1" + strings.Repeat(" ", 56) + "a-b: x",
			"foosig: 1" + strings.Repeat(" ", 52) + "temp: ",
		} {
			if got := redactErrorMessage(harmless); got != harmless {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", harmless, got, harmless)
			}
			if got, lost := sanitizeRawText(harmless); got != harmless || lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, false)", harmless, got, lost, harmless)
			}
		}
		for _, pair := range []struct{ raw, want string }{
			{`{"msg":"hello\nworld","token":"SECRET123"}`, `{"msg":"hello\nworld","token":"[REDACTED]"}`},
			{`{"path":"C:\\users\\deadbeef","token":"SECRET123"}`, `{"path":"C:\\users\\deadbeef","token":"[REDACTED]"}`},
			{`{"msg":"hello [REDACTED]\nworld","token":"SECRET123"}`, `{"msg":"hello [REDACTED]\nworld","token":"[REDACTED]"}`},
			{`{"sep":"` + strings.Repeat(`\\`, 128) + `","token":"SECRET123"}`, `{"sep":"` + strings.Repeat(`\\`, 128) + `","token":"[REDACTED]"}`},
		} {
			if got, lost := sanitizeRawText(pair.raw); got != pair.want || lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, false)", pair.raw, got, lost, pair.want)
			}
		}
		if got, want := nestedPairEnd("token=abc rest", len("token=abc "), 0), len("token=abc "); got != want {
			t.Errorf("nestedPairEnd fallback = %d, want %d", got, want)
		}
	})

	t.Run("quoted fragments are inspected in linear time", func(t *testing.T) {
		filler := strings.Repeat("b", 100_000)
		samples := map[string]string{
			// Each fragment fails to decode at its line break.
			"broken fragments": `"a" ` + strings.Repeat("\"x\n", 64) + filler,
			// A key word lets the text through the credential pattern
			// prefilter, so scanning the rest of the text again for each
			// fragment would show.
			"broken fragments with a key word": `"a" ` + strings.Repeat("\"token\n", 64) + filler,
			// The fragment never ends, and each escaped quote in it starts
			// another fragment that would be decoded to the end of the text.
			"unterminated fragment with escaped quotes": `"a" "` + strings.Repeat(`x\"`, 62) + filler,
		}
		for name, text := range samples {
			start := time.Now()
			got, lost := sanitizeRawText(text)
			if d := time.Since(start); d > 3*time.Second {
				t.Errorf("%s: sanitizeRawText took %v for a 100 KB text", name, d)
			}
			if got != text || lost {
				t.Errorf("%s: sanitizeRawText changed text without credentials (lost = %v)", name, lost)
			}
		}
	})

	t.Run("nested sensitive keys are inspected in linear time", func(t *testing.T) {
		samples := map[string]string{
			// Each value is the next key, so each pair is matched again from
			// its value.
			"keys separated by spaces": strings.Repeat("token: ", 3_000) + "SECRET123",
			"credentials blocks":       strings.Repeat("credentials:\n  ", 2_000) + "token: SECRET123",
			// The value ends with a delimiter and a value follows it, as for a
			// key, but it is too long to be one, so it is redacted whole
			// rather than matched again from each key in it.
			"keys without spaces": "password=SECRET123-" + strings.Repeat("token:", 10_000) + " x",
		}
		for name, text := range samples {
			start := time.Now()
			got, lost := sanitizeRawText(text)
			if d := time.Since(start); d > 3*time.Second {
				t.Errorf("%s: sanitizeRawText took %v for a %d-byte text", name, d, len(text))
			}
			if strings.Contains(got, "SECRET123") {
				t.Errorf("%s: sanitizeRawText kept the credential", name)
			}
			if !lost {
				t.Errorf("%s: sanitizeRawText lost = false, want true", name)
			}
		}
	})

	t.Run("pattern words with long backslash runs are inspected in linear time", func(t *testing.T) {
		unchanged := map[string]string{
			"300 backslashes": "design review notes; " + strings.Repeat(`\`, 300) + " end",
			"50KB prose":      strings.Repeat("design review notes; ", 2_500) + strings.Repeat(`\`, 256) + " end",
			"105KB prose":     strings.Repeat("design review notes; ", 5_000) + strings.Repeat(`\`, 256) + " end",
			"55KB multiline colon prose with backslashes":               strings.Repeat("design: review=notes;\n", 2_500) + strings.Repeat(`\`, 256) + " end",
			"55KB multiline colon prose with quoted backslashes":        strings.Repeat("design: review=notes;\n", 2_500) + `"` + strings.Repeat(`\`, 300) + `"`,
			"110KB multiline colon prose with quoted backslashes":       strings.Repeat("design: review=notes;\n", 5_000) + `"` + strings.Repeat(`\`, 300) + `"`,
			"53KB contractions and basic prose with quoted backslashes": strings.Repeat("a basic user's design; ", 2_300) + `"` + strings.Repeat(`\`, 300) + `"`,
			"105KB quoted colons with Windows path":                     strings.Repeat(`see "design": user notes; `, 4_100) + `C:\\users\\alice`,
		}
		for name, text := range unchanged {
			start := time.Now()
			if got := redactErrorMessage(text); got != text {
				t.Errorf("%s: redactErrorMessage redacted clean text (len=%d)", name, len(got))
			}
			if d := time.Since(start); d > 3*time.Second {
				t.Errorf("%s: redactErrorMessage took %v for a %d-byte text", name, d, len(text))
			}
			start = time.Now()
			got, lost := sanitizeRawText(text)
			if d := time.Since(start); d > 3*time.Second {
				t.Errorf("%s: sanitizeRawText took %v for a %d-byte text", name, d, len(text))
			}
			if got != text || lost {
				t.Errorf("%s: sanitizeRawText modified clean text (len=%d, lost=%v)", name, len(got), lost)
			}
		}

		withCred := strings.Repeat("design review notes; ", 2_500) + `retry\npassword=SECRET123`
		start := time.Now()
		got, lost := sanitizeRawText(withCred)
		if d := time.Since(start); d > 3*time.Second {
			t.Errorf("escaped credential after 50KB prose: sanitizeRawText took %v", d)
		}
		if got != redactedSensitiveTextPlaceholder || !lost {
			t.Errorf("escaped credential after 50KB prose: sanitizeRawText = (%q, %v), want (%q, true)", got, lost, redactedSensitiveTextPlaceholder)
		}
	})

	t.Run("keys quoted with escapes and padded with a Unicode space fail closed", func(t *testing.T) {
		// The quotes around each key are written as escapes, and a Unicode
		// space before the delimiter keeps the decoded pair from matching
		// the credential patterns, so only the check for a quoted sensitive
		// key in the decoded text finds it.
		samples := []string{
			"[\\u0022AUTHORIZATION\\u0022\u00a0: \\u0022SECRET123\\u0022]",
			"{\\u0022password\\u0022\u00a0: \\u0022SECRET123\\u0022}",
			"[%22password%22\u3000: %22SECRET123%22]",
			"{\\x22api_key\\x22\u00a0= \\x22SECRET123\\x22}",
			"{\\u0027secret\\u0027\u2009:\\u0027SECRET123\\u0027}",
			// The same with literal quotes, and without the Unicode space.
			"[\"password\"\u00a0: \"SECRET123\"]",
			"[\\u0022password\\u0022: \\u0022SECRET123\\u0022]",
		}
		for _, s := range samples {
			if got, lost := sanitizeRawText(s); got != unparseableJSONPlaceholder || !lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", s, got, lost, unparseableJSONPlaceholder)
			}
			if got := redactErrorMessage(s); got != unparseableJSONPlaceholder {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", s, got, unparseableJSONPlaceholder)
			}
		}
	})

	t.Run("escaped keys and delimiters across long whitespace runs fail closed", func(t *testing.T) {
		// Only the spans of a text over 8 KiB around its escapes are
		// decoded, so each span must reach across the whitespace between
		// the key, the delimiter and the value, however long it is.
		prose := strings.Repeat("lorem ipsum dolor sit amet\n", 340)
		spaces := func(n int) string { return strings.Repeat(" ", n) }
		tests := []struct {
			name string
			raw  string
			// want is the result of sanitizeRawText, redactErrorMessage and
			// SmartTruncate. scrubErrorText returns
			// redactedSensitiveTextPlaceholder for every raw.
			want string
		}{
			{"escaped key, spaces, delimiter", "pass\\u0077ord" + spaces(9000) + ": SECRET123", redactedSensitiveTextPlaceholder},
			{"escaped key and delimiter, spaces", "pass\\u0077ord:" + spaces(9000) + "SECRET123", redactedSensitiveTextPlaceholder},
			{"key, spaces, escaped delimiter", "password" + spaces(9000) + "\\u003a SECRET123", redactedSensitiveTextPlaceholder},
			{"escaped key, spaces around the delimiter", prose + "pass\\u0077ord" + spaces(6000) + ":" + spaces(6000) + "SECRET123", redactedSensitiveTextPlaceholder},
			{"long word, escaped key, spaces, delimiter", strings.Repeat("x", 8300) + " pass\\u0077ord" + spaces(5000) + ": SECRET123", redactedSensitiveTextPlaceholder},
			{"escaped key, spaces and a line break, delimiter", "pass\\u0077ord" + spaces(3000) + "\n" + spaces(3000) + ": SECRET123" + spaces(3000), redactedSensitiveTextPlaceholder},
			// The span reaches 512 bytes past the escape before it is
			// widened.
			{"escaped key, 400 spaces and a line break, delimiter", prose + "pass\\u0077ord" + spaces(400) + "\n: SECRET123", redactedSensitiveTextPlaceholder},
			{"escaped key, 513 spaces and a line break, delimiter", prose + "pass\\u0077ord" + spaces(513) + "\n: SECRET123", redactedSensitiveTextPlaceholder},
			{"escaped key and delimiter, 520 spaces and a line break", prose + "pass\\u0077ord:" + spaces(520) + "\nSECRET123", redactedSensitiveTextPlaceholder},
			{"key and escaped delimiter, 600 spaces and a line break", prose + "password\\u003a" + spaces(600) + "\nSECRET123", redactedSensitiveTextPlaceholder},
			{"escaped key, 600 line breaks, delimiter", prose + "pass\\u0077ord" + strings.Repeat("\n", 600) + ": SECRET123", redactedSensitiveTextPlaceholder},
			{"escaped key and delimiter, 520 tabs and a line break", prose + "api\\u005fkey =" + strings.Repeat("\t", 520) + "\nSECRET123", redactedSensitiveTextPlaceholder},
			{"key, spaces, delimiter, spaces, value in escaped quotes", prose + "password" + spaces(6000) + ":" + spaces(6000) + "\\u0022a SECRET123\\u0022", redactedSensitiveTextPlaceholder},
			{"escaped tab, bearer, spaces, token", strings.Repeat("a ", 5000) + "\\tBearer" + spaces(9000) + "SECRET123", redactedSensitiveTextPlaceholder},
			{"quoted key, 5,000 spaces, escaped delimiter, prose", "{\"password\"" + spaces(5000) + "\\u003a \"SECRET123\"}" + strings.Repeat(" x", 2000), unparseableJSONPlaceholder},
			// Spans are joined across a gap of up to 64 KiB between quotes.
			{"quoted key, 60,000 spaces, escaped delimiter", "{\"password\"" + spaces(60_000) + "\\u003a \"SECRET123\"}", unparseableJSONPlaceholder},
			{"quoted key, 70,000 spaces, escaped delimiter", "{\"password\"" + spaces(70_000) + "\\u003a \"SECRET123\"}", unparseableJSONPlaceholder},
		}
		for _, tt := range tests {
			if got, lost := sanitizeRawText(tt.raw); got != tt.want || !lost {
				t.Errorf("%s: sanitizeRawText = (%.80q, %v), want (%q, true)", tt.name, got, lost, tt.want)
			}
			if got := redactErrorMessage(tt.raw); got != tt.want {
				t.Errorf("%s: redactErrorMessage = %.80q, want %q", tt.name, got, tt.want)
			}
			if got := scrubErrorText(tt.raw); got != redactedSensitiveTextPlaceholder {
				t.Errorf("%s: scrubErrorText = %.80q, want %q", tt.name, got, redactedSensitiveTextPlaceholder)
			}
			truncJSON, _, err := SmartTruncate(tt.raw, 1<<20)
			var trunc string
			if err != nil || json.Unmarshal(truncJSON, &trunc) != nil || trunc != tt.want {
				t.Errorf("%s: SmartTruncate = (%.80s, %v), want %q", tt.name, truncJSON, err, tt.want)
			}
		}
	})

	t.Run("a quote after an even run of backslashes is not an escaped quote", func(t *testing.T) {
		// The backslashes escape each other, so the quote does not hide a
		// quoted key from the checks on the raw text, and prose that ends
		// a Windows path with it is kept.
		prose := `[INFO] saved to "C:\\" ok`
		if got := redactErrorMessage(prose); got != prose {
			t.Errorf("redactErrorMessage(%q) = %q, want %q", prose, got, prose)
		}
		// A key quoted that way is still found, as is one quoted with
		// escaped quotes.
		for _, s := range []string{
			`[INFO] \\"password\\": \\"SECRET123\\"`,
			`[INFO] \"password\": \"SECRET123\"`,
		} {
			if got, lost := sanitizeRawText(s); got != unparseableJSONPlaceholder || !lost {
				t.Errorf("sanitizeRawText(%q) = (%q, %v), want (%q, true)", s, got, lost, unparseableJSONPlaceholder)
			}
			if got := redactErrorMessage(s); got != unparseableJSONPlaceholder {
				t.Errorf("redactErrorMessage(%q) = %q, want %q", s, got, unparseableJSONPlaceholder)
			}
		}
	})

	t.Run("key-value pairs with one kind of delimiter are redacted in linear time", func(t *testing.T) {
		// Each text repeats sensitive key-value pairs that use only ":" or
		// only "=". Looking for the next delimiter of the other kind anew
		// after each match would scan the rest of the text once per match.
		scale := 4
		if raceEnabled {
			// The race detector slows the scans about twentyfold.
			scale = 64
		}
		for _, tc := range []struct {
			name string
			unit string
			size int
			// kept are the parts of unit that redaction must remove.
			kept []string
		}{
			{"colons", "token: a: b: c: d ", 4 << 20, []string{"token: a"}},
			{"equals signs", "'a'='b' secret='' ", 4 << 20, []string{"secret=''"}},
			// Both the JSON and the text pass match each pair with a
			// regular expression, which takes longer per byte.
			{"JSON members", `"token": "x", "password": "y", `, 1 << 20, []string{`"x"`, `"y"`}},
		} {
			text := strings.Repeat(tc.unit, tc.size/scale/len(tc.unit))
			for _, redact := range []struct {
				name string
				fn   func(string) string
			}{
				{"sanitizeRawText", func(s string) string { got, _ := sanitizeRawText(s); return got }},
				{"redactErrorMessage", redactErrorMessage},
				{"scrubErrorText", scrubErrorText},
			} {
				start := time.Now()
				got := redact.fn(text)
				if d := time.Since(start); d > 3*time.Second {
					t.Errorf("%s: %s took %v for a %d-byte text", tc.name, redact.name, d, len(text))
				}
				for _, k := range tc.kept {
					if strings.Contains(got, k) {
						t.Errorf("%s: %s kept %q", tc.name, redact.name, k)
					}
				}
			}
		}
	})
}

func TestSmartTruncate_NodeBudgetLimit(t *testing.T) {
	t.Run("oversized slice stops at budget with single sentinel", func(t *testing.T) {
		items := make([]any, maxSanitizeNodes*2)
		for i := range items {
			items[i] = "x"
		}
		got, truncated, err := SmartTruncate(items, 100)
		if err != nil {
			t.Fatalf("SmartTruncate() error: %v", err)
		}
		if !truncated {
			t.Error("SmartTruncate() truncated = false, want true when node budget is exceeded")
		}
		if count := bytes.Count(got, []byte(sanitizeBudgetPlaceholder)); count != 1 {
			t.Errorf("SmartTruncate() emitted %s %d times, want 1", sanitizeBudgetPlaceholder, count)
		}
		var parsed []any
		if err := json.Unmarshal(got, &parsed); err != nil {
			t.Fatalf("json.Unmarshal: %v", err)
		}
		if len(parsed) > maxSanitizeNodes {
			t.Errorf("parsed slice len = %d, want <= %d", len(parsed), maxSanitizeNodes)
		}
		if parsed[len(parsed)-1] != sanitizeBudgetPlaceholder {
			t.Errorf("last element = %v, want %s", parsed[len(parsed)-1], sanitizeBudgetPlaceholder)
		}
	})

	t.Run("oversized sensitive temp map charges budget and stops", func(t *testing.T) {
		m := make(map[string]any, maxSanitizeNodes+500)
		for i := range maxSanitizeNodes + 500 {
			m["temp:k_"+strconv.Itoa(i)] = "secret"
		}
		got, truncated, err := SmartTruncate(m, 100)
		if err != nil {
			t.Fatalf("SmartTruncate() error: %v", err)
		}
		if !truncated {
			t.Error("SmartTruncate() truncated = false, want true when temp map exceeds node budget")
		}
		if !bytes.Contains(got, []byte(sanitizeBudgetPlaceholder)) {
			t.Errorf("SmartTruncate() did not emit %s when temp map exceeded node budget", sanitizeBudgetPlaceholder)
		}
		var parsed map[string]any
		if err := json.Unmarshal(got, &parsed); err != nil {
			t.Fatalf("json.Unmarshal: %v", err)
		}
		if len(parsed) > maxSanitizeNodes {
			t.Errorf("parsed map len = %d, want <= %d", len(parsed), maxSanitizeNodes)
		}
	})

	t.Run("JSON text shares sanitizer node budget", func(t *testing.T) {
		items := make([]any, maxSanitizeNodes-5)
		for i := range items {
			items[i] = "x"
		}
		items = append(items, `[1,2,3,4,5,6,7,8,9,10]`)
		got, truncated, err := SmartTruncate(items, 100)
		if err != nil {
			t.Fatalf("SmartTruncate() error: %v", err)
		}
		if !truncated {
			t.Error("SmartTruncate() truncated = false, want true when JSON text exhausts shared node budget")
		}
		if !bytes.Contains(got, []byte(sanitizeBudgetPlaceholder)) {
			t.Errorf("SmartTruncate() did not emit %s when JSON text exhausted shared node budget", sanitizeBudgetPlaceholder)
		}
	})
}

func TestContentSummaries(t *testing.T) {
	content := &genai.Content{Parts: []*genai.Part{
		{Text: "hello world"},
		{FunctionCall: &genai.FunctionCall{Name: "lookup"}},
		{FunctionResponse: &genai.FunctionResponse{Name: "lookup"}},
		{InlineData: &genai.Blob{MIMEType: "image/png"}},
	}}
	jsonSecretContent := &genai.Content{Parts: []*genai.Part{
		{Text: `{"api_key":"secret","note":"0123456789"}`},
	}}
	multiPartJSONSecretContent := &genai.Content{Parts: []*genai.Part{
		{Text: `{"api_key":"secret"}`},
		{Text: "plain follow-up"},
	}}
	markdownLinkContent := &genai.Content{Parts: []*genai.Part{
		{Text: "[link](https://example.test)"},
	}}
	bracketTagContent := &genai.Content{Parts: []*genai.Part{
		{Text: "[INFO] ready"},
	}}
	tests := []struct {
		name      string
		format    func(*genai.Content, int, bool) (string, bool)
		content   *genai.Content
		maxLength int
		noRedact  bool
		want      string
		wantTrunc bool
	}{
		{name: "summary", format: contentSummary, content: content, maxLength: 100, want: "hello world | Function response: lookup"},
		{name: "truncated summary", format: contentSummary, content: content, maxLength: 5, want: "hello...[TRUNCATED]", wantTrunc: true},
		{name: "summary of nil", format: contentSummary, maxLength: 100, want: ""},
		{
			name: "summary sanitizes JSON before truncating", format: contentSummary,
			content: jsonSecretContent, maxLength: 25,
			want: `{"api_key":"[REDACTED]","...[TRUNCATED]`, wantTrunc: true,
		},
		{
			name: "multi-part summary sanitizes JSON per part before joining", format: contentSummary,
			content: multiPartJSONSecretContent, maxLength: 200,
			want: `{"api_key":"[REDACTED]"} | plain follow-up`,
		},
		{
			name: "summary preserves markdown link starting with bracket", format: contentSummary,
			content: markdownLinkContent, maxLength: 200,
			want: "[link](https://example.test)", wantTrunc: false,
		},
		{
			name: "summary preserves bracketed tag", format: contentSummary,
			content: bracketTagContent, maxLength: 200,
			want: "[INFO] ready", wantTrunc: false,
		},
		{name: "response", format: formatResponseContent, content: content, maxLength: 100, want: "text: 'hello world' | call: lookup | resp: lookup | other"},
		{
			name: "truncated response", format: formatResponseContent, content: content, maxLength: 5,
			want: "text: 'hello...[TRUNCATED]' | call: lookup | resp: lookup | other", wantTrunc: true,
		},
		{
			name: "response sanitizes JSON before truncating", format: formatResponseContent,
			content: jsonSecretContent, maxLength: 25,
			want: `text: '{"api_key":"[REDACTED]","...[TRUNCATED]'`, wantTrunc: true,
		},
		{name: "response of nil", format: formatResponseContent, maxLength: 100, want: "None"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, truncated := tt.format(tt.content, tt.maxLength, !tt.noRedact)
			if got != tt.want || truncated != tt.wantTrunc {
				t.Errorf("got (%q, %v), want (%q, %v)", got, truncated, tt.want, tt.wantTrunc)
			}
		})
	}
}

func TestFormatContentParts_RedactionAndPreservation(t *testing.T) {
	t.Run("function response with URL and query parameter preserves trailing query params", func(t *testing.T) {
		content := &genai.Content{
			Parts: []*genai.Part{
				{
					FunctionResponse: &genai.FunctionResponse{
						Name: "convert_currency",
						Response: map[string]any{
							"error": "API error: GET https://api.exchangerate.host/convert?api_key=SECRET_12345&to=USD",
						},
					},
				},
			},
		}

		parts := FormatContentParts(content, 1000)
		if len(parts) != 1 {
			t.Fatalf("len(parts) = %d, want 1", len(parts))
		}
		attrs, ok := parts[0]["part_attributes"].(string)
		if !ok {
			t.Fatalf("part_attributes type = %T, want string", parts[0]["part_attributes"])
		}
		if strings.Contains(attrs, "SECRET_12345") {
			t.Errorf("part_attributes leaked credential: %s", attrs)
		}
		if !strings.Contains(attrs, "api_key=%5BREDACTED%5D") && !strings.Contains(attrs, "api_key=[REDACTED]") {
			t.Errorf("part_attributes did not redact api_key: %s", attrs)
		}
		var decoded struct {
			FunctionResponse struct {
				Response struct {
					Error string `json:"error"`
				} `json:"response"`
			} `json:"function_response"`
		}
		if err := json.Unmarshal([]byte(attrs), &decoded); err != nil {
			t.Fatalf("json.Unmarshal(attrs): %v", err)
		}
		if !strings.Contains(decoded.FunctionResponse.Response.Error, "&to=USD") {
			t.Errorf("decoded error %q lost query param &to=USD", decoded.FunctionResponse.Response.Error)
		}
	})

	t.Run("plain text part with Authorization header scrubs credential", func(t *testing.T) {
		content := &genai.Content{
			Parts: []*genai.Part{
				{
					Text: "Failed with error: Authorization: Bearer secret-bearer-token-12345",
				},
			},
		}

		parts := FormatContentParts(content, 1000)
		if len(parts) != 1 {
			t.Fatalf("len(parts) = %d, want 1", len(parts))
		}
		text, ok := parts[0]["text"].(string)
		if !ok {
			t.Fatalf("text type = %T, want string", parts[0]["text"])
		}
		if strings.Contains(text, "secret-bearer-token-12345") {
			t.Errorf("text leaked credential: %s", text)
		}
		if !strings.Contains(text, "Authorization: [REDACTED]") && !strings.Contains(text, "Bearer [REDACTED]") {
			t.Errorf("text did not redact credential: %s", text)
		}
	})
}

// collectParts returns the collector after it has taken in the messages,
// which are numbered from 0 as the messages of a request are.
func collectParts(opts partsOptions, contents ...*genai.Content) *partsCollector {
	c := &partsCollector{opts: opts}
	for i, content := range contents {
		c.addContent(content, i)
	}
	return c
}

// singlePart returns a message that holds part.
func singlePart(part *genai.Part) *genai.Content {
	return &genai.Content{Parts: []*genai.Part{part}}
}

// partAttributes decodes the part_attributes of an entry.
func partAttributes(t *testing.T, entry map[string]any) any {
	t.Helper()
	s, ok := entry["part_attributes"].(string)
	if !ok {
		t.Fatalf("part_attributes is %T, want a JSON string; entry: %v", entry["part_attributes"], entry)
	}
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		t.Fatalf("part_attributes is not JSON: %v\n%s", err, s)
	}
	return v
}

func TestFormatContentParts_StructuredParts(t *testing.T) {
	tests := []struct {
		name string
		part *genai.Part
		// The entry apart from part_attributes, which is compared as JSON.
		wantMIMEType, wantText, wantAttrs string
	}{
		{
			name:         "function call",
			part:         &genai.Part{FunctionCall: &genai.FunctionCall{Name: "lookup", Args: map[string]any{"q": "weather"}}},
			wantMIMEType: "application/json",
			wantText:     "Function: lookup",
			wantAttrs:    `{"function_call":{"args":{"q":"weather"},"name":"lookup"},"function_name":"lookup"}`,
		},
		{
			name:         "function response",
			part:         &genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "lookup", Response: map[string]any{"temp": 21}}},
			wantMIMEType: "application/json",
			wantText:     "Function response: lookup",
			wantAttrs:    `{"function_response":{"name":"lookup","response":{"temp":21}}}`,
		},
		{
			name:         "function response without a name",
			part:         &genai.Part{FunctionResponse: &genai.FunctionResponse{Response: map[string]any{"ok": true}}},
			wantMIMEType: "application/json",
			wantText:     "Function response: unknown",
			wantAttrs:    `{"function_response":{"response":{"ok":true}}}`,
		},
		{
			name:         "executable code",
			part:         &genai.Part{ExecutableCode: &genai.ExecutableCode{Code: "print(1)", Language: genai.LanguagePython}},
			wantMIMEType: "text/plain",
			wantText:     "print(1)",
			wantAttrs:    `{"executable_code":{"code":"print(1)","language":"PYTHON"}}`,
		},
		{
			name:         "code execution result",
			part:         &genai.Part{CodeExecutionResult: &genai.CodeExecutionResult{Outcome: genai.OutcomeOK, Output: "1"}},
			wantMIMEType: "text/plain",
			wantText:     "1",
			wantAttrs:    `{"code_execution_result":{"outcome":"OUTCOME_OK","output":"1"}}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, truncated := formatContentParts(singlePart(tt.part), 100, true)

			if truncated {
				t.Error("formatContentParts() truncated = true, want false")
			}
			if len(got) != 1 {
				t.Fatalf("formatContentParts() returned %d entries, want 1: %v", len(got), got)
			}
			checkJSON(t, "part_attributes", partAttributes(t, got[0]), tt.wantAttrs)
			delete(got[0], "part_attributes")
			want := map[string]any{"part_index": 0, "storage_mode": "INLINE", "mime_type": tt.wantMIMEType, "text": tt.wantText}
			if diff := cmp.Diff(want, got[0]); diff != "" {
				t.Errorf("entry mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestPartsCollector_PartAttributesRedactSensitiveKeys(t *testing.T) {
	tests := []struct {
		name   string
		redact bool
		want   string
	}{
		{
			name:   "redaction on",
			redact: true,
			want:   `{"function_response":{"name":"login","response":{"api_key":"[REDACTED]","user":"ann"}}}`,
		},
		{
			name:   "redaction off",
			redact: false,
			want:   `{"function_response":{"name":"login","response":{"api_key":"SECRET-KEY","user":"ann"}}}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			part := &genai.Part{FunctionResponse: &genai.FunctionResponse{
				Name: "login", Response: map[string]any{"api_key": "SECRET-KEY", "user": "ann"},
			}}

			c := collectParts(partsOptions{maxLen: 100, redact: tt.redact}, singlePart(part))

			if len(c.parts) != 1 {
				t.Fatalf("got %d entries, want 1: %v", len(c.parts), c.parts)
			}
			checkJSON(t, "part_attributes", partAttributes(t, c.parts[0]), tt.want)
		})
	}
}

// TestPartsCollector_TruncatedFlag checks which parts make the row truncated.
func TestPartsCollector_TruncatedFlag(t *testing.T) {
	const maxLen = 30
	long := strings.Repeat("x", 50)
	tests := []struct {
		name string
		part *genai.Part
		want bool
	}{
		{name: "short text", part: &genai.Part{Text: "short"}},
		{name: "text at the limit", part: &genai.Part{Text: strings.Repeat("x", maxLen)}},
		{name: "long text", part: &genai.Part{Text: long}, want: true},
		{name: "inline data", part: &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: make([]byte, 1000)}}},
		{name: "function call", part: &genai.Part{FunctionCall: &genai.FunctionCall{Name: "lookup"}}},
		{name: "function call with a long name", part: &genai.Part{FunctionCall: &genai.FunctionCall{Name: long}}, want: true},
		{
			name: "function response",
			part: &genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "f", Response: map[string]any{"k": "v"}}},
		},
		{
			name: "function response with a long value",
			part: &genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "f", Response: map[string]any{"k": long}}},
			want: true,
		},
		{name: "executable code", part: &genai.Part{ExecutableCode: &genai.ExecutableCode{Code: "print(1)"}}},
		{name: "long executable code", part: &genai.Part{ExecutableCode: &genai.ExecutableCode{Code: long}}, want: true},
		{name: "long code output", part: &genai.Part{CodeExecutionResult: &genai.CodeExecutionResult{Output: long}}, want: true},
		{name: "file URI", part: &genai.Part{FileData: &genai.FileData{FileURI: "https://example.com/a.png"}}},
		{name: "file URI with a secret", part: &genai.Part{FileData: &genai.FileData{FileURI: "https://example.com/a.png?key=abc"}}, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, got := formatContentParts(singlePart(tt.part), maxLen, true); got != tt.want {
				t.Errorf("formatContentParts() truncated = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFormatContentParts_EmptyParts(t *testing.T) {
	content := &genai.Content{Parts: []*genai.Part{nil, {}, {Text: ""}, {InlineData: nil}}}

	got := FormatContentParts(content, 100)

	// Every part keeps its index, whether or not there is anything to log.
	want := []map[string]any{
		{"part_index": 0, "storage_mode": "INLINE"},
		{"mime_type": "text/plain", "part_attributes": "{}", "part_index": 1, "storage_mode": "INLINE"},
		{"mime_type": "text/plain", "part_attributes": "{}", "part_index": 2, "storage_mode": "INLINE"},
		{"mime_type": "text/plain", "part_attributes": "{}", "part_index": 3, "storage_mode": "INLINE"},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("FormatContentParts() mismatch (-want +got):\n%s", diff)
	}
}

func TestPartsCollector_InlineDataWithoutMIMEType(t *testing.T) {
	content := singlePart(&genai.Part{InlineData: &genai.Blob{Data: []byte{1}}})
	// Without offloading, a missing MIME type is written as it is.
	want := []map[string]any{{
		"part_index": 0, "storage_mode": "INLINE", "mime_type": "", "text": "[BINARY DATA]",
	}}
	// Offloading needs a content type for the object, so the part records the
	// generic binary type that the object is stored with.
	wantOffloaded := []map[string]any{{
		"part_index": 0, "storage_mode": "INLINE", "mime_type": "application/octet-stream", "text": "[BINARY DATA]",
	}}

	t.Run("exported function", func(t *testing.T) {
		if diff := cmp.Diff(want, FormatContentParts(content, 100)); diff != "" {
			t.Errorf("FormatContentParts() mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("offloading off", func(t *testing.T) {
		c := collectParts(partsOptions{maxLen: 100}, content)
		if diff := cmp.Diff(want, c.parts); diff != "" {
			t.Errorf("content_parts mismatch (-want +got):\n%s", diff)
		}
		if len(c.pending) != 0 {
			t.Errorf("queued uploads = %v, want none", c.pending)
		}
	})
	t.Run("offloading on", func(t *testing.T) {
		c := collectParts(partsOptions{maxLen: 100, offload: true, inlineLimit: 10}, content)
		if diff := cmp.Diff(wantOffloaded, c.parts); diff != "" {
			t.Errorf("content_parts mismatch (-want +got):\n%s", diff)
		}
		// The object is stored with the same type that the row names.
		wantPending := []pendingUpload{{slot: 0, mimeType: "application/octet-stream", data: []byte{1}}}
		if diff := cmp.Diff(wantPending, c.pending, cmp.AllowUnexported(pendingUpload{})); diff != "" {
			t.Errorf("queued uploads mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("offloading on, upload budget exhausted", func(t *testing.T) {
		c := collectParts(partsOptions{
			maxLen:       100,
			offload:      true,
			inlineLimit:  10,
			reserveBytes: func(int) bool { return false },
		}, content)
		wantKept := []map[string]any{{
			"part_index": 0, "storage_mode": "INLINE", "mime_type": "application/octet-stream", "text": uploadFailedText,
		}}
		if diff := cmp.Diff(wantKept, c.parts); diff != "" {
			t.Errorf("content_parts mismatch (-want +got):\n%s", diff)
		}
		if len(c.pending) != 0 {
			t.Errorf("queued uploads = %v, want none", c.pending)
		}
	})
}

// TestPartsCollector_Offloading is the decision table of what is queued for
// upload and what stays in the row.
func TestPartsCollector_Offloading(t *testing.T) {
	const inlineLimit = 10
	on := partsOptions{maxLen: 100, offload: true, inlineLimit: inlineLimit}
	textEntry := func(text string) map[string]any {
		return map[string]any{"part_index": 0, "storage_mode": "INLINE", "mime_type": "text/plain", "text": text}
	}
	imageEntry := map[string]any{"part_index": 0, "storage_mode": "INLINE", "mime_type": "image/png", "text": "[BINARY DATA]"}
	image := func(data ...byte) *genai.Part {
		return &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: data}}
	}
	text := func(n int) string { return strings.Repeat("a", n) }
	tests := []struct {
		name string
		opts partsOptions
		part *genai.Part
		// wantEntry, if set, is the only content_parts entry.
		wantEntry   map[string]any
		wantPending []pendingUpload
	}{
		{
			name:      "offloading off keeps long text inline",
			opts:      partsOptions{maxLen: 100, inlineLimit: inlineLimit},
			part:      &genai.Part{Text: text(40)},
			wantEntry: textEntry(text(40)),
		},
		{
			name:      "offloading off keeps images inline",
			opts:      partsOptions{maxLen: 100, inlineLimit: inlineLimit},
			part:      image(1, 2, 3),
			wantEntry: imageEntry,
		},
		{
			name:      "text at the limit stays inline",
			opts:      on,
			part:      &genai.Part{Text: text(inlineLimit)},
			wantEntry: textEntry(text(inlineLimit)),
		},
		{
			name:        "text over the limit is queued",
			opts:        on,
			part:        &genai.Part{Text: text(inlineLimit + 1)},
			wantEntry:   textEntry(text(inlineLimit + 1)),
			wantPending: []pendingUpload{{slot: 0, mimeType: "text/plain", isText: true, text: text(inlineLimit + 1)}},
		},
		{
			// The limit counts bytes: six runes of two bytes are twelve.
			name:        "text over the limit in bytes is queued",
			opts:        on,
			part:        &genai.Part{Text: strings.Repeat("é", 6)},
			wantPending: []pendingUpload{{slot: 0, mimeType: "text/plain", isText: true, text: strings.Repeat("é", 6)}},
		},
		{
			name:      "text at the limit in bytes stays inline",
			opts:      on,
			part:      &genai.Part{Text: strings.Repeat("é", 5)},
			wantEntry: textEntry(strings.Repeat("é", 5)),
		},
		{
			// The row cannot keep all of this inline, so the object must.
			name:        "text cut by MaxContentLen is queued in full",
			opts:        partsOptions{maxLen: 30, offload: true, inlineLimit: 100},
			part:        &genai.Part{Text: text(40)},
			wantEntry:   textEntry(text(30) + "...[TRUNCATED]"),
			wantPending: []pendingUpload{{slot: 0, mimeType: "text/plain", isText: true, text: text(40), fallbackTruncated: true}},
		},
		{
			name:      "text at MaxContentLen stays inline",
			opts:      partsOptions{maxLen: 30, offload: true, inlineLimit: 100},
			part:      &genai.Part{Text: text(30)},
			wantEntry: textEntry(text(30)),
		},
		{
			name:      "without a length limit only the inline limit counts",
			opts:      partsOptions{maxLen: -1, offload: true, inlineLimit: 100},
			part:      &genai.Part{Text: text(40)},
			wantEntry: textEntry(text(40)),
		},
		{
			name:        "image is queued",
			opts:        on,
			part:        image(1, 2, 3),
			wantEntry:   imageEntry,
			wantPending: []pendingUpload{{slot: 0, mimeType: "image/png", data: []byte{1, 2, 3}}},
		},
		{
			name:        "image without data is queued too",
			opts:        on,
			part:        &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png"}},
			wantEntry:   imageEntry,
			wantPending: []pendingUpload{{slot: 0, mimeType: "image/png"}},
		},
		{
			name: "function call is not queued",
			opts: on,
			part: &genai.Part{FunctionCall: &genai.FunctionCall{Name: strings.Repeat("f", 40)}},
		},
		{
			name: "function response is not queued",
			opts: on,
			part: &genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "f", Response: map[string]any{"k": text(40)}}},
		},
		{
			name: "code is not queued",
			opts: on,
			part: &genai.Part{ExecutableCode: &genai.ExecutableCode{Code: text(40)}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := collectParts(tt.opts, singlePart(tt.part))

			if tt.wantEntry != nil {
				if diff := cmp.Diff([]map[string]any{tt.wantEntry}, c.parts); diff != "" {
					t.Errorf("content_parts mismatch (-want +got):\n%s", diff)
				}
			}
			if diff := cmp.Diff(tt.wantPending, c.pending, cmp.AllowUnexported(pendingUpload{})); diff != "" {
				t.Errorf("queued uploads mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestPartsCollector_FileDataObjectRefs(t *testing.T) {
	tests := []struct {
		name         string
		offload      bool
		connectionID string
		uri          string
		wantURI      string
		// wantRef is the object_ref of the entry; nil means there is none.
		wantRef       map[string]any
		wantTruncated bool
	}{
		{
			name: "Cloud Storage file with a connection", offload: true, connectionID: "us.conn",
			uri: "gs://bucket/doc.pdf", wantURI: "gs://bucket/doc.pdf",
			wantRef: map[string]any{"uri": "gs://bucket/doc.pdf", "details": "{}", "authorizer": "us.conn"},
		},
		{
			name: "Cloud Storage file without a connection", offload: true,
			uri: "gs://bucket/doc.pdf", wantURI: "gs://bucket/doc.pdf",
			wantRef: map[string]any{"uri": "gs://bucket/doc.pdf", "details": "{}"},
		},
		{
			name: "other file", offload: true, connectionID: "us.conn",
			uri: "https://example.com/doc.pdf", wantURI: "https://example.com/doc.pdf",
		},
		{
			name: "offloading off", connectionID: "us.conn",
			uri: "gs://bucket/doc.pdf", wantURI: "gs://bucket/doc.pdf",
		},
		{
			name: "Cloud Storage file with a secret in the URI", offload: true, connectionID: "us.conn",
			uri: "gs://bucket/doc.pdf?key=abc", wantURI: "gs://bucket/doc.pdf?key=%5BREDACTED%5D", wantTruncated: true,
		},
		{
			name: "Cloud Storage file with user information", offload: true, connectionID: "us.conn",
			uri: "gs://user:pw@bucket/doc.pdf", wantURI: redactedURIPlaceholder, wantTruncated: true,
		},
		{name: "no URI", offload: true, connectionID: "us.conn"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			part := &genai.Part{FileData: &genai.FileData{FileURI: tt.uri, MIMEType: "application/pdf"}}

			c := collectParts(partsOptions{maxLen: 100, redact: true, offload: tt.offload, inlineLimit: 10, connectionID: tt.connectionID}, singlePart(part))

			want := map[string]any{"part_index": 0, "storage_mode": "EXTERNAL_URI", "mime_type": "application/pdf", "uri": tt.wantURI}
			if tt.wantRef != nil {
				want["object_ref"] = tt.wantRef
			}
			if diff := cmp.Diff([]map[string]any{want}, c.parts); diff != "" {
				t.Errorf("content_parts mismatch (-want +got):\n%s", diff)
			}
			if c.truncated != tt.wantTruncated {
				t.Errorf("truncated = %v, want %v", c.truncated, tt.wantTruncated)
			}
			if len(c.pending) != 0 {
				t.Errorf("queued uploads = %v, want none: the file is stored already", c.pending)
			}
		})
	}
}

// TestPartsCollector_SlotsAndOrdinals checks that a queued upload names its
// entry and its message correctly when there are several messages, and
// entries that are not uploaded.
func TestPartsCollector_SlotsAndOrdinals(t *testing.T) {
	png := func(b byte) *genai.Part {
		return &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{b}}}
	}
	long := strings.Repeat("a", 40)
	opts := partsOptions{maxLen: 100, offload: true, inlineLimit: 10}

	c := collectParts(opts,
		&genai.Content{Parts: []*genai.Part{{Text: "short"}, png(1)}},
		// A nil message has no parts, but keeps its number.
		nil,
		&genai.Content{Parts: []*genai.Part{nil, {FunctionCall: &genai.FunctionCall{Name: "f"}}, png(2), {Text: long}}},
	)

	// Slots count the entries of all messages; part indexes restart in each.
	want := []pendingUpload{
		{slot: 1, contentOrd: 0, partIdx: 1, mimeType: "image/png", data: []byte{1}},
		{slot: 4, contentOrd: 2, partIdx: 2, mimeType: "image/png", data: []byte{2}},
		{slot: 5, contentOrd: 2, partIdx: 3, mimeType: "text/plain", isText: true, text: long},
	}
	if diff := cmp.Diff(want, c.pending, cmp.AllowUnexported(pendingUpload{})); diff != "" {
		t.Errorf("queued uploads mismatch (-want +got):\n%s", diff)
	}
	if got := len(c.parts); got != 6 {
		t.Fatalf("got %d content_parts entries, want 6: %v", got, c.parts)
	}
	for _, u := range c.pending {
		if got, want := c.parts[u.slot]["part_index"], u.partIdx; got != want {
			t.Errorf("the entry in slot %d has part_index %v, want %v", u.slot, got, want)
		}
	}
}

func TestPartsCollector_QueuedDataIsACopy(t *testing.T) {
	blob := []byte{1, 2, 3}
	part := &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: blob}}
	c := collectParts(partsOptions{maxLen: 100, offload: true, inlineLimit: 10}, singlePart(part))

	// The caller changes the bytes, and then the message.
	blob[0] = 0xEE
	part.InlineData.Data = nil
	part.InlineData.MIMEType = "image/gif"

	want := []pendingUpload{{slot: 0, mimeType: "image/png", data: []byte{1, 2, 3}}}
	if diff := cmp.Diff(want, c.pending, cmp.AllowUnexported(pendingUpload{})); diff != "" {
		t.Errorf("queued uploads mismatch (-want +got):\n%s", diff)
	}
}

func TestPartsCollector_TextAndStructuredRedaction(t *testing.T) {
	secretJSON := `{"api_key":"sk-secret-key-999","msg":"` + strings.Repeat("x", 40) + `"}`
	codeJSON := `{"password":"pw-secret-123"}`

	t.Run("redact enabled", func(t *testing.T) {
		c := collectParts(
			partsOptions{maxLen: 500, redact: true, offload: true, inlineLimit: 20},
			&genai.Content{Parts: []*genai.Part{
				{Text: secretJSON},
				{ExecutableCode: &genai.ExecutableCode{Language: genai.LanguagePython, Code: codeJSON}},
				{CodeExecutionResult: &genai.CodeExecutionResult{Outcome: genai.OutcomeOK, Output: codeJSON}},
			}},
		)
		if len(c.pending) != 1 {
			t.Fatalf("pending len = %d, want 1", len(c.pending))
		}
		if strings.Contains(c.pending[0].text, "sk-secret-key-999") || !strings.Contains(c.pending[0].text, redactedValue) {
			t.Errorf("queued upload text was not redacted: %q", c.pending[0].text)
		}
		for i, part := range c.parts {
			text, _ := part["text"].(string)
			if strings.Contains(text, "sk-secret-key-999") || strings.Contains(text, "pw-secret-123") {
				t.Errorf("part %d text contains unredacted secret: %q", i, text)
			}
			if !strings.Contains(text, redactedValue) {
				t.Errorf("part %d text lacks %q: %q", i, redactedValue, text)
			}
		}
	})

	t.Run("redact disabled", func(t *testing.T) {
		c := collectParts(
			partsOptions{maxLen: 500, redact: false, offload: true, inlineLimit: 20},
			singlePart(&genai.Part{Text: secretJSON}),
		)
		if len(c.pending) != 1 || !strings.Contains(c.pending[0].text, "sk-secret-key-999") {
			t.Errorf("pending text with redact=false = %v, want original secret preserved", c.pending)
		}
	})
}

func TestNormalizeFunctionResponse_PythonSnakeCaseParity(t *testing.T) {
	willCont := true
	fr := &genai.FunctionResponse{
		ID:           "call-1",
		Name:         "lookup_doc",
		WillContinue: &willCont,
		Scheduling:   genai.FunctionResponseSchedulingWhenIdle,
		Response: map[string]any{
			"CallerCamelKey": "untouched",
			"api_key":        "sk-hidden",
		},
		Parts: []*genai.FunctionResponsePart{
			{
				InlineData: &genai.FunctionResponseBlob{
					MIMEType:    "text/plain",
					Data:        []byte("hello"),
					DisplayName: "greeting.txt",
				},
			},
			{
				InlineData: &genai.FunctionResponseBlob{
					MIMEType:    "image/png",
					Data:        []byte{0x89, 'P', 'N', 'G', 0xff},
					DisplayName: "pixel.png",
				},
			},
			{
				FileData: &genai.FunctionResponseFileData{
					FileURI:     "gs://bucket/file.pdf",
					MIMEType:    "application/pdf",
					DisplayName: "report.pdf",
				},
			},
		},
	}

	c := collectParts(partsOptions{maxLen: 1000, redact: true}, singlePart(&genai.Part{FunctionResponse: fr}))
	if len(c.parts) != 1 {
		t.Fatalf("got %d parts, want 1", len(c.parts))
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(c.parts[0]["part_attributes"].(string)), &got); err != nil {
		t.Fatalf("part_attributes is not JSON: %v", err)
	}
	want := map[string]any{
		"function_response": map[string]any{
			"id":            "call-1",
			"name":          "lookup_doc",
			"will_continue": true,
			"scheduling":    string(genai.FunctionResponseSchedulingWhenIdle),
			"response": map[string]any{
				"CallerCamelKey": "untouched",
				"api_key":        redactedValue,
			},
			"parts": []any{
				map[string]any{
					"inline_data": map[string]any{
						"mime_type":    "text/plain",
						"data":         "hello",
						"display_name": "greeting.txt",
					},
				},
				map[string]any{
					"inline_data": map[string]any{
						"mime_type":    "image/png",
						"data":         binaryDataPlaceholder,
						"display_name": "pixel.png",
					},
				},
				map[string]any{
					"file_data": map[string]any{
						"file_uri":     "gs://bucket/file.pdf",
						"mime_type":    "application/pdf",
						"display_name": "report.pdf",
					},
				},
			},
		},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("part_attributes mismatch (-want +got):\n%s", diff)
	}
}

// The bytes of a function response's inline media are sanitized like any other
// payload: UTF-8 data is redacted and truncated as text, and binary data is
// replaced by a placeholder.
func TestNormalizeFunctionResponse_InlineMediaDataIsSanitized(t *testing.T) {
	const (
		credentialText = "db password: hunter2\napi_key=SECRET123\nAuthorization: Bearer abc.def.ghi\n{\"token\": \"TOKENVALUE\"}"
		keyedJSON      = `{"password": "p4ssw0rd", "user": "alice"}`
	)

	// inlineMedia collects a function response that carries one inline media
	// part and returns the collector, the part_attributes written for it, and
	// the data it holds there.
	inlineMedia := func(t *testing.T, opts partsOptions, mimeType string, data []byte) (*partsCollector, string, any) {
		t.Helper()
		c := collectParts(opts, singlePart(&genai.Part{FunctionResponse: &genai.FunctionResponse{
			Name: "fetch",
			Parts: []*genai.FunctionResponsePart{{
				InlineData: &genai.FunctionResponseBlob{MIMEType: mimeType, Data: data},
			}},
		}}))
		if len(c.parts) != 1 {
			t.Fatalf("got %d parts, want 1", len(c.parts))
		}
		attrs, _ := c.parts[0]["part_attributes"].(string)
		parts, _ := lookup(partAttributes(t, c.parts[0]), "function_response", "parts")
		list, _ := parts.([]any)
		if len(list) != 1 {
			t.Fatalf("function_response.parts = %v, want 1 part; part_attributes: %s", parts, attrs)
		}
		got, ok := lookup(list[0], "inline_data", "data")
		if !ok {
			t.Fatalf("part has no inline_data.data: %v", list[0])
		}
		return c, attrs, got
	}

	t.Run("credentials in UTF-8 data are redacted", func(t *testing.T) {
		_, attrs, data := inlineMedia(t, partsOptions{maxLen: 1000, redact: true}, "text/plain", []byte(credentialText))
		for _, secret := range []string{"hunter2", "SECRET123", "abc.def.ghi", "TOKENVALUE"} {
			if strings.Contains(attrs, secret) {
				t.Errorf("part_attributes leaks %q: %s", secret, attrs)
			}
		}
		if s, _ := data.(string); !strings.Contains(s, redactedValue) {
			t.Errorf("data = %q, want it to contain %q", data, redactedValue)
		}
	})

	t.Run("sensitive keys in JSON data are redacted", func(t *testing.T) {
		_, attrs, data := inlineMedia(t, partsOptions{maxLen: 1000, redact: true}, "application/json", []byte(keyedJSON))
		if strings.Contains(attrs, "p4ssw0rd") {
			t.Errorf("part_attributes leaks the password: %s", attrs)
		}
		if s, _ := data.(string); !strings.Contains(s, redactedValue) || !strings.Contains(s, "alice") {
			t.Errorf("data = %q, want the password redacted and the user kept", data)
		}
	})

	t.Run("binary data is replaced by a placeholder", func(t *testing.T) {
		c, _, data := inlineMedia(t, partsOptions{maxLen: 1000, redact: true}, "image/png", []byte{0x89, 'P', 'N', 'G', 0xff})
		if data != binaryDataPlaceholder {
			t.Errorf("data = %v, want %q", data, binaryDataPlaceholder)
		}
		if !c.truncated {
			t.Error("collector is not marked truncated after binary data was dropped")
		}
	})

	t.Run("long data is truncated", func(t *testing.T) {
		c, _, data := inlineMedia(t, partsOptions{maxLen: 10, redact: true}, "text/plain", []byte(strings.Repeat("a", 50)))
		if want := strings.Repeat("a", 10) + truncationSuffix; data != want {
			t.Errorf("data = %v, want %q", data, want)
		}
		if !c.truncated {
			t.Error("collector is not marked truncated after data was cut")
		}
	})

	t.Run("data is kept as text when redaction is off", func(t *testing.T) {
		_, _, data := inlineMedia(t, partsOptions{maxLen: 1000, redact: false}, "text/plain", []byte(credentialText))
		if data != credentialText {
			t.Errorf("data = %v, want %q", data, credentialText)
		}
	})
}

func TestPartsCollector_QueuedTextDoesNotMarkTruncated(t *testing.T) {
	longText := strings.Repeat("a", 50)
	c := collectParts(partsOptions{maxLen: 20, offload: true, inlineLimit: 100}, singlePart(&genai.Part{Text: longText}))
	if len(c.pending) != 1 {
		t.Fatalf("pending len = %d, want 1", len(c.pending))
	}
	if c.truncated || c.nonUploadTruncated {
		t.Errorf("truncated=%v nonUploadTruncated=%v, want both false when text is queued for offload", c.truncated, c.nonUploadTruncated)
	}

	// When pending upload budget rejects the reservation, truncation is recorded immediately.
	cRejected := collectParts(partsOptions{
		maxLen:       20,
		offload:      true,
		inlineLimit:  100,
		reserveBytes: func(int) bool { return false },
	}, singlePart(&genai.Part{Text: longText}))
	if len(cRejected.pending) != 0 {
		t.Fatalf("pending len = %d, want 0 when reservation fails", len(cRejected.pending))
	}
	if !cRejected.truncated || !cRejected.nonUploadTruncated {
		t.Errorf("truncated=%v nonUploadTruncated=%v, want both true when reservation fails", cRejected.truncated, cRejected.nonUploadTruncated)
	}
}

func TestContentSummary_PartWithTextAndFileDataPrecedence(t *testing.T) {
	c := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{
				Text:     "should be ignored in summary",
				FileData: &genai.FileData{FileURI: "gs://bucket/file.png", MIMEType: "image/png"},
			},
		},
	}
	m := newMessageSummary(c, 0, 100, false)
	if len(m.pieces) != 0 {
		t.Fatalf("m.pieces len = %d, want 0 (FileData takes precedence over Text in contentSummary)", len(m.pieces))
	}
}

func TestContentSummary_ContentDeniedFailedUploadTruncatedPinsIsTruncated(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: strings.Repeat("x", 50)},
				},
			},
		},
	}
	parts := &partsCollector{
		opts: partsOptions{
			maxLen:       10,
			offload:      true,
			inlineLimit:  5,
			reserveBytes: func(int) bool { return true },
		},
	}
	_, _, builder := buildLLMRequestContent(req, 10, false, true, parts)
	if builder == nil {
		t.Fatal("builder is nil")
	}
	if len(parts.pending) != 1 {
		t.Fatalf("pending len = %d, want 1", len(parts.pending))
	}
	u := parts.pending[0]
	if !u.fallbackTruncated {
		t.Fatalf("fallbackTruncated = false, want true")
	}

	row := map[string]any{"event_type": EventLLMRequest}
	builder.retainInlineSummary(u.contentOrd, u.partIdx, "xxxxxxxxxx[TRUNCATED]", u.fallbackTruncated)
	builder.applyToRow(row)

	if _, ok := row[columnContent]; ok {
		t.Errorf("row[%q] is set, want omitted when contentDenied is true", columnContent)
	}
	if got, _ := row["is_truncated"].(bool); !got {
		t.Errorf("row[is_truncated] = %v, want true (pinned by hasFailedUploadTruncated)", row["is_truncated"])
	}
}
