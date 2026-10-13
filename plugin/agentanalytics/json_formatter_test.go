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
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"google.golang.org/genai"
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
			want: `failed: {"temp:cache=[REDACTED]":"[REDACTED]"}`,
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
