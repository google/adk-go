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

package helper

import (
	"encoding/base64"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestNames(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		{
			name: "a",
			want: "a",
		},
		{
			name: "A",
			want: "a",
		},
		{
			name: "aa",
			want: "aa",
		},
		{
			name: "Aa",
			want: "aa",
		},
		{
			name: "AaAa",
			want: "aa_aa",
		},
		{
			name: "ArtifactDelta",
			want: "artifact_delta",
		},
		{
			name: "RequestedToolConfirmations",
			want: "requested_tool_confirmations",
		},
		{
			name: "ID",
			want: "id",
		},
		{
			name: "InvocationID",
			want: "invocation_id",
		},
		{
			name: "LongRunningToolIDs",
			want: "long_running_tool_i_ds", // special case, handled by exception - see pathName
		},
	}
	for _, tt := range tests {
		snake := convertName("", tt.name)
		if snake != tt.want {
			t.Errorf("convertName(%q) = %q, want %q", tt.name, snake, tt.want)
		}
	}
}

func TestEvent(t *testing.T) {
	event := session.Event{
		ID: "1",
		LongRunningToolIDs: []string{
			"1",
			"2",
		},
	}
	o, err := convertSnake("", "", event)
	if err != nil {
		t.Errorf("convertSnake() failed: %v", err)
	}
	if m, ok := o.(map[string]any); ok {
		if arr, ok := m["long_running_tool_ids"]; ok {
			t.Logf("long_running_tool_ids: %v %T", arr, arr)
			if arr, ok := arr.([]any); ok {
				if len(arr) == 2 {
					if arr[0] != "1" {
						t.Errorf("long_running_tool_ids[0] is not 1")
					}
					if arr[1] != "2" {
						t.Errorf("long_running_tool_ids[1] is not 2")
					}

				} else {
					t.Errorf("long_running_tool_ids is not an array of length 2")
				}
			} else {
				t.Errorf("long_running_tool_ids is not an array")
			}
		} else {
			t.Errorf("long_running_tool_ids not found")
		}
	} else {
		t.Errorf("o is not a map")
	}
}

func TestEventLogProbs(t *testing.T) {
	event := session.Event{
		ID: "1",
		LongRunningToolIDs: []string{
			"1",
			"2",
		},
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Parts: []*genai.Part{
					{
						Text: "Hello",
					},
				},
			},
		},
	}
	o, err := convertSnake("", "", event)
	if err != nil {
		t.Errorf("convertSnake() failed: %v", err)
	}
	if m, ok := o.(map[string]any); ok {
		if arr, ok := m["long_running_tool_ids"]; ok {
			t.Logf("long_running_tool_ids: %v %T", arr, arr)
			if arr, ok := arr.([]any); ok {
				if len(arr) == 2 {
					if arr[0] != "1" {
						t.Errorf("long_running_tool_ids[0] is not 1")
					}
					if arr[1] != "2" {
						t.Errorf("long_running_tool_ids[1] is not 2")
					}

				} else {
					t.Errorf("long_running_tool_ids is not an array of length 2")
				}
			} else {
				t.Errorf("long_running_tool_ids is not an array")
			}
		} else {
			t.Errorf("long_running_tool_ids not found")
		}
	} else {
		t.Errorf("o is not a map")
	}
}

func TestEmbedded(t *testing.T) {
	type A struct {
		anInteger int
		aString   string
	}
	type B struct {
		A
		anotherString string
	}
	b := B{
		A: A{
			anInteger: 1,
			aString:   "a",
		},
		anotherString: "b",
	}
	got, err := convertSnake("", "", b)
	if err != nil {
		t.Errorf("convertSnake() failed: %v", err)
	}

	want := map[string]any{
		"an_integer":     int64(1),
		"a_string":       "a",
		"another_string": "b",
	}

	diff := cmp.Diff(got, want)
	if diff != "" {
		t.Errorf("convertSnake() = %v, want %v, diff: \n%v", got, want, diff)
	}
}

func TestOmitEmpty(t *testing.T) {
	type A struct {
		OmittableArr         []int       `json:"optional_array,omitempty"`
		MandatoryArr         []int       `json:"must_array"`
		OmmitableIntToIntMap map[int]int `json:"optional_map,omitempty"`
		MandatoryIntToIntMap map[int]int `json:"must_map"`
	}
	tests := []struct {
		name string
		a    A
		want map[string]any
	}{
		{
			name: "nil, nil, nil, nil",
			a:    A{},
			want: map[string]any{
				"must_array": []any{},
				"must_map":   map[string]any{},
			},
		},
		{
			name: "empty, nil, nil, nil",
			a: A{
				OmittableArr: []int{},
			},
			want: map[string]any{
				"must_array": []any{},
				"must_map":   map[string]any{},
			},
		},
		{
			name: "nil, [1], nil, nil",
			a: A{
				OmittableArr: []int{1},
			}, want: map[string]any{
				"must_array": []any{},
				"must_map":   map[string]any{},
				"optional_array": []any{
					int64(1),
				},
			},
		},
	}
	for _, tc := range tests {
		got, err := convertSnake("", "", tc.a)
		if err != nil {
			t.Errorf("convertSnake() failed: %v", err)
		}
		diff := cmp.Diff(got, tc.want)
		if diff != "" {
			t.Errorf("convertSnake() = %v, want %v, diff: \n%v", got, tc.want, diff)
		}

	}
}

func TestByteSlice(t *testing.T) {
	type withBytes struct {
		Signature []byte `json:"signature,omitempty"`
		Name      string `json:"name,omitempty"`
	}

	tests := []struct {
		name  string
		input withBytes
		want  map[string]any
	}{
		{
			name:  "non-empty byte slice is base64 encoded",
			input: withBytes{Signature: []byte{0x01, 0x02, 0x03}, Name: "a"},
			want:  map[string]any{"signature": "AQID", "name": "a"},
		},
		{
			name:  "empty byte slice is dropped by omitempty",
			input: withBytes{Signature: []byte{}, Name: "a"},
			want:  map[string]any{"name": "a"},
		},
		{
			name:  "nil byte slice is dropped by omitempty",
			input: withBytes{Name: "a"},
			want:  map[string]any{"name": "a"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := convertSnake("", "", tt.input)
			if err != nil {
				t.Fatalf("convertSnake() failed: %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("convertSnake() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestThoughtSignature(t *testing.T) {
	// A thought signature is what thinking models put on a part in practice, and
	// before byte slices were handled it failed conversion of the whole event.
	signature := []byte("thought-signature-bytes")
	event := session.Event{
		ID: "1",
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "hello", ThoughtSignature: signature},
				},
			},
		},
	}

	got, err := convertSnake("", "", event)
	if err != nil {
		t.Fatalf("convertSnake() failed: %v", err)
	}

	m, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("convertSnake() returned %T, want map[string]any", got)
	}
	content, ok := m["content"].(map[string]any)
	if !ok {
		t.Fatalf("content is %T, want map[string]any", m["content"])
	}
	parts, ok := content["parts"].([]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("parts is %#v, want a slice of one element", content["parts"])
	}
	part, ok := parts[0].(map[string]any)
	if !ok {
		t.Fatalf("parts[0] is %T, want map[string]any", parts[0])
	}
	want := base64.StdEncoding.EncodeToString(signature)
	if part["thought_signature"] != want {
		t.Errorf("thought_signature = %v, want %v", part["thought_signature"], want)
	}
}
