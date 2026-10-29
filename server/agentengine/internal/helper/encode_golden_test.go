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
	"encoding/json"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// TestConvertSnakeMatchesLegacy verifies that ConvertSnake produces identical
// JSON bytes to the uncached reference implementation across Agent Engine event shapes.
func TestConvertSnakeMatchesLegacy(t *testing.T) {
	type Embedded struct {
		Inner string `json:"inner"`
		Count int    `json:"count,omitempty"`
	}
	type mixed struct {
		Embedded
		Name      string `json:"name"`
		Skip      string `json:"-"`
		Untagged  string `json:""`
		NoTag     bool
		Zero      float64         `json:"zero,omitempty"`
		Bytes     []byte          `json:"bytes"`
		Raw       json.RawMessage `json:"raw,omitempty"`
		EmptyRaw  json.RawMessage `json:"emptyRaw,omitempty"`
		When      time.Time       `json:"when"`
		Ptr       *Embedded       `json:"ptr"`
		NilPtr    *Embedded       `json:"nilPtr,omitempty"`
		Any       any             `json:"any"`
		NilAny    any             `json:"nilAny"`
		Nested    map[string]any  `json:"nested"`
		NilMap    map[string]int  `json:"nilMap"`
		NilSlice  []string        `json:"nilSlice"`
		Ints      []int           `json:"ints"`
		Role      genai.Role      `json:"role"`
		OmitZeroI int             `json:"omitZeroI,omitzero"`
		hidden    int
	}

	frEvent := benchTextEvent()
	frEvent.LLMResponse.Content = &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
		FunctionResponse: &genai.FunctionResponse{
			ID: "call-1", Name: "lookup_weather",
			Response: map[string]any{"result": map[string]any{"temp": 21.5, "unit": "celsius"}},
		},
	}}}
	richEvent := benchTextEvent()
	richEvent.Branch = "root.child"
	richEvent.IsolationScope = "scope-1"
	richEvent.Routes = []string{"a", "b"}
	richEvent.Output = map[string]any{"answer": 42.0}
	richEvent.NodeInfo = &session.NodeInfo{Path: "wf/node@1", MessageAsOutput: true}
	richEvent.RequestedInput = &session.RequestInput{InterruptID: "approve", Message: "ok?"}
	richEvent.LLMResponse.CustomMetadata = map[string]any{"k": []any{"x", 1.0}}
	richEvent.LLMResponse.Partial = true
	richEvent.LLMResponse.AvgLogprobs = -0.25
	richEvent.LLMResponse.Content.Parts = append(richEvent.LLMResponse.Content.Parts, &genai.Part{
		Thought: true, Text: "thinking", ThoughtSignature: []byte{1, 2, 3},
	}, &genai.Part{
		InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("png-bytes"), DisplayName: "pic"},
	})
	richEvent.Actions = session.EventActions{
		StateDelta:        map[string]any{"turn": 3.0, "nested": map[string]any{"a": "b"}},
		ArtifactDelta:     map[string]int64{"file.txt": 2},
		Escalate:          true,
		SkipSummarization: true,
		TransferToAgent:   "other",
		RequestedToolConfirmations: map[string]toolconfirmation.ToolConfirmation{
			"call-9": {Hint: "please confirm", Confirmed: false},
		},
		Compaction: &session.EventCompaction{
			StartTimestamp:   time.Unix(1, 0),
			EndTimestamp:     time.Unix(9, 0),
			CompactedContent: genai.NewContentFromText("summary", genai.RoleModel),
		},
	}
	errEvent := session.Event{ID: "e", LLMResponse: model.LLMResponse{ErrorCode: "500", ErrorMessage: "boom"}}

	for _, tc := range []struct {
		name  string
		value any
	}{
		{"nil", nil},
		{"string", "plain"},
		{"text event", benchTextEvent()},
		{"text event pointer", ptrTo(benchTextEvent())},
		{"function call event", benchFunctionCallEvent()},
		{"function response event", frEvent},
		{"rich event", richEvent},
		{"error event", errEvent},
		{"empty event", session.Event{}},
		{"envelope", benchEnvelope(benchFunctionCallEvent())},
		{"error map", map[string]any{"error": "boom"}},
		{"mixed struct", mixed{
			Embedded: Embedded{Inner: "in", Count: 0},
			Name:     "n", Skip: "s", Untagged: "u", NoTag: true,
			Bytes: []byte("abc"), Raw: json.RawMessage(`{"a":1}`),
			When:   time.Unix(1_700_000_000, 500_000_000),
			Ptr:    &Embedded{Inner: "p", Count: 2},
			Any:    Embedded{Inner: "dyn"},
			Nested: map[string]any{"list": []any{1.0, "two", nil}, "m": map[string]string{"x": "y"}},
			Ints:   []int{1, 2, 3},
			Role:   genai.RoleModel,
			hidden: 7,
		}},
		{"mixed struct zero", mixed{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want, err := json.Marshal(legacyConvertSnake(tc.value))
			if err != nil {
				t.Fatalf("legacy: %v", err)
			}
			got, err := json.Marshal(ConvertSnake(tc.value))
			if err != nil {
				t.Fatalf("live: %v", err)
			}
			if string(got) != string(want) {
				t.Errorf("ConvertSnake output differs from the reference\n got: %s\nwant: %s", got, want)
			}
		})
	}
}

func ptrTo[T any](v T) *T { return &v }
