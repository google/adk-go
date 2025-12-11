// Copyright 2025 Google LLC
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

package adka2a

import (
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

func TestPartsTwoWayConversion(t *testing.T) {
	testCases := []struct {
		name                   string
		a2aPart                a2a.Part
		genaiPart              *genai.Part
		longRunningFunctionIDs []string
	}{
		{
			name:      "text",
			a2aPart:   a2a.TextPart{Text: "Hello"},
			genaiPart: &genai.Part{Text: "Hello"},
		},
		{
			name:      "thought",
			a2aPart:   a2a.TextPart{Text: "Hello", Metadata: map[string]any{ToA2AMetaKey("thought"): true}},
			genaiPart: &genai.Part{Text: "Hello", Thought: true},
		},
		{
			name: "file uri",
			a2aPart: a2a.FilePart{
				File: a2a.FileURI{URI: "ftp://cat.com", FileMeta: a2a.FileMeta{MimeType: "image/jpeg", Name: "cat.jpeg"}},
			},
			genaiPart: &genai.Part{
				FileData: &genai.FileData{FileURI: "ftp://cat.com", MIMEType: "image/jpeg", DisplayName: "cat.jpeg"},
			},
		},
		{
			name: "file bytes",
			a2aPart: a2a.FilePart{
				File: a2a.FileBytes{Bytes: "/w==", FileMeta: a2a.FileMeta{MimeType: "image/jpeg", Name: "cat.jpeg"}},
			},
			genaiPart: &genai.Part{
				InlineData: &genai.Blob{Data: []byte{0xfF}, MIMEType: "image/jpeg", DisplayName: "cat.jpeg"},
			},
		},
		{
			name: "function call",
			a2aPart: a2a.DataPart{
				Data: map[string]any{
					"id":   "get_weather",
					"args": map[string]any{"city": "Warsaw"},
					"name": "GetWeather",
				},
				Metadata: map[string]any{
					a2aDataPartMetaTypeKey:        a2aDataPartTypeFunctionCall,
					a2aDataPartMetaLongRunningKey: false,
				},
			},
			genaiPart: &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   "get_weather",
					Args: map[string]any{"city": "Warsaw"},
					Name: "GetWeather",
				},
			},
		},
		{
			name: "long running function call",
			a2aPart: a2a.DataPart{
				Data: map[string]any{
					"id":   "get_weather",
					"args": map[string]any{"city": "Warsaw"},
					"name": "GetWeather",
				},
				Metadata: map[string]any{
					a2aDataPartMetaTypeKey:        a2aDataPartTypeFunctionCall,
					a2aDataPartMetaLongRunningKey: true,
				},
			},
			genaiPart: &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   "get_weather",
					Args: map[string]any{"city": "Warsaw"},
					Name: "GetWeather",
				},
			},
			longRunningFunctionIDs: []string{"get_weather"},
		},
		{
			name: "function response",
			a2aPart: a2a.DataPart{
				Data: map[string]any{
					"id":         "get_weather",
					"scheduling": string(genai.FunctionResponseSchedulingInterrupt),
					"response":   map[string]any{"temperature": "7C"},
					"name":       "GetWeather",
				},
				Metadata: map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeFunctionResponse},
			},
			genaiPart: &genai.Part{
				FunctionResponse: &genai.FunctionResponse{
					ID:         "get_weather",
					Scheduling: genai.FunctionResponseSchedulingInterrupt,
					Response:   map[string]any{"temperature": "7C"},
					Name:       "GetWeather",
				},
			},
		},
		{
			name: "code execution result",
			a2aPart: a2a.DataPart{
				Data:     map[string]any{"outcome": string(genai.OutcomeOK), "output": "4"},
				Metadata: map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeCodeExecResult},
			},
			genaiPart: &genai.Part{
				CodeExecutionResult: &genai.CodeExecutionResult{
					Outcome: genai.OutcomeOK,
					Output:  "4",
				},
			},
		},
		{
			name: "code execution result",
			a2aPart: a2a.DataPart{
				Data:     map[string]any{"code": "print(2+2)", "language": string(genai.LanguagePython)},
				Metadata: map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeCodeExecutableCode},
			},
			genaiPart: &genai.Part{
				ExecutableCode: &genai.ExecutableCode{
					Code:     "print(2+2)",
					Language: genai.LanguagePython,
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			toA2A, err := ToA2AParts([]*genai.Part{tc.genaiPart}, tc.longRunningFunctionIDs)
			if err != nil {
				t.Errorf("toA2AParts() error = %v, want nil", err)
			}
			if diff := cmp.Diff([]a2a.Part{tc.a2aPart}, toA2A); diff != "" {
				t.Errorf("toA2AParts() wrong result (+got,-want)\ngot = %v\nwant = %v\ndiff = %s", toA2A, tc.a2aPart, diff)
			}

			toGenAI, err := ToGenAIParts([]a2a.Part{tc.a2aPart})
			if err != nil {
				t.Errorf("toGenAIParts() error = %v, want nil", err)
			}
			if diff := cmp.Diff([]*genai.Part{tc.genaiPart}, toGenAI); diff != "" {
				t.Errorf("toGenAIParts() wrong result (+got,-want)\ngot = %v\nwant = %v\ndiff = %s", toA2A, tc.a2aPart, diff)
			}
		})
	}
}

func TestPartsOneWayConversion(t *testing.T) {
	part := a2a.DataPart{Data: map[string]any{"arbitrary": "data"}}
	wantGenAI := &genai.Part{Text: `{"arbitrary":"data"}`}

	gotGenAI, err := ToGenAIParts([]a2a.Part{part})
	if err != nil {
		t.Fatalf("toGenAI() error = %v, want nil", err)
	}
	if diff := cmp.Diff([]*genai.Part{wantGenAI}, gotGenAI); diff != "" {
		t.Fatalf("toGenAI() wrong result (+got,-want)\ngot = %v\nwant = %v\ndiff = %s", gotGenAI, part, diff)
	}

	wantA2A := a2a.TextPart{Text: `{"arbitrary":"data"}`}
	gotA2A, err := ToA2AParts(gotGenAI, nil)
	if err != nil {
		t.Fatalf("toA2AParts() error = %v, want nil", err)
	}
	if diff := cmp.Diff([]a2a.Part{wantA2A}, gotA2A); diff != "" {
		t.Fatalf("toA2AParts() wrong result (+got,-want)\ngot = %v\nwant = %v\ndiff = %s", gotA2A, wantA2A, diff)
	}
}
