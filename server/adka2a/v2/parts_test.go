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
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/converters"
)

func TestPartsTwoWayConversion(t *testing.T) {
	testCases := []struct {
		name                   string
		a2aPart                *a2a.Part
		genaiPart              *genai.Part
		longRunningFunctionIDs []string
	}{
		{
			name:      "text",
			a2aPart:   a2a.NewTextPart("Hello"),
			genaiPart: &genai.Part{Text: "Hello"},
		},
		{
			name: "thought",
			a2aPart: func() *a2a.Part {
				p := a2a.NewTextPart("Hello")
				p.SetMeta(ToA2AMetaKey("thought"), true)
				return p
			}(),
			genaiPart: &genai.Part{Text: "Hello", Thought: true},
		},
		{
			name: "file uri",
			a2aPart: func() *a2a.Part {
				p := a2a.NewFileURLPart("ftp://cat.com", "image/jpeg")
				p.Filename = "cat.jpeg"
				return p
			}(),
			genaiPart: &genai.Part{
				FileData: &genai.FileData{FileURI: "ftp://cat.com", MIMEType: "image/jpeg", DisplayName: "cat.jpeg"},
			},
		},
		{
			name: "file bytes",
			a2aPart: func() *a2a.Part {
				p := a2a.NewRawPart([]byte{0xfF})
				p.MediaType = "image/jpeg"
				p.Filename = "cat.jpeg"
				return p
			}(),
			genaiPart: &genai.Part{
				InlineData: &genai.Blob{Data: []byte{0xfF}, MIMEType: "image/jpeg", DisplayName: "cat.jpeg"},
			},
		},
		{
			name: "function call",
			a2aPart: func() *a2a.Part {
				p := a2a.NewDataPart(map[string]any{
					"id":   "get_weather",
					"args": map[string]any{"city": "Warsaw"},
					"name": "GetWeather",
				})
				p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeFunctionCall)
				p.SetMeta(a2aDataPartMetaLongRunningKey, false)
				return p
			}(),
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
			a2aPart: func() *a2a.Part {
				p := a2a.NewDataPart(map[string]any{
					"id":   "get_weather",
					"args": map[string]any{"city": "Warsaw"},
					"name": "GetWeather",
				})
				p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeFunctionCall)
				p.SetMeta(a2aDataPartMetaLongRunningKey, true)
				return p
			}(),
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
			a2aPart: func() *a2a.Part {
				p := a2a.NewDataPart(map[string]any{
					"id":         "get_weather",
					"scheduling": string(genai.FunctionResponseSchedulingInterrupt),
					"response":   map[string]any{"temperature": "7C"},
					"name":       "GetWeather",
				})
				p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeFunctionResponse)
				return p
			}(),
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
			a2aPart: func() *a2a.Part {
				p := a2a.NewDataPart(map[string]any{"outcome": string(genai.OutcomeOK), "output": "4"})
				p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeCodeExecResult)
				return p
			}(),
			genaiPart: &genai.Part{
				CodeExecutionResult: &genai.CodeExecutionResult{
					Outcome: genai.OutcomeOK,
					Output:  "4",
				},
			},
		},
		{
			name: "code execution result",
			a2aPart: func() *a2a.Part {
				p := a2a.NewDataPart(map[string]any{"code": "print(2+2)", "language": string(genai.LanguagePython)})
				p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeCodeExecutableCode)
				return p
			}(),
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
			if diff := cmp.Diff([]*a2a.Part{tc.a2aPart}, toA2A); diff != "" {
				t.Errorf("toA2AParts() wrong result (+got,-want)\ngot = %v\nwant = %v\ndiff = %s", toA2A, tc.a2aPart, diff)
			}

			toGenAI, err := ToGenAIParts([]*a2a.Part{tc.a2aPart})
			if err != nil {
				t.Errorf("toGenAIParts() error = %v, want nil", err)
			}
			if diff := cmp.Diff([]*genai.Part{tc.genaiPart}, toGenAI); diff != "" {
				t.Errorf("toGenAIParts() wrong result (+got,-want)\ngot = %v\nwant = %v\ndiff = %s", toA2A, tc.a2aPart, diff)
			}
		})
	}
}

func TestPartsDataPartConversionRoundTrip(t *testing.T) {
	a2aPart := a2a.NewDataPart(map[string]any{"arbitrary": "data"})
	wantGenAI := &genai.Part{InlineData: &genai.Blob{Data: []byte("<a2a_datapart_json>{\"arbitrary\":\"data\"}</a2a_datapart_json>"), MIMEType: "text/plain"}}

	gotGenAI, err := ToGenAIParts([]*a2a.Part{a2aPart})
	if err != nil {
		t.Fatalf("toGenAI() error = %v, want nil", err)
	}
	if diff := cmp.Diff([]*genai.Part{wantGenAI}, gotGenAI); diff != "" {
		t.Fatalf("toGenAI() wrong result (+got,-want)\ngot = %v\nwant = %v\ndiff = %s", gotGenAI, a2aPart, diff)
	}

	gotbackA2A, err := ToA2AParts(gotGenAI, nil)
	if err != nil {
		t.Fatalf("toA2AParts() error = %v, want nil", err)
	}
	if diff := cmp.Diff([]*a2a.Part{a2aPart}, gotbackA2A); diff != "" {
		t.Fatalf("toA2AParts() wrong result (+got,-want)\ngot = %v\nwant = %v\ndiff = %s", gotbackA2A, a2aPart, diff)
	}
}

func TestCodeExecutionPartStructFields(t *testing.T) {
	cerType := reflect.TypeFor[genai.CodeExecutionResult]()
	var cerFields []string
	for i := range cerType.NumField() {
		cerFields = append(cerFields, cerType.Field(i).Name)
	}
	if diff := cmp.Diff([]string{"Outcome", "Output", "ID"}, cerFields); diff != "" {
		t.Fatalf("genai.CodeExecutionResult fields changed (-want +got):\n%s", diff)
	}

	ecType := reflect.TypeFor[genai.ExecutableCode]()
	var ecFields []string
	for i := range ecType.NumField() {
		ecFields = append(ecFields, ecType.Field(i).Name)
	}
	if diff := cmp.Diff([]string{"Code", "Language", "ID"}, ecFields); diff != "" {
		t.Fatalf("genai.ExecutableCode fields changed (-want +got):\n%s", diff)
	}
}

func TestCodeExecutionPartEquivalence(t *testing.T) {
	genaiParts := []*genai.Part{
		{CodeExecutionResult: &genai.CodeExecutionResult{}},
		{CodeExecutionResult: &genai.CodeExecutionResult{Outcome: genai.OutcomeOK, Output: "42\n", ID: "id-1"}},
		{CodeExecutionResult: &genai.CodeExecutionResult{Outcome: genai.OutcomeFailed, Output: "błąd wykonania 🚀 " + strings.Repeat("x", 4096)}},
		{CodeExecutionResult: &genai.CodeExecutionResult{Outcome: genai.OutcomeOK, Output: "invalid-\xff-utf8", ID: "id-\xfe"}},
		{ExecutableCode: &genai.ExecutableCode{}},
		{ExecutableCode: &genai.ExecutableCode{Code: "print('cześć')", Language: genai.LanguagePython, ID: "id-2"}},
		{ExecutableCode: &genai.ExecutableCode{Code: "invalid-\xff-code", Language: genai.LanguagePython}},
	}

	for i, part := range genaiParts {
		gotMap, gotErr := toA2ACodeExecMap(part)
		wantMap, wantErr := referenceToA2ACodeExecMap(part)
		if (gotErr != nil) != (wantErr != nil) {
			t.Fatalf("case %d: error = %v, want %v", i, gotErr, wantErr)
		}
		if diff := cmp.Diff(wantMap, gotMap); diff != "" {
			t.Errorf("case %d: toA2ADataPart map mismatch (-want +got):\n%s", i, diff)
		}
	}

	a2aCases := []struct {
		name     string
		metaType string
		data     any
	}{
		{
			name:     "canonical code execution result",
			metaType: a2aDataPartTypeCodeExecResult,
			data:     map[string]any{"outcome": "OUTCOME_OK", "output": "wynik 42", "id": "c-1"},
		},
		{
			name:     "code execution result with nil value",
			metaType: a2aDataPartTypeCodeExecResult,
			data:     map[string]any{"outcome": "OUTCOME_OK", "output": nil, "id": nil},
		},
		{
			name:     "code execution result with case-variant and unknown keys",
			metaType: a2aDataPartTypeCodeExecResult,
			data:     map[string]any{"Outcome": "OUTCOME_OK", "OUTPUT": "42", "extra": 99},
		},
		{
			name:     "code execution result with invalid utf8",
			metaType: a2aDataPartTypeCodeExecResult,
			data:     map[string]any{"outcome": "OUTCOME_OK", "output": "bad-\xff-utf8"},
		},
		{
			name:     "code execution result with wrong value type",
			metaType: a2aDataPartTypeCodeExecResult,
			data:     map[string]any{"output": 123},
		},
		{
			name:     "code execution result with unmarshalable value",
			metaType: a2aDataPartTypeCodeExecResult,
			data:     map[string]any{"output": make(chan int)},
		},
		{
			name:     "canonical executable code",
			metaType: a2aDataPartTypeCodeExecutableCode,
			data:     map[string]any{"code": "print(1)", "language": "PYTHON", "id": "e-1"},
		},
		{
			name:     "executable code with case-variant and nil keys",
			metaType: a2aDataPartTypeCodeExecutableCode,
			data:     map[string]any{"Code": "print(1)", "language": nil, "unknown": true},
		},
		{
			name:     "executable code with invalid utf8",
			metaType: a2aDataPartTypeCodeExecutableCode,
			data:     map[string]any{"code": "bad-\xff", "language": "PYTHON"},
		},
		{
			name:     "executable code with wrong value type",
			metaType: a2aDataPartTypeCodeExecutableCode,
			data:     map[string]any{"code": []int{1, 2}},
		},
	}

	for _, tc := range a2aCases {
		t.Run(tc.name, func(t *testing.T) {
			p := a2a.NewDataPart(tc.data)
			p.SetMeta(a2aDataPartMetaTypeKey, tc.metaType)

			got, gotErr := toGenAIDataPart(p)
			want, wantErr := referenceToGenAIDataPart(p)
			if (gotErr != nil) != (wantErr != nil) {
				t.Fatalf("error = %v, want %v", gotErr, wantErr)
			}
			if gotErr != nil {
				if gotErr.Error() != wantErr.Error() {
					t.Errorf("error message = %q, want %q", gotErr.Error(), wantErr.Error())
				}
				return
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("toGenAIDataPart mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func toA2ACodeExecMap(part *genai.Part) (any, error) {
	a2aPart, err := toA2ADataPart(part, nil)
	if err != nil {
		return nil, err
	}
	return a2aPart.Data(), nil
}

func referenceToA2ACodeExecMap(part *genai.Part) (any, error) {
	if part.CodeExecutionResult != nil {
		return converters.ToMapStructure(part.CodeExecutionResult)
	}
	return converters.ToMapStructure(part.ExecutableCode)
}

func referenceToGenAIDataPart(part *a2a.Part) (*genai.Part, error) {
	raw, err := json.Marshal(part.Data())
	if err != nil {
		return nil, err
	}
	switch part.Metadata[a2aDataPartMetaTypeKey] {
	case a2aDataPartTypeCodeExecResult:
		var val genai.CodeExecutionResult
		if err := json.Unmarshal(raw, &val); err != nil {
			return nil, err
		}
		return &genai.Part{CodeExecutionResult: &val}, nil
	case a2aDataPartTypeCodeExecutableCode:
		var val genai.ExecutableCode
		if err := json.Unmarshal(raw, &val); err != nil {
			return nil, err
		}
		return &genai.Part{ExecutableCode: &val}, nil
	default:
		return nil, nil
	}
}
