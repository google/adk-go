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
	"strings"
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
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
			name: "file bytes with video metadata",
			a2aPart: func() *a2a.Part {
				p := a2a.NewRawPart([]byte{0xfF})
				p.MediaType = "video/mp4"
				p.SetMeta("adk_video_metadata", map[string]any{"startOffset": "10s", "endOffset": "25s", "fps": 2.0})
				return p
			}(),
			genaiPart: &genai.Part{
				InlineData:    &genai.Blob{Data: []byte{0xfF}, MIMEType: "video/mp4"},
				VideoMetadata: &genai.VideoMetadata{StartOffset: 10 * time.Second, EndOffset: 25 * time.Second, FPS: new(2.0)},
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

func TestToGenAIPartsEmptyText(t *testing.T) {
	thought := a2a.NewTextPart("")
	thought.SetMeta(ToA2AMetaKey("thought"), true)

	testCases := []struct {
		name  string
		parts []*a2a.Part
		want  []*genai.Part
	}{
		{
			name:  "empty text",
			parts: []*a2a.Part{a2a.NewTextPart("")},
			want:  []*genai.Part{{Text: ""}},
		},
		{
			name:  "empty text after text",
			parts: []*a2a.Part{a2a.NewTextPart("Hello"), a2a.NewTextPart("")},
			want:  []*genai.Part{{Text: "Hello"}, {Text: ""}},
		},
		{
			name:  "empty thought",
			parts: []*a2a.Part{thought},
			want:  []*genai.Part{{Text: "", Thought: true}},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ToGenAIParts(tc.parts)
			if err != nil {
				t.Fatalf("ToGenAIParts() error = %v, want nil", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ToGenAIParts() wrong result (-want +got):\n%s", diff)
			}
		})
	}
}

func TestToGenAIPartsReadsVideoMetadata(t *testing.T) {
	videoPart := func(meta any) *a2a.Part {
		p := a2a.NewRawPart([]byte{0xfF})
		p.MediaType = "video/mp4"
		p.SetMeta("adk_video_metadata", meta)
		return p
	}

	testCases := []struct {
		name string
		part *a2a.Part
		want *genai.Part
	}{
		{
			name: "fps only",
			part: videoPart(map[string]any{"fps": 30.0}),
			want: &genai.Part{
				InlineData:    &genai.Blob{Data: []byte{0xfF}, MIMEType: "video/mp4"},
				VideoMetadata: &genai.VideoMetadata{FPS: new(30.0)},
			},
		},
		{
			name: "fractional offsets",
			part: videoPart(map[string]any{"startOffset": "1.5s", "endOffset": "3.25s"}),
			want: &genai.Part{
				InlineData:    &genai.Blob{Data: []byte{0xfF}, MIMEType: "video/mp4"},
				VideoMetadata: &genai.VideoMetadata{StartOffset: 1500 * time.Millisecond, EndOffset: 3250 * time.Millisecond},
			},
		},
		{
			name: "null is absent",
			part: videoPart(nil),
			want: &genai.Part{InlineData: &genai.Blob{Data: []byte{0xfF}, MIMEType: "video/mp4"}},
		},
		{
			name: "unprefixed key is ignored",
			part: func() *a2a.Part {
				p := a2a.NewRawPart([]byte{0xfF})
				p.MediaType = "video/mp4"
				p.SetMeta("video_metadata", map[string]any{"fps": 30.0})
				return p
			}(),
			want: &genai.Part{InlineData: &genai.Blob{Data: []byte{0xfF}, MIMEType: "video/mp4"}},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ToGenAIPart(tc.part)
			if err != nil {
				t.Fatalf("ToGenAIPart() error = %v, want nil", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ToGenAIPart() wrong result (-want +got):\n%s", diff)
			}
		})
	}
}

func TestToGenAIPartsRejectsMalformedVideoMetadata(t *testing.T) {
	testCases := []struct {
		name string
		meta any
	}{
		{name: "not an object", meta: "peer-supplied-text"},
		{name: "unparsable offset", meta: map[string]any{"startOffset": "peer-supplied-offset"}},
		{name: "not encodable", meta: make(chan int)},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			p := a2a.NewRawPart([]byte{0xfF})
			p.MediaType = "video/mp4"
			p.SetMeta("adk_video_metadata", tc.meta)

			got, err := ToGenAIPart(p)
			if err == nil {
				t.Fatalf("ToGenAIPart() = %v, want error", got)
			}
			// The error goes back to the A2A client and into logs, so it must
			// not quote what the peer sent.
			if strings.Contains(err.Error(), "peer-supplied") {
				t.Errorf("ToGenAIPart() error = %q, want no peer-supplied value in it", err)
			}
		})
	}
}
