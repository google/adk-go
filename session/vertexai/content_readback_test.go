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

package vertexai

import (
	"testing"
	"time"

	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
	"google.golang.org/protobuf/types/known/durationpb"
	"google.golang.org/protobuf/types/known/structpb"
)

// TestAiplatformToGenaiContent_ReadsBackSentFields checks that a stored event
// without raw_event keeps the part fields that internal/vertexai sends and
// adk-python reads back from the content column: video metadata, a function
// call's partial arguments and WillContinue, and a function response's parts.
func TestAiplatformToGenaiContent_ReadsBackSentFields(t *testing.T) {
	tests := []struct {
		name string
		part *aiplatformpb.Part
		want *genai.Part
	}{
		{
			name: "video metadata",
			part: &aiplatformpb.Part{
				Data: &aiplatformpb.Part_FileData{FileData: &aiplatformpb.FileData{MimeType: "video/mp4", FileUri: "gs://b/v.mp4"}},
				Metadata: &aiplatformpb.Part_VideoMetadata{VideoMetadata: &aiplatformpb.VideoMetadata{
					StartOffset: durationpb.New(2 * time.Second),
					EndOffset:   durationpb.New(5 * time.Second),
				}},
			},
			want: &genai.Part{
				FileData:      &genai.FileData{MIMEType: "video/mp4", FileURI: "gs://b/v.mp4"},
				VideoMetadata: &genai.VideoMetadata{StartOffset: 2 * time.Second, EndOffset: 5 * time.Second},
			},
		},
		{
			name: "function call partial arguments and WillContinue",
			part: &aiplatformpb.Part{Data: &aiplatformpb.Part_FunctionCall{FunctionCall: &aiplatformpb.FunctionCall{
				Id:   "c1",
				Name: "f",
				PartialArgs: []*aiplatformpb.PartialArg{
					{JsonPath: "$.s", Delta: &aiplatformpb.PartialArg_StringValue{StringValue: "ab"}, WillContinue: true},
					{JsonPath: "$.n", Delta: &aiplatformpb.PartialArg_NumberValue{NumberValue: 1.5}},
					{JsonPath: "$.b", Delta: &aiplatformpb.PartialArg_BoolValue{BoolValue: true}},
					{JsonPath: "$.z", Delta: &aiplatformpb.PartialArg_NullValue{NullValue: structpb.NullValue_NULL_VALUE}},
				},
				WillContinue: true,
			}}},
			want: &genai.Part{FunctionCall: &genai.FunctionCall{
				ID:   "c1",
				Name: "f",
				Args: map[string]any{},
				PartialArgs: []*genai.PartialArg{
					{JsonPath: "$.s", StringValue: "ab", WillContinue: new(true)},
					{JsonPath: "$.n", NumberValue: new(1.5)},
					{JsonPath: "$.b", BoolValue: new(true)},
					{JsonPath: "$.z", NULLValue: "NULL_VALUE"},
				},
				WillContinue: new(true),
			}},
		},
		{
			name: "function response parts",
			part: &aiplatformpb.Part{Data: &aiplatformpb.Part_FunctionResponse{FunctionResponse: &aiplatformpb.FunctionResponse{
				Id:   "c1",
				Name: "f",
				Parts: []*aiplatformpb.FunctionResponsePart{
					{Data: &aiplatformpb.FunctionResponsePart_InlineData{InlineData: &aiplatformpb.FunctionResponseBlob{MimeType: "image/png", Data: []byte{1, 2}, DisplayName: "img"}}},
					{Data: &aiplatformpb.FunctionResponsePart_FileData{FileData: &aiplatformpb.FunctionResponseFileData{MimeType: "text/plain", FileUri: "gs://b/f.txt", DisplayName: "doc"}}},
				},
			}}},
			want: &genai.Part{FunctionResponse: &genai.FunctionResponse{
				ID:       "c1",
				Name:     "f",
				Response: map[string]any{},
				Parts: []*genai.FunctionResponsePart{
					{InlineData: &genai.FunctionResponseBlob{MIMEType: "image/png", Data: []byte{1, 2}, DisplayName: "img"}},
					{FileData: &genai.FunctionResponseFileData{MIMEType: "text/plain", FileURI: "gs://b/f.txt", DisplayName: "doc"}},
				},
			}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := aiplatformToGenaiContent(&aiplatformpb.SessionEvent{
				Content: &aiplatformpb.Content{Role: "model", Parts: []*aiplatformpb.Part{tt.part}},
			})

			want := &genai.Content{Role: "model", Parts: []*genai.Part{tt.want}}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("aiplatformToGenaiContent() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
