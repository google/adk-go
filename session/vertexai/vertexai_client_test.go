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

	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestAiplatformToGenaiContent_FunctionCallID(t *testing.T) {
	args, err := structpb.NewStruct(map[string]any{"query": "hello"})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := structpb.NewStruct(map[string]any{"result": "world"})
	if err != nil {
		t.Fatal(err)
	}

	event := &aiplatformpb.SessionEvent{
		Content: &aiplatformpb.Content{
			Role: "model",
			Parts: []*aiplatformpb.Part{
				{
					Data: &aiplatformpb.Part_FunctionCall{
						FunctionCall: &aiplatformpb.FunctionCall{
							Id:   "call-123",
							Name: "search",
							Args: args,
						},
					},
				},
				{
					Data: &aiplatformpb.Part_FunctionResponse{
						FunctionResponse: &aiplatformpb.FunctionResponse{
							Id:       "call-123",
							Name:     "search",
							Response: resp,
						},
					},
				},
			},
		},
	}

	content := aiplatformToGenaiContent(event)
	if content == nil {
		t.Fatal("expected non-nil content")
	}
	if len(content.Parts) != 2 {
		t.Fatalf("got %d parts, want 2", len(content.Parts))
	}

	fc := content.Parts[0].FunctionCall
	if fc == nil {
		t.Fatal("expected FunctionCall part")
	}
	if fc.ID != "call-123" {
		t.Errorf("FunctionCall.ID = %q, want %q", fc.ID, "call-123")
	}
	if fc.Name != "search" {
		t.Errorf("FunctionCall.Name = %q, want %q", fc.Name, "search")
	}

	fr := content.Parts[1].FunctionResponse
	if fr == nil {
		t.Fatal("expected FunctionResponse part")
	}
	if fr.ID != "call-123" {
		t.Errorf("FunctionResponse.ID = %q, want %q", fr.ID, "call-123")
	}
	if fr.Name != "search" {
		t.Errorf("FunctionResponse.Name = %q, want %q", fr.Name, "search")
	}
}
