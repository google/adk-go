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

package loadartifactstool_test

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/artifact"
	artifactinternal "google.golang.org/adk/internal/artifact"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/loadartifactstool"
)

func TestLoadArtifactsTool_Run(t *testing.T) {
	loadArtifactsTool := loadartifactstool.New()
	tc := createToolContext(t)

	toolImpl, ok := loadArtifactsTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("loadArtifactsTool does not implement FunctionTool")
	}

	tests := []struct {
		name    string
		args    map[string]any
		want    map[string]any
		wantErr bool
	}{
		{
			name: "basic string slice",
			args: map[string]any{
				"artifact_names": []string{"file1", "file2"},
			},
			want: map[string]any{
				"artifact_names": []string{"file1", "file2"},
			},
		},
		{
			name: "empty args",
			args: map[string]any{},
			want: map[string]any{
				"artifact_names": []string{},
			},
		},
		{
			name: "any slice with strings",
			args: map[string]any{
				"artifact_names": []any{"fileA", "fileB"},
			},
			want: map[string]any{
				"artifact_names": []string{"fileA", "fileB"},
			},
		},
		{
			name: "empty string slice",
			args: map[string]any{
				"artifact_names": []string{},
			},
			want: map[string]any{
				"artifact_names": []string{},
			},
		},
		{
			name: "empty any slice",
			args: map[string]any{
				"artifact_names": []any{},
			},
			want: map[string]any{
				"artifact_names": []string{},
			},
		},
		{
			name: "nil value",
			args: map[string]any{
				"artifact_names": nil,
			},
			want: map[string]any{
				"artifact_names": []string{},
			},
		},
		{
			name: "incorrect type (not a slice)",
			args: map[string]any{
				"artifact_names": "not a slice",
			},
			wantErr: true,
		},
		{
			name: "any slice with non-string",
			args: map[string]any{
				"artifact_names": []any{"fileA", 123},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := toolImpl.Run(tc, tt.args)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Run() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}

			if diff := cmp.Diff(tt.want, result); diff != "" {
				t.Errorf("Run() result diff (-want +got):\n%s", diff)
			}
		})
	}
}

func TestLoadArtifactsTool_ProcessRequest(t *testing.T) {
	loadArtifactsTool := loadartifactstool.New()

	tc := createToolContext(t)
	artifacts := map[string]*genai.Part{
		"file1.txt": {Text: "content1"},
		"file2.pdf": {Text: "content2"},
	}
	for name, part := range artifacts {
		_, err := tc.Artifacts().Save(t.Context(), name, part)
		if err != nil {
			t.Fatalf("Failed to save artifact %s: %v", name, err)
		}
	}

	llmRequest := &model.LLMRequest{}

	requestProcessor, ok := loadArtifactsTool.(toolinternal.RequestProcessor)
	if !ok {
		t.Fatal("loadArtifactsTool does not implement RequestProcessor")
	}

	err := requestProcessor.ProcessRequest(tc, llmRequest)
	if err != nil {
		t.Fatalf("ProcessRequest failed: %v", err)
	}

	instruction := llmRequest.Config.SystemInstruction.Parts[0].Text
	if !strings.Contains(instruction, "You have a list of artifacts") {
		t.Errorf("Instruction should contain 'You have a list of artifacts', but got: %v", instruction)
	}
	if !strings.Contains(instruction, `"file1.txt"`) || !strings.Contains(instruction, `"file2.pdf"`) {
		t.Errorf("Instruction should contain artifact names, but got: %v", instruction)
	}
	if len(llmRequest.Contents) > 0 {
		t.Errorf("Expected no contents, but got: %v", llmRequest.Contents)
	}
}

func TestLoadArtifactsTool_ProcessRequest_Artifacts_LoadArtifactsFunctionCall(t *testing.T) {
	loadArtifactsTool := loadartifactstool.New()

	tc := createToolContext(t)
	artifacts := map[string]*genai.Part{
		"doc1.txt": {Text: "This is the content of doc1.txt"},
	}
	for name, part := range artifacts {
		_, err := tc.Artifacts().Save(t.Context(), name, part)
		if err != nil {
			t.Fatalf("Failed to save artifact %s: %v", name, err)
		}
	}

	functionResponse := &genai.FunctionResponse{
		Name: "load_artifacts",
		Response: map[string]any{
			"artifact_names": []string{"doc1.txt"},
		},
	}
	llmRequest := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "model",
				Parts: []*genai.Part{
					genai.NewPartFromFunctionResponse(functionResponse.Name, functionResponse.Response),
				},
			},
		},
	}

	requestProcessor, ok := loadArtifactsTool.(toolinternal.RequestProcessor)
	if !ok {
		t.Fatal("loadArtifactsTool does not implement RequestProcessor")
	}

	err := requestProcessor.ProcessRequest(tc, llmRequest)
	if err != nil {
		t.Fatalf("ProcessRequest failed: %v", err)
	}

	if len(llmRequest.Contents) != 2 {
		t.Fatalf("Expected 2 content, but got: %v", llmRequest.Contents)
	}

	appendedContent := llmRequest.Contents[1]
	if appendedContent.Role != "user" {
		t.Errorf("Appended Content Role: got %v, want 'user'", appendedContent.Role)
	}
	if len(appendedContent.Parts) != 2 {
		t.Fatalf("Expected 2 parts in appended content, but got: %v", appendedContent.Parts)
	}
	if appendedContent.Parts[0].Text != "Artifact doc1.txt is:" {
		t.Errorf("First part of appended content: got %v, want 'Artifact doc1.txt is:'", appendedContent.Parts[0].Text)
	}
	if appendedContent.Parts[1].Text != "This is the content of doc1.txt" {
		t.Errorf("Second part of appended content: got %v, want 'This is the content of doc1.txt'", appendedContent.Parts[1].Text)
	}
}

func TestLoadArtifactsTool_ProcessRequest_Artifacts_SafeInlineData(t *testing.T) {
	tests := []struct {
		name              string
		artifactName      string
		part              *genai.Part
		wantText          string
		wantTextContains  []string
		wantInlineMIME    string
		wantInlineData    []byte
		wantInlineCleared bool
	}{
		{
			name:              "csv becomes text",
			artifactName:      "data.csv",
			part:              genai.NewPartFromBytes([]byte("a,b\n1,2\n"), "application/csv"),
			wantText:          "a,b\n1,2\n",
			wantInlineCleared: true,
		},
		{
			name:           "pdf stays inline",
			artifactName:   "report.pdf",
			part:           genai.NewPartFromBytes([]byte("%PDF-1.7"), "application/pdf"),
			wantInlineMIME: "application/pdf",
			wantInlineData: []byte("%PDF-1.7"),
		},
		{
			name:         "unsupported binary becomes placeholder",
			artifactName: "deck.pptx",
			part: genai.NewPartFromBytes(
				[]byte{0x50, 0x4b, 0x03, 0x04},
				"application/vnd.openxmlformats-officedocument.presentationml.presentation",
			),
			wantTextContains: []string{
				"deck.pptx",
				"application/vnd.openxmlformats-officedocument.presentationml.presentation",
				"4 bytes",
			},
			wantInlineCleared: true,
		},
		{
			name:              "empty mime valid utf8 becomes text",
			artifactName:      "notes",
			part:              genai.NewPartFromBytes([]byte("plain notes"), ""),
			wantText:          "plain notes",
			wantInlineCleared: true,
		},
		{
			name:              "empty inline data becomes placeholder",
			artifactName:      "empty.csv",
			part:              genai.NewPartFromBytes(nil, "application/csv"),
			wantTextContains:  []string{"empty.csv", "no inline data"},
			wantInlineCleared: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loadArtifactsTool := loadartifactstool.New()
			tc := createToolContext(t)
			if _, err := tc.Artifacts().Save(t.Context(), tt.artifactName, tt.part); err != nil {
				t.Fatalf("Failed to save artifact %s: %v", tt.artifactName, err)
			}

			llmRequest := &model.LLMRequest{
				Contents: []*genai.Content{
					{
						Role: "model",
						Parts: []*genai.Part{
							genai.NewPartFromFunctionResponse("load_artifacts", map[string]any{
								"artifact_names": []string{tt.artifactName},
							}),
						},
					},
				},
			}

			requestProcessor, ok := loadArtifactsTool.(toolinternal.RequestProcessor)
			if !ok {
				t.Fatal("loadArtifactsTool does not implement RequestProcessor")
			}
			if err := requestProcessor.ProcessRequest(tc, llmRequest); err != nil {
				t.Fatalf("ProcessRequest failed: %v", err)
			}

			if len(llmRequest.Contents) != 2 {
				t.Fatalf("Expected 2 contents, got: %v", llmRequest.Contents)
			}
			gotPart := llmRequest.Contents[1].Parts[1]
			if tt.wantInlineMIME != "" {
				if gotPart.InlineData == nil {
					t.Fatal("Expected inline data, got nil")
				}
				if gotPart.InlineData.MIMEType != tt.wantInlineMIME {
					t.Errorf("Inline MIME type = %q, want %q", gotPart.InlineData.MIMEType, tt.wantInlineMIME)
				}
				if diff := cmp.Diff(tt.wantInlineData, gotPart.InlineData.Data); diff != "" {
					t.Errorf("Inline data diff (-want +got):\n%s", diff)
				}
			}
			if tt.wantInlineCleared && gotPart.InlineData != nil {
				t.Fatalf("Expected inline data to be cleared, got: %v", gotPart.InlineData)
			}
			if tt.wantText != "" && gotPart.Text != tt.wantText {
				t.Errorf("Text = %q, want %q", gotPart.Text, tt.wantText)
			}
			for _, want := range tt.wantTextContains {
				if !strings.Contains(gotPart.Text, want) {
					t.Errorf("Text = %q, want substring %q", gotPart.Text, want)
				}
			}
		})
	}
}

func TestLoadArtifactsTool_ProcessRequest_Artifacts_OtherFunctionCall(t *testing.T) {
	loadArtifactsTool := loadartifactstool.New()

	tc := createToolContext(t)
	artifacts := map[string]*genai.Part{
		"doc1.txt": {Text: "content1"},
	}
	for name, part := range artifacts {
		_, err := tc.Artifacts().Save(t.Context(), name, part)
		if err != nil {
			t.Fatalf("Failed to save artifact %s: %v", name, err)
		}
	}

	functionResponse := &genai.FunctionResponse{
		Name: "other_function",
		Response: map[string]any{
			"some_key": "some_value",
		},
	}
	llmRequest := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "model",
				Parts: []*genai.Part{
					genai.NewPartFromFunctionResponse(functionResponse.Name, functionResponse.Response),
				},
			},
		},
	}

	requestProcessor, ok := loadArtifactsTool.(toolinternal.RequestProcessor)
	if !ok {
		t.Fatal("loadArtifactsTool does not implement RequestProcessor")
	}

	err := requestProcessor.ProcessRequest(tc, llmRequest)
	if err != nil {
		t.Fatalf("ProcessRequest failed: %v", err)
	}
	if len(llmRequest.Contents) != 1 {
		t.Fatalf("Expected 1 content, but got: %v", llmRequest.Contents)
	}
	if llmRequest.Contents[0].Role != "model" {
		t.Errorf("Content Role: got %v, want 'model'", llmRequest.Contents[0].Role)
	}
}

func createToolContext(t *testing.T) tool.Context {
	t.Helper()

	artifacts := &artifactinternal.Artifacts{
		Service:   artifact.InMemoryService(),
		AppName:   "app",
		UserID:    "user",
		SessionID: "session",
	}

	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Artifacts: artifacts,
	})

	return toolinternal.NewToolContext(ctx, "", nil, nil)
}
