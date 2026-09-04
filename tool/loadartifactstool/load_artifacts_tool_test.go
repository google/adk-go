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

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/loadartifactstool"
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

func TestLoadArtifactsTool_ProcessRequest_Artifacts_LoadArtifactsFunctionCall_AnySlice(t *testing.T) {
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

	// Simulate the function response after a structpb round-trip, where
	// []string becomes []any ([]interface{}).
	functionResponse := &genai.FunctionResponse{
		Name: "load_artifacts",
		Response: map[string]any{
			"artifact_names": []any{"doc1.txt"},
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
		t.Errorf("First part: got %v, want 'Artifact doc1.txt is:'", appendedContent.Parts[0].Text)
	}
	if appendedContent.Parts[1].Text != "This is the content of doc1.txt" {
		t.Errorf("Second part: got %v, want 'This is the content of doc1.txt'", appendedContent.Parts[1].Text)
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

// TestLoadArtifactsTool_ProcessRequest_NilArtifacts verifies that ProcessRequest
// returns a descriptive error instead of panicking when no artifact service is
// configured (see https://github.com/google/adk-go/issues/283).
func TestLoadArtifactsTool_ProcessRequest_NilArtifacts(t *testing.T) {
	loadArtifactsTool := loadartifactstool.New()

	// Construct a context with no artifact service configured.
	invocationCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	tc := agent.NewToolContext(invocationCtx, "", nil, nil)

	requestProcessor, ok := loadArtifactsTool.(toolinternal.RequestProcessor)
	if !ok {
		t.Fatal("loadArtifactsTool does not implement RequestProcessor")
	}

	llmRequest := &model.LLMRequest{}
	err := requestProcessor.ProcessRequest(tc, llmRequest)
	if err == nil {
		t.Fatal("ProcessRequest should return an error when no artifact service is configured, but got nil")
	}
	if !strings.Contains(err.Error(), "artifact service") {
		t.Errorf("error should mention the missing artifact service, got: %v", err)
	}
}

// TestLoadArtifactsTool_ProcessRequest_NilArtifacts_WithFunctionCall verifies
// that the nil-artifact guard in ProcessRequest also covers the
// processLoadArtifactsFunctionCall path, not only appendInitialInstructions:
// with a request whose last content is a load_artifacts function response,
// ProcessRequest must still return the descriptive error instead of panicking
// on the unguarded ctx.Artifacts() read in that path.
func TestLoadArtifactsTool_ProcessRequest_NilArtifacts_WithFunctionCall(t *testing.T) {
	loadArtifactsTool := loadartifactstool.New()

	// Construct a context with no artifact service configured.
	invocationCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	tc := agent.NewToolContext(invocationCtx, "", nil, nil)

	functionResponse := &genai.FunctionResponse{
		Name: "load_artifacts",
		Response: map[string]any{
			"artifact_names": []any{"doc1.txt"},
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
	if err == nil {
		t.Fatal("ProcessRequest should return an error when no artifact service is configured, but got nil")
	}
	if !strings.Contains(err.Error(), "artifact service") {
		t.Errorf("error should mention the missing artifact service, got: %v", err)
	}
}

func TestLoadArtifactsTool_ProcessRequest_Artifacts_MultiPartFunctionResponse(t *testing.T) {
	otherFR := &genai.FunctionResponse{
		Name: "other_function",
		Response: map[string]any{
			"status": "ok",
		},
	}
	loadArtifactsFR := &genai.FunctionResponse{
		Name: "load_artifacts",
		Response: map[string]any{
			"artifact_names": []string{"doc1.txt"},
		},
	}

	tests := []struct {
		name  string
		parts []*genai.Part
	}{
		{
			name: "load_artifacts after other function",
			parts: []*genai.Part{
				genai.NewPartFromFunctionResponse(otherFR.Name, otherFR.Response),
				genai.NewPartFromFunctionResponse(loadArtifactsFR.Name, loadArtifactsFR.Response),
			},
		},
		{
			name: "load_artifacts before other function",
			parts: []*genai.Part{
				genai.NewPartFromFunctionResponse(loadArtifactsFR.Name, loadArtifactsFR.Response),
				genai.NewPartFromFunctionResponse(otherFR.Name, otherFR.Response),
			},
		},
		{
			name: "nil part alongside load_artifacts",
			parts: []*genai.Part{
				nil,
				genai.NewPartFromFunctionResponse(loadArtifactsFR.Name, loadArtifactsFR.Response),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loadArtifactsTool := loadartifactstool.New()
			tc := createToolContext(t)
			if _, err := tc.Artifacts().Save(t.Context(), "doc1.txt", &genai.Part{Text: "This is the content of doc1.txt"}); err != nil {
				t.Fatalf("Failed to save artifact: %v", err)
			}

			llmRequest := &model.LLMRequest{
				Contents: []*genai.Content{
					{
						Role:  genai.RoleUser,
						Parts: tt.parts,
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
				t.Fatalf("Expected 2 contents, but got: %v", len(llmRequest.Contents))
			}

			appendedContent := llmRequest.Contents[1]
			if appendedContent.Role != genai.RoleUser {
				t.Errorf("Appended Content Role: got %v, want %v", appendedContent.Role, genai.RoleUser)
			}
			if len(appendedContent.Parts) != 2 {
				t.Fatalf("Expected 2 parts in appended content, but got: %v", len(appendedContent.Parts))
			}
			if appendedContent.Parts[0].Text != "Artifact doc1.txt is:" {
				t.Errorf("First part of appended content: got %v, want 'Artifact doc1.txt is:'", appendedContent.Parts[0].Text)
			}
			if appendedContent.Parts[1].Text != "This is the content of doc1.txt" {
				t.Errorf("Second part of appended content: got %v, want 'This is the content of doc1.txt'", appendedContent.Parts[1].Text)
			}
		})
	}
}

func TestLoadArtifactsTool_ProcessRequest_MIMEConversion(t *testing.T) {
	tests := []struct {
		name         string
		artifactName string
		part         *genai.Part
		wantInline   bool
		wantMIMEType string
		wantData     []byte
		wantText     string
	}{
		{
			name:         "text-like type is inlined as text",
			artifactName: "data.csv",
			part:         genai.NewPartFromBytes([]byte("col1,col2\n1,2\n"), "application/csv; charset=utf-8"),
			wantText:     "col1,col2\n1,2\n",
		},
		{
			name:         "media type keeps its inline data",
			artifactName: "photo.png",
			part:         genai.NewPartFromBytes([]byte{0x89, 'P', 'N', 'G'}, "image/png"),
			wantInline:   true,
			wantMIMEType: "image/png",
			wantData:     []byte{0x89, 'P', 'N', 'G'},
		},
		{
			name:         "document type keeps its inline data",
			artifactName: "file.pdf",
			part:         genai.NewPartFromBytes([]byte("%PDF-1.4"), "application/pdf"),
			wantInline:   true,
			wantMIMEType: "application/pdf",
			wantData:     []byte("%PDF-1.4"),
		},
		{
			name:         "type casing is ignored",
			artifactName: "notes.txt",
			part:         genai.NewPartFromBytes([]byte("line one\n"), "TEXT/PLAIN"),
			wantText:     "line one\n",
		},
		{
			name:         "unsupported binary type becomes a placeholder",
			artifactName: "slides.pptx",
			part:         genai.NewPartFromBytes([]byte{1, 2, 3}, "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
			wantText:     `[Binary artifact: "slides.pptx", type: "application/vnd.openxmlformats-officedocument.presentationml.presentation", size: 3 bytes. Content cannot be displayed inline.]`,
		},
		{
			name:         "text type holding invalid UTF-8 is decoded with replacement characters",
			artifactName: "broken.txt",
			part:         genai.NewPartFromBytes([]byte{0xff, 0xfe, 0xfd}, "text/plain"),
			wantText:     "\uFFFD",
		},
		{
			name:         "json type is inlined as text",
			artifactName: "config.json",
			part:         genai.NewPartFromBytes([]byte(`{"a":1}`), "application/json"),
			wantText:     `{"a":1}`,
		},
		{
			name:         "xml type is inlined as text",
			artifactName: "feed.xml",
			part:         genai.NewPartFromBytes([]byte("<a/>"), "application/xml"),
			wantText:     "<a/>",
		},
		{
			name:         "audio type keeps its inline data",
			artifactName: "clip.mp3",
			part:         genai.NewPartFromBytes([]byte{0x49, 0x44, 0x33}, "audio/mpeg"),
			wantInline:   true,
			wantMIMEType: "audio/mpeg",
			wantData:     []byte{0x49, 0x44, 0x33},
		},
		{
			name:         "video type keeps its inline data",
			artifactName: "clip.mp4",
			part:         genai.NewPartFromBytes([]byte{0x00, 0x00, 0x00, 0x18}, "video/mp4"),
			wantInline:   true,
			wantMIMEType: "video/mp4",
			wantData:     []byte{0x00, 0x00, 0x00, 0x18},
		},
		{
			name:         "supported-type check strips casing and parameters",
			artifactName: "photo2.png",
			part:         genai.NewPartFromBytes([]byte{0x89, 'P', 'N', 'G'}, "IMAGE/PNG; foo=bar"),
			wantInline:   true,
			wantMIMEType: "IMAGE/PNG; foo=bar",
			wantData:     []byte{0x89, 'P', 'N', 'G'},
		},
		{
			name:         "text in a legacy encoding is decoded lossily",
			artifactName: "sales.csv",
			part:         genai.NewPartFromBytes([]byte("caf\xe9, 12\n"), "application/csv"),
			wantText:     "caf\uFFFD, 12\n",
		},
		{
			name:         "svg image is delivered as text",
			artifactName: "diagram.svg",
			part:         genai.NewPartFromBytes([]byte(`<svg xmlns="http://www.w3.org/2000/svg"/>`), "image/svg+xml"),
			wantText:     `<svg xmlns="http://www.w3.org/2000/svg"/>`,
		},
		{
			name:         "legacy svg subtype is delivered as text",
			artifactName: "old.svg",
			part:         genai.NewPartFromBytes([]byte("<svg/>"), "image/svg"),
			wantText:     "<svg/>",
		},
		{
			name:         "xml image subtype is delivered as text",
			artifactName: "vector.xml",
			part:         genai.NewPartFromBytes([]byte("<v/>"), "image/xml"),
			wantText:     "<v/>",
		},
		{
			name:         "svg application subtype is delivered as text",
			artifactName: "chart.svg",
			part:         genai.NewPartFromBytes([]byte("<svg id=\"c\"/>"), "application/svg+xml"),
			wantText:     "<svg id=\"c\"/>",
		},
		{
			name:         "declared charset is honoured when decoding",
			artifactName: "sales2.csv",
			part:         genai.NewPartFromBytes([]byte("caf\xe9,12\n"), "text/csv; charset=windows-1252"),
			wantText:     "café,12\n",
		},
		{
			name:         "unknown charset falls back to lossy decoding",
			artifactName: "odd.txt",
			part:         genai.NewPartFromBytes([]byte{0xff}, "text/plain; charset=no-such-charset"),
			wantText:     "\uFFFD",
		},
		{
			name:         "control characters in the label do not dodge the text path",
			artifactName: "config2.json",
			part:         genai.NewPartFromBytes([]byte(`{"b":2}`), "application/json\x00"),
			wantText:     `{"b":2}`,
		},
		{
			name:         "control characters in the label do not reopen the inline path",
			artifactName: "diagram2.svg",
			part:         genai.NewPartFromBytes([]byte("<svg/>"), "image/svg+xml\x00"),
			wantText:     "<svg/>",
		},
		{
			name:         "missing type defaults to octet-stream",
			artifactName: "unknown.bin",
			part:         genai.NewPartFromBytes([]byte{1, 2, 3, 4}, ""),
			wantText:     `[Binary artifact: "unknown.bin", type: "application/octet-stream", size: 4 bytes. Content cannot be displayed inline.]`,
		},
		{
			name:         "empty data becomes a placeholder",
			artifactName: "empty.bin",
			part:         &genai.Part{InlineData: &genai.Blob{MIMEType: "application/octet-stream"}},
			wantText:     `[Artifact: "empty.bin", type: "application/octet-stream". No inline data was provided.]`,
		},
		{
			name:         "empty data is reported even for a supported type",
			artifactName: "empty.png",
			part:         &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png"}},
			wantText:     `[Artifact: "empty.png", type: "image/png". No inline data was provided.]`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := processLoadArtifactRequest(t, tt.artifactName, tt.part)

			if !tt.wantInline {
				if got.InlineData != nil {
					t.Fatalf("InlineData: got %v, want nil", got.InlineData)
				}
				if got.Text != tt.wantText {
					t.Errorf("Text: got %q, want %q", got.Text, tt.wantText)
				}
				return
			}

			if got.InlineData == nil {
				t.Fatal("InlineData: got nil, want inline data preserved")
			}
			if got.InlineData.MIMEType != tt.wantMIMEType {
				t.Errorf("InlineData.MIMEType: got %q, want %q", got.InlineData.MIMEType, tt.wantMIMEType)
			}
			if diff := cmp.Diff(tt.wantData, got.InlineData.Data); diff != "" {
				t.Errorf("InlineData.Data diff (-want +got):\n%s", diff)
			}
		})
	}
}

func processLoadArtifactRequest(t *testing.T, artifactName string, part *genai.Part) *genai.Part {
	t.Helper()

	loadArtifactsTool := loadartifactstool.New()
	tc := createToolContext(t)
	if _, err := tc.Artifacts().Save(t.Context(), artifactName, part); err != nil {
		t.Fatalf("Failed to save artifact %s: %v", artifactName, err)
	}

	llmRequest := loadArtifactsRequest(artifactName)
	requestProcessor, ok := loadArtifactsTool.(toolinternal.RequestProcessor)
	if !ok {
		t.Fatal("loadArtifactsTool does not implement RequestProcessor")
	}

	if err := requestProcessor.ProcessRequest(tc, llmRequest); err != nil {
		t.Fatalf("ProcessRequest failed: %v", err)
	}
	if len(llmRequest.Contents) != 2 {
		t.Fatalf("Expected 2 contents, got %d", len(llmRequest.Contents))
	}

	appendedContent := llmRequest.Contents[1]
	if appendedContent.Role != genai.RoleUser {
		t.Errorf("Appended Content Role: got %v, want %v", appendedContent.Role, genai.RoleUser)
	}
	if len(appendedContent.Parts) != 2 {
		t.Fatalf("Expected 2 parts in appended content, got %d", len(appendedContent.Parts))
	}
	if got, want := appendedContent.Parts[0].Text, "Artifact "+artifactName+" is:"; got != want {
		t.Errorf("First part of appended content: got %q, want %q", got, want)
	}
	return appendedContent.Parts[1]
}

func loadArtifactsRequest(artifactName string) *model.LLMRequest {
	return &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "model",
				Parts: []*genai.Part{
					genai.NewPartFromFunctionResponse("load_artifacts", map[string]any{
						"artifact_names": []string{artifactName},
					}),
				},
			},
		},
	}
}

func createToolContext(t *testing.T) agent.Context {
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

	return agent.NewToolContext(ctx, "", nil, nil)
}
