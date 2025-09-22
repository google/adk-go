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

package tool_test

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/artifactsinternal"
	"google.golang.org/adk/internal/toolinternal"

	"google.golang.org/adk/artifactservice"
	"google.golang.org/adk/llm"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

func TestLoadArtifactsTool_Run(t *testing.T) {

	loadArtifactsTool := tool.NewLoadArtifactsTool()

	tc := createToolContext(t)

	args := map[string]any{
		"artifact_names": []string{"file1", "file2"},
	}
	toolImpl, ok := loadArtifactsTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("loadArtifactsTool does not implement FunctionTool")
	}
	result, err := toolImpl.Run(tc, args)
	if err != nil {
		t.Fatalf("Run with args failed: %v", err)
	}
	expected := map[string]any{
		"artifact_names": []string{"file1", "file2"},
	}
	if diff := cmp.Diff(expected, result); diff != "" {
		t.Errorf("Run with args result diff (-want +got):\n%s", diff)
	}

	// Test without artifact names
	args = map[string]any{}
	result, err = toolImpl.Run(tc, args)
	if err != nil {
		t.Fatalf("Run without args failed: %v", err)
	}
	expected = map[string]any{
		"artifact_names": []string{},
	}
	if diff := cmp.Diff(expected, result); diff != "" {
		t.Errorf("Run without args result diff (-want +got):\n%s", diff)
	}
}

func TestLoadArtifactsTool_ProcessRequest(t *testing.T) {
	loadArtifactsTool := tool.NewLoadArtifactsTool()

	tc := createToolContext(t)
	artifacts := map[string]*genai.Part{
		"file1.txt": {Text: "content1"},
		"file2.pdf": {Text: "content2"},
	}
	for name, part := range artifacts {
		err := tc.Artifacts().Save(name, *part)
		if err != nil {
			t.Fatalf("Failed to save artifact %s: %v", name, err)
		}
	}

	llmRequest := &llm.Request{}

	requestProcessor, ok := loadArtifactsTool.(toolinternal.RequestProcessor)
	if !ok {
		t.Fatal("loadArtifactsTool does not implement RequestProcessor")
	}

	err := requestProcessor.ProcessRequest(tc, llmRequest)
	if err != nil {
		t.Fatalf("ProcessRequest failed: %v", err)
	}

	instruction := llmRequest.GenerateConfig.SystemInstruction.Parts[0].Text
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
	loadArtifactsTool := tool.NewLoadArtifactsTool()

	tc := createToolContext(t)
	artifacts := map[string]*genai.Part{
		"doc1.txt": {Text: "This is the content of doc1.txt"},
	}
	for name, part := range artifacts {
		err := tc.Artifacts().Save(name, *part)
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
	llmRequest := &llm.Request{
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

func TestLoadArtifactsTool_ProcessRequest_Artifacts_OtherFunctionCall(t *testing.T) {
	loadArtifactsTool := tool.NewLoadArtifactsTool()

	tc := createToolContext(t)
	artifacts := map[string]*genai.Part{
		"doc1.txt": {Text: "content1"},
	}
	for name, part := range artifacts {
		err := tc.Artifacts().Save(name, *part)
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
	llmRequest := &llm.Request{
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

	sessionId := session.ID{
		AppName:   "app",
		UserID:    "user",
		SessionID: "session",
	}

	artifactsImpl := artifactsinternal.NewArtifacts(artifactservice.Mem(), sessionId)

	agentCtx := agent.NewContext(t.Context(), nil, nil, artifactsImpl, nil, nil, "")

	return tool.NewContext(agentCtx, "", nil)
}
