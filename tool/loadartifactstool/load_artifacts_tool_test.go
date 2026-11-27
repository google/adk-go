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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net"
	"net/url"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/artifact"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/loadartifactstool"
)

func TestLoadArtifactsTool_Run(t *testing.T) {
	loadArtifactsTool := loadartifactstool.New()
	tc := createToolContext(t)

	toolImpl, ok := loadArtifactsTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("loadArtifactsTool does not implement FunctionTool")
	}

	// Arguments the model can correct get an error result, not a Go error.
	invalidArtifactNamesResult := map[string]any{
		"error":      "'artifact_names' must be a list of strings.",
		"error_code": "INVALID_ARGUMENTS",
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
			want: invalidArtifactNamesResult,
		},
		{
			name: "incorrect type (object)",
			args: map[string]any{
				"artifact_names": map[string]any{"name": "fileA"},
			},
			want: invalidArtifactNamesResult,
		},
		{
			name: "any slice with non-string",
			args: map[string]any{
				"artifact_names": []any{"fileA", 123},
			},
			want: invalidArtifactNamesResult,
		},
		{
			name: "any slice with empty string",
			args: map[string]any{
				"artifact_names": []any{"fileA", ""},
			},
			want: map[string]any{
				"artifact_names": []string{"fileA", ""},
			},
		},
		{
			name: "any slice with null",
			args: map[string]any{
				"artifact_names": []any{"fileA", nil},
			},
			want: invalidArtifactNamesResult,
		},
		{
			name: "specific version as float64 from JSON",
			args: map[string]any{
				"artifact_names": []any{"design.md"},
				"version":        float64(2),
			},
			want: map[string]any{
				"artifact_names": []string{"design.md"},
				"version":        2,
			},
		},
		{
			name: "version 0 is kept",
			args: map[string]any{
				"artifact_names": []any{"design.md"},
				"version":        float64(0),
			},
			want: map[string]any{
				"artifact_names": []string{"design.md"},
				"version":        0,
			},
		},
		{
			name: "artifact_versions selector without version",
			args: map[string]any{
				"artifact_versions": []any{
					map[string]any{"name": "design.md"},
					map[string]any{"name": "design.md", "version": float64(0)},
				},
			},
			want: map[string]any{
				"artifact_names": []string{},
				"artifact_versions": []map[string]any{
					{"name": "design.md"},
					{"name": "design.md", "version": 0},
				},
			},
		},
		{
			name: "artifact_versions selector list",
			args: map[string]any{
				"artifact_versions": []any{
					map[string]any{"name": "design.md", "version": float64(1)},
					map[string]any{"name": "design.md", "version": float64(2)},
				},
			},
			want: map[string]any{
				"artifact_names": []string{},
				"artifact_versions": []map[string]any{
					{"name": "design.md", "version": 1},
					{"name": "design.md", "version": 2},
				},
			},
		},
		{
			name: "list_versions boolean",
			args: map[string]any{
				"artifact_names": []any{"design.md"},
				"list_versions":  true,
			},
			want: map[string]any{
				"artifact_names": []string{"design.md"},
				"list_versions":  true,
			},
		},
		{
			name: "list_versions string slice",
			args: map[string]any{
				"list_versions": []any{"design.md"},
			},
			want: map[string]any{
				"artifact_names": []string{"design.md"},
				"list_versions":  true,
			},
		},
		{
			name: "negative version is rejected",
			args: map[string]any{
				"artifact_names": []any{"design.md"},
				"version":        -1,
			},
			wantErr: true,
		},
		{
			name: "fractional version is rejected",
			args: map[string]any{
				"artifact_names": []any{"design.md"},
				"version":        1.5,
			},
			wantErr: true,
		},
		{
			name: "artifact_versions entry missing name is rejected",
			args: map[string]any{
				"artifact_versions": []any{
					map[string]any{"version": 1},
				},
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
			"artifact_names": []string{"unrelated.txt"},
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

			if _, err := tc.Artifacts().Save(t.Context(), "unrelated.txt", genai.NewPartFromText("unrelated body")); err != nil {
				t.Fatalf("Save() error = %v", err)
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

func TestLoadArtifactsTool_ParallelCalls(t *testing.T) {
	const (
		app, user, sid    = "app", "user", "session"
		january, february = "january body", "february body"
	)
	backendError := errors.New("storage unavailable")
	for _, tc := range []struct {
		name        string
		firstArgs   []any
		secondFile  string
		failLoadFor string
		loadErr     error
		wantBodies  []string
		wantErr     error
	}{
		{name: "two successful calls", firstArgs: []any{"january.txt"}, secondFile: "february.txt", wantBodies: []string{january, february}},
		{name: "invalid arguments before successful call", firstArgs: []any{123}, secondFile: "february.txt", wantBodies: []string{february}},
		{name: "missing file follows successful call", firstArgs: []any{"january.txt"}, secondFile: "missing.txt", wantBodies: []string{january}},
		{name: "missing file precedes successful call", firstArgs: []any{"missing.txt"}, secondFile: "february.txt", wantBodies: []string{february}},
		{name: "invalid filename follows successful call", firstArgs: []any{"january.txt"}, secondFile: "dir/february.txt", wantBodies: []string{january}},
		{name: "file disappears after listing", firstArgs: []any{"january.txt"}, secondFile: "february.txt", failLoadFor: "january.txt", loadErr: fs.ErrNotExist, wantBodies: []string{february}},
		{name: "backend failure", firstArgs: []any{"january.txt"}, secondFile: "february.txt", failLoadFor: "february.txt", loadErr: backendError, wantErr: backendError},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			artifacts := artifact.InMemoryService()
			for name, text := range map[string]string{"january.txt": january, "february.txt": february} {
				if _, err := artifacts.Save(ctx, &artifact.SaveRequest{
					AppName: app, UserID: user, SessionID: sid, FileName: name,
					Part: genai.NewPartFromText(text),
				}); err != nil {
					t.Fatalf("Save() error = %v", err)
				}
			}
			first := genai.NewPartFromFunctionCall("load_artifacts", map[string]any{"artifact_names": tc.firstArgs})
			first.FunctionCall.ID = "call-1"
			second := genai.NewPartFromFunctionCall("load_artifacts", map[string]any{"artifact_names": []any{tc.secondFile}})
			second.FunctionCall.ID = "call-2"
			mock := &testutil.MockModel{Responses: []*genai.Content{
				{Role: genai.RoleModel, Parts: []*genai.Part{first, second}},
				genai.NewContentFromText("done", genai.RoleModel),
			}}
			a, err := llmagent.New(llmagent.Config{
				Name: "reader", Model: mock, Tools: []tool.Tool{loadartifactstool.New()},
			})
			if err != nil {
				t.Fatalf("llmagent.New() error = %v", err)
			}
			sessions := session.InMemoryService()
			if _, err := sessions.Create(ctx, &session.CreateRequest{AppName: app, UserID: user, SessionID: sid}); err != nil {
				t.Fatalf("Create() error = %v", err)
			}
			if tc.loadErr != nil {
				artifacts = &failingArtifactLoad{Service: artifacts, name: tc.failLoadFor, err: tc.loadErr}
			}
			r, err := runner.New(runner.Config{
				AppName: app, Agent: a, SessionService: sessions, ArtifactService: artifacts,
			})
			if err != nil {
				t.Fatalf("runner.New() error = %v", err)
			}
			prompt := genai.NewContentFromText("Compare January and February", genai.RoleUser)
			for _, err := range r.Run(ctx, user, sid, prompt, agent.RunConfig{}) {
				if err != nil {
					if tc.wantErr != nil && errors.Is(err, tc.wantErr) {
						return
					}
					t.Fatalf("Run() returned an unexpected error of type %T", err)
				}
			}
			if tc.wantErr != nil {
				t.Fatal("Run() succeeded despite a backend failure")
			}
			if len(mock.Requests) != 2 {
				t.Fatalf("model requests = %d, want 2", len(mock.Requests))
			}
			var responses int
			var bodies []string
			for _, content := range mock.Requests[1].Contents {
				if content == nil {
					t.Fatal("request includes nil content")
				}
				for _, part := range content.Parts {
					if part.FunctionResponse != nil && part.FunctionResponse.Name == "load_artifacts" {
						responses++
					}
					if part.Text == january || part.Text == february {
						bodies = append(bodies, part.Text)
					}
				}
			}
			if responses != 2 {
				t.Errorf("load_artifacts responses = %d, want 2", responses)
			}
			if len(bodies) != len(tc.wantBodies) {
				t.Fatalf("attached artifact bodies = %d, want %d", len(bodies), len(tc.wantBodies))
			}
			if !slices.Equal(bodies, tc.wantBodies) {
				t.Error("artifact bodies do not match the expected tool response order")
			}
		})
	}
}

type failingArtifactLoad struct {
	artifact.Service
	name string
	err  error
}

func (s *failingArtifactLoad) Load(ctx context.Context, req *artifact.LoadRequest) (*artifact.LoadResponse, error) {
	if req.FileName == s.name {
		return nil, s.err
	}
	return s.Service.Load(ctx, req)
}

func TestLoadArtifactsTool_ProcessRequest_MalformedResponse(t *testing.T) {
	for _, tc := range []struct {
		name      string
		malformed any
	}{
		{"not a list", "bad.txt"},
		{"list with a non-string element", []any{"bad.txt", 123}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := createToolContext(t)
			for _, name := range []string{"ok.txt", "bad.txt"} {
				if _, err := ctx.Artifacts().Save(t.Context(), name, genai.NewPartFromText("body")); err != nil {
					t.Fatalf("Save() returned an error of type %T", err)
				}
			}
			processor := loadartifactstool.New().(toolinternal.RequestProcessor)
			malformed := genai.NewPartFromFunctionResponse("load_artifacts", map[string]any{"artifact_names": tc.malformed})
			valid := genai.NewPartFromFunctionResponse("load_artifacts", map[string]any{"artifact_names": []any{"ok.txt"}})
			t.Run("single", func(t *testing.T) {
				req := &model.LLMRequest{Contents: []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{malformed}}}}
				if err := processor.ProcessRequest(ctx, req); err == nil {
					t.Fatal("ProcessRequest() accepted a malformed response")
				}
			})
			t.Run("parallel", func(t *testing.T) {
				req := &model.LLMRequest{Contents: []*genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{malformed, valid}}}}
				if err := processor.ProcessRequest(ctx, req); err != nil {
					t.Fatalf("ProcessRequest() returned an error of type %T", err)
				}
				if len(req.Contents) != 2 || req.Contents[1].Parts[0].Text != "Artifact ok.txt is:" {
					t.Fatal("ProcessRequest() did not attach exactly the valid artifact")
				}
			})
		})
	}
}

func TestLoadArtifactsTool_ProcessRequest_LoadError(t *testing.T) {
	tests := []struct {
		name          string
		artifactNames []string
	}{
		{
			name:          "missing artifact",
			artifactNames: []string{"missing.txt"},
		},
		{
			name:          "missing artifact alongside a saved one",
			artifactNames: []string{"doc1.txt", "missing.txt"},
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
						Role: genai.RoleUser,
						Parts: []*genai.Part{
							genai.NewPartFromFunctionResponse("load_artifacts", map[string]any{
								"artifact_names": tt.artifactNames,
							}),
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
				t.Fatal("ProcessRequest should return an error when an artifact cannot be loaded, but got nil")
			}
			if got := strings.Count(err.Error(), "failed to load artifact"); got != 1 {
				t.Errorf("error names the failure %d times, want once: %v", got, err)
			}
			if !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("errors.Is(err, fs.ErrNotExist) = false, want true, got: %v", err)
			}
		})
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

func TestLoadArtifactsTool_ProcessRequest_FirstRevisionIsZero(t *testing.T) {
	tc := createToolContext(t)
	for _, body := range []string{"draft v0", "draft v1"} {
		if _, err := tc.Artifacts().Save(t.Context(), "design.md", genai.NewPartFromText(body)); err != nil {
			t.Fatalf("Save(%q): %v", body, err)
		}
	}
	tests := []struct {
		name       string
		response   map[string]any
		wantHeader string
		wantBody   string
	}{
		{
			name:       "top-level version 0",
			response:   map[string]any{"artifact_names": []any{"design.md"}, "version": float64(0)},
			wantHeader: "Artifact design.md (version 0) is:",
			wantBody:   "draft v0",
		},
		{
			name:       "artifact_versions version 0",
			response:   map[string]any{"artifact_versions": []any{map[string]any{"name": "design.md", "version": float64(0)}}},
			wantHeader: "Artifact design.md (version 0) is:",
			wantBody:   "draft v0",
		},
		{
			name:       "artifact_versions without version loads the latest",
			response:   map[string]any{"artifact_versions": []any{map[string]any{"name": "design.md"}}},
			wantHeader: "Artifact design.md is:",
			wantBody:   "draft v1",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &model.LLMRequest{Contents: []*genai.Content{{
				Role:  genai.RoleUser,
				Parts: []*genai.Part{genai.NewPartFromFunctionResponse("load_artifacts", tt.response)},
			}}}
			if err := loadartifactstool.New().(toolinternal.RequestProcessor).ProcessRequest(tc, req); err != nil {
				t.Fatalf("ProcessRequest: %v", err)
			}
			if len(req.Contents) != 2 {
				t.Fatalf("len(req.Contents) = %d, want 2", len(req.Contents))
			}
			if got := req.Contents[1].Parts[0].Text; got != tt.wantHeader {
				t.Errorf("header = %q, want %q", got, tt.wantHeader)
			}
			if got := req.Contents[1].Parts[1].Text; got != tt.wantBody {
				t.Errorf("body = %q, want %q", got, tt.wantBody)
			}
		})
	}
}

func TestLoadArtifactsTool_Declaration_RevisionsStartAtZero(t *testing.T) {
	decl := loadartifactstool.New().(toolinternal.FunctionTool).Declaration()
	props := decl.Parameters.Properties
	version, ok := props["version"]
	if !ok {
		t.Fatal("declaration has no version parameter")
	}
	want := "Optional revision number to load for each artifact in artifact_names. Revisions are numbered from 0. If omitted, the latest revision is loaded."
	if got := version.Description; got != want {
		t.Errorf("version description = %q, want %q", got, want)
	}
	selectors, ok := props["artifact_versions"]
	if !ok || selectors.Items == nil || selectors.Items.Properties["version"] == nil {
		t.Fatal("declaration has no artifact_versions[].version parameter")
	}
	want = "Revision number to load. Revisions are numbered from 0. If omitted, the latest revision is loaded."
	if got := selectors.Items.Properties["version"].Description; got != want {
		t.Errorf("artifact_versions[].version description = %q, want %q", got, want)
	}
}

func TestLoadArtifactsTool_ProcessRequest_RevisionsAndMetadata(t *testing.T) {
	t.Parallel()

	loadArtifactsTool := loadartifactstool.New()
	requestProcessor := loadArtifactsTool.(toolinternal.RequestProcessor)

	svc := artifact.InMemoryService()
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Artifacts: &artifactinternal.Artifacts{
			Service:   svc,
			AppName:   "app",
			UserID:    "user",
			SessionID: "session",
		},
	})
	tc := agent.NewToolContext(ic, "call-1", &session.EventActions{}, nil)

	if _, err := tc.Artifacts().Save(t.Context(), "design.md", genai.NewPartFromText("draft v1")); err != nil {
		t.Fatalf("Save v1: %v", err)
	}
	reqAfterOneRevision := &model.LLMRequest{}
	if err := requestProcessor.ProcessRequest(tc, reqAfterOneRevision); err != nil {
		t.Fatalf("ProcessRequest after 1 revision: %v", err)
	}
	instructionAfterOne := reqAfterOneRevision.Config.SystemInstruction.Parts[0].Text

	for _, body := range []string{"draft v2", "draft v3"} {
		if _, err := tc.Artifacts().Save(t.Context(), "design.md", genai.NewPartFromText(body)); err != nil {
			t.Fatalf("Save(%q): %v", body, err)
		}
	}

	t.Run("default instruction size is unchanged when multiple revisions exist", func(t *testing.T) {
		reqAfterThree := &model.LLMRequest{}
		if err := requestProcessor.ProcessRequest(tc, reqAfterThree); err != nil {
			t.Fatalf("ProcessRequest after 3 revisions: %v", err)
		}
		got := reqAfterThree.Config.SystemInstruction.Parts[0].Text
		if got != instructionAfterOne {
			t.Errorf("instruction after 3 revisions = %q, want unchanged %q", got, instructionAfterOne)
		}
	})

	t.Run("loads a specific revision via top-level version", func(t *testing.T) {
		llmReq := &model.LLMRequest{
			Contents: []*genai.Content{{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					genai.NewPartFromFunctionResponse("load_artifacts", map[string]any{
						"artifact_names": []any{"design.md"},
						"version":        float64(0),
					}),
				},
			}},
		}
		if err := requestProcessor.ProcessRequest(tc, llmReq); err != nil {
			t.Fatalf("ProcessRequest: %v", err)
		}
		if len(llmReq.Contents) != 2 {
			t.Fatalf("len(Contents) = %d, want 2", len(llmReq.Contents))
		}
		parts := llmReq.Contents[1].Parts
		if got, want := parts[0].Text, "Artifact design.md (version 0) is:"; got != want {
			t.Errorf("header = %q, want %q", got, want)
		}
		if got, want := parts[1].Text, "draft v1"; got != want {
			t.Errorf("body = %q, want %q", got, want)
		}
	})

	t.Run("loads multiple specific revisions via artifact_versions", func(t *testing.T) {
		llmReq := &model.LLMRequest{
			Contents: []*genai.Content{{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					genai.NewPartFromFunctionResponse("load_artifacts", map[string]any{
						"artifact_versions": []any{
							map[string]any{"name": "design.md", "version": float64(0)},
							map[string]any{"name": "design.md", "version": float64(2)},
						},
					}),
				},
			}},
		}
		if err := requestProcessor.ProcessRequest(tc, llmReq); err != nil {
			t.Fatalf("ProcessRequest: %v", err)
		}
		if len(llmReq.Contents) != 3 {
			t.Fatalf("len(Contents) = %d, want 3", len(llmReq.Contents))
		}
		if got, want := llmReq.Contents[1].Parts[1].Text, "draft v1"; got != want {
			t.Errorf("first loaded body = %q, want %q", got, want)
		}
		if got, want := llmReq.Contents[2].Parts[1].Text, "draft v3"; got != want {
			t.Errorf("second loaded body = %q, want %q", got, want)
		}
	})

	t.Run("enumerates revisions with metadata when list_versions is true", func(t *testing.T) {
		llmReq := &model.LLMRequest{
			Contents: []*genai.Content{{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					genai.NewPartFromFunctionResponse("load_artifacts", map[string]any{
						"artifact_names": []any{"design.md"},
						"list_versions":  true,
					}),
				},
			}},
		}
		if err := requestProcessor.ProcessRequest(tc, llmReq); err != nil {
			t.Fatalf("ProcessRequest: %v", err)
		}
		if len(llmReq.Contents) != 2 {
			t.Fatalf("len(Contents) = %d, want 2", len(llmReq.Contents))
		}
		parts := llmReq.Contents[1].Parts
		if got, want := parts[0].Text, "Versions of artifact design.md:"; got != want {
			t.Errorf("header = %q, want %q", got, want)
		}
		var summaries []struct {
			Version    int64  `json:"version"`
			CreateTime string `json:"create_time"`
			MimeType   string `json:"mime_type"`
		}
		if err := json.Unmarshal([]byte(parts[1].Text), &summaries); err != nil {
			t.Fatalf("json.Unmarshal(%q): %v", parts[1].Text, err)
		}
		if strings.Contains(parts[1].Text, "canonical_uri") || strings.Contains(parts[1].Text, "custom_metadata") {
			t.Errorf("version summary exposed internal URI or custom metadata: %s", parts[1].Text)
		}
		if len(summaries) != 3 {
			t.Fatalf("len(summaries) = %d, want 3", len(summaries))
		}
		for i, wantVer := range []int64{0, 1, 2} {
			if summaries[i].Version != wantVer {
				t.Errorf("summaries[%d].Version = %d, want %d", i, summaries[i].Version, wantVer)
			}
			if summaries[i].MimeType != "text/plain" {
				t.Errorf("summaries[%d].MimeType = %q, want \"text/plain\"", i, summaries[i].MimeType)
			}
			if summaries[i].CreateTime == "" {
				t.Errorf("summaries[%d] missing create_time: %+v", i, summaries[i])
			}
		}
	})

	t.Run("omits model-supplied artifact names from versioned load and list errors", func(t *testing.T) {
		for _, tcErr := range []struct {
			name     string
			ctx      agent.Context
			response map[string]any
			secret   string
		}{
			{
				name: "LoadVersion error",
				ctx:  tc,
				response: map[string]any{
					"artifact_versions": []any{
						map[string]any{"name": "patient-jane-doe-hiv-results.pdf", "version": float64(7)},
					},
				},
				secret: "patient-jane-doe-hiv-results.pdf",
			},
			{
				name: "Versions error",
				ctx:  tc,
				response: map[string]any{
					"artifact_names": []any{"never-saved-x.pdf"},
					"list_versions":  true,
				},
				secret: "never-saved-x.pdf",
			},
			{
				name: "GetArtifactVersion error",
				ctx: agent.NewToolContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
					Artifacts: &artifactinternal.Artifacts{
						Service:   failingGetVersionService{Service: svc},
						AppName:   "app",
						UserID:    "user",
						SessionID: "session",
					},
				}), "call-meta", &session.EventActions{}, nil),
				response: map[string]any{
					"artifact_names": []any{"design.md"},
					"list_versions":  true,
				},
				secret: "design.md",
			},
		} {
			t.Run(tcErr.name, func(t *testing.T) {
				llmReq := &model.LLMRequest{
					Contents: []*genai.Content{{
						Role: genai.RoleUser,
						Parts: []*genai.Part{
							genai.NewPartFromFunctionResponse("load_artifacts", tcErr.response),
						},
					}},
				}
				err := requestProcessor.ProcessRequest(tcErr.ctx, llmReq)
				if !errors.Is(err, fs.ErrNotExist) {
					t.Fatalf("ProcessRequest error = %v, want errors.Is(_, fs.ErrNotExist)", err)
				}
				if strings.Contains(err.Error(), tcErr.secret) {
					t.Errorf("ProcessRequest error %q embeds model-supplied artifact name %q", err.Error(), tcErr.secret)
				}
			})
		}
	})
}

type failingGetVersionService struct {
	artifact.Service
}

func (f failingGetVersionService) GetArtifactVersion(context.Context, *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	return nil, fs.ErrNotExist
}

func TestLoadArtifactsTool_ProcessRequest_ServiceErrorsOmitIdentifiers(t *testing.T) {
	const (
		userID    = "user-7f3a"
		sessionID = "session-91c2"
		fileName  = "patient-jane-doe.pdf"
	)
	base := artifact.InMemoryService()
	if _, err := base.Save(t.Context(), &artifact.SaveRequest{
		AppName: "app", UserID: userID, SessionID: sessionID, FileName: fileName, Part: genai.NewPartFromText("v1"),
	}); err != nil {
		t.Fatalf("Save: %v", err)
	}
	dnsErr := &net.DNSError{Err: "no such host", Name: "storage.example", IsNotFound: true}

	tests := []struct {
		name    string
		failing string
		cause   error
		// response is the load_artifacts function response in the request; nil
		// sends none, so only the artifact listing runs.
		response map[string]any
		wantErr  string
	}{
		{
			name:    "listing artifacts",
			failing: "List",
			cause:   context.Canceled,
			wantErr: "failed to list artifacts: context canceled",
		},
		{
			name:     "artifact_versions with a version",
			failing:  "Load",
			cause:    fs.ErrNotExist,
			response: map[string]any{"artifact_versions": []any{map[string]any{"name": fileName, "version": float64(7)}}},
			wantErr:  "failed to load artifact at artifact_versions[0] (version 7): file does not exist",
		},
		{
			name:     "in-process artifact_versions with a version",
			failing:  "Load",
			cause:    fs.ErrNotExist,
			response: map[string]any{"artifact_versions": []map[string]any{{"name": fileName, "version": 7}}},
			wantErr:  "failed to load artifact at artifact_versions[0] (version 7): file does not exist",
		},
		{
			name:     "artifact_names with top-level version",
			failing:  "Load",
			cause:    fs.ErrNotExist,
			response: map[string]any{"artifact_names": []any{fileName}, "version": float64(7)},
			wantErr:  "failed to load artifact at artifact_names[0] (version 7): file does not exist",
		},
		{
			name:     "artifact_versions without a version",
			failing:  "Load",
			cause:    fs.ErrNotExist,
			response: map[string]any{"artifact_versions": []any{map[string]any{"name": fileName}}},
			wantErr:  "failed to load artifact at artifact_versions[0]: file does not exist",
		},
		{
			name:     "artifact_versions with version 0",
			failing:  "Load",
			cause:    context.DeadlineExceeded,
			response: map[string]any{"artifact_versions": []any{map[string]any{"name": fileName, "version": float64(0)}}},
			wantErr:  "failed to load artifact at artifact_versions[0] (version 0): context deadline exceeded",
		},
		{
			name:     "artifact_names",
			failing:  "Load",
			cause:    fs.ErrNotExist,
			response: map[string]any{"artifact_names": []any{fileName}},
			wantErr:  "failed to load artifact at artifact_names[0]: file does not exist",
		},
		{
			name:     "artifact_names denied by the service",
			failing:  "Load",
			cause:    fs.ErrPermission,
			response: map[string]any{"artifact_names": []any{fileName}},
			wantErr:  "failed to load artifact at artifact_names[0]: permission denied",
		},
		{
			name:     "list_versions listing revisions",
			failing:  "Versions",
			cause:    dnsErr,
			response: map[string]any{"artifact_names": []any{fileName}, "list_versions": true},
			wantErr:  "failed to list versions of artifact at artifact_names[0]: artifact service error of type *net.DNSError",
		},
		{
			name:     "list_versions reading revision metadata",
			failing:  "GetArtifactVersion",
			cause:    fs.ErrNotExist,
			response: map[string]any{"artifact_names": []any{fileName}, "list_versions": true},
			wantErr:  "failed to get version 0 metadata for artifact at artifact_names[0]: file does not exist",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Artifacts: &artifactinternal.Artifacts{
					Service:   identifyingErrorService{Service: base, failing: tt.failing, cause: tt.cause},
					AppName:   "app",
					UserID:    userID,
					SessionID: sessionID,
				},
			})
			tc := agent.NewToolContext(ic, "call-1", &session.EventActions{}, nil)
			req := &model.LLMRequest{}
			if tt.response != nil {
				req.Contents = []*genai.Content{{
					Role:  genai.RoleUser,
					Parts: []*genai.Part{genai.NewPartFromFunctionResponse("load_artifacts", tt.response)},
				}}
			}

			err := loadartifactstool.New().(toolinternal.RequestProcessor).ProcessRequest(tc, req)
			if err == nil {
				t.Fatal("ProcessRequest succeeded, want an error")
			}
			for _, id := range []string{fileName, userID, sessionID, "storage.example"} {
				if strings.Contains(err.Error(), id) {
					t.Errorf("ProcessRequest error carries %q", id)
				}
			}
			if got := err.Error(); got != tt.wantErr {
				t.Errorf("ProcessRequest error = %q, want %q", got, tt.wantErr)
			}
			if !errors.Is(err, tt.cause) {
				t.Errorf("errors.Is(err, %v) = false, want true", tt.cause)
			}
			if _, ok := errors.AsType[*url.Error](err); !ok {
				t.Error("errors.AsType[*url.Error](err) = false, want true")
			}
		})
	}
}

func TestLoadArtifactsTool_ProcessRequest_ListVersionsUnsupported(t *testing.T) {
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Artifacts: unversionedArtifacts{&artifactinternal.Artifacts{
			Service:   artifact.InMemoryService(),
			AppName:   "app",
			UserID:    "user",
			SessionID: "session",
		}},
	})
	tc := agent.NewToolContext(ic, "call-1", &session.EventActions{}, nil)
	req := &model.LLMRequest{Contents: []*genai.Content{{
		Role: genai.RoleUser,
		Parts: []*genai.Part{genai.NewPartFromFunctionResponse("load_artifacts", map[string]any{
			"artifact_names": []any{"design.md"},
			"list_versions":  true,
		})},
	}}}

	err := loadartifactstool.New().(toolinternal.RequestProcessor).ProcessRequest(tc, req)
	if got, want := fmt.Sprint(err), "failed to list versions of artifact at artifact_names[0]: unsupported operation"; got != want {
		t.Errorf("ProcessRequest error = %q, want %q", got, want)
	}
	if !errors.Is(err, errors.ErrUnsupported) {
		t.Errorf("errors.Is(err, errors.ErrUnsupported) = false, want true, got: %v", err)
	}
}

func TestLoadArtifactsTool_ProcessRequest_ErrorsIdentifyFailingItem(t *testing.T) {
	tests := []struct {
		name     string
		response map[string]any
		wantErr  string
	}{
		{
			name: "second of two selectors with the same version",
			response: map[string]any{"artifact_versions": []any{
				map[string]any{"name": "doc1.txt", "version": float64(0)},
				map[string]any{"name": "missing.txt", "version": float64(0)},
			}},
			wantErr: "failed to load artifact at artifact_versions[1] (version 0): file does not exist",
		},
		{
			name: "selector after artifact_names",
			response: map[string]any{
				"artifact_names":    []any{"doc1.txt"},
				"artifact_versions": []any{map[string]any{"name": "missing.txt"}},
			},
			wantErr: "failed to load artifact at artifact_versions[0]: file does not exist",
		},
		{
			name:     "list_versions over several names",
			response: map[string]any{"artifact_names": []any{"doc1.txt", "missing.txt"}, "list_versions": true},
			wantErr:  "failed to list versions of artifact at artifact_names[1]: file does not exist",
		},
		{
			name:     "name given only in list_versions",
			response: map[string]any{"artifact_names": []any{"doc1.txt"}, "list_versions": []any{"doc1.txt", "missing.txt"}},
			wantErr:  "failed to list versions of artifact at list_versions[1]: file does not exist",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tc := createToolContext(t)
			if _, err := tc.Artifacts().Save(t.Context(), "doc1.txt", genai.NewPartFromText("v1")); err != nil {
				t.Fatalf("Save: %v", err)
			}
			req := &model.LLMRequest{Contents: []*genai.Content{{
				Role:  genai.RoleUser,
				Parts: []*genai.Part{genai.NewPartFromFunctionResponse("load_artifacts", tt.response)},
			}}}

			err := loadartifactstool.New().(toolinternal.RequestProcessor).ProcessRequest(tc, req)
			if got := fmt.Sprint(err); got != tt.wantErr {
				t.Errorf("ProcessRequest error = %q, want %q", got, tt.wantErr)
			}
			if !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("errors.Is(err, fs.ErrNotExist) = false, want true")
			}
		})
	}
}

func TestLoadArtifactsTool_ProcessRequest_ParallelErrorNamesCall(t *testing.T) {
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Artifacts: unversionedArtifacts{&artifactinternal.Artifacts{
			Service:   artifact.InMemoryService(),
			AppName:   "app",
			UserID:    "user",
			SessionID: "session",
		}},
	})
	tc := agent.NewToolContext(ic, "call-1", &session.EventActions{}, nil)
	for _, name := range []string{"doc1.txt", "doc2.txt"} {
		if _, err := tc.Artifacts().Save(t.Context(), name, genai.NewPartFromText("body")); err != nil {
			t.Fatalf("Save(%q): %v", name, err)
		}
	}
	req := &model.LLMRequest{Contents: []*genai.Content{{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			genai.NewPartFromFunctionResponse("load_artifacts", map[string]any{"artifact_names": []any{"doc1.txt"}}),
			genai.NewPartFromFunctionResponse("load_artifacts", map[string]any{"list_versions": []any{"doc2.txt"}}),
		},
	}}}

	err := loadartifactstool.New().(toolinternal.RequestProcessor).ProcessRequest(tc, req)
	if got, want := fmt.Sprint(err), "failed to list versions of artifact at load_artifacts[1].list_versions[0]: unsupported operation"; got != want {
		t.Errorf("ProcessRequest error = %q, want %q", got, want)
	}
}

func TestLoadArtifactsTool_ProcessRequest_InvalidNameKeepsReason(t *testing.T) {
	tests := []struct {
		name     string
		response map[string]any
		wantErr  string
	}{
		{
			name:     "artifact_names with a slash",
			response: map[string]any{"artifact_names": []any{"reports/q3.pdf"}},
			wantErr:  "failed to load artifact at artifact_names[0]: artifact service error of type *errors.errorString; the name contains a path separator, which the in-memory and GCS artifact services reject",
		},
		{
			name:     "artifact_versions with a backslash",
			response: map[string]any{"artifact_versions": []any{map[string]any{"name": `reports\q3.pdf`, "version": float64(2)}}},
			wantErr:  "failed to load artifact at artifact_versions[0] (version 2): artifact service error of type *errors.errorString; the name contains a path separator, which the in-memory and GCS artifact services reject",
		},
		{
			name:     "artifact_names with an empty name",
			response: map[string]any{"artifact_names": []any{""}},
			wantErr:  "failed to load artifact at artifact_names[0]: artifact service error of type *errors.errorString; the name is empty",
		},
		{
			name:     "list_versions with a slash",
			response: map[string]any{"artifact_names": []any{"reports/q3.pdf"}, "list_versions": true},
			wantErr:  "failed to list versions of artifact at artifact_names[0]: artifact service error of type *errors.errorString; the name contains a path separator, which the in-memory and GCS artifact services reject",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tc := createToolContext(t)
			req := &model.LLMRequest{Contents: []*genai.Content{{
				Role:  genai.RoleUser,
				Parts: []*genai.Part{genai.NewPartFromFunctionResponse("load_artifacts", tt.response)},
			}}}

			err := loadartifactstool.New().(toolinternal.RequestProcessor).ProcessRequest(tc, req)
			if got := fmt.Sprint(err); got != tt.wantErr {
				t.Errorf("ProcessRequest error = %q, want %q", got, tt.wantErr)
			}
		})
	}
}

func TestLoadArtifactsTool_ProcessRequest_ServiceAcceptsNestedName(t *testing.T) {
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Artifacts: &artifactinternal.Artifacts{
			Service:   nestedNameService{},
			AppName:   "app",
			UserID:    "user",
			SessionID: "session",
		},
	})
	tc := agent.NewToolContext(ic, "call-1", &session.EventActions{}, nil)
	req := &model.LLMRequest{Contents: []*genai.Content{{
		Role:  genai.RoleUser,
		Parts: []*genai.Part{genai.NewPartFromFunctionResponse("load_artifacts", map[string]any{"artifact_names": []any{"reports/q3.pdf"}})},
	}}}

	if err := loadartifactstool.New().(toolinternal.RequestProcessor).ProcessRequest(tc, req); err != nil {
		t.Fatalf("ProcessRequest: %v", err)
	}
	if got := len(req.Contents); got != 2 {
		t.Fatalf("len(req.Contents) = %d, want 2", got)
	}
	if got, want := req.Contents[1].Parts[1].Text, "q3 body"; got != want {
		t.Errorf("loaded artifact text = %q, want %q", got, want)
	}

	missing := &model.LLMRequest{Contents: []*genai.Content{{
		Role:  genai.RoleUser,
		Parts: []*genai.Part{genai.NewPartFromFunctionResponse("load_artifacts", map[string]any{"artifact_names": []any{"reports/q4.pdf"}})},
	}}}
	err := loadartifactstool.New().(toolinternal.RequestProcessor).ProcessRequest(tc, missing)
	if got, want := fmt.Sprint(err), "failed to load artifact at artifact_names[0]: file does not exist"; got != want {
		t.Errorf("ProcessRequest error for a missing nested name = %q, want %q", got, want)
	}

	unreachable := &model.LLMRequest{Contents: []*genai.Content{{
		Role:  genai.RoleUser,
		Parts: []*genai.Part{genai.NewPartFromFunctionResponse("load_artifacts", map[string]any{"artifact_names": []any{"reports/offline.pdf"}})},
	}}}
	err = loadartifactstool.New().(toolinternal.RequestProcessor).ProcessRequest(tc, unreachable)
	if got, want := fmt.Sprint(err), "failed to load artifact at artifact_names[0]: artifact service error of type *net.DNSError; the name contains a path separator, which the in-memory and GCS artifact services reject"; got != want {
		t.Errorf("ProcessRequest error for a storage failure on a nested name = %q, want %q", got, want)
	}
}

// nestedNameService is an artifact service that, unlike the in-tree ones,
// accepts file names containing a path separator. It lists "reports/q3.pdf",
// answers a Load of it with the text "q3 body", fails "reports/offline.pdf"
// with a DNS error and reports every other name as not found.
type nestedNameService struct {
	artifact.Service
}

func (nestedNameService) List(context.Context, *artifact.ListRequest) (*artifact.ListResponse, error) {
	return &artifact.ListResponse{FileNames: []string{"reports/q3.pdf"}}, nil
}

func (nestedNameService) Load(_ context.Context, req *artifact.LoadRequest) (*artifact.LoadResponse, error) {
	switch req.FileName {
	case "reports/q3.pdf":
		return &artifact.LoadResponse{Part: genai.NewPartFromText("q3 body")}, nil
	case "reports/offline.pdf":
		return nil, fmt.Errorf("get: %w", &net.DNSError{Err: "no such host", Name: "storage.example"})
	}
	return nil, fmt.Errorf("artifact %q: %w", req.FileName, fs.ErrNotExist)
}

// identifyingErrorService fails the operation named by failing with an error
// that, like the GCS service's, names the object path and a request URL built
// from the request's user, session and file name, and wraps cause. Every other
// operation goes to the embedded service.
type identifyingErrorService struct {
	artifact.Service
	failing string
	cause   error
}

func (s identifyingErrorService) err(userID, sessionID, fileName string) error {
	object := "app/" + userID + "/" + sessionID + "/" + fileName
	return fmt.Errorf("artifact '%s': %w", object, &url.Error{
		Op:  "Get",
		URL: "https://storage.example/b/bucket/o?prefix=" + url.QueryEscape(object),
		Err: s.cause,
	})
}

func (s identifyingErrorService) List(ctx context.Context, req *artifact.ListRequest) (*artifact.ListResponse, error) {
	if s.failing == "List" {
		return nil, s.err(req.UserID, req.SessionID, "")
	}
	return s.Service.List(ctx, req)
}

func (s identifyingErrorService) Load(ctx context.Context, req *artifact.LoadRequest) (*artifact.LoadResponse, error) {
	if s.failing == "Load" {
		return nil, s.err(req.UserID, req.SessionID, req.FileName)
	}
	return s.Service.Load(ctx, req)
}

func (s identifyingErrorService) Versions(ctx context.Context, req *artifact.VersionsRequest) (*artifact.VersionsResponse, error) {
	if s.failing == "Versions" {
		return nil, s.err(req.UserID, req.SessionID, req.FileName)
	}
	return s.Service.Versions(ctx, req)
}

func (s identifyingErrorService) GetArtifactVersion(ctx context.Context, req *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	if s.failing == "GetArtifactVersion" {
		return nil, s.err(req.UserID, req.SessionID, req.FileName)
	}
	return s.Service.GetArtifactVersion(ctx, req)
}

// unversionedArtifacts exposes only the agent.Artifacts methods, so neither
// version listing nor a backing service is reachable through it.
type unversionedArtifacts struct {
	agent.Artifacts
}
