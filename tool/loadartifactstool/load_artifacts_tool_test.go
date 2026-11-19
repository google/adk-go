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
	"errors"
	"io/fs"
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
		name string
		args map[string]any
		want map[string]any
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := toolImpl.Run(tc, tt.args)
			if err != nil {
				t.Fatalf("Run() error = %v, want nil", err)
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

// TestLoadArtifactsTool_ParallelCallsAttachEveryBody reproduces
// https://github.com/google/adk-go/issues/2415: the model answers with two
// load_artifacts calls in one turn, so the next request must carry both bodies.
func TestLoadArtifactsTool_ParallelCallsAttachEveryBody(t *testing.T) {
	const app, user, sid = "app", "client", "session"
	ctx := t.Context()
	artifacts := artifact.InMemoryService()

	calls := &genai.Content{Role: genai.RoleModel}
	for _, file := range []struct{ name, text string }{
		{"january.txt", "january body"},
		{"february.txt", "february body"},
	} {
		if _, err := artifacts.Save(ctx, &artifact.SaveRequest{
			AppName: app, UserID: user, SessionID: sid,
			FileName: file.name, Part: genai.NewPartFromText(file.text),
		}); err != nil {
			t.Fatalf("Save(%s) failed: %v", file.name, err)
		}
		part := genai.NewPartFromFunctionCall("load_artifacts", map[string]any{
			"artifact_names": []any{file.name},
		})
		part.FunctionCall.ID = file.name
		calls.Parts = append(calls.Parts, part)
	}

	fake := &testutil.MockModel{Responses: []*genai.Content{
		calls,
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{
		Name: "reader", Model: fake, Tools: []tool.Tool{loadartifactstool.New()},
	})
	if err != nil {
		t.Fatalf("llmagent.New failed: %v", err)
	}
	sessions := session.InMemoryService()
	if _, err := sessions.Create(ctx, &session.CreateRequest{
		AppName: app, UserID: user, SessionID: sid,
	}); err != nil {
		t.Fatalf("session Create failed: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName: app, Agent: a, SessionService: sessions, ArtifactService: artifacts,
	})
	if err != nil {
		t.Fatalf("runner.New failed: %v", err)
	}

	prompt := genai.NewContentFromText("Compare January and February", genai.RoleUser)
	for _, err := range r.Run(ctx, user, sid, prompt, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run failed: %v", err)
		}
	}
	if len(fake.Requests) != 2 {
		t.Fatalf("model requests = %d, want 2", len(fake.Requests))
	}

	var responses, bodies int
	for _, content := range fake.Requests[1].Contents {
		for _, part := range content.Parts {
			if part.FunctionResponse != nil && part.FunctionResponse.Name == "load_artifacts" {
				responses++
			}
			switch part.Text {
			case "january body", "february body":
				bodies++
			}
		}
	}
	if responses != 2 || bodies != 2 {
		t.Errorf("tool responses = %d, artifact bodies = %d; want 2 of each", responses, bodies)
	}
}

// TestLoadArtifactsTool_ParallelCallsSkipMissingFile covers the follow-up on
// PR #2416: with two calls in one turn, one artifact that cannot be loaded must
// not keep the other body from the model, in either order.
func TestLoadArtifactsTool_ParallelCallsSkipMissingFile(t *testing.T) {
	for _, order := range []struct {
		name   string
		first  string
		second string
		body   string
	}{
		{name: "missing file follows a saved one", first: "january.txt", second: "missing.txt", body: "january body"},
		{name: "missing file precedes a saved one", first: "missing.txt", second: "february.txt", body: "february body"},
	} {
		t.Run(order.name, func(t *testing.T) {
			const app, user, sid = "app", "client", "session"
			ctx := t.Context()
			artifacts := artifact.InMemoryService()
			if _, err := artifacts.Save(ctx, &artifact.SaveRequest{
				AppName: app, UserID: user, SessionID: sid,
				FileName: "january.txt", Part: genai.NewPartFromText("january body"),
			}); err != nil {
				t.Fatalf("Save(january.txt) failed: %v", err)
			}
			if _, err := artifacts.Save(ctx, &artifact.SaveRequest{
				AppName: app, UserID: user, SessionID: sid,
				FileName: "february.txt", Part: genai.NewPartFromText("february body"),
			}); err != nil {
				t.Fatalf("Save(february.txt) failed: %v", err)
			}

			calls := &genai.Content{Role: genai.RoleModel}
			for _, name := range []string{order.first, order.second} {
				part := genai.NewPartFromFunctionCall("load_artifacts", map[string]any{
					"artifact_names": []any{name},
				})
				part.FunctionCall.ID = name
				calls.Parts = append(calls.Parts, part)
			}

			fake := &testutil.MockModel{Responses: []*genai.Content{
				calls,
				genai.NewContentFromText("done", genai.RoleModel),
			}}
			a, err := llmagent.New(llmagent.Config{
				Name: "reader", Model: fake, Tools: []tool.Tool{loadartifactstool.New()},
			})
			if err != nil {
				t.Fatalf("llmagent.New failed: %v", err)
			}
			sessions := session.InMemoryService()
			if _, err := sessions.Create(ctx, &session.CreateRequest{
				AppName: app, UserID: user, SessionID: sid,
			}); err != nil {
				t.Fatalf("session Create failed: %v", err)
			}
			r, err := runner.New(runner.Config{
				AppName: app, Agent: a, SessionService: sessions, ArtifactService: artifacts,
			})
			if err != nil {
				t.Fatalf("runner.New failed: %v", err)
			}

			prompt := genai.NewContentFromText("Read the file", genai.RoleUser)
			for _, err := range r.Run(ctx, user, sid, prompt, agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("Run failed: %v", err)
				}
			}
			if len(fake.Requests) != 2 {
				t.Fatalf("model requests = %d, want 2", len(fake.Requests))
			}
			bodies := 0
			for _, content := range fake.Requests[1].Contents {
				for _, part := range content.Parts {
					if part.Text == order.body {
						bodies++
					}
				}
			}
			if bodies != 1 {
				t.Errorf("artifact bodies reaching the model = %d, want 1", bodies)
			}
		})
	}
}
