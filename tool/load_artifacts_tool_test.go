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
	"context"
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/llm"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// FakeArtifacts is a test double for agent.Artifacts.
type FakeArtifacts struct {
	data    map[string]genai.Part
	ListErr error
	LoadErr error
}

func NewFakeArtifacts(initialData map[string]*genai.Part) *FakeArtifacts {
	d := make(map[string]genai.Part)
	for k, v := range initialData {
		if v != nil {
			d[k] = *v
		}
	}
	return &FakeArtifacts{data: d}
}

func (f *FakeArtifacts) Save(name string, data genai.Part) error {
	return fmt.Errorf("Save not implemented in fake")
}

func (f *FakeArtifacts) Load(name string) (genai.Part, error) {
	if f.LoadErr != nil {
		return genai.Part{}, f.LoadErr
	}
	part, ok := f.data[name]
	if !ok {
		return genai.Part{}, fmt.Errorf("artifact %q not found", name)
	}
	return part, nil
}

func (f *FakeArtifacts) LoadVersion(name string, version int) (genai.Part, error) {
	return genai.Part{}, fmt.Errorf("LoadVersion not implemented in fake")
}

func (f *FakeArtifacts) List() ([]string, error) {
	if f.ListErr != nil {
		return nil, f.ListErr
	}
	keys := make([]string, 0, len(f.data))
	for k := range f.data {
		keys = append(keys, k)
	}
	sort.Strings(keys) // Ensure deterministic order
	return keys, nil
}

var _ agent.Artifacts = (*FakeArtifacts)(nil)

// FakeAgentContext is a test double for agent.Context.
type FakeAgentContext struct {
	context.Context
	artifacts agent.Artifacts
}

func NewFakeAgentContext(ctx context.Context, artifacts agent.Artifacts) *FakeAgentContext {
	return &FakeAgentContext{
		Context:   ctx,
		artifacts: artifacts,
	}
}

func (f *FakeAgentContext) Artifacts() agent.Artifacts {
	return f.artifacts
}

// Implement other agent.Context methods as minimally as possible for the tests.
func (f *FakeAgentContext) UserContent() *genai.Content             { return nil }
func (f *FakeAgentContext) InvocationID() string                    { return "test-invocation" }
func (f *FakeAgentContext) Branch() string                          { return "main" }
func (f *FakeAgentContext) Agent() agent.Agent                      { return nil }
func (f *FakeAgentContext) Session() session.Session                { return nil }
func (f *FakeAgentContext) End()                                    {}
func (f *FakeAgentContext) Ended() bool                             { return false }
func (f *FakeAgentContext) Value(key any) any                       { return f.Context.Value(key) }
func (f *FakeAgentContext) Deadline() (deadline time.Time, ok bool) { return f.Context.Deadline() }
func (f *FakeAgentContext) Done() <-chan struct{}                   { return f.Context.Done() }
func (f *FakeAgentContext) Err() error                              { return f.Context.Err() }

var _ agent.Context = (*FakeAgentContext)(nil)

func TestLoadArtifactsTool_Run(t *testing.T) {
	ctx := context.Background()
	loadArtifactsTool := tool.NewLoadArtifactsTool()
	fakeArtifacts := NewFakeArtifacts(nil)
	agentCtx := NewFakeAgentContext(ctx, fakeArtifacts)
	tc := tool.NewContext(agentCtx, "", nil)

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
	ctx := context.Background()
	loadArtifactsTool := tool.NewLoadArtifactsTool()
	artifacts := map[string]*genai.Part{
		"file1.txt": {Text: "content1"},
		"file2.pdf": {Text: "content2"},
	}
	fakeArtifacts := NewFakeArtifacts(artifacts)
	agentCtx := NewFakeAgentContext(ctx, fakeArtifacts)
	tc := tool.NewContext(agentCtx, "", nil)

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
	ctx := context.Background()
	loadArtifactsTool := tool.NewLoadArtifactsTool()

	artifacts := map[string]*genai.Part{
		"doc1.txt": {Text: "This is the content of doc1.txt"},
	}
	fakeArtifacts := NewFakeArtifacts(artifacts)
	agentCtx := NewFakeAgentContext(ctx, fakeArtifacts)
	tc := tool.NewContext(agentCtx, "", nil)

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
	ctx := context.Background()
	loadArtifactsTool := tool.NewLoadArtifactsTool()
	artifacts := map[string]*genai.Part{
		"doc1.txt": {Text: "content1"},
	}
	fakeArtifacts := NewFakeArtifacts(artifacts)
	agentCtx := NewFakeAgentContext(ctx, fakeArtifacts)
	tc := tool.NewContext(agentCtx, "", nil)

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
