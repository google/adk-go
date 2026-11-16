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

package artifact_test

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/artifact"
	"google.golang.org/adk/internal/artifact/tests"
)

func TestInMemoryArtifactService(t *testing.T) {
	factory := func(t *testing.T) (artifact.Service, error) {
		return artifact.InMemoryService(), nil
	}
	tests.TestArtifactService(t, "InMemory", factory)
}

// adk-python's InMemoryArtifactService deep-copies the part on save and on
// load (in_memory_artifact_service.py: `model_copy(deep=True)`), so a caller
// that edits a part it saved or loaded cannot rewrite the stored version.
func TestInMemoryService_DoesNotAliasStoredParts(t *testing.T) {
	ctx := t.Context()
	s := artifact.InMemoryService()
	req := func(p *genai.Part) *artifact.SaveRequest {
		return &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f.txt", Part: p}
	}
	part := genai.NewPartFromText("v1")
	if _, err := s.Save(ctx, req(part)); err != nil {
		t.Fatal(err)
	}
	part.Text = "edited after save"

	load := func() *genai.Part {
		t.Helper()
		resp, err := s.Load(ctx, &artifact.LoadRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f.txt"})
		if err != nil {
			t.Fatal(err)
		}
		return resp.Part
	}
	got := load()
	if got.Text != "v1" {
		t.Errorf("Load() after editing the saved part = %q, want %q", got.Text, "v1")
	}
	got.Text = "edited after load"
	if again := load(); again.Text != "v1" {
		t.Errorf("Load() after editing a loaded part = %q, want %q", again.Text, "v1")
	}
}

func fullPart() *genai.Part {
	fps := 24.0
	return &genai.Part{
		Text:                "x",
		InlineData:          &genai.Blob{Data: []byte("abc"), MIMEType: "text/plain"},
		FileData:            &genai.FileData{FileURI: "gs://b/o"},
		ThoughtSignature:    []byte("sig"),
		VideoMetadata:       &genai.VideoMetadata{FPS: &fps, StartOffset: time.Second},
		PartMetadata:        map[string]any{"src": map[string]any{"name": "a"}, "list": []any{map[string]any{"n": "1"}}},
		MediaResolution:     &genai.PartMediaResolution{Level: genai.PartMediaResolutionLevelMediaResolutionLow},
		FunctionCall:        &genai.FunctionCall{Name: "fn", Args: map[string]any{"a": map[string]any{"b": "c"}}},
		FunctionResponse:    &genai.FunctionResponse{Name: "fn", Response: map[string]any{"r": []any{"s"}}},
		ExecutableCode:      &genai.ExecutableCode{Code: "print(1)"},
		CodeExecutionResult: &genai.CodeExecutionResult{Output: "1"},
		ToolCall:            &genai.ToolCall{ID: "t", Args: map[string]any{"q": []any{"a"}}},
		ToolResponse:        &genai.ToolResponse{ID: "t", Response: map[string]any{"r": map[string]any{"k": "v"}}},
	}
}

// mutateFullPart edits every member of a part built by fullPart in place.
func mutateFullPart(p *genai.Part) {
	p.InlineData.Data[0] = 'X'
	p.InlineData.MIMEType = "edited"
	p.FileData.FileURI = "edited"
	p.ThoughtSignature[0] = 'X'
	*p.VideoMetadata.FPS = 60
	p.VideoMetadata.StartOffset = time.Hour
	p.PartMetadata["src"].(map[string]any)["name"] = "b"
	p.PartMetadata["list"].([]any)[0].(map[string]any)["n"] = "2"
	p.MediaResolution.Level = genai.PartMediaResolutionLevelMediaResolutionHigh
	p.FunctionCall.Args["a"].(map[string]any)["b"] = "edited"
	p.FunctionResponse.Response["r"].([]any)[0] = "edited"
	p.ExecutableCode.Code = "edited"
	p.CodeExecutionResult.Output = "edited"
	p.ToolCall.Args["q"].([]any)[0] = "edited"
	p.ToolResponse.Response["r"].(map[string]any)["k"] = "edited"
}

func TestInMemoryService_DoesNotAliasStoredPartFields(t *testing.T) {
	ctx := t.Context()
	s := artifact.InMemoryService()
	load := func(version int64) *genai.Part {
		t.Helper()
		resp, err := s.Load(ctx, &artifact.LoadRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f.bin", Version: version})
		if err != nil {
			t.Fatal(err)
		}
		return resp.Part
	}
	save := func(p *genai.Part) {
		t.Helper()
		if _, err := s.Save(ctx, &artifact.SaveRequest{AppName: "app", UserID: "u", SessionID: "s", FileName: "f.bin", Part: p}); err != nil {
			t.Fatal(err)
		}
	}

	t.Run("edit after save", func(t *testing.T) {
		part := fullPart()
		save(part)
		mutateFullPart(part)
		if diff := cmp.Diff(fullPart(), load(0)); diff != "" {
			t.Errorf("Load() after editing the saved part mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("edit after load", func(t *testing.T) {
		mutateFullPart(load(0))
		if diff := cmp.Diff(fullPart(), load(0)); diff != "" {
			t.Errorf("Load() after editing a loaded part mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("edit after versioned load", func(t *testing.T) {
		// Two versions, so Version: 1 takes the explicit-version path in Load.
		save(genai.NewPartFromText("v2"))
		mutateFullPart(load(1))
		if diff := cmp.Diff(fullPart(), load(1)); diff != "" {
			t.Errorf("Load(Version: 1) after editing a version-1 load mismatch (-want +got):\n%s", diff)
		}
	})
}

// The case from the review of the first version of the copy, where only some
// of the pointer members of genai.Part were detached.
func TestInMemoryService_ReviewRepro(t *testing.T) {
	ctx := t.Context()
	s := artifact.InMemoryService()
	fps := 24.0
	p := &genai.Part{
		Text:            "x",
		VideoMetadata:   &genai.VideoMetadata{FPS: &fps},
		PartMetadata:    map[string]any{"src": map[string]any{"name": "a"}},
		MediaResolution: &genai.PartMediaResolution{Level: genai.PartMediaResolutionLevelMediaResolutionLow},
	}
	if _, err := s.Save(ctx, &artifact.SaveRequest{AppName: "a", UserID: "u", SessionID: "s", FileName: "f", Part: p}); err != nil {
		t.Fatal(err)
	}
	fps = 60
	p.PartMetadata["src"].(map[string]any)["name"] = "b"
	p.MediaResolution.Level = genai.PartMediaResolutionLevelMediaResolutionHigh

	resp, err := s.Load(ctx, &artifact.LoadRequest{AppName: "a", UserID: "u", SessionID: "s", FileName: "f"})
	if err != nil {
		t.Fatal(err)
	}
	if got := *resp.Part.VideoMetadata.FPS; got != 24 {
		t.Errorf("VideoMetadata.FPS = %v, want 24", got)
	}
	if got := resp.Part.PartMetadata["src"].(map[string]any)["name"]; got != "a" {
		t.Errorf("PartMetadata[src][name] = %v, want a", got)
	}
	if got := resp.Part.MediaResolution.Level; got != genai.PartMediaResolutionLevelMediaResolutionLow {
		t.Errorf("MediaResolution.Level = %v, want %v", got, genai.PartMediaResolutionLevelMediaResolutionLow)
	}
}
