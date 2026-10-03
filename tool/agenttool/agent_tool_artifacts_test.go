// Copyright 2026 Google LLC
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

package agenttool_test

import (
	"context"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/agenttool"
)

func artifactParent(t *testing.T, name string, child agent.Agent) agent.Agent {
	t.Helper()
	wrapped := agenttool.New(child, nil).(toolinternal.FunctionTool)
	parent, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				event := session.NewEvent(ctx, ctx.InvocationID())
				toolCtx := agent.NewToolContext(ctx, "child_call", &event.Actions, nil)
				_, err := wrapped.Run(toolCtx, map[string]any{"request": "run"})
				if err != nil {
					yield(nil, err)
					return
				}
				event.Content = genai.NewContentFromText("done", "model")
				yield(event, nil)
			}
		},
	})
	if err != nil {
		t.Fatal("parent construction failed")
	}
	return parent
}

func TestAgentTool_Run_Artifacts(t *testing.T) {
	for _, nested := range []bool{false, true} {
		name := "direct"
		if nested {
			name = "nested"
		}
		t.Run(name, func(t *testing.T) { testAgentToolArtifacts(t, nested) })
	}
}

func testAgentToolArtifacts(t *testing.T, nested bool) {
	t.Helper()
	store := artifact.InMemoryService()
	_, err := store.Save(t.Context(), &artifact.SaveRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "input.txt",
		Part: genai.NewPartFromText("input"),
	})
	if err != nil {
		t.Fatal("parent fixture save failed")
	}

	var childLoaded, childListed, childSaved bool
	child, err := agent.New(agent.Config{
		Name: "child",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				input, loadErr := ctx.Artifacts().Load(ctx, "input.txt")
				childLoaded = loadErr == nil && input.Part.Text == "input"
				files, listErr := ctx.Artifacts().List(ctx)
				childListed = listErr == nil && len(files.FileNames) == 1
				event := session.NewEvent(ctx, ctx.InvocationID())
				cb := agent.NewCallbackContextWithArtifactTracking(ctx, &event.Actions)
				_, saveErr := cb.Artifacts().Save(ctx, "output.txt", genai.NewPartFromText("output"))
				childSaved = saveErr == nil
				if saveErr != nil {
					yield(nil, saveErr)
					return
				}
				if _, userSaveErr := cb.Artifacts().Save(ctx, "user:prefs.txt", genai.NewPartFromText("user-scoped")); userSaveErr != nil {
					yield(nil, userSaveErr)
					return
				}
				event.Content = genai.NewContentFromText("done", "model")
				yield(event, nil)
			}
		},
	})
	if err != nil {
		t.Fatal("child construction failed")
	}
	if nested {
		child = artifactParent(t, "middle", child)
	}
	r, err := runner.New(runner.Config{
		AppName: "app", Agent: artifactParent(t, "parent", child), SessionService: session.InMemoryService(),
		ArtifactService: store, AutoCreateSession: true,
	})
	if err != nil {
		t.Fatal("runner construction failed")
	}
	var parentDelta bool
	for event, err := range r.Run(t.Context(), "user", "session", genai.NewContentFromText("run", "user"), agent.RunConfig{}) {
		if err != nil {
			t.Fatal("parent run failed")
		}
		if event != nil && event.Actions.ArtifactDelta["output.txt"] > 0 {
			parentDelta = true
		}
	}
	if !childSaved {
		t.Fatal("child save was not exercised")
	}
	if !childLoaded {
		t.Error("child cannot load parent's input artifact")
	}
	if !childListed {
		t.Error("child cannot list parent's input artifact")
	}
	output, loadErr := store.Load(t.Context(), &artifact.LoadRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "output.txt",
	})
	if loadErr != nil || output.Part.Text != "output" {
		t.Error("parent cannot load child's saved artifact")
	}
	userOutput, userLoadErr := store.Load(t.Context(), &artifact.LoadRequest{
		AppName: "app", UserID: "user", SessionID: "other_session", FileName: "user:prefs.txt",
	})
	if userLoadErr != nil || userOutput.Part.Text != "user-scoped" {
		t.Error("parent cannot load child's user-scoped artifact across sessions")
	}
	if !parentDelta {
		t.Error("parent event lacks child's artifact delta")
	}
}

type staticJSONModel struct {
	text string
}

func (m staticJSONModel) Name() string { return "static-json" }

func (m staticJSONModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content: genai.NewContentFromText(m.text, genai.RoleModel),
		}, nil)
	}
}

func TestAgentTool_Run_OutputArtifactWithOutputSchema(t *testing.T) {
	t.Parallel()

	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"title": {Type: genai.TypeString},
		},
		Required: []string{"title"},
	}
	child, err := llmagent.New(llmagent.Config{
		Name:           "spec_writer",
		Model:          staticJSONModel{text: `{"title":"v1"}`},
		OutputSchema:   schema,
		OutputArtifact: "spec.json",
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	store := artifact.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             artifactParent(t, "parent", child),
		SessionService:    session.InMemoryService(),
		ArtifactService:   store,
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}

	var deltaVer int64
	for ev, err := range r.Run(t.Context(), "user", "session", genai.NewContentFromText("run", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run: %v", err)
		}
		if ev != nil && ev.Actions.ArtifactDelta["spec.json"] > 0 {
			deltaVer = ev.Actions.ArtifactDelta["spec.json"]
		}
	}
	if deltaVer != 1 {
		t.Errorf("parent ArtifactDelta[\"spec.json\"] = %d, want 1", deltaVer)
	}
	loaded, err := store.Load(t.Context(), &artifact.LoadRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "spec.json", Version: 1,
	})
	if err != nil || loaded.Part.Text != `{"title":"v1"}` {
		t.Errorf("loaded spec.json = (%v, %v), want {\"title\":\"v1\"}", loaded, err)
	}
}
