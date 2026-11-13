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
	"errors"
	"io/fs"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/toolinternal"
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
	if !parentDelta {
		t.Error("parent event lacks child's artifact delta")
	}
}

// artifactVersionResult is what one ArtifactVersion call inside a sub-agent
// returned.
type artifactVersionResult struct {
	got *artifact.ArtifactVersion
	err error
}

// It fails if a sub-agent's version lookup does not reach the outer runner's
// artifact service with the outer session's scope, if a version of 0 is not
// reported as not found, or if a sub-agent whose outer runner has no artifact
// service gets anything but "artifact service is not initialized", for version
// 0 as well (the check of the forwarding chain precedes the version check).
func TestAgentTool_Run_ArtifactVersionAndNoService(t *testing.T) {
	for _, nested := range []bool{false, true} {
		name := "direct"
		if nested {
			name = "nested"
		}
		t.Run("with service/"+name, func(t *testing.T) {
			store := artifact.InMemoryService()
			for _, text := range []string{"first", "second"} {
				_, err := store.Save(t.Context(), &artifact.SaveRequest{
					AppName: "app", UserID: "user", SessionID: "session", FileName: "input.txt",
					Part: genai.NewPartFromText(text),
				})
				if err != nil {
					t.Fatal("parent fixture save failed")
				}
			}

			results := runArtifactVersionLookups(t, store, nested)

			if got, err := results["latest"].got, results["latest"].err; err != nil || got == nil || got.Version != 2 ||
				got.CanonicalURI != "memory://apps/app/users/user/sessions/session/artifacts/input.txt/versions/2" {
				t.Errorf("ArtifactVersion(latest) = %+v, %v; want version 2 of the outer session's input.txt", got, err)
			}
			if got, err := results["1"].got, results["1"].err; err != nil || got == nil || got.Version != 1 ||
				got.CanonicalURI != "memory://apps/app/users/user/sessions/session/artifacts/input.txt/versions/1" {
				t.Errorf("ArtifactVersion(1) = %+v, %v; want version 1 of the outer session's input.txt", got, err)
			}
			if got, err := results["0"].got, results["0"].err; got != nil || !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("ArtifactVersion(0) = %+v, %v; want nil and an error wrapping fs.ErrNotExist", got, err)
			}
		})
		t.Run("no service/"+name, func(t *testing.T) {
			const want = "artifact service is not initialized"

			results := runArtifactVersionLookups(t, nil, nested)

			for _, version := range []string{"latest", "1", "0"} {
				got, err := results[version].got, results[version].err
				if got != nil || err == nil || err.Error() != want || errors.Is(err, errors.ErrUnsupported) {
					t.Errorf("ArtifactVersion(%s) = %+v, %v; want nil, %q", version, got, err, want)
				}
			}
		})
	}
}

// runArtifactVersionLookups runs a sub-agent, behind one AgentTool or two when
// nested, under a runner with store as its artifact service, and returns what
// agent.ArtifactVersion of "input.txt" returned in the sub-agent's tool context
// for the latest version and for versions 1 and 0.
func runArtifactVersionLookups(t *testing.T, store artifact.Service, nested bool) map[string]artifactVersionResult {
	t.Helper()
	v0, v1 := int64(0), int64(1)
	results := map[string]artifactVersionResult{}
	child, err := agent.New(agent.Config{
		Name: "child",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				event := session.NewEvent(ctx, ctx.InvocationID())
				toolCtx := agent.NewToolContext(ctx, "lookup_call", &event.Actions, nil)
				for name, version := range map[string]*int64{"latest": nil, "1": &v1, "0": &v0} {
					got, err := agent.ArtifactVersion(toolCtx, "input.txt", version)
					results[name] = artifactVersionResult{got: got, err: err}
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
	for _, err := range r.Run(t.Context(), "user", "session", genai.NewContentFromText("run", "user"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("parent run failed: %v", err)
		}
	}
	if len(results) != 3 {
		t.Fatalf("sub-agent made %d lookups, want 3", len(results))
	}
	return results
}
