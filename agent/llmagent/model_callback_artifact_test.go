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

package llmagent_test

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// TestModelCallbackArtifactSaveRecordedOnEvent checks that artifacts saved from
// before and after model callbacks show up in the model response event's
// ArtifactDelta, the same way state set in those callbacks shows up in
// StateDelta.
func TestModelCallbackArtifactSaveRecordedOnEvent(t *testing.T) {
	ctx := t.Context()
	sessionService := session.InMemoryService()

	saveArtifact := func(cctx agent.Context, name string) error {
		_, err := cctx.Artifacts().Save(cctx, name, genai.NewPartFromText(name))
		return err
	}
	a, err := llmagent.New(llmagent.Config{
		Name:  "root_agent",
		Model: &FakeLLM{},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(cctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
				return nil, saveArtifact(cctx, "before.txt")
			},
		},
		AfterModelCallbacks: []llmagent.AfterModelCallback{
			func(cctx agent.Context, _ *model.LLMResponse, _ error) (*model.LLMResponse, error) {
				return nil, saveArtifact(cctx, "after.txt")
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:         "test_app",
		Agent:           a,
		SessionService:  sessionService,
		ArtifactService: artifact.InMemoryService(),
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	createResp, err := sessionService.Create(ctx, &session.CreateRequest{AppName: "test_app", UserID: "u"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	var modelEvent *session.Event
	for ev, err := range r.Run(ctx, "u", createResp.Session.ID(), genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev.Author == "root_agent" && ev.Content != nil {
			modelEvent = ev
		}
	}
	if modelEvent == nil {
		t.Fatal("no model response event")
	}
	for _, name := range []string{"before.txt", "after.txt"} {
		if _, ok := modelEvent.Actions.ArtifactDelta[name]; !ok {
			t.Errorf("model response event ArtifactDelta = %v, missing %q", modelEvent.Actions.ArtifactDelta, name)
		}
	}
}
