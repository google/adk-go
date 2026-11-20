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
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/session"
)

// TestNew_BothOutputSchemas fails if New accepts an agent with both
// OutputSchema and OutputJsonSchema set and silently uses one of them.
func TestNew_BothOutputSchemas(t *testing.T) {
	a, err := llmagent.New(llmagent.Config{
		Name:             "a",
		OutputSchema:     &genai.Schema{Type: genai.TypeObject},
		OutputJsonSchema: &jsonschema.Schema{Type: "object"},
	})
	if err == nil {
		t.Fatal("New() error = nil, want an error naming OutputSchema and OutputJsonSchema")
	}
	if a != nil {
		t.Errorf("New() agent = %v, want nil", a)
	}
	for _, field := range []string{"OutputSchema", "OutputJsonSchema"} {
		if !strings.Contains(err.Error(), field) {
			t.Errorf("New() error = %q, want it to contain %q", err, field)
		}
	}
}

// TestOutputJSONSchema_EmptyStreamedFinalChunkIsNotSaved fails if the guard
// that skips an all-whitespace final text reads only OutputSchema, so an agent
// with OutputJsonSchema saves " " under its OutputKey.
func TestOutputJSONSchema_EmptyStreamedFinalChunkIsNotSaved(t *testing.T) {
	m := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText(" ", genai.RoleModel)}}
	a, err := llmagent.New(llmagent.Config{
		Name:             "a",
		Model:            m,
		OutputKey:        "out",
		OutputJsonSchema: &jsonschema.Schema{Type: "object"},
	})
	if err != nil {
		t.Fatal(err)
	}
	r := testutil.NewTestAgentRunner(t, a)
	stream := r.RunContentWithConfig(t, "s", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{StreamingMode: agent.StreamingModeSSE})
	if _, err := testutil.CollectEvents(stream); err != nil {
		t.Fatal(err)
	}

	resp, err := r.SessionService().Get(t.Context(), &session.GetRequest{AppName: "test_app", UserID: "test_user", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}
	if v, err := resp.Session.State().Get("out"); err == nil {
		t.Errorf("session state %q = %q, want no value", "out", v)
	}
}
