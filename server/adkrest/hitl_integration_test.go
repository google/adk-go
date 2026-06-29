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

package adkrest_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/server/adkrest"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// TestRESTHITL_TwoFullCycles_SameSession drives the HITL pause/resume
// flow over the REST API the Dev UI uses, for two full cycles in one
// session. This is the closest in-process guard for "the Web UI works":
// the Angular frontend itself lives in another repo, but it relies on
// (1) the server resuming each cycle correctly and (2) each pause
// carrying a distinct interrupt ID — the frontend keys its reply box on
// that ID and will not re-prompt for one it already resolved. A reused
// ID passed the console flow but silently broke the Web UI, so the
// distinct-ID assertion below is the regression this test exists for.
func TestRESTHITL_TwoFullCycles_SameSession(t *testing.T) {
	srv := httptest.NewServer(newHITLServer(t))
	defer srv.Close()

	sid := createSession(t, srv.URL)

	id1, greet1 := restCycle(t, srv.URL, sid, "Wojtek")
	if greet1 != "Hello, Wojtek!" {
		t.Fatalf("cycle 1 greeting = %q, want %q", greet1, "Hello, Wojtek!")
	}

	id2, greet2 := restCycle(t, srv.URL, sid, "Karol")
	if greet2 != "Hello, Karol!" {
		t.Fatalf("cycle 2 greeting = %q, want %q", greet2, "Hello, Karol!")
	}

	if id1 == "" || id2 == "" {
		t.Fatalf("each pause must carry an interrupt ID; got %q and %q", id1, id2)
	}
	if id1 == id2 {
		t.Fatalf("interrupt IDs must differ across runs in one session "+
			"(the Dev UI won't re-prompt for a reused ID); both = %q", id1)
	}
}

// restCycle runs one HITL cycle over REST: a fresh turn that pauses,
// then a resume turn delivering name.
func restCycle(t *testing.T, baseURL, sid, name string) (interruptID, greeting string) {
	t.Helper()

	paused := runTurn(t, baseURL, sid, genai.NewContentFromText("hi", genai.RoleUser))
	interruptID = pauseInterruptID(paused)
	if interruptID == "" {
		t.Fatalf("fresh turn did not pause on an interrupt; events = %+v", paused)
	}

	reply := &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       interruptID,
				Name:     workflow.WorkflowInputFunctionCallName,
				Response: map[string]any{"payload": name},
			},
		}},
	}
	resumed := runTurn(t, baseURL, sid, reply)
	return interruptID, greetingOutput(resumed)
}

// restEvent is the subset of the REST Event JSON this test inspects.
type restEvent struct {
	LongRunningToolIDs []string `json:"longRunningToolIds"`
	Output             any      `json:"output"`
	RequestedInput     *struct {
		InterruptID string `json:"interruptId"`
	} `json:"requestedInput"`
}

func pauseInterruptID(events []restEvent) string {
	for _, ev := range events {
		if ev.RequestedInput != nil && ev.RequestedInput.InterruptID != "" {
			return ev.RequestedInput.InterruptID
		}
		if len(ev.LongRunningToolIDs) > 0 {
			return ev.LongRunningToolIDs[0]
		}
	}
	return ""
}

func greetingOutput(events []restEvent) string {
	for _, ev := range events {
		if s, ok := ev.Output.(string); ok && s != "" {
			return s
		}
	}
	return ""
}

func runTurn(t *testing.T, baseURL, sid string, msg *genai.Content) []restEvent {
	t.Helper()
	body := map[string]any{
		"appName":    hitlApp,
		"userId":     hitlUser,
		"sessionId":  sid,
		"newMessage": msg,
	}
	var events []restEvent
	postJSON(t, baseURL+"/run", body, &events)
	return events
}

func createSession(t *testing.T, baseURL string) string {
	t.Helper()
	var resp struct {
		ID string `json:"id"`
	}
	postJSON(t, fmt.Sprintf("%s/apps/%s/users/%s/sessions", baseURL, hitlApp, hitlUser),
		map[string]any{}, &resp)
	if resp.ID == "" {
		t.Fatal("create session returned an empty ID")
	}
	return resp.ID
}

func postJSON(t *testing.T, url string, body, out any) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	resp, err := http.Post(url, "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST %s: status %d, body %s", url, resp.StatusCode, data)
	}
	if err := json.Unmarshal(data, out); err != nil {
		t.Fatalf("decode response from %s: %v\nbody: %s", url, err, data)
	}
}

const (
	hitlApp  = "hitl_simple"
	hitlUser = "u"
)

// newHITLServer builds the REST handler around a workflow agent shaped
// like examples/workflow/hitl_simple: ask_name pauses on a unique
// interrupt ID per request, greet returns the greeting for the reply.
func newHITLServer(t *testing.T) *adkrest.Server {
	t.Helper()
	ask := workflow.NewEmittingFunctionNode[any, any]("ask_name",
		func(ic agent.Context, _ any, emit func(*session.Event) error) (any, error) {
			if err := emit(workflow.NewRequestInputEvent(ic, session.RequestInput{
				InterruptID: "ask_name-" + uuid.NewString(),
				Message:     "What's your name?",
			})); err != nil {
				return nil, err
			}
			return nil, workflow.ErrNodeInterrupted
		},
		workflow.NodeConfig{},
	)
	greet := workflow.NewFunctionNode("greet",
		func(_ agent.Context, name string) (string, error) {
			if name == "" {
				name = "stranger"
			}
			return "Hello, " + name + "!", nil
		},
		workflow.NodeConfig{},
	)
	a, err := workflowagent.New(workflowagent.Config{
		Name:  hitlApp,
		Edges: workflow.Chain(workflow.Start, ask, greet),
	})
	if err != nil {
		t.Fatalf("workflowagent.New() error = %v", err)
	}

	srv, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService: session.InMemoryService(),
		AgentLoader:    agent.NewSingleLoader(a),
	})
	if err != nil {
		t.Fatalf("adkrest.NewServer() error = %v", err)
	}
	return srv
}
