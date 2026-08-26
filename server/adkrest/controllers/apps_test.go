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

package controllers_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/server/adkrest/controllers"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type orderArgs struct {
	OrderID string `json:"orderId"`
}

type orderResult struct {
	Status string `json:"status"`
}

func newOrderTool(t *testing.T) tool.Tool {
	t.Helper()
	ft, err := functiontool.New(functiontool.Config{
		Name:        "lookup_order",
		Description: "Looks up an order by id.",
	}, func(ctx agent.Context, args orderArgs) (orderResult, error) {
		return orderResult{Status: "shipped"}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New failed: %v", err)
	}
	return ft
}

// appInfoRequest serves GET /apps/{app}/app-info from a router wired to loader,
// and returns the recorded response.
func appInfoRequest(t *testing.T, loader agent.Loader, app string) *httptest.ResponseRecorder {
	t.Helper()
	controller := controllers.NewAppsAPIController(loader)
	router := mux.NewRouter()
	router.HandleFunc("/apps/{app_name}/app-info", controller.AppInfoHandler).Methods(http.MethodGet)

	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/apps/"+app+"/app-info", nil))
	return rr
}

func TestAppInfoHandler(t *testing.T) {
	subAgent, err := llmagent.New(llmagent.Config{
		Name:        "support_agent",
		Description: "Answers support questions.",
		Instruction: "Help the customer.",
		Tools:       []tool.Tool{newOrderTool(t)},
	})
	if err != nil {
		t.Fatalf("llmagent.New failed: %v", err)
	}
	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "concierge",
		Description: "Routes customer requests.",
		Instruction: "Delegate to the right agent.",
		SubAgents:   []agent.Agent{subAgent},
	})
	if err != nil {
		t.Fatalf("llmagent.New failed: %v", err)
	}

	rr := appInfoRequest(t, agent.NewSingleLoader(rootAgent), "concierge")
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", rr.Code, http.StatusOK, rr.Body.String())
	}

	// Decode into a generic map: the exact JSON keys are the wire contract, so
	// assert on them rather than on Go field names.
	var got map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("failed to unmarshal response: %v; body: %s", err, rr.Body.String())
	}

	for key, want := range map[string]any{
		"name":          "concierge",
		"rootAgentName": "concierge",
		"description":   "Routes customer requests.",
		"language":      "go",
	} {
		if got[key] != want {
			t.Errorf("%s = %v, want %v", key, got[key], want)
		}
	}

	agents, ok := got["agents"].(map[string]any)
	if !ok {
		t.Fatalf("agents is %T, want an object", got["agents"])
	}
	if len(agents) != 2 {
		t.Errorf("len(agents) = %d, want 2", len(agents))
	}

	root, ok := agents["concierge"].(map[string]any)
	if !ok {
		t.Fatalf("agents[concierge] is %T, want an object", agents["concierge"])
	}
	// Sub-agent names are camelCase on the wire, unlike adk-python's sub_agents.
	if diff := cmp.Diff([]any{"support_agent"}, root["subAgents"]); diff != "" {
		t.Errorf("subAgents mismatch (-want +got):\n%s", diff)
	}

	support, ok := agents["support_agent"].(map[string]any)
	if !ok {
		t.Fatalf("agents[support_agent] is %T, want an object", agents["support_agent"])
	}
	if got, want := support["instruction"], "Help the customer."; got != want {
		t.Errorf("instruction = %v, want %v", got, want)
	}
	tools, ok := support["tools"].([]any)
	if !ok {
		t.Fatalf("tools is %T, want an array", support["tools"])
	}
	if len(tools) != 1 {
		t.Fatalf("len(tools) = %d, want 1 (the built-in tool has no declaration)", len(tools))
	}
}

func TestAppInfoHandlerEmptyToolsIsNotNull(t *testing.T) {
	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "plain",
		Description: "No tools at all.",
		Instruction: "Answer briefly.",
	})
	if err != nil {
		t.Fatalf("llmagent.New failed: %v", err)
	}

	rr := appInfoRequest(t, agent.NewSingleLoader(rootAgent), "plain")
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusOK)
	}
	// The contract requires tools to be present, and forbids null fields.
	if body := rr.Body.String(); !strings.Contains(body, `"tools":[]`) {
		t.Errorf("response does not contain an empty tools array; body: %s", body)
	}
}

// TestAppInfoHandlerWorkflowRoot covers an app whose root agent is not an LLM
// agent. adk-python answers 400 here; this server describes the app instead.
func TestAppInfoHandlerWorkflowRoot(t *testing.T) {
	writer, err := llmagent.New(llmagent.Config{
		Name:        "writer",
		Description: "Writes a draft.",
		Instruction: "Write about the topic.",
	})
	if err != nil {
		t.Fatalf("llmagent.New failed: %v", err)
	}
	pipeline, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:        "writing_pipeline",
			Description: "Writes, then edits.",
			SubAgents:   []agent.Agent{writer},
		},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New failed: %v", err)
	}

	rr := appInfoRequest(t, agent.NewSingleLoader(pipeline), "writing_pipeline")
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", rr.Code, http.StatusOK, rr.Body.String())
	}

	var got map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	if got["rootAgentName"] != "writing_pipeline" {
		t.Errorf("rootAgentName = %v, want writing_pipeline", got["rootAgentName"])
	}

	agents, ok := got["agents"].(map[string]any)
	if !ok {
		t.Fatalf("agents is %T, want an object", got["agents"])
	}
	if _, ok := agents["writing_pipeline"]; !ok {
		t.Error("agents is missing the workflow root itself")
	}
	if _, ok := agents["writer"]; !ok {
		t.Error("agents is missing writer; the sub-tree below a non-LLM agent must still be walked")
	}
}

func TestAppInfoHandlerUnknownApp(t *testing.T) {
	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "known",
		Description: "The only app.",
		Instruction: "Answer.",
	})
	if err != nil {
		t.Fatalf("llmagent.New failed: %v", err)
	}

	rr := appInfoRequest(t, agent.NewSingleLoader(rootAgent), "missing")
	if rr.Code != http.StatusNotFound {
		t.Errorf("status = %d, want %d; body: %s", rr.Code, http.StatusNotFound, rr.Body.String())
	}
}
