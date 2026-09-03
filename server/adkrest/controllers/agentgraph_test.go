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

package controllers_test

import (
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/server/adkrest/controllers"
	"google.golang.org/adk/v2/server/adkrest/internal/services"
)

const graphAppName = "test-app"

// newGraphAgent builds a plain agent with the given children. The graph
// endpoints only read names, descriptions and the sub-agent tree, so a custom
// agent with no Run function is enough.
func newGraphAgent(t *testing.T, name, description string, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name:        name,
		Description: description,
		SubAgents:   subAgents,
	})
	if err != nil {
		t.Fatalf("agent.New(%q) failed: %v", name, err)
	}
	return a
}

// graphController wraps root in a single-agent loader reachable as
// [graphAppName].
func graphController(t *testing.T, root agent.Agent) *controllers.AgentGraphAPIController {
	t.Helper()
	return controllers.NewAgentGraphAPIController(agent.NewSingleLoader(root))
}

// callGraph invokes a graph handler with app_name set to appName and the given
// query parameters.
func callGraph(t *testing.T, h http.HandlerFunc, appName string, query url.Values) *httptest.ResponseRecorder {
	t.Helper()
	target := "/dev/apps/" + appName + "/build_graph"
	if encoded := query.Encode(); encoded != "" {
		target += "?" + encoded
	}
	req := httptest.NewRequest(http.MethodGet, target, nil)
	req = mux.SetURLVars(req, map[string]string{"app_name": appName})
	rr := httptest.NewRecorder()
	h(rr, req)
	return rr
}

// nodeQuery builds the query string the UI sends for one breadcrumb level.
func nodeQuery(node string) url.Values {
	return url.Values{"node": []string{node}}
}

// TestBuildGraphHandlerRootShape pins the response shape. The UI walks children
// through sub_agents, so the key has to be there and has to be a list even when
// the agent has no children: a null makes the walk throw.
func TestBuildGraphHandlerRootShape(t *testing.T) {
	root := newGraphAgent(t, graphAppName, "the root agent")
	rr := callGraph(t, graphController(t, root).BuildGraphHandler, graphAppName, nil)

	if got, want := rr.Code, http.StatusOK; got != want {
		t.Fatalf("build_graph status = %d, want %d; body: %s", got, want, rr.Body.String())
	}
	if got, want := rr.Body.String(), `"sub_agents":[]`; !strings.Contains(got, want) {
		t.Errorf("build_graph raw body = %q, want it to contain %q, not a null sub_agents", got, want)
	}

	var body map[string]json.RawMessage
	if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil {
		t.Fatalf("json.Unmarshal(%q) failed: %v", rr.Body.String(), err)
	}
	if diff := cmp.Diff([]string{"readme", "root_agent"}, slices.Sorted(maps.Keys(body))); diff != "" {
		t.Errorf("build_graph top-level field names mismatch (-want +got):\n%s", diff)
	}

	var rootAgent map[string]any
	if err := json.Unmarshal(body["root_agent"], &rootAgent); err != nil {
		t.Fatalf("json.Unmarshal(root_agent) failed: %v", err)
	}
	if diff := cmp.Diff([]string{"description", "name", "sub_agents"}, slices.Sorted(maps.Keys(rootAgent))); diff != "" {
		t.Errorf("build_graph root_agent field names mismatch (-want +got):\n%s", diff)
	}
	if got, want := rootAgent["name"], graphAppName; got != want {
		t.Errorf("build_graph root_agent name = %v, want %q", got, want)
	}
	if got, want := rootAgent["description"], "the root agent"; got != want {
		t.Errorf("build_graph root_agent description = %v, want %q", got, want)
	}
	if subAgents, ok := rootAgent["sub_agents"].([]any); !ok {
		t.Errorf("build_graph root_agent sub_agents = %v (%T), want an empty JSON list", rootAgent["sub_agents"], rootAgent["sub_agents"])
	} else if len(subAgents) != 0 {
		t.Errorf("build_graph root_agent sub_agents = %v, want it empty", subAgents)
	}

	var readme string
	if err := json.Unmarshal(body["readme"], &readme); err != nil {
		t.Fatalf("json.Unmarshal(readme) failed: %v", err)
	}
	if readme != "" {
		t.Errorf("build_graph readme = %q, want it empty", readme)
	}
}

func TestBuildGraphHandlerNestedTree(t *testing.T) {
	grandchild := newGraphAgent(t, "grandchild", "leaf")
	child := newGraphAgent(t, "child", "middle", grandchild)
	root := newGraphAgent(t, graphAppName, "the root agent", child)

	rr := callGraph(t, graphController(t, root).BuildGraphHandler, graphAppName, nil)
	if got, want := rr.Code, http.StatusOK; got != want {
		t.Fatalf("build_graph status = %d, want %d; body: %s", got, want, rr.Body.String())
	}

	var got controllers.AppInfoResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("json.Unmarshal(%q) failed: %v", rr.Body.String(), err)
	}

	want := controllers.AppInfoResponse{
		RootAgent: controllers.AgentNode{
			Name:        graphAppName,
			Description: "the root agent",
			SubAgents: []controllers.AgentNode{{
				Name:        "child",
				Description: "middle",
				SubAgents: []controllers.AgentNode{{
					Name:        "grandchild",
					Description: "leaf",
					SubAgents:   []controllers.AgentNode{},
				}},
			}},
		},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("build_graph nested tree mismatch (-want +got):\n%s", diff)
	}
}

// TestBuildGraphImageHandlerWithoutNodeIsKeyedByRoot pins the map shape the UI
// preloads every breadcrumb level from. It maps the root agent's own name to
// the root level, so the key is part of the contract.
func TestBuildGraphImageHandlerWithoutNodeIsKeyedByRoot(t *testing.T) {
	child := newGraphAgent(t, "child", "middle")
	root := newGraphAgent(t, graphAppName, "the root agent", child)

	rr := callGraph(t, graphController(t, root).BuildGraphImageHandler, graphAppName, nil)
	if got, want := rr.Code, http.StatusOK; got != want {
		t.Fatalf("build_graph_image status = %d, want %d; body: %s", got, want, rr.Body.String())
	}

	var got map[string]controllers.DotGraph
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("json.Unmarshal(%q) failed: %v", rr.Body.String(), err)
	}
	if diff := cmp.Diff([]string{graphAppName}, slices.Sorted(maps.Keys(got))); diff != "" {
		t.Errorf("build_graph_image map keys mismatch (-want +got):\n%s", diff)
	}
	if got[graphAppName].DotSrc == "" {
		t.Error("build_graph_image root entry dotSrc is empty, want Graphviz source")
	}
	if !strings.Contains(got[graphAppName].DotSrc, "child") {
		t.Errorf("build_graph_image root dotSrc = %q, want it to include the sub-agent", got[graphAppName].DotSrc)
	}
}

// TestBuildGraphImageHandlerWithNodeIsBare checks the other shape: asked for one
// subtree, the endpoint answers {"dotSrc": …} rather than a map.
func TestBuildGraphImageHandlerWithNodeIsBare(t *testing.T) {
	child := newGraphAgent(t, "child", "middle")
	root := newGraphAgent(t, graphAppName, "the root agent", child)

	rr := callGraph(t, graphController(t, root).BuildGraphImageHandler, graphAppName, nodeQuery("child"))
	if got, want := rr.Code, http.StatusOK; got != want {
		t.Fatalf("build_graph_image?node=child status = %d, want %d; body: %s", got, want, rr.Body.String())
	}

	var body map[string]json.RawMessage
	if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil {
		t.Fatalf("json.Unmarshal(%q) failed: %v", rr.Body.String(), err)
	}
	if diff := cmp.Diff([]string{"dotSrc"}, slices.Sorted(maps.Keys(body))); diff != "" {
		t.Errorf("build_graph_image?node=child field names mismatch (-want +got):\n%s", diff)
	}

	var got controllers.DotGraph
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("json.Unmarshal(%q) failed: %v", rr.Body.String(), err)
	}
	if !strings.Contains(got.DotSrc, "child") {
		t.Errorf("build_graph_image?node=child dotSrc = %q, want it to draw the child", got.DotSrc)
	}
	if strings.Contains(got.DotSrc, graphAppName) {
		t.Errorf("build_graph_image?node=child dotSrc = %q, want the child subtree, not the root graph", got.DotSrc)
	}
}

// TestBuildGraphImageHandlerUnknownNodeIs404 is the regression that matters
// most here. An unresolvable node used to fall back to the root agent and
// answer 200, telling the caller a graph belonged to a node it did not.
func TestBuildGraphImageHandlerUnknownNodeIs404(t *testing.T) {
	child := newGraphAgent(t, "child", "middle")
	root := newGraphAgent(t, graphAppName, "the root agent", child)

	rr := callGraph(t, graphController(t, root).BuildGraphImageHandler, graphAppName, nodeQuery("no_such_agent"))

	if got, want := rr.Code, http.StatusNotFound; got != want {
		t.Errorf("build_graph_image?node=no_such_agent status = %d, want %d; body: %s", got, want, rr.Body.String())
	}
	if got := rr.Body.String(); strings.Contains(got, "dotSrc") {
		t.Errorf("build_graph_image?node=no_such_agent body = %q, want an error rather than a fallback graph", got)
	}
}

// TestBuildGraphImageHandlerResolvesByPath builds a tree with the same agent
// name at two depths. A name search finds whichever one it reaches first; only
// walking the breadcrumb path returns the one the caller asked for.
func TestBuildGraphImageHandlerResolvesByPath(t *testing.T) {
	sharedUnderA := newGraphAgent(t, "shared", "under branch_a", newGraphAgent(t, "leaf_a", "a"))
	branchA := newGraphAgent(t, "branch_a", "first branch", sharedUnderA)

	sharedUnderB := newGraphAgent(t, "shared", "under branch_b", newGraphAgent(t, "leaf_b", "b"))
	mid := newGraphAgent(t, "mid", "middle", sharedUnderB)
	branchB := newGraphAgent(t, "branch_b", "second branch", mid)

	root := newGraphAgent(t, graphAppName, "the root agent", branchA, branchB)
	handler := graphController(t, root).BuildGraphImageHandler

	tests := []struct {
		name       string
		node       string
		wantLeaf   string
		wantNoLeaf string
	}{
		{
			name:       "shallow duplicate",
			node:       "branch_a/shared",
			wantLeaf:   "leaf_a",
			wantNoLeaf: "leaf_b",
		},
		{
			name:       "deep duplicate",
			node:       "branch_b/mid/shared",
			wantLeaf:   "leaf_b",
			wantNoLeaf: "leaf_a",
		},
		{
			name:       "path repeating the root name",
			node:       graphAppName + "/branch_b/mid/shared",
			wantLeaf:   "leaf_b",
			wantNoLeaf: "leaf_a",
		},
		{
			// The UI suffixes a segment with "@" and a discriminator when the
			// same agent shows up more than once in a trace.
			name:       "segments carrying an @ suffix",
			node:       "branch_a@1/shared@2",
			wantLeaf:   "leaf_a",
			wantNoLeaf: "leaf_b",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rr := callGraph(t, handler, graphAppName, nodeQuery(tt.node))
			if got, want := rr.Code, http.StatusOK; got != want {
				t.Fatalf("build_graph_image?node=%s status = %d, want %d; body: %s", tt.node, got, want, rr.Body.String())
			}

			var got controllers.DotGraph
			if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
				t.Fatalf("json.Unmarshal(%q) failed: %v", rr.Body.String(), err)
			}
			if !strings.Contains(got.DotSrc, tt.wantLeaf) {
				t.Errorf("build_graph_image?node=%s dotSrc = %q, want it to contain %q", tt.node, got.DotSrc, tt.wantLeaf)
			}
			if strings.Contains(got.DotSrc, tt.wantNoLeaf) {
				t.Errorf("build_graph_image?node=%s dotSrc = %q, want it not to contain %q from the other branch", tt.node, got.DotSrc, tt.wantNoLeaf)
			}
		})
	}
}

// TestBuildGraphImageHandlerHonoursDarkMode checks the palette follows the
// query parameter. The UI preloads both renderings and switches between them
// without a round trip, so one palette for both requests leaves half the users
// with an unreadable graph.
func TestBuildGraphImageHandlerHonoursDarkMode(t *testing.T) {
	root := newGraphAgent(t, graphAppName, "the root agent", newGraphAgent(t, "child", "middle"))
	handler := graphController(t, root).BuildGraphImageHandler

	dotFor := func(darkMode string) string {
		t.Helper()
		rr := callGraph(t, handler, graphAppName, url.Values{"dark_mode": []string{darkMode}})
		if got, want := rr.Code, http.StatusOK; got != want {
			t.Fatalf("build_graph_image?dark_mode=%s status = %d, want %d; body: %s", darkMode, got, want, rr.Body.String())
		}
		var got map[string]controllers.DotGraph
		if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
			t.Fatalf("json.Unmarshal(%q) failed: %v", rr.Body.String(), err)
		}
		return got[graphAppName].DotSrc
	}

	light := dotFor("false")
	dark := dotFor("true")

	if light == dark {
		t.Errorf("build_graph_image returned the same dotSrc for dark_mode=false and dark_mode=true:\n%s", light)
	}
	if !strings.Contains(light, services.LightTheme.Background) {
		t.Errorf("build_graph_image?dark_mode=false dotSrc = %q, want the light background %s", light, services.LightTheme.Background)
	}
	if !strings.Contains(dark, services.DarkTheme.Background) {
		t.Errorf("build_graph_image?dark_mode=true dotSrc = %q, want the dark background %s", dark, services.DarkTheme.Background)
	}
	if strings.Contains(light, services.DarkTheme.Background) {
		t.Errorf("build_graph_image?dark_mode=false dotSrc = %q, want it not to carry the dark background %s", light, services.DarkTheme.Background)
	}
}

func TestGraphHandlersUnknownApp(t *testing.T) {
	root := newGraphAgent(t, graphAppName, "the root agent")
	controller := graphController(t, root)

	tests := []struct {
		name    string
		handler http.HandlerFunc
	}{
		{name: "build_graph", handler: controller.BuildGraphHandler},
		{name: "build_graph_image", handler: controller.BuildGraphImageHandler},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rr := callGraph(t, tt.handler, "not-an-app", nil)
			if got, want := rr.Code, http.StatusNotFound; got != want {
				t.Errorf("%s for an unknown app status = %d, want %d; body: %s", tt.name, got, want, rr.Body.String())
			}
		})
	}
}

func TestGraphHandlersNilAgentLoader(t *testing.T) {
	controller := controllers.NewAgentGraphAPIController(nil)

	tests := []struct {
		name    string
		handler http.HandlerFunc
	}{
		{name: "build_graph", handler: controller.BuildGraphHandler},
		{name: "build_graph_image", handler: controller.BuildGraphImageHandler},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rr := callGraph(t, tt.handler, graphAppName, nil)
			if got, want := rr.Code, http.StatusServiceUnavailable; got != want {
				t.Errorf("%s with no agent loader status = %d, want %d; body: %s", tt.name, got, want, rr.Body.String())
			}
		})
	}
}
