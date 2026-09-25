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

package controllers

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/server/adkrest/internal/services"
)

// AgentGraphAPIController serves the agent-structure endpoints that back the
// "Agent structure" panel of the ADK web UI.
type AgentGraphAPIController struct {
	agentLoader agent.Loader
}

// NewAgentGraphAPIController creates a new AgentGraphAPIController.
func NewAgentGraphAPIController(agentLoader agent.Loader) *AgentGraphAPIController {
	return &AgentGraphAPIController{agentLoader: agentLoader}
}

// AgentNode describes one agent in the tree returned by BuildGraphHandler.
// The web UI walks children by looking for a "sub_agents" array whose entries
// carry a "name", so those two field names are part of the client contract.
type AgentNode struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	SubAgents   []AgentNode `json:"sub_agents"`
}

// AppInfoResponse is the body of the build_graph endpoint.
type AppInfoResponse struct {
	RootAgent AgentNode `json:"root_agent"`
	// Readme is always empty: ADK Go agents are defined in code, so there is no
	// agent directory to read a README from.
	Readme string `json:"readme"`
}

// DotGraph carries Graphviz DOT source. The UI renders it client-side.
type DotGraph struct {
	DotSrc string `json:"dotSrc"`
}

// BuildGraphHandler returns the agent tree for an app, used by the UI to build
// the breadcrumb navigation over the agent structure.
func (c *AgentGraphAPIController) BuildGraphHandler(rw http.ResponseWriter, req *http.Request) {
	rootAgent, ok := c.loadAgent(rw, req)
	if !ok {
		return
	}
	EncodeJSONResponse(AppInfoResponse{RootAgent: toAgentNode(rootAgent)}, http.StatusOK, rw)
}

// BuildGraphImageHandler returns Graphviz DOT source for an app's agent tree.
//
// The response shape depends on the request, because the UI calls this endpoint
// two different ways. With a "node" query parameter it asks for one subtree and
// expects a bare {"dotSrc": …}. Without one it preloads every level and expects
// a map keyed by agent path; the root agent's own name maps to the root level.
func (c *AgentGraphAPIController) BuildGraphImageHandler(rw http.ResponseWriter, req *http.Request) {
	rootAgent, ok := c.loadAgent(rw, req)
	if !ok {
		return
	}

	// The UI preloads a light and a dark rendering and switches between them
	// without a round trip, so the palette has to follow this parameter.
	theme := services.ThemeFor(req.URL.Query().Get("dark_mode") == "true")

	node := req.URL.Query().Get("node")
	if node == "" {
		dot, err := services.GetAgentGraphWithTheme(req.Context(), rootAgent, nil, theme)
		if err != nil {
			http.Error(rw, err.Error(), http.StatusInternalServerError)
			return
		}
		EncodeJSONResponse(map[string]DotGraph{rootAgent.Name(): {DotSrc: dot}}, http.StatusOK, rw)
		return
	}

	target, err := resolveAgentPath(rootAgent, node)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusNotFound)
		return
	}
	dot, err := services.GetAgentGraphWithTheme(req.Context(), target, nil, theme)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}
	EncodeJSONResponse(DotGraph{DotSrc: dot}, http.StatusOK, rw)
}

// resolveAgentPath walks a slash-separated breadcrumb path from the root agent.
//
// It deliberately does not fall back to a name search. An unresolvable path
// used to return the root agent's graph with 200, which told the caller that
// the graph it was looking at belonged to the node it asked for. Two agents
// sharing a name at different depths resolved to whichever one a depth-first
// search reached first.
func resolveAgentPath(root agent.Agent, path string) (agent.Agent, error) {
	current := root
	for i, segment := range strings.Split(path, "/") {
		// The UI suffixes a segment with "@" and an invocation discriminator
		// when the same agent appears more than once in a trace.
		name, _, _ := strings.Cut(segment, "@")
		if name == "" {
			continue
		}
		// The path may or may not repeat the root agent's own name first,
		// depending on which breadcrumb the UI built it from.
		if i == 0 && name == root.Name() {
			continue
		}
		next := current.FindSubAgent(name)
		if next == nil {
			return nil, fmt.Errorf("agent %q not found under %q in path %q", name, current.Name(), path)
		}
		current = next
	}
	return current, nil
}

// loadAgent resolves the app_name path variable, writing the error response
// itself and reporting false when it cannot.
func (c *AgentGraphAPIController) loadAgent(rw http.ResponseWriter, req *http.Request) (agent.Agent, bool) {
	if c.agentLoader == nil {
		http.Error(rw, "no agent loader configured", http.StatusServiceUnavailable)
		return nil, false
	}
	appName := mux.Vars(req)["app_name"]
	if appName == "" {
		http.Error(rw, "app_name parameter is required", http.StatusBadRequest)
		return nil, false
	}
	loaded, err := c.agentLoader.LoadAgent(appName)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusNotFound)
		return nil, false
	}
	return loaded, true
}

func toAgentNode(a agent.Agent) AgentNode {
	node := AgentNode{
		Name:        a.Name(),
		Description: a.Description(),
		SubAgents:   []AgentNode{},
	}
	for _, sub := range a.SubAgents() {
		node.SubAgents = append(node.SubAgents, toAgentNode(sub))
	}
	return node
}
