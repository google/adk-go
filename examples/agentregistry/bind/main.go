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

// Package main builds an LlmAgent from a capability rather than an address: you
// name the tool you need, and the Google Cloud Agent Registry decides which MCP
// server provides it.
package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"slices"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agentregistry"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"
)

func main() {
	ctx := context.Background()

	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		log.Fatal("GOOGLE_CLOUD_PROJECT must be set")
	}
	location := cmp.Or(os.Getenv("GOOGLE_CLOUD_LOCATION"), "global")
	// Enabling the Cloud Run API auto-registers a server declaring this tool, so
	// the default works without registering anything. Prefer a distinctive name:
	// a generic one like "list_services" is declared by several servers.
	want := cmp.Or(os.Getenv("REGISTRY_TOOL"), "deploy_service_from_image")

	client, err := agentregistry.New(ctx, agentregistry.Config{
		ProjectID: project,
		Location:  location,
	})
	if err != nil {
		log.Fatalf("Failed to create the registry client: %v", err)
	}

	server, err := pickProvider(ctx, client, want)
	if err != nil {
		log.Fatalf("Failed to find a provider for %q: %v", want, explain(err))
	}
	log.Printf("Tool %q is provided by %q (%s)", want, server.DisplayName, server.Name)

	// Requests to *.googleapis.com endpoints reuse the registry's credentials;
	// anything else needs WithMCPHTTPClient/WithMCPHeaders.
	toolset, err := client.MCPToolset(ctx, server.Name)
	if err != nil {
		log.Fatalf("Failed to connect to MCP server %q: %v", server.Name, err)
	}

	// The empty config resolves the backend from the environment: a Gemini API
	// key, or Vertex AI when GOOGLE_GENAI_USE_VERTEXAI is set.
	m, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{})
	if err != nil {
		log.Fatalf("Failed to create the model: %v", err)
	}

	root, err := llmagent.New(llmagent.Config{
		Name:        "registry_hub",
		Model:       m,
		Description: fmt.Sprintf("Agent wired to the registered provider of %q.", want),
		Instruction: "Answer using the tools discovered in the Agent Registry. " +
			"Name the tool you used.",
		Toolsets: []tool.Toolset{toolset},
	})
	if err != nil {
		log.Fatalf("Failed to create the agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(root),
	}
	l := full.NewLauncher()
	if err := l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

// explain condenses a registry failure into something a human can act on. The
// service reports a denial as a screenful of JSON, so unwrap the typed
// [agentregistry.APIError] and keep the parts that identify the fix.
func explain(err error) error {
	var apiErr *agentregistry.APIError
	if !errors.As(err, &apiErr) {
		return err
	}
	var envelope struct {
		Error struct {
			Message string `json:"message"`
			Status  string `json:"status"`
		} `json:"error"`
	}
	detail := apiErr.Body
	if json.Unmarshal([]byte(apiErr.Body), &envelope) == nil && envelope.Error.Message != "" {
		detail = fmt.Sprintf("%s: %s", envelope.Error.Status, envelope.Error.Message)
	}
	if apiErr.StatusCode == http.StatusForbidden {
		detail += "\nGrant roles/agentregistry.viewer on this project, or point GOOGLE_CLOUD_PROJECT at one where you have it."
	}
	return fmt.Errorf("HTTP %d — %s", apiErr.StatusCode, detail)
}

// pickProvider resolves a capability to the MCP server that will serve it, by
// scanning the catalog for servers that declare a tool called name.
//
// This is the lookup a hardcoded endpoint cannot do — the answer depends on what
// is registered in this project right now, and it comes from the catalog's own
// metadata rather than from connecting to every candidate in turn.
//
// Tool names are not unique across servers, so a capability can have several
// providers. This takes the first and names the rest; a real application would
// apply its own policy, such as a trusted publisher or an explicit allowlist.
func pickProvider(ctx context.Context, c *agentregistry.Client, name string) (*agentregistry.MCPServer, error) {
	var matches []*agentregistry.MCPServer
	for server, err := range c.AllMCPServers(ctx) {
		if err != nil {
			return nil, err
		}
		if slices.ContainsFunc(server.Tools, func(t agentregistry.Tool) bool { return t.Name == name }) {
			matches = append(matches, server)
		}
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("no registered MCP server declares it; run the discover sample to see what is available")
	}
	if len(matches) > 1 {
		others := make([]string, 0, len(matches)-1)
		for _, m := range matches[1:] {
			others = append(others, m.DisplayName)
		}
		log.Printf("%q is also declared by %s", name, strings.Join(others, ", "))
	}
	return matches[0], nil
}
