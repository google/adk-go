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
	// Output here is a transcript, not a log: the timestamp prefix is noise.
	log.SetFlags(0)

	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		log.Fatal("GOOGLE_CLOUD_PROJECT must be set")
	}
	location := cmp.Or(os.Getenv("GOOGLE_CLOUD_LOCATION"), "global")
	// Enabling the Cloud Logging API auto-registers a server declaring this tool,
	// so the default works without registering anything. Prefer a distinctive
	// name: a generic one like "list_services" is declared by several servers.
	want := cmp.Or(os.Getenv("REGISTRY_TOOL"), "list_log_names")

	client, err := agentregistry.New(ctx, agentregistry.Config{
		ProjectID: project,
		Location:  location,
	})
	if err != nil {
		log.Fatalf("Failed to create the registry client: %v", err)
	}

	server, err := pickProvider(ctx, client, want)
	if err != nil {
		// Name the location: GOOGLE_CLOUD_LOCATION also steers the catalog, so a
		// region meant for the model reads as "nobody provides it".
		log.Fatalf("Failed to find a provider for %q in projects/%s/locations/%s: %s", want, project, location, explain(err))
	}
	log.Printf("Tool %q is provided by %q (%s)", want, cmp.Or(server.DisplayName, server.Name), server.Name)

	// Requests to *.googleapis.com endpoints reuse the registry's credentials;
	// anything else gets http.DefaultClient, so pass WithMCPHTTPClient or
	// WithMCPHeaders when it needs credentials. This resolves the endpoint only:
	// the MCP session opens on the first model turn.
	toolset, err := client.MCPToolset(ctx, server.Name)
	if err != nil {
		log.Fatalf("Failed to build a toolset for MCP server %q: %s", server.Name, explain(err))
	}

	// The empty config resolves the backend from the environment: a Gemini API
	// key, or Vertex AI when GOOGLE_GENAI_USE_VERTEXAI is "1" or "true". Both
	// have to work here, hence a versioned model: Vertex AI does not serve the
	// "-latest" aliases.
	m, err := gemini.NewModel(ctx, "gemini-3.5-flash", &genai.ClientConfig{})
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

// explain renders a registry failure as something a human can act on. The
// service reports a denial as a screenful of JSON, so unwrap the typed
// [agentregistry.APIError] and keep the parts that identify the fix.
//
// It returns text, not an error: wrapping the result back into one with %w
// would re-append the envelope it exists to suppress.
func explain(err error) string {
	var apiErr *agentregistry.APIError
	if !errors.As(err, &apiErr) {
		return err.Error()
	}
	var envelope struct {
		Error struct {
			Message string `json:"message"`
			Status  string `json:"status"`
		} `json:"error"`
	}
	detail := apiErr.Body
	if json.Unmarshal([]byte(apiErr.Body), &envelope) == nil && envelope.Error.Message != "" {
		detail = envelope.Error.Message
		if envelope.Error.Status != "" {
			detail = envelope.Error.Status + ": " + detail
		}
	}
	if apiErr.StatusCode == http.StatusForbidden {
		detail += "\nGrant roles/agentregistry.viewer on this project, or point GOOGLE_CLOUD_PROJECT at one where you have it."
	}
	return fmt.Sprintf("HTTP %d — %s", apiErr.StatusCode, detail)
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
		return nil, errors.New("no registered MCP server declares this tool; run the discover sample to see what is available")
	}
	if len(matches) > 1 {
		others := make([]string, 0, len(matches)-1)
		for _, m := range matches[1:] {
			others = append(others, cmp.Or(m.DisplayName, m.Name))
		}
		log.Printf("%q is also declared by %s", name, strings.Join(others, ", "))
	}
	return matches[0], nil
}
