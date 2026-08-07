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

// Package main prints the A2A agents, MCP servers, and model endpoints
// registered in a Google Cloud Agent Registry.
package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log"
	"net/http"
	"os"
	"strings"

	"google.golang.org/adk/v2/agentregistry"
)

func main() {
	ctx := context.Background()

	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		log.Fatal("GOOGLE_CLOUD_PROJECT must be set")
	}
	// The registry's default location, as in adk-python.
	location := cmp.Or(os.Getenv("GOOGLE_CLOUD_LOCATION"), "global")

	client, err := agentregistry.New(ctx, agentregistry.Config{
		ProjectID: project,
		Location:  location,
	})
	if err != nil {
		log.Fatalf("Failed to create the registry client: %v", err)
	}

	// List options apply to every collection: the filter is evaluated
	// server-side, the page size bounds one round trip (not the total).
	opts := []agentregistry.ListOption{agentregistry.WithPageSize(10)}
	if filter := os.Getenv("REGISTRY_FILTER"); filter != "" {
		opts = append(opts, agentregistry.WithFilter(filter))
	}

	fmt.Printf("Catalog of projects/%s/locations/%s\n", project, location)

	if err := printAll("A2A agents", client.AllAgents(ctx, opts...),
		func(a *agentregistry.Agent) entry {
			return entry{
				displayName: a.DisplayName,
				resource:    a.Name,
				detail:      summary("skills", skillNames(a.Skills)),
			}
		}); err != nil {
		log.Fatalf("Failed to list agents: %v", explain(err))
	}

	if err := printAll("MCP servers", client.AllMCPServers(ctx, opts...),
		func(s *agentregistry.MCPServer) entry {
			return entry{
				displayName: s.DisplayName,
				resource:    s.Name,
				detail:      summary("tools", toolNames(s.Tools)),
			}
		}); err != nil {
		log.Fatalf("Failed to list MCP servers: %v", explain(err))
	}

	if err := printAll("Model endpoints", client.AllEndpoints(ctx, opts...),
		func(e *agentregistry.Endpoint) entry {
			return entry{
				displayName: e.DisplayName,
				resource:    e.Name,
				detail:      endpointURL(e),
			}
		}); err != nil {
		log.Fatalf("Failed to list endpoints: %v", explain(err))
	}
}

// entry is the one-line-per-resource summary printed for every catalog kind.
type entry struct {
	displayName string
	// resource is the full resource name: the input to the Get* methods and to
	// the RemoteAgent/MCPToolset factory helpers.
	resource string
	detail   string
}

// printAll drains an auto-paging discovery iterator, printing one entry per
// resource. Those iterators report a failed page fetch as a single (nil, error)
// and then stop, so returning on the first error ends the iteration.
func printAll[T any](title string, seq iter.Seq2[*T, error], row func(*T) entry) error {
	fmt.Printf("\n%s:\n", title)
	count := 0
	for v, err := range seq {
		if err != nil {
			return err
		}
		count++
		e := row(v)
		fmt.Printf("  %s\n", cmp.Or(e.displayName, "(no display name)"))
		fmt.Printf("    %s\n", e.resource)
		if e.detail != "" {
			fmt.Printf("    %s\n", e.detail)
		}
	}
	if count == 0 {
		fmt.Println("  (none)")
	}
	return nil
}

func skillNames(skills []agentregistry.Skill) []string {
	names := make([]string, 0, len(skills))
	for _, s := range skills {
		names = append(names, cmp.Or(s.Name, s.ID))
	}
	return names
}

// toolNames returns the tools the registry declares for an MCP server. The live
// tool set is whatever the server reports over MCP once a toolset connects.
func toolNames(tools []agentregistry.Tool) []string {
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		names = append(names, t.Name)
	}
	return names
}

func endpointURL(e *agentregistry.Endpoint) string {
	for _, i := range e.Interfaces {
		if i.URL != "" {
			return "url: " + i.URL
		}
	}
	return ""
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

// summary renders "label (n): a, b, c". The count is always exact, and the tail
// is elided only for a list long enough to bury the resource name above it.
// Returns "" when there is nothing to show.
func summary(label string, items []string) string {
	if len(items) == 0 {
		return ""
	}
	const limit = 8
	line := fmt.Sprintf("%s (%d): ", label, len(items))
	if len(items) <= limit {
		return line + strings.Join(items, ", ")
	}
	return line + strings.Join(items[:limit], ", ") + ", ..."
}
