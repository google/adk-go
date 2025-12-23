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

// Package provides an example ADK agent that uses BigQuery via MCP.
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"golang.org/x/oauth2"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/mcptoolset"
)

// In order to run this example you need to:
//    - set the environment variable `GCP_PROJECT_ID`
//    - set the environment variable `GOOGLE_API_KEY`
//    - set the environment variable `TEST_TOKEN`   (you can use command like "export TEST_TOKEN=$(gcloud auth print-access-token)")
//    - ensure you have enabled "BigQuery API" (bigquery.googleapis.com) for project indicated in `GCP_PROJECT_ID`
//	  - ensure you have access to the project indidated in `GCP_PROJECT_ID`
// You can try prompt like:
//    `select server date using googlesql from project ` + value of `GCP_PROJECT_ID`

// TransportWithHeaders adds "X-Goog-User-Project" header to BigQuery MCP calls
type TransportWithHeaders struct {
	parent  http.RoundTripper
	project string // GCP Project ID
}

func (t *TransportWithHeaders) RoundTrip(req *http.Request) (*http.Response, error) {
	if t.project != "" {
		req = req.Clone(req.Context())
		req.Header.Add("X-Goog-User-Project", t.project)
	}
	return t.parent.RoundTrip(req)
}

func main() {
	project := os.Getenv("GCP_PROJECT_ID")
	apiKey := os.Getenv("GOOGLE_API_KEY")
	token := os.Getenv("TEST_TOKEN")

	// Create context that cancels on interrupt signal (Ctrl+C)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	oauthClient := oauth2.NewClient(ctx, oauth2.StaticTokenSource(
		&oauth2.Token{AccessToken: token},
	))

	transport := &mcp.StreamableClientTransport{
		Endpoint: "https://bigquery.googleapis.com/mcp",

		HTTPClient: &http.Client{
			Transport: &TransportWithHeaders{parent: oauthClient.Transport, project: project},
		},
	}
	mcpToolSet, err := mcptoolset.New(mcptoolset.Config{
		Transport: transport,
		// WARNING: we need to filter out "get_table_info" tool because of errors is causes ("reference to undefined schema at $defs.RangePartitioning.properties.range")
		// we need now just 'execute_sql' tool
		ToolFilter: tool.StringPredicate([]string{"execute_sql"}),
	})
	if err != nil {
		log.Fatalf("Failed to create MCP tool set: %v", err)
	}

	// Create LLMAgent with MCP tool set
	a, err := llmagent.New(llmagent.Config{
		Name:        "helper_agent",
		Model:       model,
		Description: "Helper agent.",
		Instruction: "You are a helpful assistant that helps users with various tasks.",
		Toolsets: []tool.Toolset{
			mcpToolSet,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),
	}
	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
