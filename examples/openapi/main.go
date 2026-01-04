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

// Example demonstrating OpenAPIToolset with GitHub API authentication including OAuth2 flow.
//
// This example creates an agent that can:
// - List authenticated user's repositories
// - Get repository details
// - List issues for a repository
//
// Usage (Bearer Token - simplest):
//
//	export GITHUB_TOKEN=your_personal_access_token
//	export GOOGLE_API_KEY=your_google_api_key
//	go run main.go
//
// Usage (OAuth2 Authorization Code Flow - default):
//
//  1. Create GitHub OAuth App at: https://github.com/settings/developers
//  2. Set Authorization callback URL to: http://localhost:8080/callback
//  3. Run:
//     export GITHUB_CLIENT_ID=your_oauth_app_client_id
//     export GITHUB_CLIENT_SECRET=your_secret
//     export GOOGLE_API_KEY=your_key
//     go run main.go
//
// Usage (Custom Port):
//
//	go run main.go --oauth-port 3000
//
// Usage (OAuth2 Device Flow - when supported):
//
//	export GITHUB_CLIENT_ID=your_oauth_app_client_id
//	export GITHUB_CLIENT_SECRET=your_oauth_app_client_secret
//	export GOOGLE_API_KEY=your_google_api_key
//	go run main.go --oauth-device-flow
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/auth"
	"google.golang.org/adk/examples/openapi/oauth2handler"
	"google.golang.org/adk/examples/openapi/support"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/openapitoolset"
	"google.golang.org/genai"
)

// Command line flags
var (
	useDeviceFlow = flag.Bool("oauth-device-flow", false, "Use Device Authorization Flow instead of Authorization Code Flow (not all providers support this)")
	oauthPort     = flag.Int("oauth-port", 8777, "Port for OAuth2 callback server (default: 8080)")
)

// Simplified GitHub OpenAPI spec for common operations.
const githubOpenAPISpec = `
openapi: "3.0.0"
info:
  title: GitHub API
  version: "1.0"
servers:
  - url: https://api.github.com
paths:
  /user:
    get:
      operationId: get_authenticated_user
      summary: Get the authenticated user
      description: Returns the authenticated user's profile information.
      responses:
        "200":
          description: Successful response

  /user/repos:
    get:
      operationId: list_user_repos
      summary: List repositories for the authenticated user
      description: Lists repositories that the authenticated user has access to.
      parameters:
        - name: sort
          in: query
          description: "Sort field: created, updated, pushed, full_name"
          schema:
            type: string
        - name: per_page
          in: query
          description: Number of results per page
          schema:
            type: integer
      responses:
        "200":
          description: Successful response

  /repos/{owner}/{repo}:
    get:
      operationId: get_repo
      summary: Get a repository
      description: Gets information about a specific repository.
      parameters:
        - name: owner
          in: path
          required: true
          schema:
            type: string
        - name: repo
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Successful response

  /repos/{owner}/{repo}/issues:
    get:
      operationId: list_repo_issues
      summary: List issues for a repository
      description: List issues in a repository.
      parameters:
        - name: owner
          in: path
          required: true
          schema:
            type: string
        - name: repo
          in: path
          required: true
          schema:
            type: string
        - name: state
          in: query
          description: "Issue state: open, closed, all"
          schema:
            type: string
            default: open
        - name: per_page
          in: query
          description: Number of results per page
          schema:
            type: integer
      responses:
        "200":
          description: Successful response
`

func main() {
	flag.Parse()

	ctx := context.Background()

	// Auto-detect auth mode based on environment variables
	var authScheme auth.AuthScheme
	var authCredential *auth.AuthCredential

	githubToken := os.Getenv("GITHUB_TOKEN")
	if githubToken != "" {
		// Use Bearer token auth (preferred, simpler)
		fmt.Println("Using Bearer token authentication (GITHUB_TOKEN)")
		authScheme, authCredential = auth.BearerTokenCredential(githubToken)
	} else {
		// Use OAuth2 auth
		clientID := os.Getenv("GITHUB_CLIENT_ID")
		clientSecret := os.Getenv("GITHUB_CLIENT_SECRET")
		if clientID == "" || clientSecret == "" {
			log.Fatal("Either GITHUB_TOKEN or both GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET must be set")
		}

		flowType := "Authorization Code"
		if *useDeviceFlow {
			flowType = "Device"
		}
		fmt.Printf("Using OAuth2 %s authentication (GITHUB_CLIENT_ID/GITHUB_CLIENT_SECRET)\n", flowType)

		// GitHub OAuth2 endpoints
		authScheme, authCredential = auth.OAuth2AuthorizationCode(
			clientID,
			clientSecret,
			"https://github.com/login/oauth/authorize",
			"https://github.com/login/oauth/access_token",
			map[string]string{
				"repo":      "Full control of private repositories",
				"read:user": "Read access to profile info",
			},
		)
	}

	// Create OpenAPI toolset with the GitHub spec
	githubToolset, err := openapitoolset.New(openapitoolset.Config{
		SpecStr:        githubOpenAPISpec,
		SpecStrType:    "yaml",
		AuthScheme:     authScheme,
		AuthCredential: authCredential,
		ToolNamePrefix: "github_",
	})
	if err != nil {
		log.Fatalf("Failed to create GitHub toolset: %v", err)
	}

	// List available tools
	tools, err := githubToolset.Tools(nil)
	if err != nil {
		log.Fatalf("Failed to get tools: %v", err)
	}

	fmt.Println("Available GitHub API tools:")
	for _, t := range tools {
		fmt.Printf("  - %s: %s\n", t.Name(), t.Description())
	}
	fmt.Println()

	// Create the model
	model, err := gemini.NewModel(ctx, "gemini-2.0-flash-exp", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Create the agent with GitHub tools
	a, err := llmagent.New(llmagent.Config{
		Name:        "github_assistant",
		Description: "An assistant that can interact with GitHub API",
		Model:       model,
		Instruction: `You are a helpful GitHub assistant. You can:
- Get information about the authenticated user
- List and get repository information
- List issues for repositories

When asked about repositories, provide helpful summaries of the information.`,
		Toolsets: []tool.Toolset{githubToolset},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Create services
	sessionService := session.InMemoryService()
	artifactService := artifact.InMemoryService()

	// Create runner
	// Create OAuth2 handler first (needed for runner config)
	var flowType oauth2handler.FlowType
	if *useDeviceFlow {
		flowType = oauth2handler.FlowTypeDevice
	} else {
		flowType = oauth2handler.FlowTypeAuthCode
	}
	oauth2Handler := oauth2handler.New(flowType, *oauthPort, "/callback")
	defer oauth2Handler.Close()

	// Create runner
	r, err := runner.New(runner.Config{
		AppName:         "github_example",
		Agent:           a,
		SessionService:  sessionService,
		ArtifactService: artifactService,
	})

	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	fmt.Printf("\nOAuth2 Configuration:\n")
	fmt.Printf("  Callback URL: http://localhost:%d/callback\n", *oauthPort)
	fmt.Printf("  Configure this URL in your OAuth provider's settings\n\n")

	// Run interactive loop
	if err := support.RunInteractive(
		ctx,
		"github_example",
		"user123",
		"GitHub Assistant (type 'quit' to exit)",
		r,
		sessionService,
		oauth2Handler,
	); err != nil {
		log.Fatal(err)
	}
}
