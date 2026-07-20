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

// Package main shows how to attach credentials to an MCP toolset with the ADK
// auth subsystem, using any of the built-in providers — selected with AUTH_MODE.
//
// Common env (needs a real streamable-HTTP MCP server):
//
//	export GOOGLE_API_KEY=...           # for the Gemini model
//	export MCP_ENDPOINT=https://...     # streamable-HTTP MCP server URL
//	go run ./examples/mcpauth web
//
// AUTH_MODE selects the provider (default "gcp"):
//
//	gcp     GCP_AUTH_RESOURCE=projects/*/locations/*/authProviders|connectors/*
//	        (per-user; supports interactive 3-legged consent via GCP_CONTINUE_URI)
//	apikey    API_KEY=...  [API_KEY_HEADER=X-Api-Key]   # header API key
//	static    BEARER_TOKEN=...                          # static bearer token
//	oauth2cc  OAUTH_CLIENT_ID=... OAUTH_CLIENT_SECRET=... OAUTH_TOKEN_URL=... [OAUTH_SCOPES="a b"]
//	          # 2-legged OAuth (client credentials)
//	adc       (Application Default Credentials; e.g. *.googleapis.com servers)
//	sa        [SA_KEY_FILE=/path/key.json] (else ADC)   # service account
//
// The GCP interactive consent flow only completes against a real resource; the
// code is otherwise a faithful, buildable template for wiring auth into mcptoolset.
package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"strings"

	"golang.org/x/oauth2/clientcredentials"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/auth/gcp"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/mcptoolset"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	model, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	provider := providerFromEnv(ctx)

	// Setting Config.Auth makes mcptoolset own transport construction: it builds
	// a streamable-HTTP transport to Endpoint whose client applies the resolved
	// credential to every request via a context-aware RoundTripper.
	mcpToolSet, err := mcptoolset.New(mcptoolset.Config{
		Endpoint: mustEnv("MCP_ENDPOINT"),
		Auth:     provider,
	})
	if err != nil {
		log.Fatalf("Failed to create MCP tool set: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "auth_mcp_agent",
		Model:       model,
		Description: "Agent that calls an authenticated MCP server.",
		Instruction: "You are a helpful assistant. Use the available MCP tools to answer.",
		Toolsets:    []tool.Toolset{mcpToolSet},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// How auth resolves at run time:
	//
	//   - Non-interactive (API key, static bearer, ADC, service account, GCP
	//     server-side): the provider mints a credential and the transport attaches
	//     it. Nothing surfaces to the user.
	//
	//   - Interactive (GCP 3-legged): if the resource requires user consent, the
	//     tool pauses and ADK emits an "adk_request_credential" function call (see
	//     tool/authconsent) carrying the consent URL. The client — ADK Web, or the
	//     "web"/console launcher below — presents the URL; once the user consents,
	//     ADK resumes the original tool call automatically.
	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),
	}
	l := full.NewLauncher()
	if err := l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

// providerFromEnv builds the auth.CredentialProvider selected by AUTH_MODE,
// illustrating how each built-in provider wires into mcptoolset.Config.Auth. A
// missing or invalid setting is fatal, matching the rest of this example main.
func providerFromEnv(ctx context.Context) auth.CredentialProvider {
	const cloudPlatform = "https://www.googleapis.com/auth/cloud-platform"

	switch strings.ToLower(os.Getenv("AUTH_MODE")) {
	case "", "gcp":
		// GCP Agent Identity / IAM Connector: per-user, and the only mode that can
		// drive interactive 3-legged consent (via ContinueURI). The backing service
		// is chosen from the resource name (authProviders/* vs connectors/*).
		p, err := gcp.NewProvider(gcp.Scheme{
			Name:        mustEnv("GCP_AUTH_RESOURCE"),
			Scopes:      []string{cloudPlatform},
			ContinueURI: os.Getenv("GCP_CONTINUE_URI"), // 3-legged only; empty otherwise
		})
		if err != nil {
			log.Fatalf("create GCP provider: %v", err)
		}
		return p
	case "apikey":
		return auth.APIKey(envOr("API_KEY_HEADER", "X-Api-Key"), mustEnv("API_KEY"))
	case "static":
		return auth.StaticToken(mustEnv("BEARER_TOKEN"))
	case "oauth2cc":
		// 2-legged OAuth (client credentials): the token endpoint issues a token
		// from the client id/secret; TokenSource caches and refreshes it.
		cc := clientcredentials.Config{
			ClientID:     mustEnv("OAUTH_CLIENT_ID"),
			ClientSecret: mustEnv("OAUTH_CLIENT_SECRET"),
			TokenURL:     mustEnv("OAUTH_TOKEN_URL"),
			Scopes:       strings.Fields(os.Getenv("OAUTH_SCOPES")),
		}
		return auth.TokenSourceProvider(cc.TokenSource(ctx))
	case "adc":
		return auth.ADC(cloudPlatform)
	case "sa":
		// From an explicit JSON key file, or Application Default Credentials when
		// SA_KEY_FILE is unset.
		var key []byte
		if f := os.Getenv("SA_KEY_FILE"); f != "" {
			b, err := os.ReadFile(f)
			if err != nil {
				log.Fatalf("read SA_KEY_FILE: %v", err)
			}
			key = b
		}
		return auth.ServiceAccount(auth.ServiceAccountConfig{JSONKey: key, Scopes: []string{cloudPlatform}})
	default:
		log.Fatalf("unknown AUTH_MODE %q (want gcp|apikey|static|oauth2cc|adc|sa)", os.Getenv("AUTH_MODE"))
		return nil // unreachable: log.Fatalf exits the process
	}
}

func mustEnv(name string) string {
	v := os.Getenv(name)
	if v == "" {
		log.Fatalf("environment variable %s must be set", name)
	}
	return v
}

func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}
