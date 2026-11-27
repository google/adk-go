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

// Command skillregistry is an example agent whose skills are served from the
// Skill Registry of the Gemini Enterprise Agent Platform.
//
// It reads skills from a real project, so it needs Google Cloud credentials and
// three environment variables:
//
//	GOOGLE_CLOUD_PROJECT   project that owns the Skill Registry
//	GOOGLE_CLOUD_LOCATION  region hosting it, e.g. "global" or "us-central1"
//	GOOGLE_API_KEY         API key for the Gemini model
//
// Authenticate with Application Default Credentials first, then run it from the
// repository root:
//
//	gcloud auth application-default login
//	GOOGLE_CLOUD_PROJECT=my-proj GOOGLE_CLOUD_LOCATION=global \
//	  GOOGLE_API_KEY=... go run ./examples/skillregistry console
//
// The source is wrapped in simplecache so repeated reads are served from memory
// and refreshed in the background.
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/skilltoolset"
	"google.golang.org/adk/v2/tool/skilltoolset/skill/skillregistry"
	"google.golang.org/adk/v2/tool/skilltoolset/skill/skillregistry/simplecache"
)

func main() {
	ctx := context.Background()

	projectID := mustEnv("GOOGLE_CLOUD_PROJECT")
	location := mustEnv("GOOGLE_CLOUD_LOCATION")

	// A low-level client for the Skill Registry. Credentials come from the
	// environment (Application Default Credentials).
	client, err := skillregistry.NewClient(ctx, skillregistry.Config{
		ProjectID: projectID,
		Location:  location,
	})
	if err != nil {
		log.Fatalf("Failed to create Skill Registry client: %v", err)
	}

	// Wrap the client in an in-memory cache: the initial WarmUp blocks until the
	// skill list and frontmatters are loaded, then a background goroutine keeps
	// them fresh. StopAutorefresh releases that goroutine on exit.
	cache, err := simplecache.NewCache(simplecache.CacheConfig{Client: client})
	if err != nil {
		log.Fatalf("Failed to create skill cache: %v", err)
	}

	source, err := simplecache.NewCachedSkillRegistrySource(ctx, cache)
	if err != nil {
		log.Fatalf("Failed to create skill source: %v", err)
	}

	// A quick connectivity check: report how many skills the registry served.
	frontmatters, err := source.ListFrontmatters(ctx)
	if err != nil {
		log.Fatalf("Failed to list skills: %v", err)
	}
	log.Printf("Loaded %d skill(s) from the Skill Registry", len(frontmatters))

	model, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	skillToolset, err := skilltoolset.New(ctx, skilltoolset.Config{Source: source})
	if err != nil {
		log.Fatalf("Failed to create skill toolset: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "skills_agent",
		Model:       model,
		Description: "Agent to demonstrate using skills from the Skill Registry.",
		Instruction: "You are a helpful assistant. Use the available skills when they fit the task.",
		Toolsets:    []tool.Toolset{skillToolset},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),
	}

	l := full.NewLauncher()
	if err := l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

// mustEnv returns the value of the named environment variable or exits if it is
// unset.
func mustEnv(name string) string {
	v := os.Getenv(name)
	if v == "" {
		log.Fatalf("Environment variable %s is not set", name)
	}
	return v
}
