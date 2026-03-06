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

package main

import (
	"context"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/internal/configurable"
	"google.golang.org/adk/internal/configurable/conformance"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
)

func main() {
	// 1. Get the Current Working Directory (where the user typed 'adk')
	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Error getting current directory: %v", err)
	}

	// Register callbacks for the conformance agents
	err = conformance.RegisterCallbacks()
	if err != nil {
		log.Fatalf("Error registering callbacks: %v", err)
	}
	err = conformance.RegisterFunctions()
	if err != nil {
		log.Fatalf("Error registering functions: %v", err)
	}

	fmt.Printf("🔍 Scanning for 'root_agent.yaml' in: %s\n", cwd)

	// 2. Crawl folder structure to find all configs
	var agentConfigs []string

	err = filepath.WalkDir(cwd, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			// Report error but continue walking other files
			fmt.Printf("Warning: skipping %q due to error: %v\n", path, err)
			return nil
		}

		// Check if it matches the filename we are looking for
		if !d.IsDir() && d.Name() == "root_agent.yaml" {
			agentConfigs = append(agentConfigs, path)
		}
		return nil
	})
	if err != nil {
		log.Fatalf("Error walking the path: %v", err)
	}

	// 3. Check if we found anything
	if len(agentConfigs) == 0 {
		fmt.Printf("❌ No 'root_agent.yaml' files found in %s or subdirectories\n", cwd)
		os.Exit(1)
	}

	fmt.Printf("🚀 Found %d agent config(s)\n", len(agentConfigs))
	agentsMap := make(map[string]agent.Agent, len(agentConfigs))
	// 4. Iterate and Load all agents found
	for _, configPath := range agentConfigs {
		fmt.Printf("➡️  Loading agent from: %s\n", configPath)

		// This reads the YAML, finds the 'agent_class', and calls the registered factory.
		myAgent, err := configurable.FromConfig(context.Background(), configPath)
		if err != nil {
			log.Printf("⚠️  Error loading agent at %s: %v", configPath, err)
			continue // Skip this one and try the next
		}
		fmt.Printf("✅ Agent loaded successfully: %s\n", myAgent.Name())

		folderName := filepath.Base(filepath.Dir(configPath))
		fmt.Printf("✅ Agent folder name: %s\n", folderName)

		if _, ok := agentsMap[folderName]; ok {
			log.Printf("⚠️  Agent %s already exists, skipping", folderName)
			continue
		}
		agentsMap[folderName] = myAgent
	}

	ctx := context.Background()

	loader, err := conformance.NewConformanceAgentLoader(agentsMap)
	if err != nil {
		log.Fatalf("Error loading agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: loader,
		PluginConfig: runner.PluginConfig{
			Plugins: []*plugin.Plugin{
				// TODO: Add replay plugin
				// replayplugin.MustNew(),
			},
		},
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
