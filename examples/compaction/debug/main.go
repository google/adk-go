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

// Debug example that shows exactly when compaction happens.
// This example provides visual markers to help verify compaction is triggered.

package main

import (
	"context"
	"flag"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/compaction"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	// Command-line flags
	interval := flag.Int("interval", 2, "Compaction interval (number of turns)")
	overlap := flag.Int("overlap", 1, "Number of turns to overlap in compaction")
	verbose := flag.Bool("verbose", true, "Enable verbose logging")

	// Parse flags - separate from launcher args
	flag.CommandLine.Parse(os.Args[1:])
	launcherArgs := flag.Args()

	log.Println()
	log.Println("╔════════════════════════════════════════════════════════════╗")
	log.Println("║ Session History Compaction - DEBUG Example                ║")
	log.Println("║                                                            ║")
	log.Println("║ This example shows WHEN and HOW compaction is triggered    ║")
	log.Println("╚════════════════════════════════════════════════════════════╝")
	log.Println()

	if *verbose {
		log.Printf("[CONFIG] Compaction Interval: %d turns\n", *interval)
		log.Printf("[CONFIG] Overlap Size: %d turns\n", *overlap)
		log.Println()
		log.Println("[GUIDE] What to expect:")
		log.Println("  1. Type messages to have a conversation")
		log.Println("  2. Watch for '🎯 COMPACTION' markers")
		log.Println("  3. Compaction happens automatically in background")
		log.Println("  4. Each turn shows the expected next compaction point")
		log.Println()
	}

	// Create model
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Create agent
	a, err := llmagent.New(llmagent.Config{
		Name:  "debug-agent",
		Model: model,
		Instruction: `You are a helpful AI assistant.
Provide concise, informative responses.
Maintain context from the conversation history.`,
		Tools: []tool.Tool{
			geminitool.GoogleSearch{},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Create session service
	sessionService := session.InMemoryService()

	// Configure compaction
	compactionConfig := &compaction.Config{
		Enabled:            true,
		CompactionInterval: *interval,
		OverlapSize:        *overlap,
		Model:              "",
		SystemPrompt:       "Summarize the conversation history concisely.",
	}

	if err := compactionConfig.Validate(); err != nil {
		log.Fatalf("Invalid compaction config: %v", err)
	}

	if *verbose {
		log.Println("[INFO] Compaction configuration validated ✓")
		log.Println()
		log.Println("=== How Compaction Works ===")
		log.Printf("  After turn %d: 1st compaction\n", *interval)
		log.Printf("  After turn %d: 2nd compaction\n", *interval*2)
		log.Printf("  And so on every %d turns...\n", *interval)
		log.Println()
		log.Println("Type 'quit' to exit, or start a conversation:")
		log.Println()
	}

	// Create launcher config
	config := &launcher.Config{
		AgentLoader:      agent.NewSingleLoader(a),
		SessionService:   sessionService,
		CompactionConfig: compactionConfig,
	}

	// Run launcher
	l := full.NewLauncher()
	if err = l.Execute(ctx, config, launcherArgs); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
