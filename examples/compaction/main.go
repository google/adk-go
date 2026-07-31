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

// Package main provides an example ADK agent with context compaction enabled.
//
// Compaction keeps an agent's prompt small as its conversation grows: older
// turns are summarized into a single event, and later prompts carry that
// summary instead of the raw turns. Two triggers are available, and this
// example arms both:
//
//   - Sliding window fires after every CompactionInterval completed turns.
//   - Tail retention fires mid-turn, once a prompt reaches TokenThreshold.
//
// Setting EventsCompactionConfig on [launcher.Config] enables compaction for
// every surface the launcher serves: the console, the web UI, A2A and Agent
// Engine.
//
// Run it and hold a conversation of several turns:
//
//	GOOGLE_API_KEY=... go run ./examples/compaction console
//	GOOGLE_API_KEY=... go run ./examples/compaction web
//
// After every two turns a compaction event is appended to the session, and the
// turns it covers stop being sent to the model. The summary is bookkeeping
// rather than conversation, so it is not streamed back with the agent's reply;
// look for it in the session's event list, for example in the web UI.
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
	"google.golang.org/adk/v2/session/compaction"
)

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Model:       model,
		Description: "A general purpose assistant with a long memory.",
		Instruction: "You are a helpful assistant. Keep your answers to a sentence or two.",
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),

		// Summarizer is left nil, so the runner summarizes with the root
		// agent's own model.
		EventsCompactionConfig: &compaction.Config{
			// Sliding window: after every 2 completed turns, summarize the
			// turns since the last compaction, carrying 1 earlier turn forward
			// so consecutive summaries overlap and context is not lost at the
			// seam. Runs once a turn has finished.
			CompactionInterval: 2,
			OverlapSize:        1,

			// Tail retention: if a prompt ever reaches 32k tokens, summarize
			// everything but the 10 most recent events before the next model
			// call. This runs *during* a turn, so it also catches a single
			// long tool-calling turn that inflates the prompt on its own.
			//
			// The two triggers are independent; either alone is a valid setup.
			TokenThreshold:     32_000,
			EventRetentionSize: 10,
		},
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
