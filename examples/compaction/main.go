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
// example arms the sliding window:
//
//   - Sliding window fires after every CompactionInterval completed turns. It
//     replaces each group of turns with one summary, a constant-factor
//     reduction. Summaries are never re-summarized, so the prompt still grows
//     with the length of the conversation.
//   - Tail retention fires mid-turn once a prompt reaches TokenThreshold, and
//     keeps one rolling summary plus the most recent events. This is the
//     trigger that puts a ceiling on prompt size.
//
// Arm one or the other, not both. The sliding window summarizes the events tail
// retention would have worked on, so with both enabled tail retention never
// finds enough uncovered events to fire and the ceiling never applies. The
// commented pair below swaps this example over to it.
//
// Setting EventsCompactionConfig on [launcher.Config] enables compaction on
// every surface that reads that config. The launcher used here, full.NewLauncher,
// serves the console, the web UI, A2A, the Pub/Sub and Eventarc triggers, and
// the REST API. Agent Engine reads the same field but is served by its own
// handler rather than by this launcher.
//
// Run it and hold a conversation of several turns:
//
//	GOOGLE_API_KEY=... go run ./examples/compaction console
//	GOOGLE_API_KEY=... go run ./examples/compaction web webui
//
// The web command needs at least one sublauncher named after it, as above.
// Running it bare exits with "no active sublaunchers found".
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

			// Tail retention, commented out on purpose. Enabling it here would
			// do nothing: the sliding window above summarizes every group of
			// two turns, so the events this trigger looks at, the ones no
			// compaction covers yet, never reach EventRetentionSize and it
			// never fires.
			//
			// To try it, delete the two sliding-window settings above and
			// uncomment these. It runs *during* a turn, so it also catches a
			// single long tool-calling turn that inflates the prompt on its
			// own, and it is the trigger that bounds prompt size rather than
			// just reducing it.
			//
			//	TokenThreshold:     32_000,
			//	EventRetentionSize: 10,
		},
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
