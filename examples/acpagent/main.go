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

// Package main runs a coding agent harness as an ADK agent and gives it a tool
// written in Go.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/exp/acpagent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type ticketArgs struct {
	Title string `json:"title" jsonschema:"one-line summary of the work"`
	Body  string `json:"body" jsonschema:"what to do and where, with file paths"`
}

type ticketResult struct {
	ID string `json:"id"`
}

// fileTicket records a ticket in the session state. A real tool would call an
// issue tracker; the state shows that the harness's call ran inside the ADK
// session.
func fileTicket(ctx agent.Context, args ticketArgs) (ticketResult, error) {
	tickets, _ := ctx.State().Get("tickets")
	list, _ := tickets.([]any)
	id := fmt.Sprintf("T-%d", len(list)+1)
	list = append(list, map[string]any{"id": id, "title": args.Title, "body": args.Body})
	if err := ctx.State().Set("tickets", list); err != nil {
		return ticketResult{}, err
	}
	return ticketResult{ID: id}, nil
}

func run(ctx context.Context) error {
	command := strings.Fields(os.Getenv("ACP_COMMAND"))
	if len(command) == 0 {
		command = []string{"opencode", "acp"}
	}

	ticket, err := functiontool.New(functiontool.Config{
		Name:        "file_ticket",
		Description: "Files a ticket for follow-up work and returns its ID.",
	}, fileTicket)
	if err != nil {
		return fmt.Errorf("creating the tool: %w", err)
	}

	harness, closer, err := acpagent.New(ctx, acpagent.Config{
		Name:        "coder",
		Description: "Reads the code in the current directory and files tickets.",
		Command:     command,
		Stderr:      os.Stderr,
		// Kind "other" lets a harness which asks before calling an MCP tool
		// call file_ticket without stopping.
		AllowTools: []string{"read", "search", "think", "other"},
		Tools:      []tool.Tool{ticket},
	})
	if err != nil {
		return fmt.Errorf("starting the harness: %w", err)
	}
	defer func() {
		if err := closer.Close(); err != nil {
			log.Printf("Stopping the harness: %v", err)
		}
	}()

	config := &launcher.Config{AgentLoader: agent.NewSingleLoader(harness)}
	l := full.NewLauncher()
	if err := l.Execute(ctx, config, os.Args[1:]); err != nil {
		return fmt.Errorf("%w\n\n%s", err, l.CommandLineSyntax())
	}
	return nil
}

func main() {
	log.SetFlags(0)
	if err := run(context.Background()); err != nil {
		log.Fatal(err)
	}
}
