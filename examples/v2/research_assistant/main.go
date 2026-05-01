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

// Research assistant agent. Demonstrates toolregistry: only list_tools
// and load_tool are exposed to the LLM upfront. The model discovers
// what's available, loads what it needs, then uses the loaded tool on
// the next turn. Keeps LLM context lean for tool-heavy domains.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/toolregistry"
)

const instruction = `You are a research assistant.

You have access to a dynamic registry of research tools. The registry is
much larger than what you should pull into context at once.

Workflow on EVERY task:

  1. Call list_tools(query="<keyword>") to discover what is available
     for this task. Use the user's intent to pick the keyword.
  2. Call load_tool(name="<name>") to activate the tool you need. The
     tool will be available for use starting on your next turn.
  3. Use the loaded tool to answer.

If you already loaded a tool earlier in this conversation, you can use
it directly without re-loading.

Do NOT speculate about a tool that wasn't returned by list_tools. If
nothing matches, say so plainly.`

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, modelName(), &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	reg := buildRegistry()
	ts := toolregistry.NewToolset(reg)

	a, err := llmagent.New(llmagent.Config{
		Name:        "research_assistant",
		Model:       model,
		Description: "A research assistant that discovers and loads tools on demand.",
		Instruction: instruction,
		Toolsets:    []tool.Toolset{ts},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	cfg := &launcher.Config{AgentLoader: agent.NewSingleLoader(a)}
	l := full.NewLauncher()
	if err = l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

func modelName() string {
	if v := os.Getenv("GOOGLE_GENAI_MODEL"); v != "" {
		return v
	}
	return "gemini-2.5-flash"
}

// buildRegistry registers a small library of research tools. Replace
// each tool body with a real implementation (search engine, vector
// store, etc.) when adapting this for production.
func buildRegistry() *toolregistry.Registry {
	reg := toolregistry.New()
	must(reg.RegisterTool(mustTool(webSearchTool()), toolregistry.Info{
		Name:        "web_search",
		Description: "Search the web. Returns up to 5 result titles + URLs.",
		Tags:        []string{"web", "discovery"},
		Hints:       "Use when the user asks for general information or recent news.",
	}))
	must(reg.RegisterTool(mustTool(fetchURLTool()), toolregistry.Info{
		Name:        "fetch_url",
		Description: "Fetch a URL and return its plain-text content.",
		Tags:        []string{"web", "content"},
		Hints:       "Use after web_search to read a specific result in full.",
	}))
	must(reg.RegisterTool(mustTool(summarizeTool()), toolregistry.Info{
		Name:        "summarize",
		Description: "Summarize a long piece of text into 3-5 bullet points.",
		Tags:        []string{"text"},
		Hints:       "Use to condense fetched articles before drafting.",
	}))
	must(reg.RegisterTool(mustTool(citationCheckTool()), toolregistry.Info{
		Name:        "citation_check",
		Description: "Verify that a quoted claim appears verbatim in the cited source.",
		Tags:        []string{"text", "fact-check"},
		Hints:       "Use before publishing a draft that contains direct quotes.",
	}))
	must(reg.RegisterTool(mustTool(saveNoteTool()), toolregistry.Info{
		Name:        "save_note",
		Description: "Save a short note to the researcher's notebook.",
		Tags:        []string{"notes"},
	}))
	return reg
}

// ---- tool implementations (stubs ready to be swapped for real ones) ----

func webSearchTool() (tool.Tool, error) {
	type args struct {
		Query string `json:"query"`
	}
	type hit struct {
		Title string `json:"title"`
		URL   string `json:"url"`
	}
	type result struct {
		Hits []hit `json:"hits"`
	}
	return functiontool.New[args, result](
		functiontool.Config{Name: "web_search", Description: "Search the web."},
		func(_ tool.Context, a args) (result, error) {
			return result{Hits: []hit{
				{Title: "ADK overview", URL: "https://example.com/adk"},
				{Title: "Building agents with ADK", URL: "https://example.com/adk/agents"},
			}}, nil
		},
	)
}

func fetchURLTool() (tool.Tool, error) {
	type args struct {
		URL string `json:"url"`
	}
	return functiontool.New[args, string](
		functiontool.Config{Name: "fetch_url", Description: "Fetch URL contents as plain text."},
		func(_ tool.Context, a args) (string, error) {
			return fmt.Sprintf("Article body for %s. (replace this stub with a real HTTP fetch)", a.URL), nil
		},
	)
}

func summarizeTool() (tool.Tool, error) {
	type args struct {
		Text string `json:"text"`
	}
	return functiontool.New[args, []string](
		functiontool.Config{Name: "summarize", Description: "Summarize text into 3-5 bullets."},
		func(_ tool.Context, a args) ([]string, error) {
			n := len(a.Text)
			return []string{
				fmt.Sprintf("Source length: %d characters.", n),
				"(stub) Replace with a real summarization call when adapting.",
			}, nil
		},
	)
}

func citationCheckTool() (tool.Tool, error) {
	type args struct {
		Quote string `json:"quote"`
		URL   string `json:"url"`
	}
	type result struct {
		Verified bool   `json:"verified"`
		Note     string `json:"note,omitempty"`
	}
	return functiontool.New[args, result](
		functiontool.Config{Name: "citation_check", Description: "Verify a quoted claim against a source."},
		func(_ tool.Context, _ args) (result, error) {
			return result{Verified: true, Note: "(stub) replace with a real check."}, nil
		},
	)
}

func saveNoteTool() (tool.Tool, error) {
	type args struct {
		Note string `json:"note"`
	}
	return functiontool.New[args, string](
		functiontool.Config{Name: "save_note", Description: "Save a research note."},
		func(_ tool.Context, a args) (string, error) {
			return "saved: " + a.Note, nil
		},
	)
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
func mustTool(t tool.Tool, err error) tool.Tool {
	if err != nil {
		log.Fatalf("tool build: %v", err)
	}
	return t
}
