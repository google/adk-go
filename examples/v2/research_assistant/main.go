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

// research_assistant demonstrates dynamic tool loading via the
// toolregistry. The agent ships with only list_tools and load_tool
// declared on the LLM request; the LLM discovers and activates the
// research-specific tools it needs as the conversation proceeds.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/toolregistry"
)

func main() {
	reg := toolregistry.New()

	// Register a small library of research tools. In production these
	// would wrap real services — search engines, vector stores, the
	// fetch/scrape API, etc.
	must(reg.RegisterTool(mustTool(searchTool()), toolregistry.Info{
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

	// In an actual agent, the toolset goes into llmagent.Config.Toolsets:
	//
	//   reg := toolregistry.New()
	//   ...register tools...
	//   ts := toolregistry.NewToolset(reg)
	//   agent, _ := llmagent.New(llmagent.Config{
	//       Model: gemini, Toolsets: []tool.Toolset{ts}, ...})
	//
	// Here we exercise the toolset directly so the demo is offline.
	ts := toolregistry.NewToolset(reg)

	// Initial state: nothing loaded. Toolset surfaces only list_tools +
	// load_tool. The LLM sees a tiny set, not the full library.
	rctx := newStubReadonlyContext(map[string]any{})
	tools, _ := ts.Tools(rctx)
	fmt.Println("=== turn 1: agent boot, only discovery tools active ===")
	printTools(tools)

	// LLM calls list_tools(query="search"). We invoke the catalog
	// directly to show what the model sees.
	fmt.Println("\n=== turn 2: list_tools(query=\"search\") ===")
	infos := reg.List(toolregistry.Filter{Query: "search"})
	for _, i := range infos {
		fmt.Printf("  %s — %s [%v]\n", i.Name, i.Description, i.Tags)
	}

	// LLM picks web_search and calls load_tool("web_search"). The
	// LoadTool handler writes it to session state.
	fmt.Println("\n=== turn 3: load_tool(\"web_search\") ===")
	state := newStubState(map[string]any{})
	state.Set(toolregistry.StateKeyLoadedTools, []string{"web_search"})

	// Next turn: the toolset surfaces web_search alongside the
	// always-on discovery tools.
	rctx = newStubReadonlyContext(state.m)
	tools, _ = ts.Tools(rctx)
	fmt.Println("\n=== turn 4: web_search now active ===")
	printTools(tools)

	// As the agent works, it loads more tools. Compare to the naive
	// upfront-declare approach: the LLM would have paid the
	// FunctionDeclaration cost for all 5 tools every turn.
	state.Set(toolregistry.StateKeyLoadedTools, []string{"web_search", "fetch_url", "summarize"})
	rctx = newStubReadonlyContext(state.m)
	tools, _ = ts.Tools(rctx)
	fmt.Println("\n=== turn 5: agent loaded fetch_url + summarize for the actual work ===")
	printTools(tools)

	fmt.Println("\nKey property: the agent never had to reason about citation_check or")
	fmt.Println("save_note for this query — they stayed dormant in the registry.")
}

// ---------- tool implementations (stubs) ----------

func searchTool() (tool.Tool, error) {
	type args struct {
		Query string `json:"query"`
	}
	type result struct {
		Hits []map[string]string `json:"hits"`
	}
	return functiontool.New[args, result](
		functiontool.Config{Name: "web_search", Description: "Search the web."},
		func(_ tool.Context, a args) (result, error) {
			return result{Hits: []map[string]string{
				{"title": "ADK research overview", "url": "https://example.com/adk"},
			}}, nil
		},
	)
}

func fetchURLTool() (tool.Tool, error) {
	type args struct {
		URL string `json:"url"`
	}
	return functiontool.New[args, string](
		functiontool.Config{Name: "fetch_url", Description: "Fetch URL contents."},
		func(_ tool.Context, a args) (string, error) {
			return "Article body for " + a.URL, nil
		},
	)
}

func summarizeTool() (tool.Tool, error) {
	type args struct {
		Text string `json:"text"`
	}
	return functiontool.New[args, []string](
		functiontool.Config{Name: "summarize", Description: "Summarize text."},
		func(_ tool.Context, a args) ([]string, error) {
			return []string{"point 1", "point 2", "point 3"}, nil
		},
	)
}

func citationCheckTool() (tool.Tool, error) {
	type args struct {
		Quote string `json:"quote"`
		URL   string `json:"url"`
	}
	return functiontool.New[args, bool](
		functiontool.Config{Name: "citation_check", Description: "Verify quote."},
		func(_ tool.Context, _ args) (bool, error) { return true, nil },
	)
}

func saveNoteTool() (tool.Tool, error) {
	type args struct {
		Note string `json:"note"`
	}
	return functiontool.New[args, string](
		functiontool.Config{Name: "save_note", Description: "Save a note."},
		func(_ tool.Context, _ args) (string, error) { return "saved", nil },
	)
}

// ---------- helpers ----------

func mustTool(t tool.Tool, err error) tool.Tool {
	if err != nil {
		log.Fatalf("tool build: %v", err)
	}
	return t
}
func must(err error) {
	if err != nil {
		log.Fatalf("%v", err)
	}
}
func printTools(tools []tool.Tool) {
	for _, t := range tools {
		fmt.Printf("  - %-15s %s\n", t.Name(), t.Description())
	}
}

var _ = context.Background
var _ = errors.New
