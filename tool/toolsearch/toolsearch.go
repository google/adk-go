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

// Package toolsearch provides a gating toolset that hides all but a small
// core set of tools from the model until it calls search_tools to discover them.
package toolsearch

import (
	"cmp"
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/ranksearch"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

const (
	// ToolName is the name of the search tool exposed to the model.
	ToolName = "search_tools"

	defaultMaxResults = 8
	stateKeyPrefix    = "tool_search:discovered:"
	maxDescLen        = 200

	// minScoreRatio drops matches scoring below this fraction of the top
	// match's BM25 score, trimming the long tail of weak matches (e.g. tools
	// that only matched a common verb like "list") so the model is not handed
	// near-irrelevant candidates.
	minScoreRatio = 0.1

	toolDescription = "Discover tools that are not yet loaded. Describe your task in natural language, " +
		"use select:tool_a,tool_b for exact names, or a regex such as get_.*_file. " +
		"Natural-language results match tool names, descriptions, and argument metadata. " +
		"Results may include an optional connected_skill hint; load the relevant skill if your application supports it. " +
		"After searching, finish this model response. Call discovered tools in the next model response, not alongside search_tools."
)

// Config tunes the gating toolset.
type Config struct {
	// AgentName namespaces the discovered-set state keys so multiple agents in a
	// session don't share their discovery sets. It must not contain a colon.
	AgentName string
	// CoreToolNames are always advertised alongside search_tools; everything
	// else in the base toolset is gated until the model discovers it.
	CoreToolNames []string
	// GatedToolNames are the names of the tools hidden behind search_tools. When
	// set, they are listed in the search_tools description so the model knows what
	// it can search for (and can select: them by exact name) instead of guessing
	// blind. Names only — no schemas — so it stays cheap and fully static.
	GatedToolNames []string
	// SkillAnnotations maps tool names to skill names so search results carry
	// a connected_skill hint pointing the model at the relevant skill.
	SkillAnnotations map[string]string
	// MaxResults caps BM25 and regex matches, not exact selections.
	// Defaults to 8 when zero or negative.
	MaxResults int
}

// New exposes CoreToolNames up front and gates other tools behind search_tools.
// search_tools is omitted only when every base tool is already core.
func New(base tool.Toolset, cfg Config) (tool.Toolset, error) {
	if strings.Contains(cfg.AgentName, ":") {
		return nil, fmt.Errorf("toolsearch: agent name %q must not contain a colon", cfg.AgentName)
	}
	maxResults := cfg.MaxResults
	if maxResults <= 0 {
		maxResults = defaultMaxResults
	}

	coreNames := nameSet(cfg.CoreToolNames)

	searchTool, err := functiontool.New(
		functiontool.Config{
			Name: ToolName,
			Description: ranksearch.DescribeSearch(
				toolDescription,
				cfg.GatedToolNames,
			),
		},
		func(ctx agent.Context, args searchArgs) (searchOutput, error) {
			return executeSearch(
				ctx, args, base, cfg.AgentName, coreNames, cfg.SkillAnnotations, maxResults,
			)
		},
	)
	if err != nil {
		return nil, fmt.Errorf("toolsearch: build search tool: %w", err)
	}

	return &gatingToolset{
		base:             base,
		coreNames:        coreNames,
		agentName:        cfg.AgentName,
		searchTool:       searchTool,
		skillAnnotations: cfg.SkillAnnotations,
	}, nil
}

// nameSet builds a lookup set from a slice of tool names.
func nameSet(names []string) map[string]bool {
	set := make(map[string]bool, len(names))
	for _, n := range names {
		set[n] = true
	}
	return set
}

// RevealTools adds tools to the same session discovery state used by search_tools.
//
// Each tool is stored under its own key, so reveals from several function calls
// in one model response are all kept: each call writes its own state delta and
// does not see the others'.
func RevealTools(state session.State, agentName string, names ...string) error {
	if strings.Contains(agentName, ":") {
		return fmt.Errorf("toolsearch: agent name %q must not contain a colon", agentName)
	}
	discovered := discoveredNames(state, agentName)
	seen := nameSet(discovered)
	next := len(discovered)
	for _, name := range names {
		if name == "" || seen[name] {
			continue
		}
		if err := state.Set(discoveredKeyPrefix(agentName)+name, next); err != nil {
			return err
		}
		seen[name] = true
		next++
	}
	return nil
}

func discoveredKeyPrefix(agentName string) string { return stateKeyPrefix + agentName + ":" }

// discoveredNames returns the agent's discovered tools in discovery order.
// Calls in one model response that each saw the same prior state can store
// equal positions, and those ties are broken by name.
func discoveredNames(state session.ReadonlyState, agentName string) []string {
	type entry struct {
		name  string
		order float64
	}
	prefix := discoveredKeyPrefix(agentName)
	var entries []entry
	for key, value := range state.All() {
		name, ok := strings.CutPrefix(key, prefix)
		if !ok || name == "" {
			continue
		}
		// The database session service decodes state with encoding/json, so the
		// stored int comes back as a float64. A custom service may return another
		// type, and such a tool is kept, sorted last, rather than dropped.
		order := math.Inf(1)
		switch v := value.(type) {
		case int:
			order = float64(v)
		case float64:
			order = v
		}
		entries = append(entries, entry{name, order})
	}
	slices.SortFunc(entries, func(a, b entry) int {
		return cmp.Or(cmp.Compare(a.order, b.order), strings.Compare(a.name, b.name))
	})
	names := make([]string, len(entries))
	for i, e := range entries {
		names[i] = e.name
	}
	return names
}

type searchArgs struct {
	Query string `json:"query" jsonschema:"A natural-language task, keyword, regex pattern, or 'select:name1,name2' to load tools by exact name. Natural phrases like 'find documents' are ranked by relevance. Regex patterns like 'get_.*_file' are supported for name matching. Max 200 characters."` //nolint:lll
}

type searchOutput struct {
	Matches  []searchMatch `json:"matches"`
	Note     string        `json:"note,omitempty"`
	NextStep string        `json:"next_step,omitempty"`
}

type searchMatch struct {
	Name           string `json:"name"`
	Description    string `json:"description"`
	ConnectedSkill string `json:"connected_skill,omitempty" jsonschema:"Load if potentially relevant; generally load one."`
}

func executeSearch(
	ctx agent.Context,
	args searchArgs,
	base tool.Toolset,
	agentName string,
	coreNames map[string]bool,
	skillAnnotations map[string]string,
	maxResults int,
) (searchOutput, error) {
	baseTools, err := base.Tools(ctx)
	if err != nil {
		return searchOutput{Note: "tool catalog unavailable"}, nil //nolint:nilerr
	}

	// Already-discovered tools and core tools are both excluded from results —
	// core tools are always visible, so returning them wastes result slots.
	already := discoveredNames(ctx.ReadonlyState(), agentName)
	alreadyAvailable := make(map[string]bool, len(already)+len(coreNames))
	for _, n := range already {
		alreadyAvailable[n] = true
	}
	for n := range coreNames {
		alreadyAvailable[n] = true
	}

	searchableTools := append([]tool.Tool(nil), baseTools...)
	connectedSkillByTool := make(map[string]string, len(skillAnnotations))
	maps.Copy(connectedSkillByTool, skillAnnotations)
	matches, note := ranksearch.Rank(buildItems(searchableTools), args.Query, alreadyAvailable, ranksearch.Config{
		ItemNoun:      "tool",
		MaxResults:    maxResults,
		MinScoreRatio: minScoreRatio,
		MaxDescLen:    maxDescLen,
	})
	if len(matches) == 0 {
		return searchOutput{Note: note}, nil
	}

	searchMatches := make([]searchMatch, len(matches))
	matchedNames := make([]string, len(matches))
	var hasConnectedSkill bool
	for i, match := range matches {
		connectedSkill := connectedSkillByTool[match.Name]
		matchedNames[i] = match.Name
		searchMatches[i] = searchMatch{
			Name:           match.Name,
			Description:    match.Description,
			ConnectedSkill: connectedSkill,
		}
		if connectedSkill != "" {
			hasConnectedSkill = true
		}
	}
	if err := RevealTools(ctx.State(), agentName, matchedNames...); err != nil {
		return searchOutput{}, fmt.Errorf("toolsearch: persist discovered tools: %w", err)
	}

	nextStep := "The matched tools are now available. " +
		"End this response now and call the appropriate tool in your next response."
	if hasConnectedSkill {
		nextStep = "End this model response now. In the next response, if connected_skill could be relevant to your task " +
			"and your application supports skills, load only the most relevant one unless the task requires multiple. " +
			"Then use the matched tools."
	}

	return searchOutput{Matches: searchMatches, Note: note, NextStep: nextStep}, nil
}

// buildItems turns the packable tools in catalog into ranksearch items. Tools
// that don't implement requestProcessor are skipped: the model could discover
// them but would get an "unknown tool" error when calling them.
func buildItems(catalog []tool.Tool) []ranksearch.Item {
	items := make([]ranksearch.Item, 0, len(catalog))
	for _, t := range catalog {
		if _, ok := t.(requestProcessor); !ok {
			continue
		}
		items = append(items, ranksearch.Item{
			Name:        t.Name(),
			Description: t.Description(),
			Tokens:      ranksearch.BuildTokens(t.Name(), t.Description(), argTokens(t)...),
		})
	}
	return items
}

// declarer is the structural hook ADK runnable tools implement to expose their
// argument schema. We type-assert to it (like requestProcessor) to index
// argument names and descriptions without importing ADK internals.
type declarer interface {
	Declaration() *genai.FunctionDeclaration
}

// argTokens returns a tool's argument names and descriptions for inclusion in
// its search token bag. Tools without a Declaration contribute nothing.
func argTokens(t tool.Tool) []string {
	d, ok := t.(declarer)
	if !ok {
		return nil
	}
	decl := d.Declaration()
	if decl == nil {
		return nil
	}
	var extra []string
	// A FunctionDeclaration carries its argument schema in one of two mutually
	// exclusive fields. ADK functiontool populates
	// the raw ParametersJsonSchema; the typed Parameters is handled too in case a
	// tool sets it instead. If an ADK upgrade ever changes the ParametersJsonSchema
	// representation, the type assertion below stops matching and arguments drop
	// out of the index — TestSearch_IndexesArguments detects when that
	// happens, since it asserts arg-only discovery through a real functiontool.
	if js, ok := decl.ParametersJsonSchema.(*jsonschema.Schema); ok {
		for argName, argSchema := range js.Properties {
			extra = append(extra, argName)
			if argSchema != nil {
				extra = append(extra, argSchema.Description)
			}
		}
	}
	if decl.Parameters != nil {
		for argName, argSchema := range decl.Parameters.Properties {
			extra = append(extra, argName)
			if argSchema != nil {
				extra = append(extra, argSchema.Description)
			}
		}
	}
	return extra
}

type gatingToolset struct {
	base             tool.Toolset
	coreNames        map[string]bool
	agentName        string
	searchTool       tool.Tool
	skillAnnotations map[string]string
}

func (g *gatingToolset) Name() string { return "SearchGatedToolset" }

// requestProcessor matches the per-tool hook ADK uses to pack a tool's
// declaration into the outgoing LLM request. functiontool and the other ADK
// tools implement it; we type-assert to it structurally to avoid importing
// ADK's internal toolinternal package.
type requestProcessor interface {
	ProcessRequest(ctx agent.Context, req *model.LLMRequest) error
}

// Tools returns search_tools, the core tools, and the tools discovered so far
// under Config.AgentName. The flow calls Tools before every model step, so a tool
// discovered by search_tools is callable on the next step of the same
// invocation. A live session is the exception: it resolves its tools once,
// before it first connects, and its reconnects reuse that list, so tools
// discovered during it appear only in a later session.
//
// Discovered tools are appended after the core tools in discovery order (the
// order they were persisted to state) rather than catalog order. The tool list
// therefore only ever grows at the tail across turns, which preserves as much
// of the provider's prompt-cache prefix as a client-side toolset can.
func (g *gatingToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	baseTools, err := g.base.Tools(ctx)
	if err != nil {
		return nil, err
	}

	// Degenerate guard: if every base tool is already in the core set, return
	// them all without search_tools — there is nothing left to discover.
	allInCore := true
	for _, t := range baseTools {
		if !g.coreNames[t.Name()] {
			allInCore = false
			break
		}
	}
	if allInCore {
		return baseTools, nil
	}

	byName := make(map[string]tool.Tool, len(baseTools))
	for _, t := range baseTools {
		byName[t.Name()] = t
	}

	visible := []tool.Tool{g.searchTool}
	for _, t := range baseTools {
		if g.coreNames[t.Name()] {
			visible = append(visible, t)
		}
	}
	for _, name := range discoveredNames(ctx.ReadonlyState(), g.agentName) {
		if g.coreNames[name] {
			continue
		}
		if t, ok := byName[name]; ok {
			visible = append(visible, t)
		}
	}
	return visible, nil
}

// ProcessRequest forwards to the base toolset when it implements the hook, so
// the base can still inject its own request state, such as instructions.
// Tools needs no help here: the flow packs whatever it returns.
func (g *gatingToolset) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if rp, ok := g.base.(requestProcessor); ok {
		if err := rp.ProcessRequest(ctx, req); err != nil {
			return fmt.Errorf("toolsearch: base toolset ProcessRequest: %w", err)
		}
	}
	return nil
}
