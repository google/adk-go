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
//
// Gating decides which tools the model sees, not which tools a caller may use.
// Pass a base toolset that is already filtered for the caller, since
// search_tools can reveal any tool in it that the model can call.
package toolsearch

import (
	"cmp"
	"errors"
	"fmt"
	"maps"
	"math"
	"net/url"
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
	// CoreToolNames are always advertised alongside search_tools; everything
	// else in the base toolset is gated until the model discovers it.
	CoreToolNames []string
	// GatedToolNames are the names of the tools hidden behind search_tools. When
	// set, they are listed in the search_tools description so the model knows what
	// it can search for (and can select: them by exact name) instead of guessing
	// blind. Names only — no schemas — so it stays cheap and fully static. Names
	// that are also in CoreToolNames are left out, since those are already active,
	// and so is ToolName, which New reserves for the search tool. Entries should
	// name callable base tools: any other name is still listed, and select: then
	// reports it as not found.
	GatedToolNames []string
	// SkillAnnotations maps tool names to skill names so search results carry
	// a connected_skill hint pointing the model at the relevant skill.
	SkillAnnotations map[string]string
	// MaxResults caps BM25 and regex matches, not exact selections.
	// Defaults to 8 when zero or negative.
	MaxResults int
}

// New exposes CoreToolNames up front and gates other tools behind search_tools.
// search_tools is omitted only when every base tool is already core. New copies
// cfg, so later changes to its slices and maps have no effect.
//
// ToolName is reserved for the search tool: it cannot be a core tool, and a
// base tool by that name is never searchable or exposed. For the same reason an
// agent can use at most one gating toolset, since two on one agent each expose
// a tool named ToolName and the framework rejects every model step with a
// duplicate-tool error. Combine the catalogs into one base toolset instead.
func New(base tool.Toolset, cfg Config) (tool.Toolset, error) {
	if base == nil {
		return nil, errors.New("toolsearch: base toolset is nil")
	}
	if slices.Contains(cfg.CoreToolNames, ToolName) {
		return nil, fmt.Errorf("toolsearch: core tool name %q is reserved for the search tool", ToolName)
	}
	maxResults := cfg.MaxResults
	if maxResults <= 0 {
		maxResults = defaultMaxResults
	}

	g := &gatingToolset{
		base:             base,
		coreNames:        nameSet(cfg.CoreToolNames),
		skillAnnotations: maps.Clone(cfg.SkillAnnotations),
	}
	var gated []string
	for _, name := range cfg.GatedToolNames {
		if !g.coreNames[name] && name != ToolName {
			gated = append(gated, name)
		}
	}

	searchTool, err := functiontool.New(
		functiontool.Config{
			Name:        ToolName,
			Description: ranksearch.DescribeSearch(toolDescription, gated),
		},
		func(ctx agent.Context, args searchArgs) (searchOutput, error) {
			return executeSearch(ctx, args, base, g.coreNames, g.skillAnnotations, maxResults)
		},
	)
	if err != nil {
		return nil, fmt.Errorf("toolsearch: build search tool: %w", err)
	}
	g.searchTool = searchTool
	return g, nil
}

// nameSet builds a lookup set from a slice of tool names.
func nameSet(names []string) map[string]bool {
	set := make(map[string]bool, len(names))
	for _, n := range names {
		set[n] = true
	}
	return set
}

// RevealTools adds tools to the session discovery state used by search_tools,
// so they become visible on the calling agent's next model step. During a live
// session they appear only in a later session. Discoveries are kept per agent,
// keyed by ctx.AgentName(). Empty and already revealed names are ignored. A
// name is callable only while the base toolset returns a tool by that name
// that the flow can pack into a request.
//
// Each tool is stored under its own key, so reveals from several function calls
// in one model response are all kept. Those calls run concurrently and share
// the session state, so a call may or may not see a sibling's reveals. The
// relative order of tools revealed in one model response is therefore
// unspecified, but once recorded it does not change.
func RevealTools(ctx agent.Context, names ...string) error {
	state, agentName := ctx.State(), ctx.AgentName()
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

// discoveredKeyPrefix escapes the agent name so it holds no colon. Both agent
// and tool names may contain one, and an unescaped agent "a" revealing tool
// "b:t" would write the key agent "a:b" reads as tool "t".
func discoveredKeyPrefix(agentName string) string {
	return stateKeyPrefix + url.QueryEscape(agentName) + ":"
}

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
		// stored int comes back as a float64, and a custom service may decode it
		// as an int64. A tool with any other type is kept, sorted last, rather
		// than dropped.
		order := math.Inf(1)
		switch v := value.(type) {
		case int:
			order = float64(v)
		case int64:
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
	coreNames map[string]bool,
	skillAnnotations map[string]string,
	maxResults int,
) (searchOutput, error) {
	baseTools, err := base.Tools(ctx)
	if err != nil {
		return searchOutput{}, fmt.Errorf("toolsearch: list base tools: %w", err)
	}

	// Already-discovered tools and core tools are both excluded from results —
	// core tools are always visible, so returning them wastes result slots.
	already := discoveredNames(ctx.ReadonlyState(), ctx.AgentName())
	alreadyAvailable := make(map[string]bool, len(already)+len(coreNames))
	for _, n := range already {
		alreadyAvailable[n] = true
	}
	for n := range coreNames {
		alreadyAvailable[n] = true
	}

	matches, note := ranksearch.Rank(buildItems(baseTools), args.Query, alreadyAvailable, ranksearch.Config{
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
		connectedSkill := skillAnnotations[match.Name]
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
	if err := RevealTools(ctx, matchedNames...); err != nil {
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

// buildItems turns the tools in catalog that the model can call into ranksearch
// items. A tool must pack itself into the request and declare a function. A
// model-side built-in such as geminitool.GoogleSearch packs but declares no
// function, so a call to it would be rejected as an unknown tool. Search
// therefore never offers one: list it in CoreToolNames instead. A base tool
// named ToolName is skipped because it would collide with the search tool.
func buildItems(catalog []tool.Tool) []ranksearch.Item {
	items := make([]ranksearch.Item, 0, len(catalog))
	for _, t := range catalog {
		if !callable(t) || t.Name() == ToolName {
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

// callable reports whether the flow can both pack t into a request and dispatch
// a function call to it.
func callable(t tool.Tool) bool {
	if _, ok := t.(requestProcessor); !ok {
		return false
	}
	d, ok := t.(declarer)
	return ok && d.Declaration() != nil
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
	if js, ok := decl.ParametersJsonSchema.(*jsonschema.Schema); ok && js != nil {
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

// Tools returns search_tools, the core tools, and the tools the calling agent
// has discovered. The flow calls Tools before every model step, so a tool
// discovered by search_tools is callable on the next step of the same
// invocation. A live session is the exception: it resolves its tools once,
// before it first connects, and its reconnects reuse that list, so tools
// discovered during it appear only in a later session.
//
// Discovered tools are appended after the core tools in discovery order (the
// order they were persisted to state) rather than catalog order. While the base
// catalog stays the same, the list therefore only grows at the tail across
// turns, which preserves as much of the provider's prompt-cache prefix as a
// client-side toolset can. A discovered tool the base stops returning drops out
// of the list.
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
	for _, name := range discoveredNames(ctx.ReadonlyState(), ctx.AgentName()) {
		if g.coreNames[name] {
			continue
		}
		// RevealTools accepts any name, so skip a tool the flow cannot pack, and
		// one named ToolName, which would duplicate the search tool.
		t, ok := byName[name]
		if _, packable := t.(requestProcessor); ok && packable && name != ToolName {
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
