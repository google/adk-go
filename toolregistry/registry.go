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

// Package toolregistry implements dynamic tool discovery and loading. It
// addresses the context-window cost of declaring every available tool
// upfront on an LLM agent: tools live in a Registry; the LLM lists what
// matches its current need via list_tools and activates a tool by name
// via load_tool. Only the tools the agent has actively loaded — plus the
// always-on list_tools / load_tool — surface in subsequent LLM requests.
//
// Loaded tools are tracked in session state under StateKeyLoadedTools so
// activation persists across turns within an invocation chain.
//
// Wire by adding a Toolset to an LlmAgent's Tools list:
//
//	reg := toolregistry.New()
//	reg.RegisterTool("calculator", calculatorTool, toolregistry.Info{
//	    Description: "Perform arithmetic on two numbers.",
//	    Tags:        []string{"math"},
//	    Hints:       "Use when the user asks for a calculation.",
//	})
//	agent := llmagent.New(llmagent.Config{
//	    Model: model,
//	    Toolsets: []tool.Toolset{toolregistry.NewToolset(reg)},
//	})
package toolregistry

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	"google.golang.org/adk/tool"
)

// StateKeyLoadedTools is the session-state key under which the names of
// currently-loaded tools are tracked. Stored as []string. Unprefixed so
// it persists across turns within an invocation chain (the temp: prefix
// would clear it after each invocation).
const StateKeyLoadedTools = "_adk_loaded_tools"

// Info is the metadata the LLM sees when calling list_tools. Kept
// lightweight — a tool's full FunctionDeclaration only goes into the
// LLM request after the tool is loaded.
type Info struct {
	// Name is the tool's stable identifier. Must be non-empty and unique
	// within a Registry.
	Name string `json:"name"`

	// Description is the human / LLM-facing one-liner. Echoed in
	// list_tools output and on the underlying tool when loaded.
	Description string `json:"description,omitempty"`

	// Tags categorize the tool (e.g. "math", "search", "file"). list_tools
	// can filter by tag, letting the agent narrow scope before reading
	// long descriptions.
	Tags []string `json:"tags,omitempty"`

	// Hints is an optional "when to use" sentence the LLM sees in
	// list_tools output. Useful for tools whose description focuses on
	// what they do rather than when to invoke them.
	Hints string `json:"hints,omitempty"`
}

// Filter narrows what list_tools returns. Empty fields match everything.
type Filter struct {
	// Query is a case-insensitive substring matched against Name,
	// Description, and Tags.
	Query string

	// Tags require ALL listed tags to be present on the tool.
	Tags []string
}

// Builder constructs a tool on first activation. The factory pattern lets
// callers register cheap metadata without paying for tool construction
// up front (relevant for tools that wrap remote services).
type Builder func() (tool.Tool, error)

// entry is one row in the registry.
type entry struct {
	info  Info
	build Builder

	// cached is the lazily-built tool. Access guarded by registryMu.
	cached tool.Tool
}

// Registry is the central catalog of dynamically-loadable tools.
//
// Concurrency: all methods are safe for concurrent use.
type Registry struct {
	mu      sync.RWMutex
	entries map[string]*entry
}

// New constructs an empty Registry.
func New() *Registry {
	return &Registry{entries: map[string]*entry{}}
}

// Register adds a tool factory keyed by info.Name. Registering an
// existing name replaces the prior entry — useful for tests and for
// hot-reloading scenarios.
func (r *Registry) Register(info Info, build Builder) error {
	if info.Name == "" {
		return errors.New("toolregistry: Register: Info.Name must not be empty")
	}
	if build == nil {
		return errors.New("toolregistry: Register: Builder must not be nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.entries[info.Name] = &entry{info: info, build: build}
	return nil
}

// RegisterTool is a convenience wrapper for the common case where the
// tool is already constructed. Equivalent to Register with a Builder
// that returns the supplied tool.
func (r *Registry) RegisterTool(t tool.Tool, info Info) error {
	if t == nil {
		return errors.New("toolregistry: RegisterTool: tool must not be nil")
	}
	if info.Name == "" {
		info.Name = t.Name()
	}
	return r.Register(info, func() (tool.Tool, error) { return t, nil })
}

// Get returns the tool registered under name. The first call constructs
// the tool via its Builder; subsequent calls return the cached value.
func (r *Registry) Get(name string) (tool.Tool, error) {
	r.mu.RLock()
	e, ok := r.entries[name]
	r.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("toolregistry: tool %q not found", name)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if e.cached != nil {
		return e.cached, nil
	}
	t, err := e.build()
	if err != nil {
		return nil, fmt.Errorf("toolregistry: build %q: %w", name, err)
	}
	e.cached = t
	return t, nil
}

// Has reports whether name is registered.
func (r *Registry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.entries[name]
	return ok
}

// List returns metadata for every registered tool that matches f.
// Results are sorted by Name for deterministic output.
func (r *Registry) List(f Filter) []Info {
	r.mu.RLock()
	defer r.mu.RUnlock()

	q := strings.ToLower(strings.TrimSpace(f.Query))
	out := make([]Info, 0, len(r.entries))
	for _, e := range r.entries {
		if !matchesFilter(e.info, q, f.Tags) {
			continue
		}
		out = append(out, e.info)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Names returns the sorted list of registered tool names.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.entries))
	for name := range r.entries {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// matchesFilter reports whether info matches q (case-insensitive
// substring across Name / Description / Tags) and contains every tag in
// requiredTags.
func matchesFilter(info Info, q string, requiredTags []string) bool {
	if q != "" {
		hay := strings.ToLower(info.Name) + " " +
			strings.ToLower(info.Description) + " " +
			strings.ToLower(strings.Join(info.Tags, " "))
		if !strings.Contains(hay, q) {
			return false
		}
	}
	if len(requiredTags) == 0 {
		return true
	}
	tagSet := map[string]struct{}{}
	for _, t := range info.Tags {
		tagSet[strings.ToLower(t)] = struct{}{}
	}
	for _, want := range requiredTags {
		if _, ok := tagSet[strings.ToLower(want)]; !ok {
			return false
		}
	}
	return true
}
