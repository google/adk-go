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

package toolsearch

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/ranksearch"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/geminitool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// testAgent is the agent name the stub contexts report.
const testAgent = "test_agent"

// stubTool is a minimal tool.Tool for testing.
type stubTool struct {
	name string
	desc string
}

func (t *stubTool) Name() string        { return t.name }
func (t *stubTool) Description() string { return t.desc }
func (t *stubTool) IsLongRunning() bool { return false }

// Declaration makes stubTool a function tool, which search requires.
func (t *stubTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: t.name, Description: t.desc}
}

func (t *stubTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	if req.Tools == nil {
		req.Tools = map[string]any{}
	}
	req.Tools[t.name] = t
	return nil
}

// nonPackableStubTool implements tool.Tool but deliberately omits ProcessRequest,
// simulating a tool that cannot be packed into an LLM request.
type nonPackableStubTool struct {
	name string
	desc string
}

func (t *nonPackableStubTool) Name() string        { return t.name }
func (t *nonPackableStubTool) Description() string { return t.desc }
func (t *nonPackableStubTool) IsLongRunning() bool { return false }

// fakeState implements both session.ReadonlyState and session.State.
type fakeState struct {
	data map[string]any
}

func newFakeState(data map[string]any) *fakeState {
	if data == nil {
		data = map[string]any{}
	}
	return &fakeState{data: data}
}

func (s *fakeState) Get(key string) (any, error) {
	v, ok := s.data[key]
	if !ok {
		return nil, fmt.Errorf("key not found: %s", key)
	}
	return v, nil
}

func (s *fakeState) Set(key string, val any) error {
	s.data[key] = val
	return nil
}

func (s *fakeState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range s.data {
			if !yield(k, v) {
				return
			}
		}
	}
}

// stubReadonlyContext implements agent.ReadonlyContext backed by fakeState.
type stubReadonlyContext struct {
	context.Context
	state *fakeState
}

func (c *stubReadonlyContext) UserContent() *genai.Content          { return nil }
func (c *stubReadonlyContext) InvocationID() string                 { return "" }
func (c *stubReadonlyContext) AgentName() string                    { return testAgent }
func (c *stubReadonlyContext) ReadonlyState() session.ReadonlyState { return c.state }
func (c *stubReadonlyContext) UserID() string                       { return "" }
func (c *stubReadonlyContext) AppName() string                      { return "" }
func (c *stubReadonlyContext) SessionID() string                    { return "" }
func (c *stubReadonlyContext) Branch() string                       { return "" }

func newCtx(state *fakeState) agent.ReadonlyContext {
	return &stubReadonlyContext{Context: context.Background(), state: state}
}

// fakeToolContext satisfies agent.Context, which ProcessRequest requires. It
// embeds StrictContextMock (rather than composing stubReadonlyContext, which
// would ambiguously promote overlapping methods) so any v2 method this test
// never exercises panics loudly instead of silently returning a zero value.
type fakeToolContext struct {
	agent.StrictContextMock
	state     *fakeState
	agentName string
}

func (c *fakeToolContext) AgentName() string                                    { return c.agentName }
func (c *fakeToolContext) ReadonlyState() session.ReadonlyState                 { return c.state }
func (c *fakeToolContext) State() session.State                                 { return c.state }
func (c *fakeToolContext) Artifacts() agent.Artifacts                           { return nil }
func (c *fakeToolContext) FunctionCallID() string                               { return "" }
func (c *fakeToolContext) Actions() *session.EventActions                       { return nil }
func (c *fakeToolContext) ToolConfirmation() *toolconfirmation.ToolConfirmation { return nil }
func (c *fakeToolContext) RequestConfirmation(_ string, _ any) error            { return nil }

func (c *fakeToolContext) SearchMemory(_ context.Context, _ string) (*memory.SearchResponse, error) {
	return nil, nil
}

var _ agent.Context = (*fakeToolContext)(nil)

func newToolCtx(state *fakeState) agent.Context {
	return newAgentToolCtx(state, testAgent)
}

func newAgentToolCtx(state *fakeState, agentName string) agent.Context {
	return &fakeToolContext{
		StrictContextMock: agent.NewStrictContextMock(context.Background()),
		state:             state,
		agentName:         agentName,
	}
}

// packableStubTool packs itself into the request, recording how many times its
// ProcessRequest was invoked and, when packOrder is set, in what order.
type packableStubTool struct {
	stubTool
	packCount *int
	packOrder *[]string
}

func (t *packableStubTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	if req.Tools == nil {
		req.Tools = map[string]any{}
	}
	req.Tools[t.name] = t
	if t.packCount != nil {
		*t.packCount++
	}
	if t.packOrder != nil {
		*t.packOrder = append(*t.packOrder, t.name)
	}
	return nil
}

// staticToolset always returns the same tools.
type staticToolset struct {
	tools []tool.Tool
}

func (s *staticToolset) Name() string                                       { return "static" }
func (s *staticToolset) Tools(_ agent.ReadonlyContext) ([]tool.Tool, error) { return s.tools, nil }

// packableBaseToolset is a staticToolset that also implements requestProcessor,
// simulating a base toolset that injects its own state
// into the LLM request.
type packableBaseToolset struct {
	staticToolset
	packCount *int
}

func (b *packableBaseToolset) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	if req.Tools == nil {
		req.Tools = map[string]any{}
	}
	req.Tools["base_injected"] = struct{}{}
	if b.packCount != nil {
		*b.packCount++
	}
	return nil
}

type toolDef struct {
	name, desc string
}

func makeTools(defs ...toolDef) []tool.Tool {
	tools := make([]tool.Tool, len(defs))
	for i, d := range defs {
		tools[i] = &stubTool{name: d.name, desc: d.desc}
	}
	return tools
}

// checkNames reports names in want that are missing from got and names in
// notWant that are present in got.
func checkNames(t *testing.T, got, want, notWant []string) {
	t.Helper()
	for _, n := range want {
		if !slices.Contains(got, n) {
			t.Errorf("names = %v, want %q included", got, n)
		}
	}
	for _, n := range notWant {
		if slices.Contains(got, n) {
			t.Errorf("names = %v, want %q excluded", got, n)
		}
	}
}

// discoveredState returns state in which test_agent discovered names in order.
func discoveredState(t *testing.T, names ...string) *fakeState {
	t.Helper()
	state := newFakeState(nil)
	if err := RevealTools(newToolCtx(state), names...); err != nil {
		t.Fatalf("RevealTools() error = %v", err)
	}
	return state
}

func mustNew(t *testing.T, base tool.Toolset, cfg Config) *gatingToolset {
	t.Helper()
	ts, err := New(base, cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return ts.(*gatingToolset)
}

func mustToolNames(t *testing.T, ts tool.Toolset, ctx agent.ReadonlyContext) []string {
	t.Helper()
	tools, err := ts.Tools(ctx)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	return toolNames(tools)
}

func mustSearch(t *testing.T, ctx agent.Context, query string, base tool.Toolset, gts *gatingToolset, maxResults int) searchOutput {
	t.Helper()
	out, err := executeSearch(ctx, searchArgs{Query: query}, base, gts.coreNames, gts.skillAnnotations, maxResults)
	if err != nil {
		t.Fatalf("executeSearch(%q) error = %v", query, err)
	}
	return out
}

func mustProcessRequest(t *testing.T, gts *gatingToolset, state *fakeState, req *model.LLMRequest) {
	t.Helper()
	if err := gts.ProcessRequest(newToolCtx(state), req); err != nil {
		t.Fatalf("ProcessRequest() error = %v", err)
	}
}

// TestGatingToolset_EmptyState verifies that only search_tools + core tools
// are advertised when no tools have been discovered yet.
func TestGatingToolset_EmptyState(t *testing.T) {
	base := &staticToolset{tools: makeTools(
		toolDef{"get_current_time", "get time"},
		toolDef{"list_books", "list all books"},
		toolDef{"get_author", "get an author"},
	)}
	ts := mustNew(t, base, Config{
		CoreToolNames: []string{"get_current_time"},
	})

	names := mustToolNames(t, ts, newCtx(newFakeState(nil)))
	checkNames(t, names, []string{ToolName, "get_current_time"}, []string{"list_books", "get_author"})
}

// TestGatingToolset_WithDiscoveredState verifies that tools written to session
// state appear in the next Tools() call.
func TestGatingToolset_WithDiscoveredState(t *testing.T) {
	base := &staticToolset{tools: makeTools(
		toolDef{"get_current_time", "get time"},
		toolDef{"list_books", "list all books"},
		toolDef{"get_author", "get an author"},
	)}
	ts := mustNew(t, base, Config{
		CoreToolNames: []string{"get_current_time"},
	})

	state := discoveredState(t, "list_books")

	names := mustToolNames(t, ts, newCtx(state))
	checkNames(t, names, []string{ToolName, "get_current_time", "list_books"}, []string{"get_author"})
}

func TestRevealTools_AppendsUniqueNamesInOrder(t *testing.T) {
	// Discovery order deliberately differs from name order.
	state := discoveredState(t, "zeta_tool", "shared_tool")

	if err := RevealTools(
		newToolCtx(state),
		"shared_tool",
		"skill_tool_b",
		"skill_tool_a",
		"skill_tool_b",
	); err != nil {
		t.Fatalf("RevealTools() error = %v", err)
	}

	want := []string{"zeta_tool", "shared_tool", "skill_tool_b", "skill_tool_a"}
	if diff := cmp.Diff(want, discoveredNames(state, testAgent)); diff != "" {
		t.Errorf("discoveredNames() mismatch (-want +got):\n%s", diff)
	}
}

func TestSearch_ExcludesPreviouslyRevealedTools(t *testing.T) {
	base := &staticToolset{tools: makeTools(
		toolDef{"list_notes", "list notes"},
		toolDef{"list_books", "list books"},
	)}
	state := newFakeState(nil)
	if err := RevealTools(newToolCtx(state), "list_notes"); err != nil {
		t.Fatalf("RevealTools() error = %v", err)
	}

	out, err := executeSearch(
		newToolCtx(state),
		searchArgs{Query: "select:list_notes,list_books"},
		base,
		nil,
		nil,
		8,
	)
	if err != nil {
		t.Fatalf("executeSearch() error = %v", err)
	}
	if diff := cmp.Diff([]string{"list_books"}, matchNames(out.Matches)); diff != "" {
		t.Errorf("executeSearch() matches mismatch (-want +got):\n%s", diff)
	}
}

// TestGatingToolset_DiscoveredToolsInDiscoveryOrder verifies that discovered
// tools are appended after core tools in discovery order, not catalog order.
// This keeps the serialized tool list append-only across turns, which is what
// preserves the provider's prompt-cache prefix.
func TestGatingToolset_DiscoveredToolsInDiscoveryOrder(t *testing.T) {
	base := &staticToolset{tools: makeTools(
		toolDef{"a_tool", "catalog-first tool"},
		toolDef{"get_current_time", "get time"},
		toolDef{"z_tool", "catalog-last tool"},
	)}
	ts := mustNew(t, base, Config{
		CoreToolNames: []string{"get_current_time"},
	})

	// z_tool was discovered before a_tool.
	state := discoveredState(t, "z_tool", "a_tool")

	want := []string{ToolName, "get_current_time", "z_tool", "a_tool"}
	if diff := cmp.Diff(want, mustToolNames(t, ts, newCtx(state))); diff != "" {
		t.Errorf("Tools() must follow discovery order, not catalog order (-want +got):\n%s", diff)
	}
}

// TestGatingToolset_DegenerateGuard verifies that search_tools is omitted when
// all base tools fit in the core set.
func TestGatingToolset_DegenerateGuard(t *testing.T) {
	base := &staticToolset{tools: makeTools(
		toolDef{"get_current_time", "get time"},
		toolDef{"show_help", "show help"},
	)}
	ts := mustNew(t, base, Config{
		CoreToolNames: []string{"get_current_time", "show_help"},
	})

	names := mustToolNames(t, ts, newCtx(newFakeState(nil)))
	checkNames(t, names, []string{"get_current_time", "show_help"}, []string{ToolName})
}

// TestProcessRequest_LeavesToolPackingToTheFlow checks that ProcessRequest
// does not pack discovered tools itself. Tools already returns them and the
// flow packs everything Tools returns, so packing here too would fail the run
// with a duplicate-tool error.
func TestProcessRequest_LeavesToolPackingToTheFlow(t *testing.T) {
	packs := 0
	base := &staticToolset{tools: []tool.Tool{
		&packableStubTool{stubTool: stubTool{name: "list_books", desc: "list"}, packCount: &packs},
		&packableStubTool{stubTool: stubTool{name: "get_author", desc: "get"}},
	}}
	gts := mustNew(t, base, Config{})

	req := &model.LLMRequest{Tools: map[string]any{ToolName: struct{}{}}}
	mustProcessRequest(t, gts, discoveredState(t, "list_books"), req)

	if packs != 0 {
		t.Errorf("discovered tool packed %d times by ProcessRequest, want 0", packs)
	}
}

// TestSearch_CatalogErrorIsReturned checks that a failing base toolset surfaces
// as an error from search_tools rather than as an empty result.
func TestSearch_CatalogErrorIsReturned(t *testing.T) {
	errCatalog := errors.New("catalog down")
	base := &failingToolset{err: errCatalog}
	gts := mustNew(t, base, Config{})

	_, err := executeSearch(newToolCtx(newFakeState(nil)), searchArgs{Query: "books"}, base, gts.coreNames, nil, 8)
	if !errors.Is(err, errCatalog) {
		t.Errorf("executeSearch() error = %v, want it to wrap %v", err, errCatalog)
	}
}

// failingToolset returns err from Tools.
type failingToolset struct{ err error }

func (f *failingToolset) Name() string                                       { return "failing" }
func (f *failingToolset) Tools(_ agent.ReadonlyContext) ([]tool.Tool, error) { return nil, f.err }

// TestProcessRequest_ForwardsToBase verifies that ProcessRequest is forwarded to
// the base toolset when it implements requestProcessor, so the base can inject
// its own state.
func TestProcessRequest_ForwardsToBase(t *testing.T) {
	basePacks := 0
	base := &packableBaseToolset{
		staticToolset: staticToolset{tools: makeTools(
			toolDef{"get_current_time", "ask"},
			toolDef{"list_books", "list"},
		)},
		packCount: &basePacks,
	}
	gts := mustNew(t, base, Config{CoreToolNames: []string{"get_current_time"}})

	req := &model.LLMRequest{Tools: map[string]any{ToolName: struct{}{}}}
	mustProcessRequest(t, gts, newFakeState(nil), req)

	if _, ok := req.Tools["base_injected"]; !ok {
		t.Error("base toolset ProcessRequest was not forwarded")
	}
	if basePacks != 1 {
		t.Errorf("base ProcessRequest called %d times, want 1", basePacks)
	}
}

// TestSearchTool_AdvertisesGatedToolsInDescription verifies that the gated tool
// names passed in Config are listed in the search_tools description the model
// receives, so it knows what it can search for and select: by name.
func TestSearchTool_AdvertisesGatedToolsInDescription(t *testing.T) {
	base := &staticToolset{tools: makeTools(
		toolDef{"get_current_time", "ask"},
		toolDef{"list_books", "list books"},
		toolDef{"list_publishers", "list publishers"},
	)}
	ts := mustNew(t, base, Config{
		CoreToolNames:  []string{"get_current_time"},
		GatedToolNames: []string{"list_publishers", "list_books", "get_current_time"},
	})

	tools, err := ts.Tools(newCtx(newFakeState(nil)))
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}

	var desc string
	for _, tl := range tools {
		if tl.Name() == ToolName {
			desc = tl.Description()
		}
	}
	if desc == "" {
		t.Fatal("search_tools is missing or has an empty description")
	}
	for _, want := range []string{"list_publishers", "list_books"} {
		if !strings.Contains(desc, want) {
			t.Errorf("search_tools description does not advertise gated tool %q", want)
		}
	}
	if strings.Contains(desc, "get_current_time") {
		t.Error("search_tools description lists core tool get_current_time")
	}
}

func TestSearch_ReportsOptionalConnectedSkillAndRevealsTool(t *testing.T) {
	base := &staticToolset{tools: append(makeTools(toolDef{"get_current_time", "ask"}),
		&stubTool{
			name: "list_recent_files",
			desc: "list recently modified files",
		},
	)}
	gts := mustNew(t, base, Config{
		CoreToolNames: []string{"get_current_time"},
		SkillAnnotations: map[string]string{
			"list_recent_files": "file-browser",
		},
	})

	checkNames(t, mustToolNames(t, gts, newCtx(newFakeState(nil))), []string{ToolName}, nil)

	state := newFakeState(nil)
	out := mustSearch(t, newToolCtx(state), "recent files", base, gts, 8)
	if len(out.Matches) != 1 {
		t.Fatalf("executeSearch() returned %d matches, want 1", len(out.Matches))
	}
	if got := out.Matches[0]; got.Name != "list_recent_files" || got.ConnectedSkill != "file-browser" {
		t.Errorf("match = %+v, want list_recent_files with connected_skill file-browser", got)
	}
	if diff := cmp.Diff([]string{"list_recent_files"}, discoveredNames(state, testAgent)); diff != "" {
		t.Errorf("discoveredNames() mismatch (-want +got):\n%s", diff)
	}
	for _, want := range []string{"could be relevant to your task", "load only the most relevant one"} {
		if !strings.Contains(out.NextStep, want) {
			t.Errorf("next_step = %q, want it to contain %q", out.NextStep, want)
		}
	}

	checkNames(t, mustToolNames(t, gts, newCtx(state)), []string{"list_recent_files"}, nil)
}

// TestSearch_CappedResultsIncludeNote verifies that executeSearch plumbs the
// MaxResults cap through to the ranker and surfaces the truncation note.
func TestSearch_CappedResultsIncludeNote(t *testing.T) {
	base := &staticToolset{tools: makeTools(
		toolDef{"tool_a", "some tool"},
		toolDef{"tool_b", "some tool"},
		toolDef{"tool_c", "some tool"},
	)}
	gts := mustNew(t, base, Config{MaxResults: 2})

	out := mustSearch(t, newToolCtx(newFakeState(nil)), "tool", base, gts, 2)
	if len(out.Matches) != 2 {
		t.Errorf("executeSearch() returned %d matches, want 2", len(out.Matches))
	}
	if out.Note == "" {
		t.Error("executeSearch() note is empty, want a truncation note")
	}
}

// TestSearch_SelectByName verifies executeSearch handles the "select:a,b" form
// and persists the selected tools to the discovered-state so later turns surface
// them.
func TestSearch_SelectByName(t *testing.T) {
	base := &staticToolset{tools: makeTools(
		toolDef{"patch_book", "patch a book"},
		toolDef{"update_book", "update a book"},
		toolDef{"list_books", "list books"},
	)}
	gts := mustNew(t, base, Config{})

	state := newFakeState(nil)
	out := mustSearch(t, newToolCtx(state), "select:patch_book,update_book,unknown_tool", base, gts, 8)
	checkNames(t, matchNames(out.Matches), []string{"patch_book", "update_book"}, []string{"list_books"})
	if !strings.Contains(out.Note, "unknown_tool") {
		t.Errorf("note = %q, want the missing tool name %q in it", out.Note, "unknown_tool")
	}
	if diff := cmp.Diff([]string{"patch_book", "update_book"}, discoveredNames(state, testAgent)); diff != "" {
		t.Errorf("selected tools must be persisted to discovered state (-want +got):\n%s", diff)
	}
}

// TestBuildItems_SkipsNonPackableTools verifies that tools without ProcessRequest
// are excluded from the searchable set — the model would discover them but
// receive an "unknown tool" error when trying to call them.
func TestBuildItems_SkipsNonPackableTools(t *testing.T) {
	catalog := []tool.Tool{
		&stubTool{name: "packable_tool", desc: "can be called"},
		&nonPackableStubTool{name: "non_packable_tool", desc: "cannot be called"},
	}
	checkNames(t, itemNames(buildItems(catalog)), []string{"packable_tool"}, []string{"non_packable_tool"})
}

// TestSearch_IndexesArguments verifies a tool is discoverable when the query
// matches only an argument's description — not its name, description, or argument
// names. It uses a real functiontool so the argument schema is exposed the same
// way functiontool exposes it (ParametersJsonSchema).
func TestSearch_IndexesArguments(t *testing.T) {
	updateNote, err := functiontool.New(
		functiontool.Config{Name: "update_note", Description: "modify a note"},
		func(_ agent.Context, _ updateNoteArgs) (struct{}, error) { return struct{}{}, nil },
	)
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}

	base := &staticToolset{tools: []tool.Tool{
		updateNote,
		&stubTool{name: "list_images", desc: "list all images"},
	}}
	gts := mustNew(t, base, Config{})

	// "comment" appears only in the body argument's description, nowhere in the
	// tool name, description, or argument names.
	out := mustSearch(t, newToolCtx(newFakeState(nil)), "comment", base, gts, 8)
	checkNames(t, matchNames(out.Matches), []string{"update_note"}, []string{"list_images"})
}

// updateNoteArgs is the argument struct for the functiontool used in
// TestSearch_IndexesArguments; the jsonschema tag becomes the property
// description that the search must index.
type updateNoteArgs struct {
	Body string `json:"body" jsonschema:"the comment text to attach"`
}

func toolNames(tools []tool.Tool) []string {
	names := make([]string, len(tools))
	for i, t := range tools {
		names[i] = t.Name()
	}
	return names
}

func itemNames(items []ranksearch.Item) []string {
	names := make([]string, len(items))
	for i, it := range items {
		names[i] = it.Name
	}
	return names
}

func matchNames(matches []searchMatch) []string {
	names := make([]string, len(matches))
	for i, m := range matches {
		names[i] = m.Name
	}
	return names
}

func TestSearch_AnnotatesBaseToolWithSkillAnnotation(t *testing.T) {
	base := &staticToolset{tools: makeTools(
		toolDef{"list_folders", "list folders in the workspace"},
		toolDef{"create_folder", "create a new folder"},
		toolDef{"list_notes", "list notes"},
	)}
	gts := mustNew(t, base, Config{
		SkillAnnotations: map[string]string{
			"list_folders":  "folder-browser",
			"create_folder": "folder-browser",
		},
	})

	out := mustSearch(t, newToolCtx(newFakeState(nil)), "folders", base, gts, 8)
	if len(out.Matches) == 0 {
		t.Fatal("executeSearch() returned no matches")
	}
	for _, match := range out.Matches {
		switch match.Name {
		case "list_folders", "create_folder":
			if match.ConnectedSkill != "folder-browser" {
				t.Errorf("%s connected_skill = %q, want folder-browser", match.Name, match.ConnectedSkill)
			}
		case "list_notes":
			if match.ConnectedSkill != "" {
				t.Errorf("unannotated tool list_notes has connected_skill %q", match.ConnectedSkill)
			}
		}
	}
}

// parallelCallModel returns all calls in its first response and records the
// tools declared in the second request.
type parallelCallModel struct {
	calls    []*genai.FunctionCall
	step     int
	declared []string
}

func (*parallelCallModel) Name() string { return "parallel-call-model" }

func (m *parallelCallModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		defer func() { m.step++ }()
		if m.step == 0 {
			parts := make([]*genai.Part, len(m.calls))
			for i, c := range m.calls {
				parts[i] = &genai.Part{FunctionCall: c}
			}
			yield(&model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: parts}}, nil)
			return
		}
		m.declared = nil
		for _, t := range req.Config.Tools {
			for _, d := range t.FunctionDeclarations {
				m.declared = append(m.declared, d.Name)
			}
		}
		yield(&model.LLMResponse{Content: genai.NewContentFromText("done", genai.RoleModel)}, nil)
	}
}

// TestSearch_ParallelCallsKeepAllDiscoveries runs two search_tools calls from
// one model response through the real runner. Each call gets its own state
// delta, so a discovery written by one must survive the other.
func TestSearch_ParallelCallsKeepAllDiscoveries(t *testing.T) {
	newTool := func(name string) tool.Tool {
		ft, err := functiontool.New(functiontool.Config{Name: name, Description: name},
			func(_ agent.Context, _ struct{}) (struct{}, error) { return struct{}{}, nil })
		if err != nil {
			t.Fatalf("functiontool.New(%q) error = %v", name, err)
		}
		return ft
	}
	gated, err := New(&staticToolset{tools: []tool.Tool{newTool("add_numbers"), newTool("multiply_numbers")}},
		Config{})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	llm := &parallelCallModel{calls: []*genai.FunctionCall{
		{ID: "1", Name: ToolName, Args: map[string]any{"query": "select:add_numbers"}},
		{ID: "2", Name: ToolName, Args: map[string]any{"query": "select:multiply_numbers"}},
	}}
	a, err := llmagent.New(llmagent.Config{Name: "calculator", Model: llm, Toolsets: []tool.Toolset{gated}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	for _, err := range r.Run(t.Context(), "user", "session", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}
	checkNames(t, llm.declared, []string{"add_numbers", "multiply_numbers"}, nil)
}

// TestDiscoveredNames_OrderAndFiltering covers state read back from the database
// session service, where the stored int arrives as a float64, ties from calls in
// one response, a value of an unexpected type, and keys that belong to something
// else.
func TestDiscoveredNames_OrderAndFiltering(t *testing.T) {
	state := newFakeState(map[string]any{
		stateKeyPrefix + "test_agent:z_tool":       float64(0),
		stateKeyPrefix + "test_agent:e_tool":       1,
		stateKeyPrefix + "test_agent:c_tool":       1,
		stateKeyPrefix + "test_agent:b_tool":       1,
		stateKeyPrefix + "test_agent:d_tool":       1,
		stateKeyPrefix + "test_agent:a_tool":       1,
		stateKeyPrefix + "test_agent:bad_value":    "1",
		stateKeyPrefix + "test_agent:":             0,
		stateKeyPrefix + "test_agent_2:other_tool": 0,
		"unrelated": 0,
	})
	want := []string{"z_tool", "a_tool", "b_tool", "c_tool", "d_tool", "e_tool", "bad_value"}
	if diff := cmp.Diff(want, discoveredNames(state, testAgent)); diff != "" {
		t.Errorf("discoveredNames() mismatch (-want +got):\n%s", diff)
	}
}

// TestRevealTools_KeepsDiscoveriesPerAgent checks that agents sharing a session
// see only their own discoveries. Agent and tool names may both contain a
// colon, so agent "a" revealing "b:t" must not hand agent "a:b" the tool "t".
func TestRevealTools_KeepsDiscoveriesPerAgent(t *testing.T) {
	agents := []string{"a", "a:b", "other"}
	for _, tc := range []struct{ agent, tool string }{
		{"a", "b:t"},
		{"a:b", "t"},
		{"other", "github:create_issue"},
	} {
		state := newFakeState(nil)
		if err := RevealTools(newAgentToolCtx(state, tc.agent), tc.tool); err != nil {
			t.Fatalf("RevealTools(%q) error = %v", tc.agent, err)
		}
		for _, agentName := range agents {
			var want []string
			if agentName == tc.agent {
				want = []string{tc.tool}
			}
			if diff := cmp.Diff(want, discoveredNames(state, agentName), cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("after %q revealed %q, discoveredNames(%q) mismatch (-want +got):\n%s", tc.agent, tc.tool, agentName, diff)
			}
		}
	}
}

// TestGatingToolset_SkipsDiscoveredToolsTheFlowCannotPack checks that a revealed
// tool without ProcessRequest stays out of Tools. The flow rejects such a tool
// and fails the whole step, and RevealTools accepts any name.
func TestGatingToolset_SkipsDiscoveredToolsTheFlowCannotPack(t *testing.T) {
	base := &staticToolset{tools: []tool.Tool{
		&stubTool{name: "packable_tool", desc: "can be called"},
		&nonPackableStubTool{name: "non_packable_tool", desc: "cannot be called"},
	}}
	ts := mustNew(t, base, Config{})

	names := mustToolNames(t, ts, newCtx(discoveredState(t, "packable_tool", "non_packable_tool")))
	checkNames(t, names, []string{"packable_tool"}, []string{"non_packable_tool"})
}

func TestNew_RejectsInvalidConfig(t *testing.T) {
	if _, err := New(nil, Config{}); err == nil {
		t.Error("New(nil) error = nil, want an error")
	}
	if _, err := New(&staticToolset{}, Config{CoreToolNames: []string{ToolName}}); err == nil {
		t.Errorf("New() with core tool %q error = nil, want an error", ToolName)
	}
}

// TestSearch_BaseToolNamedSearchToolsIsNeverExposed runs a base catalog that
// contains a tool named search_tools through the real runner. Selecting it must
// not reveal it, and a recorded name must not make Tools return a second
// search_tools, which the framework rejects on every later step.
func TestSearch_BaseToolNamedSearchToolsIsNeverExposed(t *testing.T) {
	newTool := func(name string) tool.Tool {
		ft, err := functiontool.New(functiontool.Config{Name: name, Description: name},
			func(_ agent.Context, _ struct{}) (struct{}, error) { return struct{}{}, nil })
		if err != nil {
			t.Fatalf("functiontool.New(%q) error = %v", name, err)
		}
		return ft
	}
	base := &staticToolset{tools: []tool.Tool{newTool(ToolName), newTool("other_tool")}}
	gated, err := New(base, Config{})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	llm := &parallelCallModel{calls: []*genai.FunctionCall{
		{ID: "1", Name: ToolName, Args: map[string]any{"query": "select:" + ToolName}},
	}}
	a, err := llmagent.New(llmagent.Config{Name: "agent", Model: llm, Toolsets: []tool.Toolset{gated}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	for turn := 1; turn <= 2; turn++ {
		for _, err := range r.Run(t.Context(), "user", "session", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("turn %d: Run() error = %v", turn, err)
			}
		}
	}

	out := mustSearch(t, newToolCtx(newFakeState(nil)), "select:"+ToolName, base, gated.(*gatingToolset), 8)
	checkNames(t, matchNames(out.Matches), nil, []string{ToolName})

	// A name recorded before, e.g. by RevealTools, must not duplicate it either.
	names := mustToolNames(t, gated, newCtx(discoveredState(t, ToolName, "other_tool")))
	if diff := cmp.Diff([]string{ToolName, "other_tool"}, names); diff != "" {
		t.Errorf("Tools() mismatch (-want +got):\n%s", diff)
	}
}

func TestGatingToolset_ToolsReturnsBaseError(t *testing.T) {
	errCatalog := errors.New("catalog down")
	gts := mustNew(t, &failingToolset{err: errCatalog}, Config{})
	if _, err := gts.Tools(newCtx(newFakeState(nil))); !errors.Is(err, errCatalog) {
		t.Errorf("Tools() error = %v, want it to wrap %v", err, errCatalog)
	}
}

// TestSearch_SkipsToolsWithoutFunctionDeclaration checks that a model-side
// built-in, which packs but declares no function, is not offered by search: a
// function call to it would be rejected as an unknown tool.
func TestSearch_SkipsToolsWithoutFunctionDeclaration(t *testing.T) {
	base := &staticToolset{tools: []tool.Tool{
		geminitool.GoogleSearch{},
		&stubTool{name: "web_lookup", desc: "search the web"},
	}}
	gts := mustNew(t, base, Config{})
	for _, query := range []string{"select:google_search", "google search web"} {
		out := mustSearch(t, newToolCtx(newFakeState(nil)), query, base, gts, 8)
		checkNames(t, matchNames(out.Matches), nil, []string{"google_search"})
	}
}

// typedNilSchemaTool declares a typed-nil *jsonschema.Schema.
type typedNilSchemaTool struct{ stubTool }

func (*typedNilSchemaTool) Declaration() *genai.FunctionDeclaration {
	var schema *jsonschema.Schema
	return &genai.FunctionDeclaration{Name: "typed_nil", ParametersJsonSchema: schema}
}

func TestSearch_TypedNilSchemaDoesNotPanic(t *testing.T) {
	base := &staticToolset{tools: []tool.Tool{&typedNilSchemaTool{stubTool{name: "typed_nil", desc: "typed nil schema"}}}}
	gts := mustNew(t, base, Config{})
	out := mustSearch(t, newToolCtx(newFakeState(nil)), "typed", base, gts, 8)
	checkNames(t, matchNames(out.Matches), []string{"typed_nil"}, nil)
}

func TestNew_CopiesSkillAnnotations(t *testing.T) {
	base := &staticToolset{tools: makeTools(toolDef{"list_folders", "list folders"})}
	annotations := map[string]string{"list_folders": "folder-browser"}
	gts := mustNew(t, base, Config{SkillAnnotations: annotations})
	annotations["list_folders"] = "changed-after-new"

	tools, err := gts.Tools(newCtx(newFakeState(nil)))
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	var search tool.Tool
	for _, tl := range tools {
		if tl.Name() == ToolName {
			search = tl
		}
	}
	runner, ok := search.(interface {
		Run(agent.Context, any) (map[string]any, error)
	})
	if !ok {
		t.Fatal("search tool has no Run method")
	}
	got, err := runner.Run(newToolCtx(newFakeState(nil)), map[string]any{"query": "select:list_folders"})
	if err != nil {
		t.Fatalf("search Run() error = %v", err)
	}
	if !strings.Contains(fmt.Sprint(got), "folder-browser") || strings.Contains(fmt.Sprint(got), "changed-after-new") {
		t.Errorf("search result = %v, want the connected skill given to New", got)
	}
}

// TestSearch_OverlappingParallelCallsNeverReportNoMatch sends two overlapping
// select: calls in one model response through the real runner. Whichever call
// runs second must say the tool is already available, not that nothing matched.
func TestSearch_OverlappingParallelCallsNeverReportNoMatch(t *testing.T) {
	newTool := func(name string) tool.Tool {
		ft, err := functiontool.New(functiontool.Config{Name: name, Description: name},
			func(_ agent.Context, _ struct{}) (struct{}, error) { return struct{}{}, nil })
		if err != nil {
			t.Fatalf("functiontool.New(%q) error = %v", name, err)
		}
		return ft
	}
	for i := 0; i < 20; i++ {
		gated, err := New(&staticToolset{tools: []tool.Tool{newTool("zebra_tool"), newTool("apple_tool")}}, Config{})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		llm := &parallelCallModel{calls: []*genai.FunctionCall{
			{ID: "1", Name: ToolName, Args: map[string]any{"query": "select:zebra_tool"}},
			{ID: "2", Name: ToolName, Args: map[string]any{"query": "select:zebra_tool,apple_tool"}},
		}}
		a, err := llmagent.New(llmagent.Config{Name: "agent", Model: llm, Toolsets: []tool.Toolset{gated}})
		if err != nil {
			t.Fatalf("llmagent.New() error = %v", err)
		}
		r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
		if err != nil {
			t.Fatalf("runner.New() error = %v", err)
		}
		for ev, err := range r.Run(t.Context(), "user", "session", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if ev == nil || ev.LLMResponse.Content == nil {
				continue
			}
			for _, p := range ev.LLMResponse.Content.Parts {
				if fr := p.FunctionResponse; fr != nil && strings.Contains(fmt.Sprint(fr.Response["note"]), "no tools matched") {
					t.Fatalf("run %d: a search_tools call reported no match: %v", i, fr.Response)
				}
			}
		}
		checkNames(t, llm.declared, []string{"zebra_tool", "apple_tool"}, nil)
	}
}
