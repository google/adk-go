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

package tool

import (
	"fmt"
	"iter"
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
)

// WithToolNamePrefix returns a Toolset whose tools are the tools of ts, each
// offered to the model and dispatched as "<prefix>_<name>".
//
// An empty prefix returns ts unchanged. A nil ts panics with "toolset must not
// be nil", even when prefix is empty.
//
// Each call to Tools lists ts again and returns, in the same order, one
// prefixed tool per tool of ts. A prefixed tool keeps everything else of the
// tool it stands for: its description, IsLongRunning, Run or RunStream, its
// request hook, its DefersResponse, DisplayResultOnSkipSummarization and
// ResponseScheduling answers and, for a tool that runs an agent, such as
// agenttool's, the agent it wraps. Its declaration is a copy of the original's with the prefixed name,
// or nil when the original's is nil. The original tool and its declaration are
// never modified, so listing any number of times never prefixes a name twice.
// Running a prefixed tool's request hook runs the original's and registers the
// prefixed tool, under its prefixed name, in place of the original. It fails
// with a duplicate tool error when the request already holds that name.
//
// When combined with [FilterToolset], the order of wrapping determines which
// names the predicate sees: wrapping FilterToolset with WithToolNamePrefix
// filters on the original unprefixed names (matching adk-python's tool_filter),
// whereas wrapping WithToolNamePrefix with FilterToolset filters on the
// prefixed names. When combined with [WithConfirmation], the outer wrapper's
// tool name is the name handed to the [ConfirmationProvider].
//
//	tool.WithToolNamePrefix(tool.FilterToolset(ts, tool.AllowedToolsPredicate([]string{"search"})), "docs") // offers docs_search
//	tool.FilterToolset(tool.WithToolNamePrefix(ts, "docs"), tool.AllowedToolsPredicate([]string{"docs_search"})) // offers docs_search
//
// The returned toolset's request hook calls the request hook of ts, when ts has
// one. Close on the returned toolset closes ts when that implements
// [io.Closer], and Unwrap returns ts. The returned toolset holds no state of
// its own, so it is safe for concurrent use when ts is.
func WithToolNamePrefix(ts Toolset, prefix string) Toolset {
	if ts == nil {
		panic("toolset must not be nil")
	}
	if prefix == "" {
		return ts
	}
	return &prefixedToolset{toolset: ts, prefix: prefix}
}

// requestProcessor mirrors adk/internal/toolinternal.RequestProcessor. It is
// declared here rather than imported because toolinternal imports this package.
type requestProcessor interface {
	ProcessRequest(ctx agent.Context, req *model.LLMRequest) error
}

// agentWrapper mirrors the method of adk/internal/toolinternal.AgentWrapper,
// for the same reason.
type agentWrapper interface {
	WrappedAgent() agent.Agent
}

type declarer interface {
	Declaration() *genai.FunctionDeclaration
}

type prefixedToolset struct {
	toolset Toolset
	prefix  string
}

func (p *prefixedToolset) Name() string { return p.toolset.Name() }

func (p *prefixedToolset) Tools(ctx agent.ReadonlyContext) ([]Tool, error) {
	tools, err := p.toolset.Tools(ctx)
	if err != nil {
		return nil, err
	}
	prefixed := make([]Tool, 0, len(tools))
	for _, t := range tools {
		prefixed = append(prefixed, newPrefixedTool(t, p.prefix+"_"+t.Name()))
	}
	return prefixed, nil
}

func (p *prefixedToolset) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return processToolsetRequest(p.toolset, ctx, req)
}

func (p *prefixedToolset) Close() error { return utils.CloseIfCloser(p.toolset) }

func (p *prefixedToolset) Unwrap() Toolset { return p.toolset }

// newPrefixedTool returns the stand-in of t named name. The stand-in has the
// method set the flow dispatches on that t has: streaming is checked first, as
// the flow does, so a tool with both Run and RunStream is run as a stream. A
// stand-in of a tool that wraps an agent wraps it too, so the walks that close
// a runner's toolsets reach that agent through the stand-in.
func newPrefixedTool(t Tool, name string) Tool {
	base := &prefixedTool{Tool: t, name: name}
	aw, isAgent := t.(agentWrapper)
	switch rt := t.(type) {
	case streamingRunnableTool:
		st := &prefixedStreamingTool{prefixedDeclaredTool: &prefixedDeclaredTool{prefixedTool: base, decl: rt}, fn: rt}
		if isAgent {
			return &prefixedAgentStreamingTool{prefixedStreamingTool: st, aw: aw}
		}
		return st
	case runnableTool:
		ft := &prefixedFunctionTool{prefixedDeclaredTool: &prefixedDeclaredTool{prefixedTool: base, decl: rt}, fn: rt}
		if isAgent {
			return &prefixedAgentFunctionTool{prefixedFunctionTool: ft, aw: aw}
		}
		return ft
	case declarer:
		dt := &prefixedDeclaredTool{prefixedTool: base, decl: rt}
		if isAgent {
			return &prefixedAgentDeclaredTool{prefixedDeclaredTool: dt, aw: aw}
		}
		return dt
	default:
		if isAgent {
			return &prefixedAgentTool{prefixedTool: base, aw: aw}
		}
		return base
	}
}

// prefixedTool is the stand-in of a tool without a Declaration method, such as
// the model-side built-ins in tool/geminitool, and the base of the other
// stand-ins. It embeds the Tool interface so that only Description and
// IsLongRunning are promoted from the original.
type prefixedTool struct {
	Tool
	name string
}

func (p *prefixedTool) Name() string { return p.name }

func (p *prefixedTool) DefersResponse() bool { return defersResponse(p.Tool) }

func (p *prefixedTool) DisplayResultOnSkipSummarization() bool {
	return displaysResultOnSkipSummarization(p.Tool)
}

func (p *prefixedTool) ResponseScheduling() genai.FunctionResponseScheduling {
	return responseScheduling(p.Tool)
}

func (p *prefixedTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return p.process(p, ctx, req)
}

// process runs the original's request hook on req and moves what the hook
// registered under the original's name to the prefixed name: the entry in
// req.Tools becomes self, the stand-in the flow dispatches to, and each
// declaration the hook appended under the original's name is replaced by a
// copy that carries the prefixed name. Other edits the hook makes to req, such
// as instructions, are left in place.
//
// An entry req already holds under the original's name is hidden from the
// hook and restored afterwards, so a tool offered both as itself and through a
// prefix does not fail as a duplicate of itself.
func (p *prefixedTool) process(self Tool, ctx agent.Context, req *model.LLMRequest) error {
	rp, ok := p.Tool.(requestProcessor)
	if !ok {
		return fmt.Errorf("tool %q does not implement RequestProcessor() method", p.name)
	}
	orig := p.Tool.Name()
	prior, hadPrior := req.Tools[orig]
	if hadPrior {
		delete(req.Tools, orig)
	}
	marks := declarationLengths(req)
	err := rp.ProcessRequest(ctx, req)
	_, packed := req.Tools[orig]
	if packed {
		delete(req.Tools, orig)
	}
	if hadPrior {
		req.Tools[orig] = prior
	}
	if err != nil {
		return err
	}
	if packed {
		if _, dup := req.Tools[p.name]; dup {
			marks.drop(req, orig)
			return fmt.Errorf("duplicate tool: %q", p.name)
		}
		req.Tools[p.name] = self
	}
	marks.rename(req, orig, p.name)
	return nil
}

// prefixedDeclaredTool is the stand-in of a tool that has a Declaration method
// and neither Run nor RunStream.
type prefixedDeclaredTool struct {
	*prefixedTool
	decl declarer
}

func (p *prefixedDeclaredTool) Declaration() *genai.FunctionDeclaration {
	return renamedDeclaration(p.decl.Declaration(), p.name)
}

func (p *prefixedDeclaredTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return p.process(p, ctx, req)
}

// prefixedFunctionTool is the stand-in of a function tool.
type prefixedFunctionTool struct {
	*prefixedDeclaredTool
	fn runnableTool
}

func (p *prefixedFunctionTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	return p.fn.Run(ctx, args)
}

func (p *prefixedFunctionTool) RunRaw(ctx agent.Context, args any) (any, error) {
	return runRaw(p.fn, ctx, args)
}

func (p *prefixedFunctionTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return p.process(p, ctx, req)
}

// prefixedStreamingTool is the stand-in of a streaming function tool.
type prefixedStreamingTool struct {
	*prefixedDeclaredTool
	fn streamingRunnableTool
}

func (p *prefixedStreamingTool) RunStream(ctx agent.Context, args any) iter.Seq2[string, error] {
	return p.fn.RunStream(ctx, args)
}

func (p *prefixedStreamingTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return p.process(p, ctx, req)
}

// The agent variants add WrappedAgent to the stand-in of each kind. Each
// defines its own ProcessRequest so that the entry registered in the request
// is the variant itself, the object Tools returned.

type prefixedAgentStreamingTool struct {
	*prefixedStreamingTool
	aw agentWrapper
}

func (p *prefixedAgentStreamingTool) WrappedAgent() agent.Agent { return p.aw.WrappedAgent() }

func (p *prefixedAgentStreamingTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return p.process(p, ctx, req)
}

type prefixedAgentFunctionTool struct {
	*prefixedFunctionTool
	aw agentWrapper
}

func (p *prefixedAgentFunctionTool) WrappedAgent() agent.Agent { return p.aw.WrappedAgent() }

func (p *prefixedAgentFunctionTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return p.process(p, ctx, req)
}

type prefixedAgentDeclaredTool struct {
	*prefixedDeclaredTool
	aw agentWrapper
}

func (p *prefixedAgentDeclaredTool) WrappedAgent() agent.Agent { return p.aw.WrappedAgent() }

func (p *prefixedAgentDeclaredTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return p.process(p, ctx, req)
}

type prefixedAgentTool struct {
	*prefixedTool
	aw agentWrapper
}

func (p *prefixedAgentTool) WrappedAgent() agent.Agent { return p.aw.WrappedAgent() }

func (p *prefixedAgentTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	return p.process(p, ctx, req)
}

// renamedDeclaration returns a shallow copy of d named name, or nil when d is
// nil. The original may be a declaration its tool stores and returns on every
// call, so it is never written.
func renamedDeclaration(d *genai.FunctionDeclaration, name string) *genai.FunctionDeclaration {
	if d == nil {
		return nil
	}
	c := *d
	c.Name = name
	return &c
}

// declarationMarks records, for each entry of a request's Config.Tools, how
// many function declarations it held, so the declarations a request hook
// appends afterwards can be found. A hook that packs a tool appends its
// declaration to an existing entry or appends a new entry (toolutils.PackTool),
// so every declaration it adds lies at or beyond the recorded length, or in an
// entry past the recorded ones.
type declarationMarks []int

func declarationLengths(req *model.LLMRequest) declarationMarks {
	if req == nil || req.Config == nil || len(req.Config.Tools) == 0 {
		return nil
	}
	marks := make(declarationMarks, len(req.Config.Tools))
	for i, t := range req.Config.Tools {
		if t != nil {
			marks[i] = len(t.FunctionDeclarations)
		}
	}
	return marks
}

// rename replaces each declaration named from that was added since m was
// taken by a copy named to.
func (m declarationMarks) rename(req *model.LLMRequest, from, to string) {
	m.editAdded(req, func(added []*genai.FunctionDeclaration) []*genai.FunctionDeclaration {
		for i, d := range added {
			if d != nil && d.Name == from {
				added[i] = renamedDeclaration(d, to)
			}
		}
		return added
	})
}

// drop removes each declaration named name that was added since m was taken.
func (m declarationMarks) drop(req *model.LLMRequest, name string) {
	m.editAdded(req, func(added []*genai.FunctionDeclaration) []*genai.FunctionDeclaration {
		return slices.DeleteFunc(added, func(d *genai.FunctionDeclaration) bool {
			return d != nil && d.Name == name
		})
	})
}

// editAdded hands edit the declarations added to each entry of
// req.Config.Tools since m was taken, and keeps the slice edit returns, which
// must be a prefix of the one it was given, edited in place.
func (m declarationMarks) editAdded(req *model.LLMRequest, edit func([]*genai.FunctionDeclaration) []*genai.FunctionDeclaration) {
	if req.Config == nil {
		return
	}
	for i, t := range req.Config.Tools {
		if t == nil {
			continue
		}
		start := 0
		if i < len(m) {
			start = m[i]
		}
		if start > len(t.FunctionDeclarations) {
			continue
		}
		kept := edit(t.FunctionDeclarations[start:])
		t.FunctionDeclarations = t.FunctionDeclarations[:start+len(kept)]
	}
}
