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

package services

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"sync"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/adkcontext"
	llmagentinternal "google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// toolsetWaitBudget is how long a graph waits for its toolsets to list their
// tools, counted from the start of the graph. Every toolset in the tree starts
// listing at once, so each gets the whole budget. A toolset still listing when
// it runs out is left out of that graph and logged.
//
// The graph only stops waiting; it never cancels a listing. Canceling an MCP
// toolset while it connects makes the MCP SDK close the session and kill a
// stdio server's process, and mcptoolset cannot start that process again, so
// every later run of the agent would fail. Left alone, the listing finishes in
// the background and the agent's runs reuse the session it opened. Because
// nothing is canceled, the graph's response is bounded by this budget alone,
// with no MCP shutdown time on top, which keeps it well inside the web
// launcher's default 15s write timeout.
//
// It is a variable so tests can shorten it.
var toolsetWaitBudget = 5 * time.Second

// errStillListing is recorded for a toolset that was still listing its tools
// when the graph stopped waiting for it.
var errStillListing = errors.New("toolset was still listing its tools")

// toolsetUse is one agent's use of one toolset. A toolset can list different
// tools for different agents (tool.FilterToolset on ctx.AgentName(), for
// example), so its tools are listed per use.
type toolsetUse struct {
	agent   agent.Agent
	toolset tool.Toolset
}

// toolsetListing is one call to Toolset.Tools. done is closed once the call has
// returned, and tools and err are set before that.
type toolsetListing struct {
	done  chan struct{}
	tools []tool.Tool
	err   error
}

var (
	runningMu sync.Mutex
	// running holds the listings that have not returned yet, so a graph built
	// while one is running waits for it instead of starting another. Without
	// it, an MCP server that never answers would leave one more goroutine
	// behind on every graph request. A listing is removed once it returns.
	running = map[toolsetUse]*toolsetListing{}
)

// keyable reports whether use can be a map key. A toolset or agent of an
// uncomparable type, such as a struct holding a slice, would panic as one.
func (use toolsetUse) keyable() (ok bool) {
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	return use == use
}

// listTools returns the listing of use's tools, joining the one already running
// for use, if any, or starting one.
//
// The listing runs on its own goroutine under a context that keeps ctx's values
// but not its cancellation or deadline, so neither the graph's budget nor the
// request ending can cancel it. See [toolsetWaitBudget].
func listTools(ctx context.Context, use toolsetUse) *toolsetListing {
	shared := use.keyable()
	if shared {
		runningMu.Lock()
		defer runningMu.Unlock()
		if l, ok := running[use]; ok {
			return l
		}
	}
	l := &toolsetListing{done: make(chan struct{})}
	if shared {
		running[use] = l
	}
	listCtx := newGraphContext(context.WithoutCancel(ctx), use.agent)
	go func() {
		defer func() {
			// A panic here would end the process: nothing above this goroutine
			// can recover it.
			if r := recover(); r != nil {
				l.tools, l.err = nil, &toolsetPanicError{value: r}
			}
			if shared {
				runningMu.Lock()
				delete(running, use)
				runningMu.Unlock()
			}
			close(l.done)
		}()
		l.tools, l.err = use.toolset.Tools(listCtx)
	}()
	return l
}

// toolsetLister lists the tools of the toolsets in one graph.
type toolsetLister struct {
	// ctx is the request the graph is built for.
	ctx context.Context
	// waiting is done once the graph has waited toolsetWaitBudget.
	waiting context.Context
	// started holds the listings start began, by use.
	started map[toolsetUse]*toolsetListing
}

// newToolsetLister returns a lister for one graph built for ctx. Its budget
// starts now. Call stop once the graph is built.
func newToolsetLister(ctx context.Context) (l *toolsetLister, stop context.CancelFunc) {
	waiting, stop := context.WithTimeout(ctx, toolsetWaitBudget)
	return &toolsetLister{ctx: ctx, waiting: waiting, started: map[toolsetUse]*toolsetListing{}}, stop
}

// start begins listing the toolsets of every llmagent in the tree under root,
// so that they list in parallel rather than one after another as the graph is
// drawn.
func (l *toolsetLister) start(root agent.Agent) {
	if root == nil {
		return
	}
	if llmAgent, ok := root.(llmagentinternal.Agent); ok {
		for _, toolset := range llmagentinternal.Reveal(llmAgent).Toolsets {
			use := toolsetUse{agent: root, toolset: toolset}
			if toolset == nil || !use.keyable() {
				continue
			}
			if _, ok := l.started[use]; !ok {
				l.started[use] = listTools(l.ctx, use)
			}
		}
	}
	for _, sub := range root.SubAgents() {
		l.start(sub)
	}
}

// tools returns the tools toolset lists for a, waiting for them until the
// graph's budget runs out. It returns errStillListing if they are not ready by
// then.
func (l *toolsetLister) tools(a agent.Agent, toolset tool.Toolset) ([]tool.Tool, error) {
	use := toolsetUse{agent: a, toolset: toolset}
	var listing *toolsetListing
	if use.keyable() {
		listing = l.started[use]
	}
	if listing == nil {
		listing = listTools(l.ctx, use)
	}
	select {
	case <-listing.done:
		return listing.tools, listing.err
	case <-l.waiting.Done():
	}
	// Both can be ready at once, and select picks either.
	select {
	case <-listing.done:
		return listing.tools, listing.err
	default:
		return nil, errStillListing
	}
}

// toolsetPanicError is a panic recovered from Toolset.Tools.
type toolsetPanicError struct {
	value any
}

func (e *toolsetPanicError) Error() string {
	return fmt.Sprintf("toolset.Tools panicked: %v", e.value)
}

// logSkippedToolset records that the graph of a leaves out toolset because
// listing its tools failed with err. The graph still renders, so the caller
// gets no error, and without this line a developer has no way to see why the
// toolset's tools are missing.
//
// It logs types, not the error or panic value: those can carry a server URL
// or credentials.
func logSkippedToolset(ctx context.Context, a agent.Agent, toolset tool.Toolset, err error) {
	kind, errType := "error", fmt.Sprintf("%T", err)
	var p *toolsetPanicError
	switch {
	case errors.Is(err, errStillListing):
		kind, errType = "timeout", ""
	case errors.As(err, &p):
		kind, errType = "panic", fmt.Sprintf("%T", p.value)
	}
	slog.WarnContext(ctx, "agent graph: skipping a toolset that could not list its tools", //nolint:forbidigo // the graph still renders without this toolset, so there is no error to return
		"agent", a.Name(),
		"toolset", toolsetName(toolset),
		"toolset_type", fmt.Sprintf("%T", toolset),
		"kind", kind,
		"error_type", errType,
	)
}

// toolsetName returns toolset.Name(), or "" if Name panics. A wrapper around a
// nil toolset, such as tool.WithConfirmation(nil, ...), passes the nil check
// and then panics in both Tools and Name.
func toolsetName(toolset tool.Toolset) (name string) {
	defer func() {
		if recover() != nil {
			name = ""
		}
	}()
	return toolset.Name()
}

// graphContext is the [agent.ReadonlyContext] a graph lists toolsets with.
// The graph is drawn outside any run, so there is no user content, session,
// user or invocation. Those read as empty instead of panicking, the way
// exp/appinfo lists toolsets, so a toolset that reads them, such as
// tool/toolsearch reading state, still lists its tools.
//
// It reports no invocation identity, even if its parent carries one, so a
// toolset that acts as a user, such as an MCP toolset with per-user auth,
// fails closed instead of acting as whoever requested the graph.
type graphContext struct {
	context.Context
	agentName string
}

func newGraphContext(ctx context.Context, a agent.Agent) graphContext {
	return graphContext{Context: ctx, agentName: a.Name()}
}

func (c graphContext) Value(key any) any {
	if key == adkcontext.IdentityKey {
		return nil
	}
	return c.Context.Value(key)
}

func (c graphContext) AgentName() string { return c.agentName }

func (graphContext) UserContent() *genai.Content { return nil }
func (graphContext) InvocationID() string        { return "" }
func (graphContext) UserID() string              { return "" }
func (graphContext) AppName() string             { return "" }
func (graphContext) SessionID() string           { return "" }
func (graphContext) Branch() string              { return "" }

func (graphContext) ReadonlyState() session.ReadonlyState { return emptyState{} }

var _ agent.ReadonlyContext = graphContext{}

// emptyState is a [session.ReadonlyState] that holds nothing.
type emptyState struct{}

func (emptyState) Get(string) (any, error) { return nil, session.ErrStateKeyNotExist }

func (emptyState) All() iter.Seq2[string, any] {
	return func(func(string, any) bool) {}
}
