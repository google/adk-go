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

package flowtool

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// nodeResult is the per-node entry in the outputs map returned to the LLM.
type nodeResult struct {
	Output string `json:"output"`
	Error  string `json:"error,omitempty"`
}

// outputs is the path-keyed accumulator. Concurrent writes go through Set.
type outputs struct {
	mu   sync.Mutex
	data map[string]nodeResult
	last string // path of the last completed node, used for final_output
}

func newOutputs() *outputs {
	return &outputs{data: map[string]nodeResult{}}
}

func (o *outputs) Set(path string, r nodeResult) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.data[path] = r
	if r.Error == "" {
		o.last = path
	}
}

func (o *outputs) Get(path string) (nodeResult, bool) {
	o.mu.Lock()
	defer o.mu.Unlock()
	r, ok := o.data[path]
	return r, ok
}

func (o *outputs) snapshot() map[string]any {
	o.mu.Lock()
	defer o.mu.Unlock()
	out := make(map[string]any, len(o.data))
	for k, v := range o.data {
		entry := map[string]any{"output": v.Output}
		if v.Error != "" {
			entry["error"] = v.Error
		}
		out[k] = entry
	}
	return out
}

func (o *outputs) finalOutput() string {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.last == "" {
		return ""
	}
	return o.data[o.last].Output
}

// templateRe matches {{nodes.<path>.output}} where <path> may contain
// letters, digits, underscores, dots, and bracketed indices.
var templateRe = regexp.MustCompile(`\{\{\s*nodes\.([^}]+?)\.output\s*\}\}`)

func (o *outputs) renderTemplate(s string) (string, error) {
	var firstErr error
	out := templateRe.ReplaceAllStringFunc(s, func(match string) string {
		m := templateRe.FindStringSubmatch(match)
		if len(m) != 2 {
			return match
		}
		path := m[1]
		r, ok := o.Get(path)
		if !ok {
			if firstErr == nil {
				firstErr = fmt.Errorf("flowtool: template references unknown path %q", path)
			}
			return match
		}
		if r.Error != "" {
			return ""
		}
		return r.Output
	})
	if firstErr != nil {
		return "", firstErr
	}
	return out, nil
}

// executor runs a parsed + validated spec, populating an outputs map and
// returning the final output text.
type executor struct {
	tool      *flowTool
	toolCtx   tool.Context
	parentCtx context.Context
	stateSeed map[string]any
	outputs   *outputs
}

// run dispatches on the spec kind. The supplied input is the value passed
// from upstream (initial_input at the root). Returns the output of this
// subtree (or "" on failure).
func (e *executor) run(ctx context.Context, s *Spec, input string) (string, error) {
	switch s.Type {
	case KindAgent:
		return e.runAgent(ctx, s, input)
	case KindSeq:
		return e.runSeq(ctx, s, input)
	case KindParallel:
		return e.runParallel(ctx, s, input)
	default:
		return "", fmt.Errorf("flowtool: unknown spec type %q", s.Type)
	}
}

func (e *executor) runSeq(ctx context.Context, s *Spec, input string) (string, error) {
	cur := input
	for i := range s.Nodes {
		if ctx.Err() != nil {
			return cur, ctx.Err()
		}
		out, err := e.run(ctx, &s.Nodes[i], cur)
		if err != nil {
			return cur, err
		}
		cur = out
	}
	return cur, nil
}

func (e *executor) runParallel(ctx context.Context, s *Spec, input string) (string, error) {
	type result struct {
		path   string
		output string
		err    error
	}
	results := make([]result, len(s.Nodes))

	concurrency := e.tool.maxConcurrency
	if concurrency <= 0 {
		concurrency = len(s.Nodes)
	}
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	for i := range s.Nodes {
		i := i
		child := &s.Nodes[i]
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			out, err := e.run(ctx, child, input)
			results[i] = result{path: child.Path, output: out, err: err}
		}()
	}
	wg.Wait()

	// Best-effort: collect successes + failures into a labeled map output
	// for downstream nodes. Errors do not abort siblings.
	var b strings.Builder
	for _, r := range results {
		if r.err != nil {
			fmt.Fprintf(&b, "## %s\n[error: %v]\n\n", r.path, r.err)
			continue
		}
		fmt.Fprintf(&b, "## %s\n%s\n\n", r.path, r.output)
	}
	return strings.TrimRight(b.String(), "\n"), nil
}

func (e *executor) runAgent(ctx context.Context, s *Spec, input string) (string, error) {
	if ctx.Err() != nil {
		e.outputs.Set(s.Path, nodeResult{Error: ctx.Err().Error()})
		return "", ctx.Err()
	}

	rendered := input
	if s.Input != "" {
		r, err := e.outputs.renderTemplate(s.Input)
		if err != nil {
			e.outputs.Set(s.Path, nodeResult{Error: err.Error()})
			return "", err
		}
		rendered = r
	}

	a, err := e.tool.resolveAgent(s.Agent)
	if err != nil {
		e.outputs.Set(s.Path, nodeResult{Error: err.Error()})
		return "", err
	}

	out, err := e.invokeAgent(ctx, a, rendered)
	if err != nil {
		e.outputs.Set(s.Path, nodeResult{Output: out, Error: err.Error()})
		return out, err
	}
	e.outputs.Set(s.Path, nodeResult{Output: out})
	return out, nil
}

// invokeAgent runs a single agent.Agent in an isolated in-memory session
// and returns the concatenated text of its last response.
//
// Mirrors tool/agenttool's pattern. State inheritance is read-only and
// gated by the constructor option.
func (e *executor) invokeAgent(ctx context.Context, a agent.Agent, input string) (string, error) {
	sessionService := session.InMemoryService()

	r, err := runner.New(runner.Config{
		AppName:         a.Name(),
		Agent:           a,
		SessionService:  sessionService,
		ArtifactService: artifact.InMemoryService(),
		MemoryService:   memory.InMemoryService(),
	})
	if err != nil {
		return "", fmt.Errorf("flowtool: runner.New for %q: %w", a.Name(), err)
	}

	state := map[string]any{}
	if e.tool.inheritState {
		for k, v := range e.stateSeed {
			state[k] = v
		}
	}

	subSession, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: a.Name(),
		UserID:  e.toolCtx.UserID(),
		State:   state,
	})
	if err != nil {
		return "", fmt.Errorf("flowtool: create sub-session for %q: %w", a.Name(), err)
	}

	content := genai.NewContentFromText(input, genai.RoleUser)
	eventCh := r.Run(ctx, subSession.Session.UserID(), subSession.Session.ID(), content, agent.RunConfig{
		StreamingMode: agent.StreamingModeSSE,
	})

	var lastText string
	for ev, err := range eventCh {
		if err != nil {
			return lastText, err
		}
		if ev == nil {
			continue
		}
		if ev.ErrorCode != "" || ev.ErrorMessage != "" {
			return lastText, fmt.Errorf("agent %q: code=%q msg=%q", a.Name(), ev.ErrorCode, ev.ErrorMessage)
		}
		if ev.LLMResponse.Partial {
			continue
		}
		if t := extractText(ev.LLMResponse.Content); t != "" {
			lastText = t
		}
	}
	return lastText, nil
}

func extractText(c *genai.Content) string {
	if c == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range c.Parts {
		if p != nil && p.Text != "" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}
