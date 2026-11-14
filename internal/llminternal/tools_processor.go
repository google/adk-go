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

package llminternal

import (
	"errors"
	"fmt"
	"iter"
	"sync"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// toolProcessor populates f.Tools on every step by re-evaluating each
// Toolset. Toolsets may return different tools based on session state that
// was modified by an earlier step in the same Run(), so the tool list must
// be rebuilt before each model call rather than being cached across steps.
// The toolsets are listed concurrently, and a toolset whose listing fails is
// skipped for the step (see ResolveTools). On a listing error it does not
// change f.Tools.
func toolProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		llmAgent, ok := ctx.Agent().(Agent)
		if !ok {
			yield(nil, fmt.Errorf("agent %v is not an LLMAgent", ctx.Agent().Name()))
			return
		}
		own, groups, err := resolveToolGroups(icontext.NewReadonlyContext(ctx), llmAgent)
		if err != nil {
			yield(nil, err)
			return
		}
		f.Tools = flattenTools(own, groups)
	}
}

// ResolveTools returns the tools of the LLM agent a for ctx: its own Tools,
// then the tools each of its Toolsets lists, in declared order, in a new
// slice. Toolsets are listed concurrently, each with ctx, and every
// listing's result lands at its declared position. A toolset whose listing
// fails contributes no tools and the call goes on. When a listing failed and
// ctx is not nil and ctx.Err() is non-nil, the call returns an error instead:
// the failure of the first such toolset in declared order, as
// "failed to extract tools from the tool set %q: %w", joined with ctx.Err()
// when it does not wrap it, so errors.Is(err, context.Canceled) (or
// DeadlineExceeded) holds. A panic in a listing is re-raised in the calling
// goroutine after all listings have returned.
//
// ctx may be nil; toolsets then receive nil.
func ResolveTools(ctx agent.ReadonlyContext, a Agent) ([]tool.Tool, error) {
	own, groups, err := resolveToolGroups(ctx, a)
	if err != nil {
		return nil, err
	}
	return flattenTools(own, groups), nil
}

// resolveToolGroups lists as ResolveTools does, and returns the own tools
// and, index-aligned with Reveal(a).Toolsets, the tools each toolset listed;
// the entry of a toolset skipped because its listing failed is nil.
//
// adk-python lists concurrently with asyncio.gather and skips a toolset whose
// listing raises (agents/llm_agent.py, _convert_tool_union_to_tools); it
// catches Exception, not cancellation, so a failure under a canceled context
// still ends the step here. Python's error log line is not ported: library
// code here does not write to a process-wide log.
func resolveToolGroups(ctx agent.ReadonlyContext, a Agent) (own []tool.Tool, groups [][]tool.Tool, err error) {
	state := Reveal(a)
	toolsets := state.Toolsets
	groups = make([][]tool.Tool, len(toolsets))
	errs := make([]error, len(toolsets))
	panics := make([]any, len(toolsets))
	var wg sync.WaitGroup
	for i, toolset := range toolsets {
		wg.Go(func() {
			defer func() { panics[i] = recover() }()
			groups[i], errs[i] = toolset.Tools(ctx)
		})
	}
	wg.Wait()
	for _, p := range panics {
		if p != nil {
			panic(p)
		}
	}
	var ctxErr error
	if ctx != nil {
		ctxErr = ctx.Err()
	}
	for i, listErr := range errs {
		if listErr == nil {
			continue
		}
		if ctxErr != nil {
			err := fmt.Errorf("failed to extract tools from the tool set %q: %w", toolsets[i].Name(), listErr)
			if !errors.Is(err, ctxErr) {
				err = errors.Join(err, ctxErr)
			}
			return nil, nil, err
		}
		groups[i] = nil
	}
	return state.Tools, groups, nil
}

// flattenTools returns own followed by every group, in a new slice that
// shares no backing array with its inputs.
func flattenTools(own []tool.Tool, groups [][]tool.Tool) []tool.Tool {
	n := len(own)
	for _, g := range groups {
		n += len(g)
	}
	tools := make([]tool.Tool, 0, n)
	tools = append(tools, own...)
	for _, g := range groups {
		tools = append(tools, g...)
	}
	return tools
}
