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
	"google.golang.org/adk/v2/agent"
)

// AsToolset wraps a single Tool so it can be passed where a Toolset is
// expected, for example to keep a declared order across plain tools and
// toolsets in llmagent.Config.Toolsets.
//
// Caller provides: a non-nil Tool.
// Guarantees: the returned Toolset's Name is t.Name(); its Tools returns
// exactly []Tool{t}, in a new slice on every call, and a nil error for any
// context, nil included. The Toolset has no request hook (no
// ProcessRequest method): the flow runs t's own hook once, through the tools
// the Toolset lists, as it does for a tool in Config.Tools.
// On failure: AsToolset panics when t is nil, with a message naming
// AsToolset.
// Zero value: N/A.
// Concurrency: safe for concurrent use when t is.
func AsToolset(t Tool) Toolset {
	if t == nil {
		panic("tool.AsToolset: tool must not be nil")
	}
	return &singleToolset{tool: t}
}

// singleToolset is the Toolset AsToolset returns. It deliberately has no
// ProcessRequest, Close or Unwrap method: the flow reaches the tool's own
// hook through Tools, and it wraps a tool, not a toolset.
type singleToolset struct {
	tool Tool
}

func (s *singleToolset) Name() string { return s.tool.Name() }

func (s *singleToolset) Tools(agent.ReadonlyContext) ([]Tool, error) {
	return []Tool{s.tool}, nil
}

// WrappedTool returns the wrapped tool, so that code walking an agent tree
// (Runner.Close) can reach an agent tool placed through AsToolset without
// listing the toolset.
func (s *singleToolset) WrappedTool() Tool { return s.tool }
