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
	"context"
	"iter"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/parentmap"
)

// TreeAncestors yields a's parent, then its parent, and so on up to the root
// of a's tree, with the parent rule transfers use (see parentOf): parents, the
// run's parent map, while it knows the current agent, then the parent links
// agent.New records. It yields nothing when a has no parent. It is for a
// caller that holds the run's parent map rather than a context carrying it,
// such as the runner's router. Safe for concurrent use when parents is not
// written during the walk.
func TreeAncestors(parents parentmap.Map, a agent.Agent) iter.Seq[agent.Agent] {
	return ancestors(parentmap.ToContext(context.Background(), parents), a)
}
