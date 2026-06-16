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

package workflow

import (
	"google.golang.org/adk/agent"
)

type NodeContext = agent.Context

// newNodeContext wraps parent for a top-level (static) activation.
func newNodeContext(parent agent.InvocationContext, resumeInputs map[string]any) agent.Context {
	return agent.NewNodeContext(parent, resumeInputs)
}

// newDynamicNodeContext wraps parent for either a dynamic-node
// activation or one of its children, attaching path, runID, and the
// sub-scheduler RunNode reaches from the orchestrator body. Children
// pass the sub-scheduler's counter (or WithRunID) value as runID; a
// dynamic node's own activation passes runID="" — it is not itself a
// sub-scheduler child. Child inherits resumeInputs so HITL responses
// reach dynamic children.
func newDynamicNodeContext(parent NodeContext, path, runID string, sub agent.DynamicSubScheduler, outputForAncestors []string) agent.Context {
	return agent.NewDynamicNodeContext(parent, path, runID, sub, outputForAncestors)
}
