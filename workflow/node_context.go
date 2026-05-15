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

import "google.golang.org/adk/agent"

// nodeContext is the per-node InvocationContext seen inside Node.Run.
// It wraps the workflow's incoming agent.InvocationContext and adds
// engine-supplied metadata — currently only the upstream node name
// (TriggeredBy).
//
// TODO(wolo): replace once context-unification work lands.
type nodeContext struct {
	agent.InvocationContext
	triggeredBy string
}

// newNodeContext returns a nodeContext wrapping parent with the given
// upstream-node name. triggeredBy is empty for the initial START
// activation.
func newNodeContext(parent agent.InvocationContext, triggeredBy string) *nodeContext {
	return &nodeContext{InvocationContext: parent, triggeredBy: triggeredBy}
}

// TriggeredBy returns the name of the upstream node whose output
// scheduled this node activation. Empty for the initial START
// trigger and for non-workflow invocations (where the wrapper is not
// used).
func (c *nodeContext) TriggeredBy() string { return c.triggeredBy }
