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

package telemetrytestcase

import (
	"google.golang.org/adk/internal/telemetry/telemetrytest"
)

// WorkflowDynamicCase is the expected root span for a workflowagent
// whose single graph node is a dynamic orchestrator that delegates
// to two FunctionNode children inline via workflow.RunNode.
//
// The shape asserts that RunNode-driven delegation is instrumented:
// each delegated child emits its own "invoke_node" span nested under
// the dynamic node's span (rather than appearing flat under the
// workflow or, as before instrumentation, not at all). The dynamic
// node itself is still activated by the top scheduler, so it nests
// directly under the workflow span.
var WorkflowDynamicCase = &telemetrytest.SpanDigest{
	Name: "invoke_workflow my_workflow",
	Attributes: map[string]any{
		"gen_ai.operation.name":  "invoke_workflow",
		"gen_ai.workflow.name":   "my_workflow",
		"gen_ai.conversation.id": telemetrytest.PRESENT,
	},
	Children: []*telemetrytest.SpanDigest{
		{
			Name: "invoke_node router_node",
			Attributes: map[string]any{
				"gen_ai.operation.name":  "invoke_node",
				"gen_ai.node.name":       "router_node",
				"gen_ai.conversation.id": telemetrytest.PRESENT,
			},
			Children: []*telemetrytest.SpanDigest{
				{
					Name: "invoke_node upper_node",
					Attributes: map[string]any{
						"gen_ai.operation.name":  "invoke_node",
						"gen_ai.node.name":       "upper_node",
						"gen_ai.conversation.id": telemetrytest.PRESENT,
					},
				},
				{
					Name: "invoke_node suffix_node",
					Attributes: map[string]any{
						"gen_ai.operation.name":  "invoke_node",
						"gen_ai.node.name":       "suffix_node",
						"gen_ai.conversation.id": telemetrytest.PRESENT,
					},
				},
			},
		},
	},
}
