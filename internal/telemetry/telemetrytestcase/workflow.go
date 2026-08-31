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
	"google.golang.org/adk/v2/internal/telemetry/telemetrytest"
)

// WorkflowCase is the expected span+log tree for the canonical
// workflow scenario on the happy path:
var WorkflowCase = &telemetrytest.SpanDigest{
	Name: "invoke_workflow my_workflow",
	Attributes: map[string]any{
		"gen_ai.conversation.id": telemetrytest.PRESENT,
		"gen_ai.operation.name":  "invoke_workflow",
		"gen_ai.workflow.name":   "my_workflow",
	},
	Children: []*telemetrytest.SpanDigest{
		{
			Name: "invoke_node static_node",
			Attributes: map[string]any{
				"gen_ai.conversation.id": telemetrytest.PRESENT,
				"gen_ai.node.name":       "static_node",
				"gen_ai.operation.name":  "invoke_node",
			},
		},
		{
			Name: "invoke_node router_node",
			Attributes: map[string]any{
				"gen_ai.conversation.id": telemetrytest.PRESENT,
				"gen_ai.node.name":       "router_node",
				"gen_ai.operation.name":  "invoke_node",
			},
			Children: []*telemetrytest.SpanDigest{
				{
					Name: "invoke_agent task_agent",
					Attributes: map[string]any{
						"gcp.vertex.agent.invocation_id": telemetrytest.PRESENT,
						"gen_ai.agent.description":       "collaborative task_agent",
						"gen_ai.agent.name":              "task_agent",
						"gen_ai.conversation.id":         telemetrytest.PRESENT,
						"gen_ai.operation.name":          "invoke_agent",
					},
					Children: []*telemetrytest.SpanDigest{
						{
							Name: "generate_content mock",
							Attributes: map[string]any{
								"gcp.vertex.agent.event_id":      telemetrytest.PRESENT,
								"gcp.vertex.agent.invocation_id": telemetrytest.PRESENT,
								"gen_ai.operation.name":          "generate_content",
								"gen_ai.request.model":           "mock",
								"gen_ai.response.finish_reasons": []string{""},
							},
							Logs: []*telemetrytest.LogDigest{
								{EventName: "gen_ai.system.message", Attributes: map[string]any{}, Body: map[string]any{"content": "<elided>"}},
								{EventName: "gen_ai.user.message", Attributes: map[string]any{}, Body: map[string]any{"content": "<elided>"}},
								{EventName: "gen_ai.choice", Attributes: map[string]any{}, Body: map[string]any{"content": "<elided>", "index": float64(0)}},
							},
						},
					},
				},
				{
					Name: "invoke_agent single_turn_agent",
					Attributes: map[string]any{
						"gcp.vertex.agent.invocation_id": telemetrytest.PRESENT,
						"gen_ai.agent.description":       "collaborative single_turn_agent",
						"gen_ai.agent.name":              "single_turn_agent",
						"gen_ai.conversation.id":         telemetrytest.PRESENT,
						"gen_ai.operation.name":          "invoke_agent",
					},
					Children: []*telemetrytest.SpanDigest{
						{
							Name: "generate_content mock",
							Attributes: map[string]any{
								"gcp.vertex.agent.event_id":      telemetrytest.PRESENT,
								"gcp.vertex.agent.invocation_id": telemetrytest.PRESENT,
								"gen_ai.operation.name":          "generate_content",
								"gen_ai.request.model":           "mock",
								"gen_ai.response.finish_reasons": []string{""},
							},
							Logs: []*telemetrytest.LogDigest{
								{EventName: "gen_ai.system.message", Attributes: map[string]any{}, Body: map[string]any{"content": "<elided>"}},
								{EventName: "gen_ai.user.message", Attributes: map[string]any{}, Body: map[string]any{"content": "<elided>"}},
								{EventName: "gen_ai.choice", Attributes: map[string]any{}, Body: map[string]any{"content": "<elided>", "index": float64(0)}},
							},
						},
					},
				},
				{
					Name: "invoke_node echo_node",
					Attributes: map[string]any{
						"gen_ai.conversation.id": telemetrytest.PRESENT,
						"gen_ai.node.name":       "echo_node",
						"gen_ai.operation.name":  "invoke_node",
					},
				},
				{
					Name: "invoke_node echo_node",
					Attributes: map[string]any{
						"gen_ai.conversation.id": telemetrytest.PRESENT,
						"gen_ai.node.name":       "echo_node",
						"gen_ai.operation.name":  "invoke_node",
					},
				},
			},
		},
	},
}

// WorkflowErrorStaticNodeCase: the static node fails, so the
// workflow stops before the dynamic node runs. Both the node span
// and the workflow span carry Error status.
var WorkflowErrorStaticNodeCase = &telemetrytest.SpanDigest{
	Name:   "invoke_workflow my_workflow",
	Status: "Error",
	Attributes: map[string]any{
		"gen_ai.conversation.id": telemetrytest.PRESENT,
		"gen_ai.operation.name":  "invoke_workflow",
		"gen_ai.workflow.name":   "my_workflow",
	},
	Children: []*telemetrytest.SpanDigest{
		{
			Name:   "invoke_node static_node",
			Status: "Error",
			Attributes: map[string]any{
				"gen_ai.conversation.id": telemetrytest.PRESENT,
				"gen_ai.node.name":       "static_node",
				"gen_ai.operation.name":  "invoke_node",
			},
		},
	},
}

// WorkflowErrorDynamicNodeCase: the dynamic orchestrator body fails
// before delegating, so router_node errors with no child spans while
// the preceding static_node succeeded.
var WorkflowErrorDynamicNodeCase = &telemetrytest.SpanDigest{
	Name:   "invoke_workflow my_workflow",
	Status: "Error",
	Attributes: map[string]any{
		"gen_ai.conversation.id": telemetrytest.PRESENT,
		"gen_ai.operation.name":  "invoke_workflow",
		"gen_ai.workflow.name":   "my_workflow",
	},
	Children: []*telemetrytest.SpanDigest{
		{
			Name: "invoke_node static_node",
			Attributes: map[string]any{
				"gen_ai.conversation.id": telemetrytest.PRESENT,
				"gen_ai.node.name":       "static_node",
				"gen_ai.operation.name":  "invoke_node",
			},
		},
		{
			Name:   "invoke_node router_node",
			Status: "Error",
			Attributes: map[string]any{
				"gen_ai.conversation.id": telemetrytest.PRESENT,
				"gen_ai.node.name":       "router_node",
				"gen_ai.operation.name":  "invoke_node",
			},
		},
	},
}

// WorkflowErrorFirstAgentCase: the first collaborative agent
// (task_agent) fails. Error status propagates from generate_content
// up through invoke_agent task_agent and the dynamic router_node to
// the workflow span.
var WorkflowErrorFirstAgentCase = &telemetrytest.SpanDigest{
	Name:   "invoke_workflow my_workflow",
	Status: "Error",
	Attributes: map[string]any{
		"gen_ai.conversation.id": telemetrytest.PRESENT,
		"gen_ai.operation.name":  "invoke_workflow",
		"gen_ai.workflow.name":   "my_workflow",
	},
	Children: []*telemetrytest.SpanDigest{
		{
			Name: "invoke_node static_node",
			Attributes: map[string]any{
				"gen_ai.conversation.id": telemetrytest.PRESENT,
				"gen_ai.node.name":       "static_node",
				"gen_ai.operation.name":  "invoke_node",
			},
		},
		{
			Name:   "invoke_node router_node",
			Status: "Error",
			Attributes: map[string]any{
				"gen_ai.conversation.id": telemetrytest.PRESENT,
				"gen_ai.node.name":       "router_node",
				"gen_ai.operation.name":  "invoke_node",
			},
			Children: []*telemetrytest.SpanDigest{
				{
					Name:   "invoke_agent task_agent",
					Status: "Error",
					Attributes: map[string]any{
						"gcp.vertex.agent.invocation_id": telemetrytest.PRESENT,
						"gen_ai.agent.description":       "collaborative task_agent",
						"gen_ai.agent.name":              "task_agent",
						"gen_ai.conversation.id":         telemetrytest.PRESENT,
						"gen_ai.operation.name":          "invoke_agent",
					},
					Children: []*telemetrytest.SpanDigest{
						{
							Name:   "generate_content mock",
							Status: "Error",
							Attributes: map[string]any{
								"gcp.vertex.agent.invocation_id": telemetrytest.PRESENT,
								"gen_ai.operation.name":          "generate_content",
								"gen_ai.request.model":           "mock",
							},
							Logs: []*telemetrytest.LogDigest{
								{EventName: "gen_ai.system.message", Attributes: map[string]any{}, Body: map[string]any{"content": "<elided>"}},
								{EventName: "gen_ai.user.message", Attributes: map[string]any{}, Body: map[string]any{"content": "<elided>"}},
							},
						},
					},
				},
			},
		},
	},
}

// WorkflowErrorSecondAgentCase: the first collaborative agent
// succeeds and the second (single_turn_agent) fails; the workflow
// stops before reaching the cached echo_node delegations.
var WorkflowErrorSecondAgentCase = &telemetrytest.SpanDigest{
	Name:   "invoke_workflow my_workflow",
	Status: "Error",
	Attributes: map[string]any{
		"gen_ai.conversation.id": telemetrytest.PRESENT,
		"gen_ai.operation.name":  "invoke_workflow",
		"gen_ai.workflow.name":   "my_workflow",
	},
	Children: []*telemetrytest.SpanDigest{
		{
			Name: "invoke_node static_node",
			Attributes: map[string]any{
				"gen_ai.conversation.id": telemetrytest.PRESENT,
				"gen_ai.node.name":       "static_node",
				"gen_ai.operation.name":  "invoke_node",
			},
		},
		{
			Name:   "invoke_node router_node",
			Status: "Error",
			Attributes: map[string]any{
				"gen_ai.conversation.id": telemetrytest.PRESENT,
				"gen_ai.node.name":       "router_node",
				"gen_ai.operation.name":  "invoke_node",
			},
			Children: []*telemetrytest.SpanDigest{
				{
					Name: "invoke_agent task_agent",
					Attributes: map[string]any{
						"gcp.vertex.agent.invocation_id": telemetrytest.PRESENT,
						"gen_ai.agent.description":       "collaborative task_agent",
						"gen_ai.agent.name":              "task_agent",
						"gen_ai.conversation.id":         telemetrytest.PRESENT,
						"gen_ai.operation.name":          "invoke_agent",
					},
					Children: []*telemetrytest.SpanDigest{
						{
							Name: "generate_content mock",
							Attributes: map[string]any{
								"gcp.vertex.agent.event_id":      telemetrytest.PRESENT,
								"gcp.vertex.agent.invocation_id": telemetrytest.PRESENT,
								"gen_ai.operation.name":          "generate_content",
								"gen_ai.request.model":           "mock",
								"gen_ai.response.finish_reasons": []string{""},
							},
							Logs: []*telemetrytest.LogDigest{
								{EventName: "gen_ai.system.message", Attributes: map[string]any{}, Body: map[string]any{"content": "<elided>"}},
								{EventName: "gen_ai.user.message", Attributes: map[string]any{}, Body: map[string]any{"content": "<elided>"}},
								{EventName: "gen_ai.choice", Attributes: map[string]any{}, Body: map[string]any{"content": "<elided>", "index": float64(0)}},
							},
						},
					},
				},
				{
					Name:   "invoke_agent single_turn_agent",
					Status: "Error",
					Attributes: map[string]any{
						"gcp.vertex.agent.invocation_id": telemetrytest.PRESENT,
						"gen_ai.agent.description":       "collaborative single_turn_agent",
						"gen_ai.agent.name":              "single_turn_agent",
						"gen_ai.conversation.id":         telemetrytest.PRESENT,
						"gen_ai.operation.name":          "invoke_agent",
					},
					Children: []*telemetrytest.SpanDigest{
						{
							Name:   "generate_content mock",
							Status: "Error",
							Attributes: map[string]any{
								"gcp.vertex.agent.invocation_id": telemetrytest.PRESENT,
								"gen_ai.operation.name":          "generate_content",
								"gen_ai.request.model":           "mock",
							},
							Logs: []*telemetrytest.LogDigest{
								{EventName: "gen_ai.system.message", Attributes: map[string]any{}, Body: map[string]any{"content": "<elided>"}},
								{EventName: "gen_ai.user.message", Attributes: map[string]any{}, Body: map[string]any{"content": "<elided>"}},
							},
						},
					},
				},
			},
		},
	},
}
