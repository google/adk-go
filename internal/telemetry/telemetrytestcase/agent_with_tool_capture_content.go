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

// AgentWithToolCaptureContentCase is the expected root span for
// the same scenario as [AgentWithToolCase] but with content capture
// enabled, via OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT env var.
var AgentWithToolCaptureContentCase = &telemetrytest.SpanDigest{
	Name: "invoke_agent some_root_agent",
	Attributes: map[string]any{
		"gen_ai.operation.name":          "invoke_agent",
		"gen_ai.agent.name":              "some_root_agent",
		"gen_ai.agent.description":       "A sample root agent.",
		"gen_ai.conversation.id":         telemetrytest.PRESENT,
		"gcp.vertex.agent.invocation_id": telemetrytest.PRESENT,
	},
	Children: []*telemetrytest.SpanDigest{
		{
			Name: "generate_content mock",
			Attributes: map[string]any{
				"gen_ai.operation.name":          "generate_content",
				"gen_ai.request.model":           "mock",
				"gcp.vertex.agent.event_id":      telemetrytest.PRESENT,
				"gcp.vertex.agent.invocation_id": telemetrytest.PRESENT,
				"gen_ai.response.finish_reasons": []string{""},
			},
			Logs: []*telemetrytest.LogDigest{
				{
					EventName:  "gen_ai.system.message",
					Attributes: map[string]any{},
					Body: map[string]any{
						"content": "you are helpful\n\nYou are an agent. Your internal name is \"some_root_agent\". The description about you is \"A sample root agent.\".",
					},
				},
				{
					EventName:  "gen_ai.user.message",
					Attributes: map[string]any{},
					Body: map[string]any{
						"content": map[string]any{
							"role":  "user",
							"parts": []any{map[string]any{"text": "hello"}},
						},
					},
				},
				{
					EventName:  "gen_ai.choice",
					Attributes: map[string]any{},
					// Turn 1: model emits a function call.
					Body: map[string]any{
						"index": float64(0),
						"content": map[string]any{
							"role": "model",
							"parts": []any{map[string]any{
								"functionCall": map[string]any{
									"name": "some_tool",
									"args": map[string]any{"arg1": "val1"},
								},
							}},
						},
					},
				},
			},
		},
		{
			Name: "execute_tool some_tool",
			Attributes: map[string]any{
				"gen_ai.operation.name":           "execute_tool",
				"gen_ai.tool.name":                "some_tool",
				"gen_ai.tool.description":         "A sample tool.",
				"gen_ai.tool.call.id":             telemetrytest.PRESENT,
				"gcp.vertex.agent.event_id":       telemetrytest.PRESENT,
				"gcp.vertex.agent.tool_call_args": telemetrytest.PRESENT,
				"gcp.vertex.agent.tool_response":  telemetrytest.PRESENT,
			},
		},
		{
			Name: "generate_content mock",
			Attributes: map[string]any{
				"gen_ai.operation.name":          "generate_content",
				"gen_ai.request.model":           "mock",
				"gcp.vertex.agent.event_id":      telemetrytest.PRESENT,
				"gcp.vertex.agent.invocation_id": telemetrytest.PRESENT,
				"gen_ai.response.finish_reasons": []string{""},
			},
			Logs: []*telemetrytest.LogDigest{
				{
					EventName:  "gen_ai.system.message",
					Attributes: map[string]any{},
					Body: map[string]any{
						"content": "you are helpful\n\nYou are an agent. Your internal name is \"some_root_agent\". The description about you is \"A sample root agent.\".",
					},
				},
				{
					EventName:  "gen_ai.user.message",
					Attributes: map[string]any{},
					Body: map[string]any{
						"content": map[string]any{
							"role":  "user",
							"parts": []any{map[string]any{"text": "hello"}},
						},
					},
				},
				{
					EventName:  "gen_ai.user.message",
					Attributes: map[string]any{},
					// Turn 2 carries the model's previous turn
					// (the function call) as a content entry.
					Body: map[string]any{
						"content": map[string]any{
							"role": "model",
							"parts": []any{map[string]any{
								"functionCall": map[string]any{
									"name": "some_tool",
									"args": map[string]any{"arg1": "val1"},
								},
							}},
						},
					},
				},
				{
					EventName:  "gen_ai.user.message",
					Attributes: map[string]any{},
					// Turn 2 carries the function-response turn.
					Body: map[string]any{
						"content": map[string]any{
							"role": "user",
							"parts": []any{map[string]any{
								"functionResponse": map[string]any{
									"name":     "some_tool",
									"response": map[string]any{"result": "processed val1"},
								},
							}},
						},
					},
				},
				{
					EventName:  "gen_ai.choice",
					Attributes: map[string]any{},
					// Turn 2: final text response.
					Body: map[string]any{
						"index": float64(0),
						"content": map[string]any{
							"role":  "model",
							"parts": []any{map[string]any{"text": "text response"}},
						},
					},
				},
			},
		},
	},
}
