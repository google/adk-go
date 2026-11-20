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

package workflow_test

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/openaimodel"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/workflow"
)

type schemaTransport func(*http.Request) (*http.Response, error)

func (f schemaTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestAgentNode_OpenAIOutputSchemaTool(t *testing.T) {
	for _, api := range []openaimodel.API{openaimodel.APIChatCompletions, openaimodel.APIResponses} {
		for _, optIn := range []bool{false, true} {
			for _, withTools := range []bool{false, true} {
				for _, declared := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/optIn=%v/tools=%v/declared=%v", api, optIn, withTools, declared), func(t *testing.T) {
						originalSchema, err := json.Marshal(weatherSchema)
						if err != nil {
							t.Fatal(err)
						}
						calls := 0
						client := &http.Client{Transport: schemaTransport(func(r *http.Request) (*http.Response, error) {
							calls++
							var body map[string]any
							if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
								t.Fatal(err)
							}
							var native bool
							if api == openaimodel.APIChatCompletions {
								native = body["response_format"] != nil
							} else if text, ok := body["text"].(map[string]any); ok {
								native = text["format"] != nil
							}
							if native == optIn {
								t.Error("native response format does not match the opt-in")
							}
							responseTools, lookupTools := 0, 0
							if tools, ok := body["tools"].([]any); ok {
								for _, item := range tools {
									decl := item.(map[string]any)
									if api == openaimodel.APIChatCompletions {
										decl = decl["function"].(map[string]any)
									}
									if decl["name"] == "set_model_response" {
										responseTools++
										// Both OpenAI APIs require standard lower-case JSON Schema
										// types even when the source is a *genai.Schema.
										params, ok := decl["parameters"].(map[string]any)
										if !ok || params["type"] != "object" {
											t.Error("set_model_response parameters must be an object JSON Schema")
										} else if props, ok := params["properties"].(map[string]any); !ok {
											t.Error("set_model_response properties missing")
										} else {
											for name, wantType := range map[string]string{"city": "string", "temp": "integer"} {
												prop, ok := props[name].(map[string]any)
												if !ok || prop["type"] != wantType {
													t.Errorf("set_model_response %s schema has wrong type", name)
												}
											}
										}
									}
									if decl["name"] == "lookup" {
										lookupTools++
									}
								}
							}
							wantResponseTools, wantLookupTools := 0, 0
							if optIn {
								wantResponseTools = 1
							}
							if withTools {
								wantLookupTools = 1
							}
							if responseTools != wantResponseTools {
								t.Error("serialized set_model_response does not match the opt-in")
							}
							if lookupTools != wantLookupTools {
								t.Error("configured lookup tool was not preserved")
							}
							reply := `{"id":"chat_1","object":"chat.completion","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"{\"city\":\"Paris\",\"temp\":31}"}}]}`
							if api == openaimodel.APIResponses {
								reply = `{"id":"resp_1","object":"response","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"{\"city\":\"Paris\",\"temp\":31}"}]}]}`
							}
							if responseTools > 0 {
								if api == openaimodel.APIChatCompletions {
									reply = `{"id":"chat_1","object":"chat.completion","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"set_model_response","arguments":"{\"city\":\"Paris\",\"temp\":31}"}}]}}]}`
								} else {
									reply = `{"id":"resp_1","object":"response","status":"completed","output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"set_model_response","arguments":"{\"city\":\"Paris\",\"temp\":31}"}]}`
								}
							}
							return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(reply)), Request: r}, nil
						})}
						llm, err := openaimodel.NewModel(t.Context(), "deepseek-chat", &openaimodel.ClientConfig{
							API: api, APIKey: "test", HTTPClient: client, UseOutputSchemaTool: optIn,
						})
						if err != nil {
							t.Fatal(err)
						}
						cfg := llmagent.Config{Name: "weather", Model: llm}
						if withTools {
							cfg.Tools = []tool.Tool{lookupTool(t)}
						}
						if declared {
							cfg.OutputSchema = weatherSchema
						}
						a, err := llmagent.New(cfg)
						if err != nil {
							t.Fatal(err)
						}
						node, err := workflow.NewAgentNodeTyped[string, weatherOut](a, workflow.NodeConfig{})
						if err != nil {
							t.Fatal(err)
						}
						// Reusing the same agent must not accumulate response tools or mutate its schema.
						for range 2 {
							out := runNode(t, node, a)
							got, err := node.ValidateOutput(out)
							if err != nil {
								t.Fatal(err)
							}
							if diff := cmp.Diff(map[string]any{"city": "Paris", "temp": float64(31)}, got); diff != "" {
								t.Error("structured output mismatch")
							}
						}
						finalSchema, err := json.Marshal(weatherSchema)
						if err != nil {
							t.Fatal(err)
						}
						if string(originalSchema) != string(finalSchema) {
							t.Error("agent schema changed across runs")
						}
						if calls != 2 {
							t.Errorf("HTTP calls = %d, want 2", calls)
						}
					})
				}
			}
		}
	}
}
