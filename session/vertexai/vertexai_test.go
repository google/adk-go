// Copyright 2025 Google LLC
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

package vertexai

import (
	"testing"

	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"google.golang.org/adk/util/vertexai"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestAiplatformToGenaiContent_FunctionCallMapping(t *testing.T) {
	makeArgs := func(m map[string]any) *structpb.Struct {
		s, err := structpb.NewStruct(m)
		if err != nil {
			t.Fatalf("failed to create struct: %v", err)
		}
		return s
	}

	tests := []struct {
		name        string
		input       *aiplatformpb.SessionEvent
		wantID      string
		wantName    string
		wantArgKey  string
		wantArgVal  string
		isResponse  bool
		wantRespKey string
		wantRespVal string
	}{
		{
			name: "FunctionCall preserves ID, Name, and Args",
			input: &aiplatformpb.SessionEvent{
				Content: &aiplatformpb.Content{
					Role: "model",
					Parts: []*aiplatformpb.Part{
						{
							Data: &aiplatformpb.Part_FunctionCall{
								FunctionCall: &aiplatformpb.FunctionCall{
									Id:   "call-id-abc",
									Name: "my_tool",
									Args: makeArgs(map[string]any{"param": "value"}),
								},
							},
						},
					},
				},
			},
			wantID:     "call-id-abc",
			wantName:   "my_tool",
			wantArgKey: "param",
			wantArgVal: "value",
		},
		{
			name: "FunctionCall with empty ID is preserved as empty",
			input: &aiplatformpb.SessionEvent{
				Content: &aiplatformpb.Content{
					Role: "model",
					Parts: []*aiplatformpb.Part{
						{
							Data: &aiplatformpb.Part_FunctionCall{
								FunctionCall: &aiplatformpb.FunctionCall{
									Id:   "",
									Name: "tool_no_id",
									Args: makeArgs(map[string]any{"x": "y"}),
								},
							},
						},
					},
				},
			},
			wantID:     "",
			wantName:   "tool_no_id",
			wantArgKey: "x",
			wantArgVal: "y",
		},
		{
			name:       "FunctionResponse preserves ID, Name, and Response",
			isResponse: true,
			input: &aiplatformpb.SessionEvent{
				Content: &aiplatformpb.Content{
					Role: "user",
					Parts: []*aiplatformpb.Part{
						{
							Data: &aiplatformpb.Part_FunctionResponse{
								FunctionResponse: &aiplatformpb.FunctionResponse{
									Id:       "call-id-abc",
									Name:     "my_tool",
									Response: makeArgs(map[string]any{"result": "ok"}),
								},
							},
						},
					},
				},
			},
			wantID:      "call-id-abc",
			wantName:    "my_tool",
			wantRespKey: "result",
			wantRespVal: "ok",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := aiplatformToGenaiContent(tt.input)
			if got == nil || len(got.Parts) == 0 {
				t.Fatal("expected at least one part, got nil or empty")
			}
			if tt.isResponse {
				fr := got.Parts[0].FunctionResponse
				if fr == nil {
					t.Fatal("expected FunctionResponse part, got nil")
				}
				if fr.ID != tt.wantID {
					t.Errorf("FunctionResponse.ID = %q, want %q", fr.ID, tt.wantID)
				}
				if fr.Name != tt.wantName {
					t.Errorf("FunctionResponse.Name = %q, want %q", fr.Name, tt.wantName)
				}
				if got, ok := fr.Response[tt.wantRespKey]; !ok || got != tt.wantRespVal {
					t.Errorf("FunctionResponse.Response[%q] = %v, want %q", tt.wantRespKey, got, tt.wantRespVal)
				}
			} else {
				fc := got.Parts[0].FunctionCall
				if fc == nil {
					t.Fatal("expected FunctionCall part, got nil")
				}
				if fc.ID != tt.wantID {
					t.Errorf("FunctionCall.ID = %q, want %q", fc.ID, tt.wantID)
				}
				if fc.Name != tt.wantName {
					t.Errorf("FunctionCall.Name = %q, want %q", fc.Name, tt.wantName)
				}
				if got, ok := fc.Args[tt.wantArgKey]; !ok || got != tt.wantArgVal {
					t.Errorf("FunctionCall.Args[%q] = %v, want %q", tt.wantArgKey, got, tt.wantArgVal)
				}
			}
		})
	}
}

func TestGetReasoningEngineID(t *testing.T) {
	tests := []struct {
		name             string
		existingEngineID string // Field: c.reasoningEngine
		inputAppName     string // Argument: appName
		expectedID       string
		expectError      bool
	}{
		{
			name:             "Client already has engine ID configured",
			existingEngineID: "999",
			inputAppName:     "irrelevant-input",
			expectedID:       "999",
			expectError:      false,
		},
		{
			name:             "Input is a direct numeric ID",
			existingEngineID: "",
			inputAppName:     "123456",
			expectedID:       "123456",
			expectError:      false,
		},
		{
			name:             "Input is a valid full resource path",
			existingEngineID: "",
			inputAppName:     "projects/my-project/locations/us-central1/reasoningEngines/555123",
			expectedID:       "555123",
			expectError:      false,
		},
		{
			name:             "Input is valid path with dashes and underscores in project/location",
			existingEngineID: "",
			inputAppName:     "projects/my_project-1/locations/us_central-1/reasoningEngines/888",
			expectedID:       "888",
			expectError:      false,
		},
		{
			name:             "Input is malformed (ID is not numeric)",
			existingEngineID: "",
			inputAppName:     "projects/proj/locations/loc/reasoningEngines/abc",
			expectedID:       "",
			expectError:      true,
		},
		{
			name:             "Input is malformed (missing path components)",
			existingEngineID: "",
			inputAppName:     "locations/us-central1/reasoningEngines/123",
			expectedID:       "",
			expectError:      true,
		},
		{
			name:             "Input is random text",
			existingEngineID: "",
			inputAppName:     "some-random-app-name",
			expectedID:       "",
			expectError:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup the client with the test case state
			c := &vertexAiClient{
				agentEngineData: &vertexai.AgentEngineData{
					ReasoningEngine: tt.existingEngineID,
				},
			}

			// Execute
			got, err := c.getReasoningEngineID(tt.inputAppName)

			// Check Error Expectation
			if (err != nil) != tt.expectError {
				t.Errorf("getReasoningEngineID() error = %v, expectError %v", err, tt.expectError)
				return
			}

			// Check Returned Value
			if got != tt.expectedID {
				t.Errorf("getReasoningEngineID() got = %v, want %v", got, tt.expectedID)
			}
		})
	}
}
