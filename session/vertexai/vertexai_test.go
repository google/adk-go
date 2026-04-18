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

func TestAiplatformToGenaiContent_PreservesFunctionCallAndResponseIDs(t *testing.T) {
	callID := "test-call-id-123"
	argsStruct, err := structpb.NewStruct(map[string]any{"param": "value"})
	if err != nil {
		t.Fatalf("failed to create args struct: %v", err)
	}
	respStruct, err := structpb.NewStruct(map[string]any{"result": "ok"})
	if err != nil {
		t.Fatalf("failed to create response struct: %v", err)
	}

	sessionEvent := &aiplatformpb.SessionEvent{
		Content: &aiplatformpb.Content{
			Role: "model",
			Parts: []*aiplatformpb.Part{
				{
					Data: &aiplatformpb.Part_FunctionCall{
						FunctionCall: &aiplatformpb.FunctionCall{
							Id:   callID,
							Name: "my_tool",
							Args: argsStruct,
						},
					},
				},
			},
		},
	}
	gotCall := aiplatformToGenaiContent(sessionEvent)
	if gotCall == nil || len(gotCall.Parts) == 0 || gotCall.Parts[0].FunctionCall == nil {
		t.Fatal("expected FunctionCall part, got nil")
	}
	if got := gotCall.Parts[0].FunctionCall.ID; got != callID {
		t.Errorf("FunctionCall.ID = %q, want %q", got, callID)
	}

	sessionEvent2 := &aiplatformpb.SessionEvent{
		Content: &aiplatformpb.Content{
			Role: "user",
			Parts: []*aiplatformpb.Part{
				{
					Data: &aiplatformpb.Part_FunctionResponse{
						FunctionResponse: &aiplatformpb.FunctionResponse{
							Id:       callID,
							Name:     "my_tool",
							Response: respStruct,
						},
					},
				},
			},
		},
	}
	gotResp := aiplatformToGenaiContent(sessionEvent2)
	if gotResp == nil || len(gotResp.Parts) == 0 || gotResp.Parts[0].FunctionResponse == nil {
		t.Fatal("expected FunctionResponse part, got nil")
	}
	if got := gotResp.Parts[0].FunctionResponse.ID; got != callID {
		t.Errorf("FunctionResponse.ID = %q, want %q", got, callID)
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
