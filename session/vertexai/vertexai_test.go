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

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/util/vertexai"

	"google.golang.org/genai"
	"google.golang.org/protobuf/types/known/structpb"

	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
)

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

func TestAiplatformToGenaiContentPreservesFunctionIDs(t *testing.T) {
	args, err := structpb.NewStruct(map[string]any{"city": "Stockholm"})
	if err != nil {
		t.Fatalf("structpb.NewStruct(args) failed: %v", err)
	}
	response, err := structpb.NewStruct(map[string]any{"temperature": 21})
	if err != nil {
		t.Fatalf("structpb.NewStruct(response) failed: %v", err)
	}

	content := aiplatformToGenaiContent(&aiplatformpb.SessionEvent{
		Content: &aiplatformpb.Content{
			Role: string(genai.RoleModel),
			Parts: []*aiplatformpb.Part{
				{
					Data: &aiplatformpb.Part_FunctionCall{
						FunctionCall: &aiplatformpb.FunctionCall{
							Id:   "call-123",
							Name: "get_weather",
							Args: args,
						},
					},
				},
				{
					Data: &aiplatformpb.Part_FunctionResponse{
						FunctionResponse: &aiplatformpb.FunctionResponse{
							Id:       "call-123",
							Name:     "get_weather",
							Response: response,
						},
					},
				},
			},
		},
	})

	if content == nil {
		t.Fatal("aiplatformToGenaiContent() returned nil content")
	}
	if got, want := len(content.Parts), 2; got != want {
		t.Fatalf("len(content.Parts) = %d, want %d", got, want)
	}

	functionCall := content.Parts[0].FunctionCall
	if functionCall == nil {
		t.Fatal("content.Parts[0].FunctionCall is nil")
	}
	if got, want := functionCall.ID, "call-123"; got != want {
		t.Errorf("FunctionCall.ID = %q, want %q", got, want)
	}

	functionResponse := content.Parts[1].FunctionResponse
	if functionResponse == nil {
		t.Fatal("content.Parts[1].FunctionResponse is nil")
	}
	if got, want := functionResponse.ID, "call-123"; got != want {
		t.Errorf("FunctionResponse.ID = %q, want %q", got, want)
	}
}

func TestToStructPB(t *testing.T) {
	tests := []struct {
		name        string
		input       any
		expectError bool
		validate    func(t *testing.T, s *structpb.Struct)
	}{
		{
			name:        "simple map representing function call args or function response",
			input:       map[string]any{"city": "Stockholm"},
			expectError: false,
			validate: func(t *testing.T, s *structpb.Struct) {
				if got, want := s.Fields["city"].GetStringValue(), "Stockholm"; got != want {
					t.Errorf("city = %q, want %q", got, want)
				}
			},
		},
		{
			name:        "invalid input",
			input:       "hello",
			expectError: true,
		},
		{
			name: "custom struct representing possible state delta",
			input: struct {
				StringValue string
				BoolValue   bool
				IntValue    int32
				ArrayValue  []string
			}{
				StringValue: "value",
				BoolValue:   false,
				IntValue:    123,
				ArrayValue:  []string{"value"},
			},
			expectError: false,
			validate: func(t *testing.T, s *structpb.Struct) {
				if _, exists := s.Fields["StringValue"]; !exists {
					t.Error("expected 'StringValue' field to exist")
				}
				if _, exists := s.Fields["BoolValue"]; !exists {
					t.Error("expected 'Boolvalue' field to exist")
				}
				if _, exists := s.Fields["IntValue"]; !exists {
					t.Error("expected 'IntValue' field to exist")
				}
				if _, exists := s.Fields["ArrayValue"]; !exists {
					t.Error("expected 'ArrayValue' field to exist")
				}
			},
		},
		{
			name: "custom struct representing possible state delta respects json tags and omitempty",
			input: struct {
				StringValue      string   `json:"string_value"`
				BoolValue        bool     `json:"bool_value"`
				IntValue         int32    `json:"int_value"`
				ArrayValue       []string `json:"array_value"`
				EmptyStringValue string   `json:"empty_string_value,omitempty"`
			}{
				StringValue:      "value",
				BoolValue:        false,
				IntValue:         123,
				ArrayValue:       []string{"value"},
				EmptyStringValue: "",
			},
			expectError: false,
			validate: func(t *testing.T, s *structpb.Struct) {
				if _, exists := s.Fields["string_value"]; !exists {
					t.Error("expected 'string_value' field to exist")
				}
				if _, exists := s.Fields["bool_value"]; !exists {
					t.Error("expected 'bool_value' field to exist")
				}
				if _, exists := s.Fields["int_value"]; !exists {
					t.Error("expected 'int_value' field to exist")
				}
				if _, exists := s.Fields["array_value"]; !exists {
					t.Error("expected 'array_value' field to exist")
				}
				if _, exists := s.Fields["empty_string_value"]; exists {
					t.Error("unexpected 'empty_string_value' field")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := toStructPB(tt.input)
			if (err != nil) != tt.expectError {
				t.Errorf("toStructPB() error = %v, expectError %v", err, tt.expectError)
			}
			if !tt.expectError && tt.validate != nil {
				tt.validate(t, result)
			}
		})
	}
}

func TestCreateAiplatformpbContent(t *testing.T) {
	tests := []struct {
		name        string
		event       *session.Event
		expectError bool
	}{
		{
			name: "simple function call args",
			event: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Parts: []*genai.Part{
							genai.NewPartFromFunctionCall("tool", map[string]any{
								"city": "Stockholm",
							}),
						},
						Role: genai.RoleUser,
					},
				},
			},
			expectError: false,
		},
		{
			name: "simple function response",
			event: &session.Event{
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Parts: []*genai.Part{
							genai.NewPartFromFunctionResponse("tool", map[string]any{
								"city": "Stockholm",
							}),
						},
						Role: genai.RoleUser,
					},
				},
			},
			expectError: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := createAiplatformpbContent(tt.event)
			if (err != nil) != tt.expectError {
				t.Errorf("createAiplatformpbContent() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

func TestCreateAiplatformpbMetadata(t *testing.T) {
	tests := []struct {
		name        string
		event       *session.Event
		expectError bool
	}{
		{
			name: "simple custom metadata",
			event: &session.Event{
				LLMResponse: model.LLMResponse{
					CustomMetadata: map[string]any{
						"key": "value",
					},
				},
			},
			expectError: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := createAiplatformpbMetadata(tt.event)
			if (err != nil) != tt.expectError {
				t.Errorf("createAiplatformpbMetadata() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

func TestCreateAiplatformpbEventActionsIncludesArtifactDelta(t *testing.T) {
	event := &session.Event{
		Actions: session.EventActions{
			StateDelta:    map[string]any{"user:theme": "dark"},
			ArtifactDelta: map[string]int64{"chart.html": 3, "table.csv": 7},
		},
	}

	actions, err := createAiplatformpbEventActions(event)
	if err != nil {
		t.Fatalf("createAiplatformpbEventActions() failed: %v", err)
	}
	if actions == nil {
		t.Fatal("createAiplatformpbEventActions() returned nil")
	}
	if actions.StateDelta == nil {
		t.Fatal("actions.StateDelta is nil")
	}
	if got, want := actions.StateDelta.AsMap()["user:theme"], "dark"; got != want {
		t.Errorf("actions.StateDelta[user:theme] = %v, want %v", got, want)
	}
	if got, want := actions.ArtifactDelta["chart.html"], int32(3); got != want {
		t.Errorf("actions.ArtifactDelta[chart.html] = %d, want %d", got, want)
	}
	if got, want := actions.ArtifactDelta["table.csv"], int32(7); got != want {
		t.Errorf("actions.ArtifactDelta[table.csv] = %d, want %d", got, want)
	}
}

func TestAiplatformToSessionEventActionsIncludesArtifactDelta(t *testing.T) {
	stateDelta, err := structpb.NewStruct(map[string]any{"user:theme": "dark"})
	if err != nil {
		t.Fatalf("structpb.NewStruct() failed: %v", err)
	}

	actions := aiplatformToSessionEventActions(&aiplatformpb.EventActions{
		StateDelta:    stateDelta,
		ArtifactDelta: map[string]int32{"chart.html": 3, "table.csv": 7},
	})

	if got, want := actions.StateDelta["user:theme"], "dark"; got != want {
		t.Errorf("actions.StateDelta[user:theme] = %v, want %v", got, want)
	}
	if got, want := actions.ArtifactDelta["chart.html"], int64(3); got != want {
		t.Errorf("actions.ArtifactDelta[chart.html] = %d, want %d", got, want)
	}
	if got, want := actions.ArtifactDelta["table.csv"], int64(7); got != want {
		t.Errorf("actions.ArtifactDelta[table.csv] = %d, want %d", got, want)
	}
}

func TestAiplatformToSessionEventActionsHandlesNil(t *testing.T) {
	actions := aiplatformToSessionEventActions(nil)

	if actions.StateDelta != nil {
		t.Errorf("actions.StateDelta = %v, want nil", actions.StateDelta)
	}
	if actions.ArtifactDelta != nil {
		t.Errorf("actions.ArtifactDelta = %v, want nil", actions.ArtifactDelta)
	}
}
