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

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/util/vertexai"

	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"google.golang.org/genai"
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

// TestRawEventRoundTrip pins that fields lacking a dedicated SessionEvent
// column survive a raw_event write/read round-trip — NodeInfo in
// particular, which the legacy field-based path dropped — and that the
// fields already persisted via dedicated columns are not degraded.
func TestRawEventRoundTrip(t *testing.T) {
	tests := []struct {
		name  string
		event *session.Event
	}{
		{
			name: "workflow fields",
			event: &session.Event{
				InvocationID:   "inv-1",
				Author:         "agent-x",
				Branch:         "a.b",
				IsolationScope: "scope-1",
				Routes:         []string{"approve"},
				Output:         "the-output",
				NodeInfo: &session.NodeInfo{
					Path:            "wf/child@1",
					MessageAsOutput: true,
					OutputFor:       []string{"wf/child@1", "wf"},
				},
			},
		},
		{
			name: "content with text and function call",
			event: &session.Event{
				Author: "agent-x",
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Role: string(genai.RoleModel),
						Parts: []*genai.Part{
							{Text: "hello"},
							{FunctionCall: &genai.FunctionCall{
								ID:   "call-1",
								Name: "get_weather",
								Args: map[string]any{"city": "Stockholm"},
							}},
						},
					},
				},
			},
		},
		{
			name: "structured output",
			event: &session.Event{
				Author: "agent-x",
				Output: map[string]any{"score": float64(42), "label": "ok"},
			},
		},
		{
			// Typed map[string]int64 keeps its int type (unlike any-typed
			// Output/StateDelta; see TestRawEventNumericContract).
			name: "typed int64 artifact delta preserved",
			event: &session.Event{
				Author: "agent-x",
				Output: "x",
				Actions: session.EventActions{
					ArtifactDelta: map[string]int64{"file.png": 7},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := eventToRawEvent(tc.event)
			if err != nil {
				t.Fatalf("eventToRawEvent() error = %v", err)
			}
			got, err := eventFromRawEvent(raw)
			if err != nil {
				t.Fatalf("eventFromRawEvent() error = %v", err)
			}
			if diff := cmp.Diff(tc.event, got); diff != "" {
				t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestRawEventNumericContract pins the documented contract (see
// eventToRawEvent): integers in the any-typed Output and StateDelta come
// back as float64.
func TestRawEventNumericContract(t *testing.T) {
	event := &session.Event{
		Author: "agent-x",
		Output: int64(9007199254740993), // 2^53 + 1
		Actions: session.EventActions{
			StateDelta: map[string]any{"count": 3},
		},
	}
	raw, err := eventToRawEvent(event)
	if err != nil {
		t.Fatalf("eventToRawEvent() error = %v", err)
	}
	got, err := eventFromRawEvent(raw)
	if err != nil {
		t.Fatalf("eventFromRawEvent() error = %v", err)
	}
	if _, ok := got.Output.(float64); !ok {
		t.Errorf("Output type = %T, want float64", got.Output)
	}
	if v, ok := got.Actions.StateDelta["count"].(float64); !ok || v != 3 {
		t.Errorf("StateDelta[count] = %#v, want float64(3)", got.Actions.StateDelta["count"])
	}
}

// TestEventNeedsRawEvent guards the invariant that plain events keep
// their legacy wire format (no raw_event) while events carrying state
// without a dedicated SessionEvent column opt into raw_event. Changing
// this for plain events would invalidate the recorded replay fixtures.
func TestEventNeedsRawEvent(t *testing.T) {
	tests := []struct {
		name  string
		event *session.Event
		want  bool
	}{
		{name: "plain event", event: &session.Event{Author: "user"}, want: false},
		{name: "with content only", event: &session.Event{
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("hi", genai.RoleUser)},
		}, want: false},
		{name: "with output", event: &session.Event{Output: "x"}, want: true},
		{name: "with node info", event: &session.Event{NodeInfo: &session.NodeInfo{Path: "wf"}}, want: true},
		{name: "with isolation scope", event: &session.Event{IsolationScope: "s"}, want: true},
		{name: "with routes", event: &session.Event{Routes: []string{"approve"}}, want: true},
		{name: "with requested input", event: &session.Event{
			RequestedInput: &session.RequestInput{InterruptID: "i"},
		}, want: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := eventNeedsRawEvent(tc.event); got != tc.want {
				t.Errorf("eventNeedsRawEvent() = %v, want %v", got, tc.want)
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
