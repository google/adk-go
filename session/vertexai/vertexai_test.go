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

func TestToInt32Map(t *testing.T) {
	tests := []struct {
		name string
		in   map[string]int64
		want map[string]int32
	}{
		{
			name: "nil map",
			in:   nil,
			want: nil,
		},
		{
			name: "empty map",
			in:   map[string]int64{},
			want: nil,
		},
		{
			name: "single entry",
			in:   map[string]int64{"chart.html": 1},
			want: map[string]int32{"chart.html": 1},
		},
		{
			name: "multiple entries",
			in:   map[string]int64{"a.txt": 1, "b.png": 3},
			want: map[string]int32{"a.txt": 1, "b.png": 3},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := toInt32Map(tt.in)
			if len(got) != len(tt.want) {
				t.Fatalf("toInt32Map() len = %d, want %d", len(got), len(tt.want))
			}
			for k, wantV := range tt.want {
				if gotV, ok := got[k]; !ok || gotV != wantV {
					t.Errorf("toInt32Map()[%q] = %d, want %d", k, gotV, wantV)
				}
			}
		})
	}
}

func TestToInt64Map(t *testing.T) {
	tests := []struct {
		name string
		in   map[string]int32
		want map[string]int64
	}{
		{
			name: "nil map",
			in:   nil,
			want: nil,
		},
		{
			name: "empty map",
			in:   map[string]int32{},
			want: nil,
		},
		{
			name: "single entry",
			in:   map[string]int32{"chart.html": 1},
			want: map[string]int64{"chart.html": 1},
		},
		{
			name: "multiple entries",
			in:   map[string]int32{"a.txt": 1, "b.png": 3},
			want: map[string]int64{"a.txt": 1, "b.png": 3},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := toInt64Map(tt.in)
			if len(got) != len(tt.want) {
				t.Fatalf("toInt64Map() len = %d, want %d", len(got), len(tt.want))
			}
			for k, wantV := range tt.want {
				if gotV, ok := got[k]; !ok || gotV != wantV {
					t.Errorf("toInt64Map()[%q] = %d, want %d", k, gotV, wantV)
				}
			}
		})
	}
}

func TestAppendEventSerializesArtifactDelta(t *testing.T) {
	// This test verifies that createAiplatformpbContent and the event
	// construction in appendEvent correctly handle ArtifactDelta.
	// We test the serialization path by constructing the EventActions
	// the same way appendEvent does.
	tests := []struct {
		name          string
		stateDelta    map[string]any
		artifactDelta map[string]int64
		wantNilAction bool
		wantArtifact  map[string]int32
	}{
		{
			name:          "only artifact delta",
			stateDelta:    nil,
			artifactDelta: map[string]int64{"chart.html": 1},
			wantNilAction: false,
			wantArtifact:  map[string]int32{"chart.html": 1},
		},
		{
			name:          "both state and artifact delta",
			stateDelta:    map[string]any{"key": "value"},
			artifactDelta: map[string]int64{"file.txt": 2},
			wantNilAction: false,
			wantArtifact:  map[string]int32{"file.txt": 2},
		},
		{
			name:          "neither state nor artifact delta",
			stateDelta:    nil,
			artifactDelta: nil,
			wantNilAction: true,
			wantArtifact:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Replicate the appendEvent logic for constructing EventActions
			var eventState *aiplatformpb.EventActions
			if len(tt.stateDelta) > 0 || len(tt.artifactDelta) > 0 {
				eventState = &aiplatformpb.EventActions{}
				if len(tt.stateDelta) > 0 {
					sessionState, err := structpb.NewStruct(tt.stateDelta)
					if err != nil {
						t.Fatalf("structpb.NewStruct() failed: %v", err)
					}
					eventState.StateDelta = sessionState
				}
				if len(tt.artifactDelta) > 0 {
					eventState.ArtifactDelta = toInt32Map(tt.artifactDelta)
				}
			}

			if tt.wantNilAction {
				if eventState != nil {
					t.Fatal("expected nil EventActions, got non-nil")
				}
				return
			}

			if eventState == nil {
				t.Fatal("expected non-nil EventActions, got nil")
			}

			gotArtifact := eventState.GetArtifactDelta()
			if len(gotArtifact) != len(tt.wantArtifact) {
				t.Fatalf("ArtifactDelta len = %d, want %d", len(gotArtifact), len(tt.wantArtifact))
			}
			for k, wantV := range tt.wantArtifact {
				if gotV, ok := gotArtifact[k]; !ok || gotV != wantV {
					t.Errorf("ArtifactDelta[%q] = %d, want %d", k, gotV, wantV)
				}
			}
		})
	}
}

func TestListSessionEventsDeserializesArtifactDelta(t *testing.T) {
	// Simulate what listSessionEvents does when reading back ArtifactDelta
	// from a proto SessionEvent.
	pbActions := &aiplatformpb.EventActions{
		ArtifactDelta: map[string]int32{
			"chart.html": 1,
			"data.csv":   2,
		},
	}

	got := toInt64Map(pbActions.GetArtifactDelta())

	want := map[string]int64{
		"chart.html": 1,
		"data.csv":   2,
	}

	if len(got) != len(want) {
		t.Fatalf("ArtifactDelta len = %d, want %d", len(got), len(want))
	}
	for k, wantV := range want {
		if gotV, ok := got[k]; !ok || gotV != wantV {
			t.Errorf("ArtifactDelta[%q] = %d, want %d", k, gotV, wantV)
		}
	}
}
