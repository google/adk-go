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

package adka2a

import (
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func TestMetadataTwoWayConversion(t *testing.T) {
	testCases := []struct {
		name    string
		event   *session.Event
		a2aMeta map[string]any
	}{
		{
			name: "error code",
			event: &session.Event{
				LLMResponse: model.LLMResponse{ErrorCode: "102"},
			},
			a2aMeta: map[string]any{metadataErrorCodeKey: "102"},
		},
		{
			name: "custom metadata",
			event: &session.Event{
				LLMResponse: model.LLMResponse{CustomMetadata: map[string]any{"nested": map[string]any{"key": "value"}}},
			},
			a2aMeta: map[string]any{metadataCustomMetaKey: map[string]any{"nested": map[string]any{"key": "value"}}},
		},
		{
			name: "usage",
			event: &session.Event{
				LLMResponse: model.LLMResponse{UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					CacheTokensDetails:      []*genai.ModalityTokenCount{{Modality: "text", TokenCount: 1}},
					PromptTokenCount:        1,
					ToolUsePromptTokenCount: 1,
				}},
			},
			a2aMeta: map[string]any{metadataUsageKey: map[string]any{
				"cacheTokensDetails":      []any{map[string]any{"modality": "text", "tokenCount": float64(1)}},
				"promptTokenCount":        float64(1),
				"toolUsePromptTokenCount": float64(1),
			}},
		},
		{
			name: "grounding",
			event: &session.Event{
				LLMResponse: model.LLMResponse{GroundingMetadata: &genai.GroundingMetadata{
					RetrievalQueries:  []string{"hello"},
					RetrievalMetadata: &genai.RetrievalMetadata{GoogleSearchDynamicRetrievalScore: 23.3},
				}},
			},
			a2aMeta: map[string]any{
				metadataGroundingKey: map[string]any{
					"retrievalQueries":  []any{string("hello")},
					"retrievalMetadata": map[string]any{"googleSearchDynamicRetrievalScore": 23.3},
				},
			},
		},
		{
			name:    "actions",
			event:   &session.Event{Actions: session.EventActions{TransferToAgent: "another", Escalate: true}},
			a2aMeta: map[string]any{metadataTransferToAgentKey: "another", metadataEscalateKey: true},
		},
		{
			name: "composite",
			event: &session.Event{
				LLMResponse: model.LLMResponse{
					GroundingMetadata: &genai.GroundingMetadata{
						SourceFlaggingUris: []*genai.GroundingMetadataSourceFlaggingURI{{SourceID: "id1"}},
					},
					UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
						CandidatesTokenCount: 12,
						ThoughtsTokenCount:   42,
					},
					CustomMetadata: map[string]any{
						"nested": map[string]any{"key": "value"},
					},
				},
				Actions: session.EventActions{TransferToAgent: "another", Escalate: true},
			},
			a2aMeta: map[string]any{
				metadataGroundingKey:       map[string]any{"sourceFlaggingUris": []any{map[string]any{"sourceId": "id1"}}},
				metadataUsageKey:           map[string]any{"candidatesTokenCount": float64(12), "thoughtsTokenCount": float64(42)},
				metadataCustomMetaKey:      map[string]any{"nested": map[string]any{"key": "value"}},
				metadataTransferToAgentKey: "another",
				metadataEscalateKey:        true,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			meta, err := toEventMeta(invocationMeta{}, tc.event)
			if err != nil {
				t.Errorf("toEventMeta() error = %v, want nil", err)
			}

			meta = setActionsMeta(meta, tc.event.Actions)
			if diff := cmp.Diff(tc.a2aMeta, meta); diff != "" {
				t.Errorf("toEventMeta() wrong result (+got,-want)\ngot = %v\nwant = %v\ndiff = %s", meta, tc.a2aMeta, diff)
			}

			event := &session.Event{}
			err = processA2AMeta(&a2a.TaskStatusUpdateEvent{Metadata: meta}, event)
			if err != nil {
				t.Errorf("processA2AMeta() error = %v, want nil", err)
			}
			if diff := cmp.Diff(tc.event, event); diff != "" {
				t.Errorf("processA2AMeta() wrong result (+got,-want)\ngot = %v\nwant = %v\ndiff = %s", event, tc.event, diff)
			}
		})
	}
}
