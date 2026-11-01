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
	"maps"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/converters"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestMetadataTwoWayConversion(t *testing.T) {
	testCases := []struct {
		name    string
		event   *session.Event
		a2aMeta map[string]any
		// wantAfterProcessA2AMeta overrides event for the processA2AMeta
		// direction. Needed only when TransferToAgent is set, since that's
		// never restored from peer metadata (see toEventActions).
		wantAfterProcessA2AMeta *session.Event
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
			// processA2AMeta never restores TransferToAgent from peer metadata.
			wantAfterProcessA2AMeta: &session.Event{Actions: session.EventActions{Escalate: true}},
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
			// processA2AMeta never restores TransferToAgent from peer metadata;
			// everything else round-trips.
			wantAfterProcessA2AMeta: &session.Event{
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
				Actions: session.EventActions{Escalate: true},
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
			// processA2AMeta never restores TransferToAgent from peer
			// metadata (see TransferToAgentFromMeta for how a caller
			// restores it explicitly, after making its own trust decision
			// about the remote peer).
			err = processA2AMeta(&a2a.TaskStatusUpdateEvent{Metadata: meta}, event)
			if err != nil {
				t.Errorf("processA2AMeta() error = %v, want nil", err)
			}
			want := tc.event
			if tc.wantAfterProcessA2AMeta != nil {
				want = tc.wantAfterProcessA2AMeta
			}
			if diff := cmp.Diff(want, event); diff != "" {
				t.Errorf("processA2AMeta() wrong result (+got,-want)\ngot = %v\nwant = %v\ndiff = %s", event, want, diff)
			}
		})
	}
}

func TestToEventMetaAndArtifactMakerEquivalence(t *testing.T) {
	invMeta := invocationMeta{
		userID:    "user-1",
		sessionID: "sess-1",
		agentName: "agent-1",
		eventMeta: map[string]any{
			ToA2AMetaKey("app_name"):   "app-1",
			ToA2AMetaKey("user_id"):    "user-1",
			ToA2AMetaKey("session_id"): "sess-1",
		},
	}

	cases := []struct {
		name  string
		event *session.Event
	}{
		{
			name:  "empty event with typed nil metadata pointers",
			event: &session.Event{},
		},
		{
			name: "partial chunk with invocation author branch",
			event: &session.Event{
				InvocationID: "inv-42",
				Author:       "agent-1",
				Branch:       "root.agent-1",
				LLMResponse: model.LLMResponse{
					Partial: true,
					Content: genai.NewContentFromText("chunk", genai.RoleModel),
				},
			},
		},
		{
			name: "non-nil zero metadata structs",
			event: &session.Event{
				InvocationID: "inv-zero",
				LLMResponse: model.LLMResponse{
					CitationMetadata:  &genai.CitationMetadata{},
					GroundingMetadata: &genai.GroundingMetadata{},
					UsageMetadata:     &genai.GenerateContentResponseUsageMetadata{},
				},
			},
		},
		{
			name: "populated metadata and error code with unicode",
			event: &session.Event{
				InvocationID: "inv-full",
				Author:       "autor-żółć",
				Branch:       "gałąź.1",
				LLMResponse: model.LLMResponse{
					ErrorCode: "429",
					CitationMetadata: &genai.CitationMetadata{
						Citations: []*genai.Citation{{Title: "Źródło", URI: "https://example.com"}},
					},
					GroundingMetadata: &genai.GroundingMetadata{
						WebSearchQueries: []string{"zapytanie"},
					},
					UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
						PromptTokenCount:     100,
						CandidatesTokenCount: 50,
						TotalTokenCount:      150,
					},
					CustomMetadata: map[string]any{"klucz": "wartość"},
				},
				Actions: session.EventActions{Escalate: true},
			},
		},
	}

	reqCtx := &a2asrv.ExecutorContext{TaskID: "task-1", ContextID: "ctx-1"}
	ignoreArtifactID := cmpopts.IgnoreFields(a2a.Artifact{}, "ID")

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotMeta, gotErr := toEventMeta(invMeta, tc.event)
			wantMeta, wantErr := referenceToEventMeta(invMeta, tc.event)
			if (gotErr != nil) != (wantErr != nil) {
				t.Fatalf("toEventMeta error = %v, want %v", gotErr, wantErr)
			}
			if diff := cmp.Diff(wantMeta, gotMeta); diff != "" {
				t.Errorf("toEventMeta mismatch (-want +got):\n%s", diff)
			}

			maker := newArtifactMaker(reqCtx)
			refMaker := newArtifactMaker(reqCtx)
			parts := []*a2a.Part{a2a.NewTextPart("part")}

			gotArt, err := maker.transform(tc.event, parts, gotMeta)
			if err != nil {
				t.Fatalf("maker.transform failed: %v", err)
			}
			wantArt, err := referenceArtifactMakerTransform(refMaker, tc.event, parts, wantMeta)
			if err != nil {
				t.Fatalf("referenceArtifactMakerTransform failed: %v", err)
			}
			if diff := cmp.Diff(wantArt, gotArt, ignoreArtifactID); diff != "" {
				t.Errorf("artifactMaker.transform mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func referenceToEventMeta(meta invocationMeta, event *session.Event) (map[string]any, error) {
	result := make(map[string]any)
	maps.Copy(result, meta.eventMeta)

	for k, v := range map[string]string{
		"invocation_id": event.InvocationID,
		"author":        event.Author,
		"branch":        event.Branch,
	} {
		if v != "" {
			result[ToA2AMetaKey(k)] = v
		}
	}

	if err := referenceAddMeta(result, metadataCitationKey, event.CitationMetadata); err != nil {
		return nil, err
	}
	if err := referenceAddMeta(result, metadataGroundingKey, event.GroundingMetadata); err != nil {
		return nil, err
	}
	if err := referenceAddMeta(result, metadataUsageKey, event.UsageMetadata); err != nil {
		return nil, err
	}
	if event.CustomMetadata != nil {
		result[metadataCustomMetaKey] = event.CustomMetadata
	}
	if event.LLMResponse.ErrorCode != "" {
		result[metadataErrorCodeKey] = event.LLMResponse.ErrorCode
	}

	return result, nil
}

func referenceAddMeta(result map[string]any, key string, data any) error {
	if data == nil {
		return nil
	}
	v, err := converters.ToMapStructure(data)
	if err != nil {
		return err
	}
	if v == nil {
		return nil
	}
	result[key] = v
	return nil
}

func referenceArtifactMakerTransform(m *artifactMaker, event *session.Event, parts []*a2a.Part, meta map[string]any) (*a2a.TaskArtifactUpdateEvent, error) {
	result := a2a.NewArtifactEvent(m.reqCtx, parts...)
	if artifactID, ok := m.lastAgentPartialArtifact[event.Author]; ok {
		result.Artifact.ID = artifactID
		result.Append = event.Partial
	}
	result.LastChunk = !event.Partial
	if event.Partial {
		m.lastAgentPartialArtifact[event.Author] = result.Artifact.ID
	} else {
		delete(m.lastAgentPartialArtifact, event.Author)
	}
	if len(meta) > 0 {
		if result.Artifact.Metadata == nil {
			result.Artifact.Metadata = make(map[string]any)
		}
		maps.Copy(result.Artifact.Metadata, meta)
	}
	return result, nil
}
