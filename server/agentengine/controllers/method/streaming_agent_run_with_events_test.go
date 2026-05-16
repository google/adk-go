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

package method

import (
	"context"
	"encoding/json"
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/model"
	"google.golang.org/adk/server/agentengine/internal/models"
	"google.golang.org/adk/session"
)

type agentSpaceStreamResponse struct {
	Events    []simpleEvent `json:"events"`
	SessionID string        `json:"session_id"`
}

type streamAwareLLM struct{}

func (streamAwareLLM) Name() string {
	return "stream-aware-llm"
}

func (streamAwareLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if stream {
			if !yield(&model.LLMResponse{
				Content: genai.NewContentFromText("partial response", genai.RoleModel),
				Partial: true,
			}, nil) {
				return
			}
		}
		yield(&model.LLMResponse{
			Content: genai.NewContentFromText("final response", genai.RoleModel),
		}, nil)
	}
}

func TestDecodeStreamingAgentRunWithEventsRequest(t *testing.T) {
	payload := []byte(`{
		"class_method": "streaming_agent_run_with_events",
		"input": {
			"request_json": "{\"message\":{\"role\":\"user\",\"parts\":[{\"text\":\"Hi\"}]},\"session_id\":\"projects/619863079366/locations/global/collections/default_collection/engines/launchpad-secchot/sessions/18317904751077773933\",\"user_id\":\"jan@example.com\"}"
		}
	}`)
	var req models.StreamingAgentRunWithEventsRequest
	if err := json.Unmarshal(payload, &req); err != nil {
		t.Fatalf("json.Unmarshal() failed: %v", err)
	}

	got, requestedSessionID, err := decodeStreamingAgentRunWithEventsRequest(&req)
	if err != nil {
		t.Fatalf("decodeStreamingAgentRunWithEventsRequest() failed: %v", err)
	}

	want := &models.StreamingAgentRunWithEventsRunRequest{
		UserID:    "jan@example.com",
		SessionID: "projects/619863079366/locations/global/collections/default_collection/engines/launchpad-secchot/sessions/18317904751077773933",
		Message: genai.Content{
			Role:  "user",
			Parts: []*genai.Part{{Text: "Hi"}},
		},
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("decodeStreamingAgentRunWithEventsRequest() mismatch (-want +got):\n%s", diff)
	}
	if requestedSessionID != want.SessionID {
		t.Errorf("requestedSessionID = %q, want %q", requestedSessionID, want.SessionID)
	}
}

func TestEnsureBackendSession_CreateBackendSession(t *testing.T) {
	sessionService := session.InMemoryService()
	handler := NewStreamingAgentRunWithEventsHandler(&launcher.Config{SessionService: sessionService}, "app", "streaming_agent_run_with_events", "async_stream")
	req := &models.StreamingAgentRunWithEventsRunRequest{
		UserID:    "jan@example.com",
		SessionID: "projects/619863079366/locations/global/collections/default_collection/engines/launchpad-secchot/sessions/1737792313033595937",
	}
	requestedSessionID := req.SessionID

	if err := handler.ensureBackendSession(t.Context(), req, requestedSessionID); err != nil {
		t.Fatalf("ensureBackendSession() failed: %v", err)
	}
	if req.SessionID == "" || req.SessionID == requestedSessionID {
		t.Fatalf("SessionID = %q, want generated backend session ID", req.SessionID)
	}

	got, err := sessionService.Get(t.Context(), &session.GetRequest{
		AppName:   "app",
		UserID:    "jan@example.com",
		SessionID: req.SessionID,
	})
	if err != nil {
		t.Fatalf("Get() failed: %v", err)
	}
	if got.Session.ID() != req.SessionID {
		t.Errorf("stored SessionID = %q, want %q", got.Session.ID(), req.SessionID)
	}
}

func TestEnsureBackendSession_ReuseReturnedBackendSession(t *testing.T) {
	sessionService := session.InMemoryService()
	created, err := sessionService.Create(t.Context(), &session.CreateRequest{
		AppName: "app",
		UserID:  "jan@example.com",
	})
	if err != nil {
		t.Fatalf("Create() failed: %v", err)
	}

	handler := NewStreamingAgentRunWithEventsHandler(&launcher.Config{SessionService: sessionService}, "app", "streaming_agent_run_with_events", "async_stream")
	req := &models.StreamingAgentRunWithEventsRunRequest{
		UserID:    "jan@example.com",
		SessionID: created.Session.ID(),
	}

	if err := handler.ensureBackendSession(t.Context(), req, req.SessionID); err != nil {
		t.Fatalf("ensureBackendSession() failed: %v", err)
	}
	if req.SessionID != created.Session.ID() {
		t.Errorf("SessionID = %q, want existing backend session %q", req.SessionID, created.Session.ID())
	}
}

func TestStreamJSONL_AgentSpaceResponseEnvelope(t *testing.T) {
	const (
		appName           = "app"
		userID            = "jan@example.com"
		externalSessionID = "projects/619863079366/locations/global/collections/default_collection/engines/launchpad-secchot/sessions/5383754056151277294"
	)

	a, err := llmagent.New(llmagent.Config{
		Name: "Echo",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(cc agent.CallbackContext) (*genai.Content, error) {
				return cc.UserContent(), nil
			},
		},
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(a),
		SessionService: session.InMemoryService(),
	}
	h := NewStreamingAgentRunWithEventsHandler(config, appName, "streaming_agent_run_with_events", "async_stream")

	requestJSON := `{"message":{"role":"user","parts":[{"text":"Please"}]},"session_id":"` + externalSessionID + `","user_id":"` + userID + `"}`
	payload, err := json.Marshal(models.StreamingAgentRunWithEventsRequest{
		ClassMethod: "streaming_agent_run_with_events",
		Input: models.StreamingAgentRunWithEventsInput{
			RequestJSON: requestJSON,
		},
	})
	if err != nil {
		t.Fatalf("json.Marshal() failed: %v", err)
	}

	w := newStringWriter()
	if err := h.streamJSONL(t.Context(), w, payload); err != nil {
		t.Fatalf("streamJSONL() failed: %v", err)
	}

	var got agentSpaceStreamResponse
	if err := json.Unmarshal([]byte(w.sb.String()), &got); err != nil {
		t.Fatalf("json.Unmarshal() failed: %v", err)
	}
	if got.SessionID == "" || got.SessionID == externalSessionID {
		t.Fatalf("SessionID = %q, want generated backend session ID", got.SessionID)
	}
	if len(got.Events) != 1 {
		t.Fatalf("len(Events) = %d, want 1", len(got.Events))
	}

	wantContent := genai.NewContentFromText("Please", genai.RoleUser)
	if diff := cmp.Diff(wantContent, got.Events[0].Content); diff != "" {
		t.Errorf("event content mismatch (-want +got):\n%s", diff)
	}
}

func TestStreamJSONL_AgentSpaceAcceptsReturnedBackendSessionID(t *testing.T) {
	const (
		appName           = "app"
		userID            = "jan@example.com"
		externalSessionID = "projects/619863079366/locations/global/collections/default_collection/engines/launchpad-secchot/sessions/17538412265490363072"
	)

	a, err := llmagent.New(llmagent.Config{
		Name: "Echo",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(cc agent.CallbackContext) (*genai.Content, error) {
				return cc.UserContent(), nil
			},
		},
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(a),
		SessionService: session.InMemoryService(),
	}
	h := NewStreamingAgentRunWithEventsHandler(config, appName, "streaming_agent_run_with_events", "async_stream")

	run := func(message, sessionID string) agentSpaceStreamResponse {
		t.Helper()
		requestJSON := `{"message":{"role":"user","parts":[{"text":"` + message + `"}]},"session_id":"` + sessionID + `","user_id":"` + userID + `"}`
		payload, err := json.Marshal(models.StreamingAgentRunWithEventsRequest{
			ClassMethod: "streaming_agent_run_with_events",
			Input: models.StreamingAgentRunWithEventsInput{
				RequestJSON: requestJSON,
			},
		})
		if err != nil {
			t.Fatalf("json.Marshal() failed: %v", err)
		}

		w := newStringWriter()
		if err := h.streamJSONL(t.Context(), w, payload); err != nil {
			t.Fatalf("streamJSONL() failed: %v", err)
		}

		var got agentSpaceStreamResponse
		if err := json.Unmarshal([]byte(w.sb.String()), &got); err != nil {
			t.Fatalf("json.Unmarshal() failed: %v", err)
		}
		return got
	}

	first := run("Hi", externalSessionID)
	second := run("Again", first.SessionID)

	if first.SessionID == "" || first.SessionID == externalSessionID {
		t.Fatalf("first SessionID = %q, want generated backend session ID", first.SessionID)
	}
	if second.SessionID != first.SessionID {
		t.Fatalf("second SessionID = %q, want returned backend session ID %q", second.SessionID, first.SessionID)
	}

	list, err := config.SessionService.List(t.Context(), &session.ListRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		t.Fatalf("List() failed: %v", err)
	}
	if len(list.Sessions) != 1 {
		t.Fatalf("len(Sessions) = %d, want 1", len(list.Sessions))
	}
}

func TestStreamJSONL_AgentSpaceUsesNonStreamingMode(t *testing.T) {
	const (
		appName           = "app"
		userID            = "jan@example.com"
		externalSessionID = "projects/619863079366/locations/global/collections/default_collection/engines/launchpad-secchot/sessions/8130118279508262912"
	)

	a, err := llmagent.New(llmagent.Config{
		Name:  "StreamAware",
		Model: streamAwareLLM{},
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(a),
		SessionService: session.InMemoryService(),
	}
	h := NewStreamingAgentRunWithEventsHandler(config, appName, "streaming_agent_run_with_events", "async_stream")

	requestJSON := `{"message":{"role":"user","parts":[{"text":"What is your capabilities"}]},"session_id":"` + externalSessionID + `","user_id":"` + userID + `"}`
	payload, err := json.Marshal(models.StreamingAgentRunWithEventsRequest{
		ClassMethod: "streaming_agent_run_with_events",
		Input: models.StreamingAgentRunWithEventsInput{
			RequestJSON: requestJSON,
		},
	})
	if err != nil {
		t.Fatalf("json.Marshal() failed: %v", err)
	}

	w := newStringWriter()
	if err := h.streamJSONL(t.Context(), w, payload); err != nil {
		t.Fatalf("streamJSONL() failed: %v", err)
	}

	var got agentSpaceStreamResponse
	if err := json.Unmarshal([]byte(w.sb.String()), &got); err != nil {
		t.Fatalf("json.Unmarshal() failed: %v", err)
	}
	if len(got.Events) != 1 {
		t.Fatalf("len(Events) = %d, want 1", len(got.Events))
	}

	wantContent := genai.NewContentFromText("final response", genai.RoleModel)
	if diff := cmp.Diff(wantContent, got.Events[0].Content); diff != "" {
		t.Errorf("event content mismatch (-want +got):\n%s", diff)
	}
}

func TestStreamingAgentRunWithEventsHandlerMetadata(t *testing.T) {
	handler := NewStreamingAgentRunWithEventsHandler(nil, "", "streaming_agent_run_with_events", "stream")

	got, err := handler.Metadata()
	if err != nil {
		t.Fatalf("Metadata() failed: %v", err)
	}

	want := map[string]any{
		"api_mode": "stream",
		"name":     "streaming_agent_run_with_events",
		"parameters": map[string]any{
			"properties": map[string]any{
				"request_json": map[string]any{
					"type": "string",
				},
			},
			"required": []any{"request_json"},
			"type":     "object",
		},
	}

	if diff := cmp.Diff(want, got.AsMap(), cmpopts.IgnoreMapEntries(func(k string, _ any) bool {
		return k == "description"
	})); diff != "" {
		t.Errorf("Metadata() mismatch (-want +got):\n%s", diff)
	}
}
