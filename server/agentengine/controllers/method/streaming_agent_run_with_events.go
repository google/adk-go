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
	"fmt"
	"iter"
	"log"
	"net/http"

	"google.golang.org/genai"
	"google.golang.org/protobuf/types/known/structpb"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/agentengine/internal/helper"
	"google.golang.org/adk/server/agentengine/internal/models"
	"google.golang.org/adk/session"
)

type streamingAgentRunWithEventsHandler struct {
	config        *launcher.Config
	methodName    string
	apiMode       string
	agentEngineID string
}

// NewStreamingAgentRunWithEventsHandler creates a new handler for streaming_agent_run_with_events.
func NewStreamingAgentRunWithEventsHandler(config *launcher.Config, agentEngineID, methodName, apiMode string) *streamingAgentRunWithEventsHandler {
	return &streamingAgentRunWithEventsHandler{config: config, agentEngineID: agentEngineID, methodName: methodName, apiMode: apiMode}
}

// Handle generates stream of json-encoded responses based on the payload. Errors are also emitted as errors.
func (s *streamingAgentRunWithEventsHandler) Handle(ctx context.Context, rw http.ResponseWriter, payload []byte) error {
	streamErr := s.streamJSONL(ctx, rw, payload)
	// streamJSONL will return error only before streaming. In that case we can handle it with HTTP Status, which is done in upstream.
	if streamErr != nil {
		err := fmt.Errorf("s.streamJSONL() failed: %w", streamErr)
		return err
	}
	return nil
}

// streamJSONL streams a single line for each event or error.
func (s *streamingAgentRunWithEventsHandler) streamJSONL(ctx context.Context, rw http.ResponseWriter, payload []byte) error {
	var req models.StreamingAgentRunWithEventsRequest
	if err := json.Unmarshal(payload, &req); err != nil {
		err = fmt.Errorf("json.Unmarshal() failed for models.StreamingAgentRunWithEventsRequest: %w", err)
		log.Print(err.Error())
		return err
	}

	runReq, requestedSessionID, err := decodeStreamingAgentRunWithEventsRequest(&req)
	if err != nil {
		err = fmt.Errorf("decodeStreamingAgentRunWithEventsRequest() failed: %w", err)
		log.Print(err.Error())
		return err
	}
	if err := s.ensureBackendSession(ctx, runReq, requestedSessionID); err != nil {
		err = fmt.Errorf("s.ensureBackendSession() failed: %w", err)
		log.Print(err.Error())
		return err
	}

	events, err := s.run(ctx, runReq, &runReq.Message, s.config)
	if err != nil {
		err = fmt.Errorf("s.run() failed: %w", err)
		log.Print(err.Error())
		return err
	}

	rw.Header().Set("Content-Type", "application/json")
	rw.Header().Set("Cache-Control", "no-cache")
	rw.Header().Set("Connection", "keep-alive")
	// from this moment on we must not return error. Instead, it should be handled by using helper.EmitJSONError

	for event, err := range events {
		log.Printf("Processing event: %+v err: %+v\n", event, err)
		if err != nil {
			log.Printf("error in events: %v\n", err)
			e := helper.EmitJSONError(rw, err)
			if e != nil {
				e = fmt.Errorf("helper.EmitJSONError() failed: %w", e)
				log.Print(e.Error())
			}
			break
		}
		if event == nil {
			continue
		}
		if event.LLMResponse.Content == nil {
			continue
		}

		err = helper.EmitJSON(rw, models.StreamingAgentRunWithEventsResponse{
			Events:    []*session.Event{event},
			SessionID: runReq.SessionID,
		})
		if err != nil {
			e := fmt.Errorf("helper.EmitJSON() failed: %w", err)
			log.Print(e.Error())
			e = helper.EmitJSONError(rw, e)
			if e != nil {
				e = fmt.Errorf("helper.EmitJSONError() failed: %w", e)
				log.Print(e.Error())
			}
			break
		}
	}
	return nil
}

// decodeStreamingAgentRunWithEventsRequest decodes input.request_json and returns
// the caller-requested session_id before backend ADK session normalization.
func decodeStreamingAgentRunWithEventsRequest(req *models.StreamingAgentRunWithEventsRequest) (*models.StreamingAgentRunWithEventsRunRequest, string, error) {
	var runReq models.StreamingAgentRunWithEventsRunRequest
	if err := json.Unmarshal([]byte(req.Input.RequestJSON), &runReq); err != nil {
		return nil, "", fmt.Errorf("json.Unmarshal(input.request_json) failed: %w", err)
	}
	return &runReq, runReq.SessionID, nil
}

// ensureBackendSession normalizes Gemini Enterprise requests so
// the runner always receives a backend ADK session ID.
//
// On the first turn, the embedded session_id is usually a Gemini Enterprise /
// Discovery Engine session resource:
//
//	projects/{project}/locations/global/collections/default_collection/engines/{engine}/sessions/{session}
//
// VertexAISessionService cannot use that resource name as a caller-provided
// backend session ID. This method treats the incoming session_id as either a
// backend ADK session ID returned by a previous response, or an external
// first-turn resource that needs a newly-created backend session.
func (s *streamingAgentRunWithEventsHandler) ensureBackendSession(ctx context.Context, req *models.StreamingAgentRunWithEventsRunRequest, requestedSessionID string) error {
	if requestedSessionID == "" {
		return nil
	}
	if req.UserID == "" {
		return fmt.Errorf("user_id is required for backend session handling")
	}
	if s.config == nil || s.config.SessionService == nil {
		return fmt.Errorf("session service is required for backend session handling")
	}

	getResp, err := s.config.SessionService.Get(ctx, &session.GetRequest{
		AppName:   s.agentEngineID,
		UserID:    req.UserID,
		SessionID: requestedSessionID,
	})
	if err == nil && getResp.Session != nil {
		req.SessionID = getResp.Session.ID()
		return nil
	}

	createResp, err := s.config.SessionService.Create(ctx, &session.CreateRequest{
		AppName: s.agentEngineID,
		UserID:  req.UserID,
	})
	if err != nil {
		return fmt.Errorf("sessionService.Create() failed: %w", err)
	}

	req.SessionID = createResp.Session.ID()
	return nil
}

// Name implements MethodHandler.
func (s *streamingAgentRunWithEventsHandler) Name() string {
	return s.methodName
}

var _ MethodHandler = (*streamingAgentRunWithEventsHandler)(nil)

// Metadata implements MethodHandler.
func (s *streamingAgentRunWithEventsHandler) Metadata() (*structpb.Struct, error) {
	classAsyncMethod, err := structpb.NewStruct(map[string]any{
		"api_mode": s.apiMode,
		"name":     s.methodName,
		"parameters": map[string]any{
			"properties": map[string]any{
				"request_json": map[string]any{
					"type": "string",
				},
			},
			"required": []any{
				"request_json",
			},
			"type": "object",
		},
		"description": `Streams responses asynchronously from the ADK application.

In general, you should use async_stream_query instead, as it has a more
structured API. This method is primarily meant for invocation from Gemini Enterprise.

Args:
    request_json (str):
        Required. The request to stream responses for.

`,
	})
	if err != nil {
		return nil, fmt.Errorf("cannot create %s: %w", s.Name(), err)
	}
	return classAsyncMethod, nil
}

func (s *streamingAgentRunWithEventsHandler) run(ctx context.Context, req *models.StreamingAgentRunWithEventsRunRequest, message *genai.Content, config *launcher.Config) (iter.Seq2[*session.Event, error], error) {
	rootAgent := config.AgentLoader.RootAgent()

	r, err := runner.New(runner.Config{
		AppName:           s.agentEngineID,
		Agent:             rootAgent,
		SessionService:    config.SessionService,
		ArtifactService:   config.ArtifactService,
		PluginConfig:      config.PluginConfig,
		AutoCreateSession: true,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create runner: %v", err)
	}

	// The path mirrors Python AdkApp.streaming_agent_run_with_events,
	// which does not force SSE mode. Sending both partial and final events here
	// causes Gemini Enterprise to render duplicate answer text.
	return r.Run(ctx, req.UserID, req.SessionID, message, agent.RunConfig{
		StreamingMode: agent.StreamingModeNone,
	}), nil
}
