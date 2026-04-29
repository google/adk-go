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

type streamQueryHandler struct {
	config        *launcher.Config
	methodName    string
	apiMode       string
	agentEngineID string
}

// NewStreamQueryHandler creates a new streamQueryHandler. It can be used to serve "async_stream_query" method
func NewStreamQueryHandler(config *launcher.Config, agentEngineID, methodName, apiMode string) *streamQueryHandler {
	return &streamQueryHandler{config: config, agentEngineID: agentEngineID, methodName: methodName, apiMode: apiMode}
}

// Handle generates stream of json-encoded responses based on the payload. Error are also emitted as errors
func (s *streamQueryHandler) Handle(ctx context.Context, rw http.ResponseWriter, payload []byte) error {
	streamErr := s.streamJSONL(ctx, rw, payload)
	// streamJSONL will return error only before streaming. In that case we can handle it with HTTP Status, which is done in upstream
	if streamErr != nil {
		err := fmt.Errorf("s.streamJSONL() failed: %w", streamErr)
		return err
	}
	return nil
}

// streamJSONL streams a single line for each event or error
func (s *streamQueryHandler) streamJSONL(ctx context.Context, rw http.ResponseWriter, payload []byte) error {
	var req models.StreamQueryRequest

	// try to unmarshal models.StreamQueryRequest first
	err := json.Unmarshal(payload, &req)
	if err != nil {
		// try to unmarshal models.StreamQueryTextRequest
		var reqText models.StreamQueryTextRequest
		errText := json.Unmarshal(payload, &reqText)
		if errText != nil {
			// cannot unmarshall to models.StreamQueryRequest and models.StreamQueryTextRequest
			err = fmt.Errorf("json.Unmarshal() failed both for models.StreamQueryRequest (%v) and models.StreamQueryTextRequest (%v)", err, errText)
			log.Print(err.Error())
			return err
		}
		// got text, create a full content based on that text
		req = models.StreamQueryRequest{
			ClassMethod: reqText.ClassMethod,
			Input: models.StreamQueryInput{
				UserID:    reqText.Input.UserID,
				SessionID: reqText.Input.SessionID,
				Message:   *genai.NewContentFromText(reqText.Input.Message, genai.RoleUser),
			},
		}
	}
	requestedSessionID, err := normalizeStreamQueryRequest(&req)
	if err != nil {
		err = fmt.Errorf("normalizeStreamQueryRequest() failed: %w", err)
		log.Print(err.Error())
		return err
	}
	if err := s.ensureBackendSession(ctx, &req, requestedSessionID); err != nil {
		err = fmt.Errorf("s.ensureBackendSession() failed: %w", err)
		log.Print(err.Error())
		return err
	}

	events, err := s.run(ctx, &req, &req.Input.Message, s.config)
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

		chunk := s.responseChunk(&req, event)
		err = helper.EmitJSON(rw, chunk)
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

func (s *streamQueryHandler) isAgentSpaceMethod() bool {
	return s.methodName == "streaming_agent_run_with_events"
}

func (s *streamQueryHandler) responseChunk(req *models.StreamQueryRequest, event *session.Event) any {
	if !s.isAgentSpaceMethod() {
		return *event
	}

	// AgentSpace / Gemini Enterprise expects the Python AdkApp-style stream
	// envelope, not a raw ADK event JSON line. Return the backend ADK session ID
	// so Gemini Enterprise can use it on later turns, matching Python AdkApp.
	return models.StreamingAgentRunWithEventsResponse{
		Events:    []*session.Event{event},
		SessionID: req.Input.SessionID,
	}
}

// normalizeStreamQueryRequest normalizes the two stream request shapes handled
// by this endpoint.
//
// Standard async_stream_query requests send message, user_id, and session_id as
// direct input fields. Gemini Enterprise / AgentSpace instead calls
// streaming_agent_run_with_events with the actual request encoded as a JSON
// string in input.request_json. When request_json is present, this function
// decodes it and replaces req.Input with the embedded request.
//
// The returned string is the caller-requested session_id before any backend ADK
// session normalization. For direct async_stream_query requests this is simply
// req.Input.SessionID. For Gemini Enterprise requests it is the session_id from
// the decoded request_json, which may be either a first-turn Gemini Enterprise
// session resource or a backend ADK session ID returned by a previous response.
func normalizeStreamQueryRequest(req *models.StreamQueryRequest) (string, error) {
	if req.Input.RequestJSON == "" {
		return req.Input.SessionID, nil
	}

	var input models.StreamQueryInput
	if err := json.Unmarshal([]byte(req.Input.RequestJSON), &input); err != nil {
		return "", fmt.Errorf("json.Unmarshal(input.request_json) failed: %w", err)
	}
	requestedSessionID := input.SessionID

	req.Input = input
	return requestedSessionID, nil
}

// ensureBackendSession normalizes AgentSpace / Gemini Enterprise requests so
// the runner always receives a backend ADK session ID.
//
// Gemini Enterprise calls streaming_agent_run_with_events with the actual
// request encoded in input.request_json. On the first turn, the embedded
// session_id is usually a Gemini Enterprise / Discovery Engine session resource:
//
//	projects/{project}/locations/global/collections/default_collection/engines/{engine}/sessions/{session}
//
// VertexAISessionService cannot use that resource name as a caller-provided
// backend session ID. This method therefore treats the incoming session_id as
// either:
//
//  1. A returned backend ADK session ID from a previous response. If it exists
//     in the configured session service, reuse it directly.
//  2. A first-turn Gemini Enterprise session resource. If no backend session
//     exists with that ID, create a new backend ADK session with a generated ID.
//
// The selected backend ID replaces req.Input.SessionID before the runner is
// invoked. The response envelope then returns that backend ID so Gemini
// Enterprise can send it on later turns, matching Python AdkApp behavior.
func (s *streamQueryHandler) ensureBackendSession(ctx context.Context, req *models.StreamQueryRequest, requestedSessionID string) error {
	if requestedSessionID == "" {
		return nil
	}
	if req.Input.UserID == "" {
		return fmt.Errorf("user_id is required for backend session handling")
	}
	if s.config == nil || s.config.SessionService == nil {
		return fmt.Errorf("session service is required for backend session handling")
	}

	getResp, err := s.config.SessionService.Get(ctx, &session.GetRequest{
		AppName:   s.agentEngineID,
		UserID:    req.Input.UserID,
		SessionID: requestedSessionID,
	})
	if err == nil && getResp.Session != nil {
		req.Input.SessionID = getResp.Session.ID()
		return nil
	}

	createResp, err := s.config.SessionService.Create(ctx, &session.CreateRequest{
		AppName: s.agentEngineID,
		UserID:  req.Input.UserID,
	})
	if err != nil {
		return fmt.Errorf("sessionService.Create() failed: %w", err)
	}

	req.Input.SessionID = createResp.Session.ID()
	return nil
}

// Name implements MethodHandler.
func (s *streamQueryHandler) Name() string {
	return s.methodName
}

var _ MethodHandler = (*streamQueryHandler)(nil)

// Metadata implements MethodHandler.
func (s *streamQueryHandler) Metadata() (*structpb.Struct, error) {
	if s.methodName == "streaming_agent_run_with_events" {
		return s.agentSpaceMetadata()
	}

	classAsyncMethod, err := structpb.NewStruct(map[string]any{
		"api_mode": s.apiMode,
		"name":     s.methodName,
		"parameters": map[string]any{
			"properties": map[string]any{
				"user_id": map[string]any{
					"type": "string",
				},
				"session_id": map[string]any{
					"nullable": true,
					"type":     "string",
				},
				"message": map[string]any{
					"anyOf": []any{
						map[string]any{
							"type": "string",
						},
						map[string]any{
							"additionalProperties": true,
							"type":                 "object",
						},
					},
				},
			},
			"required": []any{
				"message",
				"user_id",
			},
			"type": "object",
		},
		"description": `Streams responses asynchronously from the ADK application.
Args:
    message (genai.Content):
        Required. The message to stream responses for.
    user_id (str):
        Required. The ID of the user.
    session_id (str):
        Optional. The ID of the session. If not provided, a new session will be created for the user.

Yields:
    Single lines with JSON encoded event each. Errors are also emitted as JSON.

`,
	})
	if err != nil {
		return nil, fmt.Errorf("cannot create %s: %w", s.Name(), err)
	}
	return classAsyncMethod, nil
}

func (s *streamQueryHandler) agentSpaceMetadata() (*structpb.Struct, error) {
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
structured API. This method is primarily meant for invocation from AgentSpace
and Gemini Enterprise.

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

func (s *streamQueryHandler) run(ctx context.Context, req *models.StreamQueryRequest, message *genai.Content, config *launcher.Config) (iter.Seq2[*session.Event, error], error) {
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

	streamingMode := agent.StreamingModeSSE
	if s.methodName == "streaming_agent_run_with_events" {
		// The AgentSpace path mirrors Python AdkApp.streaming_agent_run_with_events,
		// which does not force SSE mode. Sending both partial and final events here
		// causes Gemini Enterprise to render duplicate answer text.
		streamingMode = agent.StreamingModeNone
	}

	return r.Run(ctx, req.Input.UserID, req.Input.SessionID, message, agent.RunConfig{
		StreamingMode: streamingMode,
	}), nil
}
