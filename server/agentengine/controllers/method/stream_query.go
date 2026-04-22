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

	err := json.Unmarshal(payload, &req)
	if err != nil {
		err = fmt.Errorf("json.Unmarshal() failed: %v", err)
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

		chunk := *event
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

// Name implements MethodHandler.
func (s *streamQueryHandler) Name() string {
	return s.methodName
}

var _ MethodHandler = (*streamQueryHandler)(nil)

// Metadata implements MethodHandler.
func (s *streamQueryHandler) Metadata() (*structpb.Struct, error) {
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
					"additionalProperties": true,
					"type":                 "object",
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

func (s *streamQueryHandler) run(ctx context.Context, req *models.StreamQueryRequest, message *genai.Content, config *launcher.Config) (iter.Seq2[*session.Event, error], error) {
	rootAgent := config.AgentLoader.RootAgent()

	r, err := runner.New(runner.Config{
		AppName:         s.agentEngineID,
		Agent:           rootAgent,
		SessionService:  config.SessionService,
		ArtifactService: config.ArtifactService,
		PluginConfig:    config.PluginConfig,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create runner: %v", err)
	}

	return r.Run(ctx, req.Input.UserID, req.Input.SessionID, message, agent.RunConfig{
		StreamingMode: agent.StreamingModeSSE,
	}), nil
}
