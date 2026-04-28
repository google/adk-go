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
	"net/http"

	"google.golang.org/protobuf/types/known/structpb"

	"google.golang.org/adk/server/agentengine/internal/models"
	"google.golang.org/adk/session"
)

type deleteSessionHandler struct {
	sessionservice session.Service
	agentEngineID  string
	methodName     string
	apiMode        string
}

// NewDeleteSessionHandler creates a new deleteSessionHandler. It can be used to serve "async_delete_session" method
func NewDeleteSessionHandler(sessionservice session.Service, agentEngineID, methodName, apiMode string) *deleteSessionHandler {
	return &deleteSessionHandler{sessionservice: sessionservice, agentEngineID: agentEngineID, methodName: methodName, apiMode: apiMode}
}

// Metadata implements MethodHandler.
func (g *deleteSessionHandler) Metadata() (*structpb.Struct, error) {
	metadata, err := structpb.NewStruct(map[string]any{
		"api_mode": g.apiMode,
		"name":     g.methodName,
		"parameters": map[string]any{
			"properties": map[string]any{
				"session_id": map[string]any{
					"type": "string",
				},
				"user_id": map[string]any{
					"type": "string",
				},
			},
			"required": []any{
				"user_id",
				"session_id",
			},
			"type": "object",
		},
		"description": `Deletes a session for the given user.

Args:
    user_id (str):
        Required. The ID of the user.
    session_id (str):
        Required. The ID of the session.

Returns:
	on success returns an empty string. On error returns an error message.

`,
	})
	if err != nil {
		return nil, fmt.Errorf("cannot create metadata for %s: %w", g.methodName, err)
	}
	return metadata, nil
}

// Handle implements MethodHandler.
func (g *deleteSessionHandler) Handle(ctx context.Context, rw http.ResponseWriter, payload []byte) error {
	var req models.DeleteSessionRequest

	err := json.Unmarshal(payload, &req)
	if err != nil {
		return fmt.Errorf("json.Unmarshal() failed: %v", err)
	}

	ssReq := &session.DeleteRequest{
		AppName:   g.agentEngineID,
		UserID:    req.Input.UserID,
		SessionID: req.Input.SessionID,
	}
	err = g.sessionservice.Delete(ctx, ssReq)
	if err != nil {
		return fmt.Errorf("g.sessionservice.Delete() failed: %v", err)
	}

	result := models.DeleteSessionResponse{}
	err = json.NewEncoder(rw).Encode(result)
	if err != nil {
		return fmt.Errorf("json.NewEncoder failed: %v", err)
	}
	return nil
}

// Name implements MethodHandler.
func (g *deleteSessionHandler) Name() string {
	return g.methodName
}

var _ MethodHandler = (*deleteSessionHandler)(nil)
