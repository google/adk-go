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

type createSessionHandler struct {
	sessionService session.Service
	agentEngineID  string
	methodName     string
	apiMode        string
}

// NewCreateSessionHandler creates a new createSessionHandler. It can be used to serve "asynch_create_session" method
func NewCreateSessionHandler(sessionService session.Service, agentEngineID, methodName, apiMode string) *createSessionHandler {
	return &createSessionHandler{sessionService: sessionService, agentEngineID: agentEngineID, methodName: methodName, apiMode: apiMode}
}

// Metadata implements MethodHandler.
func (c *createSessionHandler) Metadata() (*structpb.Struct, error) {
	metadata, err := structpb.NewStruct(map[string]any{
		"api_mode": c.apiMode,
		"name":     c.methodName,
		"parameters": map[string]any{
			"properties": map[string]any{
				"user_id": map[string]any{
					"type": "string",
				},
				"state": map[string]any{
					"nullable": true,
					"type":     "object",
				},
			},
			"required": []any{
				"user_id",
			},
			"type": "object",
		},
		"description": `Creates a new session.
		
Args:
    user_id (str):
	        Required. The ID of the user.
    state (dict[str, Any]):
        Optional. The initial state of the session.

Returns:
    Session: The newly created session instance.
`,
	})
	if err != nil {
		return nil, fmt.Errorf("cannot create metadata for %s: %w", c.methodName, err)
	}
	return metadata, nil
}

// Handle implements MethodHandler.
func (c *createSessionHandler) Handle(ctx context.Context, rw http.ResponseWriter, payload []byte) error {
	var req models.CreateSessionRequest

	err := json.Unmarshal(payload, &req)
	if err != nil {
		return fmt.Errorf("json.Unmarshal() failed: %v", err)
	}

	ssReq := &session.CreateRequest{
		AppName: c.agentEngineID,
		UserID:  req.Input.UserID,
		State:   req.Input.State,
	}
	resp, err := c.sessionService.Create(ctx, ssReq)
	if err != nil {
		return fmt.Errorf("c.sessionservice.Create() failed: %v", err)
	}

	sd := models.FromSession(resp.Session)

	result := models.CreateSessionResponse{
		Output: sd,
	}
	err = json.NewEncoder(rw).Encode(result)
	if err != nil {
		return fmt.Errorf("json.NewEncoder failed: %v", err)
	}
	return nil
}

// Name implements MethodHandler.
func (c *createSessionHandler) Name() string {
	return c.methodName
}

var _ MethodHandler = (*createSessionHandler)(nil)
