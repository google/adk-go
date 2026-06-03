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

type listSessionHandler struct {
	sessionservice session.Service
	agentEngineID  string
	methodName     string
	apiMode        string
}

// NewListSessionHandler creates a new listSessionHandler. It can be used to serve "async_list_session" method
func NewListSessionHandler(sessionservice session.Service, agentEngineID, methodName, apiMode string) *listSessionHandler {
	return &listSessionHandler{sessionservice: sessionservice, agentEngineID: agentEngineID, methodName: methodName, apiMode: apiMode}
}

// Metadata implements MethodHandler.
func (l *listSessionHandler) Metadata() (*structpb.Struct, error) {
	metadata, err := structpb.NewStruct(map[string]any{
		"api_mode": l.apiMode,
		"name":     l.methodName,
		"parameters": map[string]any{
			"properties": map[string]any{
				"user_id": map[string]any{
					"type": "string",
				},
			},
			"required": []any{
				"user_id",
			},
			"type": "object",
		},
		"description": `List sessions for the given user.

Args:
    user_id (str):
        Required. The ID of the user.

Returns:
    ListSessionsResponse: The list of sessions with data.
`,
	})
	if err != nil {
		return nil, fmt.Errorf("cannot create metadata for %s: %w", l.methodName, err)
	}
	return metadata, nil
}

// Handle implements MethodHandler.
func (l *listSessionHandler) Handle(ctx context.Context, rw http.ResponseWriter, payload []byte) error {
	var req models.ListSessionRequest

	err := json.Unmarshal(payload, &req)
	if err != nil {
		return fmt.Errorf("json.Unmarshal() failed: %v", err)
	}

	ssReq := &session.ListRequest{
		AppName: l.agentEngineID,
		UserID:  req.Input.UserID,
	}
	resp, err := l.sessionservice.List(ctx, ssReq)
	if err != nil {
		return fmt.Errorf("c.sessionservice.List() failed: %v", err)
	}

	sessions := []models.SessionData{}
	for _, sess := range resp.Sessions {
		sd := models.FromSession(sess)
		sessions = append(sessions, sd)
	}

	result := models.ListSessionResponse{
		Output: models.Sessions{
			Sessions: sessions,
		},
	}
	err = json.NewEncoder(rw).Encode(result)
	if err != nil {
		return fmt.Errorf("json.NewEncoder failed: %v", err)
	}
	return nil
}

// Name implements MethodHandler.
func (l *listSessionHandler) Name() string {
	return l.methodName
}

var _ MethodHandler = (*listSessionHandler)(nil)
