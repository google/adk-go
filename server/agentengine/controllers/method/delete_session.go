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

	"google.golang.org/adk/server/agentengine/internal/models"
	"google.golang.org/adk/session"
)

type deleteSessionHandler struct {
	sessionservice session.Service
}

func NewDeleteSessionHandler(sessionservice session.Service) *deleteSessionHandler {
	return &deleteSessionHandler{sessionservice: sessionservice}
}

func (c *deleteSessionHandler) Handle(ctx context.Context, rw http.ResponseWriter, payload []byte) error {
	var req models.DeleteSessionRequest

	err := json.Unmarshal(payload, &req)
	if err != nil {
		return fmt.Errorf("json.Unmarshal() failed: %v", err)
	}

	ssReq := &session.DeleteRequest{
		AppName:   "app",
		UserID:    req.Input.UserID,
		SessionID: req.Input.SessionID,
	}
	err = c.sessionservice.Delete(ctx, ssReq)
	if err != nil {
		return fmt.Errorf("c.sessionservice.Delete() failed: %v", err)
	}

	result := models.DeleteSessionResponse{
		Output: models.SessionData{},
	}
	err = json.NewEncoder(rw).Encode(result)
	if err != nil {
		return fmt.Errorf("json.NewEncoder failed: %v", err)
	}
	return nil
}

func (c *deleteSessionHandler) Metadata() Metadata {
	return Metadata{
		Description: "Delete a session",
		Parameters: Parameters{
			Properties: map[string]ParameterType{
				"user_id": {
					Type: "string",
				},
				"session_id": {
					Type: "string",
				},
			},
			Required: []string{"user_id", "session_id"},
			Type:     "object",
		},
		Name:    c.Name(),
		APIMode: "",
	}
}

func (c *deleteSessionHandler) Name() string {
	return "delete_session"
}

var _ MethodHandler = (*deleteSessionHandler)(nil)
