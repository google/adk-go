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

type createSessionHandler struct {
	sessionservice session.Service
}

func NewCreateSessionHandler(sessionservice session.Service) *createSessionHandler {
	return &createSessionHandler{sessionservice: sessionservice}
}

func (c *createSessionHandler) Handle(ctx context.Context, rw http.ResponseWriter, payload []byte) error {
	var req models.CreateSessionRequest

	err := json.Unmarshal(payload, &req)
	if err != nil {
		return fmt.Errorf("json.Unmarshal() failed: %v", err)
	}

	ssReq := &session.CreateRequest{
		AppName: "app",
		UserID:  req.Input.UserID,
	}
	resp, err := c.sessionservice.Create(ctx, ssReq)
	if err != nil {
		return fmt.Errorf("c.sessionservice.Create() failed: %v", err)
	}

	stateMap := make(map[string]any)
	for k, v := range resp.Session.State().All() {
		stateMap[k] = v
	}

	result := models.CreateSessionResponse{
		Output: models.SessionData{
			UserID:         resp.Session.UserID(),
			LastUpdateTime: float64(resp.Session.LastUpdateTime().UnixNano()) / 1e9, // converts nanosec to sec
			AppName:        resp.Session.AppName(),
			ID:             resp.Session.ID(),
			State:          stateMap,
			Events:         resp.Session.Events(),
		},
	}
	err = json.NewEncoder(rw).Encode(result)
	if err != nil {
		return fmt.Errorf("json.NewEncoder failed: %v", err)
	}
	return nil
}

func (c *createSessionHandler) Metadata() Metadata {
	return Metadata{
		Description: "Create a new session",
		Parameters: Parameters{
			Properties: map[string]ParameterType{
				"user_id": {
					Type: "string",
				},
			},
			Required: []string{"user_id"},
			Type:     "object",
		},
		Name:    c.Name(),
		APIMode: "",
	}
}

func (c *createSessionHandler) Name() string {
	return "create_session"
}

var _ MethodHandler = (*createSessionHandler)(nil)
