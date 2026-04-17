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

type getSessionHandler struct {
	sessionservice session.Service
}

func NewGetSessionHandler(sessionservice session.Service) *getSessionHandler {
	return &getSessionHandler{sessionservice: sessionservice}
}

func (c *getSessionHandler) Handle(ctx context.Context, rw http.ResponseWriter, payload []byte) error {
	var req models.GetSessionRequest

	err := json.Unmarshal(payload, &req)
	if err != nil {
		return fmt.Errorf("json.Unmarshal() failed: %v", err)
	}

	ssReq := &session.GetRequest{
		AppName:   "app",
		UserID:    req.Input.UserID,
		SessionID: req.Input.SessionID,
	}
	resp, err := c.sessionservice.Get(ctx, ssReq)
	if err != nil {
		return fmt.Errorf("c.sessionservice.Get() failed: %v", err)
	}

	stateMap := make(map[string]any)
	for k, v := range resp.Session.State().All() {
		stateMap[k] = v
	}

	result := models.GetSessionResponse{
		Output: models.SessionData{
			UserID:         resp.Session.UserID(),
			LastUpdateTime: float64(resp.Session.LastUpdateTime().UnixNano()) / 1e9, // converts nanosec to sec
			AppName:        resp.Session.AppName(),
			Id:             resp.Session.ID(),
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

func (c *getSessionHandler) Metadata() Metadata {
	return Metadata{
		Description: "Get a session",
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

func (c *getSessionHandler) Name() string {
	return "get_session"
}

var _ MethodHandler = (*getSessionHandler)(nil)
