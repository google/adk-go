// Copyright 2025 Google LLC
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

package models

import (
	"fmt"

	"google.golang.org/genai"
)

type RunAgentRequest struct {
	AppName    string          `json:"app_name"`
	UserId     string          `json:"user_id"`
	SessionId  string          `json:"session_id"`
	NewMessage genai.Content   `json:"new_message"`
	Streaming  bool            `json:"streaming,omitempty"`
	StateDelta *map[string]any `json:"state_delta,omitempty"`
}

// Validate checks if the required fields are not zero-ed
func (req RunAgentRequest) Validate() error {
	elements := map[string]any{
		"app_name":    req.AppName,
		"user_id":     req.UserId,
		"session_id":  req.SessionId,
		"new_message": req.NewMessage,
	}
	for name, el := range elements {
		if isZero := IsZeroValue(el); isZero {
			return fmt.Errorf("%s is required", name)
		}
	}

	return nil
}
