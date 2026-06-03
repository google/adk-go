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
	AppName string `json:"appName"`

	UserId string `json:"userId"`

	SessionId string `json:"sessionId"`

	NewMessage genai.Content `json:"newMessage"`

	Streaming bool `json:"streaming,omitempty"`

	StateDelta *map[string]any `json:"stateDelta,omitempty"`
}

// AssertRunAgentRequestRequired checks if the required fields are not zero-ed
func (req RunAgentRequest) AssertRunAgentRequestRequired() error {
	elements := map[string]any{
		"appName":    req.AppName,
		"userId":     req.UserId,
		"sessionId":  req.SessionId,
		"newMessage": req.NewMessage,
	}
	for name, el := range elements {
		if isZero := IsZeroValue(el); isZero {
			return fmt.Errorf("%s is required", name)
		}
	}

	return nil
}

// blob represents a genai.blob sent by the client, explicitly mapping mime_type.
type blob struct {
	MIMEType string `json:"mime_type,omitempty"`
	Data     []byte `json:"data,omitempty"`
}

// LiveRequest represents the client request format for real-time interactions over WebSocket.
type LiveRequest struct {
	Content       *genai.Content       `json:"content,omitempty"`
	Blob          *blob                `json:"blob,omitempty"`
	ActivityStart *genai.ActivityStart `json:"activityStart,omitempty"`
	ActivityEnd   *genai.ActivityEnd   `json:"activityEnd,omitempty"`
	Close         bool                 `json:"close,omitempty"`
}
