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

package session

import (
	"context"
	"fmt"

	"google.golang.org/genai"
)

// VertexAiSessionService
type vertexAiService struct {
	client *genai.Client
	model  string
}

func newVertexAiSessionService(ctx context.Context, model string) (Service, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Backend: genai.BackendVertexAI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Vertex AI client: %w", err)
	}

	return &vertexAiService{client: client, model: model}, nil
}

func (s *vertexAiService) Create(ctx context.Context, req *CreateRequest) (*CreateResponse, error) {
	_, err := s.client.Chats.Create(ctx, s.model, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	c := &CreateResponse{
		Session: &session{
			id: id{
				appName:   req.AppName,
				userID:    req.UserID,
				sessionID: "test-id",
			},
		},
	}

	return c, nil
}

func (s *vertexAiService) Get(ctx context.Context, req *GetRequest) (*GetResponse, error) {
	return nil, fmt.Errorf("session Get function not implemented")
}

func (s *vertexAiService) List(ctx context.Context, req *ListRequest) (*ListResponse, error) {
	return nil, fmt.Errorf("session List function not implemented")
}

func (s *vertexAiService) Delete(ctx context.Context, req *DeleteRequest) error {
	return fmt.Errorf("session Delete function not implemented")
}

func (s *vertexAiService) AppendEvent(ctx context.Context, session Session, event *Event) error {
	return fmt.Errorf("session AppendEvent function not implemented")
}
