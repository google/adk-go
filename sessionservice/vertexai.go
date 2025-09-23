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

package sessionservice

import (
	"context"
	"fmt"

	"google.golang.org/adk/session"
)

// VertexAiSessionService
type vertexAiService struct {
}

func (s *vertexAiService) Create(ctx context.Context, req *CreateRequest) (*CreateResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (s *vertexAiService) Get(ctx context.Context, req *GetRequest) (*GetResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (s *vertexAiService) List(ctx context.Context, req *ListRequest) (*ListResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (s *vertexAiService) Delete(ctx context.Context, req *DeleteRequest) error {
	return fmt.Errorf("not implemented")
}

func (s *vertexAiService) AppendEvent(ctx context.Context, session StoredSession, event *session.Event) error {
	return fmt.Errorf("not implemented")
}

var _ Service = (*vertexAiService)(nil)
