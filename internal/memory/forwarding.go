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

package memory

import (
	"context"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
)

// NewForwardingService returns a memory.Service that sends every call to
// parent, so a nested runner reads and writes its caller's memory. Searches
// use parent's app and user scope, whatever the request names. A session
// added through it is stored under that session's own app, user and ID.
// parent must not be nil.
func NewForwardingService(parent agent.Memory) memory.Service {
	return &forwardingService{parent: parent}
}

type forwardingService struct {
	parent agent.Memory
}

func (s *forwardingService) AddSessionToMemory(ctx context.Context, sess session.Session) error {
	return s.parent.AddSessionToMemory(ctx, sess)
}

func (s *forwardingService) SearchMemory(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error) {
	return s.parent.SearchMemory(ctx, req.Query)
}
