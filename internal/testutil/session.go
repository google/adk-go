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

package testutil

import (
	"context"
	"slices"
	"sync"

	"google.golang.org/adk/v2/session"
)

// RecordingSessionService wraps a session.Service and records every Get
// request it serves.
type RecordingSessionService struct {
	session.Service
	mu   sync.Mutex
	gets []session.GetRequest
}

// Get records req and delegates to the wrapped session.Service.
func (s *RecordingSessionService) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	s.mu.Lock()
	s.gets = append(s.gets, *req)
	s.mu.Unlock()
	return s.Service.Get(ctx, req)
}

// Gets returns a copy of the recorded Get requests in call order.
func (s *RecordingSessionService) Gets() []session.GetRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.gets)
}
