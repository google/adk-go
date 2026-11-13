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

package memory

import (
	"context"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
)

type Memory struct {
	Service   memory.Service
	SessionID string
	UserID    string
	AppName   string
}

func (a *Memory) AddSessionToMemory(ctx context.Context, session session.Session) error {
	return a.Service.AddSessionToMemory(ctx, session)
}

// AddEventsToMemory passes req to [memory.AddEventsToMemory] with the wrapped
// Service, so a Service that implements [memory.EventAdder] is reached even
// though Memory hides it. It returns [memory.ErrAddEventsUnsupported] when the
// Service does not implement it. req carries its own scope; Memory's AppName,
// UserID and SessionID fields are not applied to it.
func (a *Memory) AddEventsToMemory(ctx context.Context, req *memory.AddEventsRequest) error {
	return memory.AddEventsToMemory(ctx, a.Service, req)
}

// AddMemory passes req to [memory.AddMemory] with the wrapped Service, so a
// Service that implements [memory.MemoryAdder] is reached even though Memory
// hides it. It returns [memory.ErrAddMemoryUnsupported] when the Service does
// not implement it. req carries its own scope; Memory's AppName and UserID
// fields are not applied to it.
func (a *Memory) AddMemory(ctx context.Context, req *memory.AddMemoryRequest) error {
	return memory.AddMemory(ctx, a.Service, req)
}

func (a *Memory) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	return a.Service.SearchMemory(ctx, &memory.SearchRequest{
		AppName: a.AppName,
		UserID:  a.UserID,
		Query:   query,
	})
}
