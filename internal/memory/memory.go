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
	"fmt"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
)

type Memory struct {
	Service   memory.Service
	SessionID string
	UserID    string
	AppName   string
}

func (a *Memory) AddSession(session session.Session) error {
	err := a.Service.AddSession(context.Background(), session)
	if err != nil {
		return fmt.Errorf("could not add session to memory service: %w", err)
	}
	return nil
}

func (a *Memory) Search(query string) ([]memory.Entry, error) {
	searchResponse, err := a.Service.Search(context.Background(), &memory.SearchRequest{
		AppName: a.AppName,
		UserID:  a.UserID,
		Query:   query,
	})
	if err != nil {
		return nil, err
	}
	return searchResponse.Memories, nil
}
