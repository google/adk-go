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

package memoryservice

import (
	"context"
	"maps"
	"regexp"
	"strings"
	"sync"
	"time"

	"google.golang.org/adk/session"
)

// Mem returns a new in-memory implementation of the memory service. Thread-safe.
func Mem() Service {
	return &inMemoryService{
		store: make(map[key]map[sessionID][]value),
	}
}

type key struct {
	appName, userID string
}

type sessionID = string

type value struct {
	event *session.Event

	// precomputed set of words in the content for simple keyword matching.
	words map[string]struct{}
}

// inMemoryService is an in-memory implementation of Service.
type inMemoryService struct {
	mu    sync.RWMutex
	store map[key]map[sessionID][]value
}

func (s *inMemoryService) AddSession(ctx context.Context, curSession session.Session) error {
	var values []value

	for event := range curSession.Events().All() {
		if event.LLMResponse == nil || event.LLMResponse.Content == nil {
			continue
		}

		words := make(map[string]struct{})
		for _, part := range event.LLMResponse.Content.Parts {
			if part.Text == "" {
				continue
			}

			maps.Copy(words, extractWords(part.Text))
		}

		if len(words) == 0 {
			continue
		}

		values = append(values, value{
			event: event,
			words: words,
		})
	}

	k := key{
		appName: curSession.ID().AppName,
		userID:  curSession.ID().UserID,
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	v, ok := s.store[k]
	if !ok {
		s.store[k] = map[sessionID][]value{
			curSession.ID().SessionID: values,
		}
		return nil
	}

	v[curSession.ID().SessionID] = values
	return nil
}

func (s *inMemoryService) SearchMemory(ctx context.Context, req *SearchMemoryRequest) (*SearchMemoryResponse, error) {
	queryWords := extractWords(req.Query)

	k := key{
		appName: req.AppName,
		userID:  req.UserID,
	}

	s.mu.RLock()
	values, ok := s.store[k]
	s.mu.RUnlock()
	if !ok {
		return &SearchMemoryResponse{}, nil
	}

	res := &SearchMemoryResponse{}

	for _, events := range values {
		for _, e := range events {
			if checkMapsIntersect(e.words, queryWords) {
				res.Memories = append(res.Memories, MemoryEntry{
					Content:   e.event.LLMResponse.Content,
					Author:    e.event.Author,
					Timestamp: e.event.Time.Format(time.RFC3339),
				})
			}
		}
	}

	return res, nil
}

func checkMapsIntersect(m1, m2 map[string]struct{}) bool {
	if len(m1) == 0 || len(m2) == 0 {
		return false
	}

	// Iterate over the smaller map.
	if len(m1) > len(m2) {
		m1, m2 = m2, m1
	}

	for k := range m1 {
		if _, ok := m2[k]; ok {
			return true
		}
	}

	return false
}

var wordRegex = regexp.MustCompile(`[A-Za-z]+`)

func extractWords(text string) map[string]struct{} {
	res := make(map[string]struct{})

	for _, word := range wordRegex.FindAllString(text, -1) {
		res[strings.ToLower(word)] = struct{}{}
	}

	return res
}
