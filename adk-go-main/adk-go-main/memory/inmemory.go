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
	"maps"
	"strings"
	"sync"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/session"
)

// InMemoryService returns a new in-memory implementation of the memory service. Thread-safe.
func InMemoryService() Service {
	return &inMemoryService{
		store: make(map[key]map[sessionID][]value),
	}
}

type key struct {
	appName, userID string
}

type sessionID string

type value struct {
	content   *genai.Content
	author    string
	timestamp time.Time

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
		if event.LLMResponse.Content == nil {
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
			content:   event.LLMResponse.Content,
			author:    event.Author,
			timestamp: event.Timestamp,
			words:     words,
		})
	}

	k := key{
		appName: curSession.AppName(),
		userID:  curSession.UserID(),
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	v, ok := s.store[k]
	if !ok {
		v = map[sessionID][]value{}
		s.store[k] = v
	}

	sid := sessionID(curSession.ID())
	v[sid] = values
	return nil
}

func (s *inMemoryService) Search(ctx context.Context, req *SearchRequest) (*SearchResponse, error) {
	queryWords := extractWords(req.Query)

	k := key{
		appName: req.AppName,
		userID:  req.UserID,
	}

	s.mu.RLock()
	values, ok := s.store[k]
	s.mu.RUnlock()
	if !ok {
		return &SearchResponse{}, nil
	}

	res := &SearchResponse{}

	for _, events := range values {
		for _, e := range events {
			if checkMapsIntersect(e.words, queryWords) {
				res.Memories = append(res.Memories, Entry{
					Content:   e.content,
					Author:    e.author,
					Timestamp: e.timestamp,
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

func extractWords(text string) map[string]struct{} {
	res := make(map[string]struct{})

	for s := range strings.SplitSeq(text, " ") {
		if s == "" {
			continue
		}
		res[strings.ToLower(s)] = struct{}{}
	}

	return res
}
