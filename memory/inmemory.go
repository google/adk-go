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
	"slices"
	"strings"
	"sync"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
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
	id             string
	content        *genai.Content
	author         string
	timestamp      time.Time
	customMetadata map[string]any

	// precomputed set of words in the content for simple keyword matching.
	words map[string]struct{}
}

// maxSearchResults caps how many memories SearchMemory returns, matching
// adk-python's _MAX_SEARCH_RESULTS.
const maxSearchResults = 10

// inMemoryService is an in-memory implementation of Service.
type inMemoryService struct {
	mu    sync.RWMutex
	store map[key]map[sessionID][]value
}

func (s *inMemoryService) AddSessionToMemory(ctx context.Context, curSession session.Session) error {
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
			id:             event.ID,
			content:        event.LLMResponse.Content,
			author:         event.Author,
			timestamp:      event.Timestamp,
			customMetadata: event.CustomMetadata,
			words:          words,
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

func (s *inMemoryService) SearchMemory(ctx context.Context, req *SearchRequest) (*SearchResponse, error) {
	queryWords := extractWords(req.Query)

	k := key{
		appName: req.AppName,
		userID:  req.UserID,
	}

	res := &SearchResponse{}

	// Hold the read lock for the whole scan. AddSessionToMemory writes into the
	// per-user session map under the write lock, so releasing the lock before
	// iterating would race with a concurrent write and can panic with
	// "concurrent map iteration and map write".
	s.mu.RLock()
	defer s.mu.RUnlock()

	values, ok := s.store[k]
	if !ok {
		return res, nil
	}

	// Almost any two sentences share a word, so keeping every event that
	// matches at least one query word keeps most of the store, and callers
	// such as preloadmemorytool render all of it into the prompt. Score each
	// event by how many distinct query words it matches and keep the best few.
	type scoredEntry struct {
		score int
		entry Entry
	}

	var scored []scoredEntry

	// Sessions are held in a map, whose iteration order is randomized, so
	// visit them in a fixed order to keep equally scored events in a stable
	// sequence across runs. adk-python gets this from its insertion-ordered
	// dict; a Go map has no insertion order to preserve, so sort the IDs.
	for _, sid := range slices.Sorted(maps.Keys(values)) {
		for _, e := range values[sid] {
			score := countMapsIntersect(e.words, queryWords)
			if score == 0 {
				continue
			}

			scored = append(scored, scoredEntry{
				score: score,
				entry: Entry{
					ID:             e.id,
					Content:        e.content,
					Author:         e.author,
					Timestamp:      e.timestamp,
					CustomMetadata: e.customMetadata,
				},
			})
		}
	}

	// Sorting only on the score keeps events that match equally in the order
	// they were visited.
	slices.SortStableFunc(scored, func(a, b scoredEntry) int {
		return b.score - a.score
	})

	if len(scored) > maxSearchResults {
		scored = scored[:maxSearchResults]
	}

	for _, s := range scored {
		res.Memories = append(res.Memories, s.entry)
	}

	return res, nil
}

// countMapsIntersect returns how many keys the two sets share.
func countMapsIntersect(m1, m2 map[string]struct{}) int {
	if len(m1) == 0 || len(m2) == 0 {
		return 0
	}

	// Iterate over the smaller map. The size of the intersection is the same
	// either way round.
	if len(m1) > len(m2) {
		m1, m2 = m2, m1
	}

	count := 0

	for k := range m1 {
		if _, ok := m2[k]; ok {
			count++
		}
	}

	return count
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
