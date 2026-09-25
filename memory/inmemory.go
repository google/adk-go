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
	"cmp"
	"context"
	"maps"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

const maxSearchResults = 10

// InMemoryService returns a new in-memory implementation of the memory service. Thread-safe.
// Searches return at most ten entries, ranked by the number of distinct query
// words they match. Ties follow session insertion order, then event order.
func InMemoryService() Service {
	return &inMemoryService{
		store: make(map[key]*userMemories),
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

	// Cache an additional lowercase copy of the event text to avoid joining and
	// lowercasing content parts on each substring search.
	textLower string
}

type userMemories struct {
	sessions     map[sessionID][]value
	sessionOrder []sessionID
}

// inMemoryService is an in-memory implementation of Service.
type inMemoryService struct {
	mu    sync.RWMutex
	store map[key]*userMemories
}

func (s *inMemoryService) AddSessionToMemory(ctx context.Context, curSession session.Session) error {
	var values []value

	for event := range curSession.Events().All() {
		if event.LLMResponse.Content == nil {
			continue
		}

		words := make(map[string]struct{})
		var texts []string
		for _, part := range event.LLMResponse.Content.Parts {
			if part.Text == "" {
				continue
			}

			maps.Copy(words, extractWords(part.Text))
			texts = append(texts, part.Text)
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
			textLower:      strings.ToLower(strings.Join(texts, " ")),
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
		v = &userMemories{sessions: make(map[sessionID][]value)}
		s.store[k] = v
	}

	sid := sessionID(curSession.ID())
	// Replacing a session must preserve its position, as in Python's dict.
	if _, ok := v.sessions[sid]; !ok {
		v.sessionOrder = append(v.sessionOrder, sid)
	}
	v.sessions[sid] = values
	return nil
}

func (s *inMemoryService) SearchMemory(ctx context.Context, req *SearchRequest) (*SearchResponse, error) {
	// A true value allows substring matching for that non-ASCII query word.
	queryWords := make(map[string]bool)
	for word := range extractWords(req.Query) {
		queryWords[word] = strings.IndexFunc(word, func(r rune) bool { return r > unicode.MaxASCII }) >= 0
	}

	k := key{
		appName: req.AppName,
		userID:  req.UserID,
	}

	res := &SearchResponse{}
	if len(queryWords) == 0 {
		return res, nil
	}

	type scoredMemory struct {
		score int
		entry Entry
	}
	var scored []scoredMemory

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

	for _, sid := range values.sessionOrder {
		for _, e := range values.sessions[sid] {
			score := 0
			for word, allowSubstring := range queryWords {
				if _, ok := e.words[word]; ok || (allowSubstring && strings.Contains(e.textLower, word)) {
					score++
				}
			}
			if score > 0 {
				scored = append(scored, scoredMemory{
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
	}

	// Common words can match most of the store. Rank all matches before limiting
	// results so that later, more relevant events can still reach the prompt.
	slices.SortStableFunc(scored, func(a, b scoredMemory) int {
		return cmp.Compare(b.score, a.score)
	})
	for _, match := range scored[:min(len(scored), maxSearchResults)] {
		res.Memories = append(res.Memories, match.entry)
	}

	return res, nil
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
