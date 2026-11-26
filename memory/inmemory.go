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
	"errors"
	"slices"
	"sync"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/memory/internal/keyword"
	"google.golang.org/adk/v2/session"
)

// InMemoryService returns a new in-memory implementation of the memory service. Thread-safe.
// Searches return at most ten entries, ranked by the number of distinct query
// words they match. Ties follow session insertion order, then event order.
//
// The returned Service also implements [EventAdder]. AddEventsToMemory appends
// to the bucket of req.SessionID and keeps the first stored event of each
// non-empty event ID in that bucket; AddSessionToMemory replaces the bucket of
// the session.
//
// Both add methods skip nil events and events whose content is nil or holds
// no non-nil part; such an event does not reserve its ID. Thought parts are
// left out of both matching and the returned content. An event with parts but
// no searchable text, such as a blob-only or thought-only event, is stored
// and reserves its ID, but no search returns it.
//
// A query and the stored text are split into words: runs of letters, numbers
// and underscores, taken after Unicode NFC normalization and lowercased with
// full Unicode case mapping. A word that mixes Latin and other scripts, such
// as the Python in 私はPythonを使う, also yields each single-script run as a
// word, in the query as in the stored text; ASCII digits and underscores
// count as Latin. Each such run is a query word of its own, so it adds to the
// ranking as any other query word does. An event matches a query word
// when it has that word. A whole query word with non-ASCII characters also
// matches anywhere inside the event's text; a run split out of a query word
// does not. A query with no words returns no entries.
//
// Returned entries carry the stored event's ID, Author, Timestamp and
// CustomMetadata. They share the event's parts and metadata with the service
// and with the events that were added, so callers must not modify them.
// AddSessionToMemory, AddEventsToMemory and SearchMemory return an error for a
// nil session or request.
func InMemoryService() Service {
	return &inMemoryService{
		store: make(map[key]*userMemories),
	}
}

var _ EventAdder = (*inMemoryService)(nil)

type key struct {
	appName, userID string
}

// sessionKey identifies a session bucket within a user scope. The default
// bucket, for events added with no session ID, has isDefault set and an
// empty id, so it never equals the key of a session, even one whose ID is "".
type sessionKey struct {
	isDefault bool
	id        string
}

type value struct {
	id             string
	content        *genai.Content
	author         string
	timestamp      time.Time
	customMetadata map[string]any

	document keyword.Document
}

type userMemories struct {
	sessions     map[sessionKey][]value
	sessionOrder []sessionKey
}

// inMemoryService is an in-memory implementation of Service.
type inMemoryService struct {
	mu    sync.RWMutex
	store map[key]*userMemories
}

func (s *inMemoryService) AddSessionToMemory(ctx context.Context, curSession session.Session) error {
	if curSession == nil {
		return errors.New("memory: session must not be nil")
	}

	var values []value
	if events := curSession.Events(); events != nil {
		for event := range events.All() {
			if v, ok := buildValue(event); ok {
				values = append(values, v)
			}
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	sk := sessionKey{id: curSession.ID()}
	v := s.ensureUserBucket(key{appName: curSession.AppName(), userID: curSession.UserID()}, sk)
	v.sessions[sk] = values
	return nil
}

func (s *inMemoryService) AddEventsToMemory(ctx context.Context, req *AddEventsRequest) error {
	if req == nil {
		return errors.New("memory: AddEventsRequest must not be nil")
	}

	var values []value
	for _, event := range req.Events {
		if v, ok := buildValue(event); ok {
			values = append(values, v)
		}
	}

	sk := sessionKey{id: req.SessionID}
	if req.SessionID == "" {
		sk = sessionKey{isDefault: true}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	v := s.ensureUserBucket(key{appName: req.AppName, userID: req.UserID}, sk)
	existing := v.sessions[sk]
	existingIDs := make(map[string]struct{}, len(existing))
	for _, e := range existing {
		if e.id != "" {
			existingIDs[e.id] = struct{}{}
		}
	}
	for _, val := range values {
		if val.id != "" {
			if _, ok := existingIDs[val.id]; ok {
				continue
			}
			existingIDs[val.id] = struct{}{}
		}
		existing = append(existing, val)
	}
	v.sessions[sk] = existing
	return nil
}

// ensureUserBucket returns the memories of k, creating them and the bucket sk
// if absent. A new bucket takes the next position in sessionOrder even when
// nothing is stored in it, as in Python's dict. The caller must hold s.mu for
// writing.
func (s *inMemoryService) ensureUserBucket(k key, sk sessionKey) *userMemories {
	v, ok := s.store[k]
	if !ok {
		v = &userMemories{sessions: make(map[sessionKey][]value)}
		s.store[k] = v
	}
	// Replacing a session must preserve its position, as in Python's dict.
	if _, ok := v.sessions[sk]; !ok {
		v.sessionOrder = append(v.sessionOrder, sk)
		v.sessions[sk] = nil
	}
	return v
}

// buildValue returns the stored form of event. It reports false for an event
// that has no content or no non-nil part, which is neither stored nor
// reserves its ID. An event with parts but no searchable words, such as a
// blob-only event, is stored so that its ID blocks later duplicates; search
// skips it.
func buildValue(event *session.Event) (value, bool) {
	if event == nil || event.Content == nil || !hasNonNilPart(event.Content.Parts) {
		return value{}, false
	}

	content, text := keyword.VisibleContent(event.Content)

	return value{
		id:             event.ID,
		content:        content,
		author:         event.Author,
		timestamp:      event.Timestamp,
		customMetadata: event.CustomMetadata,
		document:       keyword.NewDocument(text),
	}, true
}

func (s *inMemoryService) SearchMemory(ctx context.Context, req *SearchRequest) (*SearchResponse, error) {
	if req == nil {
		return nil, errors.New("memory: SearchRequest must not be nil")
	}

	query := keyword.NewQuery(req.Query)

	k := key{
		appName: req.AppName,
		userID:  req.UserID,
	}

	res := &SearchResponse{}
	if len(query) == 0 {
		return res, nil
	}

	type scoredMemory struct {
		score int
		entry Entry
	}
	var scored []scoredMemory

	// Hold the read lock for the whole scan. The add methods write into the
	// per-user session map under the write lock, so releasing the lock before
	// iterating would race with a concurrent write and can panic with
	// "concurrent map iteration and map write".
	s.mu.RLock()
	defer s.mu.RUnlock()

	values, ok := s.store[k]
	if !ok {
		return res, nil
	}

	for _, sk := range values.sessionOrder {
		for _, e := range values.sessions[sk] {
			if score := query.Score(e.document); score > 0 {
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
	for _, match := range scored[:min(len(scored), keyword.MaxResults)] {
		res.Memories = append(res.Memories, match.entry)
	}

	return res, nil
}

func hasNonNilPart(parts []*genai.Part) bool {
	return slices.ContainsFunc(parts, func(p *genai.Part) bool { return p != nil })
}
