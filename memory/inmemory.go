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
	"maps"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"

	"golang.org/x/text/cases"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/norm"
	"golang.org/x/text/unicode/runenames"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

const maxSearchResults = 10

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
// full Unicode case mapping. An event matches a query word when it has that
// word. A query word with non-ASCII characters also matches anywhere inside
// the event's text, and a stored word that mixes Latin and other scripts,
// such as the Python in 私はPythonを使う, also yields each single-script run
// as a word. A query with no words returns no entries.
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

	// precomputed set of words in the content for simple keyword matching.
	words map[string]struct{}

	// Cache an additional lowercase copy of the event text to avoid joining and
	// lowercasing content parts on each substring search.
	textLower string
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

	// Thought parts are the model's reasoning, not conversation, so they are
	// neither searched nor returned. adk-python v2.12.0 does both; this keeps
	// internal reasoning out of memory results.
	var texts []string
	var visibleParts []*genai.Part
	for _, part := range event.Content.Parts {
		if part == nil || part.Thought {
			continue
		}
		visibleParts = append(visibleParts, part)
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}

	content := *event.Content
	content.Parts = visibleParts
	text := strings.Join(texts, " ")

	return value{
		id:             event.ID,
		content:        &content,
		author:         event.Author,
		timestamp:      event.Timestamp,
		customMetadata: event.CustomMetadata,
		words:          extractSearchableWords(text),
		textLower:      cases.Lower(language.Und).String(norm.NFC.String(text)),
	}, true
}

func (s *inMemoryService) SearchMemory(ctx context.Context, req *SearchRequest) (*SearchResponse, error) {
	if req == nil {
		return nil, errors.New("memory: SearchRequest must not be nil")
	}

	// A true value allows substring matching for that non-ASCII query word.
	queryWords := make(map[string]bool)
	for word := range extractWordsLower(req.Query) {
		queryWords[word] = hasNonASCII(word)
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
			if len(e.words) == 0 {
				continue
			}
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

// extractWordsLower returns the lowercased \w+ tokens of the NFC form of text,
// as adk-python's _extract_words_lower does. Combining marks are not word
// characters, so after NFC a mark that composes nowhere separates words.
func extractWordsLower(text string) map[string]struct{} {
	// A Caser may be stateful, so each call builds its own.
	lower := cases.Lower(language.Und)
	res := make(map[string]struct{})
	for word := range strings.FieldsFuncSeq(norm.NFC.String(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r) && r != '_'
	}) {
		res[lower.String(word)] = struct{}{}
	}
	return res
}

// extractSearchableWords returns the words an event can be matched on: the
// tokens of extractWordsLower plus, for each non-ASCII token, its runs of
// Latin and of non-Latin characters. Japanese and Chinese are written without
// spaces, so in 私はPythonを使う the run python becomes a word of its own.
func extractSearchableWords(text string) map[string]struct{} {
	words := extractWordsLower(text)
	for _, word := range slices.Collect(maps.Keys(words)) {
		if !hasNonASCII(word) {
			continue
		}
		start, runLatin := 0, false
		for i, r := range word {
			latin := isLatin(r)
			if i > 0 && latin != runLatin {
				words[word[start:i]] = struct{}{}
				start = i
			}
			runLatin = latin
		}
		words[word[start:]] = struct{}{}
	}
	return words
}

// isLatin reports whether r belongs to a Latin run, as adk-python's _is_latin
// does: an ASCII word character, or a character whose Unicode name starts
// with LATIN. The name rule leaves out ª and º, which unicode.Latin includes.
func isLatin(r rune) bool {
	if r <= unicode.MaxASCII {
		return unicode.IsLetter(r) || unicode.IsNumber(r) || r == '_'
	}
	return strings.HasPrefix(runenames.Name(r), "LATIN")
}

func hasNonNilPart(parts []*genai.Part) bool {
	return slices.ContainsFunc(parts, func(p *genai.Part) bool { return p != nil })
}

func hasNonASCII(s string) bool {
	return strings.IndexFunc(s, func(r rune) bool { return r > unicode.MaxASCII }) >= 0
}
