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

package memory_test

import (
	"context"
	"errors"
	"fmt"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func textEvent(id, author, text string) *session.Event {
	role := genai.Role(genai.RoleModel)
	if author == "user" {
		role = genai.RoleUser
	}
	return &session.Event{
		ID:          id,
		Author:      author,
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(text, role)},
	}
}

// newSession creates a session in an in-memory session service and appends
// events to it.
func newSession(ctx context.Context, sessionID string, events ...*session.Event) session.Session {
	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: sessionID})
	if err != nil {
		log.Fatal(err)
	}
	for _, e := range events {
		if err := sessions.AppendEvent(ctx, created.Session, e); err != nil {
			log.Fatal(err)
		}
	}
	got, err := sessions.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: sessionID})
	if err != nil {
		log.Fatal(err)
	}
	return got.Session
}

func searchIDs(ctx context.Context, svc memory.Service, query string) []string {
	resp, err := svc.SearchMemory(ctx, &memory.SearchRequest{AppName: "app", UserID: "user", Query: query})
	if err != nil {
		log.Fatal(err)
	}
	ids := []string{}
	for _, m := range resp.Memories {
		ids = append(ids, m.ID)
	}
	return ids
}

// Add the events of one conversation turn to memory, without re-ingesting the
// whole session, and find them with a search.
func ExampleAddEventsToMemory() {
	ctx := context.Background()
	svc := memory.InMemoryService()

	latestTurn := []*session.Event{
		textEvent("turn-2-user", "user", "My favorite project is Project Alpha."),
		textEvent("turn-2-model", "assistant", "Noted: Project Alpha."),
	}
	err := memory.AddEventsToMemory(ctx, svc, &memory.AddEventsRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "session-1",
		Events:    latestTurn,
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, err := svc.SearchMemory(ctx, &memory.SearchRequest{
		AppName: "app",
		UserID:  "user",
		Query:   "Which project is my favorite?",
	})
	if err != nil {
		log.Fatal(err)
	}
	// The user's event matches four query words and the reply one, so the
	// user's event comes first.
	for _, m := range resp.Memories {
		fmt.Printf("%s (%s): %s\n", m.ID, m.Author, m.Content.Parts[0].Text)
	}
	// Output:
	// turn-2-user (user): My favorite project is Project Alpha.
	// turn-2-model (assistant): Noted: Project Alpha.
}

// loggingMemory wraps a memory service. Embedding the Service interface
// promotes only the methods of Service, so loggingMemory does not implement
// EventAdder, even though the service it wraps does.
type loggingMemory struct {
	memory.Service
}

// A service that does not implement EventAdder makes AddEventsToMemory return
// ErrAddEventsUnsupported, which also matches errors.ErrUnsupported. A caller
// can then add the whole session with AddSessionToMemory instead.
func ExampleAddEventsToMemory_unsupported() {
	ctx := context.Background()
	svc := loggingMemory{Service: memory.InMemoryService()}

	err := memory.AddEventsToMemory(ctx, svc, &memory.AddEventsRequest{
		AppName: "app",
		UserID:  "user",
		Events:  []*session.Event{textEvent("e1", "user", "I live in Lisbon.")},
	})
	fmt.Println(errors.Is(err, memory.ErrAddEventsUnsupported))
	fmt.Println(errors.Is(err, errors.ErrUnsupported))

	if errors.Is(err, memory.ErrAddEventsUnsupported) {
		sess := newSession(ctx, "session-1", textEvent("e1", "user", "I live in Lisbon."))
		if err := svc.AddSessionToMemory(ctx, sess); err != nil {
			log.Fatal(err)
		}
	}
	fmt.Println(searchIDs(ctx, svc, "Lisbon"))
	// Output:
	// true
	// true
	// [e1]
}

// The in-memory service keeps the first stored copy of an event ID within a
// session bucket: adding an event with that ID again does not change the
// memory.
func ExampleEventAdder() {
	ctx := context.Background()
	svc := memory.InMemoryService()

	adder, ok := svc.(memory.EventAdder)
	if !ok {
		log.Fatal("the service cannot add events")
	}
	add := func(events ...*session.Event) {
		err := adder.AddEventsToMemory(ctx, &memory.AddEventsRequest{AppName: "app", UserID: "user", Events: events})
		if err != nil {
			log.Fatal(err)
		}
	}
	add(textEvent("fact-1", "user", "The launch is on Monday."))
	add(
		textEvent("fact-1", "user", "The launch is on Friday."),
		textEvent("fact-2", "user", "The launch venue is Hall B."),
	)

	resp, err := svc.SearchMemory(ctx, &memory.SearchRequest{AppName: "app", UserID: "user", Query: "launch"})
	if err != nil {
		log.Fatal(err)
	}
	for _, m := range resp.Memories {
		fmt.Printf("%s: %s\n", m.ID, m.Content.Parts[0].Text)
	}
	// Output:
	// fact-1: The launch is on Monday.
	// fact-2: The launch venue is Hall B.
}

// Search matches words regardless of case and Unicode normalization, finds a
// Latin word written inside Japanese text, and matches a non-ASCII query word
// anywhere inside the text.
func ExampleInMemoryService() {
	ctx := context.Background()
	svc := memory.InMemoryService()

	sess := newSession(ctx, "session-1",
		textEvent("go", "user", "I use Go at work."),
		textEvent("ja", "user", "私はPythonでADKを使っています"),
		// "Cafe" followed by U+0301 COMBINING ACUTE ACCENT: the decomposed (NFD)
		// spelling of "Café".
		textEvent("cafe", "user", "We met at Cafe\u0301 Central."),
	)
	if err := svc.AddSessionToMemory(ctx, sess); err != nil {
		log.Fatal(err)
	}

	// "cafe" finds nothing: after normalization the stored word is "café", and
	// an ASCII query word matches whole words only. So does "thon", which is
	// only part of the word "python".
	for _, query := range []string{"GO", "python", "CAFÉ", "ADKを", "cafe", "thon"} {
		fmt.Printf("%s: %v\n", query, searchIDs(ctx, svc, query))
	}
	// Output:
	// GO: [go]
	// python: [ja]
	// CAFÉ: [cafe]
	// ADKを: [ja]
	// cafe: []
	// thon: []
}

// AddEventsToMemory appends to the bucket of its SessionID, and an empty
// SessionID names a default bucket shared by all such calls. AddSessionToMemory
// replaces the whole bucket of its session, so the events appended to that
// bucket are gone afterwards while the default bucket keeps its events.
func ExampleInMemoryService_sessionBuckets() {
	ctx := context.Background()
	svc := memory.InMemoryService()
	addEvents := func(sessionID string, events ...*session.Event) {
		err := memory.AddEventsToMemory(ctx, svc, &memory.AddEventsRequest{
			AppName: "app", UserID: "user", SessionID: sessionID, Events: events,
		})
		if err != nil {
			log.Fatal(err)
		}
	}

	sess := newSession(ctx, "session-1", textEvent("a1", "user", "alpha one"))
	if err := svc.AddSessionToMemory(ctx, sess); err != nil {
		log.Fatal(err)
	}
	addEvents("session-1", textEvent("a2", "user", "alpha two"))
	addEvents("", textEvent("a3", "user", "alpha three"))
	fmt.Println("after adding events:", searchIDs(ctx, svc, "alpha"))

	if err := svc.AddSessionToMemory(ctx, sess); err != nil {
		log.Fatal(err)
	}
	fmt.Println("after adding the session again:", searchIDs(ctx, svc, "alpha"))
	// Output:
	// after adding events: [a1 a2 a3]
	// after adding the session again: [a1 a3]
}
