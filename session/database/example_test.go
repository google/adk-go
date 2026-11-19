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

package database_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/database"
)

func ExampleNewSessionServiceFromURL() {
	// "sqlite:////var/lib/app/sessions.db" would open a file; this in-memory
	// database lasts until Close. GORM's default logger prints to stdout, so
	// the example turns it off.
	svc, err := database.NewSessionServiceFromURL("sqlite:///:memory:", &gorm.Config{Logger: logger.Discard})
	if err != nil {
		panic(err)
	}
	defer func() { _ = svc.(io.Closer).Close() }()

	ctx := context.Background()
	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "app", UserID: "user", SessionID: "s1",
		State: map[string]any{"user:theme": "dark"},
	})
	if err != nil {
		panic(err)
	}
	if err := svc.AppendEvent(ctx, created.Session, &session.Event{
		Author:  "user",
		Actions: session.EventActions{StateDelta: map[string]any{"app:visits": 1}},
	}); err != nil {
		panic(err)
	}

	got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		panic(err)
	}
	theme, _ := got.Session.State().Get("user:theme")
	visits, _ := got.Session.State().Get("app:visits")
	fmt.Println("session:", got.Session.ID())
	fmt.Println("user:theme =", theme)
	fmt.Println("app:visits =", visits)
	fmt.Println("events:", got.Session.Events().Len())
	// Output:
	// session: s1
	// user:theme = dark
	// app:visits = 1
	// events: 1
}

// New takes its options as a Config. With AppendOrderTiebreak, events that
// share a timestamp come back in the order they were appended, as in
// adk-python's SqliteSessionService; without it they are ordered by event ID,
// as in adk-python's DatabaseSessionService.
func ExampleNew() {
	dir, err := os.MkdirTemp("", "sessions")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(dir) }()

	ctx := context.Background()
	at := time.Date(2026, time.November, 13, 9, 0, 0, 0, time.UTC)
	for _, tiebreak := range []bool{false, true} {
		svc, err := database.New(database.Config{
			Dialector:           sqlite.Open(filepath.Join(dir, fmt.Sprintf("tiebreak-%t.db", tiebreak))),
			AppendOrderTiebreak: tiebreak,
		})
		if err != nil {
			log.Fatal(err)
		}
		if err := database.AutoMigrate(svc); err != nil {
			log.Fatal(err)
		}
		created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s1"})
		if err != nil {
			log.Fatal(err)
		}
		// Two events with one timestamp, appended as "second", then "first".
		for _, id := range []string{"second", "first"} {
			if err := svc.AppendEvent(ctx, created.Session, &session.Event{ID: id, Author: "user", Timestamp: at}); err != nil {
				log.Fatal(err)
			}
		}
		got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"})
		if err != nil {
			log.Fatal(err)
		}
		var ids []string
		for e := range got.Session.Events().All() {
			ids = append(ids, e.ID)
		}
		fmt.Printf("AppendOrderTiebreak=%t: %v\n", tiebreak, ids)
		// The service opened the file, so Close closes it.
		if err := svc.(io.Closer).Close(); err != nil {
			log.Fatal(err)
		}
	}

	// Output:
	// AppendOrderTiebreak=false: [first second]
	// AppendOrderTiebreak=true: [second first]
}

// A session value loaded before another write was stored is stale:
// AppendEvent through it returns an error wrapping session.ErrStaleSession and
// changes neither storage nor the value.
func ExampleNew_staleSession() {
	dir, err := os.MkdirTemp("", "sessions")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(dir) }()

	svc, err := database.New(database.Config{Dialector: sqlite.Open(filepath.Join(dir, "sessions.db"))})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = svc.(io.Closer).Close() }()
	if err := database.AutoMigrate(svc); err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		log.Fatal(err)
	}
	// Two values of the same stored session, as two requests would hold.
	first := created.Session
	second := session.CloneSession(first)

	later := first.LastUpdateTime().Add(time.Second)
	if err := svc.AppendEvent(ctx, first, &session.Event{Author: "user", Timestamp: later}); err != nil {
		log.Fatal(err)
	}
	err = svc.AppendEvent(ctx, second, &session.Event{Author: "user", Timestamp: later.Add(time.Second)})
	fmt.Println("stale:", errors.Is(err, session.ErrStaleSession))
	fmt.Println("events in the stale value:", second.Events().Len())

	// A fresh Get gives a value that can append again.
	got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("stored events:", got.Session.Events().Len())

	// Output:
	// stale: true
	// events in the stale value: 0
	// stored events: 1
}

// A URL NewSessionServiceFromURL cannot open gives an error that matches
// ErrInvalidDatabaseURL and never shows the password or query values.
func ExampleNewSessionServiceFromURL_invalidURL() {
	for _, dbURL := range []string{
		"sessions.db",
		"postgresql://admin:secret@db.example.com/sessions?sslmode=require",
	} {
		_, err := database.NewSessionServiceFromURL(dbURL)
		fmt.Println(err)
		fmt.Println("  invalid:", errors.Is(err, database.ErrInvalidDatabaseURL),
			"unsupported:", errors.Is(err, errors.ErrUnsupported))
	}

	// Output:
	// Invalid database URL format or argument '<unparseable database URL>'.
	//   invalid: true unsupported: false
	// unsupported database URL scheme "postgresql" in 'postgresql://admin:***@db.example.com/sessions?sslmode=REDACTED': open a gorm.Dialector for that driver and pass it to database.New or database.NewSessionService
	//   invalid: true unsupported: true
}
