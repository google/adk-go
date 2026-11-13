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
	"fmt"
	"io"

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
