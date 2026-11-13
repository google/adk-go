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

package session_test

import (
	"context"
	"fmt"
	"log"

	"google.golang.org/adk/v2/session"
)

// A caller that needs a session's state but not its history sets OmitEvents.
func ExampleGetRequest_omitEvents() {
	ctx := context.Background()
	svc := session.InMemoryService()

	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "support",
		UserID:    "alice",
		SessionID: "s1",
		State:     map[string]any{"topic": "billing"},
	})
	if err != nil {
		log.Fatal(err)
	}
	if err := svc.AppendEvent(ctx, created.Session, session.NewEvent(ctx, "inv-1")); err != nil {
		log.Fatal(err)
	}

	got, err := svc.Get(ctx, &session.GetRequest{
		AppName:    "support",
		UserID:     "alice",
		SessionID:  "s1",
		OmitEvents: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	topic, err := got.Session.State().Get("topic")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("events:", got.Session.Events().Len())
	fmt.Println("topic:", topic)

	// Output:
	// events: 0
	// topic: billing
}

// An empty UserID lists every user's sessions; FilterEmptyUserID makes it name
// the user "" instead.
func ExampleListRequest_filterEmptyUserID() {
	ctx := context.Background()
	svc := session.InMemoryService()

	for _, req := range []*session.CreateRequest{
		{AppName: "support", UserID: "", SessionID: "s-anon"},
		{AppName: "support", UserID: "alice", SessionID: "s-alice"},
	} {
		if _, err := svc.Create(ctx, req); err != nil {
			log.Fatal(err)
		}
	}

	for _, filter := range []bool{false, true} {
		list, err := svc.List(ctx, &session.ListRequest{
			AppName:           "support",
			UserID:            "",
			FilterEmptyUserID: filter,
		})
		if err != nil {
			log.Fatal(err)
		}
		var ids []string
		for _, s := range list.Sessions {
			ids = append(ids, s.ID())
		}
		fmt.Printf("FilterEmptyUserID=%t: %v\n", filter, ids)
	}

	// Output:
	// FilterEmptyUserID=false: [s-anon s-alice]
	// FilterEmptyUserID=true: [s-anon]
}
