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
	"errors"
	"fmt"
	"log"

	"google.golang.org/adk/v2/session"
)

// GetUserState reads a user's "user:" state without a session. A service
// that cannot read user state on its own reports errors.ErrUnsupported.
func ExampleGetUserState() {
	ctx := context.Background()
	svc := session.InMemoryService()

	if _, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "support",
		UserID:  "alice",
		State:   map[string]any{"user:language": "fr", "topic": "billing"},
	}); err != nil {
		log.Fatal(err)
	}

	for _, user := range []string{"alice", "bob"} {
		resp, err := session.GetUserState(ctx, svc, &session.GetUserStateRequest{AppName: "support", UserID: user})
		if err != nil {
			log.Fatal(err)
		}
		// Keys come back without the "user:" prefix; session-scoped keys
		// such as "topic" are not user state.
		fmt.Printf("%s: %v\n", user, resp.State)
	}

	_, err := session.GetUserState(ctx, sessionsOnly{svc}, &session.GetUserStateRequest{AppName: "support", UserID: "alice"})
	fmt.Println("unsupported:", errors.Is(err, errors.ErrUnsupported))

	// Output:
	// alice: map[language:fr]
	// bob: map[]
	// unsupported: true
}

// sessionsOnly embeds the session.Service interface, which promotes only the
// methods of session.Service, so it does not implement
// session.UserStateReader even though the service it wraps does.
type sessionsOnly struct {
	session.Service
}

// CloneSession makes a copy of a held session that a later AppendEvent can
// update without touching the original value.
func ExampleCloneSession() {
	ctx := context.Background()
	svc := session.InMemoryService()

	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "support", UserID: "alice", SessionID: "s1"})
	if err != nil {
		log.Fatal(err)
	}
	original := created.Session
	clone := session.CloneSession(original)

	event := session.NewEvent(ctx, "inv-1")
	event.Author = "user"
	event.Actions.StateDelta = map[string]any{"step": 1}
	if err := svc.AppendEvent(ctx, clone, event); err != nil {
		log.Fatal(err)
	}

	step, err := clone.State().Get("step")
	if err != nil {
		log.Fatal(err)
	}
	_, err = original.State().Get("step")
	fmt.Println("clone events:", clone.Events().Len(), "step:", step)
	fmt.Println("original events:", original.Events().Len(), "has step:", !errors.Is(err, session.ErrStateKeyNotExist))

	stored, err := svc.Get(ctx, &session.GetRequest{AppName: "support", UserID: "alice", SessionID: "s1"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("stored events:", stored.Session.Events().Len())

	// Output:
	// clone events: 1 step: 1
	// original events: 0 has step: false
	// stored events: 1
}

// The in-memory service reports a taken session ID with ErrAlreadyExists and
// a missing session with ErrNotFound, strips white space around session IDs,
// and treats deleting a missing session as success.
func ExampleInMemoryService_errors() {
	ctx := context.Background()
	svc := session.InMemoryService()

	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "support", UserID: "alice", SessionID: "s1"}); err != nil {
		log.Fatal(err)
	}

	// " s1 " is stripped to "s1", which is taken.
	_, err := svc.Create(ctx, &session.CreateRequest{AppName: "support", UserID: "alice", SessionID: " s1 "})
	fmt.Println("duplicate:", err, errors.Is(err, session.ErrAlreadyExists))

	// The user "" is a user like any other: alice's s1 is not hers.
	_, err = svc.Get(ctx, &session.GetRequest{AppName: "support", UserID: "", SessionID: "s1"})
	fmt.Println("other user:", errors.Is(err, session.ErrNotFound))

	_, err = svc.Get(ctx, &session.GetRequest{AppName: "support", UserID: "alice", SessionID: "s1", NumRecentEvents: -1})
	fmt.Println("negative count fails:", err != nil, "not found:", errors.Is(err, session.ErrNotFound))

	err = svc.Delete(ctx, &session.DeleteRequest{AppName: "support", UserID: "alice", SessionID: "missing"})
	fmt.Println("delete missing:", err)

	// AppendEvent identifies the session by app, user and ID, whatever value
	// carries them. Here they name a session deleted after the Get.
	got, err := svc.Get(ctx, &session.GetRequest{AppName: "support", UserID: "alice", SessionID: "s1"})
	if err != nil {
		log.Fatal(err)
	}
	if err := svc.Delete(ctx, &session.DeleteRequest{AppName: "support", UserID: "alice", SessionID: "s1"}); err != nil {
		log.Fatal(err)
	}
	err = svc.AppendEvent(ctx, got.Session, session.NewEvent(ctx, "inv-1"))
	fmt.Println("append to deleted:", errors.Is(err, session.ErrNotFound))

	// Output:
	// duplicate: session s1 already exists true
	// other user: true
	// negative count fails: true not found: false
	// delete missing: <nil>
	// append to deleted: true
}
