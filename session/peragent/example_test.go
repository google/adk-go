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

package peragent_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/peragent"
)

func ExampleNew() {
	// In an app, the agents root is the directory that holds the agent
	// folders; here it is an empty directory.
	root, err := os.MkdirTemp("", "agents")
	if err != nil {
		panic(err)
	}
	defer func() { _ = os.RemoveAll(root) }()

	svc, err := peragent.New(peragent.Config{AgentsRoot: root})
	if err != nil {
		panic(err)
	}
	defer func() { _ = svc.(io.Closer).Close() }()

	ctx := context.Background()
	if _, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "multi_agent.greeter", UserID: "user", SessionID: "s1",
		State: map[string]any{"topic": "weather"},
	}); err != nil {
		panic(err)
	}

	got, err := svc.Get(ctx, &session.GetRequest{AppName: "multi_agent.greeter", UserID: "user", SessionID: "s1"})
	if err != nil {
		panic(err)
	}
	topic, _ := got.Session.State().Get("topic")
	fmt.Println("session:", got.Session.ID(), "topic:", topic)

	// The app's sessions live in its own folder; a dot in the app name is a
	// path separator.
	if _, err := os.Stat(filepath.Join(root, "multi_agent", "greeter", ".adk", "session.db")); err == nil {
		fmt.Println("stored in multi_agent/greeter/.adk/session.db")
	}
	// Output:
	// session: s1 topic: weather
	// stored in multi_agent/greeter/.adk/session.db
}

// Only Create makes an app's database. The other methods report what an
// empty database would for an app that has none, and an app name that
// resolves outside the agents root is an error.
func ExampleNew_appWithoutDatabase() {
	root, err := os.MkdirTemp("", "agents")
	if err != nil {
		panic(err)
	}
	defer func() { _ = os.RemoveAll(root) }()

	svc, err := peragent.New(peragent.Config{AgentsRoot: root})
	if err != nil {
		panic(err)
	}
	defer func() { _ = svc.(io.Closer).Close() }()

	ctx := context.Background()
	_, err = svc.Get(ctx, &session.GetRequest{AppName: "weather", UserID: "user", SessionID: "s1"})
	fmt.Println("get:", errors.Is(err, session.ErrNotFound))
	list, err := svc.List(ctx, &session.ListRequest{AppName: "weather", UserID: "user"})
	if err != nil {
		panic(err)
	}
	fmt.Println("list:", len(list.Sessions))
	userState, err := session.GetUserState(ctx, svc, &session.GetUserStateRequest{AppName: "weather", UserID: "user"})
	if err != nil {
		panic(err)
	}
	fmt.Println("user state:", userState.State)
	if _, err := os.Stat(filepath.Join(root, "weather")); errors.Is(err, os.ErrNotExist) {
		fmt.Println("no weather folder")
	}

	_, err = svc.Create(ctx, &session.CreateRequest{AppName: "../outside", UserID: "user"})
	fmt.Println("outside the root:", errors.Is(err, peragent.ErrInvalidAppName))

	// Output:
	// get: true
	// list: 0
	// user state: map[]
	// no weather folder
	// outside the root: true
}
