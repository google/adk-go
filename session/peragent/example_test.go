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
