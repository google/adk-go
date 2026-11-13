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

package redis_test

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"

	"github.com/alicebob/miniredis/v2"
	goredis "github.com/redis/go-redis/v9"

	"google.golang.org/adk/session/redis"

	"google.golang.org/adk/v2/session"
)

func ExampleNewService() {
	// An in-process server stands in for Redis here.
	mr, err := miniredis.Run()
	if err != nil {
		panic(err)
	}
	defer mr.Close()

	svc, err := redis.NewService(redis.Config{URI: "redis://" + mr.Addr() + "/0"})
	if err != nil {
		panic(err)
	}
	defer func() { _ = svc.Close() }()

	ctx := context.Background()
	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "app", UserID: "user", SessionID: "s1",
		State: map[string]any{"user:theme": "dark", "draft": "hello"},
	})
	if err != nil {
		panic(err)
	}
	if err := svc.AppendEvent(ctx, created.Session, &session.Event{
		Actions: session.EventActions{StateDelta: map[string]any{"app:visits": 1}},
	}); err != nil {
		panic(err)
	}

	got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		panic(err)
	}
	for _, k := range slices.Sorted(maps.Keys(maps.Collect(got.Session.State().All()))) {
		v, _ := got.Session.State().Get(k)
		fmt.Printf("%s = %v\n", k, v)
	}
	fmt.Println("events:", got.Session.Events().Len())
	// Output:
	// app:visits = 1
	// draft = hello
	// user:theme = dark
	// events: 1
}

// A service can share a client the caller already has. Close then leaves that
// client open, and a taken session ID is reported with
// session.ErrAlreadyExists.
func ExampleConfig_client() {
	mr, err := miniredis.Run()
	if err != nil {
		panic(err)
	}
	defer mr.Close()

	client := goredis.NewClient(&goredis.Options{Addr: mr.Addr()})
	defer func() { _ = client.Close() }()

	svc, err := redis.NewService(redis.Config{Client: client, KeyPrefix: "support:"})
	if err != nil {
		panic(err)
	}

	ctx := context.Background()
	req := &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s1"}
	if _, err := svc.Create(ctx, req); err != nil {
		panic(err)
	}
	_, err = svc.Create(ctx, req)
	fmt.Println(err, errors.Is(err, session.ErrAlreadyExists))
	fmt.Println("key:", mr.Keys())

	if err := svc.Close(); err != nil {
		panic(err)
	}
	fmt.Println("client after Close:", client.Ping(ctx).Val())

	// Output:
	// session s1 already exists true
	// key: [support:app:user:s1]
	// client after Close: PONG
}
