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

// Package redis provides a Redis-backed implementation of [session.Service]
// and [session.UserStateReader].
//
// The package is its own module, google.golang.org/adk/session/redis, so a
// program that does not store sessions in Redis does not depend on a Redis
// client.
//
// A [Service] keeps each session in one JSON document and the "app:" and
// "user:" state in documents shared by every session of the app or user,
// under these keys, where the prefix is Config.KeyPrefix ("adk:session:" by
// default):
//
//	<prefix><app>:<user>:<session ID>   the session and its events
//	<prefix>user_state:<app>:<user>     "user:" state, without the prefix
//	<prefix>app_state:<app>             "app:" state, without the prefix
//
// This is the layout of adk-python's RedisSessionService. The parts are
// joined with ":" unescaped, so the keys of an app named "user_state" or
// "app_state" can collide with the state keys of another app.
//
// Every write sets the TTL of each key it writes (Config.TTL, 7 days by
// default), and reads never refresh it. App and user state therefore expire
// a TTL after a session last wrote to them, even while sessions that only
// write session-scoped state stay alive.
//
// A typical setup:
//
//	svc, err := redis.NewService(redis.Config{URI: "redis://localhost:6379/0"})
//	if err != nil {
//		return err
//	}
//	defer svc.Close()
//	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
package redis
