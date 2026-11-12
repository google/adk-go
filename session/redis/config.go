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

package redis

import (
	"time"

	goredis "github.com/redis/go-redis/v9"
)

const (
	// DefaultHost is the default Redis server hostname used when Config.Host is empty.
	DefaultHost = "localhost"
	// DefaultPort is the default Redis server port used when Config.Port is zero.
	DefaultPort = 6379
	// DefaultTTL is the default key expiration (7 days) used when Config.TTL is zero.
	DefaultTTL = 7 * 24 * time.Hour
	// DefaultKeyPrefix is the default prefix for all Redis keys written by Service
	// when Config.KeyPrefix is empty.
	DefaultKeyPrefix = "adk:session:"
)

// Config configures a Redis-backed [Service].
//
// Connection precedence matches adk-python's RedisSessionService:
//  1. If Client is non-nil, Service uses Client directly and ignores URI,
//     Host, Port, Password, SSL, and DB. Close does not close a caller-supplied
//     Client.
//  2. Otherwise, if URI is non-empty (for example
//     "redis://[:password@]host:port/db" or "rediss://..."), NewService parses
//     URI with [goredis.ParseURL] and ignores Host, Port, Password, SSL, and DB.
//  3. Otherwise, NewService connects to Host (default [DefaultHost]) and Port
//     (default [DefaultPort]) with Password, SSL (TLS with minimum version
//     TLS 1.2), and DB.
//
// The zero value Config{} is valid and uses DefaultHost, DefaultPort, DB 0,
// no SSL, a TTL of DefaultTTL (7 days), and DefaultKeyPrefix
// ("adk:session:").
type Config struct {
	// URI is a Redis connection URI (e.g. "redis://[:password@]host:port/db"
	// or "rediss://[:password@]host:port/db"). When non-empty and Client is
	// nil, URI takes precedence over Host, Port, Password, SSL, and DB.
	URI string
	// Host is the Redis server hostname. Defaults to [DefaultHost] ("localhost")
	// when empty.
	Host string
	// Port is the Redis server port. Defaults to [DefaultPort] (6379) when zero.
	// A negative Port is rejected by [NewService].
	Port int
	// Password is the optional password for Redis authentication.
	Password string
	// SSL enables TLS for Redis connections when building a client from
	// Host and Port.
	SSL bool
	// DB is the Redis database index. Defaults to 0. A negative DB is rejected
	// by [NewService].
	DB int
	// TTL is the time-to-live applied to session, user-state, and app-state
	// keys on write. Zero means [DefaultTTL] (7 days). A negative TTL means
	// keys are written without expiration. A positive TTL must be a whole
	// number of seconds.
	TTL time.Duration
	// KeyPrefix is prepended to every Redis key written by the service.
	// Defaults to [DefaultKeyPrefix] ("adk:session:") when empty.
	KeyPrefix string
	// Client is an optional pre-configured Redis client. When non-nil, it
	// takes precedence over URI and individual connection fields, and
	// [Service.Close] leaves it open.
	Client *goredis.Client
}
