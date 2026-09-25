// Copyright 2025 Google LLC
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

// Package memory defines the entities to interact with agent memory (long-term knowledge).
package memory

import (
	"context"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

// Service is a definition of the memory service.
//
// The service ingests sessions into memory so that it can be used for
// user queries across user-scoped sessions.
type Service interface {
	// AddSession adds a session to the memory service.
	//
	// A session can be added multiple times during its lifetime.
	AddSessionToMemory(ctx context.Context, s session.Session) error
	// SearchMemory returns memory entries relevant to the given query.
	// Empty slice is returned if there are no matches.
	SearchMemory(ctx context.Context, req *SearchRequest) (*SearchResponse, error)
}

// AddEventsToMemoryer is an optional capability of a [Service] that ingests an
// explicit list of events, as an extension discovered by a type assertion. It
// is deliberately kept off [Service] so that existing implementations — in
// particular custom backends — keep satisfying [Service] without having to
// implement a method they do not support; this mirrors adk-python, where
// add_events_to_memory is a concrete method that raises NotImplementedError
// rather than an abstract one.
//
// To use it, assert the capability where it is wanted:
//
//	if ingester, ok := svc.(memory.AddEventsToMemoryer); ok {
//		ingester.AddEventsToMemory(ctx, req)
//	}
//
// and fall back to [Service.AddSessionToMemory] when the assertion fails or the
// call returns the backend's unsupported error.
//
// AddEventsToMemory is intended for callers who want to persist only a subset of
// events (e.g. the latest turn) rather than re-ingesting the full session via
// [Service.AddSessionToMemory]. Implementations must treat Events as an
// incremental update and must not assume it represents the full session.
//
// The direction is not symmetric: AddSessionToMemory replaces the stored events
// for a session wholesale, so a later full-session call for the same session
// overwrites any events that were added incrementally and are absent from that
// call's session. Incremental and full-session ingestion of the same session are
// therefore not safely mixable.
type AddEventsToMemoryer interface {
	AddEventsToMemory(ctx context.Context, req *AddEventsToMemoryRequest) error
}

// AddEventsToMemoryRequest represents a request for
// [AddEventsToMemoryer.AddEventsToMemory].
type AddEventsToMemoryRequest struct {
	AppName string
	UserID  string
	Events  []*session.Event

	// Below are optional fields.

	// SessionID scopes the events to a session for the purpose of
	// deduplication: only the identity of the events a later call may reuse
	// is per-session. It is not an isolation boundary — SearchMemory returns
	// entries for an (app, user) pair irrespective of SessionID, so events
	// stored under one SessionID surface in searches for any other session of
	// the same user. Implementations may ignore it if not applicable.
	SessionID string
	// CustomMetadata is optional, implementation-defined metadata for
	// memory generation (e.g. TTL).
	CustomMetadata map[string]any
}

// SearchRequest represents a request for memory search.
type SearchRequest struct {
	Query   string
	UserID  string
	AppName string
}

// SearchResponse represents the response from a memory search.
type SearchResponse struct {
	Memories []Entry
}

// Entry represents a single memory entry.
type Entry struct {
	// ID is the unique identifier of the memory.
	ID string
	// Content contains the main content of the memory.
	Content *genai.Content
	// Author of the memory.
	Author string
	// Timestamp shows when the original content of this memory happened.
	// This string will be forwarded to LLM. Preferred format is ISO 8601 format.
	Timestamp time.Time
	// CustomMetadata contains optional custom metadata associated with the memory.
	CustomMetadata map[string]any
}
