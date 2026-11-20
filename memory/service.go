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
	"errors"
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
	// It is forwarded to the LLM formatted as ISO 8601.
	Timestamp time.Time
	// CustomMetadata contains optional custom metadata associated with the memory.
	CustomMetadata map[string]any
}

// ErrAddEventsUnsupported is returned by [AddEventsToMemory] when a [Service]
// does not implement [EventAdder]. It wraps [errors.ErrUnsupported]. A caller
// that receives it can ingest the full session with
// [Service.AddSessionToMemory] instead, as its message says.
//
// EXPERIMENTAL: ErrAddEventsUnsupported is experimental; its API and behavior may change or be removed in a later version.
var ErrAddEventsUnsupported error = &unsupportedError{
	msg: "memory: service does not support adding event deltas; call Service.AddSessionToMemory to ingest the full session",
}

// unsupportedError is an error that wraps [errors.ErrUnsupported] with a
// specific message.
type unsupportedError struct {
	msg string
}

func (e *unsupportedError) Error() string { return e.msg }

func (e *unsupportedError) Unwrap() error { return errors.ErrUnsupported }

// EventAdder is an optional interface implemented by a [Service] that can
// ingest an incremental delta of events without re-ingesting a full session.
//
// The in-memory service returned by [InMemoryService] implements EventAdder.
// Callers holding an arbitrary [Service] can either call the package-level
// [AddEventsToMemory] helper or test for support with a type assertion.
//
// A type that wraps a Service by embedding the Service interface does not
// implement EventAdder, even when the wrapped value does, because embedding
// promotes only the methods of Service. Such a wrapper declares
// AddEventsToMemory itself to forward the call.
//
// EXPERIMENTAL: EventAdder is experimental; its API and behavior may change or be removed in a later version.
type EventAdder interface {
	// AddEventsToMemory appends an incremental batch of events to the memory
	// service for the scope named by req.
	//
	// req must not be nil. Events already stored for the scope are kept: the
	// batch never replaces them. Implementations document how they treat
	// events that repeat an ID already stored.
	AddEventsToMemory(ctx context.Context, req *AddEventsRequest) error
}

// AddEventsRequest is the parameter for [EventAdder.AddEventsToMemory] and
// [AddEventsToMemory].
//
// EXPERIMENTAL: AddEventsRequest is experimental; its API and behavior may change or be removed in a later version.
type AddEventsRequest struct {
	// AppName is the application name for memory scope.
	AppName string
	// UserID is the user ID for memory scope.
	UserID string
	// SessionID is the optional session ID for memory partitioning.
	// In the in-memory service, an empty SessionID ("") places the events in a
	// shared default bucket per (AppName, UserID). The default bucket is
	// distinct from the bucket of every session, including a session whose ID
	// is "" added with AddSessionToMemory. Implementations for which session
	// partitioning does not apply may ignore SessionID.
	SessionID string
	// Events is the ordered slice of events to add to memory.
	Events []*session.Event
	// CustomMetadata holds optional, implementation-defined metadata for memory
	// generation. The in-memory service ignores CustomMetadata.
	CustomMetadata map[string]any
}

// AddEventsToMemory adds req.Events to svc if svc implements [EventAdder], or
// returns [ErrAddEventsUnsupported] if svc does not implement [EventAdder].
//
// svc and req must not be nil; AddEventsToMemory returns an error for either
// without calling svc. Otherwise it returns the error of
// [EventAdder.AddEventsToMemory].
//
// EXPERIMENTAL: AddEventsToMemory is experimental; its API and behavior may change or be removed in a later version.
func AddEventsToMemory(ctx context.Context, svc Service, req *AddEventsRequest) error {
	if svc == nil {
		return errors.New("memory: service must not be nil")
	}
	if req == nil {
		return errors.New("memory: AddEventsRequest must not be nil")
	}
	adder, ok := svc.(EventAdder)
	if !ok {
		return ErrAddEventsUnsupported
	}
	return adder.AddEventsToMemory(ctx, req)
}

// ErrAddMemoryUnsupported is returned by [AddMemory] when a [Service] does not
// implement [MemoryAdder]. It wraps [errors.ErrUnsupported]. A caller that
// receives it can ingest events with [AddEventsToMemory] or the full session
// with [Service.AddSessionToMemory] instead, as its message says.
var ErrAddMemoryUnsupported error = &unsupportedError{
	msg: "memory: service does not support direct memory writes; call memory.AddEventsToMemory or Service.AddSessionToMemory instead",
}

// MemoryAdder is an optional interface implemented by a [Service] that accepts
// memory entries written directly, in addition to the memories it generates
// from sessions and events.
//
// The in-memory service returned by [InMemoryService] does not implement
// MemoryAdder. Callers holding an arbitrary [Service] can either call the
// package-level [AddMemory] helper or test for support with a type assertion.
//
// As with [EventAdder], a type that wraps a Service by embedding the Service
// interface does not implement MemoryAdder, even when the wrapped value does.
// Such a wrapper declares AddMemory itself to forward the call.
type MemoryAdder interface {
	// AddMemory stores req.Memories for the scope named by req.
	//
	// req must not be nil. Implementations document how they treat entries
	// that repeat an ID already stored, and which CustomMetadata keys they
	// read.
	AddMemory(ctx context.Context, req *AddMemoryRequest) error
}

// AddMemoryRequest is the parameter for [MemoryAdder.AddMemory] and
// [AddMemory].
type AddMemoryRequest struct {
	// AppName is the application name for memory scope.
	AppName string
	// UserID is the user ID for memory scope.
	UserID string
	// Memories are the entries to store, in order.
	Memories []*Entry
	// CustomMetadata holds optional, implementation-defined metadata for the
	// write.
	CustomMetadata map[string]any
}

// AddMemory adds req.Memories to svc if svc implements [MemoryAdder], or
// returns [ErrAddMemoryUnsupported] if svc does not implement [MemoryAdder].
//
// svc and req must not be nil; AddMemory returns an error for either without
// calling svc. Otherwise it passes req to [MemoryAdder.AddMemory] unchanged
// and returns its error.
func AddMemory(ctx context.Context, svc Service, req *AddMemoryRequest) error {
	if svc == nil {
		return errors.New("memory: service must not be nil")
	}
	if req == nil {
		return errors.New("memory: AddMemoryRequest must not be nil")
	}
	adder, ok := svc.(MemoryAdder)
	if !ok {
		return ErrAddMemoryUnsupported
	}
	return adder.AddMemory(ctx, req)
}
