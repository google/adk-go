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

package adka2a

import (
	"context"
)

type a2aMetadataCtxKey struct{}

// A2AMetadata contains metadata from A2A requests that can be propagated
// through the context to downstream services like MCP tool servers.
type A2AMetadata struct {
	// TaskID is the A2A task identifier.
	TaskID string
	// ContextID is the A2A context identifier.
	ContextID string
	// RequestMetadata contains arbitrary metadata from the A2A request.
	RequestMetadata map[string]any
	// MessageMetadata contains metadata from the A2A message.
	MessageMetadata map[string]any
}

// ContextWithA2AMetadata returns a new context with A2A metadata attached.
// This can be used in BeforeExecuteCallback to attach A2A request metadata
// to the context for downstream propagation to MCP tool calls.
func ContextWithA2AMetadata(ctx context.Context, meta *A2AMetadata) context.Context {
	return context.WithValue(ctx, a2aMetadataCtxKey{}, meta)
}

// A2AMetadataFromContext retrieves A2A metadata from the context.
// Returns nil if no metadata is present.
func A2AMetadataFromContext(ctx context.Context) *A2AMetadata {
	meta, ok := ctx.Value(a2aMetadataCtxKey{}).(*A2AMetadata)
	if !ok {
		return nil
	}
	return meta
}
