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

package toolinternal

import (
	"errors"
	"testing"

	"google.golang.org/adk/agent"
	contextinternal "google.golang.org/adk/internal/context"
	"google.golang.org/adk/session"
)

func TestToolContext(t *testing.T) {
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})
	toolCtx := NewToolContext(inv, "fn1", &session.EventActions{})

	if _, ok := toolCtx.(agent.ReadonlyContext); !ok {
		t.Errorf("ToolContext(%+T) is unexpectedly not a ReadonlyContext", toolCtx)
	}
	if _, ok := toolCtx.(agent.CallbackContext); !ok {
		t.Errorf("ToolContext(%+T) is unexpectedly not a CallbackContext", toolCtx)
	}
	if got, ok := toolCtx.(agent.InvocationContext); ok {
		t.Errorf("ToolContext(%+T) is unexpectedly an InvocationContext", got)
	}
}

func TestInternalArtifacts_NilSafe(t *testing.T) {
	// Create invocation context without artifact service
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{
		Artifacts: nil,
	})
	toolCtx := NewToolContext(inv, "fn1", &session.EventActions{})

	artifacts := toolCtx.Artifacts()
	// artifacts will be nil when service not configured

	tests := []struct {
		name string
		call func() (any, error)
	}{
		{
			name: "List",
			call: func() (any, error) { return artifacts.List(t.Context()) },
		},
		{
			name: "Load",
			call: func() (any, error) { return artifacts.Load(t.Context(), "test.txt") },
		},
		{
			name: "Save",
			call: func() (any, error) { return artifacts.Save(t.Context(), "test.txt", nil) },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name+" returns error", func(t *testing.T) {
			_, err := tt.call()
			if err == nil {
				t.Error("Expected an error, got nil")
				return
			}
			if !errors.Is(err, ErrArtifactServiceNotConfigured) {
				t.Errorf("Expected ErrArtifactServiceNotConfigured, got: %v", err)
			}
		})
	}
}
