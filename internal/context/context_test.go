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

package context

import (
	"strings"
	"testing"

	"github.com/google/uuid"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/platform"
)

func TestReadonlyContext(t *testing.T) {
	inv := NewInvocationContext(t.Context(), InvocationContextParams{})
	readonly := NewReadonlyContext(t.Context(), inv)

	if got, ok := readonly.(agent.InvocationContext); ok {
		t.Errorf("ReadonlyContext(%+T) is unexpectedly an InvocationContext", got)
	}
}

func TestCallbackContext(t *testing.T) {
	inv := NewInvocationContext(t.Context(), InvocationContextParams{})
	callback := NewCallbackContext(t.Context(), inv)

	if _, ok := callback.(agent.ReadonlyContext); !ok {
		t.Errorf("CallbackContext(%+T) is unexpectedly not a ReadonlyContext", callback)
	}
}

func TestNewInvocationContextGeneratesIDWithProvider(t *testing.T) {
	ctx := platform.WithUUIDProvider(t.Context(), func() string { return "fixed" })
	inv := NewInvocationContext(ctx, InvocationContextParams{})

	if got, want := inv.InvocationID(), "e-fixed"; got != want {
		t.Errorf("InvocationID() = %q, want %q", got, want)
	}
}

func TestNewInvocationContextRespectsExplicitID(t *testing.T) {
	// An explicit InvocationID must be used verbatim, leaving the provider unused.
	ctx := platform.WithUUIDProvider(t.Context(), func() string { return "fixed" })
	inv := NewInvocationContext(ctx, InvocationContextParams{InvocationID: "explicit"})

	if got, want := inv.InvocationID(), "explicit"; got != want {
		t.Errorf("InvocationID() = %q, want %q", got, want)
	}
}

func TestNewInvocationContextDefaultID(t *testing.T) {
	inv := NewInvocationContext(t.Context(), InvocationContextParams{})

	id := inv.InvocationID()
	rest, ok := strings.CutPrefix(id, "e-")
	if !ok {
		t.Fatalf("InvocationID() = %q, want \"e-\" prefix", id)
	}
	if _, err := uuid.Parse(rest); err != nil {
		t.Errorf("InvocationID() = %q, suffix not a valid UUID: %v", id, err)
	}
}

func TestInvocationContext_LiveSessionResumptionHandle(t *testing.T) {
	inv := NewInvocationContext(t.Context(), InvocationContextParams{})

	iCtx, ok := inv.(*InvocationContext)
	if !ok {
		t.Fatalf("NewInvocationContext did not return *InvocationContext")
	}

	if iCtx.LiveSessionResumptionHandle() != "" {
		t.Errorf("expected empty handle, got %q", iCtx.LiveSessionResumptionHandle())
	}

	iCtx.SetLiveSessionResumptionHandle("test-handle")
	if iCtx.LiveSessionResumptionHandle() != "test-handle" {
		t.Errorf("expected test-handle, got %q", iCtx.LiveSessionResumptionHandle())
	}
}
