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
	"context"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/uuid"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/platform"
)

func TestReadonlyContext(t *testing.T) {
	inv := NewInvocationContext(t.Context(), InvocationContextParams{})
	readonly := NewReadonlyContext(inv)

	if got, ok := readonly.(agent.InvocationContext); ok {
		t.Errorf("ReadonlyContext(%+T) is unexpectedly an InvocationContext", got)
	}
}

func TestCallbackContext(t *testing.T) {
	inv := NewInvocationContext(t.Context(), InvocationContextParams{})
	callback := NewCallbackContext(inv)

	if _, ok := callback.(agent.ReadonlyContext); !ok {
		t.Errorf("callback context (%+T) is unexpectedly not a ReadonlyContext", callback)
	}
}

type testKey struct{}

func TestWithContext(t *testing.T) {
	baseCtx := t.Context()
	inv := NewInvocationContext(baseCtx, InvocationContextParams{
		Branch: "test-branch",
	})

	key := testKey{}
	val := "val"

	ctx := context.WithValue(baseCtx, key, val)
	got := inv.WithICDelta(&agent.InvocationContextDelta{Context: &ctx})

	if got.Value(key) != val {
		t.Errorf("WithContext() did not update context")
	}
	if diff := cmp.Diff(inv, got, cmp.AllowUnexported(InvocationContext{}), cmpopts.IgnoreFields(InvocationContext{}, "Context")); diff != "" {
		t.Errorf("WithContext() mismatch (-want +got):\n%s", diff)
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
