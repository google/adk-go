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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"github.com/sjzsdu/adk-go/agent"
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
		t.Errorf("CallbackContext(%+T) is unexpectedly not a ReadonlyContext", callback)
	}
	if got, ok := callback.(agent.InvocationContext); ok {
		t.Errorf("CallbackContext(%+T) is unexpectedly an InvocationContext", got)
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
	got := inv.WithContext(context.WithValue(baseCtx, key, val))

	if got.Value(key) != val {
		t.Errorf("WithContext() did not update context")
	}
	if diff := cmp.Diff(inv, got, cmp.AllowUnexported(InvocationContext{}), cmpopts.IgnoreFields(InvocationContext{}, "Context")); diff != "" {
		t.Errorf("WithContext() mismatch (-want +got):\n%s", diff)
	}
}
