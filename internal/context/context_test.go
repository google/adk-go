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

package context

import (
	"context"
	"testing"

	"google.golang.org/adk/agent"
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

func TestInvocationContextValueLookup(t *testing.T) {
	type ctxKey string
	baseKey := ctxKey("base")

	baseCtx := context.WithValue(t.Context(), baseKey, "base-value")
	inv := NewInvocationContext(baseCtx, InvocationContextParams{
		Values: map[string]any{
			"custom": "initial",
		},
	})

	if got := inv.Value("custom"); got != "initial" {
		t.Fatalf("Value(custom) = %v, want %q", got, "initial")
	}
	if got := inv.Value(baseKey); got != "base-value" {
		t.Fatalf("Value(baseKey) = %v, want %q", got, "base-value")
	}

	internal := inv.(*InvocationContext)
	internal.SetInvocationValue("custom", "updated")
	if got := inv.Value("custom"); got != "updated" {
		t.Fatalf("Value(custom) after update = %v, want %q", got, "updated")
	}

	internal.SetInvocationValue("custom", nil)
	if got := inv.Value("custom"); got != nil {
		t.Fatalf("Value(custom) after delete = %v, want nil", got)
	}
}
