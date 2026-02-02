// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
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

func TestWithContext(t *testing.T) {
	baseCtx := context.Background()
	params := InvocationContextParams{
		Branch: "test-branch",
	}
	inv := NewInvocationContext(baseCtx, params)

	key := "key"
	val := "val"
	newBaseCtx := context.WithValue(baseCtx, key, val)

	newInv := inv.WithContext(newBaseCtx)

	if newInv.Value(key) != val {
		t.Errorf("WithContext() did not update context")
	}
	ic, ok := newInv.(*InvocationContext)
	if !ok {
		t.Errorf("WithContext() did not return InvocationContext")
	}
	if diff := cmp.Diff(ic.params, params); diff != "" {
		t.Errorf("WithContext() params mismatch (-want +got):\n%s", diff)
	}
	if newInv.InvocationID() != inv.InvocationID() {
		t.Errorf("WithContext() changed InvocationID")
	}
}
