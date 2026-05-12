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

package agent

import (
	"testing"
)

func TestReadonlyContext_NotAnInvocationContext(t *testing.T) {
	inv := NewInvocationContext(t.Context(), InvocationContextParams{})
	readonly := NewReadonlyContext(inv)

	if got, ok := readonly.(InvocationContext); ok {
		t.Errorf("ReadonlyContext(%+T) is unexpectedly an InvocationContext", got)
	}
}

func TestInvocationOf_RoundTrip(t *testing.T) {
	inv := NewInvocationContext(t.Context(), InvocationContextParams{
		Branch: "round-trip",
	})
	readonly := NewReadonlyContext(inv)

	got := InvocationOf(readonly)
	if got == nil {
		t.Fatal("InvocationOf returned nil for a NewReadonlyContext-produced ReadonlyContext")
	}
	if got.Branch() != "round-trip" {
		t.Errorf("InvocationOf returned wrong invocation: Branch() = %q, want %q", got.Branch(), "round-trip")
	}
}
