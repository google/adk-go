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

// FINDING M2 — nodeInputToContent panics on a typed-nil *genai.Content.
//
// Bug: nodeInputToContent has `case nil` and `case *genai.Content`. A typed-nil
// pointer (*genai.Content)(nil) boxed in an `any` is NOT an untyped nil, so it
// does not match `case nil`; it falls into the `case *genai.Content` arm and
// evaluates `v.Parts`, dereferencing the nil pointer and panicking.
//
// Expected: a typed-nil *genai.Content is treated as empty input (no panic),
// e.g. returning (nil, nil) like the untyped-nil case.
//
// This test currently FAILS, demonstrating the bug.

package workflow

import (
	"testing"

	"google.golang.org/genai"
)

func TestVbugM2_NodeInputToContent_TypedNilContent(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("nodeInputToContent panicked on typed-nil *genai.Content: %v", r)
		}
	}()

	var typedNil *genai.Content // (*genai.Content)(nil)
	_, _ = nodeInputToContent(typedNil)
}
