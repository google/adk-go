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

package main

import (
	"strings"
	"testing"
)

func TestRunRecoversFromToolError(t *testing.T) {
	var out strings.Builder
	if err := run(t.Context(), &out); err != nil {
		t.Fatalf("run() error = %v", err)
	}

	const want = "Tool inventory_lookup failed: inventory service unavailable for \"sold-out\"\n" +
		"Agent: Inventory is temporarily unavailable; please try again.\n"
	if got := out.String(); got != want {
		t.Errorf("run() output = %q, want %q", got, want)
	}
}
