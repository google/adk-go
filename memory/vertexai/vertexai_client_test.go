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

package vertexai

import "testing"

// TestCreateScope verifies the MemoryBank scope isolates by both app_name and
// user_id. app_name must be present so that applications sharing a MemoryBank
// for the same user cannot read each other's memories.
func TestCreateScope(t *testing.T) {
	scope := createScope("app-a", "user-1")

	if got, want := scope["app_name"], "app-a"; got != want {
		t.Errorf("scope[app_name] = %q, want %q", got, want)
	}
	if got, want := scope["user_id"], "user-1"; got != want {
		t.Errorf("scope[user_id] = %q, want %q", got, want)
	}
	if len(scope) != 2 {
		t.Errorf("scope has %d keys (%v), want exactly app_name and user_id", len(scope), scope)
	}
}

// TestCreateScopeDistinguishesApps verifies that two applications with the same
// user produce different scopes, which is what keeps their memories isolated.
func TestCreateScopeDistinguishesApps(t *testing.T) {
	a := createScope("app-a", "user-1")
	b := createScope("app-b", "user-1")

	if a["app_name"] == b["app_name"] {
		t.Errorf("scopes for different apps share app_name %q; memories would not be isolated", a["app_name"])
	}
}
