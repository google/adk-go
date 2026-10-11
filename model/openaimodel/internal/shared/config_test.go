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

package shared

import (
	"testing"
	"time"

	"google.golang.org/genai"
)

// RequestTimeout must be safe on its own terms. Each endpoint package's
// applyGenerationConfig rejects a non-positive timeout before any request is
// built, so this guard is defense in depth — and it is tested directly rather
// than assumed unreachable, because a guard that only holds while callers keep
// the right order is not a guard.
func TestRequestTimeoutGuardsNonPositiveItself(t *testing.T) {
	for _, d := range []time.Duration{0, -time.Second} {
		timeout := d
		cfg := &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &timeout}}
		if got := RequestTimeout(cfg); got != 0 {
			t.Errorf("RequestTimeout(%v) = %v, want 0: a non-positive bound must not become a deadline", d, got)
		}
	}
	// The positive case still comes through, so the guard is not simply off.
	positive := time.Second
	cfg := &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &positive}}
	if got := RequestTimeout(cfg); got != positive {
		t.Errorf("RequestTimeout(%v) = %v, want it unchanged", positive, got)
	}
}
