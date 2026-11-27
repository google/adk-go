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

package agent_test

import (
	"testing"

	"google.golang.org/adk/v2/agent"
)

// TestRunConfigIsComparable pins that RunConfig stays comparable: callers
// compare it with == and use it as a map key. A field of map, slice or func
// type would make this file fail to compile.
func TestRunConfigIsComparable(t *testing.T) {
	labels := map[string]string{"team": "billing"}
	a := agent.RunConfig{MaxLLMCalls: 3, Labels: &labels}
	b := agent.RunConfig{MaxLLMCalls: 3, Labels: &labels}

	seen := map[agent.RunConfig]int{a: 1}

	if a != b {
		t.Errorf("RunConfig %+v != %+v, want equal", a, b)
	}
	if got := seen[b]; got != 1 {
		t.Errorf("map lookup by an equal RunConfig = %d, want 1", got)
	}
}
