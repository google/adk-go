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

	"google.golang.org/adk/v2/session"
)

func TestEventActionsFrom_DropsFrameworkFields(t *testing.T) {
	in := &session.EventActions{
		StateDelta: map[string]any{"k": "v"},
		Compaction: &session.EventCompaction{},
		AgentState: map[string]any{"current_sub_agent": "after_approval"},
		EndOfAgent: true,
	}

	got := eventActionsFrom(in)

	if got.Compaction != nil || got.AgentState != nil || got.EndOfAgent {
		t.Errorf("eventActionsFrom kept framework fields: Compaction=%v AgentState=%v EndOfAgent=%v", got.Compaction, got.AgentState, got.EndOfAgent)
	}
	if got.StateDelta["k"] != "v" {
		t.Errorf("eventActionsFrom dropped StateDelta: %v", got.StateDelta)
	}
}
