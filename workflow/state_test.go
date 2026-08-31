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

package workflow

import (
	"testing"
)

func TestRunState_EnsureNode(t *testing.T) {
	t.Run("creates inactive node when absent", func(t *testing.T) {
		s := NewRunState()
		ns := s.EnsureNode("A")
		if ns.Status != NodeInactive {
			t.Errorf("new node Status = %v, want NodeInactive", ns.Status)
		}
		if s.Nodes["A"] != ns {
			t.Error("EnsureNode should return the same pointer stored in Nodes map")
		}
	})

	t.Run("returns existing node without resetting it", func(t *testing.T) {
		s := NewRunState()
		first := s.EnsureNode("A")
		first.Status = NodeRunning
		first.Input = "in"
		first.TriggeredBy = "X"

		second := s.EnsureNode("A")
		if first != second {
			t.Error("EnsureNode should return the same pointer on repeat call")
		}
		if second.Status != NodeRunning || second.Input != "in" || second.TriggeredBy != "X" {
			t.Errorf("EnsureNode should not reset existing node, got %+v", *second)
		}
	})

	t.Run("allocates Nodes map on demand", func(t *testing.T) {
		s := &RunState{} // Nodes nil
		s.EnsureNode("A")
		if s.Nodes == nil {
			t.Error("EnsureNode should allocate Nodes map")
		}
	})
}
