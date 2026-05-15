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

	"google.golang.org/adk/agent"
)

func TestNodeContext_ResumedInput(t *testing.T) {
	parent := newMockCtx(t)

	t.Run("nil resumeInputs returns (nil, false)", func(t *testing.T) {
		c := newNodeContext(parent, nil)
		v, ok := c.ResumedInput("any_id")
		if v != nil || ok {
			t.Errorf("ResumedInput() = (%v, %v), want (nil, false)", v, ok)
		}
	})

	t.Run("populated resumeInputs returns matched payload", func(t *testing.T) {
		c := newNodeContext(parent, map[string]any{
			"approval": "yes",
			"comment":  "looks good",
		})
		if v, ok := c.ResumedInput("approval"); !ok || v != "yes" {
			t.Errorf("ResumedInput(\"approval\") = (%v, %v), want (\"yes\", true)", v, ok)
		}
	})
}

// Compile-time assertion: *nodeContext satisfies agent.InvocationContext.
var _ agent.InvocationContext = (*nodeContext)(nil)
