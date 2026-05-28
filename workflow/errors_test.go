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
	"strings"
	"testing"
)

func TestSentinels_HaveWorkflowPrefix(t *testing.T) {
	sentinels := map[string]error{
		"ErrNodeFailed":              ErrNodeFailed,
		"ErrNodeInterrupted":         ErrNodeInterrupted,
		"ErrInvalidRunNodeContext":   ErrInvalidRunNodeContext,
		"ErrInvalidRunID":            ErrInvalidRunID,
		"ErrParallelHITLUnsupported": ErrParallelHITLUnsupported,
	}
	for name, err := range sentinels {
		t.Run(name, func(t *testing.T) {
			if !strings.HasPrefix(err.Error(), "workflow:") {
				t.Errorf("%s.Error() = %q; want prefix \"workflow:\"", name, err.Error())
			}
		})
	}
}

func TestNodeRunError_Unwrap(t *testing.T) {
	t.Run("returns Cause", func(t *testing.T) {
		nre := &NodeRunError{Cause: ErrNodeFailed}
		if got := nre.Unwrap(); got != ErrNodeFailed {
			t.Errorf("Unwrap() = %v, want %v", got, ErrNodeFailed)
		}
	})
	t.Run("nil receiver returns nil", func(t *testing.T) {
		var nre *NodeRunError
		if got := nre.Unwrap(); got != nil {
			t.Errorf("(*NodeRunError)(nil).Unwrap() = %v, want nil", got)
		}
	})
}

func TestNodeRunError_Error(t *testing.T) {
	t.Run("includes ChildPath and cause", func(t *testing.T) {
		nre := &NodeRunError{ChildPath: "code_workflow/fixer@2", Cause: ErrNodeFailed}
		s := nre.Error()
		if !strings.Contains(s, "code_workflow/fixer@2") {
			t.Errorf("Error() = %q, want to contain ChildPath", s)
		}
		if !strings.Contains(s, ErrNodeFailed.Error()) {
			t.Errorf("Error() = %q, want to contain cause message", s)
		}
	})

	t.Run("falls back to ChildName when ChildPath empty", func(t *testing.T) {
		nre := &NodeRunError{ChildName: "fixer", Cause: ErrNodeInterrupted}
		s := nre.Error()
		if !strings.Contains(s, "fixer") {
			t.Errorf("Error() = %q, want to contain ChildName when ChildPath unset", s)
		}
	})

	t.Run("nil cause does not panic", func(t *testing.T) {
		nre := &NodeRunError{ChildPath: "x/y@1"}
		_ = nre.Error()
	})

	t.Run("nil receiver does not panic", func(t *testing.T) {
		var nre *NodeRunError
		_ = nre.Error()
	})
}

// Compile-time: *NodeRunError implements error.
var _ error = (*NodeRunError)(nil)
