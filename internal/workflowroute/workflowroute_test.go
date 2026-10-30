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

package workflowroute_test

import (
	"errors"
	"testing"

	"google.golang.org/adk/v2/internal/workflowroute"
)

func TestRegisterAndValidate(t *testing.T) {
	sentinel := errors.New("bad outgoing route")
	type targetNode struct{}

	workflowroute.Register(func(node any, out workflowroute.OutgoingSummary) error {
		if _, ok := node.(targetNode); ok && len(out.Routes) > 0 {
			return sentinel
		}
		return nil
	})

	if err := workflowroute.Validate("other-node", workflowroute.OutgoingSummary{
		Routes: []workflowroute.Value{{Kind: workflowroute.KindString, Label: `"a"`}},
	}); err != nil {
		t.Fatalf("Validate(other-node) unexpected error: %v", err)
	}

	if err := workflowroute.Validate(targetNode{}, workflowroute.OutgoingSummary{}); err != nil {
		t.Fatalf("Validate(empty summary) unexpected error: %v", err)
	}

	err := workflowroute.Validate(targetNode{}, workflowroute.OutgoingSummary{
		Routes: []workflowroute.Value{{Kind: workflowroute.KindString, Label: `"a"`}},
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("Validate(targetNode) error = %v, want %v", err, sentinel)
	}
}
