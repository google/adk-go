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
	"errors"
	"fmt"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

type identityBodyError struct{}

func (*identityBodyError) Error() string { return "scripted body failure" }

func TestDynamicNode_ErrorIdentity(t *testing.T) {
	sentinel := errors.New("scripted sentinel")
	for _, tc := range []struct {
		name string
		err  error
	}{
		{name: "sentinel", err: sentinel},
		{name: "typed", err: &identityBodyError{}},
		{name: "user_wrapper", err: fmt.Errorf("scripted wrapper: %w", sentinel)},
		{name: "node_run_error", err: &NodeRunError{ChildName: "child", Cause: sentinel}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			n := NewDynamicNode("body", func(agent.Context, any, func(*session.Event) error) (any, error) {
				return nil, tc.err
			}, NodeConfig{})
			var got error
			for _, err := range n.Run(agent.Promote(newMockCtx(t)), nil) {
				if err != nil {
					got = err
				}
			}
			if got != tc.err {
				t.Fatalf("body error identity changed (got %T, want %T)", got, tc.err)
			}
		})
	}
}

func TestRunNode_DynamicChildErrorIdentity(t *testing.T) {
	failure := &identityBodyError{}
	child := NewDynamicNode("child", func(agent.Context, any, func(*session.Event) error) (any, error) {
		return nil, failure
	}, NodeConfig{})
	var childErr error
	parent := NewDynamicNode("parent", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
		_, childErr = RunNode[any](ctx, child, nil)
		return nil, childErr
	}, NodeConfig{})
	var got error
	for _, err := range parent.Run(agent.Promote(newMockCtx(t)), nil) {
		if err != nil {
			got = err
		}
	}
	if got != childErr {
		t.Fatal("parent body error was replaced")
	}
	nre, ok := childErr.(*NodeRunError)
	if !ok {
		t.Fatal("existing NodeRunError wrapper was lost")
	}
	joined, ok := nre.Cause.(interface{ Unwrap() []error })
	if !ok {
		t.Fatal("existing child failure wrapper was lost")
	}
	causes := joined.Unwrap()
	if len(causes) != 2 || causes[0] != ErrNodeFailed || causes[1] != failure {
		t.Fatal("child business error identity changed inside NodeRunError")
	}
}
