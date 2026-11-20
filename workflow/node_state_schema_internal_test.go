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
	"context"
	"sync"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// The memo of derived state validators belongs to one run: every node of a run
// sees the same memo, and two runs of the same Workflow never share one, so a
// long-lived Workflow does not keep the validators of schemas built per run.
// The memo has no observable effect through the exported API, so this test
// reads it from the node contexts.
func TestStateValidatorMemo_IsPerRun(t *testing.T) {
	schema, err := (&jsonschema.Schema{
		Title:      "NodeSchema",
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"x": {Type: "integer"}},
	}).Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	var mu sync.Mutex
	var seen []*stateValidators
	record := func(ctx agent.Context) {
		mu.Lock()
		defer mu.Unlock()
		seen = append(seen, stateValidatorsFrom(ctx))
	}
	first := newFaninNode("first", NodeConfig{StateSchema: schema}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		record(ctx)
		faninEmit(ctx, yield, "ok")
	})
	child := newFaninNode("child", NodeConfig{}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		record(ctx)
		faninEmit(ctx, yield, "ok")
	})
	orch := NewDynamicNode("orch", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		_, err := RunNode[any](ctx, child, nil)
		return "done", err
	}, NodeConfig{})
	wf, err := New("wf", Chain(Start, first, orch))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	for run := range 2 {
		if _, err := runWithStore(t, wf); err != nil {
			t.Fatalf("run %d error = %v, want nil", run, err)
		}
	}

	if len(seen) != 4 {
		t.Fatalf("recorded %d contexts, want 4", len(seen))
	}
	if seen[0] == nil {
		t.Fatalf("first run's memo = nil, want one installed")
	}
	if seen[1] != seen[0] {
		t.Errorf("dynamic child's memo = %p, want the first node's (%p)", seen[1], seen[0])
	}
	if seen[3] != seen[2] {
		t.Errorf("second run: dynamic child's memo = %p, want the first node's (%p)", seen[3], seen[2])
	}
	if seen[2] == seen[0] {
		t.Errorf("second run's memo = %p, want one other than the first run's", seen[2])
	}
}

// A memo derives the validator of a schema once: later requests for the same
// schema, directly or through a node's config, get the same validator.
func TestStateValidatorMemo_OneValidatorPerSchema(t *testing.T) {
	schema, err := (&jsonschema.Schema{
		Title:      "NodeSchema",
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"x": {Type: "integer"}},
	}).Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}

	t.Run("forSchema", func(t *testing.T) {
		memo := &stateValidators{}

		first, err := memo.forSchema(schema)
		if err != nil {
			t.Fatalf("first forSchema() error = %v", err)
		}
		second, err := memo.forSchema(schema)
		if err != nil {
			t.Fatalf("second forSchema() error = %v", err)
		}

		if first != second {
			t.Errorf("second forSchema() = %p, want the first validator %p", second, first)
		}
	})

	t.Run("nodeStateValidator", func(t *testing.T) {
		ctx := context.WithValue(t.Context(), stateValidatorsKey{}, &stateValidators{})
		n := NewFunctionNode("n", func(agent.Context, any) (string, error) {
			return "", nil
		}, NodeConfig{StateSchema: schema})

		first, err := nodeStateValidator(ctx, n)
		if err != nil {
			t.Fatalf("first nodeStateValidator() error = %v", err)
		}
		second, err := nodeStateValidator(ctx, n)
		if err != nil {
			t.Fatalf("second nodeStateValidator() error = %v", err)
		}

		if first == nil || second == nil {
			t.Fatalf("nodeStateValidator() = %v, %v, want two validators", first, second)
		}
		if *first != *second {
			t.Errorf("second nodeStateValidator() = %v, want the first validator %v", *second, *first)
		}
	})
}
