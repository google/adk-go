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
	"strings"
	"testing"

	"google.golang.org/adk/session"
)

func TestRunNode_ErrInvalidRunNodeContext_OnStaticContext(t *testing.T) {
	ctx := newNodeContext(newMockCtx(t), nil) // no subScheduler attached
	_, err := RunNode[string](ctx, newStubNode("c", "x"), nil)
	if !errors.Is(err, ErrInvalidRunNodeContext) {
		t.Errorf("err = %v, want ErrInvalidRunNodeContext", err)
	}
}

func TestRunNode_ReturnsTypedOutput(t *testing.T) {
	child := newStubNode("c", "hello")
	got := runInOrchestrator[string](t, func(ctx NodeContext) (string, error) {
		return RunNode[string](ctx, child, nil)
	})
	if got != "hello" {
		t.Errorf("RunNode output = %q, want %q", got, "hello")
	}
}

func TestRunNode_OutputTypeMismatch(t *testing.T) {
	child := newStubNode("c", 42) // emits int
	_, err := runInOrchestratorWithErr[string](t, func(ctx NodeContext) (string, error) {
		return RunNode[string](ctx, child, nil) // expects string
	})
	if err == nil {
		t.Fatal("expected error for type mismatch, got nil")
	}
	if !strings.Contains(err.Error(), "does not satisfy expected") {
		t.Errorf("err = %v, want type-mismatch message", err)
	}
}

func TestRunNode_PropagatesErrNodeInterrupted(t *testing.T) {
	asker := newRequestInputNode("asker", "approve?")
	_, err := runInOrchestratorWithErr[string](t, func(ctx NodeContext) (string, error) {
		return RunNode[string](ctx, asker, nil)
	})
	if !errors.Is(err, ErrNodeInterrupted) {
		t.Errorf("err = %v, want errors.Is ErrNodeInterrupted", err)
	}
}

func TestRunNode_PropagatesErrNodeFailed(t *testing.T) {
	failer := newFailingNode("failer", errors.New("boom"))
	_, err := runInOrchestratorWithErr[string](t, func(ctx NodeContext) (string, error) {
		return RunNode[string](ctx, failer, nil)
	})
	if !errors.Is(err, ErrNodeFailed) {
		t.Errorf("err = %v, want errors.Is ErrNodeFailed", err)
	}
}

func TestRunNode_WithRunID_AppearsInChildPath(t *testing.T) {
	child := newStubNode("processor", "")
	runInOrchestrator[string](t, func(ctx NodeContext) (string, error) {
		return RunNode[string](ctx, child, nil, WithRunID("order-42"))
	})
	if got, want := child.lastPath, "orch/processor@order-42"; got != want {
		t.Errorf("child Path() = %q, want %q", got, want)
	}
}

func TestRunNode_WithRunID_InvalidRejected(t *testing.T) {
	child := newStubNode("c", "")
	_, err := runInOrchestratorWithErr[string](t, func(ctx NodeContext) (string, error) {
		return RunNode[string](ctx, child, nil, WithRunID("123")) // purely numeric
	})
	if !errors.Is(err, ErrInvalidRunID) {
		t.Errorf("err = %v, want errors.Is ErrInvalidRunID", err)
	}
}

func TestRunNode_NilChildOutput_ReturnsZero(t *testing.T) {
	child := newStubNode("c", nil)
	got := runInOrchestrator[string](t, func(ctx NodeContext) (string, error) {
		return RunNode[string](ctx, child, nil)
	})
	if got != "" {
		t.Errorf("nil child output → OUT = %q, want \"\" (zero)", got)
	}
}

// --- test helpers ---

// runInOrchestrator drives orchestratorFn inside a dynamic node so that
// the RunNode calls inside have a valid NodeContext + sub-scheduler.
func runInOrchestrator[OUT any](t *testing.T, orchestratorFn func(NodeContext) (OUT, error)) OUT {
	t.Helper()
	got, err := runInOrchestratorWithErr[OUT](t, orchestratorFn)
	if err != nil {
		t.Fatalf("orchestrator error: %v", err)
	}
	return got
}

func runInOrchestratorWithErr[OUT any](t *testing.T, orchestratorFn func(NodeContext) (OUT, error)) (OUT, error) {
	t.Helper()
	var (
		got    OUT
		gotErr error
	)
	n := NewDynamicNode[string, OUT]("orch",
		func(ctx NodeContext, _ string, _ func(*session.Event) error) (OUT, error) {
			got, gotErr = orchestratorFn(ctx)
			if gotErr != nil {
				return got, gotErr
			}
			return got, nil
		},
		NodeConfig{},
	)
	_, runErr := drainDynamicWithErr(t, n, "")
	if runErr != nil {
		return got, runErr
	}
	return got, gotErr
}
