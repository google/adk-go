package adk_test

import (
	"context"
	"errors"
	"testing"

	"github.com/google/adk-go"
)

type testAgent struct {
	run func(ctx context.Context, parentCtx *adk.InvocationContext) (adk.EventStream, error)
}

func (m *testAgent) Name() string        { return "TestAgent" }
func (m *testAgent) Description() string { return "" }
func (m *testAgent) Run(ctx context.Context, parentCtx *adk.InvocationContext) (adk.EventStream, error) {
	return m.run(ctx, parentCtx)
}

func TestNewInvocationContext_End(t *testing.T) {
	ctx := t.Context()
	waitForCancel := func(ctx context.Context, parentCtx *adk.InvocationContext) (adk.EventStream, error) {
		return func(yield func(*adk.Event, error) bool) {
			select {
			case <-ctx.Done():
				// stuck here until the context is canceled.
				yield(nil, ctx.Err())
				return
			}
		}, nil
	}
	agent := &testAgent{run: waitForCancel}

	ctx, ic := adk.NewInvocationContext(ctx, agent)
	events, err := agent.Run(ctx, ic)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	// schedule cancellation to happen after the agent starts running.
	go func() { ic.End(errors.New("end")) }()

	for ev, err := range events {
		if ev != nil || err == nil {
			t.Errorf("agent returned %v, %v, want cancellation", ev, err)
		}
		if err != nil {
			break
		}
	}
	if got := context.Cause(ctx); got.Error() != "end" {
		t.Errorf("context.Cause(ctx) = %v, want 'end'", got)
	}
}
