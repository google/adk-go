// Copyright 2025 Google LLC
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

package adk_test

import (
	"context"
	"errors"
	"iter"
	"testing"

	"github.com/google/adk-go"
)

type testAgent struct {
	run func(ctx context.Context, parentCtx *adk.InvocationContext) iter.Seq2[*adk.Event, error]
}

func (m *testAgent) Name() string        { return "TestAgent" }
func (m *testAgent) Description() string { return "" }
func (m *testAgent) Run(ctx context.Context, parentCtx *adk.InvocationContext) iter.Seq2[*adk.Event, error] {
	return m.run(ctx, parentCtx)
}

func TestNewInvocationContext_End(t *testing.T) {
	ctx := t.Context()
	waitForCancel := func(ctx context.Context, parentCtx *adk.InvocationContext) iter.Seq2[*adk.Event, error] {
		return func(yield func(*adk.Event, error) bool) {
			<-ctx.Done()
			// stuck here until the context is canceled.
			yield(nil, ctx.Err())
		}
	}
	agent := &testAgent{run: waitForCancel}

	ctx, ic := adk.NewInvocationContext(ctx, agent)
	// schedule cancellation to happen after the agent starts running.
	go func() { ic.End(errors.New("end")) }()

	for ev, err := range agent.Run(ctx, ic) {
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
