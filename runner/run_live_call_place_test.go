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

package runner

import (
	"context"
	"iter"
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/adkcontext"
	"google.golang.org/adk/v2/session"
)

// TestRunner_RunLive_NestedRunIsNotOrderedByCallingBatch pins that a live run
// started from a function call's context does not keep the call's place in the
// calling batch: code in the live run that waits for the turn of the context it
// was given does not wait for the batch's earlier calls. It fails if RunLive
// passes the caller's place on, because call 0 never returns here.
func TestRunner_RunLive_NestedRunIsNotOrderedByCallingBatch(t *testing.T) {
	order := adkcontext.NewCallOrder(2) // call 0 never finishes
	callCtx := adkcontext.WithCallPlace(t.Context(), order, 1)

	waited := make(chan error, 1)
	r, _ := newRunLiveTestRunner(t, "s", func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		waitCtx, cancel := context.WithTimeout(ctx, time.Second)
		defer cancel()
		waited <- adkcontext.AwaitEarlierCalls(waitCtx)
		return func(func(*session.Event, error) bool) {}
	})
	_, events, err := r.RunLive(callCtx, "testUser", "s", agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}
	for range events { //nolint:errcheck // drains the stream; its errors are not under test
	}

	select {
	case err := <-waited:
		if err != nil {
			t.Errorf("AwaitEarlierCalls in the live run = %v, want nil at once", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the live run never reached its sequence")
	}
}
