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

package llminternal

import (
	"context"
	"errors"
	"iter"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestRunLiveCancellationClosesConnectionBeforeIteratorIsConsumed(t *testing.T) {
	connected := make(chan struct{})
	disconnected := make(chan struct{})
	client, _ := startFakeLiveServer(t, func(_ int, conn *websocket.Conn) {
		t.Cleanup(func() { _ = conn.Close() })
		close(connected)
		blockUntilClientCloses(conn)
		close(disconnected)
	})
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()
	flow := &Flow{
		Model: &fakeLiveModel{client: client},
		RequestProcessors: []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{
			liveConfigProcessor,
		},
	}
	sess, events, err := flow.RunLive(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = sess.Close() }()
	select {
	case <-connected:
	case <-time.After(5 * time.Second):
		t.Fatal("live connection was not established")
	}
	cancel()
	select {
	case <-disconnected:
	case <-time.After(5 * time.Second):
		t.Fatal("cancellation left the live connection open until the event iterator was consumed")
	}
	completed := make(chan int, 1)
	go func() {
		var errorCount int
		for event, err := range events {
			if event != nil {
				t.Error("idle canceled session yielded an event")
			}
			if !errors.Is(err, context.Canceled) {
				t.Errorf("iterator error has type %T, want context cancellation", err)
			}
			errorCount++
		}
		completed <- errorCount
	}()
	select {
	case errorCount := <-completed:
		if errorCount != 1 {
			t.Errorf("iterator yielded %d errors, want one", errorCount)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("canceled event iterator did not finish")
	}
}
