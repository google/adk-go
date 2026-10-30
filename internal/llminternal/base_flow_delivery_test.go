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
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
)

// TestRunLiveDeliversToolResponseAfterReconnect pins the delivery half of
// #1819: a function response produced while the connection is down reaches the
// next connection exactly once. Before this fix the pending value was scoped to
// the per-connection loop, so it was discarded with the socket and the model's
// call was left unanswered.
func TestRunLiveDeliversToolResponseAfterReconnect(t *testing.T) {
	baseline, _ := runLiveStacks()

	const toolCall = `{"toolCall":{"functionCalls":[{"id":"t1","name":"slow_tool","args":{}}]}}`

	var mu sync.Mutex
	var conn2Payloads []string
	delivered := make(chan struct{})
	var once sync.Once

	client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
		if connNum == 1 {
			if err := conn.WriteMessage(websocket.TextMessage, []byte(toolCall)); err != nil {
				return
			}
			// A close frame reaches the client's reader at a known point,
			// unlike a TCP FIN, so the response below is deterministically
			// produced after the connection is already known to be gone.
			_ = conn.WriteMessage(websocket.CloseMessage,
				websocket.FormatCloseMessage(websocket.ClosePolicyViolation, "policy"))
			return
		}
		_ = conn.SetReadDeadline(time.Now().Add(10 * time.Second))
		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			mu.Lock()
			conn2Payloads = append(conn2Payloads, string(data))
			mu.Unlock()
			if strings.Contains(string(data), "functionResponse") {
				once.Do(func() { close(delivered) })
				_ = conn.WriteMessage(websocket.TextMessage, []byte(serverContentPong))
			}
		}
	})

	f := newReconnectFlow(client, testLiveReconnectPolicy(4, 20))
	f.Tools = []tool.Tool{
		&mockFunctionTool{name: "slow_tool", runFunc: func(agent.Context, map[string]any) (map[string]any, error) {
			time.Sleep(200 * time.Millisecond)
			return map[string]any{"ok": true}, nil
		}},
	}
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}

	pong := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		var once sync.Once
		for ev, err := range seq {
			if err != nil || ev == nil || ev.LLMResponse.Content == nil {
				continue
			}
			for _, p := range ev.LLMResponse.Content.Parts {
				if p.Text == "pong" {
					once.Do(func() { close(pong) })
				}
			}
		}
	}()

	select {
	case <-delivered:
	case <-pong:
	case <-time.After(10 * time.Second):
		t.Fatal("the tool response never reached the second connection")
	}
	_ = sess.Close()
	wg.Wait()

	if got, want := connCount.Load(), int32(2); got != want {
		t.Fatalf("dialled %d times, want %d (the scenario needs one reconnect)", got, want)
	}

	mu.Lock()
	count := 0
	for _, p := range conn2Payloads {
		if strings.Contains(p, "functionResponse") {
			count++
		}
	}
	mu.Unlock()
	if count != 1 {
		t.Errorf("functionResponse payloads on the second connection = %d, want exactly 1", count)
	}

	assertNoRunLiveLeak(t, baseline)
}

// TestRunLiveDoesNotResendToolResponseAfterSuccessfulWrite guards the other
// half of the delivery decision: once the sender has recorded a successful
// write, a connection that dies afterwards must not cause the response to be
// sent again. Delivery to the server is then unknown, and re-sending would
// duplicate the response for one function call. This stays at-most-once until
// #1819's open maintainer decision.
func TestRunLiveDoesNotResendToolResponseAfterSuccessfulWrite(t *testing.T) {
	baseline, _ := runLiveStacks()

	const toolCall = `{"toolCall":{"functionCalls":[{"id":"t1","name":"fast_tool","args":{}}]}}`

	var mu sync.Mutex
	var conn2Payloads []string

	client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
		if connNum == 1 {
			if err := conn.WriteMessage(websocket.TextMessage, []byte(toolCall)); err != nil {
				return
			}
			// Close only after the response reached the wire, so the write is
			// known to have succeeded and the drop happens afterwards.
			_ = conn.SetReadDeadline(time.Now().Add(5 * time.Second))
			for {
				_, data, err := conn.ReadMessage()
				if err != nil {
					return
				}
				if strings.Contains(string(data), "functionResponse") {
					return
				}
			}
		}
		// Record everything the second connection receives for a bounded
		// window, then let the client finish.
		_ = conn.SetReadDeadline(time.Now().Add(time.Second))
		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				break
			}
			mu.Lock()
			conn2Payloads = append(conn2Payloads, string(data))
			mu.Unlock()
		}
		_ = conn.WriteMessage(websocket.TextMessage, []byte(serverContentPong))
		blockUntilClientCloses(conn)
	})

	f := newReconnectFlow(client, testLiveReconnectPolicy(4, 20))
	f.Tools = []tool.Tool{
		&mockFunctionTool{name: "fast_tool", runFunc: func(agent.Context, map[string]any) (map[string]any, error) {
			return map[string]any{"ok": true}, nil
		}},
	}
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}

	pong := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		var once sync.Once
		for ev, err := range seq {
			if err != nil || ev == nil || ev.LLMResponse.Content == nil {
				continue
			}
			for _, p := range ev.LLMResponse.Content.Parts {
				if p.Text == "pong" {
					once.Do(func() { close(pong) })
				}
			}
		}
	}()

	select {
	case <-pong:
	case <-time.After(10 * time.Second):
		t.Fatal("the flow never recovered on the second connection")
	}
	_ = sess.Close()
	wg.Wait()

	if got, want := connCount.Load(), int32(2); got != want {
		t.Fatalf("dialled %d times, want %d (the scenario needs one reconnect)", got, want)
	}

	mu.Lock()
	resent := 0
	for _, p := range conn2Payloads {
		if strings.Contains(p, "functionResponse") {
			resent++
		}
	}
	mu.Unlock()
	if resent != 0 {
		t.Errorf("functionResponse payloads on the second connection = %d, want 0: the write already succeeded", resent)
	}

	assertNoRunLiveLeak(t, baseline)
}

// TestKeepPendingAfterError covers the decision the select-order-independent
// branch makes on a resume: only a reported write failure is provably unsent.
func TestKeepPendingAfterError(t *testing.T) {
	tests := []struct {
		name    string
		sent    bool
		outcome int32
		want    bool
	}{
		{"nothing handed over", false, liveWriteNotAttempted, true},
		{"nothing handed over after an earlier attempt", false, liveWriteSucceeded, true},
		{"write failed", true, liveWriteFailed, true},
		{"write succeeded", true, liveWriteSucceeded, false},
		{"outcome not recorded yet", true, liveWriteNotAttempted, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := keepPendingAfterError(tc.sent, tc.outcome); got != tc.want {
				t.Errorf("keepPendingAfterError(%t, %d) = %t, want %t", tc.sent, tc.outcome, got, tc.want)
			}
		})
	}
}
