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

package controllers

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/websocket"
)

// TestRunLiveWebSocketMessageSizeBoundary is a mechanism-level regression
// test for the /run_live message-size limit, not a full RunLiveHandler
// integration test.
//
// This checkout's Agent interface has an unexported internal() method and
// agent.New()'s returned type does not implement the liveAgent interface
// Runner.RunLive() requires, so there is no way via the public API to build
// a fake agent that supports live sessions here -- RunLiveHandler would
// always fail at r.RunLive() before its read loop (and so before
// ws.SetReadLimit's effect) is ever reachable in a test.
//
// This test instead exercises the exact websocket.Upgrader configuration
// and ws.SetReadLimit(maxLiveMessageBytes) call RunLiveHandler uses, in
// isolation, confirming: a message of exactly the limit is accepted, and
// one byte over gets close code 1009 (CloseMessageTooBig) instead of being
// silently buffered. Deleting the SetReadLimit call in RunLiveHandler does
// not fail this test (since it calls SetReadLimit directly, matching the
// handler's own configuration) -- it is a test of the limit's mechanism,
// not a regression guard on the handler's wiring. A true end-to-end guard
// needs the missing live-agent test scaffolding; see PR discussion.
func TestRunLiveWebSocketMessageSizeBoundary(t *testing.T) {
	upgrader := websocket.Upgrader{
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
	}

	handler := func(rw http.ResponseWriter, req *http.Request) {
		ws, err := upgrader.Upgrade(rw, req, nil)
		if err != nil {
			return
		}
		defer func() { _ = ws.Close() }()

		// Matches RunLiveHandler's own call exactly.
		ws.SetReadLimit(maxLiveMessageBytes)

		_, _, _ = ws.ReadMessage()
	}

	t.Run("at limit is accepted", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(handler))
		defer server.Close()

		wsURL := "ws" + strings.TrimPrefix(server.URL, "http")
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			t.Fatalf("Dial() failed: %v", err)
		}
		defer conn.Close()

		payload := make([]byte, maxLiveMessageBytes)
		if err := conn.WriteMessage(websocket.BinaryMessage, payload); err != nil {
			t.Fatalf("WriteMessage(at-limit) failed: %v", err)
		}
		// The handler's single ReadMessage call consuming this without
		// erroring, and the connection not being closed with
		// CloseMessageTooBig, is the acceptance signal here.
		if _, _, err := conn.ReadMessage(); err != nil {
			if ce, ok := err.(*websocket.CloseError); ok && ce.Code == websocket.CloseMessageTooBig {
				t.Fatalf("at-limit message was rejected with CloseMessageTooBig")
			}
			// Any other error (e.g. the server closing after its single
			// read) is expected here; the assertion above is what matters.
		}
	})

	t.Run("one byte over is rejected", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(handler))
		defer server.Close()

		wsURL := "ws" + strings.TrimPrefix(server.URL, "http")
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			t.Fatalf("Dial() failed: %v", err)
		}
		defer conn.Close()

		payload := make([]byte, maxLiveMessageBytes+1)
		if err := conn.WriteMessage(websocket.BinaryMessage, payload); err != nil {
			t.Fatalf("WriteMessage(oversized) failed: %v", err)
		}

		_, _, err = conn.ReadMessage()
		ce, ok := err.(*websocket.CloseError)
		if !ok {
			t.Fatalf("ReadMessage() error = %v, want a *websocket.CloseError", err)
		}
		if ce.Code != websocket.CloseMessageTooBig {
			t.Errorf("close code = %d, want %d (CloseMessageTooBig)", ce.Code, websocket.CloseMessageTooBig)
		}
	})
}
