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

package liveflow

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

// ConnectFn creates a new live connection, optionally resuming a previous
// session via the provided handle (empty string = new session).
type ConnectFn func(handle string) (model.LiveConnection, error)

// Reconnect retry knobs. Declared as vars (rather than consts) so tests can
// override them to keep wall time low and make context-cancellation timing
// deterministic. Production behavior is unchanged.
var (
	maxReconnectRetries  = 3
	reconnectBaseBackoff = time.Second
	reconnectGoAwaySleep = 500 * time.Millisecond
)

// connectWithRetry attempts to connect up to maxReconnectRetries+1 times,
// applying linear backoff between attempts. Returns context.Canceled (or
// the deadline error) immediately if the parent context is cancelled at
// any point — including during backoff — so callers do not wait out timers
// after a cancel.
func connectWithRetry(ctx context.Context, handle string, connectFn ConnectFn) (model.LiveConnection, error) {
	var lastErr error
	for attempt := 0; attempt <= maxReconnectRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		conn, err := connectFn(handle)
		if err == nil {
			return conn, nil
		}
		lastErr = err
		if attempt < maxReconnectRetries {
			if !sleepOrCancel(ctx, time.Duration(attempt+1)*reconnectBaseBackoff) {
				return nil, ctx.Err()
			}
		}
	}
	return nil, fmt.Errorf("failed to connect live after %d retries: %w", maxReconnectRetries, lastErr)
}

// sleepOrCancel waits for d to elapse or for ctx to be cancelled, returning
// true if the duration elapsed and false if the context was cancelled.
func sleepOrCancel(ctx context.Context, d time.Duration) bool {
	select {
	case <-time.After(d):
		return true
	case <-ctx.Done():
		return false
	}
}

// isConnectionClosed reports whether err looks like a closed transport:
// io.EOF, an unexpectedly-closed read, or a WebSocket close frame with
// code 1000 (normal) / 1006 (abnormal). When such a close arrives without
// a preceding GoAway frame, callers can decide — based on whether a
// resumption handle is available — whether to reconnect via the outer
// reconnect cycle or surface the error.
//
// Permissive on purpose: when in doubt, prefer treating it as closed so
// the outer cycle's retry budget remains the gate against runaway
// reconnects.
func isConnectionClosed(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	// gorilla/websocket's *CloseError exposes Code() int.
	var ce interface{ Code() int }
	if errors.As(err, &ce) {
		switch ce.Code() {
		case 1000, 1006:
			return true
		}
	}
	// Substring fallback for transports that render close frames as
	// plain strings without a typed wrapper. "close 1000" / "close 1006"
	// are supersets of "websocket: close 100X" so we don't need both.
	msg := err.Error()
	switch {
	case strings.Contains(msg, "close 1000"),
		strings.Contains(msg, "close 1006"):
		return true
	}
	return false
}

// isEOF returns true if the error is io.EOF or wraps it.
func isEOF(err error) bool {
	return errors.Is(err, io.EOF)
}

// applySessionResumptionUpdate mutates ctx's resumption handle in response
// to a server-supplied SessionResumptionUpdate event, if the message carries
// one. Non-resumable updates clear any stale handle so the next reconnect
// uses an empty handle.
func applySessionResumptionUpdate(ctx agent.InvocationContext, msg eventOrError) {
	if msg.event == nil || msg.event.SessionResumptionUpdate == nil {
		return
	}
	upd := msg.event.SessionResumptionUpdate
	if !upd.Resumable {
		ctx.SetLiveSessionResumptionHandle("")
		return
	}
	if upd.NewHandle != "" {
		ctx.SetLiveSessionResumptionHandle(upd.NewHandle)
	}
}

func isGoAway(msg eventOrError) bool {
	return msg.event != nil && msg.event.GoAway != nil
}
