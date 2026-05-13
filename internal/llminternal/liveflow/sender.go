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
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

// trackedSend wraps conn.Send, stamping SentAt on the request and
// recording the send time in shared timing state.
func trackedSend(ctx context.Context, conn model.LiveConnection, req *model.LiveRequest, ts *liveTimingState) error {
	now := time.Now()
	req.SentAt = now
	ts.recordSend(now)
	return conn.Send(ctx, req)
}

// queueDone reports whether the LiveRequestQueue has been closed by the
// caller (queue.Done() channel is signalled). Used to skip the close-
// reconnect path during a client-initiated shutdown — without this gate,
// closing the queue triggers conn.Close, which the receiver observes as
// an EOF and would otherwise re-enter the reconnect cycle.
func queueDone(queue *agent.LiveRequestQueue) bool {
	select {
	case <-queue.Done():
		return true
	default:
		return false
	}
}

// sendAndSignal sends a request on the connection and signals turnResetCh.
// Returns a non-nil error if the send fails.
func sendAndSignal(
	ctx context.Context,
	conn model.LiveConnection,
	req *model.LiveRequest,
	ts *liveTimingState,
	eventCh chan<- eventOrError,
	turnResetCh chan<- struct{},
) error {
	if err := trackedSend(ctx, conn, req, ts); err != nil {
		sendEvent(ctx, eventCh, eventOrError{err: err})
		return err
	}
	// Signal receiverLoop that user activity occurred.
	// Non-blocking: if signal already pending, skip — receiver will see it.
	select {
	case turnResetCh <- struct{}{}:
	default:
	}
	return nil
}

// senderLoop forwards queue messages to the live connection.
// On queue.Done(), it drains any remaining buffered messages before returning.
func (lf *LiveFlow) senderLoop(
	ctx context.Context,
	conn model.LiveConnection,
	queue *agent.LiveRequestQueue,
	ts *liveTimingState,
	eventCh chan<- eventOrError,
	turnResetCh chan<- struct{},
) {
	for {
		select {
		case req := <-queue.Events():
			if sendAndSignal(ctx, conn, req, ts, eventCh, turnResetCh) != nil {
				return
			}
		case <-queue.Done():
			lf.drainQueue(ctx, conn, queue, ts, eventCh, turnResetCh)
			return
		case <-ctx.Done():
			return
		}
	}
}

// drainQueue sends any remaining buffered messages before senderLoop exits.
func (lf *LiveFlow) drainQueue(
	ctx context.Context,
	conn model.LiveConnection,
	queue *agent.LiveRequestQueue,
	ts *liveTimingState,
	eventCh chan<- eventOrError,
	turnResetCh chan<- struct{},
) {
	for {
		select {
		case req := <-queue.Events():
			if sendAndSignal(ctx, conn, req, ts, eventCh, turnResetCh) != nil {
				return
			}
		default:
			return
		}
	}
}
