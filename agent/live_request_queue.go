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

package agent

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/adk/model"
)

// ErrQueueClosed is returned when Send is called on a closed LiveRequestQueue.
var ErrQueueClosed = errors.New("live request queue is closed")

// LiveRequestQueue is a concurrency-safe buffered channel for LiveRequest messages.
// Core invariant: ch is NEVER closed. Only done is closed on Close(). This
// eliminates all "send on closed channel" panics in M:1 producer scenarios.
//
// If Send's select picks q.ch when q.done is also ready, the message is
// enqueued and Send returns nil. The consumer (senderLoop) MUST drain
// q.Events() after Done() fires to guarantee no messages are lost.
type LiveRequestQueue struct {
	ch            chan *model.LiveRequest
	done          chan struct{}
	closeOnce     sync.Once
	modelSpeaking int32 // atomic
}

// NewLiveRequestQueue creates a new LiveRequestQueue with the given buffer size.
func NewLiveRequestQueue(bufferSize int) *LiveRequestQueue {
	if bufferSize <= 0 {
		bufferSize = 100
	}
	return &LiveRequestQueue{
		ch:   make(chan *model.LiveRequest, bufferSize),
		done: make(chan struct{}),
	}
}

// Send enqueues a LiveRequest. Returns ErrQueueClosed if the queue has been
// closed, or ctx.Err() if the context is cancelled.
//
// If q.ch <- req and <-q.done are both ready, Go's select may pick either.
// When q.ch <- req wins, Send returns nil and the message is enqueued —
// senderLoop's drain guarantees it will be processed.
func (q *LiveRequestQueue) Send(ctx context.Context, req *model.LiveRequest) error {
	req.EnqueuedAt = time.Now()
	// Fast-path: honour already-cancelled context deterministically.
	if err := ctx.Err(); err != nil {
		return err
	}
	// Fast-path: already-closed queue.
	select {
	case <-q.done:
		return ErrQueueClosed
	default:
	}
	select {
	case q.ch <- req:
		return nil
	case <-q.done:
		return ErrQueueClosed
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Close signals the queue is done. Only done is closed; ch is never closed.
// Idempotent via sync.Once.
func (q *LiveRequestQueue) Close() {
	q.closeOnce.Do(func() {
		close(q.done)
	})
}

// Events returns the read-only channel for consuming requests.
// Consumer must also select on Done() to detect queue closure.
func (q *LiveRequestQueue) Events() <-chan *model.LiveRequest {
	return q.ch
}

// Done returns a channel that is closed when the queue is closed.
func (q *LiveRequestQueue) Done() <-chan struct{} {
	return q.done
}

// Len returns the current number of buffered requests (snapshot).
func (q *LiveRequestQueue) Len() int {
	return len(q.ch)
}

// ModelSpeaking returns whether the model is currently producing audio output.
func (q *LiveRequestQueue) ModelSpeaking() bool {
	return atomic.LoadInt32(&q.modelSpeaking) == 1
}

// SetModelSpeaking atomically sets the model speaking state.
func (q *LiveRequestQueue) SetModelSpeaking(speaking bool) {
	var v int32
	if speaking {
		v = 1
	}
	atomic.StoreInt32(&q.modelSpeaking, v)
}
