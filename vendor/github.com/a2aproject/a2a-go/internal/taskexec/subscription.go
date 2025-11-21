// Copyright 2025 The A2A Authors
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

package taskexec

import (
	"context"
	"fmt"
	"iter"

	"github.com/a2aproject/a2a-go/a2a"
)

// Subscription encapsulates the logic of subscribing a channel to [Execution] events and canceling the subscription.
// A default subscription is created when an Execution is started.
type Subscription struct {
	eventsChan chan a2a.Event
	execution  *Execution
	consumed   bool
}

// newSubscription tries to subscribe a channel to Execution events. If the Execution ends,
// an "empty" subscription is returned.
func newSubscription(ctx context.Context, e *Execution) (*Subscription, error) {
	ch := make(chan a2a.Event)

	select {
	case <-ctx.Done():
		return nil, ctx.Err()

	case e.subscribeChan <- ch:
		return &Subscription{eventsChan: ch, execution: e}, nil

	case <-e.result.done:
		return &Subscription{}, nil
	}
}

// newDefaultSubscription is called on Execution which is not yet running.
// Using newSubscription() might lead to Execute() callers missing first events.
func newDefaultSubscription(e *Execution) *Subscription {
	ch := make(chan a2a.Event)
	e.subscribers[ch] = struct{}{}
	return &Subscription{eventsChan: ch, execution: e}
}

// Events returns a sequence of Events produced during Execution. If Execution resolves to an error
// the error gets reported. A sequence can only be consumed once. Subscription gets automatically
// canceled once event consumption stops.
func (s *Subscription) Events(ctx context.Context) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		if s.eventsChan == nil || s.execution == nil {
			return
		}
		if s.consumed {
			yield(nil, fmt.Errorf("subscription already consumed"))
			return
		}
		s.consumed = true

		defer s.cancel()

		executionFinished := false
		for !executionFinished {
			select {
			case <-ctx.Done():
				yield(nil, ctx.Err())
				return

			case <-s.execution.result.done:
				executionFinished = true

			case event, ok := <-s.eventsChan:
				if !ok {
					executionFinished = true
					break
				}
				if !yield(event, nil) {
					return
				}
			}
		}

		// check if execution terminated unsuccessfully and report it:
		if executionFinished {
			if _, err := s.execution.Result(ctx); err != nil {
				yield(nil, err)
			}
		}
	}
}

// cancel unsubscribes events channel from Execution events. If the Execution ends,
// the operation is a no-op.
func (s *Subscription) cancel() {
	if s == nil {
		return
	}
	if s.eventsChan == nil || s.execution == nil {
		return
	}

	for {
		select {
		case <-s.eventsChan:
			// Do not block processor goroutine on trying to notify the subscription
			// which is trying to unsubscribe from events mid-execution.

		case s.execution.unsubscribeChan <- s.eventsChan:
			return

		case <-s.execution.result.done:
			// Execution goroutine was stopped disposing all subscriptions.
			return
		}
	}
}
