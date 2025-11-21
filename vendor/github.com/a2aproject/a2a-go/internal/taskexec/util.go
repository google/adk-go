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
	"errors"
	"fmt"

	"golang.org/x/sync/errgroup"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
)

func readQueueToChannels(ctx context.Context, queue eventqueue.Reader, eventChan chan a2a.Event, errorChan chan error) {
	for {
		event, err := queue.Read(ctx)
		if err != nil {
			select {
			case errorChan <- err:
			case <-ctx.Done():
			}
			return
		}

		select {
		case eventChan <- event:
		case <-ctx.Done():
			return
		}
	}
}

type eventProducerFn func(context.Context) error
type eventConsumerFn func(context.Context) (a2a.SendMessageResult, error)

// runProducerConsumer starts producer and consumer goroutines in an error group and waits
// for both of them to finish or one of them to fail. If both complete successfuly and consumer produces a result,
// the result is returned, otherwise an error is returned.
func runProducerConsumer(ctx context.Context, producer eventProducerFn, consumer eventConsumerFn) (a2a.SendMessageResult, error) {
	group, ctx := errgroup.WithContext(ctx)

	group.Go(func() (err error) {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("event producer panic: %v", r)
			}
		}()
		err = producer(ctx)
		return
	})

	// The error is returned to cancel producer context when consumer decides to return a result and stop processing events.
	errConsumerStopped := errors.New("consumer stopped")

	var result *a2a.SendMessageResult
	group.Go(func() (err error) {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("event consumer panic: %v", r)
			}
		}()
		localResult, err := consumer(ctx)
		result = &localResult
		if err == nil {
			err = errConsumerStopped
		}
		return
	})

	if err := group.Wait(); err != nil && !errors.Is(err, errConsumerStopped) {
		return nil, err
	}

	if *result == nil {
		return nil, fmt.Errorf("bug: consumer stopped, but result unset")
	}

	return *result, nil
}
