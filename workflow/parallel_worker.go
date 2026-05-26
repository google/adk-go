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

package workflow

import (
	"context"
	"fmt"
	"iter"
	"reflect"
	"sync"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// ParallelWorker runs a wrapped node in parallel for each item in the input list.
type ParallelWorker struct {
	BaseNode
	wrapped        Node
	maxConcurrency int
}

// NewParallelWorker creates a new ParallelWorker node.
// maxConcurrency <= 0 means no limit on concurrency.
func NewParallelWorker(name string, wrapped Node, maxConcurrency int, cfg NodeConfig) (*ParallelWorker, error) {
	if wrapped.Config().RetryConfig != nil {
		return nil, fmt.Errorf("ParallelWorker %s: wrapped node %s cannot have RetryConfig", name, wrapped.Name())
	}
	return &ParallelWorker{
		BaseNode:       BaseNode{name: name, config: cfg},
		wrapped:        wrapped,
		maxConcurrency: maxConcurrency,
	}, nil
}

// Run executes the wrapped node in parallel for each item in the input list.
// It aggregates the "output" from each wrapped node execution into a list and
// yields a single final event with the aggregated list as output.
//
// RetryConfig in the wrapped nodes are not allowed, only the parent node (ParallelWorker)
// will be respected. Each failed input will be retried based on the RetryConfig independently from other inputs.
// Which means for the input: []any{"a", "b", "c"}, if "b" always fails, and MaxAttempt is 3
// the ParallelWorker will perform 1 ("a") + 3 ("b" retried) + 1 ("c") = 5 calls in total.
//
// If any of the wrapped node executions returns a non-retryable error, the workflow
// will fail fast, cancel other in-flight workers, and return this first encountered error.
//
// In case the wrapped node produces more then one output event, they will be
// aggregated into a list, and the final result will be a multi dimensional list.
//
// Intermediate non-output events emitted by the wrapped node are suppressed.
func (n *ParallelWorker) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		cancelCtx, cancelFunc := context.WithCancel(ctx)
		defer cancelFunc()
		workerCtx := ctx.WithContext(cancelCtx)

		v := reflect.ValueOf(input)
		if v.Kind() != reflect.Slice {
			yield(nil, fmt.Errorf("parallel worker %s expects a slice input, got %T", n.Name(), input))
			return
		}

		nItems := v.Len()
		if nItems == 0 {
			// Yield an empty list as output
			event := session.NewEvent(ctx.InvocationID())
			event.Actions.StateDelta["output"] = []any{}
			yield(event, nil)
			return
		}

		outputs := make([]any, nItems)
		var wg sync.WaitGroup
		wg.Add(nItems)

		var sem chan struct{}
		if n.maxConcurrency > 0 {
			sem = make(chan struct{}, n.maxConcurrency)
		}

		resCh := make(chan workerResult, nItems)

		for i := 0; i < nItems; i++ {
			item := v.Index(i).Interface()

			if sem != nil {
				select {
				case sem <- struct{}{}:
				case <-ctx.Done():
					wg.Done()
					continue
				}
			}

			go n.runWorker(workerCtx, i, item, sem, resCh, &wg)
		}

		// Goroutine to close channel when all workers are done
		go func() {
			wg.Wait()
			close(resCh)
		}()

		var firstErr error

		for res := range resCh {
			if res.err != nil {
				if firstErr == nil {
					firstErr = res.err
					cancelFunc() // Cancel all other workers!
				}
				continue
			}

			if res.ev != nil {
				if out, ok := res.ev.Actions.StateDelta["output"]; ok {
					outputs[res.index] = out
				}
			}
		}

		if firstErr != nil {
			yield(nil, firstErr)
			return
		}

		// Yield the aggregated output
		event := session.NewEvent(ctx.InvocationID())
		event.Actions.StateDelta["output"] = outputs
		yield(event, nil)
	}
}

type workerResult struct {
	index int
	ev    *session.Event
	err   error
}

func (n *ParallelWorker) runWorker(ctx agent.InvocationContext, idx int, item any, sem chan struct{}, resCh chan<- workerResult, wg *sync.WaitGroup) {
	defer wg.Done()
	defer func() {
		if sem != nil {
			<-sem
		}
	}()

	retryCfg := n.Config().RetryConfig
	failedAttempts := 0

	for {
		var workerOutputs []any
		var runErr error

		for ev, err := range n.wrapped.Run(ctx, item) {
			if err != nil {
				runErr = err
				break
			}

			if out, ok := extractOutput(ev); ok {
				workerOutputs = append(workerOutputs, out)
			}
		}

		if runErr == nil {
			// On success populate the output event.
			resCh <- workerResult{index: idx, ev: makeWorkerOutputEvent(workerOutputs)}
			return
		}

		// Failure, check if the retry config is configured.
		// If so, follow the retry logic and repeat the execution of the wrapped node on failed input.
		failedAttempts++
		if ShouldRetry(retryCfg, runErr, failedAttempts) {
			delay := CalculateDelay(retryCfg, failedAttempts)
			select {
			case <-time.After(delay):
				continue
			case <-ctx.Done():
				resCh <- workerResult{index: idx, err: ctx.Err()}
				return
			}
		}

		// Cannot retry or exhausted attempts
		resCh <- workerResult{index: idx, err: runErr}
		return
	}
}

func makeWorkerOutputEvent(outputs []any) *session.Event {
	if len(outputs) == 0 {
		return nil
	}
	var output any
	if len(outputs) == 1 {
		output = outputs[0]
	} else {
		output = outputs
	}
	return &session.Event{Actions: session.EventActions{StateDelta: map[string]any{"output": output}}}
}

func extractOutput(ev *session.Event) (any, bool) {
	if ev == nil {
		return nil, false
	}
	if ev.Output != nil {
		return ev.Output, true
	}
	if ev.Actions.StateDelta != nil {
		if out, ok := ev.Actions.StateDelta["output"]; ok {
			return out, true
		}
	}
	return nil, false
}
