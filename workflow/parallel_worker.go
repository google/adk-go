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
	"fmt"
	"iter"
	"reflect"
	"sync"

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
func NewParallelWorker(name string, wrapped Node, maxConcurrency int, cfg NodeConfig) *ParallelWorker {
	return &ParallelWorker{
		BaseNode:       BaseNode{name: name, config: cfg},
		wrapped:        wrapped,
		maxConcurrency: maxConcurrency,
	}
}

// Run executes the wrapped node in parallel for each item in the input list.
// It aggregates the "output" from each wrapped node execution into a list and
// yields a single final event with the aggregated list as output.
// Non-output events emitted by the wrapped node are yielded immediately.
func (n *ParallelWorker) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		v := reflect.ValueOf(input)
		if v.Kind() != reflect.Slice {
			yield(nil, fmt.Errorf("ParallelWorker %s expects a slice input, got %T", n.Name(), input))
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

		type result struct {
			index int
			ev    *session.Event
			err   error
		}
		resCh := make(chan result, nItems)

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

			go func(idx int, it any) {
				defer wg.Done()
				defer func() {
					if sem != nil {
						<-sem
					}
				}()

				var workerOutputs []any
				hasOutput := false

				for ev, err := range n.wrapped.Run(ctx, it) {
					if err != nil {
						select {
						case resCh <- result{index: idx, err: err}:
						case <-ctx.Done():
						}
						return
					}

					if ev != nil && ev.Actions.StateDelta != nil {
						if out, ok := ev.Actions.StateDelta["output"]; ok {
							workerOutputs = append(workerOutputs, out)
							hasOutput = true
						}
					}
				}

				var finalEv *session.Event
				if hasOutput {
					var output any
					if len(workerOutputs) == 1 {
						output = workerOutputs[0]
					} else {
						output = workerOutputs
					}
					finalEv = &session.Event{Actions: session.EventActions{StateDelta: map[string]any{"output": output}}}
				}

				select {
				case resCh <- result{index: idx, ev: finalEv}:
				case <-ctx.Done():
				}
			}(i, item)
		}

		// Goroutine to close channel when all workers are done
		go func() {
			wg.Wait()
			close(resCh)
		}()

		var firstErr error

		for res := range resCh {
			if res.err != nil && firstErr == nil {
				firstErr = res.err
				// We could cancel the context here to stop other workers,
				// but we don't own the context. The scheduler will cancel it
				// when we return an error.
			}
			if res.err != nil {
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
