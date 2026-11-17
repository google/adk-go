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
	"strconv"
	"sync"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// ParallelWorker runs a wrapped node in parallel for each item in the input list.
type ParallelWorker struct {
	BaseNode
	wrapped        Node
	maxConcurrency int
	retryCfg       *RetryConfig
}

// NewParallelWorker creates a new ParallelWorker node.
// maxConcurrency <= 0 means no limit on concurrency.
func NewParallelWorker(name string, wrapped Node, maxConcurrency int, cfg NodeConfig) (*ParallelWorker, error) {
	if wrapped.Config().RetryConfig != nil {
		return nil, fmt.Errorf("ParallelWorker %s: wrapped node %s cannot have RetryConfig", name, wrapped.Name())
	}
	retryCfg := cfg.RetryConfig
	cfg.RetryConfig = nil // Hide from scheduler, so it does not try to retry the node itself.

	return &ParallelWorker{
		BaseNode:       BaseNode{name: name, config: cfg},
		wrapped:        wrapped,
		maxConcurrency: maxConcurrency,
		retryCfg:       retryCfg,
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
// When several items fail at about the same time, which of their errors is
// returned is not specified.
//
// If ctx is done by the time the workers finish and no item has failed, Run
// returns ctx.Err() and yields no output, even if every item succeeded. An
// empty input starts no workers and always yields an empty list.
//
// A wrapped node whose context is done is not started again, and its yield
// returns false at its next event. No item is started once ctx is done.
// Without a concurrency cap every other item is started, even when a sibling
// has already failed; a failed sibling stops the started items through their
// canceled context.
//
// In case the wrapped node produces more then one output event, they will be
// aggregated into a list, and the final result will be a multi dimensional list.
//
// Intermediate non-output events emitted by the wrapped node are suppressed,
// but their state and artifact deltas are carried. Each item attempt checks its
// state writes and the state deltas of its events against the workflow's state
// schema, as a node does; a refused write fails the item like any other error.
// The deltas of every item are merged in item order, so a later item wins a key
// that two items wrote. On success they ride on the aggregated output event. On
// failure they go out on one event that carries only them, before the error,
// unless ctx is done.
func (n *ParallelWorker) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		cancelCtx, cancelFunc := ctx.WithAgentCancel()
		defer cancelFunc()
		workerCtx := ctx.WithAgentContext(cancelCtx)

		v := reflect.ValueOf(input)
		if v.Kind() != reflect.Slice {
			yield(nil, fmt.Errorf("parallel worker %s expects a slice input, got %T", n.Name(), input))
			return
		}

		nItems := v.Len()
		if nItems == 0 {
			// Yield an empty list as output
			event := session.NewEvent(ctx, ctx.InvocationID())
			event.Output = []any{}
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

		// reportFailure cancels the remaining workers. runWorker calls it
		// synchronously, before releasing its semaphore slot (see below).
		var reportOnce sync.Once
		var firstErr error
		reportFailure := func(err error) {
			reportOnce.Do(func() {
				firstErr = err
				cancelFunc()
			})
		}

		// Branch isolation: derive a per-item sub-branch so each
		// worker's wrapped node sees an isolated event history (the
		// LLM flow's history filter scopes by branch prefix). Items
		// 0..N-1 receive sub-branches name@1..name@N.
		parentBranch := workerCtx.Branch()
		wrappedName := n.wrapped.Name()

		for i := range nItems {
			item := v.Index(i).Interface()

			if sem != nil {
				select {
				case sem <- struct{}{}:
				case <-workerCtx.Done():
					wg.Done()
					continue
				}
				// Re-check with a default case: a plain two-case select
				// would pick randomly if cancellation raced the send above.
				select {
				case <-workerCtx.Done():
					<-sem
					wg.Done()
					continue
				default:
				}
			} else if ctx.Err() != nil {
				// Check the run's own context, not workerCtx: a failed
				// sibling cancels workerCtx, and without a cap the other
				// items still start.
				wg.Done()
				continue
			}

			itemBranch := deriveSubBranch(parentBranch, wrappedName+"@"+strconv.Itoa(i+1))

			itemCtx := workerCtx.WithDelta(&agent.CommonContextDelta{InvocationContextDelta: &agent.InvocationContextDelta{Branch: &itemBranch}})
			go n.runWorker(itemCtx, i, item, sem, resCh, &wg, reportFailure)
		}

		// Goroutine to close channel when all workers are done
		go func() {
			wg.Wait()
			close(resCh)
		}()

		deltas := make([]*session.EventActions, nItems)
		for res := range resCh {
			deltas[res.index] = res.delta
			if res.ev != nil {
				if out, ok := extractOutput(res.ev); ok {
					outputs[res.index] = out
				}
			}
		}

		// Items skipped at dispatch, and workers that return without an error
		// once ctx is done, report no failure, so firstErr stays nil. Without
		// this check Run would yield a partial or post-cancellation result as
		// a success.
		if firstErr == nil && ctx.Err() != nil {
			firstErr = ctx.Err()
		}
		itemWrites := &session.EventActions{}
		for _, d := range deltas {
			mergeDelta(itemWrites, d)
		}
		if firstErr != nil {
			// The item writes outlive the worker's failure, whatever its
			// cause. They stay out of the worker's own pending writes, which
			// a failure caused by a dynamic child discards, and the event
			// that carries them is registered as checked.
			if ctx.Err() == nil && (len(itemWrites.StateDelta) > 0 || len(itemWrites.ArtifactDelta) > 0) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				mergeDelta(&ev.Actions, itemWrites)
				forwardedFrom(ctx).mark(ev)
				if !yield(ev, nil) {
					return
				}
			}
			yield(nil, firstErr)
			return
		}
		if actions := ctx.Actions(); actions != nil {
			mergeDelta(actions, itemWrites)
		}

		// Yield the aggregated output
		event := session.NewEvent(ctx, ctx.InvocationID())
		event.Output = outputs
		yield(event, nil)
	}
}

type workerResult struct {
	index int
	ev    *session.Event
	err   error
	// delta holds the state and artifact deltas of every attempt of the
	// item, a later attempt winning a key two attempts wrote.
	delta *session.EventActions
}

func (n *ParallelWorker) runWorker(ctx agent.Context, idx int, item any, sem chan struct{}, resCh chan<- workerResult, wg *sync.WaitGroup, reportFailure func(error)) {
	defer wg.Done()
	defer func() {
		if sem != nil {
			<-sem
		}
	}()

	retryCfg := n.retryCfg
	failedAttempts := 0
	itemDelta := &session.EventActions{}

	for {
		workerOutputs, attemptDelta, runErr := n.runWrappedOnce(ctx, item)
		mergeDelta(itemDelta, attemptDelta)

		if runErr == nil {
			// On success populate the output event.
			resCh <- workerResult{index: idx, ev: makeWorkerOutputEvent(workerOutputs), delta: itemDelta}
			return
		}

		// Failure, check if the retry config is configured.
		// If so, follow the retry logic and repeat the execution of the wrapped node on failed input.
		failedAttempts++
		if ShouldRetry(retryCfg, runErr, failedAttempts) {
			delay := CalculateDelay(retryCfg, failedAttempts)
			select {
			case <-time.After(delay):
			case <-ctx.Done():
			}
			// A two-case select picks at random when the delay has elapsed and ctx
			// is done, so decide on ctx itself: a canceled item must not start the
			// wrapped node again.
			if err := ctx.Err(); err != nil {
				// Cancellation while parked in the retry delay is a real
				// failure too: report it before releasing sem, exactly
				// like the exhausted-attempts path below. Without this,
				// firstErr stays nil unless some other worker happens to
				// report first, so a cancellation landing here can be
				// silently dropped and the whole run yields success with
				// a nil output for this item.
				reportFailure(err)
				resCh <- workerResult{index: idx, err: err, delta: itemDelta}
				return
			}
			continue
		}

		// Cannot retry or exhausted attempts: report before releasing sem.
		reportFailure(runErr)
		resCh <- workerResult{index: idx, err: runErr, delta: itemDelta}
		return
	}
}

// runWrappedOnce runs the wrapped node once for item under its own
// invoke_node span, so each item and each retry attempt is a distinct
// span. Kept a separate function so the deferred span.end() fires per
// attempt instead of piling up across runWorker's retry loop.
//
// Each attempt is an activation of its own, with its own pending writes.
// delta collects the state and artifact deltas of the attempt's events and
// the writes the attempt has not put on an event; a canceled attempt keeps
// only the deltas of its events.
func (n *ParallelWorker) runWrappedOnce(ctx agent.Context, item any) (outputs []any, delta *session.EventActions, err error) {
	ctx, writes := beginActivation(ctx, n.wrapped)
	span, ctx := startNodeSpan(ctx, n.wrapped)
	defer func() {
		span.recordError(err, nil)
		span.end()
	}()

	delta = &session.EventActions{}
	for ev, runErr := range n.wrapped.Run(ctx, item) {
		if runErr != nil {
			err = runErr
			break
		}
		if err = writes.admit(ev); err != nil {
			break
		}
		if ev != nil {
			mergeDelta(delta, &ev.Actions)
		}
		if out, ok := extractOutput(ev); ok {
			outputs = append(outputs, out)
		}
		if ctx.Err() != nil {
			// The wrapped node's yield returns false, so it does not go on.
			// The item reports no error of its own: Run turns a done ctx
			// into the error it yields.
			break
		}
	}
	cause := err
	if cause == nil {
		cause = ctx.Err()
	}
	if flushesPending(cause) {
		writes.drainInto(delta)
	}
	return outputs, delta, err
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
	return &session.Event{Output: output}
}

func extractOutput(ev *session.Event) (any, bool) {
	if ev == nil {
		return nil, false
	}
	if ev.Output != nil {
		return ev.Output, true
	}
	return nil, false
}
