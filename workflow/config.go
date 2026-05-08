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

import "time"

// defaultRetryConfig is the default retry configuration for a node.
var defaultRetryConfig = RetryConfig{
	MaxAttempts:   5,
	InitialDelay:  time.Second,
	MaxDelay:      60 * time.Second,
	BackoffFactor: 2.0,
	Jitter:        1.0,
	ShouldRetry: func(error) bool {
		return true
	},
}

// DefaultRetryConfig returns a copy of the default retry policy
// (5 attempts, 1s initial delay, 60s cap, 2x backoff, full jitter,
// retry every error). Override fields on the returned value to
// customise:
//
//	rc := workflow.DefaultRetryConfig()
//	rc.MaxAttempts = 10
//	cfg := workflow.NodeConfig{RetryConfig: rc}
func DefaultRetryConfig() *RetryConfig {
	cfg := defaultRetryConfig
	return &cfg
}

// NodeConfig defines the configuration for a node.
//
// All fields are optional. The pointer-typed fields (RerunOnResume,
// WaitForOutput) are tri-state: nil means "use the engine default for
// this node kind", which mirrors Python's per-node-type defaults
// (e.g. AgentNode in task mode defaults WaitForOutput to true while
// other nodes default to false). Use the *Or accessor helpers to
// read these values with an explicit per-call-site default.
type NodeConfig struct {
	// ParallelWorker, when true, runs the node concurrently for each
	// item of a list-typed input. The engine collects per-item
	// outputs and emits a single aggregate output event.
	ParallelWorker bool

	// RerunOnResume controls human-in-the-loop resume behaviour. When
	// true, an interrupted node re-runs from scratch on resume; when
	// false, the resume payload is treated as the node's output.
	// nil means "use the engine default", which is true for AgentNode
	// and false elsewhere.
	RerunOnResume *bool

	// WaitForOutput, when true, keeps the node in NodeWaiting
	// (re-triggerable) until it actually yields an event carrying an
	// "output" key in Actions.StateDelta, instead of moving it to
	// NodeCompleted on first return. JoinNode and any custom fan-in
	// node sets this. nil means "use the engine default" — false for
	// most node kinds.
	WaitForOutput *bool

	// RetryConfig, when non-nil, makes the scheduler retry this node
	// on failure per the policy. nil means "no retries".
	RetryConfig *RetryConfig

	// Timeout, when > 0, bounds a single activation of the node via
	// context.WithTimeout on the per-node context. Zero (the default)
	// means the node is bounded only by the parent invocation
	// context's deadline, if any.
	Timeout time.Duration
}

// RetryConfig defines the parameters for retrying a failed node.
//
// Recommended construction is via DefaultRetryConfig and override
// the fields you want to customize:
//
//	rc := workflow.DefaultRetryConfig()
//	rc.MaxAttempts = 10
//	cfg := workflow.NodeConfig{RetryConfig: rc}
//
// Constructing via struct literal (RetryConfig{...}) is permitted
// but discouraged: any unset field defaults to its zero value, not
// to DefaultRetryConfig's value. The zero RetryConfig is a valid
// "no retry, no backoff, no jitter" policy.
//
// The struct shape is deliberately a flat set of plain values: no
// dependency on cenkalti/backoff/v5. The scheduler's retry handler
// implements the backoff math in workflow/retry.go.
type RetryConfig struct {
	// Maximum number of attempts, including the original request. If 0 or 1, it means no retries. If not specified, default to 5.
	MaxAttempts int

	// Initial delay before the first retry, in fractions of a second. If not specified, default to 1 second.
	InitialDelay time.Duration

	// Maximum delay between retries, in fractions of a second. If not specified, default to 60 seconds.
	MaxDelay time.Duration

	// BackoffFactor is the per-attempt multiplier applied to the
	// delay. A factor of 1.0 means a constant InitialDelay between
	// retries; 2.0 means classic exponential backoff. Values < 1.0
	// shrink the delay each attempt (rare but permitted).
	BackoffFactor float64

	// Jitter is a randomness factor in [0.0, 1.0]. The actual delay
	// is sampled from delay * (1 ± Jitter). Zero means deterministic
	// delays.
	Jitter float64

	// Predicate that defines when to retry (true means retry). If not specified, default to true.
	ShouldRetry func(error) bool
}
