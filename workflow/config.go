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

func DefaultRetryConfig() *RetryConfig {
	cfg := defaultRetryConfig
	return &cfg
}

// NodeConfig defines the configuration for a node.
type NodeConfig struct {
	// Enables data parallelism (runs node concurrently for each item in input collection)
	ParallelWorker bool
	// Re-runs node on resume. Defaults to true for AgentNode
	RerunOnResume *bool
	// Wait for output before triggering edges. Defaults to true for Task agents
	WaitForOutput *bool
	// Retry configuration on failure
	RetryConfig *RetryConfig
	// Max duration for node to complete. Optional for global defaults
	Timeout *time.Duration
}

// RetryConfig defines the parameters for retrying a failed node.
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
type RetryConfig struct {
	// Maximum number of attempts, including the original request. If 0 or 1, it means no retries. If not specified, default to 5.
	MaxAttempts int
	// Initial delay before the first retry, in fractions of a second. If not specified, default to 1 second.
	InitialDelay time.Duration
	// Maximum delay between retries, in fractions of a second. If not specified, default to 60 seconds.
	MaxDelay time.Duration
	// Multiplier by which the delay increases after each attempt. If not specified, default to 2.0.
	BackoffFactor float64
	// Randomness factor for the delay. Use 0.0 to remove randomness. If not specified, default to 1.0.
	Jitter float64
	// Predicate that defines when to retry (true means retry). If not specified, default to true.
	ShouldRetry func(error) bool
}
