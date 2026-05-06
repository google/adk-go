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

// ptr returns a pointer to the value passed in.
func ptr[T any](v T) *T {
	return &v
}

// defaultRetryConfig is the default retry configuration for a node.
var defaultRetryConfig = RetryConfig{
	MaxAttempts:   ptr(5),
	InitialDelay:  ptr(time.Second),
	MaxDelay:      ptr(60 * time.Second),
	BackoffFactor: ptr(2.0),
	Jitter:        ptr(1.0),
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
type RetryConfig struct {
	// Maximum number of attempts, including the original request. If 0 or 1, it means no retries. If not specified, default to 5.
	MaxAttempts *int
	// Initial delay before the first retry, in fractions of a second. If not specified, default to 1 second.
	InitialDelay *time.Duration
	// Maximum delay between retries, in fractions of a second. If not specified, default to 60 seconds.
	MaxDelay *time.Duration
	// Multiplier by which the delay increases after each attempt. If not specified, default to 2.0.
	BackoffFactor *float64
	// Randomness factor for the delay. Use 0.0 to remove randomness. If not specified, default to 1.0.
	Jitter *float64
	// Predicate that defines when to retry (true means retry). If not specified, default to true.
	ShouldRetry func(error) bool
}

func (r *RetryConfig) applyDefaults() {
	defaults := DefaultRetryConfig()
	if r.MaxAttempts == nil {
		r.MaxAttempts = defaults.MaxAttempts
	}
	if r.InitialDelay == nil {
		r.InitialDelay = defaults.InitialDelay
	}
	if r.MaxDelay == nil {
		r.MaxDelay = defaults.MaxDelay
	}
	if r.BackoffFactor == nil {
		r.BackoffFactor = defaults.BackoffFactor
	}
	if r.Jitter == nil {
		r.Jitter = defaults.Jitter
	}
	if r.ShouldRetry == nil {
		r.ShouldRetry = defaults.ShouldRetry
	}
}
