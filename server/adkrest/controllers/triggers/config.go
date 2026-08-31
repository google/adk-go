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

package triggers

import "time"

// TriggerConfig contains configuration options for triggers.
type TriggerConfig struct {
	// MaxRetries is the maximum number of times to retry a failed agent execution.
	MaxRetries int
	// BaseDelay is the base delay between retries.
	BaseDelay time.Duration
	// MaxDelay is the maximum delay between retries.
	MaxDelay time.Duration
	// MaxConcurrentRuns is the maximum number of concurrent runs.
	MaxConcurrentRuns int
}
