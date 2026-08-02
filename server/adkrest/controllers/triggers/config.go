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
	// ExpectedAudience, when non-empty, requires every request to carry a
	// Google-signed OIDC bearer token (as attached by Pub/Sub push
	// subscriptions and Eventarc triggers configured with a service account)
	// whose audience claim matches this value; requests without one are
	// rejected with 401 before the agent is invoked.
	//
	// This endpoint accepts arbitrary attacker-controlled content as agent
	// input and otherwise has no authentication of its own: on platforms
	// that don't already gate the endpoint with their own IAM check (or if
	// that gate is ever misconfigured), leaving this unset means anyone who
	// can reach the endpoint can trigger arbitrary agent runs.
	ExpectedAudience string
}
