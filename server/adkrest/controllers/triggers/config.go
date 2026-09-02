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
	// Google-signed OIDC bearer token whose audience claim equals this value
	// and whose issuer is Google; requests without one are rejected with 401
	// before the agent is invoked.
	//
	// This alone does not identify the caller. idtoken.Validate checks the
	// Google signature, the expiry and the audience string, and nothing else:
	// the audience is chosen freely by whoever mints the token, so any
	// principal that can call iam.serviceAccounts.getOpenIdToken on a service
	// account of its own can obtain a Google-signed token for this audience.
	// Set AllowedServiceAccounts to pin the check to the specific identities
	// your trigger actually delivers as.
	//
	// This endpoint accepts arbitrary attacker-controlled content as agent
	// input and otherwise has no authentication of its own: on platforms
	// that don't already gate the endpoint with their own IAM check (or if
	// that gate is ever misconfigured), leaving this unset means anyone who
	// can reach the endpoint can trigger arbitrary agent runs.
	ExpectedAudience string
	// AllowedServiceAccounts, when non-empty, additionally requires the
	// verified token to carry a verified email claim matching one of these
	// service account addresses, the identity the Pub/Sub push subscription
	// or Eventarc trigger was configured with. Requests presenting a valid
	// but unlisted principal are rejected with 403.
	//
	// Requires ExpectedAudience; it is ignored on its own. Tokens minted
	// without an email claim (getOpenIdToken defaults includeEmail to false)
	// are rejected, so the subscription must be configured to include it.
	AllowedServiceAccounts []string
}
