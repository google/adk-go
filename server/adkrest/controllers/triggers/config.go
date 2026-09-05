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
	// OIDC, when non-nil, requires every request to carry a Google-signed
	// OIDC bearer token before the agent is invoked. Nil leaves the endpoint
	// exactly as it behaved before this field existed.
	//
	// A pointer to a struct rather than the two settings inline, because a
	// []string field on TriggerConfig would make it non-comparable and so
	// stop `cfg1 == cfg2` compiling for downstream callers.
	//
	// This endpoint accepts arbitrary attacker-controlled content as agent
	// input and otherwise has no authentication of its own: on platforms
	// that don't already gate the endpoint with their own IAM check (or if
	// that gate is ever misconfigured), leaving this nil means anyone who
	// can reach the endpoint can trigger arbitrary agent runs.
	OIDC *OIDCConfig
}

// OIDCConfig verifies the Google-signed OIDC bearer token that Pub/Sub push
// subscriptions and Eventarc triggers attach when they are configured with a
// service account.
//
// The controller constructors copy this struct and its slice, so changing
// either after construction does not change what a running handler enforces.
type OIDCConfig struct {
	// ExpectedAudience is the audience claim a token must carry, matched
	// exactly. Requests without a valid Google-signed token for it are
	// rejected with 401. Required: a non-nil OIDCConfig with an empty
	// audience is a configuration error rather than a way to disable
	// verification, and is rejected by the WithConfig constructors.
	//
	// This alone does not identify the caller. idtoken.Validate checks the
	// Google signature, the expiry and the audience string, and nothing else:
	// the audience is chosen freely by whoever mints the token, so any
	// principal that can call iam.serviceAccounts.getOpenIdToken on a service
	// account of its own can obtain a Google-signed token for this audience.
	// Set AllowedServiceAccounts to pin the check to the specific identities
	// your trigger actually delivers as.
	ExpectedAudience string
	// AllowedServiceAccounts, when non-empty, additionally requires the
	// verified token to carry a verified email claim matching one of these
	// service account addresses, the identity the Pub/Sub push subscription
	// or Eventarc trigger was configured with. Requests presenting a valid
	// but unlisted principal are rejected with 403.
	//
	// Tokens minted without an email claim (getOpenIdToken defaults
	// includeEmail to false) are rejected, so the subscription must be
	// configured to include it.
	AllowedServiceAccounts []string
}
