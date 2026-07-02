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

// Package guardrail provides sentinel error types for policy guardrails.
//
// Guardrail callbacks signal a policy denial by returning *ErrGuardrailBlocked.
// Runners detect this via errors.As and route to OnGuardrailBlockedCallbacks
// rather than OnToolErrorCallbacks, so policy denials are never misreported
// as unexpected runtime failures.
package guardrail

import "fmt"

// ErrGuardrailBlocked is returned by a BeforeToolCallback when an agent's
// tool call has been denied by a policy guardrail.
//
// Use errors.As to check for this type and distinguish policy denials from
// unexpected runtime errors:
//
//	var blocked *guardrail.ErrGuardrailBlocked
//	if errors.As(err, &blocked) {
//	    // policy denial — blocked.Policy and blocked.Reason are available
//	}
type ErrGuardrailBlocked struct {
	// Policy is the name of the policy or guardrail that blocked the call.
	// May be empty if the callback does not supply a policy name.
	Policy string

	// Reason is a human-readable explanation of why the call was blocked.
	Reason string
}

// Error implements the error interface.
func (e *ErrGuardrailBlocked) Error() string {
	if e.Policy != "" {
		return fmt.Sprintf("guardrail %q blocked tool call: %s", e.Policy, e.Reason)
	}
	if e.Reason != "" {
		return fmt.Sprintf("guardrail blocked tool call: %s", e.Reason)
	}
	return "guardrail blocked tool call"
}
