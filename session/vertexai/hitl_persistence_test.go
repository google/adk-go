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

package vertexai

import (
	"testing"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/authconsent"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// A pending human-in-the-loop request lives only on EventActions: the consent
// request in RequestedCredentials, the confirmation in
// RequestedToolConfirmations. The structured actions column carries neither, so
// raw_event is the only slot that keeps them. Without it a session reloaded
// from Vertex has no record that the call is paused, and compaction then treats
// the paused call as answered at its first response and can cut between it and
// the request that unblocks it.
func TestEventNeedsRawEventForPendingHITLRequests(t *testing.T) {
	t.Parallel()

	consent := &session.Event{ID: "paused", Author: "agent", InvocationID: "inv1"}
	consent.Actions.RequestedCredentials = map[string]authconsent.AuthConfig{
		"call-1": authconsent.OAuth2Consent("https://consent.example/auth", "n", "k"),
	}
	if !eventNeedsRawEvent(consent) {
		t.Error("eventNeedsRawEvent() = false for an event with a pending consent request, so the request would be lost on reload")
	}

	confirmation := &session.Event{ID: "paused", Author: "agent", InvocationID: "inv1"}
	confirmation.Actions.RequestedToolConfirmations = map[string]toolconfirmation.ToolConfirmation{
		"call-1": {Hint: "approve?"},
	}
	if !eventNeedsRawEvent(confirmation) {
		t.Error("eventNeedsRawEvent() = false for an event with a pending confirmation, so the request would be lost on reload")
	}
}
