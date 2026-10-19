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

package agent

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/authconsent"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

func newCredentialInvocationContext(t *testing.T) InvocationContext {
	t.Helper()
	return &invocationContext{Context: t.Context(), agent: &agent{name: "credential-test"}}
}

func newCredentialToolContext(functionCallID string) *commonContext {
	return &commonContext{
		functionCallID: functionCallID,
		actions:        &session.EventActions{StateDelta: map[string]any{}},
	}
}

func TestRequestCredentialRecordsTheRequest(t *testing.T) {
	c := newCredentialToolContext("call-1")
	cfg := authconsent.OAuth2Consent("https://consent.example/auth", "n", "k")

	if err := c.RequestCredential(cfg); err != nil {
		t.Fatalf("RequestCredential() error = %v, want nil", err)
	}
	if diff := cmp.Diff(map[string]authconsent.AuthConfig{"call-1": cfg}, c.actions.RequestedCredentials); diff != "" {
		t.Errorf("RequestedCredentials diff (-want +got):\n%s", diff)
	}
	if !c.actions.SkipSummarization {
		t.Error("SkipSummarization = false, want true. Without it the agent loop runs on past the tool call being paused for consent.")
	}
}

// TestRequestCredentialWithoutFunctionCallID pins the callback-context case:
// there is no call to key the request on, so it must report that rather than
// silently recording nothing.
func TestRequestCredentialWithoutFunctionCallID(t *testing.T) {
	c := newCredentialToolContext("")

	if err := c.RequestCredential(authconsent.OAuth2Consent("https://consent.example/auth", "", "")); err == nil {
		t.Fatal("RequestCredential() error = nil, want an error when there is no function call id")
	}
	if len(c.actions.RequestedCredentials) != 0 {
		t.Errorf("RequestedCredentials = %v, want nothing recorded", c.actions.RequestedCredentials)
	}
	if c.actions.SkipSummarization {
		t.Error("SkipSummarization = true, want false: a request that failed must not pause the run")
	}
}

func TestAuthResponseDefaultsToNil(t *testing.T) {
	if got := newCredentialToolContext("call-1").AuthResponse(); got != nil {
		t.Errorf("AuthResponse() = %v, want nil on a fresh tool context", got)
	}
}

// TestWithDeltaThreadsCredentialResponse pins the resume seam. The flow has no
// other way to hand a tool its consent response, because NewToolContext's
// signature is public API and cannot grow a parameter.
func TestWithDeltaThreadsCredentialResponse(t *testing.T) {
	c := newCredentialToolContext("call-1")
	want := &authconsent.AuthConfig{CredentialKey: "k"}

	got := c.WithDelta(&CommonContextDelta{CredentialResponse: want}).AuthResponse()
	if got != want {
		t.Errorf("AuthResponse() after WithDelta = %v, want %v", got, want)
	}
	if c.AuthResponse() != nil {
		t.Error("WithDelta mutated the receiver; it must return a derived context")
	}
}

func TestWithDeltaWithoutCredentialResponseKeepsTheCurrentOne(t *testing.T) {
	c := newCredentialToolContext("call-1")
	existing := &authconsent.AuthConfig{CredentialKey: "k"}
	resumed := c.WithDelta(&CommonContextDelta{CredentialResponse: existing})

	branch := "other"
	if got := resumed.WithDelta(&CommonContextDelta{Path: &branch}).AuthResponse(); got != existing {
		t.Errorf("AuthResponse() after an unrelated delta = %v, want it preserved as %v", got, existing)
	}
}

// TestNewToolContextDoesNotInheritCredentialResponse pins that a tool context
// built from a resumed one starts clean.
//
// NewToolContext copies every field of the context it is given and then resets
// the per-call ones. AuthResponse is exactly the signal a tool reads as "I was
// resumed, proceed", so inheriting it would let an unrelated call skip the
// consent it needs.
func TestNewToolContextDoesNotInheritCredentialResponse(t *testing.T) {
	ic := newCredentialInvocationContext(t)
	// Reach the *commonContext the wrapper holds: that is the branch where
	// NewToolContext copies every field of the context it is given, and so the
	// only branch where a stale consent response can be inherited.
	wrapper := NewToolContext(ic, "call-1", nil, nil).(*toolContextWrapper)
	resumed := wrapper.context.WithDelta(&CommonContextDelta{
		CredentialResponse: &authconsent.AuthConfig{CredentialKey: "k"},
	})
	if _, ok := resumed.(*commonContext); !ok {
		t.Fatalf("setup: resumed context is %T, want *commonContext", resumed)
	}
	if resumed.AuthResponse() == nil {
		t.Fatal("setup: the resumed context has no consent response")
	}

	nested := NewToolContext(resumed, "call-2", nil, nil)
	if got := nested.AuthResponse(); got != nil {
		t.Errorf("AuthResponse() on a fresh tool context = %v, want nil. It was inherited from the "+
			"context it was built from, so an unrelated call reads it as its own consent.", got)
	}
}

// TestWithCredentialResponseKeepsTheContextImplementation pins that a resumed
// tool sees the same kind of context its first call saw.
//
// WithDelta on a tool context returns the inner context, dropping the
// tool-context wrapper. A tool resumed through a bare delta would find a live
// Session, Agent and EndInvocation where its first call found the wrapper's
// deliberate refusals, so the same tool would take different branches depending
// on whether it had been through consent.
func TestWithCredentialResponseKeepsTheContextImplementation(t *testing.T) {
	ic := newCredentialInvocationContext(t)
	fresh := NewToolContext(ic, "call-1", nil, nil)

	resumed := WithCredentialResponse(fresh, &authconsent.AuthConfig{CredentialKey: "k"})

	if resumed.AuthResponse() == nil {
		t.Fatal("AuthResponse() = nil, want the consent response threaded in")
	}
	if got, want := fmt.Sprintf("%T", resumed), fmt.Sprintf("%T", fresh); got != want {
		t.Errorf("resumed context is %s, want %s — the same implementation the tool's first call saw", got, want)
	}
	// The observable consequence, asserted directly rather than through the type.
	if fresh.Session() == nil && resumed.Session() != nil {
		t.Error("the resumed context exposes a live Session where the first call got none, " +
			"so a tool branching on Session() behaves differently after consent")
	}
}

// TestWithCredentialResponseNilIsIdentity pins the no-op case: a first call has
// no consent response and must not be re-derived.
func TestWithCredentialResponseNilIsIdentity(t *testing.T) {
	ic := newCredentialInvocationContext(t)
	fresh := NewToolContext(ic, "call-1", nil, nil)
	if got := WithCredentialResponse(fresh, nil); got != fresh {
		t.Errorf("WithCredentialResponse(ctx, nil) = %p, want the original %p", got, fresh)
	}
}

// TestRequestCredentialRefusesASecondRoundTrip pins that a tool cannot silently
// pause a run that nothing will resume.
//
// The consent event is generated on the model-response path only, so a request
// recorded during a resume reaches no client. Returning nil there would tell the
// tool its request was enqueued when it was not, and the run would end on the
// tool's error with no prompt.
func TestRequestCredentialRefusesASecondRoundTrip(t *testing.T) {
	c := newCredentialToolContext("call-1")
	resumed := c.WithDelta(&CommonContextDelta{CredentialResponse: &authconsent.AuthConfig{CredentialKey: "k"}})

	err := resumed.RequestCredential(authconsent.OAuth2Consent("https://consent.example/more-scope", "", ""))
	if err == nil {
		t.Fatal("RequestCredential() on a resumed call = nil, want an error")
	}
	if got := resumed.Actions().RequestedCredentials; len(got) != 0 {
		t.Errorf("RequestedCredentials = %v, want nothing recorded", got)
	}
	if resumed.Actions().SkipSummarization {
		t.Error("SkipSummarization = true, want false: a refused request must not pause the run")
	}
}

// TestOneHumanRoundTripPerCall pins that a tool call cannot start a second
// human-in-the-loop round-trip. The request event is emitted only after the
// model's own calls, so a request raised on a resume never reaches a client,
// and two raised on one call leave the reply unable to resume it. Each row is
// a state a tool can actually reach; every one must refuse and record nothing.
func TestOneHumanRoundTripPerCall(t *testing.T) {
	consent := authconsent.OAuth2Consent("https://consent.example/auth", "", "")
	tests := []struct {
		name  string
		setup func(c *commonContext)
		ask   func(c *commonContext) error
	}{
		{
			name:  "credential after a confirmation resume",
			setup: func(c *commonContext) { c.toolConfirmation = &toolconfirmation.ToolConfirmation{Confirmed: true} },
			ask:   func(c *commonContext) error { return c.RequestCredential(consent) },
		},
		{
			name:  "credential after requesting confirmation on the same call",
			setup: func(c *commonContext) { _ = c.RequestConfirmation("approve?", nil) },
			ask:   func(c *commonContext) error { return c.RequestCredential(consent) },
		},
		{
			name:  "confirmation after a consent resume",
			setup: func(c *commonContext) { c.credentialResponse = &authconsent.AuthConfig{} },
			ask:   func(c *commonContext) error { return c.RequestConfirmation("approve?", nil) },
		},
		{
			name:  "confirmation after requesting credential on the same call",
			setup: func(c *commonContext) { _ = c.RequestCredential(consent) },
			ask:   func(c *commonContext) error { return c.RequestConfirmation("approve?", nil) },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := newCredentialToolContext("call-1")
			tt.setup(c)
			beforeCreds := len(c.actions.RequestedCredentials)
			beforeConfs := len(c.actions.RequestedToolConfirmations)

			if err := tt.ask(c); err == nil {
				t.Fatal("second round-trip request = nil error, want a refusal")
			}
			if len(c.actions.RequestedCredentials) != beforeCreds || len(c.actions.RequestedToolConfirmations) != beforeConfs {
				t.Error("the refused request was recorded anyway")
			}
		})
	}
}

// TestRepeatedConfirmationStillAllowed is the control: the new rule is about
// mixing the two kinds, and must not change what RequestConfirmation did before.
func TestRepeatedConfirmationStillAllowed(t *testing.T) {
	c := newCredentialToolContext("call-1")
	c.toolConfirmation = &toolconfirmation.ToolConfirmation{Confirmed: true}
	if err := c.RequestConfirmation("approve again?", nil); err != nil {
		t.Errorf("RequestConfirmation() on a confirmation resume = %v, want nil as before this change", err)
	}
}
