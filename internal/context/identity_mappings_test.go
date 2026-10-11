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

package context_test

import (
	"reflect"
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

// mappingSession answers every accessor with a distinct non-empty value, so a
// field read off the wrong accessor, or left unset, is visible.
type mappingSession struct{ session.Session }

func (mappingSession) ID() string                { return "mapping-sid" }
func (mappingSession) AppName() string           { return "mapping-app" }
func (mappingSession) UserID() string            { return "mapping-user" }
func (mappingSession) State() session.State      { return nil }
func (mappingSession) Events() session.Events    { return nil }
func (mappingSession) LastUpdateTime() time.Time { return time.Time{} }

// enclosingSession is what the invocation the decorator embeds carries, distinct
// from mappingSession at every accessor.
type enclosingSession struct{ session.Session }

func (enclosingSession) ID() string                { return "enclosing-sid" }
func (enclosingSession) AppName() string           { return "enclosing-app" }
func (enclosingSession) UserID() string            { return "enclosing-user" }
func (enclosingSession) State() session.State      { return nil }
func (enclosingSession) Events() session.Events    { return nil }
func (enclosingSession) LastUpdateTime() time.Time { return time.Time{} }

// readByAgent is an agent.InvocationContext written outside package agent, so it
// carries no identity marker and the procedure READS its session — which is the
// arm that runs agent.identityOf. Reaching that mapping needs a type agent does
// not own, which is why this fixture is here rather than a constructor call.
type readByAgent struct {
	agent.InvocationContext
	own session.Session
}

func (r readByAgent) Session() session.Session { return r.own }

// TestIdentityMappingsAgree pins the two places a session becomes an
// agent.Identity against each other.
//
// agent.identityOf and (*icontext.InvocationContext).Value build the same struct
// from the same three accessors, and they are separate literals because
// internal/context cannot import package agent's unexported helper without a
// cycle. Nothing about a duplicated literal keeps it in step: add a field to
// agent.Identity, populate it in package agent alone, and every other test in
// both packages stays green while the two mappings silently disagree.
//
// The field sweep is the second half. Equality alone passes when a new field is
// populated by NEITHER mapping, which leaves it permanently zero for every
// caller — so each field is also required to be set.
//
// It sweeps string fields only. Requiring every field to be non-zero would fire
// on a field whose correct value here happens to be the zero one, a bool most
// obviously, and that false alarm costs more than the case it would catch. A
// non-string field added later is therefore not covered by the sweep and wants a
// deliberate decision at this line.
func TestIdentityMappingsAgree(t *testing.T) {
	s := mappingSession{}

	// The internal/context mapping: this type owns its session and answers the
	// key from its own copy of the literal.
	fromInternal, ok := agent.IdentityFromContext(
		icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: s}))
	if !ok {
		t.Fatal("IdentityFromContext() over icontext.InvocationContext: ok = false, want true")
	}

	// The package agent mapping: an invocation agent does not own, so the
	// procedure reads its Session() through agent.identityOf.
	// A DIFFERENT session on the enclosing invocation, so this arm fails on its own
	// terms if the procedure ever reads the enclosing session instead of the
	// decorator's. With both set to s the test would compare one mapping with itself.
	base := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: enclosingSession{}})
	fromAgent, ok := agent.IdentityFromContext(
		agent.Promote(readByAgent{InvocationContext: base, own: s}))
	if !ok {
		t.Fatal("IdentityFromContext() over an invocation agent reads: ok = false, want true")
	}

	if fromAgent != fromInternal {
		t.Errorf("the two session-to-Identity mappings disagree:\n  package agent:    %+v\n  internal/context: %+v\n"+
			"They are separate literals held together by nothing but this test. Update both.",
			fromAgent, fromInternal)
	}

	v := reflect.ValueOf(fromAgent)
	for i := range v.NumField() {
		if v.Field(i).Kind() == reflect.String && v.Field(i).IsZero() {
			t.Errorf("Identity.%s is zero for a session that sets every accessor, so no mapping "+
				"populates it. Either both mappings must set it, or this fixture needs a value for it.",
				v.Type().Field(i).Name)
		}
	}
}
