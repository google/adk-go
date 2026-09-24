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
	"context"
	"sync"
	"testing"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
)

// TestIdentityReadsRaceTheInvocationMutators is a tripwire, not a detector.
//
// It reports nothing today and is meant to: the fields the readers read and the
// fields the writers write are disjoint, so there is no race for ThreadSanitizer
// to find. Read the green -race gate accordingly — it does not certify this path,
// it only says the path has not yet been broken.
//
// Until this test, nothing here started a goroutine at all, so a green -race run
// said nothing whatever about either of the two identity-answering Value methods
// in this package — it covered agent's commonContext and nothing else.
//
// What it pins is the reads-versus-writes question, which is currently settled
// by field layout alone: EndInvocation and SetLiveSessionResumptionHandle write
// params fields that happen to be disjoint from the params.Session the identity
// path reads, and nothing says they must stay disjoint. Move the session behind
// either writer, or have one rewrite params wholesale, and this turns red under
// -race where nothing else in the repository would. Measured: rewriting params
// wholesale in EndInvocation is caught on 10 runs out of 10.
//
// The identity is asserted as well as read, so the test still discriminates
// without -race: it belongs to the invocation's own session throughout,
// whatever the mutators are doing.
func TestIdentityReadsRaceTheInvocationMutators(t *testing.T) {
	const (
		readers = 32
		rounds  = 200
	)
	want := agent.Identity{UserID: "mapping-user", AppName: "mapping-app", SessionID: "mapping-sid"}

	ic := icontext.NewInvocationContext(t.Context(),
		icontext.InvocationContextParams{Session: mappingSession{}})

	// Both identity-answering Value methods in this package. ReadonlyContext
	// routes through agent.Promote, so it exercises a different path to the same
	// session.
	targets := []struct {
		name string
		ctx  context.Context
	}{
		{"InvocationContext", ic},
		{"ReadonlyContext", icontext.NewReadonlyContext(ic)},
		// Behind a non-ADK wrapper, which is the shape a transport actually holds.
		{"behind a plain wrapper", context.WithValue(ic, wrapKey{}, "x")},
	}

	var wg sync.WaitGroup
	bad := make(chan string, readers*len(targets))

	for _, tc := range targets {
		for range readers {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for range rounds {
					id, ok := agent.IdentityFromContext(tc.ctx)
					if !ok || id != want {
						select {
						case bad <- tc.name + ": got " + id.UserID + "/" + id.AppName + "/" + id.SessionID:
						default:
						}
						return
					}
				}
			}()
		}
	}

	// The writers the layout argument rests on, running throughout.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for range rounds {
			ic.EndInvocation()
		}
	}()
	if setter, ok := ic.(interface{ SetLiveSessionResumptionHandle(string) }); ok {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range rounds {
				setter.SetLiveSessionResumptionHandle(string(rune('a' + i%26)))
			}
		}()
	} else {
		t.Error("InvocationContext no longer has SetLiveSessionResumptionHandle; this test " +
			"exists to race the identity read against the params writers, so find the new ones")
	}

	wg.Wait()
	close(bad)
	for msg := range bad {
		t.Errorf("a concurrent read saw the wrong identity — %s, want %v", msg, want)
	}
}
