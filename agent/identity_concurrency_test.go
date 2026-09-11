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
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/v2/session"
)

// The identity procedure is read from more than one goroutine by design:
// IdentityFromContext exists so an http.RoundTripper deep beneath a tool call can
// recover the acting user, and that runs on whatever goroutine the caller is on.
// A fanned-out agent adds the other half — several children derived from one
// parent invocation, read at the same time. Neither shape was covered.

// TestIdentityIsRaceFreeAcrossConcurrentReaders reads one invocation through
// every derivation at once.
//
// The four shapes are the ones a reader actually holds: the invocation itself, a
// promotion, a fresh context over it, and a derived context.Context that knows
// nothing about ADK. All four must report the same user, because they all speak
// for the same invocation.
func TestIdentityIsRaceFreeAcrossConcurrentReaders(t *testing.T) {
	parent := &invocationContext{Context: t.Context(), session: matrixOwner("alice")}
	want := Identity{UserID: "alice", AppName: "app", SessionID: "sid-alice"}

	const readers = 64
	var wg sync.WaitGroup
	start := make(chan struct{})
	errs := make(chan error, readers)

	for i := range readers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start // release together, so the reads genuinely overlap

			shape := i % 4
			var ctx context.Context
			switch shape {
			case 0:
				ctx = parent
			case 1:
				ctx = Promote(parent)
			case 2:
				ctx = NewContext(parent)
			case 3:
				ctx = context.WithValue(NewContext(parent), unrelatedKey{}, i)
			}

			if got, ok := IdentityFromContext(ctx); !ok || got != want {
				errs <- fmt.Errorf("reader %d (shape %d): IdentityFromContext() = %+v, %v; want %+v, true",
					i, shape, got, ok, want)
			}
		}()
	}

	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

// generationalInvocation hands out a fresh session per Session() call, every
// field of it stamped with the same generation number.
//
// It is read rather than asked, because embedding the InvocationContext
// interface promotes only that interface's methods and the marker is not one of
// them — so this takes the branch that calls Session(), which is the branch under
// test.
type generationalInvocation struct {
	InvocationContext
	calls atomic.Int64
}

func (g *generationalInvocation) Session() session.Session {
	tag := strconv.FormatInt(g.calls.Add(1), 10)
	return &matrixSession{id: "sid-" + tag, app: "app-" + tag, user: "user-" + tag}
}

// TestIdentityReadsTheSessionOnceAndNeverTears pins the reason identityOf takes
// one Session value and then reads three fields off it, rather than calling
// Session() per field.
//
// Its comment says re-reading risks a torn identity and nothing held it to that.
// A session accessor that returns a different value per call is not exotic: a
// wrapper rebuilding a view produces one, and llmagent wraps sessions already. If
// the three fields were read through three calls, an identity could name one
// user with another's session, and under concurrency it would do so rarely enough
// to survive review.
func TestIdentityReadsTheSessionOnceAndNeverTears(t *testing.T) {
	inv := &generationalInvocation{InvocationContext: &invocationContext{Context: t.Context()}}
	ctx := Promote(inv)

	if _, ok := IdentityFromContext(ctx); !ok {
		t.Fatal("IdentityFromContext() reported no identity; the test cannot say anything about tearing")
	}
	if got := inv.calls.Load(); got != 1 {
		t.Errorf("Session() called %d times for one identity read, want 1: each call yields a "+
			"different session, so reading per field lets one identity mix two of them", got)
	}

	const readers = 64
	var wg sync.WaitGroup
	start := make(chan struct{})
	errs := make(chan error, readers)

	for i := range readers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start

			got, ok := IdentityFromContext(ctx)
			if !ok {
				errs <- fmt.Errorf("reader %d: IdentityFromContext() reported no identity", i)
				return
			}
			user := strings.TrimPrefix(got.UserID, "user-")
			app := strings.TrimPrefix(got.AppName, "app-")
			sid := strings.TrimPrefix(got.SessionID, "sid-")
			if user != app || user != sid {
				errs <- fmt.Errorf("reader %d: torn identity %+v — generations user=%s app=%s session=%s",
					i, got, user, app, sid)
			}
		}()
	}

	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

// swappingSession advances a generation counter on every accessor call, under a
// mutex, so no access is a data race by construction.
//
// It gives the identity read a session that CHANGES between the three accessor
// calls it makes — which is the shape the contract's read-window clause is about
// — and it does so without a second goroutine, so nothing depends on the
// scheduler. It gives ThreadSanitizer nothing of its own to report: every access
// to the counter goes through the mutex. What TSan sees is the 64 readers below
// sharing one commonContext.
type swappingSession struct {
	session.Session
	mu  sync.RWMutex
	gen int
}

func (s *swappingSession) swap() { s.mu.Lock(); s.gen++; s.mu.Unlock() }

func (s *swappingSession) now() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.gen
}

// tag advances the generation and returns the new one, so the three accessor
// calls inside one identity read see three different sessions BY CONSTRUCTION.
//
// The churn is generated by the reader rather than by a separate writer
// goroutine on purpose. Relying on a writer to land between two accessor calls
// makes the property scheduler-dependent: measured at GOMAXPROCS=1, no read ever
// spanned a swap and the control below failed every run.
func (s *swappingSession) tag() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.gen++
	return strconv.Itoa(s.gen)
}

func (s *swappingSession) ID() string      { return "sid-" + s.tag() }
func (s *swappingSession) AppName() string { return "app-" + s.tag() }
func (s *swappingSession) UserID() string  { return "user-" + s.tag() }

// TestIdentityAgainstASessionThatChangesUnderIt pins what the identity read
// promises against a session that changes between its accessor calls, and what it
// does not.
//
// Promised: every field comes from the window the read spanned. The session
// advances once per accessor call, so every field must name a generation strictly
// after the one observed going in and no later than the one observed coming out.
// A field outside that means the identity was assembled from something other than
// this session at this moment — a stale cache, a field taken off the wrong
// session, or two fields transposed. The lower bound is the one that bites: 64
// readers share the counter, so the upper end is loose, while a cached identity
// fails the lower end on its second read.
//
// Not promised: an atomic snapshot. identityOf binds one Session value and then
// makes three accessor calls on it, so a swap between the first and the third
// yields an Identity naming two generations. Holding the value removes the tear
// across Session() calls, not the tear across field reads, and nothing that keeps
// session.Session's six-accessor shape can. The bracket tolerates that and still
// fails a transposition, which a well-formedness check does not.
func TestIdentityAgainstASessionThatChangesUnderIt(t *testing.T) {
	sess := &swappingSession{}
	parent := &invocationContext{Context: t.Context(), session: sess}
	// One shared context for every reader, unlike the per-goroutine derivations
	// above: a memo on this commonContext is written by one reader and read by 63.
	ctx := Promote(parent)

	const readers = 64
	// No writer goroutine. There was one, and it earned its place only while the
	// churn came from it. Once the session advanced on every accessor call the
	// writer stopped contributing to any assertion — measured, the test stays
	// green with it deleted — and its stated ThreadSanitizer value was already
	// false, since every access to the counter is mutex-guarded. What TSan does
	// see is 64 readers sharing one commonContext, each taking the counter's
	// exclusive lock three times per read.
	var readersDone sync.WaitGroup
	errs := make(chan error, readers)
	// Counts reads where the generation moved across the call. It is a diagnostic,
	// not an independent assertion: every early return below sends to errs first,
	// so a short count implies the test is already red. The bracket at the bottom
	// of the loop is what actually catches a cached identity.
	var spanned atomic.Int64

	for i := range readers {
		readersDone.Add(1)
		go func() {
			defer readersDone.Done()
			for range 50 {
				before := sess.now()
				got, ok := IdentityFromContext(ctx)
				after := sess.now()
				if !ok {
					errs <- fmt.Errorf("reader %d: IdentityFromContext() reported no identity", i)
					return
				}
				for _, f := range [][2]string{
					{"UserID", got.UserID}, {"AppName", got.AppName}, {"SessionID", got.SessionID},
				} {
					name, want := f[0], map[string]string{"UserID": "user", "AppName": "app", "SessionID": "sid"}[f[0]]
					prefix, tag, found := strings.Cut(f[1], "-")
					if !found || prefix != want {
						errs <- fmt.Errorf("reader %d: %s = %q, want prefix %q — a field read off the wrong accessor looks exactly like this", i, name, f[1], want)
						return
					}
					gen, err := strconv.Atoi(tag)
					if err != nil {
						errs <- fmt.Errorf("reader %d: %s = %q, generation %q is not a number", i, name, f[1], tag)
						return
					}
					// Strictly above before: every accessor advances the counter and
					// then reports the new value, so no field can name the generation
					// observed going in. At or below after, which the readers share, so
					// with 64 of them running the window is wide — the bound that bites
					// is the lower one, and a cached identity trips it at once.
					if gen <= before || gen > after {
						errs <- fmt.Errorf("reader %d: %s = %q, generation %d is outside (%d,%d], the window this read spanned",
							i, name, f[1], gen, before, after)
						return
					}
				}
				if after > before {
					spanned.Add(1)
				}
			}
		}()
	}

	readersDone.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	// A cross-check on the fixture rather than on the code under test: the session
	// advances on every accessor call, so a short count means the fixture stopped
	// churning and the bracket above was measuring nothing. It cannot fail on its
	// own — a reader that bails sends to errs first — which is why it is worded as
	// a fixture check and not as the control.
	if want := int64(readers * 50); spanned.Load() != want {
		t.Errorf("%d of %d reads saw the generation move; the fixture is meant to advance on "+
			"every accessor call, so the bracket above was not measuring what it claims",
			spanned.Load(), want)
	}
}

// TestIdentityIsNotCached pins the property the Identity doc states first — that
// the lookup reads the session each time rather than caching.
//
// Nothing else covers it. Every other test builds a context, reads once, and
// throws it away, so a memo on commonContext guarded by a sync.Once passes all of
// them and passes -race too: the concurrency tests would see one frozen identity
// and agree with themselves about it. The consequence would be silent and real,
// since auth.Transport resolves per request precisely so a credential follows the
// current session rather than the one the context was born with.
func TestIdentityIsNotCached(t *testing.T) {
	sess := &swappingSession{}
	ctx := Promote(&invocationContext{Context: t.Context(), session: sess})

	first, ok := IdentityFromContext(ctx)
	if !ok {
		t.Fatal("IdentityFromContext() reported no identity")
	}
	sess.swap()
	second, ok := IdentityFromContext(ctx)
	if !ok {
		t.Fatal("IdentityFromContext() reported no identity after the swap")
	}

	if second == first {
		t.Errorf("IdentityFromContext() = %+v both before and after the session changed. "+
			"The identity is being cached, and the doc says it is read live", first)
	}
	// Ahead of the first read, not at any exact generation: the session advances
	// on every accessor call, so pinning the number here would encode how many
	// accessors identityOf happens to call rather than the property under test.
	// All three fields, not just the user. Memoizing AppName and SessionID on the
	// grounds that they "never change" while reading UserID live passes a
	// user-only check, and is exactly the half-cached shape the doc forbids.
	for _, f := range [][3]string{
		{"UserID", first.UserID, second.UserID},
		{"AppName", first.AppName, second.AppName},
		{"SessionID", first.SessionID, second.SessionID},
	} {
		if generationOf(t, f[2]) <= generationOf(t, f[1]) {
			t.Errorf("IdentityFromContext().%s went %q then %q; the second read must see a "+
				"later session than the first", f[0], f[1], f[2])
		}
	}
}

// generationOf extracts the counter a swappingSession stamped into a field.
func generationOf(t *testing.T, field string) int {
	t.Helper()
	_, tag, found := strings.Cut(field, "-")
	if !found {
		t.Fatalf("field %q is not <prefix>-<generation>", field)
	}
	n, err := strconv.Atoi(tag)
	if err != nil {
		t.Fatalf("field %q has a non-numeric generation: %v", field, err)
	}
	return n
}
