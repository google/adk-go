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

// FINDING L5 — rehydrateCache reads session event fields with no
// synchronization (best-effort, -race).
//
// BUG: When a dynamic sub-scheduler is created (on a node goroutine),
// newDynamicSubScheduler calls rehydrateCache (dynamic_scheduler.go), which
// iterates sess.Events().All() and reads each event's fields — ev.Output,
// ev.NodeInfo, ev.NodeInfo.Path — while holding only the sub-scheduler's own
// mutex (which guards resultByPath, NOT the session or the events). It does
// no synchronization against an engine that is concurrently producing /
// stamping events for the same session. The engine's scheduler.handleEvent
// writes exactly these fields (ev.Branch, ev.NodeInfo.Path) as events flow.
// rehydrateCache's correctness therefore depends entirely on (a) the Session
// backend serializing reads against writes and (b) no other goroutine
// mutating an event it is reading — neither of which the session.Session
// interface (session/session.go) guarantees.
//
// EXPECTED: rehydrateCache must read session state safely regardless of
// concurrent engine activity (it runs on a node goroutine while the run is
// in flight), i.e. it must not data-race on the events it scans.
//
// WHY LATENT / BEST-EFFORT: the bundled in-memory backend
// (session/inmemory.go) snapshots the events slice header under an RLock in
// Events(), so the *slice* access is safe, and the production event lifecycle
// happens to finish mutating an event before it becomes "history" that
// rehydrateCache would scan — together these mask the race in the shipped
// stack. Neither is required by the Session contract. This test surfaces the
// missing synchronization in rehydrateCache itself: it uses a backend that
// (like in-memory) snapshots the slice header safely, so the ONLY
// unsynchronized access left is rehydrateCache's read of the shared *Event
// fields, raced against an engine-like goroutine that writes those same
// fields (mirroring scheduler.handleEvent).
//
// HOW THIS TEST DEMONSTRATES IT: It seeds a session with events, then runs
// rehydrateCache (via newDynamicSubScheduler) repeatedly on one goroutine
// while another goroutine mutates ev.NodeInfo.Path / ev.Output / ev.Branch
// on those same events. FAILS under `go test -race` (DATA RACE; the read
// side is rehydrateCache at dynamic_scheduler.go ~161/164/168). Without
// -race it passes (no functional assertion) — it is purely a race-detector
// repro. Use -count to confirm reliability.

package workflow

import (
	"iter"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/session"
)

// vbugL5Session is a minimal, contract-conforming session.Session whose
// Events() snapshots the slice header under an RLock exactly like the
// bundled in-memory backend — so the slice access is safe and the only
// unsynchronized access in this test is rehydrateCache's read of the shared
// *Event fields.
type vbugL5Session struct {
	mu     sync.RWMutex
	events []*session.Event
}

func (s *vbugL5Session) ID() string                { return "vbug-l5" }
func (s *vbugL5Session) AppName() string           { return "app" }
func (s *vbugL5Session) UserID() string            { return "user" }
func (s *vbugL5Session) State() session.State      { return nil }
func (s *vbugL5Session) LastUpdateTime() time.Time { return time.Time{} }

func (s *vbugL5Session) Events() session.Events {
	s.mu.RLock()
	defer s.mu.RUnlock()
	// Copy the slice header (mirrors inmemory's safe snapshot); the *Event
	// pointers are still shared, which is the point.
	cp := make([]*session.Event, len(s.events))
	copy(cp, s.events)
	return vbugL5Events(cp)
}

// snapshot returns the shared event pointers for the mutator goroutine.
func (s *vbugL5Session) snapshot() []*session.Event {
	s.mu.RLock()
	defer s.mu.RUnlock()
	cp := make([]*session.Event, len(s.events))
	copy(cp, s.events)
	return cp
}

type vbugL5Events []*session.Event

func (e vbugL5Events) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, ev := range e {
			if !yield(ev) {
				return
			}
		}
	}
}
func (e vbugL5Events) Len() int                { return len(e) }
func (e vbugL5Events) At(i int) *session.Event { return e[i] }

func TestVbugL5_RehydrateCacheRacesOnSharedEventFields(t *testing.T) {
	const (
		nEvents    = 64
		rehydrates = 60
	)

	sess := &vbugL5Session{}
	for i := 0; i < nEvents; i++ {
		ev := session.NewEvent("inv")
		// Path under the sub-scheduler's parentPath prefix so the
		// rehydrateCache loop body (the racing reads) actually executes.
		ev.NodeInfo = &session.NodeInfo{Path: "dyn/c@1"}
		ev.Output = "v"
		sess.events = append(sess.events, ev)
	}

	mc := newMockCtx(t)
	mc.sess = sess
	parent := newNodeContext(mc, nil)

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		// Engine-like mutator: writes exactly the fields scheduler.handleEvent
		// stamps on events as they flow, on the same *Event objects that
		// rehydrateCache reads.
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			for _, ev := range sess.snapshot() {
				ev.NodeInfo.Path = "dyn/c@1"
				ev.Output = "v2"
				ev.Branch = "b"
			}
		}
	}()

	// Drive rehydrateCache concurrently with the mutator. Each
	// newDynamicSubScheduler call iterates the seeded events and reads their
	// fields without synchronization.
	for i := 0; i < rehydrates; i++ {
		_ = newDynamicSubScheduler(parent, "dyn", noopEmit)
	}

	close(stop)
	wg.Wait()
}
