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

package database

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"gorm.io/gorm"

	"google.golang.org/adk/v2/session"
)

// The session row is written back by the append, so it has to be locked for
// the rest of the transaction. Without the lock an append from another
// process can write the row in between this read and the commit. This matches
// adk-python, which locks the session row FOR UPDATE.
func TestDatabaseService_AppendEvent_LocksTheSessionRow(t *testing.T) {
	ctx := t.Context()
	s := emptyService(t)
	created, err := s.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	got := recordStateRowStatements(t, s, "sessions")

	event := &session.Event{
		ID:        "e1",
		Author:    "user",
		Timestamp: time.Now(),
		Actions:   session.EventActions{StateDelta: map[string]any{"s": 1}},
	}
	if err := s.AppendEvent(ctx, created.Session, event); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}

	if len(*got) != 1 || (*got)[0] != "select sessions for update" {
		t.Errorf("AppendEvent session statements:\n got %q\nwant %q", *got, []string{"select sessions for update"})
	}
}

// A failed append must not leave its event in the in-memory session. The
// event used to be appended before the write, so an append that the database
// rejected still showed up in Events() and in the session state.
func TestDatabaseService_AppendEvent_FailedAppendLeavesNoEventInMemory(t *testing.T) {
	ctx := t.Context()
	s := emptyService(t)
	created, err := s.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	sess, ok := created.Session.(*localSession)
	if !ok {
		t.Fatalf("unexpected session type %T", created.Session)
	}

	// Remove the session row so that the append fails when it reads it.
	if err := s.db.Where("id = ?", sess.ID()).Delete(&storageSession{}).Error; err != nil {
		t.Fatalf("delete session row: %v", err)
	}

	event := &session.Event{
		ID:        "e1",
		Author:    "user",
		Timestamp: time.Now(),
		Actions:   session.EventActions{StateDelta: map[string]any{"s": 1}},
	}
	if err := s.AppendEvent(ctx, sess, event); err == nil {
		t.Fatal("AppendEvent succeeded; want it to fail on the missing session row")
	}

	if got := sess.Events().Len(); got != 0 {
		t.Errorf("in-memory events after a failed append = %d; want 0", got)
	}
	if _, err := sess.State().Get("s"); err == nil {
		t.Error("session state has the key of a failed append; want it absent")
	}
}

// A temp: key is scoped to the invocation and must not reach storage, but it
// has to stay readable in the live session for the rest of the invocation.
// The in-memory and the Vertex AI session services keep it, and so does
// adk-python, which applies the temp state before trimming it.
func TestDatabaseService_AppendEvent_KeepsTempStateReadableInMemory(t *testing.T) {
	ctx := t.Context()
	s := emptyService(t)
	created, err := s.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	sess, ok := created.Session.(*localSession)
	if !ok {
		t.Fatalf("unexpected session type %T", created.Session)
	}

	event := &session.Event{
		ID:        "e1",
		Author:    "user",
		Timestamp: time.Now(),
		Actions: session.EventActions{
			StateDelta: map[string]any{"temp:k1": "v1", "k2": "v2"},
		},
	}
	if err := s.AppendEvent(ctx, sess, event); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}

	got, err := sess.State().Get("temp:k1")
	if err != nil {
		t.Errorf("temp key is not readable in the live session: %v", err)
	} else if got != "v1" {
		t.Errorf("temp key = %v; want %q", got, "v1")
	}

	// The temp key must not reach storage.
	resp, err := s.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: sess.ID()})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if _, err := resp.Session.State().Get("temp:k1"); err == nil {
		t.Error("temp key reached storage; want it absent")
	}
	if _, err := resp.Session.State().Get("k2"); err != nil {
		t.Errorf("stored state is missing the key of the event: %v", err)
	}
}

// Appends to one session are serialized, so two of them cannot be inside
// their transactions at the same time. The callback below holds the first
// append inside its transaction, so an append that is not serialized would
// reach the database while the first one is still there.
func TestDatabaseService_AppendEvent_SerializesAppendsToOneSession(t *testing.T) {
	ctx := t.Context()
	s := emptyService(t)
	created, err := s.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	var inFlight atomic.Int32
	var maxInFlight atomic.Int32
	firstInside := make(chan struct{})
	anotherInside := make(chan struct{}, 1)
	release := make(chan struct{})
	var holdFirst atomic.Bool
	record := func(tx *gorm.DB) {
		if tx.Statement.Table != "sessions" {
			return
		}
		n := inFlight.Add(1)
		defer inFlight.Add(-1)
		updateMax(&maxInFlight, n)
		if holdFirst.CompareAndSwap(false, true) {
			// Hold the first append inside its transaction.
			close(firstInside)
			<-release
			return
		}
		select {
		case anotherInside <- struct{}{}:
		default:
		}
	}
	if err := s.db.Callback().Query().Before("gorm:query").Register("test:count_session_query", record); err != nil {
		t.Fatalf("register query callback: %v", err)
	}
	t.Cleanup(func() { _ = s.db.Callback().Query().Remove("test:count_session_query") })

	const appends = 4
	errs := make([]error, appends)
	var wg sync.WaitGroup

	// The first append is held inside its transaction.
	wg.Add(1)
	go func() {
		defer wg.Done()
		event := &session.Event{
			ID:        "e0",
			Author:    "user",
			Timestamp: time.Now(),
			Actions:   session.EventActions{StateDelta: map[string]any{"k0": 0}},
		}
		errs[0] = s.AppendEvent(ctx, created.Session, event)
	}()
	select {
	case <-firstInside:
	case <-time.After(10 * time.Second):
		t.Fatal("the first append never reached the database")
	}

	// The others start while the first one is held there. A serialized
	// append waits for the first one and does not get to the database.
	for i := 1; i < appends; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			event := &session.Event{
				ID:        fmt.Sprintf("e%d", i),
				Author:    "user",
				Timestamp: time.Now(),
				Actions:   session.EventActions{StateDelta: map[string]any{fmt.Sprintf("k%d", i): i}},
			}
			errs[i] = s.AppendEvent(ctx, created.Session, event)
		}()
	}

	select {
	case <-anotherInside:
		t.Error("a second append reached the database while the first was inside its transaction")
	case <-time.After(200 * time.Millisecond):
		// The other appends are waiting for the first one.
	}
	close(release)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("AppendEvent %d: %v", i, err)
		}
	}
	if got := maxInFlight.Load(); got != 1 {
		t.Errorf("appends inside a transaction at the same time = %d; want 1", got)
	}

	// Every append is stored, and every state key survives.
	resp, err := s.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: created.Session.ID()})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got := resp.Session.Events().Len(); got != appends {
		t.Errorf("stored events = %d; want %d", got, appends)
	}
	for i := range appends {
		key := fmt.Sprintf("k%d", i)
		if _, err := resp.Session.State().Get(key); err != nil {
			t.Errorf("state[%q] is missing: %v", key, err)
		}
	}
}

// AppendEvent takes the same row locks in the same order as Create:
// app state, then user state, then the session row. Create writes app and
// user before inserting the session, so taking the session row first here
// inverts the order; on MySQL a concurrent duplicate Create then waits on the
// session row while holding the state rows and the append, holding the
// session row while waiting for the state rows, is aborted as the deadlock
// victim.
func TestDatabaseService_AppendEvent_LocksStateRowsBeforeTheSessionRow(t *testing.T) {
	ctx := t.Context()
	s := emptyService(t)
	created, err := s.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	got := recordStateRowStatements(t, s, "app_states", "user_states", "sessions")

	event := &session.Event{
		ID:        "e1",
		Author:    "user",
		Timestamp: time.Now(),
		Actions: session.EventActions{StateDelta: map[string]any{
			"app:k":  1,
			"user:k": 2,
			"k":      3,
		}},
	}
	if err := s.AppendEvent(ctx, created.Session, event); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}

	// The first lock statement for each table is the one taken by the
	// transaction; inserts of missing rows come before their lock.
	first := func(substr string) int {
		for i, stmt := range *got {
			if strings.HasPrefix(stmt, substr) {
				return i
			}
		}
		return -1
	}
	app := first("select app_states for update")
	user := first("select user_states for update")
	session := first("select sessions for update")
	if app < 0 || user < 0 || session < 0 {
		t.Fatalf("AppendEvent lock statements:\n got %q\nwant a row lock on app_states, user_states and sessions", *got)
	}
	if !(app < user && user < session) {
		t.Errorf("AppendEvent lock order:\n got %q\nwant app_states, then user_states, then sessions", *got)
	}
}

// A Create that names an existing session ID runs concurrently with an append
// to that session. The two take the same row locks in the same order, so the
// append still commits; the Create fails on the primary key instead of
// deadlocking the append.
func TestDatabaseService_AppendEvent_SurvivesConcurrentDuplicateCreate(t *testing.T) {
	ctx := t.Context()
	s := emptyService(t)
	created, err := s.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "ses1"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	var wg sync.WaitGroup
	var appendErr, createErr error
	wg.Add(2)
	go func() {
		defer wg.Done()
		event := &session.Event{
			ID:        "e1",
			Author:    "user",
			Timestamp: time.Now(),
			Actions: session.EventActions{StateDelta: map[string]any{
				"app:k":  1,
				"user:k": 2,
				"k":      3,
			}},
		}
		appendErr = s.AppendEvent(ctx, created.Session, event)
	}()
	go func() {
		defer wg.Done()
		_, createErr = s.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "ses1"})
	}()
	wg.Wait()

	if appendErr != nil {
		t.Errorf("AppendEvent alongside a duplicate Create: %v", appendErr)
	}
	if createErr == nil {
		t.Error("the duplicate Create succeeded; want it to fail on the primary key")
	}

	resp, err := s.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "ses1"})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got := resp.Session.Events().Len(); got != 1 {
		t.Errorf("stored events after the race = %d; want 1", got)
	}
	for _, key := range []string{"app:k", "user:k", "k"} {
		if _, err := resp.Session.State().Get(key); err != nil {
			t.Errorf("state[%q] is missing after the race: %v", key, err)
		}
	}
}

// An append whose request is canceled while it waits for another append to
// the same session leaves the queue and returns context.Canceled at once.
// The base implementation reached the database layer and failed there; a
// plain mutex would keep the canceled request blocked until the holder's
// transaction commits, even though the client has already gone.
func TestDatabaseService_AppendEvent_CanceledAppendLeavesTheQueue(t *testing.T) {
	ctx := t.Context()
	s := emptyService(t)
	created, err := s.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "ses1"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	key := sessionKey{appName: "app", userID: "user", sessionID: "ses1"}

	firstInside := make(chan struct{})
	release := make(chan struct{})
	var holdFirst atomic.Bool
	record := func(tx *gorm.DB) {
		if tx.Statement.Table != "sessions" {
			return
		}
		if holdFirst.CompareAndSwap(false, true) {
			close(firstInside)
			<-release
		}
	}
	if err := s.db.Callback().Query().Before("gorm:query").Register("test:hold_first_append", record); err != nil {
		t.Fatalf("register query callback: %v", err)
	}
	t.Cleanup(func() { _ = s.db.Callback().Query().Remove("test:hold_first_append") })

	firstErr := make(chan error, 1)
	go func() {
		firstErr <- s.AppendEvent(ctx, created.Session, &session.Event{
			ID:        "e0",
			Author:    "user",
			Timestamp: time.Now(),
			Actions:   session.EventActions{StateDelta: map[string]any{"k0": 0}},
		})
	}()
	select {
	case <-firstInside:
	case <-time.After(10 * time.Second):
		t.Fatal("the first append never reached the database")
	}

	// The second append queues on the session lock while the first holds it.
	cancelCtx, cancel := context.WithCancel(ctx)
	secondErr := make(chan error, 1)
	go func() {
		secondErr <- s.AppendEvent(cancelCtx, created.Session, &session.Event{
			ID:        "e1",
			Author:    "user",
			Timestamp: time.Now(),
		})
	}()

	// Wait for the goroutine above to enqueue, then cancel it.
	queueDeadline := time.Now().Add(10 * time.Second)
	for {
		s.sessionLocks.mu.Lock()
		refs := 0
		if entry, ok := s.sessionLocks.locks[key]; ok {
			refs = entry.refs
		}
		s.sessionLocks.mu.Unlock()
		if refs == 2 {
			break
		}
		if time.Now().After(queueDeadline) {
			close(release)
			t.Fatal("the canceled append never queued on the session lock")
		}
		time.Sleep(time.Millisecond)
	}
	cancel()

	select {
	case err := <-secondErr:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("canceled append error = %v; want %v", err, context.Canceled)
		}
	case <-time.After(10 * time.Second):
		close(release)
		t.Fatal("the canceled append waited for the holder to finish")
	}

	close(release)
	if err := <-firstErr; err != nil {
		t.Errorf("the holding append failed: %v", err)
	}

	resp, err := s.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "ses1"})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got := resp.Session.Events().Len(); got != 1 {
		t.Errorf("stored events = %d; want 1", got)
	}
}
