// Copyright 2025 Google LLC
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

package mcptoolset

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// gatedSessionTransport pauses connection establishment before using the real
// MCP in-memory protocol. Cancellation is exercised while the connection owner
// remains blocked, without depending on an external server.
type gatedSessionTransport struct {
	gate    chan struct{}
	entered chan struct{}
	calls   atomic.Int32
	server  *mcp.Server
}

func (g *gatedSessionTransport) Connect(ctx context.Context) (mcp.Connection, error) {
	g.calls.Add(1)
	g.entered <- struct{}{}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-g.gate:
	}
	client, server := mcp.NewInMemoryTransports()
	if _, err := g.server.Connect(ctx, server, nil); err != nil {
		return nil, err
	}
	return client.Connect(ctx)
}

func TestSessionWaitCancellation(t *testing.T) {
	for _, refresh := range []bool{false, true} {
		name := "initialize"
		if refresh {
			name = "refresh"
		}
		t.Run(name, func(t *testing.T) {
			for _, deadline := range []bool{false, true} {
				name := "cancel"
				if deadline {
					name = "deadline"
				}
				t.Run(name, func(t *testing.T) {
					transport := &gatedSessionTransport{gate: make(chan struct{}), entered: make(chan struct{}, 16), server: mcp.NewServer(&mcp.Implementation{Name: "test", Version: "1"}, nil)}
					c := newConnectionRefresher(nil, transport)
					acquire := c.getSession
					if refresh {
						acquire = c.refreshConnection
					}
					ownerCtx, ownerCancel := context.WithTimeout(t.Context(), 5*time.Second)
					defer ownerCancel()
					ownerDone := make(chan error, 1)
					go func() { _, err := acquire(ownerCtx); ownerDone <- err }()
					<-transport.entered

					if transport.calls.Load() != 1 {
						t.Fatalf("connect count=%d, want 1", transport.calls.Load())
					}

					waiterCtx, waiterCancel := context.WithCancel(t.Context())
					wantErr := error(context.Canceled)
					if deadline {
						waiterCancel()
						waiterCtx, waiterCancel = context.WithTimeout(t.Context(), 50*time.Millisecond)
						wantErr = context.DeadlineExceeded
					}
					defer waiterCancel()
					waiterDone := make(chan error, 1)
					waiterReturned := false
					go func() { _, err := acquire(waiterCtx); waiterDone <- err }()

					if deadline {
						<-waiterCtx.Done()
					} else {
						waiterCancel()
					}

					select {
					case err := <-waiterDone:
						waiterReturned = true
						if !errors.Is(err, wantErr) {
							t.Errorf("waiter error=%v, want context cancellation", err)
						}
					case <-time.After(time.Second):
						t.Error("waiter did not return while the connection owner remained blocked")
					}
					if transport.calls.Load() != 1 {
						t.Errorf("canceled waiter started an extra connection: %d", transport.calls.Load())
					}
					select {
					case <-ownerDone:
						t.Fatal("canceling waiter interrupted owner")
					default:
					}
					close(transport.gate)

					if err := <-ownerDone; err != nil {
						t.Fatalf("owner failed: %v", err)
					}
					if !waiterReturned {
						<-waiterDone
					}
					ownerSession := c.session
					if ownerSession == nil {
						t.Fatal("waiter destroyed the owner session")
					}
					defer func() {
						if err := ownerSession.Close(); err != nil {
							t.Error("failed to close owner session")
						}
					}()
					// A canceled call must not take even an available permit and disturb
					// the cached session, including the refresh path's Ping/Close sequence.
					if _, err := acquire(waiterCtx); !errors.Is(err, wantErr) {
						t.Errorf("already canceled call error=%v", err)
					}
					session, err := c.getSession(t.Context())
					if err != nil {
						t.Fatal(err)
					}
					if err := session.Ping(t.Context(), nil); err != nil {
						t.Fatal("owner session no longer usable")
					}
					if transport.calls.Load() != 1 {
						t.Errorf("session unexpectedly replaced: connects=%d", transport.calls.Load())
					}
				})
			}
		})
	}
}

func TestSessionInitializationFailureRecovery(t *testing.T) {
	transport := &gatedSessionTransport{gate: make(chan struct{}), entered: make(chan struct{}, 16), server: mcp.NewServer(&mcp.Implementation{Name: "test", Version: "1"}, nil)}
	c := newConnectionRefresher(nil, transport)
	ctx, cancel := context.WithCancel(t.Context())
	firstDone := make(chan error, 1)
	go func() { _, err := c.getSession(ctx); firstDone <- err }()
	<-transport.entered
	cancel()

	if err := <-firstDone; !errors.Is(err, context.Canceled) {
		t.Fatalf("first connection error=%v", err)
	}

	var wg sync.WaitGroup
	sessions := make(chan *mcp.ClientSession, 8)
	errs := make(chan error, 8)
	for range 8 {
		wg.Go(func() { s, err := c.getSession(t.Context()); sessions <- s; errs <- err })
	}

	<-transport.entered
	if transport.calls.Load() != 2 {
		t.Errorf("recovery must have one connection owner, calls=%d", transport.calls.Load())
	}
	close(transport.gate)
	wg.Wait()
	close(sessions)
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	defer func() {
		if err := c.session.Close(); err != nil {
			t.Error("failed to close recovered session")
		}
	}()
	for session := range sessions {
		if session != c.session {
			t.Error("waiters did not share recovered session")
		}
	}
	if transport.calls.Load() != 2 {
		t.Errorf("recovery connected more than once, calls=%d", transport.calls.Load())
	}
}
