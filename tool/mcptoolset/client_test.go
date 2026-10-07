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

package mcptoolset

import (
	"context"
	"errors"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestClosePreventsRetry(t *testing.T) {
	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "1"}, nil)
	connects := 0
	transport := connectFunc(func(ctx context.Context) (mcp.Connection, error) {
		connects++
		ct, st := mcp.NewInMemoryTransports()
		session, err := server.Connect(ctx, st, nil)
		if err != nil {
			return nil, err
		}
		t.Cleanup(func() { _ = session.Close() })
		return ct.Connect(ctx)
	})
	client := newConnectionRefresher(nil, transport)
	t.Cleanup(func() { _ = client.Close() })
	calls := 0
	_, _, err := withRetry(t.Context(), client, func(*mcp.ClientSession) (*mcp.CallToolResult, error) {
		calls++
		// The toolset closes after an operation starts, before its error is handled.
		if err := client.Close(); err != nil {
			t.Fatal(err)
		}
		return nil, mcp.ErrConnectionClosed
	})
	if !errors.Is(err, mcp.ErrConnectionClosed) {
		t.Errorf("withRetry() error = %v, want ErrConnectionClosed", err)
	}
	if calls != 1 || connects != 1 {
		t.Errorf("after Close(): %d calls and %d connections, want 1 each", calls, connects)
	}
}

type connectFunc func(context.Context) (mcp.Connection, error)

func (f connectFunc) Connect(ctx context.Context) (mcp.Connection, error) {
	return f(ctx)
}
