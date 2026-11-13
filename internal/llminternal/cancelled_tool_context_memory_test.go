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

package llminternal

import (
	"context"
	"testing"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
)

// sessionRecordingMemory records every session handed to AddSessionToMemory.
type sessionRecordingMemory struct{ got []session.Session }

func (m *sessionRecordingMemory) AddSessionToMemory(_ context.Context, s session.Session) error {
	m.got = append(m.got, s)
	return nil
}

func (m *sessionRecordingMemory) SearchMemory(context.Context, string) (*memory.SearchResponse, error) {
	return &memory.SearchResponse{}, nil
}

// A live streaming tool runs under cancelledToolContext, which wraps the tool
// context. agent.AddSessionToMemory called from it must hand the invocation's
// session to the memory service. It fails if agent.AddSessionToMemory reads the
// session only from the tool context wrapper types it knows by name, because
// the wrapped tool context's Session reports nil.
func TestAddSessionToMemoryFromAStreamingToolContext(t *testing.T) {
	resp, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatalf("session Create() = %v", err)
	}
	mem := &sessionRecordingMemory{}
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: resp.Session, Memory: mem})
	toolCtx := agent.NewToolContext(ic, "fc", nil, nil)
	cancelCtx, cancel := toolCtx.WithAgentCancel()
	defer cancel()
	streaming := &cancelledToolContext{Context: toolCtx, cancelCtx: cancelCtx}

	if err := agent.AddSessionToMemory(streaming); err != nil {
		t.Fatalf("AddSessionToMemory() = %v", err)
	}

	if len(mem.got) != 1 || mem.got[0] != resp.Session {
		t.Errorf("memory service received %v, want only the invocation's session %q", mem.got, resp.Session.ID())
	}
}
