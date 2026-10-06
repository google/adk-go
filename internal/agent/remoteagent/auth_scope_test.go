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

package remoteagent

import (
	"net/http"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2aclient"
)

// TestAttachAuthScope covers the gate that decides whether this SDK owns the
// credential scope on an outgoing call. Overwriting a scope a caller attached
// for their own auth would resolve under a key their store has never seen and
// drop their credential.
func TestAttachAuthScope(t *testing.T) {
	client := &http.Client{}
	id := CallIdentity{AppName: "shop", UserID: "alice", SessionID: "s1", AgentName: "crm"}

	t.Run("owned", func(t *testing.T) {
		ctx := AttachAuthScope(t.Context(), &A2AServerConfig{OwnsAuthScope: true, CardFetchClient: client}, id)
		sid, ok := a2aclient.SessionIDFrom(ctx)
		if !ok {
			t.Fatal("no credential scope attached")
		}
		if want := a2aclient.SessionID("shop/alice/s1/crm"); sid != want {
			t.Errorf("scope = %q, want %q", sid, want)
		}
		if CardFetchClientFrom(ctx) != client {
			t.Error("the card-fetch client was not attached, so a card fetched for this cancel would go out unauthenticated")
		}
	})

	t.Run("not owned leaves the caller's context alone", func(t *testing.T) {
		theirs := a2aclient.AttachSessionID(t.Context(), "caller-tenant")
		ctx := AttachAuthScope(theirs, &A2AServerConfig{CardFetchClient: client}, id)
		sid, _ := a2aclient.SessionIDFrom(ctx)
		if want := a2aclient.SessionID("caller-tenant"); sid != want {
			t.Errorf("scope = %q, want %q; a caller who wired their own auth owns the key", sid, want)
		}
		if CardFetchClientFrom(ctx) != nil {
			t.Error("a card-fetch client was attached to a context this SDK does not own")
		}
	})
}

// TestCardFetchClientFromEmptyContext pins that no client means an
// unauthenticated fetch through the default resolver, never a panic.
func TestCardFetchClientFromEmptyContext(t *testing.T) {
	if got := CardFetchClientFrom(t.Context()); got != nil {
		t.Errorf("CardFetchClientFrom() = %v, want nil", got)
	}
	if got := WithCardFetchClient(t.Context(), nil); CardFetchClientFrom(got) != nil {
		t.Error("WithCardFetchClient(nil) attached a client")
	}
}
