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
	"context"
	"net/http"
	"net/url"

	"github.com/a2aproject/a2a-go/v2/a2aclient"
)

// This file lives in internal/ rather than in agent/remoteagent/v2 because
// server/adka2a/v2 issues its own CancelTask against a remote subagent's client
// and needs the identical scope, and it cannot import agent/remoteagent/v2 —
// that package imports adka2a.

// CredentialScope is the key a session-aware [auth.CredentialProvider] sees via
// [a2aclient.SessionIDFrom]. It is the whole calling identity, not just the
// session id: the id alone is caller-supplied and optional, so two users each
// holding a session called "default" would otherwise resolve to one credential.
//
// agentName is part of the key because a bearer token is scoped to an audience
// as well as to a user. Without it, one provider shared between two remote
// agents in the same session would hand the second agent the token minted for
// the first.
//
// The parts are percent-encoded and joined with "/", so no two identities can
// produce the same key and the result stays printable. A provider that wants
// the parts back splits on "/" and calls [url.QueryUnescape] on each.
func CredentialScope(appName, userID, sessionID, agentName string) a2aclient.SessionID {
	return a2aclient.SessionID(url.QueryEscape(appName) + "/" + url.QueryEscape(userID) +
		"/" + url.QueryEscape(sessionID) + "/" + url.QueryEscape(agentName))
}

type cardFetchClientKey struct{}

// WithCardFetchClient carries the HTTP client that applies the remote agent's
// credential, so an agent card provider built by remoteagent.NewAgentCardProvider
// can authenticate the card fetch as adk-python does. It is a context value
// rather than a provider option because the provider is built before, and
// independently of, the agent that will call it.
func WithCardFetchClient(ctx context.Context, client *http.Client) context.Context {
	if client == nil {
		return ctx
	}
	return context.WithValue(ctx, cardFetchClientKey{}, client)
}

// CardFetchClientFrom returns the client [WithCardFetchClient] attached, or nil.
func CardFetchClientFrom(ctx context.Context) *http.Client {
	client, _ := ctx.Value(cardFetchClientKey{}).(*http.Client)
	return client
}

// CallIdentity is who is calling whom, for one outgoing A2A call.
type CallIdentity struct {
	AppName   string
	UserID    string
	SessionID string
	// AgentName is the local name of the remote agent being called.
	AgentName string
}

// AttachAuthScope returns ctx carrying what the auth transport needs to
// resolve a credential for one outgoing call: the scope, and the client that
// authenticates a card fetch.
//
// It is a no-op unless cfg.OwnsAuthScope says this SDK put the transport
// there. A caller who wired their own auth owns their own key, and overwriting
// it would silently drop their credential.
func AttachAuthScope(ctx context.Context, cfg *A2AServerConfig, id CallIdentity) context.Context {
	if !cfg.OwnsAuthScope {
		return ctx
	}
	ctx = a2aclient.AttachSessionID(ctx, CredentialScope(id.AppName, id.UserID, id.SessionID, id.AgentName))
	return WithCardFetchClient(ctx, cfg.CardFetchClient)
}
