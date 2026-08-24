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

// Package adkcontext holds the private context key under which an ADK context
// registers its invocation identity, so it can be recovered from a derived
// context.Context via agent.IdentityFromContext. It is a tiny leaf package shared
// by the agent and internal/context packages to avoid an import cycle.
package adkcontext

type ctxKey int

// IdentityKey is the context value key for the agent.Identity of an ADK context.
// It lives in an internal package with an unexported type, so no code outside
// the module can name the key. The key is therefore unforgeable, but the value
// it addresses is not: any in-process context wrapper can read the exported
// agent.Identity on its way past and hand back a different one, without ever
// naming the key. Treat the identity as trusted only as far as every wrapper in
// the chain is.
const IdentityKey ctxKey = 0

// ReadIdentity runs read, which fills in an invocation identity from a
// session.Session, and reports whether it completed.
//
// It recovers, because every ADK Value implementation reads the identity off
// session.Session — a public interface whose implementations are arbitrary code.
// A nil or typed-nil session, or a session wrapping a nil one (the shape
// llmagent.newWrappedSession produces for a nil original), panics on the first
// accessor. Value runs inside http.RoundTripper on the caller's goroutine, where
// net/http does not recover, so a broken session would take the process down;
// reporting the identity as absent instead fails the credential path closed.
func ReadIdentity(read func()) (ok bool) {
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	read()
	return true
}
