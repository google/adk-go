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

// Recovered returns what read produced, and whether it returned at all.
//
// It exists for the ADK Value implementations, which read the invocation
// identity off session.Session — a public interface whose implementations are
// arbitrary code. A nil or typed-nil session, a session wrapping a nil one (the
// shape llmagent.newWrappedSession produces for a nil original), or a Session()
// accessor that declines, all panic on the way. Value runs inside
// http.RoundTripper on the caller's goroutine, where net/http does not recover,
// so a broken session would take the process down; reporting no identity instead
// fails the credential path closed.
//
// Nothing partially built escapes: on a panic the zero value is returned. The
// cost is that a bug inside a caller's accessor surfaces as a missing identity
// rather than a stack trace — deliberate, since the alternative here is killing
// the process, and the credential path fails closed either way.
func Recovered[T any](read func() T) (v T, ok bool) {
	defer func() {
		if recover() != nil {
			var zero T
			v, ok = zero, false
		}
	}()
	return read(), true
}
