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
// Its type is unexported and declared in an internal package, so no code outside
// the module can declare a key equal to it.
//
// That is the whole of the guarantee, and it is narrower than it sounds. Two
// things it does not prevent, neither of which requires naming the type:
//
//   - Reading the identity hands this key to every context.Value implementation
//     between the reader and the invocation, so one read is enough for a wrapper
//     to keep the key and later plant an agent.Identity under it, from a context
//     descending from no invocation at all.
//   - The payload is an ordinary exported struct, so a wrapper can equally build
//     one, or substitute the one it saw on its way past.
//
// So the identity establishes which invocation a context was DERIVED from, and
// is trusted only as far as every wrapper in the chain is. It is not an
// authentication boundary. agent.IdentityFromContext states the full rule, and
// both limits above are pinned by tests in package agent.
const IdentityKey ctxKey = 0

// Recovered returns what read produced, and whether it returned at all.
//
// It exists for the ADK Value implementations, which read the invocation
// identity off session.Session — a public interface whose implementations are
// arbitrary code. A nil session, a typed-nil one whose accessors dereference the
// receiver, a session wrapping a nil one (the shape llmagent.newWrappedSession
// produces for a nil original), or a Session() accessor that declines, each panic
// on the way. A typed-nil whose accessors ignore the receiver does not, and
// answers normally. Value runs inside
// http.RoundTripper on the caller's goroutine, where net/http does not recover,
// so a broken session would take the process down. Reporting no identity instead
// fails the credential path closed.
//
// Nothing partially built escapes: on a panic the zero value is returned. The
// cost is that a bug inside a caller's accessor surfaces as a missing identity
// rather than a stack trace — deliberate, since the alternative here is killing
// the process, and the credential path fails closed either way.
//
// It contains a panic, and only a panic. An accessor that calls runtime.Goexit
// ends the calling goroutine, which no recover can undo, and a runtime throw —
// "concurrent map read and map write", say, from a session whose accessors read
// state another goroutine is writing — is not recoverable at all. Both end the
// process, so the containment this offers is narrower than "a broken session
// cannot bring us down": it covers a session that panics, not one that is
// unsafe to read from the goroutine holding the context.
func Recovered[T any](read func() T) (v T, ok bool) {
	defer func() {
		if recover() != nil {
			var zero T
			v, ok = zero, false
		}
	}()
	return read(), true
}

// Source marks a context type that answers [IdentityKey] for the invocation it
// speaks for, rather than for whatever it happens to be derived from.
//
// It is what lets the identity procedure tell one of ADK's own session-less
// views — a tool context, a callback context, the cancel-scoped context a
// streaming tool runs under — from an agent.InvocationContext written outside
// the module. One of ours is asked, because it hands the question down to the
// invocation underneath. Anything else is only read, because it cannot override
// a key it cannot name, so its embedded parent would answer with a different
// call's user.
//
// The method comes from embedding [Marker] and this package is internal, so a
// type outside the module can neither name it nor inherit it, and cannot be
// mistaken for one of ours.
type Source interface{ adkIdentitySource() }

// Marker is embedded by the ADK context types that satisfy [Source]. It carries
// no state.
type Marker struct{}

func (Marker) adkIdentitySource() {}
