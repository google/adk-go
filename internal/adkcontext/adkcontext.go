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

import "reflect"

type ctxKey int

// IdentityKey is the context value key for the agent.Identity of an ADK context.
// It lives in an internal package with an unexported type, so no code outside
// the module can name the key. The value it addresses is the exported
// agent.Identity, which any in-process context wrapper can read or substitute on
// the way through — no less than it could call the credential services directly.
const IdentityKey ctxKey = 0

// Usable reports whether the methods of interface value v can be called. Both
// ADK Value implementations read the identity off session.Session, and a
// typed-nil session survives != nil and then nil-derefs; the kind switch is
// needed because reflect.Value.IsNil panics on a struct value, which a
// six-accessor interface like session.Session may well be.
//
// It cannot promise more than that: a non-nil session whose own accessors panic
// still panics, here as anywhere else in ADK.
func Usable(v any) bool {
	if v == nil {
		return false
	}
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice, reflect.UnsafePointer:
		return !rv.IsNil()
	default:
		return true
	}
}
