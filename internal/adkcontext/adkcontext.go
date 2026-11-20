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

// Package adkcontext holds the private context keys and helpers under which ADK
// contexts expose invocation identity, context lineage, workflow node paths,
// and the runner's child-to-parent agent map across derived context.Context
// chains. It is a tiny leaf package shared by the agent, internal/context, and
// internal/agent/parentmap packages to avoid import cycles.
package adkcontext

import (
	"context"
	"reflect"
	"weak"
)

type ctxKey int

// IdentityKey is the context value key for the agent.Identity of an ADK context.
// Its type is unexported and declared in an internal package, so no code outside
// the google.golang.org/adk/v2 import-path tree can declare a key equal to it.
// Go scopes internal/ by import path rather than by module, so a separate module
// published under that prefix is INSIDE the boundary and can both name this key
// and embed [Marker].
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

// PathKey is the private context value key under which an ADK context exposes
// its non-empty workflow node path.
const PathKey ctxKey = 1

// ParentMapKey is the private context value key under which ADK's runner
// attaches the child-to-parent agent map.
const ParentMapKey ctxKey = 2

type runTokenHop uint8

// RunTokenKey is the private context value key under which ADK exposes the
// per-run identity token minted for each agent.Run execution. Using an
// unexported uint8 key type matches LineageKey so wrappers blocking ADK's
// private lineage key also block RunTokenKey cleanly.
const RunTokenKey runTokenHop = 64

const maxHopBudget uint8 = 64

type pathHop int

type parentMapHop int

type runToken struct {
	_ byte
}

// NewRunToken mints an unforgeable, comparable, non-nil per-run identity token
// owned by a single agent.Run execution.
func NewRunToken() any {
	return &runToken{}
}

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
// It contains a panic, and only a panic, and the two things it does not contain
// fail differently. A runtime throw — "concurrent map read and map write", say,
// from a session whose accessors read state another goroutine is writing — is not
// recoverable and ends the process. runtime.Goexit ends only the calling
// goroutine: measured from a RoundTrip on a net/http client goroutine, the code
// after this call never runs and the process stays alive, so what the caller sees
// is a request that vanishes with no panic, no log and no error. A program dies
// from Goexit only when it was the last goroutine.
//
// So the containment this offers is narrower than "a broken session cannot bring
// us down": it covers a session that panics, not one that is unsafe to read from
// the goroutine holding the context, and not one that never returns at all.
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
// type outside the google.golang.org/adk/v2 import-path tree can neither name it
// nor inherit it, and cannot be mistaken for one of ours.
//
// Inside that tree, embedding a marked type inherits its ANSWER and not only its
// membership. A type embedding one that answers the key terminally from its own
// session — agent's invocationContext, or internal/context's InvocationContext —
// is asked, and the promoted Value replies from the EMBEDDED session, ignoring a
// Session method the embedder overrides. Such a type must override Value too, or
// not embed a terminal answerer.
type Source interface{ adkIdentitySource() }

// Marker is embedded by the ADK context types that satisfy [Source]. It carries
// no state.
type Marker struct{}

func (Marker) adkIdentitySource() {}

// contextToken wraps a weak pointer in an unexported struct with an unexported
// field and no methods. Unlike a bare weak.Pointer[T], which exports Value(),
// contextToken gives reflect callers zero methods and an unexported field
// (CanInterface() == false) while remaining comparable with == and usable as a
// map key.
type contextToken struct {
	ptr any
}

type lineageKey uint8

// LineageKey is the starting private context key for walking the chain of ADK
// contexts from which a context.Context derives. Its type is unexported and
// declared in this internal package, so application context keys cannot collide
// with or be intercepted by lineage lookups. Each ADK context decrements the
// remaining hop budget when delegating to its parent and returns a newly
// allocated slice of opaque tokens, keeping concurrent multi-parent Value
// lookups free of shared mutable state and bounding cyclic Value chains.
const LineageKey lineageKey = 64

// IsLineageKey reports whether key is an ADK lineage lookup key.
func IsLineageKey(key any) bool {
	_, ok := key.(lineageKey)
	return ok
}

// Token returns an opaque, comparable identity token for heap-allocated ADK
// context c, or nil when c is nil.
func Token[T any](c *T) any {
	if c == nil {
		return nil
	}
	return contextToken{ptr: weak.Make(c)}
}

type lineageNode struct {
	tok   any
	extra []any
	next  *lineageNode
}

func countLineageNodes(node *lineageNode, limit int) int {
	n := 0
	for node != nil && n < limit {
		if node.tok != nil {
			n++
		}
		for _, item := range node.extra {
			if n >= limit {
				break
			}
			if sub, ok := item.(*lineageNode); ok {
				n += countLineageNodes(sub, limit-n)
			} else {
				n++
			}
		}
		node = node.next
	}
	return min(n, limit)
}

func appendLineageNodes(dst []any, node *lineageNode, limit int) []any {
	for node != nil && len(dst) < limit {
		if node.tok != nil {
			dst = append(dst, node.tok)
		}
		for _, item := range node.extra {
			if len(dst) >= limit {
				break
			}
			if sub, ok := item.(*lineageNode); ok {
				dst = appendLineageNodes(dst, sub, limit)
			} else {
				dst = append(dst, item)
			}
		}
		node = node.next
	}
	return dst
}

func collectLineageTail(rem lineageKey, parent context.Context) *lineageNode {
	if rem == 0 || parent == nil {
		return nil
	}
	nextVal, _ := Recovered(func() any { return parent.Value(rem - 1) })
	switch v := nextVal.(type) {
	case *lineageNode:
		return v
	case []any:
		if len(v) > 0 {
			return &lineageNode{extra: v}
		}
	}
	return nil
}

func flattenLineageValue(v any, limit int) []any {
	switch t := v.(type) {
	case []any:
		return t
	case *lineageNode:
		out := make([]any, 0, countLineageNodes(t, limit))
		return appendLineageNodes(out, t, limit)
	default:
		return nil
	}
}

func lineageStep[T any](key any, c *T, runTok any, parent context.Context) (any, bool) {
	rem, ok := key.(lineageKey)
	if !ok {
		return nil, false
	}
	if c == nil {
		return nil, true
	}
	tok := contextToken{ptr: weak.Make(c)}
	tail := collectLineageTail(rem, parent)
	var rtNode *lineageNode
	if rt, ok := runTok.(*runToken); ok && rt != nil {
		// Avoid duplicating an identical runToken right below when inner ADK
		// hops share the same agent run.
		if tail == nil || (tail.tok != rt && (tail.next == nil || tail.next.tok != rt)) {
			rtNode = &lineageNode{tok: rt, next: tail}
		}
	}
	if rem < LineageKey {
		if rtNode != nil {
			return &lineageNode{tok: tok, next: rtNode}, true
		}
		return &lineageNode{tok: tok, next: tail}, true
	}
	if rtNode == nil && tail == nil {
		return []any{tok}, true
	}
	nextNode := tail
	if rtNode != nil {
		nextNode = rtNode
	}
	head := &lineageNode{tok: tok, next: nextNode}
	return flattenLineageValue(head, 2*(int(rem)+1)), true
}

// LineageStep handles an ADK lineage lookup on heap-allocated ADK context c
// with embedded parent context parent. When key is not a lineage lookup key, it
// returns (nil, false) so the caller delegates key unchanged. Inner recursive
// hops return an immutable linked node in O(1) without copying ancestor tails,
// and the outermost LineageKey hop flattens the chain once into []any in O(D).
func LineageStep[T any](key any, c *T, parent context.Context) (any, bool) {
	return lineageStep(key, c, nil, parent)
}

// LineageStepWithRunToken behaves like LineageStep and also emits a non-nil
// runToken allocated by NewRunToken so child runs launched after WithICDelta
// or WithContext copies still match the launching run's activation token.
func LineageStepWithRunToken[T any](key any, c *T, runTok any, parent context.Context) (any, bool) {
	return lineageStep(key, c, runTok, parent)
}

// ForwardLineage forwards a lineage lookup across a forwarding-only ADK context
// without its own node identity, decrementing the hop budget so cyclic chains
// through it terminate cleanly.
func ForwardLineage(key any, parent context.Context) (any, bool) {
	rem, ok := key.(lineageKey)
	if !ok {
		return nil, false
	}
	if parent == nil || rem == 0 {
		return nil, true
	}
	v, _ := Recovered(func() any { return parent.Value(rem - 1) })
	if rem < LineageKey {
		return v, true
	}
	return flattenLineageValue(v, 2*(int(rem)+1)), true
}

// WalkLineage returns the slice of opaque ADK context tokens observed along ctx
// via Value(LineageKey), or nil if ctx is nil, panics, or carries none.
func WalkLineage(ctx context.Context) []any {
	if ctx == nil {
		return nil
	}
	v, _ := Recovered(func() any { return ctx.Value(LineageKey) })
	return flattenLineageValue(v, 2*(int(LineageKey)+1))
}

func pathBudget(key any) (int, bool) {
	if key == PathKey {
		return int(maxHopBudget), true
	}
	h, ok := key.(pathHop)
	if !ok || int(h) < 0 || int(h) > int(maxHopBudget) {
		return 0, false
	}
	return int(h), true
}

// IsPathKey reports whether key is the private ADK node-path key or one of its
// internal hop-budget keys.
func IsPathKey(key any) bool {
	_, ok := pathBudget(key)
	return ok
}

func parentMapBudget(key any) (int, bool) {
	if key == ParentMapKey {
		return int(maxHopBudget), true
	}
	h, ok := key.(parentMapHop)
	if !ok || int(h) < 0 || int(h) > int(maxHopBudget) {
		return 0, false
	}
	return int(h), true
}

// IsParentMapKey reports whether key is the private ADK parent-map key or one of
// its internal hop-budget keys.
func IsParentMapKey(key any) bool {
	_, ok := parentMapBudget(key)
	return ok
}

// IsRunTokenKey reports whether key is the private ADK per-run token key or
// one of its internal hop-budget keys.
func IsRunTokenKey(key any) bool {
	_, ok := key.(runTokenHop)
	return ok
}

type parentMapCtx struct {
	context.Context
	m any
}

func (c *parentMapCtx) Value(key any) any {
	if _, ok := parentMapBudget(key); ok {
		return c.m
	}
	if c.Context == nil {
		return nil
	}
	return c.Context.Value(key)
}

// WithParentMap attaches the runner's child-to-parent agent map m to ctx so
// both direct Value(ParentMapKey) lookups and hop-budgeted ReadParentMap walks
// resolve across up to 64 ADK hops while terminating cleanly on ADK cycles.
func WithParentMap(ctx context.Context, m any) context.Context {
	return context.WithValue(&parentMapCtx{Context: ctx, m: m}, ParentMapKey, m)
}

// PathStep answers a hop-budgeted PathKey lookup for an ADK context. When path
// is non-empty it returns path immediately; otherwise it delegates to parent
// with a decremented hop budget until empty or exhausted at 64 ADK hops.
func PathStep(key any, path string, parent context.Context) (any, bool) {
	rem, ok := pathBudget(key)
	if !ok {
		return nil, false
	}
	if path != "" {
		return path, true
	}
	if parent == nil || rem == 0 {
		return nil, true
	}
	v, _ := Recovered(func() any { return parent.Value(pathHop(rem - 1)) })
	return v, true
}

// ParentMapStep answers a hop-budgeted ParentMapKey lookup for an ADK context,
// delegating to parent with a decremented hop budget up to 64 ADK hops.
func ParentMapStep(key any, parent context.Context) (any, bool) {
	rem, ok := parentMapBudget(key)
	if !ok {
		return nil, false
	}
	if parent == nil || rem == 0 {
		return nil, true
	}
	v, _ := Recovered(func() any { return parent.Value(parentMapHop(rem - 1)) })
	return v, true
}

// RunTokenStep answers a hop-budgeted RunTokenKey lookup for an ADK context.
// When runTok is non-nil it returns runTok immediately; otherwise it delegates
// to parent with a decremented hop budget up to 64 ADK hops.
func RunTokenStep(key, runTok any, parent context.Context) (any, bool) {
	rem, ok := key.(runTokenHop)
	if !ok {
		return nil, false
	}
	if runTok != nil {
		return runTok, true
	}
	if parent == nil || rem == 0 {
		return nil, true
	}
	v, _ := Recovered(func() any { return parent.Value(rem - 1) })
	return v, true
}

// ReadPath returns the workflow node path recovered from ctx via Value(PathKey),
// or "" if ctx is nil, panics, exceeds the hop budget, or carries no node path.
func ReadPath(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	v, ok := Recovered(func() any { return ctx.Value(PathKey) })
	if !ok {
		return ""
	}
	s, _ := v.(string)
	return s
}

// ReadRunToken returns the non-nil per-run identity token recovered from ctx via
// Value(RunTokenKey), or nil if ctx is nil, panics, or carries no valid token.
func ReadRunToken(ctx context.Context) any {
	if ctx == nil {
		return nil
	}
	v, ok := Recovered(func() any { return ctx.Value(RunTokenKey) })
	if !ok {
		return nil
	}
	tok, _ := v.(*runToken)
	if tok == nil {
		return nil
	}
	return tok
}

func isNilish(rv reflect.Value) bool {
	for range 4 {
		if !rv.IsValid() {
			return true
		}
		switch rv.Kind() {
		case reflect.Interface:
			if rv.IsNil() {
				return true
			}
			rv = rv.Elem()
		case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func, reflect.UnsafePointer:
			return rv.IsNil()
		default:
			return false
		}
	}
	return false
}

// ReadParentMap returns a shallow snapshot copy of the child-to-parent map
// recovered from ctx via Value(ParentMapKey), or nil if ctx is nil, panics, or
// carries no non-nil map. Nil and typed-nil entries are omitted so callers
// (including via agent.Promote(nil).ADKParentMap) cannot mutate the runner's
// reused map.
func ReadParentMap(ctx context.Context) any {
	if ctx == nil {
		return nil
	}
	v, ok := Recovered(func() any { return ctx.Value(ParentMapKey) })
	if !ok || v == nil {
		return nil
	}
	out, ok := Recovered(func() any {
		rv := reflect.ValueOf(v)
		if !rv.IsValid() || rv.Kind() != reflect.Map || rv.IsNil() {
			return nil
		}
		cp := reflect.MakeMapWithSize(rv.Type(), rv.Len())
		iter := rv.MapRange()
		for iter.Next() {
			val := iter.Value()
			if isNilish(val) {
				continue
			}
			cp.SetMapIndex(iter.Key(), val)
		}
		return cp.Interface()
	})
	if !ok {
		return nil
	}
	return out
}
