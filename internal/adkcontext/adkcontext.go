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
	"runtime"
	"slices"
	"sync"
	"weak"
)

type ctxKey int

type pathKeyType int

type parentMapKeyType int

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
const PathKey pathKeyType = 1

// ParentMapKey is the private context value key under which ADK's runner
// attaches the child-to-parent agent map.
const ParentMapKey parentMapKeyType = 2

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

// StepSource marks ADK's own internal context implementations whose Value
// methods participate in hop-budgeted private-key propagation. Because the
// interface method is unexported in an internal package, external context
// wrappers outside this module cannot implement it.
type StepSource interface{ adkStepSource() }

// StepMarker is embedded by ADK context types that satisfy [StepSource]. It
// carries no state.
type StepMarker struct{}

func (StepMarker) adkStepSource() {}

const maxStdlibHops = 512

var (
	contextIfaceType     = reflect.TypeFor[context.Context]()
	stepSourceType       = reflect.TypeFor[StepSource]()
	unwrapperIfaceType   = reflect.TypeFor[interface{ Unwrap() context.Context }]()
	valueCarrierType     = reflect.TypeFor[interface{ Value(any) any }]()
	syncLockerType       = reflect.TypeFor[sync.Locker]()
	pathRootKeyType      = reflect.TypeFor[pathKeyType]()
	parentMapRootKeyType = reflect.TypeFor[parentMapKeyType]()
	agentPkgPath         = deriveAgentPkgPath()
	lockFreePlanCache    sync.Map // map[reflect.Type]lockFreeContextPlan
)

type lockFreeContextPlan struct {
	ok   bool
	path []int
}

func deriveAgentPkgPath() string {
	p := reflect.TypeFor[StepMarker]().PkgPath()
	const suffix = "/internal/adkcontext"
	if len(p) > len(suffix) && p[len(p)-len(suffix):] == suffix {
		return p[:len(p)-len(suffix)] + "/agent"
	}
	return ""
}

func isStrictContextMockType(t reflect.Type) bool {
	return agentPkgPath != "" && t != nil && t.PkgPath() == agentPkgPath && t.Name() == "StrictContextMock"
}

func cachedLockFreeContextPlan(t reflect.Type) lockFreeContextPlan {
	if t == nil {
		return lockFreeContextPlan{}
	}
	if cached, ok := lockFreePlanCache.Load(t); ok {
		return cached.(lockFreeContextPlan)
	}
	plan := computeLockFreeContextPlan(t)
	actual, _ := lockFreePlanCache.LoadOrStore(t, plan)
	return actual.(lockFreeContextPlan)
}

func declaredMethodOnType(t reflect.Type, addressable bool, name string) bool {
	check := func(cand reflect.Type) bool {
		m, ok := cand.MethodByName(name)
		if !ok || !m.Func.IsValid() || m.Func.Pointer() == 0 {
			return false
		}
		fn := runtime.FuncForPC(m.Func.Pointer())
		if fn == nil {
			return false
		}
		file, _ := fn.FileLine(fn.Entry())
		return file != "" && file != "<autogenerated>"
	}
	if check(t) {
		return true
	}
	return addressable && t.Kind() != reflect.Pointer && check(reflect.PointerTo(t))
}

func derefTypeBounded(t reflect.Type, budget *int) (reflect.Type, int) {
	var seen map[reflect.Type]bool
	for t != nil && t.Kind() == reflect.Pointer {
		if seen[t] {
			return nil, 1
		}
		if *budget <= 0 {
			return nil, 2
		}
		*budget--
		if seen == nil {
			seen = make(map[reflect.Type]bool, 4)
		}
		seen[t] = true
		elem := t.Elem()
		if elem == t {
			return nil, 1
		}
		t = elem
	}
	if t == nil {
		return nil, 1
	}
	return t, 0
}

func typeHasSyncPrimitive(t reflect.Type, depth int, budget *int, inStack map[reflect.Type]bool) bool {
	if depth >= 8 || t == nil {
		return true
	}
	base, status := derefTypeBounded(t, budget)
	if status == 1 {
		return false
	}
	if status == 2 {
		return true
	}
	pkg := base.PkgPath()
	if pkg == "sync" || pkg == "sync/atomic" {
		return true
	}
	if base.Implements(syncLockerType) {
		return true
	}
	if base.Kind() != reflect.Pointer && reflect.PointerTo(base).Implements(syncLockerType) {
		return true
	}
	switch base.Kind() {
	case reflect.Slice, reflect.Array:
		return typeHasSyncPrimitive(base.Elem(), depth+1, budget, inStack)
	case reflect.Chan:
		return true
	case reflect.Map:
		return typeHasSyncPrimitive(base.Key(), depth+1, budget, inStack) ||
			typeHasSyncPrimitive(base.Elem(), depth+1, budget, inStack)
	case reflect.Struct:
		if inStack[base] {
			return false
		}
		if *budget <= 0 {
			return true
		}
		*budget--
		if inStack == nil {
			inStack = make(map[reflect.Type]bool, 8)
		}
		inStack[base] = true
		for f := range base.Fields() {
			if typeHasSyncPrimitive(f.Type, depth+1, budget, inStack) {
				return true
			}
		}
	}
	return false
}

func typeCarriesContextBounded(t reflect.Type, depth int, budget *int, inStack map[reflect.Type]bool) bool {
	if t == nil || depth >= 8 {
		return true
	}
	base, status := derefTypeBounded(t, budget)
	if status == 1 {
		return t.Implements(contextIfaceType)
	}
	if status == 2 {
		return true
	}
	switch base.Kind() {
	case reflect.Interface:
		return base.Implements(contextIfaceType)
	case reflect.Array, reflect.Slice:
		return typeCarriesContextBounded(base.Elem(), depth+1, budget, inStack)
	case reflect.Map:
		return typeCarriesContextBounded(base.Key(), depth+1, budget, inStack) ||
			typeCarriesContextBounded(base.Elem(), depth+1, budget, inStack)
	case reflect.Struct:
		if base.Implements(contextIfaceType) || reflect.PointerTo(base).Implements(contextIfaceType) {
			return true
		}
		if inStack[base] {
			return false
		}
		if *budget <= 0 {
			return true
		}
		*budget--
		if inStack == nil {
			inStack = make(map[reflect.Type]bool, 8)
		}
		inStack[base] = true
		for sf := range base.Fields() {
			if typeCarriesContextBounded(sf.Type, depth+1, budget, inStack) {
				delete(inStack, base)
				return true
			}
		}
		delete(inStack, base)
		return false
	default:
		return base.Implements(contextIfaceType) || reflect.PointerTo(base).Implements(contextIfaceType)
	}
}

func computeLockFreeContextPlan(origVT reflect.Type) lockFreeContextPlan {
	if origVT == nil || !origVT.Implements(contextIfaceType) {
		return lockFreeContextPlan{}
	}
	startAddr := origVT.Kind() == reflect.Pointer
	vt := origVT
	if startAddr {
		vt = vt.Elem()
	}
	syncBudget := 256
	if vt.Kind() != reflect.Struct || typeHasSyncPrimitive(vt, 0, &syncBudget, nil) {
		return lockFreeContextPlan{}
	}
	if declaredMethodOnType(vt, startAddr, "Value") {
		return lockFreeContextPlan{}
	}
	type node struct {
		typ         reflect.Type
		addressable bool
		path        []int
	}
	type seenKey struct {
		typ         reflect.Type
		addressable bool
	}
	level := []node{{typ: vt, addressable: startAddr}}
	seenDepth := map[seenKey]int{{typ: vt, addressable: startAddr}: 0}
	var leafPath []int
	for depth := 0; depth < 8 && len(level) > 0; depth++ {
		var next []node
		var ctxFound []int
		ctxCount := 0
		customFound := false
		for _, cur := range level {
			if cur.typ.Kind() != reflect.Struct {
				continue
			}
			for sf := range cur.typ.Fields() {
				if !sf.Anonymous {
					continue
				}
				ft := sf.Type
				fieldPath := slices.Concat(cur.path, sf.Index)
				if ft.Kind() == reflect.Interface {
					if ft.Implements(contextIfaceType) {
						ctxCount++
						ctxFound = fieldPath
					} else if ft.Implements(valueCarrierType) {
						customFound = true
					}
					continue
				}
				childAddr := cur.addressable
				base := ft
				if base.Kind() == reflect.Pointer {
					childAddr = true
					base = base.Elem()
				}
				if base.Kind() != reflect.Interface && declaredMethodOnType(base, childAddr, "Value") {
					customFound = true
					continue
				}
				if base.Kind() != reflect.Struct {
					continue
				}
				cn := seenKey{typ: base, addressable: childAddr}
				if prevDepth, ok := seenDepth[cn]; ok && prevDepth < depth+1 {
					continue
				}
				seenDepth[cn] = depth + 1
				if len(next) >= 64 {
					return lockFreeContextPlan{}
				}
				next = append(next, node{typ: base, addressable: childAddr, path: fieldPath})
			}
		}
		if customFound || ctxCount > 1 {
			return lockFreeContextPlan{}
		}
		if ctxCount == 1 {
			leafPath = ctxFound
			break
		}
		level = next
	}
	if len(leafPath) == 0 {
		return lockFreeContextPlan{}
	}
	ctxBudget := 256
	curT := vt
	for hop, idx := range leafPath {
		var status int
		curT, status = derefTypeBounded(curT, &ctxBudget)
		if status != 0 || curT.Kind() != reflect.Struct || idx < 0 || idx >= curT.NumField() {
			return lockFreeContextPlan{}
		}
		sf := curT.Field(idx)
		if !sf.Anonymous || (hop == len(leafPath)-1 && !sf.IsExported()) {
			return lockFreeContextPlan{}
		}
		for fi := range curT.NumField() {
			if fi == idx {
				continue
			}
			if typeCarriesContextBounded(curT.Field(fi).Type, 0, &ctxBudget, nil) {
				return lockFreeContextPlan{}
			}
		}
		if hop < len(leafPath)-1 {
			curT = sf.Type
		}
	}
	return lockFreeContextPlan{ok: true, path: leafPath}
}

func stepLockFreePromotedField(rv reflect.Value, path []int) (reflect.Value, bool) {
	if len(path) == 0 {
		return reflect.Value{}, false
	}
	cur := rv
	var innerSeen map[visitedCtxPtr]bool
	for hop, idx := range path {
		for derefs := 0; cur.Kind() == reflect.Pointer; derefs++ {
			if derefs >= 16 || cur.IsNil() {
				return reflect.Value{}, false
			}
			if hop > 0 {
				key := visitedCtxPtr{ptr: cur.Pointer(), typ: cur.Type()}
				if innerSeen != nil && innerSeen[key] {
					return reflect.Value{}, false
				}
				if innerSeen == nil {
					innerSeen = make(map[visitedCtxPtr]bool, 4)
				}
				innerSeen[key] = true
			}
			cur = cur.Elem()
		}
		if cur.Kind() != reflect.Struct || idx < 0 || idx >= cur.NumField() {
			return reflect.Value{}, false
		}
		fv := cur.Field(idx)
		if hop < len(path)-1 {
			cur = fv
			continue
		}
		if fv.Kind() != reflect.Interface || !fv.Type().Implements(contextIfaceType) || fv.IsNil() {
			return reflect.Value{}, false
		}
		return fv, true
	}
	return reflect.Value{}, false
}

type visitedCtxPtr struct {
	ptr uintptr
	typ reflect.Type
}

func sameComparableCtx(a, b any) bool {
	if a == nil || b == nil {
		return a == b
	}
	ta := reflect.TypeOf(a)
	tb := reflect.TypeOf(b)
	if ta != tb || !ta.Comparable() {
		return false
	}
	return a == b
}

func baseStructValue(rv reflect.Value) (reflect.Value, bool) {
	for range 8 {
		if !rv.IsValid() {
			return reflect.Value{}, false
		}
		switch rv.Kind() {
		case reflect.Pointer, reflect.Interface:
			if rv.IsNil() {
				return reflect.Value{}, false
			}
			rv = rv.Elem()
		case reflect.Struct:
			return rv, true
		default:
			return reflect.Value{}, false
		}
	}
	return reflect.Value{}, false
}

func findStdlibParentField(sv reflect.Value) (reflect.Value, bool) {
	var found reflect.Value
	count := 0
	var scan func(v reflect.Value, depth int) bool
	scan = func(v reflect.Value, depth int) bool {
		if depth >= 4 || v.Kind() != reflect.Struct || v.Type().PkgPath() != "context" {
			return true
		}
		t := v.Type()
		for i := range t.NumField() {
			sf := t.Field(i)
			fv := v.Field(i)
			if sf.Type == contextIfaceType {
				count++
				if count > 1 {
					return false
				}
				found = fv
				continue
			}
			if sf.Anonymous && sf.Type.Kind() == reflect.Struct && sf.Type.PkgPath() == "context" {
				if !scan(fv, depth+1) {
					return false
				}
			}
		}
		return true
	}
	if !scan(sv, 0) || count != 1 {
		return reflect.Value{}, false
	}
	return found, true
}

func verifyStdlibChainToStepSource(fv reflect.Value, rem, stdlibHops int, seen map[visitedCtxPtr]bool, wantRootKeyType reflect.Type) (int, bool) {
	for fv.IsValid() && fv.Kind() == reflect.Interface {
		if fv.IsNil() {
			return 0, false
		}
		elem := fv.Elem()
		if !elem.IsValid() {
			return 0, false
		}
		et := elem.Type()
		if elem.Kind() == reflect.Pointer {
			if elem.IsNil() {
				return 0, false
			}
			key := visitedCtxPtr{ptr: elem.Pointer(), typ: et}
			if seen != nil && seen[key] {
				return 0, false
			}
			if seen == nil {
				seen = make(map[visitedCtxPtr]bool, 4)
			}
			seen[key] = true
		}
		if et.Implements(stepSourceType) {
			return rem, true
		}
		if et.Implements(unwrapperIfaceType) {
			return 0, false
		}
		sv, ok := baseStructValue(elem)
		if !ok {
			return 0, false
		}
		if isStrictContextMockType(sv.Type()) {
			if rem <= 0 {
				return 0, false
			}
			cf := sv.FieldByName("Ctx")
			if !cf.IsValid() || cf.Kind() != reflect.Interface || cf.IsNil() {
				return 0, false
			}
			rem--
			fv = cf
			continue
		}
		if sv.Type().PkgPath() != "context" {
			plan := cachedLockFreeContextPlan(et)
			if !plan.ok || rem <= 0 {
				return 0, false
			}
			nextFV, ok := stepLockFreePromotedField(elem, plan.path)
			if !ok {
				return 0, false
			}
			rem--
			fv = nextFV
			continue
		}
		if wantRootKeyType != nil && sv.Type().Name() == "valueCtx" {
			if kf := sv.FieldByName("key"); kf.IsValid() && !kf.IsNil() && kf.Elem().Type() == wantRootKeyType {
				return int(maxHopBudget), true
			}
		}
		if stdlibHops >= maxStdlibHops {
			return 0, false
		}
		stdlibHops++
		nextFV, ok := findStdlibParentField(sv)
		if !ok || nextFV.IsNil() {
			return 0, false
		}
		fv = nextFV
	}
	return 0, false
}

func resolvePrivateKeyTarget(ctx context.Context, rem int, wantRootKeyType reflect.Type) (context.Context, int, bool) {
	if rem < 0 {
		return nil, 0, false
	}
	var firstPtr visitedCtxPtr
	var hasFirst bool
	var seen map[visitedCtxPtr]bool
	stdlibHops := 0
	for ctx != nil {
		rv := reflect.ValueOf(ctx)
		isPtr := rv.Kind() == reflect.Pointer
		if isPtr && rv.IsNil() {
			return nil, 0, false
		}
		if _, ok := ctx.(StepSource); ok {
			return ctx, rem, true
		}
		if isPtr {
			key := visitedCtxPtr{ptr: rv.Pointer(), typ: rv.Type()}
			if !hasFirst {
				firstPtr = key
				hasFirst = true
			} else if key == firstPtr {
				return nil, 0, false
			} else {
				if seen == nil {
					seen = make(map[visitedCtxPtr]bool, 4)
					seen[firstPtr] = true
				}
				if seen[key] {
					return nil, 0, false
				}
				seen[key] = true
			}
		}
		if u, ok := ctx.(interface{ Unwrap() context.Context }); ok {
			if rem <= 0 {
				return nil, 0, false
			}
			next, recOK := Recovered(u.Unwrap)
			rem--
			if !recOK || next == nil || sameComparableCtx(next, ctx) {
				return nil, 0, false
			}
			ctx = next
			continue
		}
		sv, ok := baseStructValue(rv)
		if !ok {
			return nil, 0, false
		}
		if isStrictContextMockType(sv.Type()) {
			if rem <= 0 {
				return nil, 0, false
			}
			cf := sv.FieldByName("Ctx")
			if !cf.IsValid() || cf.Kind() != reflect.Interface || cf.IsNil() || !cf.CanInterface() {
				return nil, 0, false
			}
			next, _ := reflect.TypeAssert[context.Context](cf)
			rem--
			if next == nil || sameComparableCtx(next, ctx) {
				return nil, 0, false
			}
			ctx = next
			continue
		}
		if sv.Type().PkgPath() != "context" {
			plan := cachedLockFreeContextPlan(rv.Type())
			if !plan.ok || rem <= 0 {
				return nil, 0, false
			}
			fv, ok := stepLockFreePromotedField(rv, plan.path)
			if !ok || !fv.CanInterface() {
				return nil, 0, false
			}
			next, _ := reflect.TypeAssert[context.Context](fv)
			rem--
			if next == nil || sameComparableCtx(next, ctx) {
				return nil, 0, false
			}
			ctx = next
			continue
		}
		if wantRootKeyType != nil && sv.Type().Name() == "valueCtx" {
			if kf := sv.FieldByName("key"); kf.IsValid() && !kf.IsNil() && kf.Elem().Type() == wantRootKeyType {
				return ctx, int(maxHopBudget), true
			}
		}
		if stdlibHops >= maxStdlibHops {
			return nil, 0, false
		}
		stdlibHops++
		fv, ok := findStdlibParentField(sv)
		if !ok || fv.IsNil() {
			return nil, 0, false
		}
		if fv.CanInterface() {
			next, _ := reflect.TypeAssert[context.Context](fv)
			if next == nil || sameComparableCtx(next, ctx) {
				return nil, 0, false
			}
			ctx = next
			continue
		}
		nextRem, verified := verifyStdlibChainToStepSource(fv, rem, stdlibHops, seen, wantRootKeyType)
		if !verified {
			return nil, 0, false
		}
		return ctx, nextRem, true
	}
	return nil, 0, false
}

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
	tgt, nextRem, ok := resolvePrivateKeyTarget(parent, int(rem)-1, nil)
	if !ok {
		return nil
	}
	nextVal, _ := Recovered(func() any { return tgt.Value(lineageKey(nextRem)) })
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
	tgt, nextRem, ok := resolvePrivateKeyTarget(parent, int(rem)-1, nil)
	if !ok {
		return nil, true
	}
	v, _ := Recovered(func() any { return tgt.Value(lineageKey(nextRem)) })
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
	tgt, nextRem, ok := resolvePrivateKeyTarget(ctx, int(maxHopBudget), nil)
	if !ok {
		return nil
	}
	v, _ := Recovered(func() any { return tgt.Value(lineageKey(nextRem)) })
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
	StepMarker
	context.Context
	m any
}

func (c *parentMapCtx) Value(key any) any {
	if c == nil {
		return nil
	}
	if _, ok := parentMapBudget(key); ok {
		return c.m
	}
	if v, ok := ForwardLineage(key, c.Context); ok {
		return v
	}
	if v, ok := PathStep(key, "", c.Context); ok {
		return v
	}
	if v, ok := RunTokenStep(key, nil, c.Context); ok {
		return v
	}
	if c.Context == nil {
		return nil
	}
	return c.Context.Value(key)
}

var _ StepSource = (*parentMapCtx)(nil)

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
	tgt, nextRem, ok := resolvePrivateKeyTarget(parent, rem-1, pathRootKeyType)
	if !ok {
		return nil, true
	}
	var lookupKey any = pathHop(nextRem)
	if nextRem == int(maxHopBudget) {
		lookupKey = PathKey
	}
	v, _ := Recovered(func() any { return tgt.Value(lookupKey) })
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
	tgt, nextRem, ok := resolvePrivateKeyTarget(parent, rem-1, parentMapRootKeyType)
	if !ok {
		return nil, true
	}
	var lookupKey any = parentMapHop(nextRem)
	if nextRem == int(maxHopBudget) {
		lookupKey = ParentMapKey
	}
	v, _ := Recovered(func() any { return tgt.Value(lookupKey) })
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
	tgt, nextRem, ok := resolvePrivateKeyTarget(parent, int(rem)-1, nil)
	if !ok {
		return nil, true
	}
	v, _ := Recovered(func() any { return tgt.Value(runTokenHop(nextRem)) })
	return v, true
}

// ReadPath returns the workflow node path recovered from ctx via Value(PathKey),
// or "" if ctx is nil, panics, exceeds the hop budget, or carries no node path.
func ReadPath(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	tgt, nextRem, ok := resolvePrivateKeyTarget(ctx, int(maxHopBudget), pathRootKeyType)
	if !ok {
		return ""
	}
	var lookupKey any = pathHop(nextRem)
	if nextRem == int(maxHopBudget) {
		lookupKey = PathKey
	}
	v, recOK := Recovered(func() any { return tgt.Value(lookupKey) })
	if !recOK {
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
	tgt, nextRem, ok := resolvePrivateKeyTarget(ctx, int(maxHopBudget), nil)
	if !ok {
		return nil
	}
	v, recOK := Recovered(func() any { return tgt.Value(runTokenHop(nextRem)) })
	if !recOK {
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
	tgt, nextRem, ok := resolvePrivateKeyTarget(ctx, int(maxHopBudget), parentMapRootKeyType)
	if !ok {
		return nil
	}
	var lookupKey any = parentMapHop(nextRem)
	if nextRem == int(maxHopBudget) {
		lookupKey = ParentMapKey
	}
	v, recOK := Recovered(func() any { return tgt.Value(lookupKey) })
	if !recOK || v == nil {
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
