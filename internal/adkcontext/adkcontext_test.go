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

// Package adkcontext_test is an external test package on purpose: for method
// sets it behaves exactly as any package outside this module does, which is the
// party the trust boundary is meant to exclude. An in-package test could satisfy
// [adkcontext.Source] with an unexported method of its own and would prove
// nothing.
package adkcontext_test

import (
	"context"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"google.golang.org/adk/v2/internal/adkcontext"
)

// TestSourceMethodIsUnexported guards the property the whole mechanism rests on.
//
// Go satisfies interfaces structurally, and an unexported method name is
// qualified by its declaring package while an exported one is not. So
// [adkcontext.Source] excludes outside types only while its single method stays
// unexported. Rename it and any type anywhere satisfies Source without importing
// this package — and agent's identity procedure ASKS a Source for the invocation
// it speaks for, rather than reading its session, which is the one arm that
// extends real trust.
//
// Nothing else checks this. The rename was tried and the entire suite stayed
// green, and the doc on the internal package invites reading the internal/
// directory as sufficient on its own, which it is not: internal/ governs who may
// import, never who may satisfy.
func TestSourceMethodIsUnexported(t *testing.T) {
	src := reflect.TypeFor[adkcontext.Source]()
	if got := src.NumMethod(); got != 1 {
		t.Fatalf("Source has %d methods, want exactly 1; this test knows how to check one", got)
	}
	// PkgPath is empty for an exported method and the declaring package's path
	// for an unexported one.
	// The method's OWN package must be internal too, not just the key's. Splitting
	// the package — Source and Marker public, the key left behind — leaves every
	// other check here green while letting any type anywhere embed Marker and be
	// ASKED for an identity. Two holes of this class have already been found by
	// hand. This is the third, and it is the same one test away.
	if m := src.Method(0); !slices.Contains(strings.Split(m.PkgPath, "/"), "internal") {
		t.Errorf("Source.%s is declared in %q, which is importable from anywhere. Any importer "+
			"could then embed Marker and be taken for one of ADK's own contexts. Keep the "+
			"marker under internal/.", m.Name, m.PkgPath)
	}
	if m := src.Method(0); m.PkgPath == "" {
		t.Errorf("Source.%s is exported. Any type outside this module can then declare a "+
			"method of that name and be taken for one of ADK's own contexts, which the "+
			"identity procedure asks for the user it speaks for instead of reading its "+
			"session. Keep the method unexported.", m.Name)
	}
}

// TestIdentityKeyTypeIsUnnameable guards the other half of the boundary.
//
// Source decides who gets ASKED for an identity. The key decides who can PLANT
// one, and it is trusted by everything rather than by one arm, so it is the
// stronger of the two properties. It holds only while its type is unexported and
// declared here: a context key compares by dynamic type as well as value, so an
// untyped `const IdentityKey = 0`, or an exported `CtxKey`, would let any package
// forge an identity with context.WithValue(ctx, 0, agent.Identity{...}) and have
// the credential path mint for it.
//
// Measured before writing this: with the constant made untyped the forgery
// succeeds and the entire suite stays green. That is the same shape as the
// unexported-method hole above, which also went unnoticed until someone tried
// the rename.
func TestIdentityKeyTypeIsUnnameable(t *testing.T) {
	kt := reflect.TypeOf(adkcontext.IdentityKey)
	if kt.PkgPath() == "" {
		t.Fatalf("IdentityKey has type %s, which any package can name and plant with "+
			"context.WithValue. The identity would then be forgeable by unrelated code.", kt)
	}
	if name := kt.Name(); name == "" || name[0] >= 'A' && name[0] <= 'Z' {
		t.Errorf("IdentityKey's type is %s.%s, which is exported; any package could declare "+
			"a value of it and plant an identity. Keep the key's type unexported.",
			kt.PkgPath(), name)
	}
	// An unexported type is not enough on its own. IdentityKey is itself
	// exported, so anything able to IMPORT this package can plant an identity
	// with it and never name the type at all — the only thing preventing that is
	// the internal/ element in the path. Both checks above survive moving this
	// package somewhere public, which is a mechanical rename, so the placement is
	// asserted too.
	if !slices.Contains(strings.Split(kt.PkgPath(), "/"), "internal") {
		t.Errorf("IdentityKey is declared in %s, which is importable from anywhere. The key "+
			"is exported, so any importer can plant an identity with it, and any importer "+
			"can embed Marker to be asked for one. Keep this package under internal/.",
			kt.PkgPath())
	}
}

// exportedLookalike is what an outside type would declare to impersonate a
// Source if the marker method were ever exported.
type exportedLookalike struct{}

func (exportedLookalike) AdkIdentitySource() {}

// unexportedLookalike declares the real name from a different package. Go treats
// that as a different method, which is the guarantee under test.
type unexportedLookalike struct{}

func (unexportedLookalike) adkIdentitySource() {}

func TestALookalikeFromAnotherPackageIsNotASource(t *testing.T) {
	// Called so the method is a genuine use rather than a suppressed warning: it
	// has to exist for the assertion below to mean anything, and a linter
	// directive would only hide the day it stops existing.
	unexportedLookalike{}.adkIdentitySource()

	for _, tc := range []struct {
		name string
		v    any
	}{
		{"exported name", exportedLookalike{}},
		{"the real name, declared elsewhere", unexportedLookalike{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, ok := tc.v.(adkcontext.Source); ok {
				t.Errorf("%T satisfies adkcontext.Source from outside the declaring package, "+
					"so the trust boundary is open: such a type is asked for the invocation it "+
					"speaks for rather than read.", tc.v)
			}
		})
	}
}

// Embedding Marker is the sanctioned way in, and it has to keep working — a
// guard that also broke the legitimate path would be swapped out rather than
// fixed.
type sanctioned struct{ adkcontext.Marker }

func TestEmbeddingMarkerStillSatisfiesSource(t *testing.T) {
	if _, ok := any(sanctioned{}).(adkcontext.Source); !ok {
		t.Error("a type embedding adkcontext.Marker does not satisfy adkcontext.Source")
	}
}

func TestLineageKeyTypeIsUnnameable(t *testing.T) {
	kt := reflect.TypeOf(adkcontext.LineageKey)
	if kt.PkgPath() == "" {
		t.Fatalf("LineageKey has type %s, which any package can name", kt)
	}
	if name := kt.Name(); name == "" || name[0] >= 'A' && name[0] <= 'Z' {
		t.Errorf("LineageKey's type is %s.%s, which is exported; keep the key's type unexported", kt.PkgPath(), name)
	}
	if !slices.Contains(strings.Split(kt.PkgPath(), "/"), "internal") {
		t.Errorf("LineageKey is declared in %s, which is importable from anywhere; keep this package under internal/", kt.PkgPath())
	}
}

type appObserverKey struct{}

func (appObserverKey) ObserveADKContext(any) bool {
	return true
}

type sampleStepCtx struct {
	context.Context
	id int
}

func (c *sampleStepCtx) Value(key any) any {
	if v, ok := adkcontext.LineageStep(key, c, c.Context); ok {
		return v
	}
	if c.Context == nil {
		return nil
	}
	return c.Context.Value(key)
}

type cyclicStepBridge struct {
	context.Context
	loop context.Context
}

func (c *cyclicStepBridge) Value(key any) any {
	return c.loop.Value(key)
}

func TestLineageStepTokenIsOpaqueImmutableAndBounded(t *testing.T) {
	if got := adkcontext.Token((*sampleStepCtx)(nil)); got != nil {
		t.Fatalf("Token(nil) = %v, want nil", got)
	}
	var nilCtx context.Context
	if got := adkcontext.WalkLineage(nilCtx); got != nil {
		t.Fatalf("WalkLineage(nilCtx) = %v, want nil", got)
	}
	if v, ok := adkcontext.LineageStep(adkcontext.LineageKey, (*sampleStepCtx)(nil), context.Background()); !ok || v != nil {
		t.Fatalf("LineageStep(nil receiver) = (%v, %v), want (nil, true)", v, ok)
	}
	if v, ok := adkcontext.LineageStep(appObserverKey{}, &sampleStepCtx{id: 1}, context.Background()); ok || v != nil {
		t.Fatalf("LineageStep(appObserverKey) = (%v, %v), want (nil, false)", v, ok)
	}

	base := context.WithValue(context.Background(), appObserverKey{}, "app-payload")
	a := &sampleStepCtx{Context: base, id: 1}
	b := &sampleStepCtx{Context: a, id: 2}

	if got := b.Value(appObserverKey{}); got != "app-payload" {
		t.Fatalf("Value(appObserverKey{}) = %v, want %q", got, "app-payload")
	}

	tokens := adkcontext.WalkLineage(b)
	if len(tokens) != 2 {
		t.Fatalf("len(WalkLineage(b)) = %d, want 2", len(tokens))
	}
	if tokens[0] != adkcontext.Token(b) || tokens[1] != adkcontext.Token(a) || tokens[0] == tokens[1] {
		t.Fatalf("token mismatch: %#v", tokens)
	}
	seen := map[any]int{tokens[0]: 1, tokens[1]: 2}
	if seen[adkcontext.Token(b)] != 1 || seen[adkcontext.Token(a)] != 2 {
		t.Fatalf("map lookup failed: %#v", seen)
	}

	rv := reflect.ValueOf(tokens[0])
	if rv.NumMethod() != 0 || rv.MethodByName("Value").IsValid() {
		t.Fatalf("token exposes methods (NumMethod=%d); weak.Pointer.Value must not be reachable", rv.NumMethod())
	}
	if rv.Kind() != reflect.Struct || rv.NumField() != 1 || rv.Field(0).CanInterface() {
		t.Fatalf("token struct field must be single and unexported (Kind=%v, NumField=%d)", rv.Kind(), rv.NumField())
	}

	// Concurrent lookups must return independent immutable slices without data races.
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			got := adkcontext.WalkLineage(b)
			if len(got) != 2 || got[0] != tokens[0] || got[1] != tokens[1] {
				t.Errorf("concurrent WalkLineage = %#v, want %#v", got, tokens)
			}
		})
	}
	wg.Wait()

	// Cyclic Value chain through ADK step terminates cleanly at LineageKey+1 hops.
	bridge := &cyclicStepBridge{Context: context.Background()}
	cyc := &sampleStepCtx{Context: bridge, id: 9}
	bridge.loop = cyc
	bounded := adkcontext.WalkLineage(cyc)
	if len(bounded) != int(adkcontext.LineageKey)+1 {
		t.Fatalf("len(WalkLineage(cyclic)) = %d, want %d", len(bounded), int(adkcontext.LineageKey)+1)
	}
}

func build64LayerChain() (context.Context, []*sampleStepCtx) {
	const depth = 64
	layers := make([]*sampleStepCtx, depth)
	cur := context.Background()
	for i := range depth {
		node := &sampleStepCtx{Context: cur, id: i + 1}
		layers[i] = node
		cur = node
	}
	return cur, layers
}

func TestWalkLineage_64LayersOrderAndLinearAllocs(t *testing.T) {
	top, layers := build64LayerChain()
	tokens := adkcontext.WalkLineage(top)
	if len(tokens) != len(layers) {
		t.Fatalf("len(WalkLineage(64 layers)) = %d, want %d", len(tokens), len(layers))
	}
	for i, wantNode := range layers {
		// Outermost token sits at index 0; deepest sits at index len(layers)-1.
		got := tokens[len(layers)-1-i]
		if want := adkcontext.Token(wantNode); got != want {
			t.Fatalf("token mismatch at layer %d: got %#v, want %#v", i, got, want)
		}
	}

	// Verify linear O(D) heap usage: 64 layers with 64 hops would copy >2,000
	// interface slots quadratically under per-hop slice copying; linked nodes
	// keep total allocs linear in depth.
	if allocs := testing.AllocsPerRun(50, func() {
		got := adkcontext.WalkLineage(top)
		if len(got) != len(layers) {
			t.Fatalf("unexpected len=%d", len(got))
		}
	}); allocs > 300 {
		t.Fatalf("WalkLineage(64 layers) allocs = %v, want linear <= 300", allocs)
	}
}

func BenchmarkWalkLineage_64Layers(b *testing.B) {
	top, layers := build64LayerChain()
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		got := adkcontext.WalkLineage(top)
		if len(got) != len(layers) {
			b.Fatalf("len(WalkLineage) = %d, want %d", len(got), len(layers))
		}
	}
}

func TestReadPathAndParentMap(t *testing.T) {
	if !adkcontext.IsPathKey(adkcontext.PathKey) || adkcontext.IsPathKey(adkcontext.IdentityKey) {
		t.Fatal("IsPathKey returned unexpected result")
	}
	if !adkcontext.IsParentMapKey(adkcontext.ParentMapKey) || adkcontext.IsParentMapKey(adkcontext.PathKey) {
		t.Fatal("IsParentMapKey returned unexpected result")
	}
	var nilCtx context.Context
	if adkcontext.ReadPath(nilCtx) != "" || adkcontext.ReadParentMap(nilCtx) != nil {
		t.Fatal("nil context should return empty path and nil parent map")
	}
	if v, ok := adkcontext.PathStep(adkcontext.PathKey, "wf/a", nil); !ok || v != "wf/a" {
		t.Fatalf("PathStep(wf/a) = (%v, %v), want (wf/a, true)", v, ok)
	}
	if v, ok := adkcontext.PathStep(adkcontext.IdentityKey, "wf/a", nil); ok || v != nil {
		t.Fatalf("PathStep(IdentityKey) = (%v, %v), want (nil, false)", v, ok)
	}
	base := context.WithValue(context.Background(), adkcontext.ParentMapKey, map[string]string{"leaf": "root"})
	withPath := context.WithValue(base, adkcontext.PathKey, "wf/parent")
	if v, ok := adkcontext.PathStep(adkcontext.PathKey, "", withPath); !ok || v != "wf/parent" {
		t.Fatalf("PathStep(delegate) = (%v, %v), want (wf/parent, true)", v, ok)
	}
	if got := adkcontext.ReadPath(withPath); got != "wf/parent" {
		t.Fatalf("ReadPath = %q, want wf/parent", got)
	}
	if m, ok := adkcontext.ReadParentMap(withPath).(map[string]string); !ok || m["leaf"] != "root" {
		t.Fatalf("ReadParentMap = %#v, want leaf->root", adkcontext.ReadParentMap(withPath))
	}
}
