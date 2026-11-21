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
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

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

func TestStepSourceMethodIsUnexported(t *testing.T) {
	src := reflect.TypeFor[adkcontext.StepSource]()
	if got := src.NumMethod(); got != 1 {
		t.Fatalf("StepSource has %d methods, want exactly 1", got)
	}
	if m := src.Method(0); !slices.Contains(strings.Split(m.PkgPath, "/"), "internal") {
		t.Errorf("StepSource.%s is declared in %q, must remain under internal/", m.Name, m.PkgPath)
	}
	if m := src.Method(0); m.PkgPath == "" {
		t.Errorf("StepSource.%s is exported; must remain unexported", m.Name)
	}
}

type appObserverKey struct{}

func (appObserverKey) ObserveADKContext(any) bool {
	return true
}

type sampleStepCtx struct {
	adkcontext.StepMarker
	context.Context
	id int
}

func (c *sampleStepCtx) Value(key any) any {
	if v, ok := adkcontext.LineageStep(key, c, c.Context); ok {
		return v
	}
	if v, ok := adkcontext.PathStep(key, "", c.Context); ok {
		return v
	}
	if v, ok := adkcontext.ParentMapStep(key, c.Context); ok {
		return v
	}
	if v, ok := adkcontext.RunTokenStep(key, nil, c.Context); ok {
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

type cyclicUnwrapBridge struct {
	loop context.Context
}

func (*cyclicUnwrapBridge) Deadline() (time.Time, bool) { return time.Time{}, false }
func (*cyclicUnwrapBridge) Done() <-chan struct{}       { return nil }
func (*cyclicUnwrapBridge) Err() error                  { return nil }
func (c *cyclicUnwrapBridge) Value(key any) any         { return c.loop.Value(key) }
func (c *cyclicUnwrapBridge) Unwrap() context.Context   { return c.loop }

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

	// Cyclic ADK step chain terminates cleanly at LineageKey+1 hops.
	cycA := &sampleStepCtx{id: 9}
	cycB := &sampleStepCtx{Context: cycA, id: 10}
	cycA.Context = cycB
	bounded := adkcontext.WalkLineage(cycA)
	if len(bounded) != int(adkcontext.LineageKey)+1 {
		t.Fatalf("len(WalkLineage(ADK cycle)) = %d, want %d", len(bounded), int(adkcontext.LineageKey)+1)
	}
	// Cyclic Unwrap() bridge terminates cleanly within the hop budget.
	ubridge := &cyclicUnwrapBridge{}
	ucyc := &sampleStepCtx{Context: ubridge, id: 11}
	ubridge.loop = ucyc
	ubounded := adkcontext.WalkLineage(ucyc)
	if len(ubounded) == 0 || len(ubounded) > int(adkcontext.LineageKey)+1 {
		t.Fatalf("len(WalkLineage(Unwrap cycle)) = %d, want 1..%d", len(ubounded), int(adkcontext.LineageKey)+1)
	}
	// Opaque external bridge without Unwrap() stops without delegating LineageKey.
	bridge := &cyclicStepBridge{Context: context.Background()}
	cyc := &sampleStepCtx{Context: bridge, id: 12}
	bridge.loop = cyc
	opaqueWalk := adkcontext.WalkLineage(cyc)
	if len(opaqueWalk) != 1 || opaqueWalk[0] != adkcontext.Token(cyc) {
		t.Fatalf("WalkLineage(opaque bridge) = %#v, want [%#v]", opaqueWalk, adkcontext.Token(cyc))
	}
}

var walkLineageTestSink []any

func buildLayerChain(depth int) (context.Context, []*sampleStepCtx) {
	layers := make([]*sampleStepCtx, depth)
	cur := context.Background()
	for i := range depth {
		node := &sampleStepCtx{Context: cur, id: i + 1}
		layers[i] = node
		cur = node
	}
	return cur, layers
}

func bytesPerRun(n int, fn func()) uint64 {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)
	fn()
	best := ^uint64(0)
	for range 3 {
		runtime.GC()
		var m0, m1 runtime.MemStats
		runtime.ReadMemStats(&m0)
		for range n {
			fn()
		}
		runtime.ReadMemStats(&m1)
		if m1.TotalAlloc >= m0.TotalAlloc {
			if per := (m1.TotalAlloc - m0.TotalAlloc) / uint64(n); per < best {
				best = per
			}
		}
	}
	return best
}

func TestWalkLineage_64LayersOrderAndLinearAllocs(t *testing.T) {
	for _, depth := range []int{16, 32, 64} {
		top, layers := buildLayerChain(depth)
		tokens := adkcontext.WalkLineage(top)
		if len(tokens) != len(layers) {
			t.Fatalf("len(WalkLineage(%d layers)) = %d, want %d", depth, len(tokens), len(layers))
		}
		for i, wantNode := range layers {
			// Outermost token sits at index 0; deepest sits at index len(layers)-1.
			got := tokens[len(layers)-1-i]
			if want := adkcontext.Token(wantNode); got != want {
				t.Fatalf("depth %d token mismatch at layer %d: got %#v, want %#v", depth, i, got, want)
			}
		}
	}

	top16, _ := buildLayerChain(16)
	top32, _ := buildLayerChain(32)
	top64, _ := buildLayerChain(64)

	measure := func(top context.Context, wantLen int) uint64 {
		return bytesPerRun(50, func() {
			got := adkcontext.WalkLineage(top)
			if len(got) != wantLen {
				t.Fatalf("unexpected len=%d, want %d", len(got), wantLen)
			}
			walkLineageTestSink = got
		})
	}
	b16 := measure(top16, 16)
	b32 := measure(top32, 32)
	b64 := measure(top64, 64)
	t.Logf("WalkLineage bytes/run (race=%v): d16=%d, d32=%d, d64=%d", raceEnabled, b16, b32, b64)

	// Verify linear O(D) heap usage: quadratic per-hop slice copying grows ~14x
	// from depth 16->64 and ~3.8x from depth 32->64; linear O(D) grows ~3.8x and ~2x.
	if b16 == 0 || b32 == 0 {
		t.Fatalf("unexpected zero measured bytes: b16=%d b32=%d b64=%d", b16, b32, b64)
	}
	if b64 > b16*7 {
		t.Fatalf("super-linear byte growth 16->64: b16=%d b64=%d (ratio %.2f > 7.0)", b16, b64, float64(b64)/float64(b16))
	}
	if b64 > b32*3 {
		t.Fatalf("super-linear byte growth 32->64: b32=%d b64=%d (ratio %.2f > 3.0)", b32, b64, float64(b64)/float64(b32))
	}
	if !raceEnabled && b64 > b16 {
		if perLayer := (b64 - b16) / 48; perLayer > 256 {
			t.Fatalf("excess per-layer allocation slope 16->64: %d bytes/layer > 256 (b16=%d b64=%d)", perLayer, b16, b64)
		}
	}

	if allocs := testing.AllocsPerRun(50, func() {
		got := adkcontext.WalkLineage(top64)
		if len(got) != 64 {
			t.Fatalf("unexpected len=%d", len(got))
		}
	}); allocs > 300 {
		t.Fatalf("WalkLineage(64 layers) allocs = %v, want linear <= 300", allocs)
	}
}

func BenchmarkWalkLineage_64Layers(b *testing.B) {
	top, layers := buildLayerChain(64)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		got := adkcontext.WalkLineage(top)
		if len(got) != len(layers) {
			b.Fatalf("len(WalkLineage) = %d, want %d", len(got), len(layers))
		}
	}
}

type samplePathStepCtx struct {
	adkcontext.StepMarker
	context.Context
	path string
}

func (c *samplePathStepCtx) Value(key any) any {
	if v, ok := adkcontext.PathStep(key, c.path, c.Context); ok {
		return v
	}
	if v, ok := adkcontext.ForwardLineage(key, c.Context); ok {
		return v
	}
	if v, ok := adkcontext.ParentMapStep(key, c.Context); ok {
		return v
	}
	if v, ok := adkcontext.RunTokenStep(key, nil, c.Context); ok {
		return v
	}
	if c.Context == nil {
		return nil
	}
	return c.Context.Value(key)
}

func TestReadPath_ADKStepCycleTerminates(t *testing.T) {
	if os.Getenv("PR2414_READPATH_CYCLE") == "1" {
		bridge := &cyclicStepBridge{Context: context.Background()}
		cyc := &samplePathStepCtx{Context: bridge}
		bridge.loop = cyc
		if got := adkcontext.ReadPath(cyc); got != "" {
			t.Fatalf("ReadPath(cyclic) = %q, want empty", got)
		}
		return
	}
	subCtx, cancel := context.WithTimeout(t.Context(), 8*time.Second)
	defer cancel()
	cmd := exec.CommandContext(subCtx, os.Args[0], "-test.run=^TestReadPath_ADKStepCycleTerminates$")
	cmd.Env = append(os.Environ(), "PR2414_READPATH_CYCLE=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("subprocess ReadPath cycle failed (%v):\n%s", err, out)
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
	base := adkcontext.WithParentMap(context.Background(), map[string]string{"leaf": "root"})
	directPath := context.WithValue(base, adkcontext.PathKey, "wf/direct")
	if got := adkcontext.ReadPath(directPath); got != "wf/direct" {
		t.Fatalf("ReadPath(direct Value) = %q, want wf/direct", got)
	}
	withPath := &samplePathStepCtx{Context: base, path: "wf/parent"}
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

type sampleParentMapStepCtx struct {
	adkcontext.StepMarker
	context.Context
}

func (c *sampleParentMapStepCtx) Value(key any) any {
	if v, ok := adkcontext.ParentMapStep(key, c.Context); ok {
		return v
	}
	if v, ok := adkcontext.ForwardLineage(key, c.Context); ok {
		return v
	}
	if v, ok := adkcontext.PathStep(key, "", c.Context); ok {
		return v
	}
	if v, ok := adkcontext.RunTokenStep(key, nil, c.Context); ok {
		return v
	}
	if c.Context == nil {
		return nil
	}
	return c.Context.Value(key)
}

type sampleRunTokenStepCtx struct {
	adkcontext.StepMarker
	context.Context
	tok any
}

func (c *sampleRunTokenStepCtx) Value(key any) any {
	if v, ok := adkcontext.LineageStepWithRunToken(key, c, c.tok, c.Context); ok {
		return v
	}
	if v, ok := adkcontext.RunTokenStep(key, c.tok, c.Context); ok {
		return v
	}
	if v, ok := adkcontext.PathStep(key, "", c.Context); ok {
		return v
	}
	if v, ok := adkcontext.ParentMapStep(key, c.Context); ok {
		return v
	}
	if c.Context == nil {
		return nil
	}
	return c.Context.Value(key)
}

func TestReadPath_63HopsFound_65HopsNil(t *testing.T) {
	root := context.Context(&samplePathStepCtx{Context: context.Background(), path: "wf/root"})
	build := func(hops int) context.Context {
		cur := root
		for range hops {
			cur = &samplePathStepCtx{Context: cur}
		}
		return cur
	}
	if got := adkcontext.ReadPath(build(63)); got != "wf/root" {
		t.Fatalf("ReadPath(63 hops) = %q, want wf/root", got)
	}
	if got := adkcontext.ReadPath(build(64)); got != "wf/root" {
		t.Fatalf("ReadPath(64 hops) = %q, want wf/root", got)
	}
	if got := adkcontext.ReadPath(build(65)); got != "" {
		t.Fatalf("ReadPath(65 hops) = %q, want empty (budget exhausted)", got)
	}
}

func TestReadParentMap_FoundThroughBudgetedADKHopsAndCycle(t *testing.T) {
	base := adkcontext.WithParentMap(context.Background(), map[string]string{"leaf": "root"})
	cur := base
	for range 10 {
		cur = &sampleParentMapStepCtx{Context: cur}
	}
	if pm, ok := adkcontext.ReadParentMap(cur).(map[string]string); !ok || pm["leaf"] != "root" {
		t.Fatalf("ReadParentMap(10 ADK hops) = %#v, want leaf->root", adkcontext.ReadParentMap(cur))
	}

	cur65 := base
	for range 65 {
		cur65 = &sampleParentMapStepCtx{Context: cur65}
	}
	if got := adkcontext.ReadParentMap(cur65); got != nil {
		t.Fatalf("ReadParentMap(65 ADK hops) = %#v, want nil", got)
	}

	bridge := &cyclicStepBridge{Context: context.Background()}
	cyc := &sampleParentMapStepCtx{Context: bridge}
	bridge.loop = cyc
	if got := adkcontext.ReadParentMap(cyc); got != nil {
		t.Fatalf("ReadParentMap(cyclic ADK hops) = %#v, want nil", got)
	}
}

func TestReadParentMap_DropsTypedNil(t *testing.T) {
	type customPtr struct{}
	type parentMap map[string]any
	var typedNil *customPtr
	src := parentMap{
		"nil_ptr": typedNil,
		"good":    "ok",
	}
	ctx := adkcontext.WithParentMap(context.Background(), src)
	got, ok := adkcontext.ReadParentMap(ctx).(parentMap)
	if !ok {
		t.Fatalf("ReadParentMap() type = %T, want parentMap", adkcontext.ReadParentMap(ctx))
	}
	if _, exists := got["nil_ptr"]; exists {
		t.Fatalf("ReadParentMap() retained typed-nil entry: %#v", got)
	}
	if got["good"] != "ok" {
		t.Fatalf("ReadParentMap()[good] = %#v, want ok", got["good"])
	}
}

type sampleForwardCtx struct {
	adkcontext.StepMarker
	context.Context
}

func (c *sampleForwardCtx) Value(key any) any {
	if v, ok := adkcontext.ForwardLineage(key, c.Context); ok {
		return v
	}
	if v, ok := adkcontext.PathStep(key, "", c.Context); ok {
		return v
	}
	if v, ok := adkcontext.ParentMapStep(key, c.Context); ok {
		return v
	}
	if v, ok := adkcontext.RunTokenStep(key, nil, c.Context); ok {
		return v
	}
	if c.Context == nil {
		return nil
	}
	return c.Context.Value(key)
}

func TestForwardLineage_BoundedCycleAndDelegation(t *testing.T) {
	inner := &sampleStepCtx{Context: context.Background(), id: 1}
	fwd := &sampleForwardCtx{Context: inner}
	if got := adkcontext.WalkLineage(fwd); len(got) != 1 || got[0] != adkcontext.Token(inner) {
		t.Fatalf("WalkLineage(ForwardLineage -> inner) = %#v, want [%#v]", got, adkcontext.Token(inner))
	}

	bridge := &cyclicStepBridge{Context: context.Background()}
	cyc := &sampleForwardCtx{Context: bridge}
	bridge.loop = cyc
	if got := adkcontext.WalkLineage(cyc); got != nil {
		t.Fatalf("WalkLineage(ForwardLineage cycle) = %#v, want nil", got)
	}
	if got := adkcontext.ReadPath(cyc); got != "" {
		t.Fatalf("ReadPath(ForwardLineage cycle) = %q, want empty", got)
	}
	if got := adkcontext.ReadParentMap(cyc); got != nil {
		t.Fatalf("ReadParentMap(ForwardLineage cycle) = %#v, want nil", got)
	}
	if got := adkcontext.ReadRunToken(cyc); got != nil {
		t.Fatalf("ReadRunToken(ForwardLineage cycle) = %#v, want nil", got)
	}
}

func TestRunToken_StepBudgetCycleAndForgeRejection(t *testing.T) {
	tokA := adkcontext.NewRunToken()
	tokB := adkcontext.NewRunToken()
	if tokA == nil || tokB == nil || tokA == tokB {
		t.Fatalf("NewRunToken failed uniqueness: %#v vs %#v", tokA, tokB)
	}
	if !adkcontext.IsRunTokenKey(adkcontext.RunTokenKey) || adkcontext.IsRunTokenKey(adkcontext.PathKey) {
		t.Fatal("IsRunTokenKey returned unexpected result")
	}

	forged := context.WithValue(context.Background(), adkcontext.RunTokenKey, new(byte))
	if got := adkcontext.ReadRunToken(forged); got != nil {
		t.Fatalf("ReadRunToken accepted forged *byte: %#v", got)
	}

	root := &sampleRunTokenStepCtx{Context: context.Background(), tok: tokA}
	if got := adkcontext.ReadRunToken(root); got != tokA {
		t.Fatalf("ReadRunToken(root) = %#v, want %#v", got, tokA)
	}
	nested := &sampleRunTokenStepCtx{Context: root, tok: tokB}
	if got := adkcontext.ReadRunToken(nested); got != tokB {
		t.Fatalf("ReadRunToken(nested) = %#v, want %#v", got, tokB)
	}

	// Build 65 distinct-token layers and confirm every contextToken and runToken survives flattening.
	cur := context.Background()
	var wantRunTokens []any
	for range 65 {
		rt := adkcontext.NewRunToken()
		wantRunTokens = append(wantRunTokens, rt)
		cur = &sampleRunTokenStepCtx{Context: cur, tok: rt}
	}
	lineage := adkcontext.WalkLineage(cur)
	for i, rt := range wantRunTokens {
		if !slices.Contains(lineage, rt) {
			t.Fatalf("WalkLineage(65 layers) missing runToken #%d (%#v) in len=%d", i, rt, len(lineage))
		}
	}

	// 65 hops with nil tok reaches budget exhaustion cleanly.
	curNil := context.Context(root)
	for range 65 {
		curNil = &sampleRunTokenStepCtx{Context: curNil}
	}
	if got := adkcontext.ReadRunToken(curNil); got != nil {
		t.Fatalf("ReadRunToken(65 nil hops) = %#v, want nil", got)
	}
}

func TestReadParentMap_ReturnsSnapshotAndFiltersNil(t *testing.T) {
	type sampleParentMap map[string]any
	src := sampleParentMap{"c": "p", "n": nil}
	ctx := context.WithValue(context.Background(), adkcontext.ParentMapKey, src)

	raw := adkcontext.ReadParentMap(ctx)
	got, ok := raw.(sampleParentMap)
	if !ok {
		t.Fatalf("ReadParentMap() type = %T, want sampleParentMap", raw)
	}
	if _, hasNil := got["n"]; hasNil {
		t.Fatalf("ReadParentMap() kept nil entry: %#v", got)
	}
	if got["c"] != "p" {
		t.Fatalf("ReadParentMap()[c] = %v, want p", got["c"])
	}
	got["evil"] = "x"
	if _, leaked := src["evil"]; leaked {
		t.Fatalf("mutating ReadParentMap() output mutated caller map: %#v", src)
	}

	// Nil map and non-map values under ParentMapKey return nil.
	nilMapCtx := context.WithValue(context.Background(), adkcontext.ParentMapKey, sampleParentMap(nil))
	if gotNil := adkcontext.ReadParentMap(nilMapCtx); gotNil != nil {
		t.Fatalf("ReadParentMap(typed-nil map) = %#v, want nil", gotNil)
	}
	nonMapCtx := context.WithValue(context.Background(), adkcontext.ParentMapKey, "not-a-map")
	if gotNonMap := adkcontext.ReadParentMap(nonMapCtx); gotNonMap != nil {
		t.Fatalf("ReadParentMap(non-map) = %#v, want nil", gotNonMap)
	}
}

type uint8BlockingCtx struct {
	context.Context
}

func (c *uint8BlockingCtx) Value(key any) any {
	if key != nil && reflect.TypeOf(key).Kind() == reflect.Uint8 {
		return nil
	}
	return c.Context.Value(key)
}

type uint8BlockingUnwrapCtx struct {
	context.Context
}

func (c *uint8BlockingUnwrapCtx) Value(key any) any {
	if key != nil && reflect.TypeOf(key).Kind() == reflect.Uint8 {
		return nil
	}
	return c.Context.Value(key)
}

func (c *uint8BlockingUnwrapCtx) Unwrap() context.Context {
	return c.Context
}

func TestPathAndParentMapSurviveUint8BlockingWrapperAfterADKStep(t *testing.T) {
	base := adkcontext.WithParentMap(context.Background(), map[string]string{"leaf": "worker"})
	innerTok := adkcontext.NewRunToken()
	innerLineage := &sampleRunTokenStepCtx{
		Context: &samplePathStepCtx{Context: base, path: "wf/left/leaf@1"},
		tok:     innerTok,
	}

	t.Run("with_unwrap", func(t *testing.T) {
		filtered := &uint8BlockingUnwrapCtx{Context: innerLineage}
		outerTok := adkcontext.NewRunToken()
		outerADK := &sampleRunTokenStepCtx{
			Context: &samplePathStepCtx{Context: &sampleParentMapStepCtx{Context: filtered}},
			tok:     outerTok,
		}
		outerEmptyTok := &sampleRunTokenStepCtx{
			Context: &samplePathStepCtx{Context: &sampleParentMapStepCtx{Context: filtered}},
		}

		if got := adkcontext.ReadPath(outerADK); got != "wf/left/leaf@1" {
			t.Fatalf("ReadPath across ADK step + uint8BlockingUnwrapCtx = %q, want wf/left/leaf@1", got)
		}
		if m, ok := adkcontext.ReadParentMap(outerADK).(map[string]string); !ok || m["leaf"] != "worker" {
			t.Fatalf("ReadParentMap across ADK step + uint8BlockingUnwrapCtx = %#v, want leaf->worker", adkcontext.ReadParentMap(outerADK))
		}
		if lineage := adkcontext.WalkLineage(outerADK); !slices.Contains(lineage, adkcontext.Token(outerADK)) || !slices.Contains(lineage, adkcontext.Token(innerLineage)) {
			t.Fatalf("WalkLineage(outerADK) missing expected tokens across Unwrap: %#v", lineage)
		}
		if got := adkcontext.ReadRunToken(outerEmptyTok); got != innerTok {
			t.Fatalf("ReadRunToken across Unwrap = %#v, want %#v", got, innerTok)
		}
	})

	t.Run("without_unwrap_opaque", func(t *testing.T) {
		filtered := &uint8BlockingCtx{Context: innerLineage}
		outerTok := adkcontext.NewRunToken()
		outerADK := &sampleRunTokenStepCtx{
			Context: &samplePathStepCtx{Context: &sampleParentMapStepCtx{Context: filtered}},
			tok:     outerTok,
		}
		outerEmptyTok := &sampleRunTokenStepCtx{
			Context: &samplePathStepCtx{Context: &sampleParentMapStepCtx{Context: filtered}},
		}

		if got := adkcontext.ReadPath(outerADK); got != "" {
			t.Fatalf("ReadPath across opaque uint8BlockingCtx = %q, want empty", got)
		}
		if got := adkcontext.ReadParentMap(outerADK); got != nil {
			t.Fatalf("ReadParentMap across opaque uint8BlockingCtx = %#v, want nil", got)
		}
		if lineage := adkcontext.WalkLineage(outerADK); len(lineage) == 0 || !slices.Contains(lineage, adkcontext.Token(outerADK)) {
			t.Fatalf("WalkLineage(outerADK) missing outer token: %#v", lineage)
		} else if slices.Contains(lineage, adkcontext.Token(innerLineage)) {
			t.Fatalf("WalkLineage penetrated opaque uint8BlockingCtx: %#v", lineage)
		}
		if got := adkcontext.ReadRunToken(outerEmptyTok); got != nil {
			t.Fatalf("ReadRunToken penetrated opaque uint8BlockingCtx: %#v, want nil", got)
		}
	})
}

type r10CycleNodeA struct {
	peer *r10CycleNodeB
}

func (*r10CycleNodeA) Deadline() (time.Time, bool) { return time.Time{}, false }
func (*r10CycleNodeA) Done() <-chan struct{}       { return nil }
func (*r10CycleNodeA) Err() error                  { return nil }
func (a *r10CycleNodeA) Value(key any) any {
	if a.peer != nil {
		return a.peer.Value(key)
	}
	return nil
}

type r10CycleNodeB struct {
	peer *r10CycleNodeA
}

func (*r10CycleNodeB) Deadline() (time.Time, bool) { return time.Time{}, false }
func (*r10CycleNodeB) Done() <-chan struct{}       { return nil }
func (*r10CycleNodeB) Err() error                  { return nil }
func (b *r10CycleNodeB) Value(key any) any {
	if b.peer != nil {
		return b.peer.Value(key)
	}
	return nil
}

type r10OpaqueCounterCtx struct {
	context.Context
	valueCalls *int
}

func (c *r10OpaqueCounterCtx) Value(key any) any {
	*c.valueCalls++
	return c.Context.Value(key)
}

func TestPrivateKeyReaders_ExternalValueCycle_DoesNotCallValue(t *testing.T) {
	if os.Getenv("ADK_R10_SUBPROCESS") == "1" {
		a := &r10CycleNodeA{}
		b := &r10CycleNodeB{peer: a}
		a.peer = b

		adkPathCtx := &samplePathStepCtx{Context: a}
		adkPMCtx := &sampleParentMapStepCtx{Context: a}
		adkRunTokCtx := &sampleRunTokenStepCtx{Context: a}
		adkFwdCtx := &sampleForwardCtx{Context: a}
		withPM := adkcontext.WithParentMap(a, map[string]string{"c": "p"})

		for _, c := range []context.Context{a, adkPathCtx, adkPMCtx, adkRunTokCtx, adkFwdCtx, withPM} {
			if got := adkcontext.ReadPath(c); got != "" {
				t.Fatalf("ReadPath(%T) = %q, want empty", c, got)
			}
			if got := adkcontext.ReadRunToken(c); got != nil {
				t.Fatalf("ReadRunToken(%T) = %#v, want nil", c, got)
			}
		}
		for _, c := range []context.Context{a, adkPMCtx, adkFwdCtx} {
			if got := adkcontext.ReadParentMap(c); got != nil {
				t.Fatalf("ReadParentMap(%T) = %#v, want nil", c, got)
			}
		}
		if got := adkcontext.ReadParentMap(withPM); got == nil {
			t.Fatalf("ReadParentMap(withPM) = nil, want map")
		}
		wcCycle := context.WithoutCancel(a)
		if got := adkcontext.ReadPath(wcCycle); got != "" {
			t.Fatalf("ReadPath(WithoutCancel(cycleA)) = %q, want empty", got)
		}
		if got := adkcontext.ReadRunToken(wcCycle); got != nil {
			t.Fatalf("ReadRunToken(WithoutCancel(cycleA)) = %#v, want nil", got)
		}
		_ = adkcontext.WalkLineage(a)
		_ = adkcontext.WalkLineage(adkFwdCtx)
		_ = adkcontext.WalkLineage(withPM)
		_ = adkcontext.WalkLineage(wcCycle)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestPrivateKeyReaders_ExternalValueCycle_DoesNotCallValue$", "-test.v")
	cmd.Env = append(os.Environ(), "ADK_R10_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("subprocess failed on external Value cycle: %v\n%s", err, out)
	}

	// Non-cyclic opaque wrapper with declared Value and no Unwrap() must never have Value called.
	calls := 0
	tok := adkcontext.NewRunToken()
	inner := &samplePathStepCtx{
		Context: adkcontext.WithParentMap(
			&sampleRunTokenStepCtx{Context: context.Background(), tok: tok},
			map[string]string{"c": "p"},
		),
		path: "wf/a",
	}
	opaque := &r10OpaqueCounterCtx{Context: inner, valueCalls: &calls}
	outerADK := &sampleForwardCtx{Context: opaque}
	for _, c := range []context.Context{opaque, outerADK} {
		if got := adkcontext.ReadPath(c); got != "" {
			t.Fatalf("ReadPath(opaque) = %q, want empty", got)
		}
		if got := adkcontext.ReadParentMap(c); got != nil {
			t.Fatalf("ReadParentMap(opaque) = %#v, want nil", got)
		}
		if got := adkcontext.ReadRunToken(c); got != nil {
			t.Fatalf("ReadRunToken(opaque) = %#v, want nil", got)
		}
	}
	if calls != 0 {
		t.Fatalf("opaque wrapper Value called %d times by private key readers, want 0", calls)
	}

	// Verify 80+ WithValue layers + WithoutCancel preserve all private keys, while
	// non-Unwrap external wrappers stop cleanly without reading side fields or calling Value.
	type deepValKey int
	deep := context.Context(inner)
	for i := range 80 {
		deep = context.WithValue(deep, deepValKey(i), i)
	}
	deep = context.WithoutCancel(deep)
	if got := adkcontext.ReadPath(deep); got != "wf/a" {
		t.Fatalf("ReadPath(WithoutCancel -> 80 WithValue) = %q, want wf/a", got)
	}
	if got := adkcontext.ReadRunToken(deep); got != tok {
		t.Fatalf("ReadRunToken(WithoutCancel -> 80 WithValue) = %#v, want %#v", got, tok)
	}
	if pm, ok := adkcontext.ReadParentMap(deep).(map[string]string); !ok || pm["c"] != "p" {
		t.Fatalf("ReadParentMap(WithoutCancel -> 80 WithValue) = %#v, want c->p", adkcontext.ReadParentMap(deep))
	}
	if lin := adkcontext.WalkLineage(deep); !slices.Contains(lin, tok) {
		t.Fatalf("WalkLineage(WithoutCancel -> 80 WithValue) missing tok: %#v", lin)
	}

	// Verify Unwrap()-equipped wrapper preserves private keys over WithoutCancel -> 80 WithValue.
	ub := &uint8BlockingUnwrapCtx{Context: deep}
	if got := adkcontext.ReadPath(ub); got != "wf/a" {
		t.Fatalf("ReadPath(uint8BlockingUnwrapCtx -> WithoutCancel -> 80 WithValue) = %q, want wf/a", got)
	}
	if got := adkcontext.ReadRunToken(ub); got != tok {
		t.Fatalf("ReadRunToken(uint8BlockingUnwrapCtx -> WithoutCancel -> 80 WithValue) = %#v, want %#v", got, tok)
	}

	// Lock-free wrapper with a second context side field (e.g. lastSeen) must stop without reading it.
	type lockFreeSideCtx struct {
		context.Context
		lastSeen context.Context
	}
	var lastSeenMu sync.Mutex
	sideW := &lockFreeSideCtx{Context: inner, lastSeen: inner}
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
				lastSeenMu.Lock()
				if i%2 == 0 {
					sideW.lastSeen = context.Background()
				} else {
					sideW.lastSeen = inner
				}
				lastSeenMu.Unlock()
				runtime.Gosched()
			}
		}
	}()
	defer func() {
		close(stop)
		<-done
	}()
	for range 100 {
		if got := adkcontext.ReadPath(sideW); got != "" {
			t.Fatalf("ReadPath(lockFreeSideCtx) = %q, want empty", got)
		}
		if got := adkcontext.ReadRunToken(sideW); got != nil {
			t.Fatalf("ReadRunToken(lockFreeSideCtx) = %#v, want nil", got)
		}
		if got := adkcontext.ReadParentMap(sideW); got != nil {
			t.Fatalf("ReadParentMap(lockFreeSideCtx) = %#v, want nil", got)
		}
		_ = adkcontext.WalkLineage(sideW)
	}
}
