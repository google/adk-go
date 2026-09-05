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
	"reflect"
	"slices"
	"strings"
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
	src := reflect.TypeOf((*adkcontext.Source)(nil)).Elem()
	if got := src.NumMethod(); got != 1 {
		t.Fatalf("Source has %d methods, want exactly 1; this test knows how to check one", got)
	}
	// PkgPath is empty for an exported method and the declaring package's path
	// for an unexported one.
	// The method's OWN package must be internal too, not just the key's. Splitting
	// the package — Source and Marker public, the key left behind — leaves every
	// other check here green while letting any type anywhere embed Marker and be
	// ASKED for an identity. Two holes of this class have already been found by
	// hand; this is the third, and it is the same one test away.
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
