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
	if m := src.Method(0); m.PkgPath == "" {
		t.Errorf("Source.%s is exported. Any type outside this module can then declare a "+
			"method of that name and be taken for one of ADK's own contexts, which the "+
			"identity procedure asks for the user it speaks for instead of reading its "+
			"session. Keep the method unexported.", m.Name)
	}
}

// exportedLookalike is what an outside type would declare to impersonate a
// Source if the marker method were ever exported.
type exportedLookalike struct{}

func (exportedLookalike) AdkIdentitySource() {}

// unexportedLookalike declares the real name from a different package. Go treats
// that as a different method, which is the guarantee under test.
type unexportedLookalike struct{}

// Never called, and that is the point: it exists to be offered to a type
// assertion and refused.
//
//nolint:unused // declared to fail an interface check, not to be invoked.
func (unexportedLookalike) adkIdentitySource() {}

func TestALookalikeFromAnotherPackageIsNotASource(t *testing.T) {
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
