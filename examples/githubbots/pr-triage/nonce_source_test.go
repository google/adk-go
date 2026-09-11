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

package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The fence around author-written text is only unforgeable while the nonce is
// unpredictable. If an attacker can guess it, they write the closing marker
// into their own description, and everything after it reads as trusted
// scaffolding -- which is the containment every injection result in this
// package rests on.
//
// Nothing else in the suite defends that. Measured: replacing the crypto/rand
// import with `rand "math/rand"`, correctly sorted, builds clean, passes
// `go vet`, passes the entire suite under -race including all three nonce
// tests, and reports 0 issues from golangci-lint.
//
// The three nonce tests are blind to it by construction, and they look like
// coverage. One asserts a fresh nonce per pull request, one asserts the draw's
// failure path aborts, one asserts length and uniqueness. math/rand satisfies
// every one of those: it yields unique, correctly-sized hex from a Read that
// can be made to fail. They pin the shape of the nonce and never its source.
//
// A repository-wide gosec rule would catch this, but that config lives in the
// root module, which this example does not own. A security property resting on
// a lint rule you cannot control is a property you can pin locally instead.
const cryptoRand = "crypto/rand"

// weakRandomness lists the packages that must never appear here. math/rand/v2
// has no Read, so it cannot be a drop-in for the draw, but it is banned anyway:
// the point is that no predictable generator has any business in this package.
var weakRandomness = []string{"math/rand", "math/rand/v2"}

// TestNonceIsDrawnFromACSPRNG checks the SOURCE, because no behavioural test
// can. A predictable generator produces output indistinguishable from a secure
// one by any property a unit test can assert -- that is what "predictable"
// means to everyone except the attacker holding the seed.
func TestNonceIsDrawnFromACSPRNG(t *testing.T) {
	files := packageFiles(t)

	// 1. No weak generator anywhere in the package, matched by import PATH so
	//    an alias cannot hide one. `rand "math/rand"` reads exactly like the
	//    real thing at the call site.
	for path, f := range files {
		for _, imp := range f.Imports {
			got := strings.Trim(imp.Path.Value, `"`)
			for _, banned := range weakRandomness {
				if got == banned {
					t.Errorf("%s imports %q. The nonce must come from a CSPRNG: a "+
						"predictable fence marker lets an author write the closing marker "+
						"themselves and escape the fence.", filepath.Base(path), banned)
				}
			}
		}
	}

	// 2. The CSPRNG is actually present, so deleting the draw fails here too.
	local, file := cryptoRandLocalName(files)
	if local == "" {
		t.Fatalf("no file in this package imports %q", cryptoRand)
	}

	// 3. The nonce draw resolves to THAT import. Catches the evasion the other
	//    two miss: keeping crypto/rand imported and legitimately used elsewhere
	//    while the nonce itself draws from something else.
	fn := findNonceFunc(t, files)
	var sawRead bool
	ast.Inspect(fn, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "Read" {
			return true
		}
		sawRead = true
		pkg, ok := sel.X.(*ast.Ident)
		if !ok || pkg.Name != local {
			t.Errorf("newNonce draws from %s.Read; %q is imported as %q in %s, so the "+
				"draw is not coming from the CSPRNG.",
				exprString(sel.X), cryptoRand, local, filepath.Base(file))
		}
		return true
	})
	if !sawRead {
		t.Error("newNonce contains no call to Read. Either the draw moved, in which " +
			"case point this test at its new home, or the nonce is no longer drawn at all.")
	}
}

// packageFiles parses every non-test Go file in this directory.
func packageFiles(t *testing.T) map[string]*ast.File {
	t.Helper()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}
	fset := token.NewFileSet()
	out := map[string]*ast.File{}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		out[name] = f
	}
	if len(out) == 0 {
		t.Fatal("parsed no package files; this test would pass vacuously")
	}
	return out
}

// cryptoRandLocalName returns the identifier crypto/rand is bound to, and the
// file importing it. Empty when no file does.
func cryptoRandLocalName(files map[string]*ast.File) (name, file string) {
	for path, f := range files {
		for _, imp := range f.Imports {
			if strings.Trim(imp.Path.Value, `"`) != cryptoRand {
				continue
			}
			if imp.Name != nil {
				return imp.Name.Name, path
			}
			return "rand", path
		}
	}
	return "", ""
}

// findNonceFunc returns the body of newNonce, wherever in the package it lives,
// so hoisting it into another file relocates the test rather than defeating it.
func findNonceFunc(t *testing.T, files map[string]*ast.File) ast.Node {
	t.Helper()
	for _, f := range files {
		var found ast.Node
		ast.Inspect(f, func(n ast.Node) bool {
			vs, ok := n.(*ast.ValueSpec)
			if !ok {
				return true
			}
			for i, id := range vs.Names {
				if id.Name == "newNonce" && i < len(vs.Values) {
					found = vs.Values[i]
					return false
				}
			}
			return true
		})
		if found != nil {
			return found
		}
	}
	t.Fatal("newNonce is not declared in this package; if it was renamed, rename it here too")
	return nil
}

// exprString renders an expression for an error message.
func exprString(e ast.Expr) string {
	if id, ok := e.(*ast.Ident); ok {
		return id.Name
	}
	return "an expression"
}
