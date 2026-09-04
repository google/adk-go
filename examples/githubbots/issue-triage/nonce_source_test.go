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
	"strings"
	"testing"
)

// The fence nonce must come from a cryptographic source, asserted at the source
// and not through the values.
//
// The whole untrusted-input design rests on an author being unable to guess the
// marker that closes their own fence. Swap crypto/rand for math/rand in
// newNonce and that assumption is gone -- an attacker who can predict the
// marker writes the closing token themselves and everything after it is read as
// instructions rather than as data.
//
// No test of the OUTPUT can catch that, which is why this one reads the source
// instead. math/rand produces hex tokens that are unique across calls, the
// right length and correctly distributed, so the freshness, charset and
// per-issue-distinctness assertions in main_test.go all pass against a fully
// predictable generator. Measured on this tree: with the import swapped and the
// block re-sorted, `go build`, `go vet`, `gofmt`, `go test -race -shuffle=on`
// and `golangci-lint run` were ALL clean. gosec does flag it (G404), and gosec
// is not among the linters this repository's CI runs -- the root .golangci.yml
// enables goheader, revive, goimports and gofumpt. So nothing in the pipeline
// stood between a predictable nonce and main.
//
// That is the general shape worth naming: a rule some OTHER configuration would
// have caught is not covered, it is merely somebody else's. A module that does
// not own the root lint config can still close the gap locally, and this is
// what closing it looks like.
//
// Scoped deliberately to the nonce. retry.go also draws from crypto/rand, for
// backoff jitter, and says in its own comment that this is not a security
// decision -- it is there because gosec reports math/rand. A file-level or
// import-level assertion would either be satisfied by that unrelated import
// while newNonce quietly used math/rand, or fire on the jitter site and
// pressure someone into weakening it. So this resolves the identifier at the
// call inside newNonce back to the import it came from.
//
// Killing mutations, all three verified: replace the import with "math/rand";
// alias it, `rand "math/rand"`, so the call site is untouched; and move the
// draw into a helper that newNonce calls.
func TestTheNonceIsDrawnFromCryptoRand(t *testing.T) {
	const file = "main.go"
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, file, nil, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", file, err)
	}

	// Import path by the name it is referred to as, so an alias cannot hide
	// which package a call actually resolves to.
	pathFor := map[string]string{}
	for _, imp := range f.Imports {
		path := strings.Trim(imp.Path.Value, `"`)
		name := path[strings.LastIndex(path, "/")+1:]
		if imp.Name != nil {
			name = imp.Name.Name
		}
		pathFor[name] = path
	}

	decl := findValueSpec(f, "newNonce")
	if decl == nil {
		t.Fatalf("no newNonce declaration in %s; this guard no longer knows what to check", file)
	}

	// Every package-qualified call newNonce makes, so the assertion is about
	// what the function does rather than about what the file imports.
	calls := map[string]bool{}
	ast.Inspect(decl, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		pkg, ok := sel.X.(*ast.Ident)
		if !ok {
			return true
		}
		calls[pathFor[pkg.Name]+"."+sel.Sel.Name] = true
		return true
	})

	if !calls["crypto/rand.Read"] {
		got := make([]string, 0, len(calls))
		for c := range calls {
			got = append(got, c)
		}
		t.Errorf("newNonce does not call crypto/rand.Read. It calls %v.\n"+
			"The fence marker has to be unguessable: a predictable nonce lets an issue author "+
			"write their own closing marker and escape the fence, which every injection test here "+
			"assumes is impossible. If the draw moved into a helper, this guard must follow it "+
			"rather than be deleted -- nothing else in the build or the CI linters catches the swap.",
			got)
	}
	// The presence check above is satisfied by a function that draws from BOTH,
	// so the weak source must also be absent.
	for name := range calls {
		if strings.HasPrefix(name, "math/rand") {
			t.Errorf("newNonce calls %s. The fence marker must come only from a cryptographic "+
				"source.", name)
		}
	}
}

// findValueSpec returns the declaration of a package-level var by name.
func findValueSpec(f *ast.File, name string) ast.Node {
	var found ast.Node
	for _, d := range f.Decls {
		gen, ok := d.(*ast.GenDecl)
		if !ok || gen.Tok != token.VAR {
			continue
		}
		for _, spec := range gen.Specs {
			vs, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			for _, id := range vs.Names {
				if id.Name == name {
					found = vs
				}
			}
		}
	}
	return found
}
