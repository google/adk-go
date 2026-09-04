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

// This file reads main.go as source rather than running it. That is an unusual
// thing for a test to do and it needs justifying.
//
// The invariant is that each concurrent review gets its OWN runner, because
// ADK's runner.Run lazily initializes mutable state on the agent and two
// concurrent Run calls on a shared runner race on it (see newReviewer). The
// regression is a hoist: build one reviewer up front and hand the same one to
// every review.
//
// Two tests already attack that at run time, and neither closes it:
//
//   - TestReviewAllGivesEachReviewItsOwnRunner is deterministic, but it calls
//     reviewAll directly with a factory of its own, so a hoist introduced in
//     sweep -- above the seam it tests -- is invisible to it.
//   - TestSweepGivesEachConcurrentReviewItsOwnRunner drives the real sweep, so
//     it does reach that hoist, but sweep builds its factory internally and
//     there is nowhere to put a barrier. Detection rests on the race detector
//     catching two workers on the same two instructions, and it is documented
//     there as probabilistic: measured 7 of 10 fresh processes.
//
// Measured here, on this tree, against three hoist mutants in fresh `go test
// -race` processes, and 0 spurious failures in 20 runs against correct code:
//
//	                                          before this test   with it
//	H1  sweep reuses its startup-check
//	    reviewer for every review             killed 3 of 5      5 of 5
//	H2  the fan-out is relocated into a new
//	    helper and hoisted inside it          killed 0 of 5      5 of 5
//	H3  reviewerFor is bypassed and the
//	    factory built inline                  --                 5 of 5
//
// All three branches below fail closed, which H1, H2 and H3 hit in turn. The
// simplest form of H3 -- dropping reviewerFor and leaving mdl and tools unused
// -- does not compile, so the compiler already refuses it; H3 is the version
// that does compile.
//
// H2 is the one that matters for the shape of the assertion. It is why this
// test asserts the PRESENCE of the right wiring rather than the absence of the
// wrong wiring: an absence check ("sweep contains no call to newReviewer") is
// satisfied vacuously the moment the code it is inspecting moves somewhere
// else. Scoping the walk to sweep's own body and requiring the call to be there
// fails closed instead -- relocating the fan-out removes the thing being
// asserted, so the assertion fails rather than passing on an empty set.
//
// It is a complement to the runtime tests, not a replacement. It cannot tell
// whether the wiring WORKS; it only tells you the composition root still looks
// the way the runtime tests assume. Deleting it costs the deterministic half of
// the guarantee, and deleting the runtime tests costs the other half.
func TestSweepPassesThePerReviewFactoryToReviewAll(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "main.go", nil, 0)
	if err != nil {
		t.Fatalf("parsing main.go: %v", err)
	}

	sweepFn := findFunc(file, "sweep")
	if sweepFn == nil {
		t.Fatal("main.go has no func sweep: the composition root was renamed or moved, " +
			"and this test can no longer see what it is asserting")
	}

	// The factory variable: the one thing in sweep assigned from reviewerFor.
	factory := identsAssignedFrom(sweepFn.Body, "reviewerFor")
	switch len(factory) {
	case 1: // as expected
	case 0:
		t.Fatal("sweep no longer builds its reviewer factory with reviewerFor(...). " +
			"If the per-review construction moved elsewhere, every review may now share " +
			"one runner, which races inside ADK (see newReviewer)")
	default:
		t.Fatalf("sweep assigns %d variables from reviewerFor; this test assumes one and "+
			"cannot tell which is handed to reviewAll: %v", len(factory), factory)
	}

	// The fan-out: reviewAll must be called HERE, in sweep's own body. Requiring
	// it is what makes relocating the fan-out a failure rather than a bypass.
	calls := callsTo(sweepFn.Body, "reviewAll")
	switch len(calls) {
	case 1: // as expected
	case 0:
		t.Fatal("sweep no longer calls reviewAll. The fan-out was relocated, so nothing " +
			"here checks that each review still gets its own runner -- which is exactly " +
			"how a hoist hides. Re-point this test at wherever reviewAll is now called " +
			"from, and check that caller passes a per-review factory")
	default:
		t.Fatalf("sweep calls reviewAll %d times; this test assumes one", len(calls))
	}

	// And the factory itself is what is handed over -- not a value built once
	// above the call and wrapped in a constant closure.
	want := factory[0]
	for _, arg := range calls[0].Args {
		if id, ok := arg.(*ast.Ident); ok && id.Name == want {
			return
		}
	}
	t.Errorf("sweep calls reviewAll without passing %s, the factory it built from "+
		"reviewerFor. A reviewer built once and shared across reviews races inside ADK "+
		"(runner.Run initializes agent state on first use); args were %s",
		want, renderArgs(calls[0].Args))
}

// findFunc returns the top-level function declaration named name, or nil.
func findFunc(file *ast.File, name string) *ast.FuncDecl {
	for _, d := range file.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok && fn.Recv == nil && fn.Name.Name == name {
			return fn
		}
	}
	return nil
}

// identsAssignedFrom returns the names of variables in body assigned the result
// of a call to callee, e.g. "x" for `x := callee(...)`.
func identsAssignedFrom(body *ast.BlockStmt, callee string) []string {
	var out []string
	ast.Inspect(body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Rhs) != 1 || len(as.Lhs) != 1 {
			return true
		}
		if !isCallTo(as.Rhs[0], callee) {
			return true
		}
		if id, ok := as.Lhs[0].(*ast.Ident); ok {
			out = append(out, id.Name)
		}
		return true
	})
	return out
}

// callsTo returns every call to the named function inside body.
func callsTo(body *ast.BlockStmt, callee string) []*ast.CallExpr {
	var out []*ast.CallExpr
	ast.Inspect(body, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok && isCallTo(call, callee) {
			out = append(out, call)
		}
		return true
	})
	return out
}

// isCallTo reports whether e is a call to the plain (non-method) function name.
func isCallTo(e ast.Expr, name string) bool {
	call, ok := e.(*ast.CallExpr)
	if !ok {
		return false
	}
	id, ok := call.Fun.(*ast.Ident)
	return ok && id.Name == name
}

// renderArgs names the arguments of a call for a failure message, so the report
// says what was passed instead of only what was missing.
func renderArgs(args []ast.Expr) string {
	out := "("
	for i, a := range args {
		if i > 0 {
			out += ", "
		}
		if id, ok := a.(*ast.Ident); ok {
			out += id.Name
			continue
		}
		out += "<expr>"
	}
	return out + ")"
}

// The nonce is what makes the untrusted fence unforgeable. Every injection
// defence in this bot rests on an attacker being unable to write the closing
// marker themselves, and that holds only while the nonce is unguessable.
//
// Nothing else in the suite notices if it stops being. Measured: swapping
// crypto/rand for math/rand in newNonce compiles and the entire suite passes,
// `-race` included. A predictable nonce cannot be detected from its output --
// math/rand produces values that are unique, well-distributed, and hex-shaped
// exactly like the real ones -- so no test of the returned string can catch
// this. The source is the only place the property is visible.
//
// The repository-wide lint rule that would also catch it lives in the root
// golangci config, which this module does not own, so the check lives here.
//
// Written in the presence form for the reason recorded above: asserting the
// absence of math/rand would pass if nonce generation moved somewhere this test
// does not look. Requiring the call to be inside newNonce fails closed instead.
func TestNonceIsDrawnFromACSPRNG(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "main.go", nil, 0)
	if err != nil {
		t.Fatalf("parsing main.go: %v", err)
	}

	// Resolve local package names to import paths, so an alias cannot disguise
	// which rand is in use: `mrand "math/rand"` and a plain `"crypto/rand"` both
	// present a selector, and only the path distinguishes them.
	paths := map[string]string{}
	for _, imp := range file.Imports {
		path := strings.Trim(imp.Path.Value, `"`)
		name := path[strings.LastIndex(path, "/")+1:]
		if imp.Name != nil {
			name = imp.Name.Name
		}
		paths[name] = path
	}

	body := funcLitFor(file, "newNonce")
	if body == nil {
		t.Fatal("main.go has no newNonce function literal. If nonce generation moved, " +
			"re-point this test at it: an unguessable nonce is what stops attacker text " +
			"closing the untrusted fence, and nothing else in this suite can see it")
	}

	found := ""
	ast.Inspect(body, func(n ast.Node) bool {
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
		if p := paths[pkg.Name]; p != "" && strings.HasSuffix(p, "rand") {
			found = p
		}
		return true
	})

	switch found {
	case "crypto/rand": // as required
	case "":
		t.Error("newNonce draws from no rand package at all, so the fence marker may be " +
			"predictable. It must call crypto/rand.Read")
	default:
		t.Errorf("newNonce draws from %q, not crypto/rand. A predictable nonce lets an "+
			"attacker pre-write the closing marker in their own issue text and escape the "+
			"untrusted fence, which every injection defence here depends on", found)
	}
}

// funcLitFor returns the body of a package-level `var name = func(...) {...}`,
// or nil. newNonce is declared that way so tests can replace it.
func funcLitFor(file *ast.File, name string) *ast.BlockStmt {
	for _, d := range file.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || gd.Tok != token.VAR {
			continue
		}
		for _, spec := range gd.Specs {
			vs, ok := spec.(*ast.ValueSpec)
			if !ok || len(vs.Names) != 1 || len(vs.Values) != 1 || vs.Names[0].Name != name {
				continue
			}
			if lit, ok := vs.Values[0].(*ast.FuncLit); ok {
				return lit.Body
			}
		}
	}
	return nil
}
