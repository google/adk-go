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
	"strings"
	"testing"
)

// transportFuncs are the functions allowed to reach GitHub without passing
// through shouldSkip, with the reason each is exempt. Adding a name here is a
// deliberate act that shows up in review, which is the point.
var transportFuncs = map[string]string{
	// The GraphQL transport itself. It is not a mutation; what matters is which
	// document its callers hand it, which readOnlyQueries below decides.
	"graphQL": "the GraphQL transport; its callers are checked by the document they pass",
}

// readOnlyQueries are the GraphQL documents known not to mutate. A call to
// c.graphQL with any other document counts as a mutation, so a write moved onto
// GraphQL -- the obvious next step, since go-github v66 has no issue-type
// support and GitHub's own answer is the updateIssue mutation -- cannot slip
// past by not looking like a REST write.
var readOnlyQueries = map[string]bool{
	"issueSearchQuery":   true,
	"issueByNumberQuery": true,
}

// The dry-run chokepoint, pinned structurally rather than by example.
//
// TestRunInDryRunWritesNothing proves the two mutations that exist today are
// suppressed, which is a statement about them and not about the next one. This
// is the counterpart of TestTools for the tool inventory: it reads the
// package's own source and fails when a function reaches GitHub without
// passing through shouldSkip.
//
// It is deliberately strict in three ways, because a pin that is easy to
// satisfy wrongly is worse than none. Any call rooted at the REST client counts
// as reaching GitHub, rather than a list of method names that a write could be
// spelled around. A call to the GraphQL transport counts unless the document is
// a known read. And shouldSkip only counts when its result actually decides
// whether the function returns, so a bare call cannot be used to silence this.
//
// Killing mutations, all verified: delete the shouldSkip call from SetType or
// AddLabel; add a new client method that writes; route a write through
// c.graphQL with a new document.
func TestEveryPathToGitHubPassesTheDryRunChokepoint(t *testing.T) {
	fset := token.NewFileSet()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read the package directory: %v", err)
	}

	// A sanity floor on the scan itself: if it finds no GitHub-reaching function
	// at all, the detector is broken and every assertion below is vacuous.
	reached := 0
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		for fnName, body := range functionBodies(file) {
			why, gated := reachesGitHub(body)
			if why == "" {
				continue
			}
			reached++
			if reason, exempt := transportFuncs[fnName]; exempt {
				t.Logf("%s: %s reaches GitHub via %s (exempt: %s)", name, fnName, why, reason)
				continue
			}
			if !gated {
				t.Errorf("%s: %s reaches GitHub via %s without a shouldSkip that decides whether it returns, "+
					"so -dry-run would not suppress it", name, fnName, why)
			}
		}
	}
	if reached < 3 {
		t.Fatalf("the scan found only %d functions reaching GitHub; expected at least graphQL, SetType and AddLabel, "+
			"so the detector is not working and its green result means nothing", reached)
	}
}

// functionBodies yields every function body in the file by name, including
// package-level function literals -- a mutation written as `var f = func(...)`
// sits in a GenDecl and would otherwise never be scanned.
func functionBodies(file *ast.File) map[string]*ast.BlockStmt {
	out := make(map[string]*ast.BlockStmt)
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if d.Body != nil {
				out[d.Name.Name] = d.Body
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				vs, ok := spec.(*ast.ValueSpec)
				if !ok {
					continue
				}
				for i, value := range vs.Values {
					lit, ok := value.(*ast.FuncLit)
					if !ok || i >= len(vs.Names) {
						continue
					}
					out[vs.Names[i].Name] = lit.Body
				}
			}
		}
	}
	return out
}

// reachesGitHub reports how a body reaches GitHub (empty if it does not), and
// whether a shouldSkip call decides whether the function returns.
func reachesGitHub(body *ast.BlockStmt) (why string, gated bool) {
	ast.Inspect(body, func(n ast.Node) bool {
		if ifStmt, ok := n.(*ast.IfStmt); ok && condCalls(ifStmt.Cond, "shouldSkip") && bodyReturns(ifStmt.Body) {
			gated = true
			return true
		}
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		// Any call rooted at the REST client, whatever it is named.
		if rootedAt(sel.X, "rest") && why == "" {
			why = "the REST client"
		}
		// A GraphQL call carrying a document that is not a known read.
		if sel.Sel.Name == "graphQL" && !readOnlyQueries[queryName(call)] {
			why = "the GraphQL transport with a document that is not a known read"
		}
		return true
	})
	return why, gated
}

func condCalls(cond ast.Expr, name string) bool {
	found := false
	ast.Inspect(cond, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == name {
				found = true
			}
		}
		return true
	})
	return found
}

func bodyReturns(body *ast.BlockStmt) bool {
	returns := false
	ast.Inspect(body, func(n ast.Node) bool {
		if _, ok := n.(*ast.ReturnStmt); ok {
			returns = true
		}
		return true
	})
	return returns
}

// queryName returns the identifier of the GraphQL document a c.graphQL call
// passes, or "" when it is not a plain identifier -- which is itself not a
// known read, so it counts as a mutation.
func queryName(call *ast.CallExpr) string {
	if len(call.Args) < 2 {
		return ""
	}
	ident, ok := call.Args[1].(*ast.Ident)
	if !ok {
		return ""
	}
	return ident.Name
}

// rootedAt reports whether expr is a selector chain rooted at a field of the
// given name, e.g. c.rest.Issues.Anything.
func rootedAt(expr ast.Expr, field string) bool {
	for {
		switch e := expr.(type) {
		case *ast.SelectorExpr:
			if e.Sel.Name == field {
				return true
			}
			expr = e.X
		case *ast.Ident:
			return e.Name == field
		default:
			return false
		}
	}
}
