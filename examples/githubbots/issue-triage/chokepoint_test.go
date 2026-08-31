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

// readOnlyRequesters are the functions allowed to issue a non-GET request
// without passing through shouldSkip, with the reason each is exempt.
var readOnlyRequesters = map[string]string{
	// GitHub's GraphQL endpoint is POST for reads too. Its two queries are
	// package-level constants and neither is a mutation.
	"graphQL": "POSTs a constant read-only GraphQL query",
}

// The dry-run chokepoint, pinned structurally rather than by example.
//
// TestRunInDryRunWritesNothing proves the two mutations that exist today are
// suppressed, which is a statement about them and not about the next one. This
// is the counterpart of TestTools for the tool inventory: it reads the
// package's own source and fails when a function issues a mutating request
// without passing through shouldSkip, so a third mutation cannot be added
// silently.
//
// Killing mutation: delete the shouldSkip call from SetType or AddLabel.
func TestEveryMutatingRequestPassesTheDryRunChokepoint(t *testing.T) {
	// os.ReadDir plus ParseFile rather than parser.ParseDir, which is
	// deprecated and which staticcheck rejects.
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read the package directory: %v", err)
	}
	fset := token.NewFileSet()

	// A sanity check on the scan itself: if the walk finds no request-issuing
	// function at all, the detector is broken and every assertion is vacuous.
	found := 0
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			mutates, gated := scanFunc(fn)
			if !mutates {
				continue
			}
			found++
			if why, exempt := readOnlyRequesters[fn.Name.Name]; exempt {
				t.Logf("%s: %s (exempt: %s)", name, fn.Name.Name, why)
				continue
			}
			if !gated {
				t.Errorf("%s: %s issues a mutating request without calling shouldSkip, so -dry-run would not suppress it",
					name, fn.Name.Name)
			}
		}
	}
	if found < 3 {
		t.Fatalf("the scan found only %d request-issuing functions; expected at least graphQL, SetType and AddLabel, so the detector is not working", found)
	}
}

// scanFunc reports whether fn issues a request that can change GitHub state,
// and whether it calls shouldSkip.
func scanFunc(fn *ast.FuncDecl) (mutates, gated bool) {
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		switch {
		case sel.Sel.Name == "shouldSkip":
			gated = true
		case sel.Sel.Name == "NewRequest":
			// c.rest.NewRequest(method, ...). Any method reaches the network.
			mutates = true
		case strings.HasPrefix(sel.Sel.Name, "Add"), strings.HasPrefix(sel.Sel.Name, "Create"),
			strings.HasPrefix(sel.Sel.Name, "Edit"), strings.HasPrefix(sel.Sel.Name, "Replace"),
			strings.HasPrefix(sel.Sel.Name, "Remove"), strings.HasPrefix(sel.Sel.Name, "Delete"):
			// A go-github service method whose name says it writes.
			if isRESTClient(sel.X) {
				mutates = true
			}
		}
		return true
	})
	return mutates, gated
}

// isRESTClient reports whether expr is rooted at the client's rest field, which
// is the only way this package reaches GitHub.
func isRESTClient(expr ast.Expr) bool {
	for {
		sel, ok := expr.(*ast.SelectorExpr)
		if !ok {
			ident, ok := expr.(*ast.Ident)
			return ok && ident.Name == "rest"
		}
		if sel.Sel.Name == "rest" {
			return true
		}
		expr = sel.X
	}
}
