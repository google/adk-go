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
	"testing"
)

// Source-level pins on how run() is WIRED.
//
// Everything else in this suite tests behavior. These do not, and the reason is
// specific: run() is the composition root. By the time it fans out it holds a
// real GitHub client and a real Gemini model, so no test can call it and there
// is no seam to inject at. That makes one particular regression invisible to
// every runtime test that could ever be written here:
//
//	sharedStack, _ := newAgentStack(cfg, mdl, tools, log)
//	triageAll(ctx, cfg, log, prs, func(ictx context.Context, number int) {
//	    triageOne(ictx, gh, cfg, log, number, func(...) string {
//	        return runAgent(actx, sharedStack.runner, sharedStack.sessions, ...)
//	    })
//	})
//
// That hands every batch worker one agent, which is the llmInternalState.Mode
// write/write race newAgentStack exists to prevent. Measured against this
// package: it compiles, `go vet`, `gofmt` and `golangci-lint` are clean, and the
// full suite passes 5 of 5 under `-race -shuffle=on`. The runtime concurrency
// pin misses it because that pin calls triageWithFreshAgent itself, so it keeps
// passing after run() stops calling it — the original defect, one level up. And
// `unused` stays quiet because triageWithFreshAgent still has test callers.
//
// So the invariant is pinned where it is written rather than where it runs.
// Parsing the source has no timing element, so unlike a race test it is
// deterministic rather than probabilistic.
//
// These are deliberately brittle to restructuring the fan-out. That is the
// point: rearranging this wiring is exactly the moment someone should have to
// come here and re-read why the agent stack is per pull request. If you are
// here because one of these failed, the question to answer is not "how do I
// make the test pass" but "does every concurrent worker still get its own
// agent".

// parseMain returns the AST of main.go.
func parseMain(t *testing.T) (*token.FileSet, *ast.File) {
	t.Helper()
	fset := token.NewFileSet()
	// Not skipped when absent: these pins are the only thing covering this
	// regression, so a missing file must fail rather than quietly pass.
	f, err := parser.ParseFile(fset, "main.go", nil, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse main.go: %v", err)
	}
	return fset, f
}

// findFunc returns the named top-level function.
func findFunc(t *testing.T, f *ast.File, name string) *ast.FuncDecl {
	t.Helper()
	for _, d := range f.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok && fn.Recv == nil && fn.Name.Name == name {
			return fn
		}
	}
	t.Fatalf("main.go declares no func %s", name)
	return nil
}

// calleeName returns the name of the function a call expression targets, for a
// plain identifier call like f(...). Returns "" for anything else.
func calleeName(call *ast.CallExpr) string {
	if id, ok := call.Fun.(*ast.Ident); ok {
		return id.Name
	}
	return ""
}

// callsTo collects every call to the named function inside n.
func callsTo(n ast.Node, name string) []*ast.CallExpr {
	var out []*ast.CallExpr
	ast.Inspect(n, func(node ast.Node) bool {
		if call, ok := node.(*ast.CallExpr); ok && calleeName(call) == name {
			out = append(out, call)
		}
		return true
	})
	return out
}

// run() must hand triageAll a worker that routes through triageWithFreshAgent,
// because that is where the per-pull-request agent stack is built. A worker that
// closes over a stack built above the fan-out shares one agent across every
// concurrent pull request.
func TestRunFansOutThroughThePerPullRequestWorker(t *testing.T) {
	_, f := parseMain(t)
	runFn := findFunc(t, f, "run")

	fanOuts := callsTo(runFn, "triageAll")
	if len(fanOuts) != 1 {
		t.Fatalf("run() calls triageAll %d times, want exactly 1", len(fanOuts))
	}
	args := fanOuts[0].Args
	if len(args) == 0 {
		t.Fatal("run()'s triageAll call has no arguments")
	}

	// The worker is the last argument.
	worker, ok := args[len(args)-1].(*ast.FuncLit)
	if !ok {
		t.Fatalf("run() passes a %T as triageAll's worker, want an inline func literal "+
			"whose body calls triageWithFreshAgent", args[len(args)-1])
	}
	if n := len(worker.Body.List); n != 1 {
		t.Fatalf("the worker literal has %d statements, want exactly 1 (a call to "+
			"triageWithFreshAgent). More than one statement usually means the agent "+
			"stack is being built or captured somewhere it should not be", n)
	}
	stmt, ok := worker.Body.List[0].(*ast.ExprStmt)
	if !ok {
		t.Fatalf("the worker's only statement is %T, want a call to triageWithFreshAgent", worker.Body.List[0])
	}
	call, ok := stmt.X.(*ast.CallExpr)
	if !ok {
		t.Fatalf("the worker's only statement is not a call: %T", stmt.X)
	}
	if got := calleeName(call); got != "triageWithFreshAgent" {
		t.Errorf("run()'s worker calls %q, want triageWithFreshAgent. Every concurrent "+
			"worker must build its own agent stack; see newAgentStack's doc comment for "+
			"the race this prevents", got)
	}
}

// The other half of the same property: run() must NOT build an agent stack
// itself. One built at the composition root can only be shared, whatever the
// worker then does with it.
func TestRunDoesNotBuildAnAgentStackItself(t *testing.T) {
	_, f := parseMain(t)
	runFn := findFunc(t, f, "run")

	if calls := callsTo(runFn, "newAgentStack"); len(calls) != 0 {
		t.Errorf("run() calls newAgentStack %d time(s). The stack must be built inside "+
			"each worker (triageWithFreshAgent), not once at the composition root, or "+
			"every batch worker shares one agent and races on its mode", len(calls))
	}
}

// And the worker must actually build one, or the two tests above would be
// satisfied by a worker that shares a stack obtained some other way.
func TestThePerPullRequestWorkerBuildsItsOwnAgentStack(t *testing.T) {
	_, f := parseMain(t)
	worker := findFunc(t, f, "triageWithFreshAgent")

	if calls := callsTo(worker, "newAgentStack"); len(calls) != 1 {
		t.Errorf("triageWithFreshAgent calls newAgentStack %d time(s), want exactly 1: "+
			"it is the function whose whole purpose is giving one pull request its own "+
			"agent", len(calls))
	}
}
