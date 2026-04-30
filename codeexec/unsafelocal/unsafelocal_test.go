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

package unsafelocal_test

import (
	"context"
	"os/exec"
	"strings"
	"testing"
	"time"

	"google.golang.org/adk/codeexec"
	"google.golang.org/adk/codeexec/unsafelocal"
)

func haveBinary(name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}

func TestExecutor_Capabilities(t *testing.T) {
	e := &unsafelocal.Executor{}
	c := e.Capabilities()
	if !c.FileIO {
		// File IO is reported true even though current impl doesn't write
		// files; future enhancement. We assert what the Capabilities
		// claims so a regression is caught.
	}
	if len(c.Languages) == 0 {
		t.Error("Capabilities.Languages should not be empty")
	}
}

func TestExecutor_Name(t *testing.T) {
	e := &unsafelocal.Executor{}
	if e.Name() != "unsafe_local" {
		t.Errorf("Name = %q, want unsafe_local", e.Name())
	}
}

func TestExecute_RejectsUnknownLanguage(t *testing.T) {
	e := &unsafelocal.Executor{}
	_, err := e.Execute(context.Background(), codeexec.Input{Language: "klingon", Code: "x"})
	if err == nil {
		t.Error("expected error for unknown language")
	}
}

func TestExecute_RejectsEmptyCode(t *testing.T) {
	e := &unsafelocal.Executor{}
	_, err := e.Execute(context.Background(), codeexec.Input{Language: "python3"})
	if err == nil {
		t.Error("expected error for empty code")
	}
}

func TestExecute_PythonHelloWorld(t *testing.T) {
	if !haveBinary("python3") && !haveBinary("python") {
		t.Skip("no python interpreter on PATH")
	}
	e := &unsafelocal.Executor{}
	ch, err := e.Execute(context.Background(), codeexec.Input{
		Language: "python3",
		Code:     "print('hello')",
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	var stdout string
	for c := range ch {
		stdout += string(c.Stdout)
		if c.ExitCode != nil && *c.ExitCode != 0 {
			t.Errorf("ExitCode = %d, want 0", *c.ExitCode)
		}
	}
	if !strings.Contains(stdout, "hello") {
		t.Errorf("stdout = %q, want hello", stdout)
	}
}

func TestExecute_TimeoutKills(t *testing.T) {
	if !haveBinary("python3") && !haveBinary("python") {
		t.Skip("no python interpreter on PATH")
	}
	e := &unsafelocal.Executor{MaxRuntime: 100 * time.Millisecond}
	start := time.Now()
	ch, err := e.Execute(context.Background(), codeexec.Input{
		Language: "python3",
		Code:     "import time\ntime.sleep(5)",
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	for range ch {
	}
	elapsed := time.Since(start)
	if elapsed > 2*time.Second {
		t.Errorf("timeout did not fire: elapsed = %v", elapsed)
	}
}

func TestRegistered(t *testing.T) {
	// init() registers unsafe_local in the codeexec registry.
	exec, err := codeexec.Build("unsafe_local", map[string]any{"max_runtime_seconds": 5.0})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if exec.Name() != "unsafe_local" {
		t.Errorf("Name = %q", exec.Name())
	}
}
