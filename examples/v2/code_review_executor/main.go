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

// code_review_executor demonstrates the codeexec interface with the
// unsafelocal backend. A code-review bot runs reviewer-supplied Python
// snippets in a subprocess sandbox.
package main

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"google.golang.org/adk/codeexec"
	_ "google.golang.org/adk/codeexec/unsafelocal" // registers "unsafe_local"
)

type Snippet struct {
	Name string
	Code string
}

func main() {
	if !havePython() {
		fmt.Println("(no python3 on PATH — skipping demo)")
		return
	}

	exec, err := codeexec.Build("unsafe_local", map[string]any{
		"max_runtime_seconds": 2.0,
	})
	if err != nil {
		fmt.Println("Build:", err)
		return
	}

	cases := []Snippet{
		{
			Name: "correct fix",
			Code: `
def add(a, b): return a + b
assert add(2, 3) == 5
print("PASS: add(2,3)==5")
`,
		},
		{
			Name: "broken fix (assertion fails)",
			Code: `
def divide(a, b): return a // b   # rounds; reviewer expected float
assert divide(7, 2) == 3.5, "expected 3.5"
print("PASS")
`,
		},
		{
			Name: "infinite loop (caught by MaxRuntime)",
			Code: `
while True: pass
`,
		},
	}

	ctx := context.Background()
	for _, c := range cases {
		fmt.Printf("=== running: %s ===\n", c.Name)
		start := time.Now()
		ch, err := exec.Execute(ctx, codeexec.Input{
			Language:    "python3",
			Code:        c.Code,
			TimeoutHint: 1500 * time.Millisecond,
		})
		if err != nil {
			fmt.Println("  Execute:", err)
			continue
		}
		var stdout, stderr strings.Builder
		var exitCode int
		var runErr error
		for chunk := range ch {
			stdout.Write(chunk.Stdout)
			stderr.Write(chunk.Stderr)
			if chunk.ExitCode != nil {
				exitCode = *chunk.ExitCode
			}
			if chunk.Err != nil {
				runErr = chunk.Err
			}
		}
		elapsed := time.Since(start)
		fmt.Printf("  exit=%d elapsed=%s\n", exitCode, elapsed.Round(time.Millisecond))
		if s := stdout.String(); s != "" {
			fmt.Printf("  stdout: %s", s)
		}
		if s := stderr.String(); s != "" {
			fmt.Printf("  stderr: %s", strings.TrimRight(s, "\n"))
			fmt.Println()
		}
		if runErr != nil {
			fmt.Printf("  run error: %v\n", runErr)
		}
		fmt.Println()
	}
}

func havePython() bool {
	_, err := exec.LookPath("python3")
	if err == nil {
		return true
	}
	_, err = exec.LookPath("python")
	return err == nil
}
