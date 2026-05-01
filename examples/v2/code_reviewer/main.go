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

// Code-reviewer agent: a Gemini agent that runs the reviewer's Python
// snippet via the codeexec/unsafelocal backend and reports back stdout,
// stderr, and exit code. Demonstrates wiring a codeexec.Executor as a
// function tool.
//
// SECURITY: unsafelocal is for local development only — no sandbox.
// For production swap in codeexec/container or another isolated backend.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/codeexec"
	_ "google.golang.org/adk/codeexec/unsafelocal" // registers "unsafe_local"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

const instruction = `You are a code-review assistant.

When the user gives you a Python snippet to evaluate (typically a fix
or a small test harness), call the run_python tool with the code. The
tool returns {stdout, stderr, exit_code, runtime_ms}.

Then write a short review:
  - Did it pass? (exit_code == 0 + no AssertionError in stderr)
  - If it failed, point to the line of stderr that explains why.
  - If it timed out, suggest a tighter algorithm.

Be concise — assume the user will see your review next to a diff.`

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, modelName(), &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	runner, err := codeexec.Build("unsafe_local", map[string]any{
		"max_runtime_seconds": 5.0,
	})
	if err != nil {
		log.Fatal(err)
	}

	runPython, err := buildRunPythonTool(runner)
	if err != nil {
		log.Fatal(err)
	}

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "code_reviewer",
		Description: "A code-review agent that runs reviewer-supplied Python snippets and reports the result.",
		Model:       model,
		Instruction: instruction,
		Tools:       []tool.Tool{runPython},
	})
	if err != nil {
		log.Fatal(err)
	}

	cfg := &launcher.Config{AgentLoader: agent.NewSingleLoader(rootAgent)}
	l := full.NewLauncher()
	if err = l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

func modelName() string {
	if v := os.Getenv("GOOGLE_GENAI_MODEL"); v != "" {
		return v
	}
	return "gemini-2.5-flash"
}

func buildRunPythonTool(executor codeexec.Executor) (tool.Tool, error) {
	type args struct {
		Code string `json:"code"`
	}
	type result struct {
		Stdout    string `json:"stdout"`
		Stderr    string `json:"stderr"`
		ExitCode  int    `json:"exit_code"`
		RuntimeMS int64  `json:"runtime_ms"`
		TimedOut  bool   `json:"timed_out"`
	}
	return functiontool.New[args, result](
		functiontool.Config{
			Name:        "run_python",
			Description: "Execute a Python snippet (python3) and return stdout/stderr/exit code. Sandboxed to 5 seconds wall clock.",
		},
		func(ctx tool.Context, a args) (result, error) {
			if a.Code == "" {
				return result{}, errors.New("code must not be empty")
			}
			start := time.Now()
			ch, err := executor.Execute(ctx, codeexec.Input{
				Language:    "python3",
				Code:        a.Code,
				TimeoutHint: 5 * time.Second,
			})
			if err != nil {
				return result{}, fmt.Errorf("executor: %w", err)
			}
			var stdout, stderr strings.Builder
			r := result{}
			for chunk := range ch {
				stdout.Write(chunk.Stdout)
				stderr.Write(chunk.Stderr)
				if chunk.ExitCode != nil {
					r.ExitCode = *chunk.ExitCode
				}
				if chunk.Err != nil {
					if strings.Contains(chunk.Err.Error(), "killed") {
						r.TimedOut = true
					}
				}
			}
			r.Stdout = stdout.String()
			r.Stderr = stderr.String()
			r.RuntimeMS = time.Since(start).Milliseconds()
			return r, nil
		},
	)
}
