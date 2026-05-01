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

// Package unsafelocal provides a code executor that runs code as a
// local subprocess on the same host as the agent process. As the name
// implies, this offers NO sandboxing — the code has the same privileges
// as the agent. It is intended only for local development against
// trusted agents.
//
// Production deployments should use codeexec/container,
// codeexec/vertexai, codeexec/gke, or codeexec/agentenginesandbox.
package unsafelocal

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"google.golang.org/adk/codeexec"
)

// Executor runs code in a fresh subprocess on each Execute call. The
// language identifier maps to the interpreter binary (python, python3,
// bash, sh, node, etc.) — see DefaultLanguageBinaries for the mapping.
type Executor struct {
	// Languages overrides DefaultLanguageBinaries. Map key is the language
	// identifier passed via Input.Language; value is the interpreter
	// binary on the host (e.g. "/usr/bin/python3"). When nil the defaults
	// apply.
	Languages map[string]string

	// MaxRuntime caps wall-clock per execution. 0 = no cap (controlled by
	// Input.TimeoutHint and ctx).
	MaxRuntime time.Duration
}

// DefaultLanguageBinaries is the language → interpreter mapping used when
// Executor.Languages is nil.
var DefaultLanguageBinaries = map[string]string{
	"python":  pythonBinary(),
	"python3": pythonBinary(),
	"bash":    "bash",
	"sh":      "sh",
	"node":    "node",
}

func pythonBinary() string {
	// Default to python3 on Unix; fall back to python on Windows where
	// the Microsoft Store launcher uses that name.
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

// Name implements codeexec.Executor.
func (e *Executor) Name() string { return "unsafe_local" }

// Capabilities implements codeexec.Executor.
func (e *Executor) Capabilities() codeexec.Capabilities {
	langs := make([]string, 0, len(e.languages()))
	for k := range e.languages() {
		langs = append(langs, k)
	}
	return codeexec.Capabilities{
		Languages:       langs,
		Stateful:        false,
		NetworkAccess:   true, // host process retains network
		InstallPackages: false,
		// FileIO is false: Input.Files is not materialized to disk and
		// Chunk.Files is never populated by this executor. Truthful
		// reporting prevents callers from assuming file plumbing they'd
		// only get from a sandboxed backend.
		FileIO:     false,
		MaxRuntime: e.MaxRuntime,
	}
}

func (e *Executor) languages() map[string]string {
	if e.Languages != nil {
		return e.Languages
	}
	return DefaultLanguageBinaries
}

// Execute implements codeexec.Executor.
func (e *Executor) Execute(ctx context.Context, in codeexec.Input) (<-chan codeexec.Chunk, error) {
	bin, ok := e.languages()[in.Language]
	if !ok {
		return nil, fmt.Errorf("unsafe_local: unsupported language %q", in.Language)
	}
	if in.Code == "" {
		return nil, errors.New("unsafe_local: empty Code")
	}

	timeout := pickTimeout(e.MaxRuntime, in.TimeoutHint)
	cctx := ctx
	var cancel context.CancelFunc
	if timeout > 0 {
		cctx, cancel = context.WithTimeout(ctx, timeout)
	}

	args, stdinSrc, err := buildArgs(bin, in)
	if err != nil {
		if cancel != nil {
			cancel()
		}
		return nil, err
	}

	cmd := exec.CommandContext(cctx, args[0], args[1:]...)
	if stdinSrc != nil {
		cmd.Stdin = stdinSrc
	}
	cmd.Env = envFor(in.Env)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	out := make(chan codeexec.Chunk, 1)

	go func() {
		defer close(out)
		if cancel != nil {
			defer cancel()
		}
		err := cmd.Run()
		chunk := codeexec.Chunk{
			Stdout: copyBytes(stdout.Bytes()),
			Stderr: copyBytes(stderr.Bytes()),
		}
		if cmd.ProcessState != nil {
			code := cmd.ProcessState.ExitCode()
			chunk.ExitCode = &code
		}
		if err != nil {
			chunk.Err = err
		}
		out <- chunk
	}()
	return out, nil
}

func pickTimeout(execMax, hint time.Duration) time.Duration {
	switch {
	case execMax <= 0:
		return hint
	case hint <= 0:
		return execMax
	case hint < execMax:
		return hint
	default:
		return execMax
	}
}

func buildArgs(bin string, in codeexec.Input) ([]string, io.Reader, error) {
	switch {
	case strings.HasPrefix(filepath.Base(bin), "python"):
		// python -c <code>
		return []string{bin, "-c", in.Code}, bytesReader(in.Stdin), nil
	case filepath.Base(bin) == "node":
		return []string{bin, "-e", in.Code}, bytesReader(in.Stdin), nil
	default:
		// shell-like: pipe the code through stdin.
		return []string{bin, "-s"}, io.MultiReader(strings.NewReader(in.Code+"\n"), bytesReader(in.Stdin)), nil
	}
}

func bytesReader(b []byte) io.Reader {
	if len(b) == 0 {
		return nil
	}
	return bytes.NewReader(b)
}

func envFor(extra map[string]string) []string {
	if len(extra) == 0 {
		return nil
	}
	out := make([]string, 0, len(extra))
	for k, v := range extra {
		out = append(out, k+"="+v)
	}
	return out
}

func copyBytes(b []byte) []byte {
	if len(b) == 0 {
		return nil
	}
	out := make([]byte, len(b))
	copy(out, b)
	return out
}

// init registers the unsafe_local executor in the codeexec registry so
// it can be referenced by name from configuration.
func init() {
	codeexec.Register("unsafe_local", func(cfg map[string]any) (codeexec.Executor, error) {
		exec := &Executor{}
		if v, ok := cfg["max_runtime_seconds"]; ok {
			if seconds, ok := v.(float64); ok {
				exec.MaxRuntime = time.Duration(seconds * float64(time.Second))
			}
		}
		return exec, nil
	})
}
