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
