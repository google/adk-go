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
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestPrepareDockerfile_BindsAllInterfaces guards that the generated Cloud Run
// container explicitly binds the web server to all interfaces. The web server
// defaults to loopback-only, which would otherwise prevent Cloud Run from
// routing traffic to the container.
func TestPrepareDockerfile_BindsAllInterfaces(t *testing.T) {
	dir := t.TempDir()
	dockerfile := filepath.Join(dir, "Dockerfile")

	oldFlags := flags
	t.Cleanup(func() { flags = oldFlags })

	flags.cloudRun.serverPort = 8080
	flags.build.execFile = "server"
	flags.build.tempDir = dir
	flags.build.dockerfileBuildPath = dockerfile
	flags.source.entryPointPath = "./main.go"

	if err := flags.prepareDockerfile(); err != nil {
		t.Fatalf("prepareDockerfile() error = %v", err)
	}
	content, err := os.ReadFile(dockerfile)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(content), `"web", "-host", "0.0.0.0", "-port", "8080"`) {
		t.Errorf("Dockerfile CMD must bind the web server to all interfaces with -host 0.0.0.0:\n%s", content)
	}
}
