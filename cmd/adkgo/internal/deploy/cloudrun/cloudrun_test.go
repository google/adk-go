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

// resetFlags restores the package-level flags var to a fresh, minimally valid
// state before each test, since computeFlags reads and writes it directly.
func resetFlags(t *testing.T, entryPointPath, serviceName string) {
	t.Helper()
	flags = deployCloudRunFlags{}
	flags.source.entryPointPath = entryPointPath
	flags.build.tempDir = t.TempDir()
	flags.cloudRun.serviceName = serviceName
}

func TestComputeFlags_RejectsUnsafeServiceName(t *testing.T) {
	// serviceName is placed in gcloud's positional SERVICE slot; a leading '-'
	// would be parsed by gcloud as a flag rather than a service name
	// (argument injection, CWE-88).
	for _, name := range []string{
		"--flags-file=/tmp/evil.yaml", // the injection payload
		"-s",
		"-agent",
		"Agent",                 // uppercase not allowed
		"agent_name",            // underscore not allowed
		"agent.name",            // dot not allowed
		"agent-",                // trailing dash not allowed
		"",                      // empty
		strings.Repeat("a", 64), // longer than 63
	} {
		resetFlags(t, "main.go", name)
		if err := flags.computeFlags(); err == nil {
			t.Errorf("computeFlags() with serviceName %q = nil, want an error", name)
		}
	}
}

func TestComputeFlags_AcceptsValidServiceNames(t *testing.T) {
	// Every value here is accepted by gcloud's own service-name validation.
	for _, name := range []string{
		"my-agent",
		"a",
		"9agent",
		"0abc",
		"ab--cd",
		strings.Repeat("a", 63),
	} {
		resetFlags(t, "main.go", name)
		if err := flags.computeFlags(); err != nil {
			t.Errorf("computeFlags() with valid serviceName %q = %v, want nil", name, err)
		}
	}
}
