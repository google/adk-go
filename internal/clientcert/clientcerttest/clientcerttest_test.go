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

package clientcerttest

import (
	"net/http"
	"os"
	"runtime"
	"testing"
)

// TestIsolateRunsOnLinux guards the tests built on Isolate and Server against
// all being skipped on the one system CI runs them on, which would still read
// as a pass.
func TestIsolateRunsOnLinux(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Isolate and Server may skip elsewhere")
	}
	t.Cleanup(func() {
		if t.Skipped() {
			t.Error("Isolate or Server skipped the test")
		}
	})
	if !Isolate(t) {
		return
	}
	for _, name := range cleared {
		if v, ok := os.LookupEnv(name); ok {
			t.Errorf("the child starts with %s=%q, want it unset", name, v)
		}
	}
	Server(t, http.NotFoundHandler())
}

// TestIsolateReportsAChildSkip always ends skipped: Isolate has to pass the
// child's skip on, neither failing the test nor counting it as run.
func TestIsolateReportsAChildSkip(t *testing.T) {
	if Isolate(t) {
		t.Skip("skipped in the child")
	}
	t.Error("Isolate returned in the parent after its child skipped")
}
