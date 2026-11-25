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
	"os/user"
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
	if os.Getenv(childEnv) == "" {
		// The kind of setting a developer's shell has. None may reach the child.
		t.Setenv("GOOGLE_API_CERTIFICATE_CONFIG", "/nonexistent/certificate_config.json")
		t.Setenv("CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE", "true")
	}
	if !Isolate(t) {
		return
	}
	PinOff()
	if env := os.Environ(); len(env) != 2 {
		t.Errorf("the child has %q after PinOff, want only HOME and the child marker", env)
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

func TestPinOff(t *testing.T) {
	// The child marker alone, as a shell could export it, is not a child.
	t.Setenv(childEnv, "TestSomethingElse")
	t.Setenv("HOME", t.TempDir())
	Setenv(t, "GOOGLE_API_USE_CLIENT_CERTIFICATE", Unset)
	PinOff()
	if got := os.Getenv("GOOGLE_API_USE_CLIENT_CERTIFICATE"); got != "false" {
		t.Errorf("GOOGLE_API_USE_CLIENT_CERTIFICATE = %q after PinOff, want false", got)
	}
}

// TestSecureConnectSkipsOnlyWhereUnsafe checks the guard in SecureConnect
// from the outside: the test must go on exactly where the auth library would
// look for SecureConnect in the child's home. Going on anywhere else would
// have a test run the machine's real helper.
func TestSecureConnectSkipsOnlyWhereUnsafe(t *testing.T) {
	if !Isolate(t) {
		return
	}
	// As SecureConnect sets it. os/user keeps its first answer.
	t.Setenv("USER", "clientcerttest")
	u, err := user.Current()
	safe := err == nil && u.HomeDir == os.Getenv("HOME")
	t.Cleanup(func() {
		if t.Skipped() == safe {
			t.Errorf("SecureConnect skipped: %v, where the library looks in the test's home: %v", t.Skipped(), safe)
		}
	})
	SecureConnect(t)
}
