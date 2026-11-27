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

// Package versiontest holds helpers for tests of code that sends ADK's
// tracking headers.
package versiontest

import "testing"

// RuntimeEnvVars returns the variables that select a runtime suffix. The
// names are literals because services parse them.
func RuntimeEnvVars() []string {
	return []string{"GOOGLE_CLOUD_AGENT_ENGINE_ID", "K_SERVICE", "CLOUD_RUN_JOB", "KUBERNETES_SERVICE_HOST"}
}

// ClearRuntimeEnv hides, for the rest of the test, the runtime the test
// itself runs on, which may be one of the runtimes ADK detects.
func ClearRuntimeEnv(tb testing.TB) {
	tb.Helper()
	for _, name := range RuntimeEnvVars() {
		tb.Setenv(name, "")
	}
}
