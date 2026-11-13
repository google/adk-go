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

package runner

import (
	"testing"

	"google.golang.org/adk/v2/tool"
)

// TestRunner_Close_ClosesPrefixedAndPlainToolsetOnce checks that a toolset
// held directly and behind two prefix wrappers is closed once, because the
// wrappers unwrap to it. A wrapper comes first, so the runner closes ts
// through the wrapper's Close.
func TestRunner_Close_ClosesPrefixedAndPlainToolsetOnce(t *testing.T) {
	ts := &countingToolset{name: "ts"}
	root := newCloseTestLLM(t, "root", []tool.Toolset{tool.WithToolNamePrefix(ts, "a"), ts, tool.WithToolNamePrefix(ts, "b")})
	r := newCloseTestRunner(t, Config{Agent: root})

	if err := r.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	if got := ts.closes.Load(); got != 1 {
		t.Errorf("closes of ts = %d, want 1", got)
	}
}
