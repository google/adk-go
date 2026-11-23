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

package mcptoolset

import (
	"testing"
	"time"

	"google.golang.org/adk/v2/tool"
)

// SetSessionBoundsForTest sets the idle timeout and size cap of the sessions
// kept with Config.Auth, restoring them when t ends.
func SetSessionBoundsForTest(t *testing.T, idle time.Duration, maxOpen int) {
	oldIdle, oldMax := sessionIdleTimeout, maxSessions
	sessionIdleTimeout, maxSessions = idle, maxOpen
	t.Cleanup(func() { sessionIdleTimeout, maxSessions = oldIdle, oldMax })
}

// SessionCountForTest returns how many sessions a toolset built with
// Config.Auth keeps.
func SessionCountForTest(ts tool.Toolset) int {
	p := ts.(*set).mcpClient.(*credentialSessions)
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.sessions)
}
