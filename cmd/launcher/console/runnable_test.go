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

package console

import (
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
)

func TestNewRunnableUsesValues(t *testing.T) {
	r, err := NewRunnable(Config{StreamingMode: agent.StreamingModeSSE, OTelToCloud: true})
	if err != nil {
		t.Fatalf("NewRunnable() failed: %v", err)
	}
	c, ok := r.(*consoleLauncher)
	if !ok {
		t.Fatalf("NewRunnable() returned %T, want *consoleLauncher", r)
	}
	if c.config.streamingMode != agent.StreamingModeSSE {
		t.Errorf("streamingMode = %q, want %q", c.config.streamingMode, agent.StreamingModeSSE)
	}
	if !c.config.otelToCloud {
		t.Error("otelToCloud = false, want true")
	}
	if c.config.shutdownTimeout != 2*time.Second {
		t.Errorf("shutdownTimeout = %v, want the 2s default", c.config.shutdownTimeout)
	}
}

func TestNewRunnableKeepsAutoStreamingMode(t *testing.T) {
	r, err := NewRunnable(Config{})
	if err != nil {
		t.Fatalf("NewRunnable() failed: %v", err)
	}
	if got := r.(*consoleLauncher).config.streamingMode; got != "" {
		t.Errorf("streamingMode = %q, want the empty auto-detect value", got)
	}
}

func TestNewRunnableRejectsInvalidStreamingMode(t *testing.T) {
	if _, err := NewRunnable(Config{StreamingMode: "bogus"}); err == nil {
		t.Fatal("NewRunnable() succeeded for an unknown streaming mode, want an error")
	}
}
