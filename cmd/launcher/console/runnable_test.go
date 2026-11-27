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
	"context"
	"strings"
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// consoleConfigOf unwraps the launcher returned by NewRunnable.
func consoleConfigOf(t *testing.T, r launcher.Runnable) *consoleConfig {
	t.Helper()
	v, ok := r.(validatingRunnable)
	if !ok {
		t.Fatalf("NewRunnable() returned %T, want validatingRunnable", r)
	}
	c, ok := v.inner.(*consoleLauncher)
	if !ok {
		t.Fatalf("wrapped launcher is %T, want *consoleLauncher", v.inner)
	}
	return c.config
}

func TestNewRunnableUsesValues(t *testing.T) {
	r, err := NewRunnable(Config{StreamingMode: agent.StreamingModeSSE, OTelToCloud: true})
	if err != nil {
		t.Fatalf("NewRunnable() failed: %v", err)
	}
	c := consoleConfigOf(t, r)
	if c.streamingMode != agent.StreamingModeSSE {
		t.Errorf("streamingMode = %q, want %q", c.streamingMode, agent.StreamingModeSSE)
	}
	if !c.otelToCloud {
		t.Error("otelToCloud = false, want true")
	}
	if c.shutdownTimeout != 2*time.Second {
		t.Errorf("shutdownTimeout = %v, want the 2s default", c.shutdownTimeout)
	}
}

func TestNewRunnableKeepsAutoStreamingMode(t *testing.T) {
	r, err := NewRunnable(Config{})
	if err != nil {
		t.Fatalf("NewRunnable() failed: %v", err)
	}
	if got := consoleConfigOf(t, r).streamingMode; got != "" {
		t.Errorf("streamingMode = %q, want the empty auto-detect value", got)
	}
}

func TestNewRunnableRejectsInvalidStreamingMode(t *testing.T) {
	if _, err := NewRunnable(Config{StreamingMode: "bogus"}); err == nil {
		t.Fatal("NewRunnable() succeeded for an unknown streaming mode, want an error")
	}
}

// countingSessionService records Create calls, so a test can tell a Run that
// rejected the config from one that got as far as creating a session.
type countingSessionService struct {
	session.Service
	creates int
}

func (s *countingSessionService) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	s.creates++
	return s.Service.Create(ctx, req)
}

func TestRunRejectsInvalidConfigBeforeCreatingASession(t *testing.T) {
	svc := &countingSessionService{Service: session.InMemoryService()}
	r, err := NewRunnable(Config{})
	if err != nil {
		t.Fatalf("NewRunnable() failed: %v", err)
	}

	err = r.Run(context.Background(), &launcher.Config{
		Compaction:     &compaction.Config{OverlapSize: 2},
		SessionService: svc,
	})
	if err == nil {
		t.Fatal("Run() started on a compaction config that cannot work")
	}
	if !strings.Contains(err.Error(), "Compaction") {
		t.Errorf("error %q does not name the field an operator has to change", err)
	}
	if svc.creates != 0 {
		t.Errorf("Create calls = %d, want 0: the config was rejected after a session was made", svc.creates)
	}
}
