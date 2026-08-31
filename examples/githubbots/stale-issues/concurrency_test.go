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

package main

import (
	"context"
	"io"
	"iter"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// stubLLM answers immediately with a final text part and calls no tools, so an
// audit completes without a network round trip.
type stubLLM struct{}

func (stubLLM) Name() string { return "stub" }

func (stubLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content: genai.NewContentFromText("Analysis for Issue: no action.", genai.RoleModel),
		}, nil)
	}
}

// The agent, session service and runner are built per audit because sharing
// them races. runner.Run writes the root agent's mode on its first call, so two
// concurrent audits on one agent are a write/write on that field — reproduced
// here under -race before the fix, at runner.go's LlmAgent node path.
//
// This is live on any real sweep: CONCURRENCY_LIMIT defaults to 3 and the search
// routinely returns more than one candidate.
func TestConcurrentAuditsDoNotShareAgentState(t *testing.T) {
	cfg := baseCfg()
	cfg.Concurrency = 8
	cfg.IssueTimeout = time.Minute
	log := discardLogger()

	var ran atomic.Int64
	err := auditAll(context.Background(), cfg, log, []int{1, 2, 3, 4, 5, 6, 7, 8},
		func(ictx context.Context, n int) error {
			r, ss, err := newAuditRunner(cfg, stubLLM{}, nil, log)
			if err != nil {
				return err
			}
			ran.Add(1)
			return runAudit(ictx, r, ss, log, n)
		})
	if err != nil {
		t.Fatalf("auditAll: %v", err)
	}
	if ran.Load() != 8 {
		t.Fatalf("drove %d audits, want 8: the fan-out did not exercise the shared state", ran.Load())
	}
}

// And the factory must actually produce a fresh agent each time. Memoizing it
// into a singleton would compile, pass every other test, and put the race back.
func TestNewAuditRunnerReturnsAFreshRunnerEachCall(t *testing.T) {
	cfg := baseCfg()
	log := discardLogger()
	r1, ss1, err := newAuditRunner(cfg, stubLLM{}, nil, log)
	if err != nil {
		t.Fatalf("newAuditRunner: %v", err)
	}
	r2, ss2, err := newAuditRunner(cfg, stubLLM{}, nil, log)
	if err != nil {
		t.Fatalf("newAuditRunner: %v", err)
	}
	if r1 == r2 {
		t.Error("both calls returned the same *runner.Runner: concurrent audits would share the agent it wraps")
	}
	if ss1 == ss2 {
		t.Error("both calls returned the same session service")
	}
}

// GET /user is a user-to-server endpoint and the workflow's GITHUB_TOKEN is an
// installation token, so identity discovery is refused on every scheduled run.
// The login therefore has to be configured, and configuring it must skip the
// call entirely rather than merely overriding its result.
func TestConfiguredBotLoginSkipsIdentityDiscovery(t *testing.T) {
	var calls int
	cfg := baseCfg()
	cfg.BotLogin = "github-actions[bot]"
	c := testClient(t, cfg, forbiddenHandler(&calls))

	if got := resolveSelfLogin(context.Background(), c.rest, cfg, discardLogger()); got != "github-actions[bot]" {
		t.Errorf("resolveSelfLogin = %q, want the configured login", got)
	}
	if calls != 0 {
		t.Errorf("made %d API calls with BOT_LOGIN set, want 0: GET /user is refused for an installation token", calls)
	}
}

// Without it the bot must still start — a local run under a personal token can
// discover the login — but the run has to say the fallback is in use, and name
// the setting that fixes it.
func TestUnconfiguredBotLoginFallsBackAndSaysSo(t *testing.T) {
	var calls int
	cfg := baseCfg()
	cfg.BotLogin = ""
	var logged strings.Builder
	log := slog.New(slog.NewTextHandler(&logged, &slog.HandlerOptions{Level: slog.LevelWarn}))
	c := testClient(t, cfg, forbiddenHandler(&calls))

	if got := resolveSelfLogin(context.Background(), c.rest, cfg, log); got != "" {
		t.Errorf("resolveSelfLogin = %q, want empty after a refused discovery call", got)
	}
	if calls == 0 {
		t.Error("no API call was made with BOT_LOGIN unset: the local discovery path is gone")
	}
	if !strings.Contains(logged.String(), "BOT_LOGIN") {
		t.Errorf("the warning does not name BOT_LOGIN, so nobody reading the log learns the fix: %q", logged.String())
	}
}

// And a login that IS discoverable is used, so the fallback is not the only path.
func TestDiscoveredBotLoginIsUsed(t *testing.T) {
	cfg := baseCfg()
	cfg.BotLogin = ""
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"login":"local-dev"}`)
	}))
	if got := resolveSelfLogin(context.Background(), c.rest, cfg, discardLogger()); got != "local-dev" {
		t.Errorf("resolveSelfLogin = %q, want the discovered login", got)
	}
}

// BOT_LOGIN must reach the config from the environment.
func TestBotLoginFromEnv(t *testing.T) {
	setRequiredCreds(t)
	t.Setenv("BOT_LOGIN", "  github-actions[bot]  ")
	cfg, err := loadConfig(nil)
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.BotLogin != "github-actions[bot]" {
		t.Errorf("BotLogin = %q, want it read and trimmed", cfg.BotLogin)
	}
}

func forbiddenHandler(calls *int) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		*calls++
		w.WriteHeader(http.StatusForbidden)
		_, _ = io.WriteString(w, `{"message":"Resource not accessible by integration"}`)
	})
}
