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
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"golang.org/x/sync/errgroup"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

const (
	appName = "github-stale-bot"
	userID  = "stale-bot"
)

func main() {
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	if err := run(context.Background(), log, os.Args[1:]); err != nil {
		log.Error("fatal", "error", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, log *slog.Logger, args []string) error {
	return runWith(ctx, log, args, audit)
}

// runWith is run with the audit step injected, so a test can drive the real
// configuration load and the real budget wiring without a model or a GitHub
// transport.
//
// The seam exists because the join is what goes untested otherwise: the two
// halves can each be well covered while nothing checks that the budget is
// actually applied to the audit.
func runWith(ctx context.Context, log *slog.Logger, args []string, auditFn func(context.Context, *Config, *slog.Logger) error) error {
	// Best-effort: load a local .env when present, for local runs only. Skipped
	// under Actions, where every setting arrives through the workflow's env:
	// block — a .env reaching a job that holds issues: write could otherwise
	// override the thresholds and label names the workflow did not set.
	if os.Getenv("GITHUB_ACTIONS") == "" {
		_ = godotenv.Load()
	}

	cfg, err := loadConfig(args)
	if err != nil {
		return err
	}
	log.Info("starting stale-issue auditor",
		"repo", cfg.Owner+"/"+cfg.Repo, "model", cfg.Model,
		"concurrency", cfg.Concurrency, "dry_run", cfg.DryRun, "run_budget", cfg.RunBudget)
	if w := maintainersWarning(cfg); w != "" {
		log.Warn(w)
	}

	return withRunBudget(ctx, cfg, log, func(ctx context.Context) error {
		return auditFn(ctx, cfg, log)
	})
}

// withRunBudget bounds the whole run at cfg.RunBudget and turns an exhausted
// budget into a diagnosis.
//
// The workflow sets timeout-minutes above the budget, so an overrun on a large
// backlog is reported here — with the budget named and a non-zero exit — rather
// than the runner killing the job mid-write and leaving nothing behind that says
// why. Per-issue timeouts cannot do this: a sweep is len(issues)/Concurrency of
// them long.
//
// It takes fn so a test can drive the real budget rather than rebuilding the
// timeout, which would assert nothing about this function.
func withRunBudget(ctx context.Context, cfg *Config, log *slog.Logger, fn func(context.Context) error) error {
	ctx, cancel := context.WithTimeout(ctx, cfg.RunBudget)
	defer cancel()
	err := fn(ctx)
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		log.Error("run budget exhausted", "budget", cfg.RunBudget)
		return fmt.Errorf("run budget of %s exhausted before the sweep finished; raise RUN_BUDGET (keeping it below the job's timeout-minutes) or narrow the sweep: %w",
			cfg.RunBudget, errors.Join(context.DeadlineExceeded, err))
	}
	return err
}

// audit builds the agent and runs it over every candidate issue. The context it
// receives already carries the run budget.
func audit(ctx context.Context, cfg *Config, log *slog.Logger) error {
	gh, err := NewGitHubClient(ctx, cfg, log)
	if err != nil {
		return fmt.Errorf("github client: %w", err)
	}

	tools, err := gh.tools()
	if err != nil {
		return err
	}

	// If a Gemini API key is set it is used directly; otherwise the genai SDK
	// auto-detects its backend (e.g. Vertex AI via ADC) from the environment.
	clientConfig := &genai.ClientConfig{}
	if cfg.GeminiAPIKey != "" {
		clientConfig.APIKey = cfg.GeminiAPIKey
	}
	m, err := gemini.NewModel(ctx, cfg.Model, clientConfig)
	if err != nil {
		return fmt.Errorf("create model: %w", err)
	}

	issues, err := candidateIssues(ctx, gh, cfg)
	if err != nil {
		return err
	}
	if len(issues) == 0 {
		log.Info("no issues matched the criteria; nothing to do")
		return nil
	}
	log.Info("auditing issues", "count", len(issues))

	// Fail loud: surface both agent-run failures (returned by auditAll) and tool
	// infrastructure errors (handed back to the model as data, tracked on the
	// client) so a scheduled/CI run exits non-zero instead of silently reporting
	// success when nothing worked.
	auditErr := auditAll(ctx, cfg, log, issues, func(ictx context.Context, n int) error {
		r, ss, err := newAuditRunner(cfg, m, tools, log)
		if err != nil {
			return fmt.Errorf("issue #%d: %w", n, err)
		}
		return runAudit(ictx, r, ss, log, n)
	})
	if auditErr != nil {
		return fmt.Errorf("one or more audits failed: %w", auditErr)
	}
	if gh.hadToolError() {
		return errors.New("one or more tool calls failed; see logs above")
	}
	return nil
}

// newAuditRunner builds the agent, session service and runner for ONE issue
// audit.
//
// They must not be shared across the fan-out. runner.Run writes the root
// agent's mode on its first call, so two concurrent audits sharing one agent
// race on that field — reproduced under -race by driving two audits through
// this exact errgroup, and live on any sweep with two candidates, since
// CONCURRENCY_LIMIT defaults to 3.
//
// The model is shared on purpose: it holds the HTTP client, and its three
// fields are set at construction and never written again.
func newAuditRunner(cfg *Config, m model.LLM, tools []tool.Tool, log *slog.Logger) (*runner.Runner, session.Service, error) {
	auditor, err := llmagent.New(llmagent.Config{
		Name:        "stale_issue_auditor",
		Model:       m,
		Description: "Audits open GitHub issues for staleness.",
		Instruction: renderPrompt(cfg),
		Tools:       tools,
		// Temperature 0 keeps the classification deterministic across runs.
		GenerateContentConfig: &genai.GenerateContentConfig{Temperature: genai.Ptr[float32](0)},
		// Observe-only: log a tool failure but don't replace the result, so the
		// model still sees the error and can react. The tool itself records the
		// failure (hadToolError) so the run can also exit non-zero.
		OnToolErrorCallbacks: []llmagent.OnToolErrorCallback{
			func(_ agent.Context, t tool.Tool, args map[string]any, err error) (map[string]any, error) {
				logToolError(log, t.Name(), args, err)
				return nil, nil
			},
		},
	})
	if err != nil {
		return nil, nil, fmt.Errorf("create agent: %w", err)
	}

	sessionService := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          auditor,
		SessionService: sessionService,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("create runner: %w", err)
	}
	return r, sessionService, nil
}

// logToolError reports a failed tool call with the argument NAMES only.
//
// The values are chosen by the model and the line lands in a public Actions log,
// which the runner parses for :: workflow commands. The names are what makes the
// line useful for debugging; the values are not worth the question.
func logToolError(log *slog.Logger, name string, args map[string]any, err error) {
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	log.Error("tool call failed", "tool", name, "args", strings.Join(keys, ","), "error", err)
}

// maintainersWarning returns a warning when no maintainers are configured. With
// an empty set, no comment can be classified as a maintainer action, so the bot
// will never mark anything stale (it can still un-stale and alert).
func maintainersWarning(cfg *Config) string {
	if len(cfg.Maintainers) == 0 {
		return "MAINTAINERS is empty: no comment will be treated as maintainer activity, so issues will never be marked stale"
	}
	return ""
}

// candidateIssues returns the issue numbers to audit: either the single issue
// requested via -issue, or all stale candidates from the search.
func candidateIssues(ctx context.Context, gh *GitHubClient, cfg *Config) ([]int, error) {
	if cfg.SingleIssue != 0 {
		return []int{cfg.SingleIssue}, nil
	}
	return gh.SearchOldOpenIssues(ctx)
}

// auditAll audits the issues with bounded concurrency. A failure on one issue
// does not abort the batch (a plain errgroup does not cancel on error), but it
// is returned so the whole run can exit non-zero.
//
// It takes runFn so a test can drive the real fan-out — each issue is audited,
// each in a session scoped to its own number — without a model or a GitHub
// transport.
func auditAll(ctx context.Context, cfg *Config, log *slog.Logger, issues []int, runFn func(context.Context, int) error) error {
	g := new(errgroup.Group)
	g.SetLimit(cfg.Concurrency)
	for _, n := range issues {
		g.Go(func() error {
			return auditIssue(ctx, cfg, n, func(ictx context.Context) error {
				return runFn(ictx, n)
			})
		})
	}
	err := g.Wait()
	log.Info("audit finished", "processed", len(issues))
	return err
}

// auditIssue scopes a session to one issue and hands it to runFn.
//
// It takes runFn so a test can drive the real function and assert the session is
// actually scoped. Reconstructing the scoping in a test would assert nothing
// about this function, and this one line is the whole cross-issue defence.
func auditIssue(ctx context.Context, cfg *Config, number int, runFn func(context.Context) error) error {
	ictx, cancel := context.WithTimeout(ctx, cfg.IssueTimeout)
	defer cancel()
	// Scope this session to the audited issue so injected instructions in the
	// issue's (untrusted) content cannot make a tool mutate a different issue.
	ictx = withAuditedIssue(ictx, number)
	return runFn(ictx)
}

// runAudit runs the agent against one already-scoped issue and logs its decision.
func runAudit(ictx context.Context, r *runner.Runner, ss session.Service, log *slog.Logger, number int) error {
	start := time.Now()
	l := log.With("issue", number)

	resp, err := ss.Create(ictx, &session.CreateRequest{AppName: appName, UserID: userID})
	if err != nil {
		l.Error("create session", "error", err)
		return fmt.Errorf("issue #%d: create session: %w", number, err)
	}

	// The issue number reaches the tools *through the model*: this message names
	// the issue, the prompt instructs the model to call get_issue_state, and the
	// model copies the number into each tool's issue_number argument. There is no
	// direct Go call path from here into the tools.
	msg := genai.NewContentFromText(fmt.Sprintf("Audit Issue #%d.", number), genai.RoleUser)

	// r.Run streams every event the agent produces: tool calls, tool results,
	// and model messages. We only want the agent's final natural-language
	// decision, so we keep the text of the last content-bearing event.
	// StreamingModeNone is used because this is a headless batch run with no UI
	// to update token-by-token (cf. StreamingModeSSE in the interactive examples).
	var (
		decision string
		runErr   error
	)
	for event, err := range r.Run(ictx, userID, resp.Session.ID(), msg, agent.RunConfig{StreamingMode: agent.StreamingModeNone}) {
		if err != nil {
			l.Error("agent run", "error", err)
			runErr = errors.Join(runErr, err)
			continue
		}
		if event.ErrorCode != "" {
			l.Error("model error", "code", event.ErrorCode, "message", event.ErrorMessage)
			runErr = errors.Join(runErr, fmt.Errorf("model error %s: %s", event.ErrorCode, event.ErrorMessage))
			continue
		}
		if event.Content == nil {
			continue
		}
		var b strings.Builder
		for _, p := range event.Content.Parts {
			b.WriteString(p.Text)
		}
		if text := b.String(); text != "" {
			decision = text
		}
	}

	l.Info("audited", "duration", time.Since(start).Round(time.Millisecond), "decision", summarize(decision))
	if runErr != nil {
		return fmt.Errorf("issue #%d: %w", number, runErr)
	}
	return nil
}

// summarize collapses the agent's final text into a single short log line.
//
// The cut is by rune, not by byte: the agent's text can contain any UTF-8, and
// slicing bytes splits a multibyte rune straddling the limit, emitting an
// invalid-UTF-8 log line. This matches the sibling spam-detection bot.
func summarize(s string) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	const maxRunes = 200
	if r := []rune(s); len(r) > maxRunes {
		return string(r[:maxRunes]) + "..."
	}
	return s
}
