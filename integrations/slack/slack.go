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

package slack

import (
	"cmp"
	"context"
	"errors"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"google.golang.org/adk/integrations/slack/internal/slackapi"
	"google.golang.org/adk/integrations/slack/mrkdwn"
)

// Config configures a Service. New requires BotToken and Backend. Handler
// needs SigningSecret to serve: without it New logs a warning and Handler
// answers 500 to every request. Every other zero value selects a default.
type Config struct {
	// BotToken is the bot user OAuth token ("xoxb-...").
	BotToken string
	// SigningSecret authenticates Events API requests. Handler answers 500
	// to every request while it is empty.
	SigningSecret string
	// Backend runs the agent turns.
	Backend Backend
	// Policy is consulted after the built-in rules, for messages they allow.
	Policy Policy

	// Channels limits the service to these channel IDs. Empty means all.
	Channels []string
	// AllowUnmentioned answers channel messages that do not mention the bot.
	AllowUnmentioned bool
	// RequireMentionInThreads stops answering unmentioned replies in threads
	// whose root message the bot posted. Replies in a thread a user started
	// need a mention either way.
	RequireMentionInThreads bool
	// Disabled refuses every message, as a kill switch.
	Disabled bool

	// NoPlaceholder skips the placeholder message posted when a turn starts.
	NoPlaceholder bool
	// Placeholder is the text shown while a turn runs. Zero means "_Thinking…_".
	Placeholder string
	// ErrorMessage is posted when a turn fails. Zero means "Sorry, something
	// went wrong on my side. Please try again.".
	ErrorMessage string
	// DisableMarkdownConversion posts model output as is instead of
	// converting Markdown to Slack mrkdwn.
	DisableMarkdownConversion bool

	// MaxConcurrentRuns is the number of turns run at once. Zero means 10,
	// and New rejects a negative value.
	MaxConcurrentRuns int
	// MaxQueued is the number of accepted events waiting for a free run.
	// Events beyond it are dropped. Zero means 100, and New rejects a
	// negative value.
	MaxQueued int
	// TurnTimeout bounds the handling of one message: learning the bot's
	// user ID, the built-in rules, the Policy and the agent turn. The wait
	// for the thread's lock is outside it. Zero means five minutes.
	TurnTimeout time.Duration
	// Deduper drops redelivered events. Zero means an in-memory LRU of
	// 10 000 event IDs kept for ten minutes, which does not span replicas.
	Deduper Deduper
	// UserIDFunc maps a message to the ADK user ID. Zero means
	// "slack:" + TeamID + ":" + Channel, so a channel shares one user.
	UserIDFunc func(*Message) string
	// HTTPClient calls the Slack Web API. Zero means a client with a 10 s timeout.
	HTTPClient *http.Client
	// Logger receives the service's diagnostics. Nil discards them. The
	// service never logs message text or tokens itself. Errors returned by
	// user code (Backend, Policy, Deduper) are logged with their text, and
	// panics in user code by type only.
	Logger *slog.Logger

	// Test hooks.
	now        func() time.Time
	apiBaseURL string
}

// Service answers Slack messages with agent turns.
type Service struct {
	secret string
	logger *slog.Logger
	now    func() time.Time
	bot    *botUser
	disp   *dispatcher
}

// New validates cfg and starts the workers that run turns. Close stops them.
func New(ctx context.Context, cfg Config) (*Service, error) {
	if cfg.BotToken == "" {
		return nil, errors.New("slack: Config.BotToken is required")
	}
	if cfg.Backend == nil {
		return nil, errors.New("slack: Config.Backend is required")
	}
	if cfg.MaxConcurrentRuns < 0 {
		return nil, errors.New("slack: Config.MaxConcurrentRuns must not be negative")
	}
	if cfg.MaxQueued < 0 {
		return nil, errors.New("slack: Config.MaxQueued must not be negative")
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	if cfg.SigningSecret == "" {
		logger.WarnContext(ctx, "slack: signing secret not configured")
	}
	now := cfg.now
	if now == nil {
		now = time.Now
	}
	var opts []slackapi.Option
	if cfg.apiBaseURL != "" {
		opts = append(opts, slackapi.WithBaseURL(cfg.apiBaseURL))
	}
	if cfg.HTTPClient != nil {
		opts = append(opts, slackapi.WithHTTPClient(cfg.HTTPClient))
	}
	api := slackapi.New(cfg.BotToken, opts...)
	bot := &botUser{api: api, logger: logger, now: now}

	placeholder := cfg.Placeholder
	if placeholder == "" {
		placeholder = "_Thinking…_"
	}
	if cfg.NoPlaceholder {
		placeholder = ""
	}
	render := mrkdwn.Render
	if cfg.DisableMarkdownConversion {
		render = func(s string) string { return s }
	}
	channels := make(map[string]bool, len(cfg.Channels))
	for _, c := range cfg.Channels {
		channels[c] = true
	}
	dc := dispatcherConfig{
		maxRuns:     cmp.Or(cfg.MaxConcurrentRuns, 10),
		maxQueued:   cmp.Or(cfg.MaxQueued, 100),
		turnTimeout: cmp.Or(cfg.TurnTimeout, 5*time.Minute),
		gate: gateConfig{
			disabled:                cfg.Disabled,
			channels:                channels,
			allowUnmentioned:        cfg.AllowUnmentioned,
			requireMentionInThreads: cfg.RequireMentionInThreads,
		},
		policy:          cfg.Policy,
		dedup:           cfg.Deduper,
		backend:         cfg.Backend,
		api:             api,
		render:          render,
		placeholder:     placeholder,
		errorText:       cmp.Or(cfg.ErrorMessage, "Sorry, something went wrong on my side. Please try again."),
		userID:          cfg.UserIDFunc,
		botUserID:       bot.get,
		cachedBotUserID: bot.cached,
		logger:          logger,
	}
	if dc.dedup == nil {
		dc.dedup = newLRUDeduper(10_000, 10*time.Minute)
	}
	if dc.userID == nil {
		dc.userID = func(m *Message) string { return "slack:" + m.TeamID + ":" + m.Channel }
	}
	return &Service{
		secret: cfg.SigningSecret,
		logger: logger,
		now:    now,
		bot:    bot,
		disp:   newDispatcher(ctx, dc),
	}, nil
}

// Handler returns the Events API endpoint. Point the app's Request URL at it.
func (s *Service) Handler() http.Handler { return http.HandlerFunc(s.serveEvents) }

// Close stops accepting events and waits for running turns until ctx is
// done. It then cancels the turns still running, waits up to two seconds for
// them to report the failure in their threads, and returns ctx's error.
// Turns still queued are dropped. Close may be called more than once.
func (s *Service) Close(ctx context.Context) error { return s.disp.close(ctx) }

// authTestBackoff spaces auth.test calls after a failure, so a Slack outage
// does not cost a call per event.
const authTestBackoff = 30 * time.Second

// botUser holds the bot's own user ID, learned from event authorizations or,
// failing that, from auth.test.
type botUser struct {
	api    *slackapi.Client
	logger *slog.Logger
	now    func() time.Time

	// fetch serializes auth.test calls. The handler never takes it.
	fetch sync.Mutex

	mu      sync.Mutex
	id      string
	retryAt time.Time
}

// record keeps the ID of the first bot authorization. Slack sends it with
// the event, so it replaces an ID learned earlier.
func (b *botUser) record(auths []authorization) {
	for _, a := range auths {
		if a.IsBot && a.UserID != "" {
			b.mu.Lock()
			b.id = a.UserID
			b.mu.Unlock()
			return
		}
	}
}

// cached returns the ID without calling Slack. It is empty until learned.
func (b *botUser) cached() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.id
}

// get returns the ID, calling auth.test when no event has carried it and no
// call failed within authTestBackoff.
func (b *botUser) get(ctx context.Context) string {
	b.fetch.Lock()
	defer b.fetch.Unlock()
	b.mu.Lock()
	id, retryAt := b.id, b.retryAt
	b.mu.Unlock()
	if id != "" || b.now().Before(retryAt) {
		return id
	}
	id, err := b.api.AuthTest(ctx)
	b.mu.Lock()
	defer b.mu.Unlock()
	if err != nil {
		b.logger.WarnContext(ctx, "slack: auth.test failed", "error", err)
		b.retryAt = b.now().Add(authTestBackoff)
		return b.id
	}
	b.id = id
	return id
}
