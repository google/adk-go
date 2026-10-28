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
	"context"
	"errors"
	"fmt"
	"log/slog"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/integrations/slack/internal/slackapi"
)

var tracer = otel.Tracer("google.golang.org/adk/integrations/slack")

var (
	errQueueFull = errors.New("slack: turn queue full")
	errStopped   = errors.New("slack: service closed")
)

type dispatcherConfig struct {
	maxRuns, maxQueued     int
	turnTimeout            time.Duration
	gate                   gateConfig
	policy                 Policy
	dedup                  Deduper
	backend                Backend
	api                    *slackapi.Client
	render                 func(string) string
	placeholder, errorText string
	userID                 func(*Message) string
	botUserID              func(context.Context) string
	cachedBotUserID        func() string
	logger                 *slog.Logger
}

// dispatcher runs turns on a fixed pool of workers fed by a bounded queue.
type dispatcher struct {
	cfg      dispatcherConfig
	queue    chan *turn
	wg       sync.WaitGroup
	locks    keyedLocks
	stopCh   chan struct{}
	stopOnce sync.Once
	// stopCtx is canceled by close once its grace period is over, which
	// cancels every running turn.
	stopCtx   context.Context
	stopTurns context.CancelFunc
}

type turn struct {
	ctx context.Context
	msg *Message
}

func newDispatcher(ctx context.Context, cfg dispatcherConfig) *dispatcher {
	d := &dispatcher{
		cfg:    cfg,
		queue:  make(chan *turn, cfg.maxQueued),
		stopCh: make(chan struct{}),
	}
	// The turns belong to the service, so only close ends them, not New's caller.
	d.stopCtx, d.stopTurns = context.WithCancel(context.WithoutCancel(ctx))
	for range cfg.maxRuns {
		d.wg.Go(d.work)
	}
	return d
}

// Submit runs the cheap synchronous checks and enqueues. It never blocks on
// the Slack API or the queue, so the HTTP handler can answer Slack in time.
// It does call the Deduper synchronously, so a Deduper that does network I/O
// delays the HTTP response.
func (d *dispatcher) Submit(ctx context.Context, m *Message) error {
	select {
	case <-d.stopCh:
		return errStopped
	default:
	}
	if m.FromBot {
		d.refused(ctx, m, ReasonFromBot)
		return nil
	}
	// Refusals that need no bot ID happen here, so an unmentioned top-level
	// channel message never takes a queue slot unless AllowUnmentioned is set.
	if reason := d.refuseEarly(m); reason != "" {
		d.refused(ctx, m, reason)
		return nil
	}
	seen, err := d.cfg.dedup.Seen(ctx, m.EventID)
	if err != nil {
		// Answering twice is better than not answering.
		d.cfg.logger.WarnContext(ctx, "slack: dedup failed", "event_id", m.EventID, "error", err)
		seen = false
	}
	if seen {
		d.drop(ctx, m, ReasonDuplicate)
		return nil
	}
	select {
	case d.queue <- &turn{ctx: ctx, msg: m}:
		return nil
	default:
		d.drop(ctx, m, ReasonQueueFull)
		return errQueueFull
	}
}

// refuseEarly returns why m is refused before it is queued, or "". A
// top-level channel message is either the plain copy of an app_mention or
// unmentioned, so it is never answered without AllowUnmentioned. Thread
// replies are left to the worker, which needs the bot's ID to judge them.
func (d *dispatcher) refuseEarly(m *Message) string {
	g := d.cfg.gate
	switch {
	case g.disabled:
		return ReasonKillSwitch
	case len(g.channels) > 0 && !g.channels[m.Channel]:
		return ReasonChannel
	case m.kind == "message" && m.ThreadTS == "" && m.ChannelType != "im" && !g.allowUnmentioned:
		if bot := d.cfg.cachedBotUserID(); bot != "" && strings.Contains(m.Text, "<@"+bot+">") {
			return ReasonMentionEcho
		}
		return ReasonNoMention
	}
	return ""
}

func (d *dispatcher) drop(ctx context.Context, m *Message, reason string) {
	d.cfg.logger.WarnContext(ctx, "slack: event dropped", "reason", reason, "channel", m.Channel, "event_id", m.EventID)
}

func (d *dispatcher) work() {
	for {
		select {
		case <-d.stopCh:
			return
		case t := <-d.queue:
			d.process(t)
		}
	}
}

// stopped reports whether close has begun, and logs the turn it drops.
func (d *dispatcher) stopped(t *turn) bool {
	select {
	case <-d.stopCh:
		d.drop(t.ctx, t.msg, ReasonShutdown)
		return true
	default:
		return false
	}
}

// turnContext is the context of a turn's gate or of its run. It outlives
// the HTTP request that delivered the turn, keeping only the request's
// values, such as the trace. It ends after budget, or when close stops
// waiting.
func (d *dispatcher) turnContext(t *turn, budget time.Duration) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(t.ctx), budget)
	stop := context.AfterFunc(d.stopCtx, cancel)
	return ctx, func() {
		stop()
		cancel()
	}
}

// turnState carries a turn through process, so that its deferred cleanup
// can finish the turn whether it returned or panicked.
type turnState struct {
	t         *turn
	sessionID string
	unlock    func()
	// The fields below are set once the turn runs.
	ctx      context.Context
	cancel   context.CancelFunc
	span     trace.Span
	dl       *deliverer
	began    time.Time
	err      error
	panicked bool
	// stage names the step running, for a panic record.
	stage string
}

// process takes one dequeued turn through the gate, the Policy and the run.
// Its recover covers everything before end: the bot-ID lookup, the gate, the
// Policy, UserIDFunc, the Backend and the replies posted during the run. end
// runs after recover has returned, so a panic in end is not caught, for
// example one in the final delivery, in an error's Error method called while
// logging, or in the Logger.
func (d *dispatcher) process(t *turn) {
	st := &turnState{t: t}
	defer func() {
		if r := recover(); r != nil {
			d.recovered(st, r)
		}
		d.end(st)
	}()
	if d.stopped(t) {
		return
	}
	// The gate and the run share one TurnTimeout budget. The wait for the
	// thread's lock is not charged to it.
	began := time.Now()
	gctx, gcancel := d.turnContext(t, d.cfg.turnTimeout)
	defer gcancel()
	bot, ok := d.allowed(gctx, st)
	if !ok {
		return
	}
	left := d.cfg.turnTimeout - time.Since(began)
	st.sessionID = SessionID(t.msg.Channel, t.msg.threadKey())
	d.locks.lock(st.sessionID)
	st.unlock = func() { d.locks.unlock(st.sessionID) }
	// Close may have begun while this worker waited for the thread's lock.
	if d.stopped(t) {
		return
	}
	st.ctx, st.cancel = d.turnContext(t, left)
	d.runTurn(st, bot)
}

// allowed resolves the bot's user ID, drops the plain-message copy of a
// mention, then applies the built-in rules and the user's Policy. It returns
// the bot's user ID, which is empty when it is not known.
func (d *dispatcher) allowed(ctx context.Context, st *turnState) (string, bool) {
	m := st.t.msg
	st.stage = "bot_id"
	bot := d.cfg.botUserID(ctx)
	st.stage = "gate"
	if m.kind == "message" {
		// Without the ID a plain message cannot be a mention, which refuses
		// rather than answers.
		m.IsMention = bot != "" && strings.Contains(m.Text, "<@"+bot+">")
		// Slack sends a channel mention twice, as app_mention and as message.
		if m.IsMention && m.ChannelType != "im" {
			d.refused(ctx, m, ReasonMentionEcho)
			return "", false
		}
	}
	gate := d.cfg.gate
	gate.botUserID = func() string { return bot }
	dec := builtinGate(gate, m)
	if dec.Allow && d.cfg.policy != nil {
		st.stage = "policy"
		var err error
		if dec, err = d.cfg.policy(ctx, m); err != nil {
			// WARN, not DEBUG: the Policy is broken, and nobody else sees it.
			d.cfg.logger.WarnContext(ctx, "slack: message not answered", "reason", ReasonPolicyError, "channel", m.Channel, "event_id", m.EventID, "error", err)
			return "", false
		}
		if !dec.Allow && dec.Reason == "" {
			dec.Reason = ReasonPolicyRefused
		}
	}
	if !dec.Allow {
		d.refused(ctx, m, dec.Reason)
	}
	return bot, dec.Allow
}

func (d *dispatcher) refused(ctx context.Context, m *Message, reason string) {
	d.cfg.logger.DebugContext(ctx, "slack: message not answered", "reason", reason, "channel", m.Channel, "event_id", m.EventID)
}

// finishTimeout bounds the final delivery, which runs after the turn's own
// context may already be done.
const finishTimeout = 5 * time.Second

func (d *dispatcher) runTurn(st *turnState, bot string) {
	m := st.t.msg
	st.stage = "turn"
	st.ctx, st.span = tracer.Start(st.ctx, "slack.turn", trace.WithAttributes(
		attribute.String("slack.channel", m.Channel),
		attribute.String("slack.thread_ts", m.threadKey()),
		attribute.String("slack.event_id", m.EventID),
		attribute.String("slack.transport", "events"),
	))
	st.dl = &deliverer{
		api:         d.cfg.api,
		channel:     m.Channel,
		threadTS:    m.threadKey(),
		placeholder: d.cfg.placeholder,
		errorText:   d.cfg.errorText,
		render:      d.cfg.render,
		logger:      d.cfg.logger,
	}
	st.dl.start(st.ctx)
	st.began = time.Now()
	text := stripOwnMention(m.Text, bot)
	if text == "" {
		// An empty user part would be stored in the session and, with
		// llmagent's default IncludeContents, resent on every later turn of
		// the thread, so a model that rejects empty parts would fail them
		// all. Sending the mention token keeps the part non-empty.
		text = m.Text
	}
	content := genai.NewContentFromText(text, genai.RoleUser)
	for ev, err := range d.cfg.backend.Run(st.ctx, d.cfg.userID(m), st.sessionID, content) {
		if err != nil {
			st.err = err
			break
		}
		if ev == nil || ev.Content == nil || ev.Partial {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p != nil && strings.TrimSpace(p.Text) != "" && !p.Thought {
				st.dl.text(st.ctx, p.Text)
			}
		}
	}
	// A Backend may stop at the deadline without yielding ctx.Err().
	if st.err == nil && st.ctx.Err() != nil {
		st.err = st.ctx.Err()
	}
}

// recovered logs a panic by its type only: the value may carry user text.
// A panic before the turn ran refuses the message without a reply.
func (d *dispatcher) recovered(st *turnState, r any) {
	m := st.t.msg
	kind, stack := fmt.Sprintf("%T", r), string(debug.Stack())
	if st.dl == nil {
		d.cfg.logger.ErrorContext(st.t.ctx, "slack: turn panicked", "stage", st.stage, "event_id", m.EventID, "panic", kind, "stack", stack)
		return
	}
	st.err = errors.New("slack: panic in turn")
	st.panicked = true
	d.cfg.logger.ErrorContext(st.ctx, "slack: turn panicked", "stage", st.stage, "event_id", m.EventID, "session_id", st.sessionID, "panic", kind, "stack", stack)
}

// end reports the turn's outcome in its thread and releases what process took.
func (d *dispatcher) end(st *turnState) {
	if st.dl != nil {
		// A timed-out or canceled turn must still report the failure in its thread.
		fctx, fcancel := context.WithTimeout(context.WithoutCancel(st.ctx), finishTimeout)
		st.dl.finish(fctx, st.err)
		fcancel()
		if st.err != nil {
			st.span.RecordError(st.err)
			st.span.SetStatus(codes.Error, "turn failed")
			if !st.panicked {
				d.cfg.logger.ErrorContext(st.ctx, "slack: turn failed", "event_id", st.t.msg.EventID, "session_id", st.sessionID,
					"backend", fmt.Sprintf("%T", d.cfg.backend), "duration", time.Since(st.began), "error", st.err)
			}
		}
	}
	if st.span != nil {
		st.span.End()
	}
	if st.cancel != nil {
		st.cancel()
	}
	if st.unlock != nil {
		st.unlock()
	}
}

// stripOwnMention removes the bot's mention tokens, which address the bot
// rather than say anything to it.
func stripOwnMention(text, botUserID string) string {
	if botUserID != "" {
		text = strings.ReplaceAll(text, "<@"+botUserID+">", "")
	}
	return strings.TrimSpace(text)
}

// closeGrace bounds the wait for canceled turns to post their error message.
const closeGrace = 2 * time.Second

func (d *dispatcher) close(ctx context.Context) error {
	d.stopOnce.Do(func() { close(d.stopCh) })
	done := make(chan struct{})
	go func() {
		d.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
	}
	d.stopTurns()
	select {
	case <-done:
	case <-time.After(closeGrace):
		return fmt.Errorf("slack: turns still running %s after cancellation: %w", closeGrace, ctx.Err())
	}
	return ctx.Err()
}

// keyedLocks serializes turns of one session, so a thread's turns see each
// other's history. Entries are deleted when unused, so the map stays as
// small as the number of active sessions.
type keyedLocks struct {
	mu    sync.Mutex
	locks map[string]*keyedLock
}

type keyedLock struct {
	mu   sync.Mutex
	refs int
}

func (k *keyedLocks) lock(key string) {
	k.mu.Lock()
	if k.locks == nil {
		k.locks = make(map[string]*keyedLock)
	}
	l, ok := k.locks[key]
	if !ok {
		l = &keyedLock{}
		k.locks[key] = l
	}
	l.refs++
	k.mu.Unlock()
	l.mu.Lock()
}

func (k *keyedLocks) unlock(key string) {
	k.mu.Lock()
	l := k.locks[key]
	l.refs--
	if l.refs == 0 {
		delete(k.locks, key)
	}
	k.mu.Unlock()
	l.mu.Unlock()
}
