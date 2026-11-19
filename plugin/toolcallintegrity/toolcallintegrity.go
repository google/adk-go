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

// Package toolcallintegrity provides a plugin that detects tampering with the
// function calls stored in a session. It stamps each function call with an
// HMAC when the event is emitted, re-verifies every stamp before each run, and
// lets a tool run only when its call has a valid stamp. Without the secret key,
// a modified call cannot be given a valid stamp.
package toolcallintegrity

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/adkcontext"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// Config configures the plugin that New returns.
type Config struct {
	// SecretKey signs the stamp of every new function call, and verifies
	// stored stamps. Required: it must be non-empty.
	SecretKey []byte

	// VerificationKeys are further keys that verify stored stamps but never
	// sign, for key rotation: put the new key in SecretKey and the old one
	// here until every stored call has been stamped with the new key. Each
	// must be non-empty. Nil means SecretKey alone.
	VerificationKeys [][]byte

	// AllowUnstampedCalls logs a warning instead of failing for a function
	// call that has an ID but no stamp, such as a call stored before the
	// plugin was installed. Such calls also run, so this reports tampering
	// that removes a stamp but does not prevent it. A stamp that is present
	// but wrong is always rejected.
	AllowUnstampedCalls bool

	// Name is the plugin name. Empty means "tool_call_integrity".
	Name string

	// Logger receives the warning AllowUnstampedCalls logs for each unstamped
	// call, at slog.LevelWarn, with the attributes "tool" and
	// "function_call_id". Nil discards the warning.
	Logger *slog.Logger
}

// ErrIntegrity matches, under errors.Is, every error the plugin's callbacks
// return when a function call fails its integrity check. The error's message
// is the check's own sentence, for example
// "transfer_money call 'fc-1' does not match its integrity stamp", without
// the text of ErrIntegrity.
var ErrIntegrity = errors.New("tool call integrity check failed")

// New returns the tool call integrity plugin.
//
// Its OnEventCallback stamps every function call that has an ID in each event
// the runner is about to store, under an ADK-internal custom metadata key, and
// fails for a non-partial event that repeats a call ID. Its BeforeRunCallback
// re-verifies every function call with an ID in the session, answered or not,
// and fails when a stamp is wrong, malformed, repeated, missing or left over.
// Its BeforeToolCallback lets a tool run only when every stored copy of its
// call verifies and the tool's name and arguments equal one of them; it
// always refuses a call whose only stored copies are in restored events (see
// PrepareRestoredEvent), and a tool context whose FunctionCallID is empty
// (ADK's own tool contexts always report an ID; see agent.NewToolContext).
// It reads the session's events and never its state, so it does not wait
// for the earlier calls of the model turn. Every check failure matches
// ErrIntegrity under errors.Is.
//
// Register the plugin before other plugins. The runner stops at the first
// OnEventCallback that returns an event, and a later plugin must not change
// function calls or replace CustomMetadata. Changing a call's arguments in
// place after the event was emitted changes the stored call, which then no
// longer verifies.
//
// New returns an error, and a nil plugin, when SecretKey is empty or a
// VerificationKeys entry is empty. It copies the keys, so later changes to the
// caller's slices have no effect. The plugin holds no mutable state and is
// safe for concurrent use by any number of runs.
func New(cfg Config) (*plugin.Plugin, error) {
	if len(cfg.SecretKey) == 0 {
		return nil, errBadKeys
	}
	keys := make([][]byte, 0, 1+len(cfg.VerificationKeys))
	keys = append(keys, bytes.Clone(cfg.SecretKey))
	for _, k := range cfg.VerificationKeys {
		if len(k) == 0 {
			return nil, errBadKeys
		}
		keys = append(keys, bytes.Clone(k))
	}
	name := cfg.Name
	if name == "" {
		name = defaultName
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	p := &integrity{keys: keys, allowUnstamped: cfg.AllowUnstampedCalls, logger: logger}
	return plugin.New(plugin.Config{
		Name:               name,
		OnEventCallback:    p.onEvent,
		BeforeRunCallback:  p.beforeRun,
		BeforeToolCallback: p.beforeTool,
	})
}

const defaultName = "tool_call_integrity"

// errBadKeys is New's error for a key set it cannot use. It does not match
// ErrIntegrity: adk-python raises ValueError here, not ToolCallIntegrityError.
var errBadKeys = errors.New("toolcallintegrity: each key must be non-empty bytes, and SecretKey must be set")

// integrity holds a plugin's configuration. It is never modified after New,
// so one value serves concurrent runs.
type integrity struct {
	keys           [][]byte // keys[0] signs; all verify. Copied by New.
	allowUnstamped bool
	logger         *slog.Logger // never nil: cfg.Logger, or a discarding logger
}

// onEvent stamps every function call with an ID in ev, in place, under
// stampKey, and refuses a non-partial event that repeats a call ID before
// anything is stamped (adk-python: on_event_callback). A partial event is
// neither stored nor executed, so a repeat there is stamped, the last copy
// winning.
func (p *integrity) onEvent(ic agent.InvocationContext, ev *session.Event) (*session.Event, error) {
	var calls []*genai.FunctionCall
	for _, fc := range ev.FunctionCalls() {
		if fc.ID != "" {
			calls = append(calls, fc)
		}
	}
	if len(calls) == 0 {
		return nil, nil
	}
	if !ev.Partial {
		seen := make(map[string]bool, len(calls))
		for _, fc := range calls {
			if seen[fc.ID] {
				return nil, integrityErrorf("event '%s' repeats function call ID '%s'", ev.ID, fc.ID)
			}
			seen[fc.ID] = true
		}
	}
	sess := ic.Session()
	// The stamp map is a map[string]any of strings, the form a serializing
	// store returns, so stored and in-memory sessions read alike.
	stamps := make(map[string]any, len(calls))
	for _, fc := range calls {
		payload, err := stampPayload(sess, ev, fc)
		if err != nil {
			return nil, err
		}
		stamps[fc.ID] = p.sign(payload)
	}
	// A new map, so a map the event shared with its creator is not written.
	metadata := make(map[string]any, len(ev.CustomMetadata)+1)
	maps.Copy(metadata, ev.CustomMetadata)
	metadata[stampKey] = stamps
	ev.CustomMetadata = metadata
	return nil, nil
}

// beforeRun re-verifies every function call with an ID in the session's
// events, skipping events without calls and restored events, whose calls
// beforeTool refuses instead (adk-python: before_run_callback). Stamps are
// per call, so the order of the events does not matter.
func (p *integrity) beforeRun(ic agent.InvocationContext) (*genai.Content, error) {
	sess := ic.Session()
	for ev := range sess.Events().All() {
		calls := ev.FunctionCalls()
		if len(calls) == 0 || isRestored(ev) {
			continue
		}
		stamps, err := stampsOf(ev)
		if err != nil {
			return nil, err
		}
		verified := make(map[string]bool, len(calls))
		for _, fc := range calls {
			// A call without an ID cannot be stamped, and beforeTool never
			// lets one run.
			if fc.ID == "" {
				continue
			}
			stamp, ok := stamps[fc.ID]
			if !ok {
				if err := p.unstamped(ic, fc.Name, fc.ID); err != nil {
					return nil, err
				}
				continue
			}
			if verified[fc.ID] {
				return nil, integrityErrorf("event '%s' has duplicate function call ID '%s'", ev.ID, fc.ID)
			}
			payload, err := stampPayload(sess, ev, fc)
			if err != nil {
				return nil, err
			}
			if !p.verify(payload, stamp) {
				return nil, integrityErrorf("%s call '%s' does not match its integrity stamp", fc.Name, fc.ID)
			}
			verified[fc.ID] = true
		}
		// A stamp with no matching call means the call was renamed, re-keyed
		// or removed.
		var leftover []string
		for id := range stamps {
			if !verified[id] {
				leftover = append(leftover, id)
			}
		}
		if len(leftover) > 0 {
			slices.Sort(leftover)
			return nil, integrityErrorf("event '%s' has integrity stamps with no matching call: ['%s']", ev.ID, strings.Join(leftover, "', '"))
		}
	}
	return nil, nil
}

// beforeTool lets the tool run only when every non-restored stored copy of
// its call verifies, and the tool's name and args equal one of them
// (adk-python: before_tool_callback). It reads args and never writes them.
func (p *integrity) beforeTool(ctx agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
	id := ctx.FunctionCallID()
	if id == "" {
		return nil, integrityErrorf("%s call has no ID, so its integrity cannot be checked", t.Name())
	}
	sess := invocationSession(ctx)
	if sess == nil {
		return nil, integrityErrorf("%s call '%s' cannot be checked: the tool context has no session", t.Name(), id)
	}
	type storedCall struct {
		ev *session.Event
		fc *genai.FunctionCall
	}
	var stored []storedCall
	restored := false
	for ev := range sess.Events().All() {
		for _, fc := range ev.FunctionCalls() {
			if fc.ID != id {
				continue
			}
			if isRestored(ev) {
				restored = true
				continue
			}
			stored = append(stored, storedCall{ev: ev, fc: fc})
		}
	}
	if len(stored) == 0 {
		if restored {
			return nil, integrityErrorf("%s call '%s' comes from restored history and cannot be executed", t.Name(), id)
		}
		return nil, p.unstamped(ctx, t.Name(), id)
	}
	// Every stored copy must verify, so a repeated ID cannot borrow the stamp
	// of another copy. An unstamped copy is trusted only when unstamped calls
	// are allowed.
	trusted := make([]*genai.FunctionCall, 0, len(stored))
	for _, c := range stored {
		stamps, err := stampsOf(c.ev)
		if err != nil {
			return nil, err
		}
		if stamp, ok := stamps[id]; !ok {
			if err := p.unstamped(ctx, t.Name(), id); err != nil {
				return nil, err
			}
		} else {
			payload, err := stampPayload(sess, c.ev, c.fc)
			if err != nil {
				return nil, err
			}
			if !p.verify(payload, stamp) {
				return nil, integrityErrorf("%s call '%s' does not match its integrity stamp", t.Name(), id)
			}
		}
		trusted = append(trusted, c.fc)
	}
	executed, err := canonicalArgs(args)
	if err != nil {
		return nil, encodeError(t.Name(), id, err)
	}
	for _, fc := range trusted {
		if fc.Name != t.Name() {
			continue
		}
		storedArgs, err := canonicalArgs(fc.Args)
		if err != nil {
			return nil, encodeError(fc.Name, fc.ID, err)
		}
		if bytes.Equal(storedArgs, executed) {
			return nil, nil
		}
	}
	return nil, integrityErrorf("%s call '%s' arguments do not match its stored call", t.Name(), id)
}

// invocationSession returns the session of the invocation ctx belongs to, as
// agent.AddSessionToMemory finds it. ADK's own context types are asked under
// adkcontext.InvocationSessionKey, because a tool context's Session reports
// no session. Any other Context's Session wins when it is not nil, so a
// wrapper that overrides Session passes its own session; when it is nil, that
// Context is asked through Value too.
func invocationSession(ctx agent.Context) session.Session {
	if _, ours := ctx.(adkcontext.Source); ours {
		if s, ok := ctx.Value(adkcontext.InvocationSessionKey).(session.Session); ok {
			return s
		}
	}
	if s := ctx.Session(); s != nil {
		return s
	}
	s, _ := ctx.Value(adkcontext.InvocationSessionKey).(session.Session)
	return s
}

const stampPrefix = "v1:"

// sign returns the stamp of payload under the signing key.
func (p *integrity) sign(payload []byte) string {
	return stampPrefix + digest(p.keys[0], payload)
}

func digest(key, payload []byte) string {
	mac := hmac.New(sha256.New, key)
	mac.Write(payload)
	return hex.EncodeToString(mac.Sum(nil))
}

// verify reports whether stamp is the stamp of payload under any of the
// keys. It tries every key, so the time it takes does not depend on which
// one matched.
func (p *integrity) verify(payload []byte, stamp string) bool {
	got, ok := strings.CutPrefix(stamp, stampPrefix)
	if !ok {
		return false
	}
	matched := false
	for _, k := range p.keys {
		if hmac.Equal([]byte(digest(k, payload)), []byte(got)) {
			matched = true
		}
	}
	return matched
}

// unstamped handles a call with an ID and no stamp: an error by default, or a
// warning when unstamped calls are allowed.
func (p *integrity) unstamped(ctx context.Context, name, callID string) error {
	if !p.allowUnstamped {
		return integrityErrorf("%s call '%s' has no integrity stamp", name, callID)
	}
	p.logger.WarnContext(ctx, "function call has no integrity stamp", "tool", name, "function_call_id", callID)
	return nil
}

// integrityError is the error of every failed integrity check.
type integrityError struct{ msg string }

func integrityErrorf(format string, args ...any) error {
	return &integrityError{msg: fmt.Sprintf(format, args...)}
}

func (e *integrityError) Error() string { return e.msg }

// Is reports whether target is ErrIntegrity.
func (e *integrityError) Is(target error) bool { return target == ErrIntegrity }
