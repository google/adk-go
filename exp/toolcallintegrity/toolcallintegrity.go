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

// Package toolcallintegrity provides an optional HMAC integrity check for
// function calls stored in the session.
//
// ADK runs some stored function calls again when it resumes, such as a tool
// call that was waiting for confirmation. Anyone who can write to the session
// store can change a stored call's arguments before that happens.
//
// The plugin returned by [New] stamps each function call with an HMAC when the
// event is emitted, checks every stamp before each run, and lets a tool run
// only when its call has a valid stamp. Without the secret key, a modified call
// cannot be given a valid stamp.
//
//	p, err := toolcallintegrity.New(toolcallintegrity.Config{SecretKeys: [][]byte{key}})
//	if err != nil {
//		return err
//	}
//	r, err := runner.New(runner.Config{..., PluginConfig: runner.PluginConfig{Plugins: []*plugin.Plugin{p}}})
//
// It is a port of adk-python's ToolCallIntegrityPlugin.
//
// EXPERIMENTAL: this package is not yet stable API. Anything here may change or
// be removed in a later version without notice.
package toolcallintegrity

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"sync"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

const (
	// DefaultName is the plugin name used when [Config.Name] is empty.
	DefaultName = "tool_call_integrity"

	// InternalMetadataPrefix is the prefix of the event custom metadata keys
	// that only ADK code writes.
	InternalMetadataPrefix = "__adk_internal_"

	// StampsMetadataKey is the event custom metadata key holding the event's
	// stamps, a map from function call ID to stamp.
	StampsMetadataKey = InternalMetadataPrefix + "fc_hmac"

	// RestoredEventKey is the event custom metadata key that marks an event
	// restored into a session from outside it. See [PrepareRestoredEvent].
	RestoredEventKey = InternalMetadataPrefix + "restored_event"

	// stampPrefix starts every stamp, so the payload format can change later.
	stampPrefix = "v1:"
)

// Error is the error the plugin reports when a function call in the session
// fails its integrity check. The plugin's callbacks return it wrapped in an
// error that names the plugin and the callback; use [errors.As] to find it.
type Error struct {
	msg string
}

func (e *Error) Error() string { return e.msg }

func errorf(format string, args ...any) *Error {
	return &Error{msg: fmt.Sprintf(format, args...)}
}

// Config configures the plugin returned by [New].
type Config struct {
	// SecretKeys are the HMAC keys. The first key stamps new calls; every key
	// is accepted when verifying, so a retired key can stay in the list during
	// rotation. At least one key is required and none may be empty.
	SecretKeys [][]byte

	// AllowUnstampedCalls logs a warning instead of failing for function calls
	// that have no stamp, such as calls stored before the plugin was installed.
	// Such calls also run, so this reports tampering that removes a stamp but
	// does not prevent it. A stamp that is present but wrong is always
	// rejected.
	AllowUnstampedCalls bool

	// Name is the plugin name. It defaults to [DefaultName].
	Name string
}

// New returns a plugin that stamps function calls and verifies them before
// runs and tool calls.
//
// Its OnEventCallback adds "v1:" + hex(HMAC-SHA256(key, payload)) for each
// function call with an ID to the event's custom metadata under
// [StampsMetadataKey], and fails for a non-partial event that repeats a call
// ID, before the event is stored. Its BeforeRunCallback recomputes the stamp
// of every function call with an ID in the session history, answered or not,
// and fails when a stamp is wrong, malformed, repeated or left over from a
// call that no longer matches. Its BeforeToolCallback lets a tool run only
// when every stored copy of its call verifies and the tool's name and
// arguments match one of them; it always refuses a call with no ID.
// Unstamped calls are rejected unless [Config.AllowUnstampedCalls] is set.
//
// Events without function calls are ignored. Events marked as restored (see
// [PrepareRestoredEvent]) are not verified, and their function calls are never
// executed. Tools that run as workflow nodes are not checked: their arguments
// come from the workflow graph, and adk-go does not run plugin tool callbacks
// for them.
//
// adk-go differs from adk-python in two ways the plugin works around. A tool
// context does not expose the session, so the before-tool check reads the
// session the before-run check saw for the same invocation. And an error from
// a BeforeToolCallback reaches the model as the tool's result rather than the
// caller of the run; so that a refused call stops the run as it does in
// adk-python, the plugin returns the refusal from its OnEventCallback when the
// refused call's response event is emitted, and ends the invocation. The
// runner reports that error to the caller and does not store the event.
//
// Register this plugin before other plugins. The plugin manager stops at the
// first OnEventCallback that returns an event, and a later plugin must not
// change function calls or replace the custom metadata.
func New(cfg Config) (*plugin.Plugin, error) {
	if len(cfg.SecretKeys) == 0 || slices.ContainsFunc(cfg.SecretKeys, func(k []byte) bool { return len(k) == 0 }) {
		return nil, errors.New("toolcallintegrity: secret key must be non-empty bytes or a non-empty list of them")
	}
	p := &integrityPlugin{
		name:           cfg.Name,
		allowUnstamped: cfg.AllowUnstampedCalls,
		sessions:       map[string]session.Session{},
		refused:        map[refusal]error{},
	}
	if p.name == "" {
		p.name = DefaultName
	}
	for _, k := range cfg.SecretKeys {
		p.keys = append(p.keys, slices.Clone(k))
	}
	return plugin.New(plugin.Config{
		Name:               p.name,
		OnEventCallback:    p.onEvent,
		BeforeRunCallback:  p.beforeRun,
		AfterRunCallback:   p.afterRun,
		BeforeToolCallback: p.beforeTool,
	})
}

// PrepareRestoredEvent returns a copy of event with internal metadata removed
// and marked as restored. Use it on events that your code copies into another
// session, so that the history passes the integrity check. The copy only keeps
// the history: its function calls are never executed. event is not modified.
func PrepareRestoredEvent(event *session.Event) (*session.Event, error) {
	b, err := json.Marshal(event)
	if err != nil {
		return nil, fmt.Errorf("toolcallintegrity: copy event: %w", err)
	}
	cp := &session.Event{}
	if err := json.Unmarshal(b, cp); err != nil {
		return nil, fmt.Errorf("toolcallintegrity: copy event: %w", err)
	}
	md := make(map[string]any, len(cp.CustomMetadata)+1)
	for k, v := range cp.CustomMetadata {
		if !strings.HasPrefix(k, InternalMetadataPrefix) {
			md[k] = v
		}
	}
	md[RestoredEventKey] = true
	cp.CustomMetadata = md
	return cp, nil
}

type refusal struct {
	invocationID string
	callID       string
}

type integrityPlugin struct {
	name           string
	keys           [][]byte
	allowUnstamped bool

	mu sync.Mutex
	// sessions holds the session of each running invocation, by invocation
	// ID, for the before-tool check.
	sessions map[string]session.Session
	// refused holds the calls BeforeToolCallback refused and have not yet
	// produced a response event, with the error to report for them.
	refused map[refusal]error
}

// wrap names the plugin and the callback, as adk-python's plugin manager
// does for an error a plugin raises.
func (p *integrityPlugin) wrap(callback string, err error) error {
	return fmt.Errorf("error in plugin '%s' during '%s' callback: %w", p.name, callback, err)
}

func (p *integrityPlugin) sign(payload []byte) string {
	mac := hmac.New(sha256.New, p.keys[0])
	mac.Write(payload)
	return stampPrefix + hex.EncodeToString(mac.Sum(nil))
}

func (p *integrityPlugin) verify(payload []byte, stamp string) bool {
	digest, ok := strings.CutPrefix(stamp, stampPrefix)
	if !ok {
		return false
	}
	valid := false
	for _, key := range p.keys {
		mac := hmac.New(sha256.New, key)
		mac.Write(payload)
		if hmac.Equal([]byte(hex.EncodeToString(mac.Sum(nil))), []byte(digest)) {
			valid = true
		}
	}
	return valid
}

func (p *integrityPlugin) unstamped(name, callID string) error {
	if !p.allowUnstamped {
		return errorf("%s call %q has no integrity stamp.", name, callID)
	}
	slog.Warn("toolcallintegrity: function call has no integrity stamp", "name", name, "call_id", callID) //nolint:forbidigo // should be changed when we apply consistent logging approach
	return nil
}

func (p *integrityPlugin) onEvent(ctx agent.InvocationContext, event *session.Event) (*session.Event, error) {
	if err := p.takeRefusal(ctx.InvocationID(), event); err != nil {
		// A tool context cannot end the invocation, so the refusal ends it
		// here, before the model is called again.
		ctx.EndInvocation()
		return nil, err
	}
	var calls []*genai.FunctionCall
	for _, fc := range functionCalls(event) {
		if fc.ID != "" {
			calls = append(calls, fc)
		}
	}
	if len(calls) == 0 {
		return nil, nil
	}
	// Stamps are keyed by call ID, so a response that repeats an ID cannot be
	// stamped, and the before-run check would reject the session on every
	// later run. The response is refused before it is stored, so none of its
	// calls run and the session stays usable. Partial events are neither
	// stored nor executed.
	if !event.Partial {
		seen := map[string]bool{}
		for _, fc := range calls {
			if seen[fc.ID] {
				return nil, p.wrap("on_event_callback", errorf("Event %q repeats function call ID %q.", event.ID, fc.ID))
			}
			seen[fc.ID] = true
		}
	}
	stamps := make(map[string]any, len(calls))
	for _, fc := range calls {
		payload, err := canonicalPayload(ctx.Session(), event, fc)
		if err != nil {
			return nil, p.wrap("on_event_callback", err)
		}
		stamps[fc.ID] = p.sign(payload)
	}
	// The metadata map may be shared with the model response, so it is
	// replaced rather than modified.
	md := make(map[string]any, len(event.CustomMetadata)+1)
	maps.Copy(md, event.CustomMetadata)
	md[StampsMetadataKey] = stamps
	event.CustomMetadata = md
	return nil, nil
}

// takeRefusal returns the refusal recorded for a function response in event,
// forgetting it.
func (p *integrityPlugin) takeRefusal(invocationID string, event *session.Event) error {
	if event.Content == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.refused) == 0 {
		return nil
	}
	for _, part := range event.Content.Parts {
		if part == nil || part.FunctionResponse == nil {
			continue
		}
		key := refusal{invocationID, part.FunctionResponse.ID}
		if err, ok := p.refused[key]; ok {
			delete(p.refused, key)
			return err
		}
	}
	return nil
}

func (p *integrityPlugin) afterRun(ctx agent.InvocationContext) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.sessions, ctx.InvocationID())
	maps.DeleteFunc(p.refused, func(k refusal, _ error) bool { return k.invocationID == ctx.InvocationID() })
}

func (p *integrityPlugin) beforeRun(ctx agent.InvocationContext) (*genai.Content, error) {
	sess := ctx.Session()
	if err := p.verifySession(sess); err != nil {
		return nil, p.wrap("before_run_callback", err)
	}
	p.mu.Lock()
	p.sessions[ctx.InvocationID()] = sess
	p.mu.Unlock()
	return nil, nil
}

func (p *integrityPlugin) verifySession(sess session.Session) error {
	for event := range sess.Events().All() {
		calls := functionCalls(event)
		// A stamp on an event without calls has nothing to execute. Calls in
		// restored events are refused by the before-tool check instead.
		if len(calls) == 0 || isRestored(event) {
			continue
		}
		stamps, err := eventStamps(event)
		if err != nil {
			return err
		}
		verified := map[string]bool{}
		for _, fc := range calls {
			// A call without an ID cannot be stamped. The before-tool check
			// refuses to run it, so there is nothing to check here.
			if fc.ID == "" {
				continue
			}
			stamp, ok := stamps[fc.ID]
			if !ok {
				if err := p.unstamped(fc.Name, fc.ID); err != nil {
					return err
				}
				continue
			}
			if verified[fc.ID] {
				return errorf("Event %q has duplicate function call ID %q.", event.ID, fc.ID)
			}
			payload, err := canonicalPayload(sess, event, fc)
			if err != nil {
				return err
			}
			if !p.verify(payload, stamp) {
				return errorf("%s call %q does not match its integrity stamp.", fc.Name, fc.ID)
			}
			verified[fc.ID] = true
		}
		// A stamp without a matching call in its event means the call was
		// renamed, re-keyed or removed.
		var unmatched []string
		for id := range stamps {
			if !verified[id] {
				unmatched = append(unmatched, id)
			}
		}
		if len(unmatched) > 0 {
			slices.Sort(unmatched)
			return errorf("Event %q has integrity stamps with no matching call: %q.", event.ID, unmatched)
		}
	}
	return nil
}

func (p *integrityPlugin) beforeTool(ctx agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
	if err := p.checkToolCall(ctx, t, args); err != nil {
		err = p.wrap("before_tool_callback", err)
		p.mu.Lock()
		p.refused[refusal{ctx.InvocationID(), ctx.FunctionCallID()}] = err
		p.mu.Unlock()
		return nil, err
	}
	return nil, nil
}

// session returns the session of the invocation ctx belongs to. A tool context
// does not expose the session, so the one the before-run check saw is used;
// ctx.Session() is the fallback for a callback called outside a run.
func (p *integrityPlugin) session(ctx agent.Context) session.Session {
	p.mu.Lock()
	sess := p.sessions[ctx.InvocationID()]
	p.mu.Unlock()
	if sess != nil {
		return sess
	}
	return ctx.Session()
}

func (p *integrityPlugin) checkToolCall(ctx agent.Context, t tool.Tool, args map[string]any) error {
	callID := ctx.FunctionCallID()
	// ADK gives every new call an ID before running it, so a call without one
	// was replayed from the session and cannot be matched to its stamp.
	if callID == "" {
		return errorf("%s call has no ID, so its integrity cannot be checked.", t.Name())
	}
	sess := p.session(ctx)
	if sess == nil {
		return errorf("%s call %q cannot be checked: the session is not available.", t.Name(), callID)
	}
	type storedCall struct {
		event *session.Event
		call  *genai.FunctionCall
	}
	var stored []storedCall
	restored := false
	for event := range sess.Events().All() {
		for _, fc := range functionCalls(event) {
			if fc.ID != callID {
				continue
			}
			if isRestored(event) {
				restored = true
			} else {
				stored = append(stored, storedCall{event, fc})
			}
		}
	}
	if len(stored) == 0 {
		if restored {
			return errorf("%s call %q comes from restored history and cannot be executed.", t.Name(), callID)
		}
		return p.unstamped(t.Name(), callID)
	}

	// Every stored copy of the call must verify, so a repeated ID cannot
	// borrow the stamp of another copy. Restored copies are never trusted; an
	// unstamped copy is trusted only when AllowUnstampedCalls logged it.
	var trusted []*genai.FunctionCall
	for _, sc := range stored {
		stamps, err := eventStamps(sc.event)
		if err != nil {
			return err
		}
		if stamp, ok := stamps[callID]; !ok {
			if err := p.unstamped(t.Name(), callID); err != nil {
				return err
			}
		} else {
			payload, err := canonicalPayload(sess, sc.event, sc.call)
			if err != nil {
				return err
			}
			if !p.verify(payload, stamp) {
				return errorf("%s call %q does not match its integrity stamp.", t.Name(), callID)
			}
		}
		trusted = append(trusted, sc.call)
	}

	// What runs must be one of those calls, whichever copy ADK took the
	// arguments from.
	executed, err := canonicalArgs(args)
	if err != nil {
		return err
	}
	for _, fc := range trusted {
		if fc.Name != t.Name() {
			continue
		}
		if stored, err := canonicalArgs(fc.Args); err == nil && bytes.Equal(stored, executed) {
			return nil
		}
	}
	return errorf("%s call %q arguments do not match its stored call.", t.Name(), callID)
}

func functionCalls(event *session.Event) []*genai.FunctionCall {
	if event == nil || event.Content == nil {
		return nil
	}
	var out []*genai.FunctionCall
	for _, part := range event.Content.Parts {
		if part != nil && part.FunctionCall != nil {
			out = append(out, part.FunctionCall)
		}
	}
	return out
}

func isRestored(event *session.Event) bool {
	restored, _ := event.CustomMetadata[RestoredEventKey].(bool)
	return restored
}

// eventStamps returns the event's stamps, failing if the metadata is
// malformed. The stamps are a map[string]any when the event went through
// JSON, as stored events do.
func eventStamps(event *session.Event) (map[string]string, error) {
	raw, ok := event.CustomMetadata[StampsMetadataKey]
	if !ok || raw == nil {
		return nil, nil
	}
	malformed := errorf("Event %q has a malformed integrity stamp.", event.ID)
	out := map[string]string{}
	switch m := raw.(type) {
	case map[string]string:
		maps.Copy(out, m)
	case map[string]any:
		for k, v := range m {
			s, ok := v.(string)
			if !ok {
				return nil, malformed
			}
			out[k] = s
		}
	default:
		return nil, malformed
	}
	for _, v := range out {
		for i := 0; i < len(v); i++ {
			if v[i] >= 0x80 {
				return nil, malformed
			}
		}
	}
	return out, nil
}

// canonicalPayload returns the bytes a stamp covers: the session, the
// invocation, branch and author of the event, and the call.
func canonicalPayload(sess session.Session, event *session.Event, fc *genai.FunctionCall) ([]byte, error) {
	args, err := jsonArgs(fc.Args)
	if err != nil {
		return nil, err
	}
	return canonicalJSON(map[string]any{
		"app_name":      sess.AppName(),
		"user_id":       sess.UserID(),
		"session_id":    sess.ID(),
		"invocation_id": event.InvocationID,
		"branch":        event.Branch,
		"author":        event.Author,
		"name":          fc.Name,
		"id":            fc.ID,
		"args":          args,
	})
}

func canonicalArgs(args map[string]any) ([]byte, error) {
	v, err := jsonArgs(args)
	if err != nil {
		return nil, err
	}
	return canonicalJSON(v)
}

// jsonArgs returns the call's arguments in the form they take after a round
// trip through JSON, as the session stores write them, so that values such
// as times and byte slices verify after storage.
func jsonArgs(args map[string]any) (any, error) {
	if len(args) == 0 {
		return map[string]any{}, nil
	}
	b, err := json.Marshal(args)
	if err != nil {
		return nil, errorf("function call arguments are not JSON: %v", err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, errorf("function call arguments are not JSON: %v", err)
	}
	return normalize(v), nil
}

// normalize writes whole numbers as integers, so that a number a session store
// keeps as a double (10.0 for 10) verifies.
func normalize(v any) any {
	switch x := v.(type) {
	case json.Number:
		s := x.String()
		if _, err := strconv.ParseInt(s, 10, 64); err == nil {
			return x
		}
		f, err := x.Float64()
		if err == nil && f == math.Trunc(f) && math.Abs(f) < 1e21 {
			return json.Number(strconv.FormatFloat(f, 'f', -1, 64))
		}
		return x
	case map[string]any:
		for k, e := range x {
			x[k] = normalize(e)
		}
		return x
	case []any:
		for i, e := range x {
			x[i] = normalize(e)
		}
		return x
	}
	return v
}

// canonicalJSON encodes v with sorted keys, no insignificant whitespace and no
// HTML escaping.
func canonicalJSON(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, errorf("cannot encode the integrity payload: %v", err)
	}
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n")), nil
}
