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

// Package toolcallintegrity provides a plugin that stamps every function call
// stored in a session with an HMAC, and refuses to run a tool whose call does
// not verify. Without the secret key, a call that was modified in the session
// store cannot be given a valid stamp.
//
// This is the Go version of adk-python's ToolCallIntegrityPlugin.
//
// Register the plugin before other plugins. The runner stops at the first
// OnEventCallback that returns an event, and a later plugin must not change
// function calls or replace CustomMetadata.
//
// Tools that run as workflow nodes are not checked: their arguments come from
// the workflow graph, and plugin tool callbacks do not run for them.
package toolcallintegrity

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

const (
	// MetadataKey is the Event.CustomMetadata key that holds the stamps of an
	// event's function calls, as a map from function call ID to stamp.
	MetadataKey = "adk_fc_hmac"

	// stampPrefix versions the payload format, so it can change later.
	stampPrefix = "v1:"
)

// ErrIntegrity is wrapped by every error the plugin returns for a function
// call that fails its integrity check.
var ErrIntegrity = errors.New("tool call integrity check failed")

// Config configures the plugin.
type Config struct {
	// Keys are the HMAC keys. Keys[0] stamps new calls, and every key is
	// accepted when verifying, so a key can be rotated by putting the new key
	// first. At least one non-empty key is required.
	Keys [][]byte

	// AllowUnstampedCalls lets a call that has no stamp, such as a call stored
	// before the plugin was installed, pass the checks and run. This leaves
	// tampering that removes a stamp undetected, so use it only while existing
	// sessions are migrated. A stamp that is present but wrong always fails.
	//
	// adk-python logs each unstamped call. ADK Go has no logger for library
	// code, so the call is not reported.
	AllowUnstampedCalls bool
}

type integrity struct {
	keys           [][]byte
	allowUnstamped bool

	mu sync.Mutex
	// trusted holds, for each running invocation, the calls a tool may run
	// as: function call ID to the canonical name and arguments of every
	// trusted copy. A tool context has no access to the session, so
	// beforeTool checks against this instead of re-reading the events.
	trusted map[string]map[string][]string
}

// New returns a plugin that stamps function calls when their event is
// emitted and verifies them before each run and before each tool call.
//
// OnEventCallback fails a non-partial event that repeats a function call ID,
// so the event is not stored. BeforeRunCallback fails the run when any stamp
// in the session is wrong, malformed, repeated or left over from a call that
// no longer matches. BeforeToolCallback fails a tool call whose ID, name or
// arguments do not match a call that was verified at the start of the run or
// stamped during it; the tool then does not run and the model gets the error.
//
// The plugin is safe for concurrent use by several runs.
func New(cfg Config) (*plugin.Plugin, error) {
	if len(cfg.Keys) == 0 {
		return nil, errors.New("toolcallintegrity: at least one key is required")
	}
	keys := make([][]byte, len(cfg.Keys))
	for i, k := range cfg.Keys {
		if len(k) == 0 {
			return nil, fmt.Errorf("toolcallintegrity: key %d is empty", i)
		}
		keys[i] = slices.Clone(k)
	}
	p := &integrity{
		keys:           keys,
		allowUnstamped: cfg.AllowUnstampedCalls,
		trusted:        map[string]map[string][]string{},
	}
	return plugin.New(plugin.Config{
		Name:               "tool_call_integrity",
		OnEventCallback:    p.onEvent,
		BeforeRunCallback:  p.beforeRun,
		AfterRunCallback:   p.afterRun,
		BeforeToolCallback: p.beforeTool,
	})
}

type payload struct {
	AppName      string         `json:"app_name"`
	UserID       string         `json:"user_id"`
	SessionID    string         `json:"session_id"`
	InvocationID string         `json:"invocation_id"`
	Branch       string         `json:"branch"`
	Author       string         `json:"author"`
	Name         string         `json:"name"`
	ID           string         `json:"id"`
	Args         map[string]any `json:"args"`
}

// canonicalPayload returns the bytes a stamp covers. encoding/json sorts map
// keys and writes a whole-number float64 like an int, so the payload is the
// same after a round trip through a session store that keeps JSON.
func canonicalPayload(sess session.Session, ev *session.Event, fc *genai.FunctionCall) ([]byte, error) {
	b, err := json.Marshal(payload{
		AppName:      sess.AppName(),
		UserID:       sess.UserID(),
		SessionID:    sess.ID(),
		InvocationID: ev.InvocationID,
		Branch:       ev.Branch,
		Author:       ev.Author,
		Name:         fc.Name,
		ID:           fc.ID,
		Args:         argsOrEmpty(fc.Args),
	})
	if err != nil {
		return nil, fmt.Errorf("%w: encoding %s call %q: %w", ErrIntegrity, fc.Name, fc.ID, err)
	}
	return b, nil
}

// canonicalCall is what beforeTool compares: the tool name and arguments.
func canonicalCall(name, id string, args map[string]any) (string, error) {
	b, err := json.Marshal(struct {
		Name string         `json:"name"`
		Args map[string]any `json:"args"`
	}{name, argsOrEmpty(args)})
	if err != nil {
		return "", fmt.Errorf("%w: encoding %s call %q: %w", ErrIntegrity, name, id, err)
	}
	return string(b), nil
}

func argsOrEmpty(args map[string]any) map[string]any {
	if args == nil {
		return map[string]any{}
	}
	return args
}

func (p *integrity) sign(data []byte) string {
	mac := hmac.New(sha256.New, p.keys[0])
	mac.Write(data)
	return stampPrefix + hex.EncodeToString(mac.Sum(nil))
}

func (p *integrity) verify(data []byte, stamp string) bool {
	digest, ok := strings.CutPrefix(stamp, stampPrefix)
	if !ok {
		return false
	}
	got, err := hex.DecodeString(digest)
	if err != nil {
		return false
	}
	for _, key := range p.keys {
		mac := hmac.New(sha256.New, key)
		mac.Write(data)
		if hmac.Equal(mac.Sum(nil), got) {
			return true
		}
	}
	return false
}

func (p *integrity) unstamped(name, id string) error {
	if p.allowUnstamped {
		return nil
	}
	return fmt.Errorf("%w: %s call %q has no integrity stamp", ErrIntegrity, name, id)
}

// stamps returns the event's stamps. A session store that keeps JSON returns
// them as map[string]any, so both forms are accepted.
func stamps(ev *session.Event) (map[string]string, error) {
	v, ok := ev.CustomMetadata[MetadataKey]
	if !ok || v == nil {
		return nil, nil
	}
	switch m := v.(type) {
	case map[string]string:
		return m, nil
	case map[string]any:
		out := make(map[string]string, len(m))
		for id, s := range m {
			str, ok := s.(string)
			if !ok {
				return nil, fmt.Errorf("%w: event %q has a malformed integrity stamp", ErrIntegrity, ev.ID)
			}
			out[id] = str
		}
		return out, nil
	default:
		return nil, fmt.Errorf("%w: event %q has a malformed integrity stamp", ErrIntegrity, ev.ID)
	}
}

func (p *integrity) onEvent(ctx agent.InvocationContext, ev *session.Event) (*session.Event, error) {
	var calls []*genai.FunctionCall
	for _, fc := range utils.FunctionCalls(ev.Content) {
		if fc.ID != "" {
			calls = append(calls, fc)
		}
	}
	if len(calls) == 0 {
		return nil, nil
	}
	// Stamps are keyed by call ID, so an event that repeats an ID cannot be
	// stamped and would fail every later run. Partial events are neither
	// stored nor executed.
	if !ev.Partial {
		seen := make(map[string]bool, len(calls))
		for _, fc := range calls {
			if seen[fc.ID] {
				return nil, fmt.Errorf("%w: event %q repeats function call ID %q", ErrIntegrity, ev.ID, fc.ID)
			}
			seen[fc.ID] = true
		}
	}
	sess := ctx.Session()
	stamped := make(map[string]string, len(calls))
	canonical := make(map[string]string, len(calls))
	for _, fc := range calls {
		data, err := canonicalPayload(sess, ev, fc)
		if err != nil {
			return nil, err
		}
		stamped[fc.ID] = p.sign(data)
		if canonical[fc.ID], err = canonicalCall(fc.Name, fc.ID, fc.Args); err != nil {
			return nil, err
		}
	}
	// Copy rather than write into a map the event may share with its caller.
	md := maps.Clone(ev.CustomMetadata)
	if md == nil {
		md = make(map[string]any, 1)
	}
	md[MetadataKey] = stamped
	ev.CustomMetadata = md

	if !ev.Partial {
		p.mu.Lock()
		defer p.mu.Unlock()
		// Only a run that passed beforeRun can execute tools, and afterRun
		// clears its entry, so an event outside such a run records nothing.
		if trusted, ok := p.trusted[ctx.InvocationID()]; ok {
			for id, c := range canonical {
				trusted[id] = append(trusted[id], c)
			}
		}
	}
	return nil, nil
}

func (p *integrity) beforeRun(ctx agent.InvocationContext) (*genai.Content, error) {
	sess := ctx.Session()
	trusted := map[string][]string{}
	for ev := range sess.Events().All() {
		calls := utils.FunctionCalls(ev.Content)
		if len(calls) == 0 {
			continue
		}
		evStamps, err := stamps(ev)
		if err != nil {
			return nil, err
		}
		verified := make(map[string]bool, len(calls))
		for _, fc := range calls {
			// A call without an ID cannot be stamped, and beforeTool refuses
			// to run it.
			if fc.ID == "" {
				continue
			}
			stamp, ok := evStamps[fc.ID]
			if !ok {
				if err := p.unstamped(fc.Name, fc.ID); err != nil {
					return nil, err
				}
			} else {
				if verified[fc.ID] {
					return nil, fmt.Errorf("%w: event %q has duplicate function call ID %q", ErrIntegrity, ev.ID, fc.ID)
				}
				data, err := canonicalPayload(sess, ev, fc)
				if err != nil {
					return nil, err
				}
				if !p.verify(data, stamp) {
					return nil, fmt.Errorf("%w: %s call %q does not match its integrity stamp", ErrIntegrity, fc.Name, fc.ID)
				}
				verified[fc.ID] = true
			}
			c, err := canonicalCall(fc.Name, fc.ID, fc.Args)
			if err != nil {
				return nil, err
			}
			trusted[fc.ID] = append(trusted[fc.ID], c)
		}
		// A stamp with no matching call means the call was renamed, re-keyed
		// or removed.
		for id := range evStamps {
			if !verified[id] {
				return nil, fmt.Errorf("%w: event %q has an integrity stamp for %q with no matching call", ErrIntegrity, ev.ID, id)
			}
		}
	}
	p.mu.Lock()
	p.trusted[ctx.InvocationID()] = trusted
	p.mu.Unlock()
	return nil, nil
}

func (p *integrity) afterRun(ctx agent.InvocationContext) {
	p.mu.Lock()
	delete(p.trusted, ctx.InvocationID())
	p.mu.Unlock()
}

func (p *integrity) beforeTool(ctx agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
	callID := ctx.FunctionCallID()
	// ADK gives every new call an ID before running it, so a call without one
	// cannot be matched to its stamp.
	if callID == "" {
		return nil, fmt.Errorf("%w: %s call has no ID, so its integrity cannot be checked", ErrIntegrity, t.Name())
	}
	executed, err := canonicalCall(t.Name(), callID, args)
	if err != nil {
		return nil, err
	}

	p.mu.Lock()
	known, ok := p.trusted[ctx.InvocationID()][callID]
	p.mu.Unlock()
	if !ok {
		return nil, p.unstamped(t.Name(), callID)
	}
	if !slices.Contains(known, executed) {
		return nil, fmt.Errorf("%w: %s call %q does not match its stored call", ErrIntegrity, t.Name(), callID)
	}
	return nil, nil
}
