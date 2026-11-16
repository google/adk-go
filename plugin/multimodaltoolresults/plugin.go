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

// Package multimodaltoolresults provides a plugin that attaches GenAI parts
// returned by tools to the model request.
package multimodaltoolresults

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"unicode"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/converters"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

const (
	// PartsReturnedByToolsKey is the invocation-scoped ("temp:") session state
	// key under which [RetentionNextModelCall] stores tool-returned parts until
	// the next model request attaches them.
	PartsReturnedByToolsKey = "temp:PARTS_RETURNED_BY_TOOLS_ID"

	// SessionPartsReturnedByToolsKey is the session-scoped state key under
	// which [RetentionSession] persists serialized non-inline tool-returned
	// parts across invocations.
	SessionPartsReturnedByToolsKey = "multimodal_tool_results_plugin:PARTS_RETURNED_BY_TOOLS_ID"
)

const defaultName = "multimodal_tool_results_plugin"

// Retention controls how long tool-returned parts stay attached to model
// requests.
type Retention string

const (
	// RetentionNextModelCall attaches saved tool-returned parts to the next
	// model request once and then clears them. Parts that the framework
	// already carries in FunctionResponse.Parts (inline data with bytes and a
	// MIME type, or file data with a URI and a MIME type) are skipped.
	RetentionNextModelCall Retention = "next_model_call"

	// RetentionSession re-attaches the latest saved non-inline parts (such as
	// file-data and text parts) to every subsequent model request in the
	// session across turns, while attaching inline-data parts only once to the
	// next model request.
	RetentionSession Retention = "session"
)

// Config configures a multimodal tool results plugin. The zero value is valid
// and selects Name "multimodal_tool_results_plugin" and Retention
// [RetentionNextModelCall].
type Config struct {
	// Name is the plugin instance's identifier, unique among the plugins
	// registered on a runner. Empty selects "multimodal_tool_results_plugin".
	Name string

	// Retention controls how long tool-returned parts stay attached to model
	// requests. Empty selects [RetentionNextModelCall]. Any value other than
	// "", [RetentionNextModelCall], or [RetentionSession] causes [New] to
	// fail.
	Retention Retention
}

// New creates a multimodal tool results plugin from cfg.
//
// Preconditions: cfg.Retention is "", [RetentionNextModelCall], or
// [RetentionSession].
//
// Postconditions: Returns a non-nil *plugin.Plugin with AfterToolCallback and
// BeforeModelCallback configured for cfg.Retention, and a nil error. cfg is
// read once and not retained.
//
// Failure: If cfg.Retention is not "", [RetentionNextModelCall], or
// [RetentionSession], New returns (nil, err) where err is a non-nil error
// whose message contains "retention must be 'next_model_call' or 'session', got <value>".
//
// Concurrency: New is pure and safe for concurrent use. The returned
// *plugin.Plugin is safe for concurrent use across parallel tool calls and
// concurrent runner invocations.
func New(cfg Config) (*plugin.Plugin, error) {
	retention := cfg.Retention
	switch retention {
	case "":
		retention = RetentionNextModelCall
	case RetentionNextModelCall, RetentionSession:
	default:
		return nil, fmt.Errorf("multimodaltoolresults: retention must be 'next_model_call' or 'session', got %s", cfg.Retention)
	}
	name := cfg.Name
	if name == "" {
		name = defaultName
	}
	m := &mtrPlugin{retention: retention}
	return plugin.New(plugin.Config{
		Name:                name,
		AfterToolCallback:   m.afterTool,
		BeforeModelCallback: m.beforeModel,
	})
}

// Invocation-scoped keys of [RetentionSession]. temp: keys are never stored
// with the session, so both are absent when an invocation starts.
const (
	// currentTurnPartsKey holds every entry the tools returned since the last
	// model request that attached them.
	currentTurnPartsKey = "temp:multimodal_tool_results_plugin:current_turn_parts"
	// sessionUpdatedKey is present once this invocation has stored parts, so
	// that later saves of the invocation append instead of replacing.
	sessionUpdatedKey = "temp:multimodal_tool_results_plugin:updated_in_invocation"
)

// mtrPlugin holds a plugin's retention and the lock over its state updates.
// The tool calls of one model turn run concurrently, so both callbacks hold mu
// from their first read of the plugin's keys to their last write.
type mtrPlugin struct {
	retention Retention
	mu        sync.Mutex // guards the plugin's read-modify-write of ctx.State()
}

// afterTool saves the parts of a part tool result (see extractPartList) for
// the next model requests. It never replaces the result, so later plugins'
// and the agent's after-tool callbacks run; adk-python returns the result
// for any other value, which stops them (multimodal_tool_results_plugin.py:113).
func (m *mtrPlugin) afterTool(ctx agent.Context, _ tool.Tool, _, result map[string]any, toolErr error) (map[string]any, error) {
	if ctx == nil || toolErr != nil {
		return nil, nil
	}
	state := ctx.State()
	if state == nil {
		return nil, nil
	}
	entries, ok := extractPartList(result)
	if !ok {
		return nil, nil
	}
	if m.retention == RetentionSession {
		return nil, m.saveForSession(state, entries)
	}
	return nil, m.saveForNextModelCall(state, entries)
}

// saveForNextModelCall appends the entries the function response does not
// carry itself to PartsReturnedByToolsKey.
func (m *mtrPlugin) saveForNextModelCall(state session.State, entries []any) error {
	kept := slices.DeleteFunc(entries, func(e any) bool { return toolinternal.MediaPart(e) != nil })
	if len(kept) == 0 {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return appendState(state, PartsReturnedByToolsKey, kept)
}

// saveForSession stores every entry except inline-data parts under
// SessionPartsReturnedByToolsKey, replacing the list on the invocation's
// first save, and appends all entries to currentTurnPartsKey.
func (m *mtrPlugin) saveForSession(state session.State, entries []any) error {
	var stored []any
	for _, e := range entries {
		p, isPart := e.(*genai.Part)
		switch {
		case !isPart || p == nil:
			stored = append(stored, e)
		case p.InlineData == nil:
			sp, err := serializePart(p)
			if err != nil {
				return fmt.Errorf("multimodaltoolresults: serialize part: %w", err)
			}
			stored = append(stored, sp)
		}
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if len(stored) > 0 {
		updated, err := hasKey(state, sessionUpdatedKey)
		if err != nil {
			return err
		}
		if updated {
			if err := appendState(state, SessionPartsReturnedByToolsKey, stored); err != nil {
				return err
			}
		} else {
			if err := setState(state, sessionUpdatedKey, true); err != nil {
				return err
			}
			if err := setState(state, SessionPartsReturnedByToolsKey, stored); err != nil {
				return err
			}
		}
	}
	return appendState(state, currentTurnPartsKey, entries)
}

// beforeModel appends the saved parts to the last content of the request.
// A request without contents leaves everything pending.
func (m *mtrPlugin) beforeModel(ctx agent.Context, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
	if ctx == nil || llmRequest == nil || len(llmRequest.Contents) == 0 {
		return nil, nil
	}
	last := llmRequest.Contents[len(llmRequest.Contents)-1]
	state := ctx.State()
	if last == nil || state == nil {
		return nil, nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.retention == RetentionSession {
		return nil, attachForSession(state, last)
	}
	return nil, attachForNextModelCall(state, last)
}

// attachForNextModelCall attaches the parts saved under
// PartsReturnedByToolsKey once and empties the list.
func attachForNextModelCall(state session.State, last *genai.Content) error {
	entries, err := readEntries(state, PartsReturnedByToolsKey)
	if err != nil || len(entries) == 0 {
		return err
	}
	if err := state.Set(PartsReturnedByToolsKey, []any{}); err != nil {
		return fmt.Errorf("multimodaltoolresults: clear temp state: %w", err)
	}
	last.Parts = append(last.Parts, partsOf(entries)...)
	return nil
}

// attachForSession attaches the stored parts that no current-turn part
// equals, followed by the current-turn parts, and empties the current-turn
// list. The stored list stays for later requests.
func attachForSession(state session.State, last *genai.Content) error {
	storedEntries, err := readEntries(state, SessionPartsReturnedByToolsKey)
	if err != nil {
		return err
	}
	currentEntries, err := readEntries(state, currentTurnPartsKey)
	if err != nil {
		return err
	}
	current := partsOf(currentEntries)

	var attach []*genai.Part
	for _, e := range storedEntries {
		var sp *genai.Part
		var ok bool
		if em, isMap := e.(map[string]any); isMap {
			sp, ok = decodePart(em)
		} else {
			sp, ok = asPart(e)
		}
		if !ok {
			continue
		}
		dup, err := containsPart(current, sp)
		if err != nil {
			return fmt.Errorf("multimodaltoolresults: compare parts: %w", err)
		}
		if !dup {
			attach = append(attach, sp)
		}
	}
	attach = append(attach, current...)

	if len(currentEntries) > 0 {
		if err := state.Set(currentTurnPartsKey, []any{}); err != nil {
			return fmt.Errorf("multimodaltoolresults: clear temp state: %w", err)
		}
	}
	last.Parts = append(last.Parts, attach...)
	return nil
}

// extractPartList returns the entries of a part tool result: a result whose
// only key is "result", holding a part or a non-empty list or array whose
// first entry is a part. Parts are returned as *genai.Part; other entries
// after the first are returned unchanged. The flow wraps a tool's non-map
// return value v as {"result": v}, so a handler that returns the one-key map
// itself is treated the same.
func extractPartList(result map[string]any) ([]any, bool) {
	if len(result) != 1 {
		return nil, false
	}
	raw, ok := result["result"]
	if !ok || raw == nil {
		return nil, false
	}
	if p, ok := asPart(raw); ok {
		return []any{p}, true
	}
	v := reflect.ValueOf(raw)
	if (v.Kind() != reflect.Slice && v.Kind() != reflect.Array) || v.Type().Elem().Kind() == reflect.Uint8 || v.Len() == 0 {
		return nil, false
	}
	first, ok := asPart(v.Index(0).Interface())
	if !ok {
		return nil, false
	}
	out := make([]any, v.Len())
	out[0] = first
	for i := 1; i < v.Len(); i++ {
		elem := v.Index(i).Interface()
		if p, ok := asPart(elem); ok {
			out[i] = p
		} else {
			out[i] = elem
		}
	}
	return out, true
}

// asPart returns v as a non-nil *genai.Part. A genai.Part value is copied.
func asPart(v any) (*genai.Part, bool) {
	switch p := v.(type) {
	case *genai.Part:
		return p, p != nil
	case genai.Part:
		cp := p
		return &cp, true
	default:
		return nil, false
	}
}

// partsOf returns the parts among entries, in order.
func partsOf(entries []any) []*genai.Part {
	var parts []*genai.Part
	for _, e := range entries {
		if p, ok := asPart(e); ok {
			parts = append(parts, p)
		}
	}
	return parts
}

func containsPart(parts []*genai.Part, p *genai.Part) (bool, error) {
	for _, q := range parts {
		eq, err := partEqual(p, q)
		if err != nil || eq {
			return eq, err
		}
	}
	return false, nil
}

// hasKey reports whether state holds key.
func hasKey(state session.State, key string) (bool, error) {
	_, err := state.Get(key)
	switch {
	case err == nil:
		return true, nil
	case errors.Is(err, session.ErrStateKeyNotExist):
		return false, nil
	default:
		return false, fmt.Errorf("multimodaltoolresults: get state %q: %w", key, err)
	}
}

func setState(state session.State, key string, v any) error {
	if err := state.Set(key, v); err != nil {
		return fmt.Errorf("multimodaltoolresults: set state: %w", err)
	}
	return nil
}

// appendState sets key to a new list holding its current entries followed by
// add. An absent key counts as an empty list; a value that is not a list is
// an error, as adk-python's += on it raises.
func appendState(state session.State, key string, add []any) error {
	v, err := state.Get(key)
	if err != nil && !errors.Is(err, session.ErrStateKeyNotExist) {
		return fmt.Errorf("multimodaltoolresults: get state %q: %w", key, err)
	}
	var prev []any
	if err == nil {
		var ok bool
		if prev, ok = listOf(v); !ok {
			return fmt.Errorf("multimodaltoolresults: state key %q holds %T, want a list", key, v)
		}
	}
	return setState(state, key, slices.Concat(prev, add))
}

// readEntries returns the entries adk-python iterates over when it reads key:
// none for an absent key or a zero value (Python skips a falsy value), none
// for a string or a map (whose characters or keys are never parts), and the
// elements of a list or array. Any other value is an error, as Python raises
// when it iterates it.
func readEntries(state session.State, key string) ([]any, error) {
	v, err := state.Get(key)
	if errors.Is(err, session.ErrStateKeyNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("multimodaltoolresults: get state %q: %w", key, err)
	}
	if l, ok := listOf(v); ok {
		return l, nil
	}
	rv := reflect.ValueOf(v)
	if !rv.IsValid() || rv.IsZero() || rv.Kind() == reflect.String || rv.Kind() == reflect.Map {
		return nil, nil
	}
	return nil, fmt.Errorf("multimodaltoolresults: state key %q holds %T, want a list", key, v)
}

// listOf returns the elements of a slice or array value.
func listOf(v any) ([]any, bool) {
	if l, ok := v.([]any); ok {
		return l, true
	}
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Slice && rv.Kind() != reflect.Array {
		return nil, false
	}
	out := make([]any, rv.Len())
	for i := range out {
		out[i] = rv.Index(i).Interface()
	}
	return out, true
}

// serializePart returns the map google-genai's Part.model_dump(mode="json")
// returns for p, which adk-python stores
// (multimodal_tool_results_plugin.py:130): every field of genai.Part as a
// snake_case key, nil for an unset field, nested messages expanded the same
// way, and bytes as URL-safe padded base64. Go cannot tell an unset string,
// bool or number from its zero value, so a zero one is nil. A part with every
// field zero is written as Part(text="") is, the only meaningful empty part.
func serializePart(p *genai.Part) (map[string]any, error) {
	m, err := converters.ToMapStructure(p)
	if err != nil {
		return nil, err
	}
	if len(m) == 0 {
		m = map[string]any{"text": ""}
	}
	return dumpStruct(reflect.TypeFor[genai.Part](), m), nil
}

// dumpStruct returns the model_dump(mode="json") map of the message type t
// whose set fields are in m: every json-tagged field of t is a key in
// snake_case, and a field missing from m is nil.
func dumpStruct(t reflect.Type, m map[string]any) map[string]any {
	out := make(map[string]any, t.NumField())
	for f := range t.Fields() {
		name, _, _ := strings.Cut(f.Tag.Get("json"), ",")
		if !f.IsExported() || name == "" || name == "-" {
			continue
		}
		out[camelToSnake(name)] = dumpValue(f.Type, m[name])
	}
	return out
}

// dumpValue converts a decoded JSON value v of Go type t: nested messages and
// lists of messages recurse, []byte becomes URL-safe padded base64, and
// every other value (including user maps such as args, response and
// partMetadata, whose keys are user data) is returned unchanged.
func dumpValue(t reflect.Type, v any) any {
	if v == nil {
		return nil
	}
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch v := v.(type) {
	case map[string]any:
		if t.Kind() == reflect.Struct {
			return dumpStruct(t, v)
		}
	case []any:
		if t.Kind() == reflect.Slice {
			out := make([]any, len(v))
			for i, e := range v {
				out[i] = dumpValue(t.Elem(), e)
			}
			return out
		}
	case string:
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
			if raw, err := base64.StdEncoding.DecodeString(v); err == nil {
				return base64.URLEncoding.EncodeToString(raw)
			}
		}
	}
	return v
}

// camelToSnake converts a lowerCamel genai JSON name to snake_case
// ("fileUri" to "file_uri"). genai's names have no acronyms in capitals.
func camelToSnake(s string) string {
	var b strings.Builder
	for i, r := range s {
		if unicode.IsUpper(r) {
			if i > 0 {
				b.WriteByte('_')
			}
			r = unicode.ToLower(r)
		}
		b.WriteRune(r)
	}
	return b.String()
}

// decodePart reads a part stored as serializePart or adk-python writes it,
// with or without its nil keys. It rejects a map with a key that is not a
// field of genai.Part, as pydantic's Part.model_validate does, so a non-part
// entry is never read as an empty part.
func decodePart(m map[string]any) (*genai.Part, bool) {
	if len(m) == 0 {
		return nil, false
	}
	camel := converters.CamelizeKeys(m)
	converters.StdBase64Fields(camel)
	b, err := json.Marshal(camel)
	if err != nil {
		return nil, false
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var p genai.Part
	if err := dec.Decode(&p); err != nil {
		return nil, false
	}
	return &p, true
}

// partEqual reports whether a and b are equal as adk-python's pydantic
// equality sees a stored part and the part it was stored from: equal when
// their stored forms are equal, so a JSON round trip (numbers as float64, an
// empty byte slice as unset) does not make them differ.
func partEqual(a, b *genai.Part) (bool, error) {
	if a == nil || b == nil {
		return a == b, nil
	}
	if reflect.DeepEqual(a, b) {
		return true, nil
	}
	ma, err := serializePart(a)
	if err != nil {
		return false, err
	}
	mb, err := serializePart(b)
	if err != nil {
		return false, err
	}
	return reflect.DeepEqual(ma, mb), nil
}
