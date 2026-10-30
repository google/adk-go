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

package agentengine

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"
	"unicode"

	"google.golang.org/genai"
)

// ErrRemote marks a failure the agent reported in the stream itself, as
// opposed to a transport or decoding error. Its message is the server's.
var ErrRemote = errors.New("agentengine: remote error")

// Event is one line of a stream-query response. It models the fields every
// ADK runtime emits; Raw holds the rest.
type Event struct {
	Author string
	// Content is nil for lines without a content object, which some agent
	// frameworks emit.
	Content *genai.Content
	// Partial marks a streamed chunk of a reply. Partial chunks are followed
	// by one aggregated non-partial event carrying the full content, so
	// consumers that want each reply once skip partial events.
	Partial bool
	// ErrorCode and ErrorMessage are set when the model call behind this
	// event failed (a safety block, a malformed function call). Such an event
	// is a regular line, not a remote error, and usually has no Content.
	ErrorCode    string
	ErrorMessage string
	// Raw is the whole decoded line, for fields this type does not model.
	Raw map[string]any
}

// decodeJSONL reassembles JSONL across chunk boundaries and decodes each line.
// The server writes one JSON object per line and reports failure as a line
// {"error": "..."}, which becomes the iterator's ErrRemote error and ends the
// stream. A final line without a trailing newline is still decoded.
func decodeJSONL(chunks iter.Seq2[[]byte, error]) iter.Seq2[*Event, error] {
	return func(yield func(*Event, error) bool) {
		var buf []byte
		emit := func(line []byte) bool {
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				return true
			}
			ev, err := decodeLine(line)
			if err != nil {
				yield(nil, err)
				return false
			}
			return yield(ev, nil)
		}
		for chunk, err := range chunks {
			if err != nil {
				yield(nil, err)
				return
			}
			// Only the new bytes can hold a newline; the retained partial
			// line was already scanned.
			scanned := len(buf)
			buf = append(buf, chunk...)
			for {
				i := bytes.IndexByte(buf[scanned:], '\n')
				if i < 0 {
					break
				}
				line := buf[:scanned+i]
				buf = buf[scanned+i+1:]
				scanned = 0
				if !emit(line) {
					return
				}
			}
		}
		emit(buf)
	}
}

func decodeLine(line []byte) (*Event, error) {
	var raw map[string]any
	if err := json.Unmarshal(line, &raw); err != nil {
		return nil, fmt.Errorf("agentengine: decoding stream line: %w", err)
	}
	if raw == nil {
		return nil, errors.New("agentengine: decoding stream line: not a JSON object")
	}
	if msg, ok := raw["error"]; ok {
		return nil, fmt.Errorf("%w: %v", ErrRemote, msg)
	}
	ev := &Event{Raw: raw}
	ev.Author, _ = raw["author"].(string)
	ev.Partial, _ = raw["partial"].(bool)
	ev.ErrorCode, _ = raw["error_code"].(string)
	ev.ErrorMessage, _ = raw["error_message"].(string)
	if c, ok := raw["content"]; ok && c != nil {
		// genai's JSON tags are camelCase, but servers emit snake_case keys,
		// so normalize before decoding.
		c = camelizeKeys(c)
		stdBase64Fields(c)
		b, err := json.Marshal(c)
		if err != nil {
			return nil, fmt.Errorf("agentengine: re-encoding content: %w", err)
		}
		var content genai.Content
		if err := json.Unmarshal(b, &content); err != nil {
			return nil, fmt.Errorf("agentengine: decoding content: %w", err)
		}
		ev.Content = &content
	}
	return ev, nil
}

// userKeyFields hold map-valued genai fields whose keys are user data (tool
// arguments and results, part metadata) and must reach the caller unchanged.
var userKeyFields = map[string]bool{"args": true, "response": true, "partMetadata": true}

// camelizeKeys renames snake_case object keys to the lowerCamelCase names in
// genai's JSON tags. Both the adk-go server (helper.ConvertSnake) and Python
// agents (model_dump_json without by_alias) emit snake_case keys. camelCase
// input passes through unchanged. Values of userKeyFields are copied without
// descending into them.
func camelizeKeys(v any) any {
	switch v := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(v))
		for k, val := range v {
			k = snakeToCamel(k)
			if userKeyFields[k] {
				out[k] = val
				continue
			}
			out[k] = camelizeKeys(val)
		}
		return out
	case []any:
		out := make([]any, len(v))
		for i, val := range v {
			out[i] = camelizeKeys(val)
		}
		return out
	default:
		return v
	}
}

func snakeToCamel(s string) string {
	if !strings.Contains(s, "_") {
		return s
	}
	var b strings.Builder
	upper := false
	for _, r := range s {
		if r == '_' {
			upper = b.Len() > 0
			continue
		}
		if upper {
			r = unicode.ToUpper(r)
			upper = false
		}
		b.WriteRune(r)
	}
	return b.String()
}

// stdBase64Fields rewrites every bytes field under content to standard
// base64 in place: thoughtSignature, and data inside an inlineData object,
// at any depth (a function response carries its own media parts). Pydantic
// emits bytes as URL-safe base64, while encoding/json decodes []byte only
// from standard base64, so a Python agent's value would otherwise fail the
// whole line. User-data maps are skipped: they are shared with Event.Raw.
// content must be a value returned by camelizeKeys, which the caller owns.
func stdBase64Fields(v any) {
	switch v := v.(type) {
	case map[string]any:
		for k, val := range v {
			if userKeyFields[k] {
				continue
			}
			switch k {
			case "thoughtSignature":
				if s, ok := val.(string); ok {
					v[k] = stdBase64(s)
				}
			case "inlineData":
				if blob, ok := val.(map[string]any); ok {
					if s, ok := blob["data"].(string); ok {
						blob["data"] = stdBase64(s)
					}
				}
			}
			stdBase64Fields(val)
		}
	case []any:
		for _, val := range v {
			stdBase64Fields(val)
		}
	}
}

// stdBase64 converts URL-safe base64 to the standard alphabet and restores
// missing padding. Standard padded input is returned unchanged.
func stdBase64(s string) string {
	if strings.ContainsAny(s, "-_") {
		s = strings.NewReplacer("-", "+", "_", "/").Replace(s)
	}
	if r := len(s) % 4; r != 0 {
		s += strings.Repeat("=", 4-r)
	}
	return s
}
