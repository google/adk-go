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

package toolcallintegrity

import (
	"google.golang.org/adk/v2/internal/eventmetadata"
	"google.golang.org/adk/v2/internal/sessionutils"
	"google.golang.org/adk/v2/internal/typeutil"
	"google.golang.org/adk/v2/session"
)

// stampKey is the custom metadata key of an event's stamp map, from call ID
// to stamp (adk-python: _HMAC_META_KEY).
const stampKey = eventmetadata.InternalPrefix + "fc_hmac"

// PrepareRestoredEvent returns a deep copy of event for appending to another
// session: its custom metadata keeps every key that does not start with
// "__adk_internal_", loses every key that does (stamps from the source
// session included), and gains the ADK-internal restored marker. The plugin
// does not verify a restored event before a run, and never lets its function
// calls run. event is not modified. A nil event gives nil.
func PrepareRestoredEvent(event *session.Event) *session.Event {
	c := sessionutils.CloneValue(event)
	eventmetadata.MarkRestored(c)
	return c
}

// stampsOf returns ev's stamp map, from call ID to stamp. An absent or nil
// value means no stamps. It accepts a map[string]any of ASCII strings, the
// form onEvent writes and a serializing store returns, and a
// map[string]string of ASCII strings; any other value is malformed
// (adk-python: _stamps and _is_stamp_map).
func stampsOf(ev *session.Event) (map[string]string, error) {
	switch v := ev.CustomMetadata[stampKey].(type) {
	case nil:
		return nil, nil
	case map[string]any:
		stamps := make(map[string]string, len(v))
		for id, s := range v {
			stamp, ok := s.(string)
			if !ok || !isASCII(stamp) {
				return nil, malformedStamp(ev)
			}
			stamps[id] = stamp
		}
		return stamps, nil
	case map[string]string:
		for _, stamp := range v {
			if !isASCII(stamp) {
				return nil, malformedStamp(ev)
			}
		}
		return v, nil
	}
	return nil, malformedStamp(ev)
}

func malformedStamp(ev *session.Event) error {
	return integrityErrorf("event '%s' has a malformed integrity stamp", ev.ID)
}

func isASCII(s string) bool {
	for i := range len(s) {
		if s[i] >= 0x80 {
			return false
		}
	}
	return true
}

// isRestored reports whether ev carries a restored marker that is true in
// Python's sense, as adk-python's _is_restored reads it, so a marker a store
// hands back as 1.0 still counts.
func isRestored(ev *session.Event) bool {
	return typeutil.IsTruthy(ev.CustomMetadata[eventmetadata.RestoredEventKey])
}
