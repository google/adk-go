// Copyright 2025 Google LLC
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

package shared

import (
	"fmt"
	"reflect"
	"strings"

	"google.golang.org/genai"
)

// FlattenContentText joins a content's text parts with newlines, and errors on
// a part without text.
func FlattenContentText(content *genai.Content) (string, error) {
	if content == nil {
		return "", nil
	}
	var b strings.Builder
	for _, part := range content.Parts {
		if part == nil {
			continue
		}
		if part.Text == "" {
			return "", fmt.Errorf("non-text system instruction part %T", part)
		}
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(part.Text)
	}
	return b.String(), nil
}

// ReplayedReasoning reports whether part is reasoning carried over from an
// earlier turn that carries nothing else, so dropping it loses nothing: the
// Responses API accepts reasoning back only as an input item referencing the
// id that produced it, an id ADK does not carry, and Chat Completions has no
// field for it, so sent as assistant text it would read as words the model
// never said.
//
// Whether to send a part's text is decided by part.Thought alone, because a
// part can carry both reasoning text and a call, and the call must survive.
func ReplayedReasoning(part *genai.Part) bool {
	if part == nil {
		return false
	}
	// A signature can arrive on a part of its own with the marker unset; there
	// is nowhere to put it in a request to either endpoint.
	if !part.Thought && len(part.ThoughtSignature) == 0 {
		return false
	}
	// Text on a part not marked as a thought is an answer, signature or not.
	if part.Text != "" && !part.Thought {
		return false
	}
	// Marking a call or anything else as a thought must not make it vanish.
	return part.FunctionCall == nil && part.FunctionResponse == nil &&
		UnsupportedPayload(part) == ""
}

// UnsupportedPayload names the first field on part that neither endpoint has a
// way to send, or "" when the part holds nothing beyond what their request
// converters account for.
//
// The test is stated as the absence of anything unaccounted for rather than as
// a list of the fields that disqualify a part, so that a field added to
// genai.Part by a later release is reported here by default instead of leaving
// the request unnoticed.
func UnsupportedPayload(part *genai.Part) string {
	if part == nil {
		return ""
	}
	rest := *part
	rest.Text = ""              // sent, or dropped when it is reasoning
	rest.Thought = false        // the marker deciding which
	rest.ThoughtSignature = nil // no request field on either endpoint carries one
	rest.FunctionCall = nil     // sent as a call
	rest.FunctionResponse = nil // sent as the call's result
	rest.VideoMetadata = nil    // qualifies media carried in another field
	rest.MediaResolution = nil  // likewise
	rest.PartMetadata = nil     // caller bookkeeping, never content

	v := reflect.ValueOf(rest)
	for i := range v.NumField() {
		if !v.Field(i).IsZero() {
			return v.Type().Field(i).Name
		}
	}
	return ""
}
