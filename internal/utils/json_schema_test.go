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

package utils_test

import (
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/internal/utils"
)

// TestStripJSONCodeFence fails if a fence is left in place, if the language
// tag or the inner space is kept, or if text that is not wholly fenced comes
// back as anything other than itself, trimmed.
func TestStripJSONCodeFence(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "plain JSON", in: `{"a": 1}`, want: `{"a": 1}`},
		{name: "json fence", in: "```json\n{\"a\": 1}\n```", want: `{"a": 1}`},
		{name: "bare fence", in: "```\n[1]\n```", want: "[1]"},
		{name: "space after the fence", in: "```   \n{\"a\": 1}\n```", want: `{"a": 1}`},
		{name: "tag and body on one line", in: "```json  {\"a\": 1}```", want: `{"a": 1}`},
		{name: "space around the fence", in: "  \n```json\n{\"a\": 1}\n```\n ", want: `{"a": 1}`},
		{name: "unclosed fence", in: " ```json\n{\"a\": 1} ", want: "```json\n{\"a\": 1}"},
		{name: "fence marker only", in: "```", want: "```"},
		{name: "empty", in: "", want: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := utils.StripJSONCodeFence(tt.in); got != tt.want {
				t.Errorf("StripJSONCodeFence(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

// TestStripJSONCodeFence_UnclosedWhitespaceIsLinear fails if the fence is
// matched by a backtracking pattern whose cost grows faster than the input on
// an unclosed fence with a long whitespace body (adk-python's ReDoS case).
func TestStripJSONCodeFence_UnclosedWhitespaceIsLinear(t *testing.T) {
	in := "```json\n" + strings.Repeat(" ", 200000) + "x"
	start := time.Now()
	got := utils.StripJSONCodeFence(in)
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("StripJSONCodeFence took %v, want under 1s", elapsed)
	}
	if got != in {
		t.Errorf("StripJSONCodeFence changed an unclosed fence (len %d, want len %d)", len(got), len(in))
	}
}

// TestWrapJSONSchema fails if the wrapper does not hold s under param, does
// not require param, leaves $defs or definitions on the inner schema instead
// of the root that "#/$defs/X" resolves against, or writes to s.
func TestWrapJSONSchema(t *testing.T) {
	newSchema := func() *jsonschema.Schema {
		return &jsonschema.Schema{
			Type:        "array",
			Items:       &jsonschema.Schema{Ref: "#/$defs/Item"},
			Defs:        map[string]*jsonschema.Schema{"Item": {Type: "object"}},
			Definitions: map[string]*jsonschema.Schema{"Old": {Type: "string"}},
		}
	}
	s := newSchema()

	got := utils.WrapJSONSchema("result", s)

	want := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"result": {Type: "array", Items: &jsonschema.Schema{Ref: "#/$defs/Item"}},
		},
		Required:    []string{"result"},
		Defs:        map[string]*jsonschema.Schema{"Item": {Type: "object"}},
		Definitions: map[string]*jsonschema.Schema{"Old": {Type: "string"}},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("WrapJSONSchema() mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(newSchema(), s); diff != "" {
		t.Errorf("WrapJSONSchema() modified its argument (-want +got):\n%s", diff)
	}
}
