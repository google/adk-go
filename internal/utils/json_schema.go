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

package utils

import (
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
)

const codeFence = "```"

// StripJSONCodeFence removes a markdown code fence that wraps the whole of
// text, and returns text trimmed of surrounding space otherwise. It ports
// adk-python's _strip_json_code_fence (utils/_schema_utils.py).
//
// The fence is found with string operations rather than a regular expression,
// so the cost stays linear in the length of text, which comes from a model.
func StripJSONCodeFence(text string) string {
	stripped := strings.TrimSpace(text)
	if len(stripped) < 2*len(codeFence) || !strings.HasPrefix(stripped, codeFence) || !strings.HasSuffix(stripped, codeFence) {
		return stripped
	}
	inner := stripped[len(codeFence) : len(stripped)-len(codeFence)]
	return strings.TrimSpace(trimFenceLanguage(inner))
}

// trimFenceLanguage removes the language tag that may open a fence's body
// ([A-Za-z0-9_]*), the spaces and tabs after it, and one newline.
func trimFenceLanguage(s string) string {
	s = strings.TrimLeftFunc(s, func(r rune) bool {
		return r == '_' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9'
	})
	s = strings.TrimLeft(s, " \t")
	return strings.TrimPrefix(s, "\n")
}

// WrapJSONSchema returns an object schema with the single required property
// param whose schema is s, and moves the root $defs and definitions of s to
// the wrapper, so "#/$defs/X" references still resolve from the root of the
// parameters. s is not modified.
func WrapJSONSchema(param string, s *jsonschema.Schema) *jsonschema.Schema {
	inner := *s
	inner.Defs = nil
	inner.Definitions = nil
	return &jsonschema.Schema{
		Type:        "object",
		Properties:  map[string]*jsonschema.Schema{param: &inner},
		Required:    []string{param},
		Defs:        s.Defs,
		Definitions: s.Definitions,
	}
}
