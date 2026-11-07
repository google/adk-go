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

package openapitoolset

import (
	"regexp"
	"strings"
)

// maxToolNameLength is adk-python's _MAX_TOOL_NAME_LENGTH: Gemini limits a
// function name to 64 characters.
const maxToolNameLength = 60

var (
	nonAlphanumeric = regexp.MustCompile(`[^a-zA-Z0-9]+`)
	lowerUpper      = regexp.MustCompile(`([a-z0-9])([A-Z])`)
	acronymWord     = regexp.MustCompile(`([A-Z]+)([A-Z][a-z])`)
	underscores     = regexp.MustCompile(`_+`)
)

// snakeCase is adk-python's _to_snake_case (tools/_gemini_schema_util.py):
// "calendar.calendars.get" becomes "calendar_calendars_get", "createUser"
// becomes "create_user" and "HTTPResponse" becomes "http_response".
func snakeCase(s string) string {
	s = nonAlphanumeric.ReplaceAllString(s, "_")
	s = lowerUpper.ReplaceAllString(s, "${1}_${2}")
	s = acronymWord.ReplaceAllString(s, "${1}_${2}")
	s = strings.ToLower(s)
	s = underscores.ReplaceAllString(s, "_")
	return strings.Trim(s, "_")
}

// pythonKeywords is Python's keyword.kwlist. adk-python names the parameters
// of a generated Python function, so it renames those that are keywords, and
// the names reach the model in the function declaration.
var pythonKeywords = map[string]bool{
	"False": true, "None": true, "True": true, "and": true, "as": true,
	"assert": true, "async": true, "await": true, "break": true, "class": true,
	"continue": true, "def": true, "del": true, "elif": true, "else": true,
	"except": true, "finally": true, "for": true, "from": true, "global": true,
	"if": true, "import": true, "in": true, "is": true, "lambda": true,
	"nonlocal": true, "not": true, "or": true, "pass": true, "raise": true,
	"return": true, "try": true, "while": true, "with": true, "yield": true,
}

// renameKeyword is adk-python's rename_python_keywords: "in" becomes
// "param_in", and anything that is not a Python keyword is returned as is.
func renameKeyword(s string) string {
	if pythonKeywords[s] {
		return "param_" + s
	}
	return s
}

// truncate returns the first n characters of s, counted as Python counts
// them, in code points.
func truncate(s string, n int) string {
	i := 0
	for pos := range s {
		if i == n {
			return s[:pos]
		}
		i++
	}
	return s
}
