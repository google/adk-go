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

package sqlitememory

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/cases"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/norm"
	"golang.org/x/text/unicode/runenames"
	"google.golang.org/genai"
)

func eventSearchText(content *genai.Content, maxBytes int) (string, bool) {
	var parts []string
	for _, part := range content.Parts {
		if part == nil || part.Thought {
			continue
		}
		if text := strings.TrimFunc(strings.ReplaceAll(part.Text, "\n", " "), isSpace); text != "" {
			parts = append(parts, text)
		}
	}
	if len(parts) == 0 {
		return "", false
	}
	text := norm.NFC.String(strings.Join(parts, " "))
	// Split before lowercasing, which can introduce a combining mark (e.g. İ).
	text = cases.Lower(language.Und).String(splitScriptRuns(text))
	if len(text) <= maxBytes {
		return text, true
	}
	for !utf8.RuneStart(text[maxBytes]) {
		maxBytes--
	}
	return text[:maxBytes], true
}

func splitScriptRuns(text string) string {
	if isASCII(text) {
		return text
	}
	var out []string
	for _, word := range strings.FieldsFunc(text, isSpace) {
		if isASCII(word) {
			out = append(out, word)
			continue
		}
		start := 0
		var latin bool
		for i, r := range word {
			next := isLatin(r)
			if i > 0 && next != latin {
				out = append(out, word[start:i])
				start = i
			}
			latin = next
		}
		out = append(out, word[start:])
	}
	return strings.Join(out, " ")
}

func isASCII(text string) bool {
	for i := range len(text) {
		if text[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

func isLatin(r rune) bool {
	if r < utf8.RuneSelf {
		return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_'
	}
	return strings.HasPrefix(runenames.Name(r), "LATIN")
}

func isSpace(r rune) bool {
	// Python also treats the four ASCII separators as whitespace.
	return unicode.IsSpace(r) || (r >= '\x1c' && r <= '\x1f')
}
