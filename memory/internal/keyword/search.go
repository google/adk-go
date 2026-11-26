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

// Package keyword implements the keyword matching used by local memory services.
package keyword

import (
	"iter"
	"strings"
	"unicode"

	"golang.org/x/text/cases"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/norm"
	"golang.org/x/text/unicode/runenames"
	"google.golang.org/genai"
)

// MaxResults is the maximum number of entries returned by a search.
const MaxResults = 10

// VisibleContent copies the content without nil parts or model reasoning and
// returns the joined text of the remaining parts. The parts are not cloned.
func VisibleContent(content *genai.Content) (*genai.Content, string) {
	var texts []string
	var visibleParts []*genai.Part
	for _, part := range content.Parts {
		if part == nil || part.Thought {
			continue
		}
		visibleParts = append(visibleParts, part)
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}

	visible := *content
	visible.Parts = visibleParts
	return &visible, strings.Join(texts, " ")
}

// Document holds the normalized text and words of an event.
type Document struct {
	words     map[string]struct{}
	textLower string
}

// NewDocument prepares searchable text for matching.
func NewDocument(text string) Document {
	return Document{
		words:     extractSearchableWords(text),
		textLower: cases.Lower(language.Und).String(norm.NFC.String(text)),
	}
}

// Query maps search words to whether they also allow substring matching.
type Query map[string]bool

// NewQuery prepares a query using the same tokenization as NewDocument.
func NewQuery(text string) Query {
	// The query is split into words like the stored text, so a Latin word
	// inside Japanese or Chinese query text matches on its own. A true value
	// allows substring matching, which only a whole non-ASCII query word gets:
	// a run split out of one, such as と from Pythonと, would match nearly any
	// Japanese text.
	queryWords := make(Query)
	for word := range extractSearchableWords(text) {
		queryWords[word] = false
	}
	for word := range extractWordsLower(text) {
		queryWords[word] = hasNonASCII(word)
	}

	return queryWords
}

// Score counts how many distinct query words match the document.
func (q Query) Score(d Document) int {
	if len(d.words) == 0 {
		return 0
	}
	score := 0
	for word, allowSubstring := range q {
		if _, ok := d.words[word]; ok || (allowSubstring && strings.Contains(d.textLower, word)) {
			score++
		}
	}
	return score
}

// extractWordsLower returns the lowercased \w+ tokens of the NFC form of text,
// as adk-python's _extract_words_lower does. Combining marks are not word
// characters, so after NFC a mark that composes nowhere separates words.
func extractWordsLower(text string) map[string]struct{} {
	// A Caser may be stateful, so each call builds its own.
	lower := cases.Lower(language.Und)
	res := make(map[string]struct{})
	for word := range tokens(text) {
		res[lower.String(word)] = struct{}{}
	}
	return res
}

// tokens returns the \w+ tokens of the NFC form of text, in their original
// case.
func tokens(text string) iter.Seq[string] {
	return strings.FieldsFuncSeq(norm.NFC.String(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r) && r != '_'
	})
}

// extractSearchableWords returns the words a query or an event is matched on,
// as adk-python's _extract_searchable_words does: the tokens of
// extractWordsLower plus, for each token with non-ASCII characters, its runs
// of Latin and of non-Latin characters. Japanese and Chinese are written
// without spaces, so in 私はPythonを使う the run python becomes a word of its
// own.
//
// A token is split before it is lowercased. Lowercasing turns İ into i and
// U+0307, which is not Latin, so splitting afterwards would make words of
// that i and of the rest of the run, such as stanbul.
func extractSearchableWords(text string) map[string]struct{} {
	lower := cases.Lower(language.Und)
	words := make(map[string]struct{})
	for word := range tokens(text) {
		words[lower.String(word)] = struct{}{}
		if !hasNonASCII(word) {
			continue
		}
		start, runLatin := 0, false
		for i, r := range word {
			latin := isLatin(r)
			if i > 0 && latin != runLatin {
				words[lower.String(word[start:i])] = struct{}{}
				start = i
			}
			runLatin = latin
		}
		words[lower.String(word[start:])] = struct{}{}
	}
	return words
}

// isLatin reports whether r belongs to a Latin run, as adk-python's _is_latin
// does: an ASCII word character, or a character whose Unicode name starts
// with LATIN. The name rule leaves out ª and º, which unicode.Latin includes.
func isLatin(r rune) bool {
	if r <= unicode.MaxASCII {
		return unicode.IsLetter(r) || unicode.IsNumber(r) || r == '_'
	}
	return strings.HasPrefix(runenames.Name(r), "LATIN")
}

func hasNonASCII(s string) bool {
	return strings.IndexFunc(s, func(r rune) bool { return r > unicode.MaxASCII }) >= 0
}
