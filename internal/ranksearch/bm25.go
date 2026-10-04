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

package ranksearch

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

const (
	bm25K1 = 1.5  // term-frequency saturation
	bm25B  = 0.75 // document-length normalization
)

// scoredDoc pairs a document index with its BM25 relevance score.
type scoredDoc struct {
	index int
	score float64
}

// rankBM25 scores each document against the query using Okapi BM25 and returns
// the indices sorted by descending score. Documents with no query-term overlap
// (score 0) are omitted. docs are pre-tokenized token bags.
//
// ponytail: IDF and length stats are rebuilt per call; add a catalog-keyed
// cache if profiling shows this is a bottleneck.
func rankBM25(query string, docs [][]string) []scoredDoc {
	queryTerms := tokenize(query)
	if len(queryTerms) == 0 || len(docs) == 0 {
		return nil
	}

	df := make(map[string]int)
	totalLen := 0
	for _, doc := range docs {
		totalLen += len(doc)
		seen := make(map[string]bool, len(doc))
		for _, tok := range doc {
			if !seen[tok] {
				df[tok]++
				seen[tok] = true
			}
		}
	}
	avgLen := float64(totalLen) / float64(len(docs))
	n := float64(len(docs))

	var scored []scoredDoc
	for i, doc := range docs {
		tf := make(map[string]int, len(doc))
		for _, tok := range doc {
			tf[tok]++
		}
		var score float64
		for _, term := range queryTerms {
			f := float64(tf[term])
			if f == 0 {
				continue
			}
			idf := math.Log(1 + (n-float64(df[term])+0.5)/(float64(df[term])+0.5))
			norm := f * (bm25K1 + 1) / (f + bm25K1*(1-bm25B+bm25B*float64(len(doc))/avgLen))
			score += idf * norm
		}
		if score > 0 {
			scored = append(scored, scoredDoc{index: i, score: score})
		}
	}
	sort.SliceStable(scored, func(a, b int) bool { return scored[a].score > scored[b].score })
	return scored
}

// tokenize lowercases s, splits on non-alphanumeric boundaries and camelCase
// humps (so "list_books" and "getBook" both split into their words), and
// applies a light plural stemmer so "books" and "book" collide.
func tokenize(s string) []string {
	var tokens []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			tokens = append(tokens, stem(strings.ToLower(cur.String())))
			cur.Reset()
		}
	}
	var prev rune
	for _, r := range s {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			if unicode.IsUpper(r) && unicode.IsLower(prev) {
				flush() // camelCase boundary
			}
			cur.WriteRune(r)
		default:
			flush()
		}
		prev = r
	}
	flush()
	return tokens
}

// stem strips a trailing plural suffix so singular and plural forms of a word
// match. It is intentionally minimal — not a full Porter stemmer. It covers the
// common "-ies", sibilant "-es", and "-s" plurals while leaving non-plural
// endings ("-ss", "-us") and short words untouched.
func stem(tok string) string {
	switch {
	case len(tok) > 4 && strings.HasSuffix(tok, "ies"):
		return tok[:len(tok)-3] + "y" // policies -> policy, activities -> activity
	case len(tok) > 4 && endsWithAny(tok, "ses", "xes", "zes", "ches", "shes"):
		return tok[:len(tok)-2] // processes -> process, statuses -> status, boxes -> box
	case strings.HasSuffix(tok, "ss") || strings.HasSuffix(tok, "us"):
		return tok // process, address, status, nexus — not plurals
	case len(tok) > 3 && strings.HasSuffix(tok, "s"):
		return tok[:len(tok)-1] // books -> book, names -> name
	default:
		return tok
	}
}

// endsWithAny reports whether s ends with any of the given suffixes.
func endsWithAny(s string, suffixes ...string) bool {
	for _, suf := range suffixes {
		if strings.HasSuffix(s, suf) {
			return true
		}
	}
	return false
}
