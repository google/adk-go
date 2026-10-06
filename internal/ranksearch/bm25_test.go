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
	"slices"
	"testing"
)

// TestRankBM25_RanksHigherOverlapFirst verifies the doc with the most matches of
// the query term ranks first and a doc with no overlap is omitted entirely.
func TestRankBM25_RanksHigherOverlapFirst(t *testing.T) {
	docs := [][]string{
		tokenize("list books list books list books list all books"),
		tokenize("get author"),
		tokenize("create book"),
	}

	ranked := rankBM25("books", docs)
	if len(ranked) == 0 {
		t.Fatal("rankBM25() returned no results")
	}
	if ranked[0].index != 0 {
		t.Errorf("top index = %d, want 0 (doc with the most 'book' matches)", ranked[0].index)
	}
	for _, r := range ranked {
		if r.index == 1 {
			t.Error("doc with no overlap must be omitted")
		}
	}
}

// TestRankBM25_DownweightsCommonTerms verifies IDF down-weights a term present in
// every doc ("list") so a rare term ("publisher") dominates ranking — the failure
// mode that plain substring matching suffers from.
func TestRankBM25_DownweightsCommonTerms(t *testing.T) {
	docs := [][]string{
		tokenize("list books"),
		tokenize("list authors"),
		tokenize("list publishers"),
	}

	ranked := rankBM25("list publishers", docs)
	if len(ranked) == 0 {
		t.Fatal("rankBM25() returned no results")
	}
	if ranked[0].index != 2 {
		t.Errorf("top index = %d, want 2 (rare term 'publisher' must outweigh the common term 'list')", ranked[0].index)
	}
}

// TestTokenize_SplitsSnakeAndCamelCase verifies the tokenizer splits both
// snake_case and camelCase into their constituent words.
func TestTokenize_SplitsSnakeAndCamelCase(t *testing.T) {
	for in, want := range map[string][]string{
		"list_books": {"list", "book"},
		"getBook":    {"get", "book"},
	} {
		if got := tokenize(in); !slices.Equal(got, want) {
			t.Errorf("tokenize(%q) = %v, want %v", in, got, want)
		}
	}
}

// TestTokenize_NoEmptyTokens checks that runs of separators produce no empty
// token. An empty token in both a query and a tool description would make the
// tool match without sharing a word with the query.
func TestTokenize_NoEmptyTokens(t *testing.T) {
	for in, want := range map[string][]string{
		"weather, forecast": {"weather", "forecast"},
		"an author. Fast.":  {"an", "author", "fast"},
		"..x..":             {"x"},
	} {
		if got := tokenize(in); !slices.Equal(got, want) {
			t.Errorf("tokenize(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestStem_PluralCollapses verifies singular and plural forms tokenize equally so
// "book" matches "books", including sibilant "-es" plurals.
func TestStem_PluralCollapses(t *testing.T) {
	for singular, plural := range map[string]string{
		"book":     "books",
		"activity": "activities",
		"name":     "names",
		"process":  "processes",
		"status":   "statuses",
		"database": "databases",
		"case":     "cases",
		"cache":    "caches",
		"size":     "sizes",
		"match":    "matches",
		"box":      "boxes",
		"use":      "uses",
		"axe":      "axes",
	} {
		if got, want := tokenize(plural), tokenize(singular); !slices.Equal(got, want) {
			t.Errorf("tokenize(%q) = %v, want %v", plural, got, want)
		}
	}
}

// TestStem_LeavesNonPluralsIntact verifies "-us" and "-ss" words are not
// over-stripped (the trailing-"s" rule must not fire on them).
func TestStem_LeavesNonPluralsIntact(t *testing.T) {
	for _, word := range []string{"status", "nexus", "process", "address"} {
		if got := tokenize(word); !slices.Equal(got, []string{word}) {
			t.Errorf("tokenize(%q) = %v, want [%s]", word, got, word)
		}
	}
}
