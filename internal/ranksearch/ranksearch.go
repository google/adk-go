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

// Package ranksearch provides a generic, dependency-free BM25 ranked search over
// named items, with exact selection, regex matching, a relevance floor,
// result limits, and optional name-list advertisement.
package ranksearch

import (
	"fmt"
	"regexp"
	"slices"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	// maxQueryLen caps the search query length to keep ranking cheap and reject
	// pasted-in garbage.
	maxQueryLen = 200
	// selectPrefix marks the exact-name query form: "select:name1,name2".
	selectPrefix = "select:"
	// nameRepeat is how many times an item's name tokens are repeated in its
	// token bag, weighting name matches above description/extra matches.
	nameRepeat = 3
)

// Item is a searchable named entity: a tool or a skill. Tokens is the
// pre-built BM25 token bag (use BuildTokens).
type Item struct {
	Name        string
	Description string
	Tokens      []string
}

// Match is a ranked result: name and (possibly truncated) description.
type Match struct {
	ID          string `json:"id,omitempty"`
	Name        string `json:"name"`
	Description string `json:"description"`
}

// Config tunes a Rank call.
type Config struct {
	// ItemNoun is the singular word used in notes ("tool", "skill"); plurals
	// are formed by appending "s".
	ItemNoun string
	// MaxResults caps fuzzy/regex matches. Zero or negative means no cap. The
	// select: form is never capped — the caller named each item explicitly.
	MaxResults int
	// MinScoreRatio drops BM25 matches scoring below this fraction of the top
	// match's score, trimming the long tail of weak matches.
	MinScoreRatio float64
	// MaxDescLen truncates returned descriptions to this many bytes (rune-safe).
	// Zero or negative leaves descriptions untouched.
	MaxDescLen int
}

// Rank dispatches on the query form and returns ranked matches plus a UX note.
// Items whose name is in alreadyAvailable are excluded — they are already
// visible to the model, so returning them wastes result slots.
//
//   - "select:a,b" loads items by exact name (no cap, reports unknown names).
//   - A query with regex metacharacters matches names/descriptions as a regex.
//     If it matches nothing and contains whitespace, it is treated as prose
//     that happens to contain punctuation and ranked by BM25 instead.
//   - Anything else is ranked by BM25 over each item's token bag.
func Rank(items []Item, query string, alreadyAvailable map[string]bool, cfg Config) ([]Match, string) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, fmt.Sprintf("provide a keyword to search for %ss", cfg.ItemNoun)
	}
	if utf8.RuneCountInString(query) > maxQueryLen {
		return nil, fmt.Sprintf(
			"query too long (max %d characters) — use a shorter keyword or pattern", maxQueryLen,
		)
	}

	if rest, ok := strings.CutPrefix(query, selectPrefix); ok {
		return selectByName(items, rest, alreadyAvailable, cfg)
	}

	eligible := make([]Item, 0, len(items))
	for _, it := range items {
		if !alreadyAvailable[it.Name] {
			eligible = append(eligible, it)
		}
	}

	var all []Match
	var note string
	switch {
	case !looksLikeRegex(query):
		all = bm25Matches(eligible, query, cfg)
	default:
		all = regexMatches(eligible, query, cfg)
		// A pattern rarely contains whitespace, and a question such as "how do
		// I list books?" does. Ranking a real pattern by its words would return
		// items the pattern excluded.
		if len(all) == 0 && strings.ContainsFunc(query, unicode.IsSpace) {
			all = bm25Matches(eligible, query, cfg)
			note = "no matches as a pattern, showing keyword matches"
		}
	}
	if len(all) == 0 {
		return nil, fmt.Sprintf("no %ss matched; try broader keywords or a different term", cfg.ItemNoun)
	}

	if cfg.MaxResults > 0 && len(all) > cfg.MaxResults {
		capped := fmt.Sprintf(
			"showing %d of %d matches — use a more specific keyword to narrow results",
			cfg.MaxResults, len(all),
		)
		if note != "" {
			capped = note + ". " + capped
		}
		return all[:cfg.MaxResults], capped
	}
	return all, note
}

// selectByName loads items by exact name, skipping already-available and
// duplicate names and reporting names that match no item.
func selectByName(items []Item, names string, alreadyAvailable map[string]bool, cfg Config) ([]Match, string) {
	byName := make(map[string]Item, len(items))
	for _, it := range items {
		byName[it.Name] = it
	}

	var matches []Match
	var notFound []string
	seen := make(map[string]bool)
	for raw := range strings.SplitSeq(names, ",") {
		name := strings.TrimSpace(raw)
		if name == "" || alreadyAvailable[name] || seen[name] {
			continue
		}
		it, ok := byName[name]
		if !ok {
			notFound = append(notFound, name)
			continue
		}
		seen[name] = true
		matches = append(matches, Match{Name: it.Name, Description: truncateRunes(it.Description, cfg.MaxDescLen)})
	}

	var note string
	if len(notFound) > 0 {
		note = fmt.Sprintf("%ss not found: %s", cfg.ItemNoun, strings.Join(notFound, ", "))
	}
	if len(matches) == 0 && note == "" {
		note = fmt.Sprintf("no %ss matched; try broader keywords or a different term", cfg.ItemNoun)
	}
	return matches, note
}

// bm25Matches ranks eligible items by BM25 relevance and trims the weak tail
// below MinScoreRatio of the top score.
func bm25Matches(eligible []Item, query string, cfg Config) []Match {
	docs := make([][]string, len(eligible))
	for i, it := range eligible {
		docs[i] = it.Tokens
	}
	ranked := rankBM25(query, docs)
	if len(ranked) == 0 {
		return nil
	}
	floor := ranked[0].score * cfg.MinScoreRatio
	var matches []Match
	for _, r := range ranked {
		if r.score < floor {
			break
		}
		it := eligible[r.index]
		matches = append(matches, Match{Name: it.Name, Description: truncateRunes(it.Description, cfg.MaxDescLen)})
	}
	return matches
}

// regexMatches matches query as a case-insensitive regex against eligible item
// names and descriptions, ranking name matches before description-only matches,
// both alphabetically.
func regexMatches(eligible []Item, query string, cfg Config) []Match {
	matchFn := buildMatchFn(query)
	type hit struct {
		match     Match
		nameMatch bool
	}
	var hits []hit
	for _, it := range eligible {
		nameHit := matchFn(it.Name)
		if nameHit || matchFn(it.Description) {
			hits = append(hits, hit{
				match:     Match{Name: it.Name, Description: truncateRunes(it.Description, cfg.MaxDescLen)},
				nameMatch: nameHit,
			})
		}
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].nameMatch != hits[j].nameMatch {
			return hits[i].nameMatch
		}
		return hits[i].match.Name < hits[j].match.Name
	})
	matches := make([]Match, len(hits))
	for i, h := range hits {
		matches[i] = h.match
	}
	return matches
}

// BuildTokens builds an item's BM25 token bag: its name (repeated to weight name
// matches highest), its description, and any extra strings (e.g. argument names
// and descriptions). Each input is tokenized.
func BuildTokens(name, description string, extra ...string) []string {
	nameToks := tokenize(name)
	toks := make([]string, 0, len(nameToks)*nameRepeat+8)
	for range nameRepeat {
		toks = append(toks, nameToks...)
	}
	toks = append(toks, tokenize(description)...)
	for _, e := range extra {
		toks = append(toks, tokenize(e)...)
	}
	return toks
}

// DescribeSearch appends the sorted item names to a search tool's base
// description so the model knows what it can search for (and select: by name).
// The list is sorted for a stable, cache-friendly declaration.
func DescribeSearch(baseDescription string, itemNames []string) string {
	if len(itemNames) == 0 {
		return baseDescription
	}
	names := slices.Clone(itemNames)
	slices.Sort(names)
	return baseDescription +
		"\n\nAvailable to load (not yet active — search or select: by name): " +
		strings.Join(names, ", ")
}

// Tokenize exposes the package tokenizer for callers that need to pre-process
// text the same way the ranker does.
func Tokenize(s string) []string { return tokenize(s) }

// looksLikeRegex reports whether query contains regex metacharacters, in which
// case it is treated as a name pattern rather than a natural-language query.
func looksLikeRegex(query string) bool {
	return strings.ContainsAny(query, `.*+?[]()|^$\{}`)
}

// buildMatchFn returns a case-insensitive regexp matcher. If query is a valid
// regexp, it is used as-is; otherwise it falls back to a literal substring match.
func buildMatchFn(query string) func(string) bool {
	re, err := regexp.Compile("(?i)" + query)
	if err != nil {
		re = regexp.MustCompile("(?i)" + regexp.QuoteMeta(query))
	}
	return re.MatchString
}

// truncateRunes caps s at maxBytes without splitting a multi-byte UTF-8 rune. A
// non-positive maxBytes leaves s untouched.
func truncateRunes(s string, maxBytes int) string {
	if maxBytes <= 0 || len(s) <= maxBytes {
		return s
	}
	end := maxBytes
	for end > 0 && !utf8.RuneStart(s[end]) {
		end--
	}
	return s[:end]
}
