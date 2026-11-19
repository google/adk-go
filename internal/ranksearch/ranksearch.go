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
	"bytes"
	"errors"
	"fmt"
	"regexp"
	"regexp/syntax"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	// maxQueryLen caps the query length, which bounds the BM25 query terms and
	// how long a regex takes to parse, and rejects pasted-in garbage. Parsing
	// is slow for case-insensitive character classes over wide ranges of runes.
	// The slowest query found, a class of 64 such ranges after its own (?i),
	// which gets it folded twice (see compileQuery), makes a search take about
	// 0.35 s.
	maxQueryLen = 200
	// maxPatternSize caps the estimated compiled size of a regex query. The
	// matcher tracks at most one state per instruction for each byte of input,
	// so this roughly bounds how many states a byte can cost. Only roughly: the
	// estimate comes from a parse without case folding (see compileQuery), and
	// a pattern that turns folding off with (?-i) can compile to many times it,
	// 19 times for one of 199 runes. It does not bound parse time (see
	// maxQueryLen). A counted repeat such as .{0,999} multiplies the size: 24 of
	// them fit in maxQueryLen and compile to about 48,000 instructions. A larger
	// pattern is not run.
	maxPatternSize = 256
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
// visible to the model, so returning them wastes result slots. When every
// match is already available, the note names those items, at most MaxResults
// of them for a ranked query, instead of reporting that nothing matched.
//
//   - "select:a,b" loads items by exact name (no cap, reports unknown names).
//   - A query with regex metacharacters is matched against names, which keeps
//     the matcher's input short: as a regex if it is a pattern, and as a
//     literal substring if it is punctuated keywords or prose (see
//     compileQuery). If no new name matches, punctuated keywords and prose are
//     ranked by BM25 instead, and a pattern reports a match among the already
//     available names or says that patterns match names only. A pattern that
//     is larger than maxPatternSize, or does not parse, is not run, and unless
//     the query is prose, the note says why.
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

	q := compileQuery(query)
	if q.notRun != "" {
		return nil, q.notRun
	}

	var eligible, available []Item
	for _, it := range items {
		if alreadyAvailable[it.Name] {
			available = append(available, it)
		} else {
			eligible = append(eligible, it)
		}
	}

	var all []Match
	var note string
	if q.hasMeta {
		all = q.nameMatches(eligible, cfg)
		if len(all) == 0 && !q.keywordFallback {
			if had := q.nameMatches(available, cfg); len(had) > 0 {
				return nil, alreadyAvailableNote(had, cfg)
			}
			return nil, fmt.Sprintf("no %s names match this pattern — patterns match names only, "+
				"so search by keywords to match descriptions", cfg.ItemNoun)
		}
		if len(all) == 0 {
			all, note = bm25Matches(eligible, query, cfg),
				fmt.Sprintf("no new %s names contain the query, showing keyword matches", cfg.ItemNoun)
		}
	} else {
		all = bm25Matches(eligible, query, cfg)
	}
	if len(all) == 0 {
		had := q.nameMatches(available, cfg)
		if len(had) == 0 {
			had = bm25Matches(available, query, cfg)
		}
		if len(had) > 0 {
			return nil, alreadyAvailableNote(had, cfg)
		}
		if q.regexHint {
			return nil, fmt.Sprintf("no %ss matched as keywords — to match %s names as a pattern, "+
				"add .* or a character class", cfg.ItemNoun, cfg.ItemNoun)
		}
		return nil, noMatchNote(cfg)
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

// compiledQuery is a query prepared once per Rank call, since preparing a
// regex can cost far more than matching it.
type compiledQuery struct {
	// hasMeta is set for a query with regex metacharacters, which is matched
	// against names before anything else.
	hasMeta bool
	// matchName matches an item name: as a regex for a pattern, and as a
	// literal substring for punctuated keywords or prose. It is nil for a
	// query without regex metacharacters and for a pattern that is not run.
	matchName func(string) bool
	// keywordFallback ranks the query by BM25 when no new name contains it,
	// because it is punctuated keywords or prose rather than a pattern.
	keywordFallback bool
	// notRun is the note explaining why a pattern was not run, if it was not.
	// It is never set for prose, which is ranked by keywords instead.
	notRun string
	// regexHint is set for punctuated keywords that would also parse as a
	// regex, such as fore.ast, so that finding nothing says how to write one.
	regexHint bool
}

// compileQuery prepares a query with regex metacharacters. It is a pattern if
// it contains one of patternMarkers or if its parse tree uses a repeat (see
// usesRepeat), judging a query that does not parse as it would be once fixed
// (see repeatsOnceParsed). A pattern is matched against names as a
// case-insensitive regex. A pattern larger than maxPatternSize, or one that
// does not parse, is not run. Any other query is punctuated keywords, and so is
// any query with whitespace, which marks prose. It is matched against names as
// a case-insensitive literal substring, then by BM25. Matching it as a regex
// would make PDF? match "update", and ranking a pattern that matched nothing by
// its words would return the items it excluded.
func compileQuery(query string) compiledQuery {
	q := compiledQuery{hasMeta: looksLikeRegex(query)}
	if !q.hasMeta {
		return q
	}
	prose := strings.ContainsFunc(query, unicode.IsSpace)
	// An invalid byte would hide a marker, as in .\xff+.
	markers := patternMarkers.MatchString(strings.ToValidUTF8(query, ""))
	// Size a parse without the (?i) the compile adds. Compiling a large pattern
	// is itself slow, and case folding is the slow part of parsing a wide
	// character class, though a query that sets (?i) itself is folded here too.
	// Folding changes which alternatives the parser can merge, and a counted
	// repeat multiplies the difference, so the estimate can be well off in
	// either direction (see maxPatternSize).
	parsed, err := syntax.Parse(query, syntax.Perl)
	isPattern := markers || repeatsOnceParsed(query, parsed, err)
	if err != nil {
		var perr *syntax.Error
		isPattern = isPattern && errors.As(err, &perr)
		if isPattern && !prose {
			q.notRun = notRunNote(perr)
		}
	} else if !prose && patternSize(parsed) > maxPatternSize {
		q.notRun = "pattern not run: counted repeats such as {1,128} make it too large — " +
			"use .*, a smaller count, or keywords"
	}
	q.keywordFallback = prose || !isPattern
	q.regexHint = !isPattern && !prose && err == nil
	switch {
	case q.keywordFallback:
		// QuoteMeta keeps invalid UTF-8, which regexp rejects even in a literal.
		q.matchName = regexp.MustCompile("(?i)" + regexp.QuoteMeta(strings.ToValidUTF8(query, "\uFFFD"))).MatchString
	case q.notRun == "":
		// A query that parses also parses with (?i) in front.
		q.matchName = regexp.MustCompile("(?i)" + query).MatchString
	}
	return q
}

// notRunNote explains why a pattern that does not parse was not run.
func notRunNote(perr *syntax.Error) string {
	if perr.Expr == "" {
		return fmt.Sprintf("pattern not run: %s — use a simpler pattern or keywords", perr.Code)
	}
	return fmt.Sprintf("pattern not run: %s: `%s` — use a simpler pattern or keywords",
		perr.Code, strings.ToValidUTF8(perr.Expr, "\uFFFD"))
}

// patternMarkers matches syntax that keywords rarely contain but patterns do: a
// character class, an alternation, an anchor, a counted repeat including the
// {,n} form Go reads as text, a (? group or (* verb, an escape such as \d, \b,
// \. or \x5f, a .+ or a possessive *+ or ?+, and a group followed by a word,
// as in (user)_file. A ? or + after an underscore marks a tool-name pattern
// such as list_files?, since keywords such as forecast? have none. After a
// hyphen or a camelCase hump it does not, since keywords such as e-mail? and
// iOS? have them, so listFiles? is read as keywords. Reading the text rather
// than the parse tree also catches a pattern the parser simplifies, such as [.]
// becoming a literal, and one that does not parse, such as ^get_.*+_record$. A
// bare ., as in node.js, and a group on its own or after a word, as in (SMTP),
// invoice(s) or fetch(), are not markers. A keyword such as US$ or C:\temp
// trips it, and is then matched as a pattern.
var patternMarkers = regexp.MustCompile(
	`[\[|^$]|\(\?|\(\*|\\.|\.\+|[*?]\+|\{,?\d|_.*[?+]|\)\w`)

// repeatsOnceParsed reports whether query, which has none of patternMarkers,
// is a pattern by its parse tree (see usesRepeat). A query that does not parse
// is fixed first. Its invalid UTF-8 and a trailing backslash are deleted. Its
// parentheses are then balanced, either by adding the missing ones at its ends
// or by deleting any fewest set of them (see anyBalancing), and its stray
// repeats are deleted after that (see deleteStrayRepeats). It is a pattern if
// any of these fixes leaves a pattern, or a query that still does not parse.
// So get_users**_record, (get_users*_record and node.(js? are patterns, and so
// are some queries that look like keywords: tel(+48, *user* and why???, and
// (beta? and C+), which closing turns into a ? or + inside a group. C++, C++17
// and (beta are keywords. A pattern that is one only by its group, such as
// (user)_file, reads as keywords once it loses its ), since (user_file cannot
// be told apart from (beta.
func repeatsOnceParsed(query string, parsed *syntax.Regexp, err error) bool {
	if err == nil {
		return usesRepeat(parsed)
	}
	b := []byte(strings.TrimSuffix(strings.ToValidUTF8(query, ""), `\`))
	opens, closes := unbalanced(b)
	closed := slices.Concat(bytes.Repeat([]byte("("), closes), b, bytes.Repeat([]byte(")"), opens))
	return balancedIsPattern(closed) || anyBalancing(b, opens, closes, balancedIsPattern)
}

// balancedIsPattern reports whether b, a query with balanced parentheses, is a
// pattern once its stray repeats are deleted, or still does not parse.
func balancedIsPattern(b []byte) bool {
	text := deleteStrayRepeats(b)
	if patternMarkers.Match(text) {
		return true
	}
	re, err := syntax.Parse(string(text), syntax.Perl)
	return err != nil || usesRepeat(re)
}

// unbalanced returns how many of the ( and how many of the ) in b have no
// partner.
func unbalanced(b []byte) (opens, closes int) {
	for _, c := range b {
		switch {
		case c == '(':
			opens++
		case c != ')':
		case opens == 0:
			closes++
		default:
			opens--
		}
	}
	return opens, closes
}

// maxBalancingSteps bounds the search in anyBalancing.
const maxBalancingSteps = 256

// anyBalancing reports whether f returns true for b with any fewest set of
// parentheses deleted that balances it, given that opens of its ( and closes of
// its ) have no partner. It reports true, as for a pattern, once the search
// takes more than maxBalancingSteps steps. f must not keep its argument.
func anyBalancing(b []byte, opens, closes int, f func([]byte) bool) bool {
	kept := make([]byte, 0, len(b))
	steps := 0
	var walk func(i, depth, opens, closes int) bool
	walk = func(i, depth, opens, closes int) bool {
		if steps++; steps > maxBalancingSteps {
			return true
		}
		n := len(kept)
		defer func() { kept = kept[:n] }()
		for ; i < len(b); i++ {
			c := b[i]
			if c == '(' && opens > 0 && walk(i+1, depth, opens-1, closes) ||
				c == ')' && closes > 0 && walk(i+1, depth, opens, closes-1) {
				return true
			}
			switch {
			case c == '(':
				depth++
			case c == ')' && depth == 0:
				return false
			case c == ')':
				depth--
			}
			kept = append(kept, c)
		}
		return depth == 0 && f(kept)
	}
	return walk(0, 0, opens, closes)
}

// deleteStrayRepeats deletes from a query with balanced parentheses each ++,
// since keywords such as C++17 have one, and then each run of *, + and ? with
// nothing to repeat, at the start or after a (, as in *node or (+48). After a (
// it keeps a ? from the run, since (? starts a flag group, which patternMarkers
// catches. So getusers++record is keywords, while patternMarkers keeps
// getx.++record a pattern.
func deleteStrayRepeats(b []byte) []byte {
	out := make([]byte, 0, len(b))
	var run []byte
	for _, c := range b {
		if isRepeatOp(c) {
			run = append(run, c)
			continue
		}
		out = append(keptOfRun(out, run), c)
		run = run[:0]
	}
	return keptOfRun(out, run)
}

// keptOfRun appends to out what deleteStrayRepeats keeps of run.
func keptOfRun(out, run []byte) []byte {
	run = bytes.ReplaceAll(run, []byte("++"), nil)
	switch {
	case len(out) == 0:
		return out
	case out[len(out)-1] == '(':
		if slices.Contains(run, '?') {
			out = append(out, '?')
		}
		return out
	}
	return append(out, run...)
}

func isRepeatOp(c byte) bool { return c == '*' || c == '+' || c == '?' }

// nameMatches returns the items whose name q matches, sorted by name.
func (q compiledQuery) nameMatches(items []Item, cfg Config) []Match {
	if q.matchName == nil {
		return nil
	}
	var matches []Match
	for _, it := range items {
		if q.matchName(it.Name) {
			matches = append(matches, Match{Name: it.Name, Description: truncateRunes(it.Description, cfg.MaxDescLen)})
		}
	}
	slices.SortFunc(matches, func(a, b Match) int { return strings.Compare(a.Name, b.Name) })
	return matches
}

func noMatchNote(cfg Config) string {
	return fmt.Sprintf("no %ss matched; try broader keywords or a different term", cfg.ItemNoun)
}

// alreadyAvailableNote tells the caller that the query did match, but only
// items it can already use, so it does not conclude they are missing.
func alreadyAvailableNote(had []Match, cfg Config) string {
	names := make([]string, 0, len(had))
	for _, m := range had {
		names = append(names, m.Name)
	}
	if cfg.MaxResults > 0 && len(names) > cfg.MaxResults {
		names = names[:cfg.MaxResults]
	}
	return fmt.Sprintf("no new %ss matched; these matching %ss are already available: %s",
		cfg.ItemNoun, cfg.ItemNoun, strings.Join(names, ", "))
}

// selectByName loads items by exact name, skipping duplicate names, reporting
// names that match no item, and reporting names that are already available.
func selectByName(items []Item, names string, alreadyAvailable map[string]bool, cfg Config) ([]Match, string) {
	byName := make(map[string]Item, len(items))
	for _, it := range items {
		byName[it.Name] = it
	}

	var matches []Match
	var notFound, had []string
	seen := make(map[string]bool)
	for raw := range strings.SplitSeq(names, ",") {
		name := strings.TrimSpace(raw)
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		if alreadyAvailable[name] {
			had = append(had, name)
			continue
		}
		it, ok := byName[name]
		if !ok {
			notFound = append(notFound, name)
			continue
		}
		matches = append(matches, Match{Name: it.Name, Description: truncateRunes(it.Description, cfg.MaxDescLen)})
	}

	var notes []string
	if len(notFound) > 0 {
		notes = append(notes, fmt.Sprintf("%ss not found: %s", cfg.ItemNoun, strings.Join(notFound, ", ")))
	}
	if len(had) > 0 {
		notes = append(notes, fmt.Sprintf("already available: %s", strings.Join(had, ", ")))
	}
	if len(matches) == 0 && len(notes) == 0 {
		notes = append(notes, noMatchNote(cfg))
	}
	return matches, strings.Join(notes, ". ")
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

// DescribeSearch appends the sorted, deduplicated item names to a search tool's
// base description so the model knows what it can search for (and select: by
// name). The list is sorted for a stable, cache-friendly declaration.
func DescribeSearch(baseDescription string, itemNames []string) string {
	if len(itemNames) == 0 {
		return baseDescription
	}
	names := slices.Clone(itemNames)
	slices.Sort(names)
	names = slices.Compact(names)
	return baseDescription +
		"\n\nAvailable to load (not yet active — search or select: by name): " +
		strings.Join(names, ", ")
}

// looksLikeRegex reports whether query contains regex metacharacters, in which
// case it is first matched against names (see compileQuery).
func looksLikeRegex(query string) bool {
	return strings.ContainsAny(query, `.*+?[]()|^$\{}`)
}

// usesRepeat reports whether re uses a repeat other than a trailing ? or + on
// a literal, as in "forecast?" or "node+". A literal there is one character
// unless written as (?:...), which patternMarkers catches. patternMarkers also
// catches every class, alternation, anchor, word boundary and counted repeat,
// so this need only look for *, + and ?. A ? or + anywhere else counts, so
// PDF?. is a pattern while PDF? is keywords.
func usesRepeat(re *syntax.Regexp) bool {
	if isTrailingPunctuation(re) {
		return false
	}
	if re.Op == syntax.OpConcat && isTrailingPunctuation(re.Sub[len(re.Sub)-1]) {
		return slices.ContainsFunc(re.Sub[:len(re.Sub)-1], hasRepeat)
	}
	return hasRepeat(re)
}

// isTrailingPunctuation reports whether re is a ? or + on a literal.
func isTrailingPunctuation(re *syntax.Regexp) bool {
	return (re.Op == syntax.OpQuest || re.Op == syntax.OpPlus) && re.Sub[0].Op == syntax.OpLiteral
}

// hasRepeat reports whether re or any part of it is a *, + or ? repeat.
func hasRepeat(re *syntax.Regexp) bool {
	switch re.Op {
	case syntax.OpStar, syntax.OpPlus, syntax.OpQuest:
		return true
	}
	return slices.ContainsFunc(re.Sub, hasRepeat)
}

// patternSize estimates how many instructions re compiles to, and stops
// counting once the estimate passes maxPatternSize. A counted repeat multiplies
// the size of its operand.
func patternSize(re *syntax.Regexp) int {
	n := 1
	if re.Op == syntax.OpLiteral {
		n = len(re.Rune)
	}
	for _, sub := range re.Sub {
		if n += patternSize(sub); n > maxPatternSize {
			return n
		}
	}
	if re.Op == syntax.OpRepeat {
		n *= max(re.Min, re.Max, 1)
	}
	return n
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
