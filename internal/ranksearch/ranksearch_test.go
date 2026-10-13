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
	"regexp"
	"regexp/syntax"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

const testMaxDescLen = 200

func testCfg() Config {
	return Config{ItemNoun: "tool", MaxResults: 8, MinScoreRatio: 0.1, MaxDescLen: testMaxDescLen}
}

// item builds an Item with a name+description token bag, the same way callers do.
func item(name, desc string, extra ...string) Item {
	return Item{Name: name, Description: desc, Tokens: BuildTokens(name, desc, extra...)}
}

func makeItems(pairs ...[2]string) []Item {
	items := make([]Item, len(pairs))
	for i, p := range pairs {
		items[i] = item(p[0], p[1])
	}
	return items
}

func names(matches []Match) []string {
	out := make([]string, len(matches))
	for i, m := range matches {
		out[i] = m.Name
	}
	return out
}

// checkNames reports names in want that are missing from got and names in
// notWant that are present in got.
func checkNames(t *testing.T, got, want, notWant []string) {
	t.Helper()
	for _, n := range want {
		if !slices.Contains(got, n) {
			t.Errorf("matches = %v, want %q included", got, n)
		}
	}
	for _, n := range notWant {
		if slices.Contains(got, n) {
			t.Errorf("matches = %v, want %q excluded", got, n)
		}
	}
}

// TestRank_CaseInsensitive verifies case-insensitive matching.
func TestRank_CaseInsensitive(t *testing.T) {
	items := makeItems(
		[2]string{"ListBooks", "Lists all Books"},
		[2]string{"get_author", "returns an Author"},
	)
	matches, _ := Rank(items, "books", nil, testCfg())
	if diff := cmp.Diff([]string{"ListBooks"}, names(matches)); diff != "" {
		t.Errorf("Rank() names mismatch (-want +got):\n%s", diff)
	}
}

// TestRank_MatchesOnDescriptionOnly verifies an item is found when the keyword
// appears only in its description, not its name.
func TestRank_MatchesOnDescriptionOnly(t *testing.T) {
	items := makeItems(
		[2]string{"fetch_records", "List all books for a library"},
		[2]string{"get_thing", "returns an unrelated thing"},
	)
	matches, _ := Rank(items, "books", nil, testCfg())
	if diff := cmp.Diff([]string{"fetch_records"}, names(matches)); diff != "" {
		t.Errorf("Rank() names mismatch (-want +got):\n%s", diff)
	}
}

// TestRank_KeywordMatch verifies a keyword matches items whose names or
// descriptions contain it.
func TestRank_KeywordMatch(t *testing.T) {
	items := makeItems(
		[2]string{"list_books", "list books"},
		[2]string{"get_book", "get a book"},
		[2]string{"list_authors", "list authors"},
	)
	matches, _ := Rank(items, "book", nil, testCfg())
	checkNames(t, names(matches), []string{"list_books", "get_book"}, []string{"list_authors"})
}

// TestRank_InvalidRegexFallsBackToLiteral verifies that an invalid regex without
// pattern syntax is treated as keywords, rather than erroring or being reported
// as a pattern that was not run.
func TestRank_InvalidRegexFallsBackToLiteral(t *testing.T) {
	items := makeItems(
		[2]string{"list_books", "list all books"},
		[2]string{"get_author", "get author"},
	)
	// "(unclosed" is an invalid regex without pattern syntax, so it is matched
	// as a literal, which matches nothing, and then by keywords.
	matches, note := Rank(items, "(unclosed", nil, testCfg())
	if len(matches) != 0 {
		t.Errorf("Rank() = %v, want no matches", names(matches))
	}
	if want := noMatchNote(testCfg()); note != want {
		t.Errorf("Rank() note = %q, want %q", note, want)
	}
}

// TestRank_RegexPattern verifies that a pattern is matched against names as a
// regex.
func TestRank_RegexPattern(t *testing.T) {
	items := makeItems(
		[2]string{"get_library_book", "retrieve a book for a library"},
		[2]string{"get_author", "retrieve an author"},
		[2]string{"list_books", "list all books"},
	)
	matches, _ := Rank(items, `get_.*_book`, nil, testCfg())
	checkNames(t, names(matches), []string{"get_library_book"}, []string{"get_author", "list_books"})
}

// TestRank_RanksNameMatchesAboveDescriptionOnly verifies BM25 ranks items whose
// (name-boosted) name carries the query term above an item that only matches the
// term in its description.
func TestRank_RanksNameMatchesAboveDescriptionOnly(t *testing.T) {
	items := makeItems(
		[2]string{"get_publisher", "retrieve a publisher record"},
		[2]string{"list_reviews", "list publisher reviews"},
		[2]string{"publisher_search", "search across all publishers"},
	)
	matches, _ := Rank(items, "publisher", nil, testCfg())
	if len(matches) != 3 {
		t.Fatalf("Rank() returned %d matches, want 3", len(matches))
	}
	if matches[2].Name != "list_reviews" {
		t.Errorf("last match = %q, want %q (description-only match ranks last)", matches[2].Name, "list_reviews")
	}
	checkNames(t, names(matches[:2]), []string{"get_publisher", "publisher_search"}, nil)
}

// TestRank_IndexesExtraTokens verifies an item is found when the query matches
// only an extra token (e.g. an argument description), not its name or
// description.
func TestRank_IndexesExtraTokens(t *testing.T) {
	items := []Item{
		item("update_note", "modify a note", "body", "the comment text to attach"),
		item("list_images", "list all images"),
	}
	matches, _ := Rank(items, "comment", nil, testCfg())
	checkNames(t, names(matches), []string{"update_note"}, []string{"list_images"})
}

// TestRank_ExcludesAlreadyAvailable verifies items in the alreadyAvailable set
// are not returned again.
func TestRank_ExcludesAlreadyAvailable(t *testing.T) {
	items := makeItems(
		[2]string{"list_books", "list books"},
		[2]string{"get_book", "get a book"},
	)
	matches, _ := Rank(items, "book", map[string]bool{"list_books": true}, testCfg())
	checkNames(t, names(matches), []string{"get_book"}, []string{"list_books"})
}

// TestRank_CappedResultsIncludeNote verifies that when more items match than
// MaxResults allows, the result is capped and a note flags the truncation.
func TestRank_CappedResultsIncludeNote(t *testing.T) {
	cfg := testCfg()
	cfg.MaxResults = 2
	items := makeItems(
		[2]string{"tool_a", "some tool"},
		[2]string{"tool_b", "some tool"},
		[2]string{"tool_c", "some tool"},
	)
	matches, note := Rank(items, "tool", nil, cfg)
	if len(matches) != 2 {
		t.Errorf("Rank() returned %d matches, want 2", len(matches))
	}
	if note == "" {
		t.Error("Rank() note is empty, want a truncation note")
	}
}

// TestRank_EmptyQueryReturnsNote verifies an empty query returns a guidance note
// rather than matching everything.
func TestRank_EmptyQueryReturnsNote(t *testing.T) {
	items := makeItems([2]string{"list_books", "list books"})
	matches, note := Rank(items, "   ", nil, testCfg())
	if len(matches) != 0 {
		t.Errorf("Rank() = %v, want no matches", names(matches))
	}
	if note == "" {
		t.Error("Rank() note is empty, want a guidance note")
	}
}

// TestRank_QueryTooLong verifies a query exceeding the limit returns an error
// note rather than searching.
func TestRank_QueryTooLong(t *testing.T) {
	items := makeItems([2]string{"list_books", "list books"})
	matches, note := Rank(items, strings.Repeat("a", maxQueryLen+1), nil, testCfg())
	if len(matches) != 0 {
		t.Errorf("Rank() = %v, want no matches", names(matches))
	}
	if !strings.Contains(note, "query too long") {
		t.Errorf("Rank() note = %q, want it to contain %q", note, "query too long")
	}
}

// TestRank_MultiWordQuery verifies BM25 ranks an item matching multiple query
// terms above one that matches none.
func TestRank_MultiWordQuery(t *testing.T) {
	items := makeItems(
		[2]string{"list_publisher_reviews", "list reviews for publishers"},
		[2]string{"get_author", "get author"},
	)
	matches, _ := Rank(items, "publisher reviews", nil, testCfg())
	checkNames(t, names(matches), []string{"list_publisher_reviews"}, []string{"get_author"})
}

// TestRank_SelectByName verifies the "select:a,b" form loads items by exact
// name, skips already-available ones, and reports unknown names.
func TestRank_SelectByName(t *testing.T) {
	items := makeItems(
		[2]string{"patch_book", "patch a book"},
		[2]string{"update_book", "update a book"},
		[2]string{"list_books", "list books"},
	)
	matches, note := Rank(items, "select:patch_book,update_book,unknown_tool", nil, testCfg())
	checkNames(t, names(matches), []string{"patch_book", "update_book"}, []string{"list_books"})
	if !strings.Contains(note, "unknown_tool") {
		t.Errorf("Rank() note = %q, want the missing name %q in it", note, "unknown_tool")
	}
}

// TestRank_SelectSkipsAlreadyAvailable verifies select leaves already-available
// names out of the matches and reports them as already available, not as not
// found.
func TestRank_SelectSkipsAlreadyAvailable(t *testing.T) {
	items := makeItems(
		[2]string{"patch_book", "patch a book"},
		[2]string{"list_books", "list books"},
	)
	matches, note := Rank(items, "select:patch_book,list_books",
		map[string]bool{"list_books": true}, testCfg())
	if diff := cmp.Diff([]string{"patch_book"}, names(matches)); diff != "" {
		t.Errorf("Rank() names mismatch (-want +got):\n%s", diff)
	}
	if want := "already available: list_books"; note != want {
		t.Errorf("Rank() note = %q, want %q", note, want)
	}
}

// TestRank_TruncatesAtRuneBoundary verifies truncation does not split a
// multi-byte UTF-8 character.
func TestRank_TruncatesAtRuneBoundary(t *testing.T) {
	desc := strings.Repeat("x", testMaxDescLen-1) + "€" + strings.Repeat("x", 10)
	// The first query is ranked by BM25 and the second matches the name.
	for _, query := range []string{"t", "^t$"} {
		matches, _ := Rank([]Item{item("t", desc)}, query, nil, testCfg())
		if len(matches) != 1 {
			t.Fatalf("Rank(%q) returned %d matches, want 1", query, len(matches))
		}
		got := matches[0].Description
		if !utf8.ValidString(got) {
			t.Errorf("Rank(%q): truncated description is not valid UTF-8", query)
		}
		if len(got) > testMaxDescLen {
			t.Errorf("Rank(%q): len(description) = %d, want <= %d", query, len(got), testMaxDescLen)
		}
	}
}

// TestDescribeSearch_AppendsSortedNames verifies item names are appended sorted.
func TestDescribeSearch_AppendsSortedNames(t *testing.T) {
	out := DescribeSearch("base.", []string{"zebra", "alpha"})
	for _, want := range []string{"base.", "alpha, zebra"} {
		if !strings.Contains(out, want) {
			t.Errorf("DescribeSearch() = %q, want it to contain %q", out, want)
		}
	}
}

func TestDescribeSearch_ListsEachNameOnce(t *testing.T) {
	out := DescribeSearch("base.", []string{"zebra", "alpha", "zebra"})
	if got := strings.Count(out, "zebra"); got != 1 {
		t.Errorf("DescribeSearch() = %q, lists zebra %d times, want once", out, got)
	}
}

// TestDescribeSearch_NoNamesIsBaseDescription verifies the description is
// unchanged when no names are provided.
func TestDescribeSearch_NoNamesIsBaseDescription(t *testing.T) {
	if got := DescribeSearch("base.", nil); got != "base." {
		t.Errorf("DescribeSearch() = %q, want %q", got, "base.")
	}
}

// TestRank_NaturalLanguageWithRegexCharacters verifies a natural-language query
// that happens to contain a regex metacharacter, such as a trailing question
// mark, still finds items when no name contains it.
func TestRank_NaturalLanguageWithRegexCharacters(t *testing.T) {
	items := makeItems(
		[2]string{"list_books", "list all books"},
		[2]string{"get_author", "get author"},
	)
	for _, query := range []string{"how do I list books?", "list the books.", "books (all of them)", "list 2*3 books"} {
		matches, _ := Rank(items, query, nil, testCfg())
		checkNames(t, names(matches), []string{"list_books"}, []string{"get_author"})
	}
}

// TestRank_PatternThatMatchesNothingStaysEmpty verifies a deliberate pattern
// with no match returns nothing, rather than keyword matches for the words in it
// that the pattern itself excluded.
func TestRank_PatternThatMatchesNothingStaysEmpty(t *testing.T) {
	items := makeItems(
		[2]string{"get_user_file", "fetch a user's file"},
		[2]string{"delete_user_record", "delete a record"},
	)
	matches, note := Rank(items, `^get_.*_record$`, nil, testCfg())
	if len(matches) != 0 {
		t.Errorf("Rank() = %v, want no matches", names(matches))
	}
	if note == "" {
		t.Error("Rank() note is empty, want a no-match note")
	}
}

// TestRank_FallbackIsAnnounced verifies the note tells the model when a query
// it may have meant as a pattern was ranked by keywords instead.
func TestRank_FallbackIsAnnounced(t *testing.T) {
	items := makeItems([2]string{"list_books", "list all books"})
	_, note := Rank(items, "how do I list books?", nil, testCfg())
	const fallbackNote = "showing keyword matches"
	if !strings.Contains(note, fallbackNote) {
		t.Errorf("Rank() note = %q, want it to contain %q", note, fallbackNote)
	}

	cfg := testCfg()
	cfg.MaxResults = 2
	items = makeItems(
		[2]string{"tool_a", "some tool"},
		[2]string{"tool_b", "some tool"},
		[2]string{"tool_c", "some tool"},
	)
	_, note = Rank(items, "which tool?", nil, cfg)
	for _, want := range []string{fallbackNote, "showing 2 of 3"} {
		if !strings.Contains(note, want) {
			t.Errorf("capped fallback note = %q, want it to contain %q", note, want)
		}
	}
}

// TestRank_OnlyAlreadyAvailableMatchesSaySo checks that a query whose only
// matches are already available names them rather than reporting no match, in
// both the ranked and the select: forms.
func TestRank_OnlyAlreadyAvailableMatchesSaySo(t *testing.T) {
	items := makeItems(
		[2]string{"list_books", "list books"},
		[2]string{"get_author", "get an author"},
	)
	available := map[string]bool{"list_books": true}
	for _, query := range []string{"books", "select:list_books"} {
		matches, note := Rank(items, query, available, testCfg())
		if len(matches) != 0 {
			t.Errorf("Rank(%q) matches = %v, want none", query, names(matches))
		}
		if !strings.Contains(note, "already available") || !strings.Contains(note, "list_books") {
			t.Errorf("Rank(%q) note = %q, want it to name list_books as already available", query, note)
		}
	}
	// Control: with nothing available, the same query does match.
	if matches, _ := Rank(items, "books", nil, testCfg()); len(matches) == 0 {
		t.Error("Rank(\"books\") with nothing available returned no matches")
	}
	// A query that matches nothing at all still says so.
	if _, note := Rank(items, "weather", available, testCfg()); note != noMatchNote(testCfg()) {
		t.Errorf("Rank(\"weather\") note = %q, want %q", note, noMatchNote(testCfg()))
	}
	// MaxResults 0 means no cap, so the note still names the item.
	unlimited := testCfg()
	unlimited.MaxResults = 0
	if _, note := Rank(items, "books", available, unlimited); !strings.Contains(note, "list_books") {
		t.Errorf("Rank(\"books\") with MaxResults 0 note = %q, want it to name list_books", note)
	}
}

// TestRank_SelectReportsUnknownNameOnce checks that a repeated unknown name is
// reported once.
func TestRank_SelectReportsUnknownNameOnce(t *testing.T) {
	_, note := Rank(makeItems([2]string{"list_books", "list books"}), "select:ghost,ghost", nil, testCfg())
	if want := "tools not found: ghost"; note != want {
		t.Errorf("Rank() note = %q, want %q", note, want)
	}
}

// TestRank_AlreadyAvailableNoteIsCapped checks that a ranked query whose matches
// are all already available names at most MaxResults of them.
func TestRank_AlreadyAvailableNoteIsCapped(t *testing.T) {
	items := makeItems(
		[2]string{"list_books", "list books"},
		[2]string{"list_authors", "list authors"},
		[2]string{"list_notes", "list notes"},
	)
	available := map[string]bool{"list_books": true, "list_authors": true, "list_notes": true}
	cfg := testCfg()
	cfg.MaxResults = 2
	_, note := Rank(items, "list", available, cfg)
	_, listed, ok := strings.Cut(note, "already available: ")
	if !ok {
		t.Fatalf("Rank() note = %q, want it to name the already-available tools", note)
	}
	if got := strings.Split(listed, ", "); len(got) != 2 {
		t.Errorf("Rank() note names %v, want 2 of the 3 matching tools", got)
	}
}

// TestRank_RelevanceFloorDropsWeakMatches checks that a match scoring below
// MinScoreRatio of the top score is dropped, and kept under a lower floor.
func TestRank_RelevanceFloorDropsWeakMatches(t *testing.T) {
	items := makeItems(
		[2]string{"list_books", "list books"},
		[2]string{"get_author", "returns the author of one of the books in the library catalog"},
	)
	for _, tc := range []struct {
		ratio float64
		want  []string
	}{
		{0.5, []string{"list_books"}},
		{0.1, []string{"list_books", "get_author"}},
	} {
		cfg := testCfg()
		cfg.MinScoreRatio = tc.ratio
		matches, _ := Rank(items, "books", nil, cfg)
		if diff := cmp.Diff(tc.want, names(matches)); diff != "" {
			t.Errorf("MinScoreRatio %v: Rank() names mismatch (-want +got):\n%s", tc.ratio, diff)
		}
	}
}

// TestRank_RegexMatchesNamesOnly checks that a regex matches item names, sorted
// by name, and not descriptions. The order decides which items survive the
// MaxResults cap.
func TestRank_RegexMatchesNamesOnly(t *testing.T) {
	items := makeItems(
		[2]string{"a_tool", "fetches books"},
		[2]string{"z_books", "unrelated"},
		[2]string{"m_books", "unrelated"},
	)
	matches, _ := Rank(items, "book.*", nil, testCfg())
	if diff := cmp.Diff([]string{"m_books", "z_books"}, names(matches)); diff != "" {
		t.Errorf("Rank() names mismatch (-want +got):\n%s", diff)
	}
}

// TestRank_RegexIsCaseInsensitive checks that a pattern matches a name whatever
// the case of either.
func TestRank_RegexIsCaseInsensitive(t *testing.T) {
	matches, _ := Rank(makeItems([2]string{"getUserFile", "returns a file"}), "getuser.*", nil, testCfg())
	if diff := cmp.Diff([]string{"getUserFile"}, names(matches)); diff != "" {
		t.Errorf("Rank() names mismatch (-want +got):\n%s", diff)
	}
}

// TestRank_InvalidPatternMatchesLiterally checks that a query that fails to
// parse the way punctuated keywords do is matched against names as a
// case-insensitive literal substring.
func TestRank_InvalidPatternMatchesLiterally(t *testing.T) {
	items := makeItems(
		[2]string{"beta(v2)_tool", "runs beta features"},
		[2]string{"other_tool", "runs (v2 features"},
	)
	matches, _ := Rank(items, "(V2", nil, testCfg())
	if diff := cmp.Diff([]string{"beta(v2)_tool"}, names(matches)); diff != "" {
		t.Errorf("Rank() names mismatch (-want +got):\n%s", diff)
	}
}

// TestRank_LargePatternIsNotRun checks that a pattern whose counted repeat
// makes it larger than maxPatternSize is not run, and that the note says why,
// while a smaller one still runs as a regex.
func TestRank_LargePatternIsNotRun(t *testing.T) {
	items := makeItems([2]string{"fix_tool", "fixes things"})
	for query, want := range map[string][]string{
		".{0,100}x": {"fix_tool"},
		".{0,999}x": nil,
		// Whitespace marks prose, which is ranked by keywords instead.
		"fixes things {0,999}": {"fix_tool"},
		// Each pattern below matches the empty string as a regex, so whether
		// fix_tool comes back shows whether it ran. A literal counts each rune,
		// and an unbounded repeat counts its minimum.
		"(?:fix_tool){0,20}": {"fix_tool"},
		"(?:fix_tool){0,30}": nil,
		"(?:x?){150,}":       nil,
	} {
		matches, note := Rank(items, query, nil, testCfg())
		if diff := cmp.Diff(want, names(matches), cmpopts.EquateEmpty()); diff != "" {
			t.Errorf("Rank(%q) names mismatch (-want +got):\n%s", query, diff)
		}
		if tooLarge := strings.Contains(note, "too large"); tooLarge != (want == nil) {
			t.Errorf("Rank(%q) note = %q, want it to say the pattern is too large: %t", query, note, want == nil)
		}
	}
}

// TestRank_PunctuatedKeywordsFallBackToKeywords checks that a query whose regex
// characters also appear in keywords, such as node.js or C++, is ranked by
// keywords when it matches no name, while one that uses a pattern operator stays
// empty, including a repeat written with + or ? instead of *.
func TestRank_PunctuatedKeywordsFallBackToKeywords(t *testing.T) {
	items := makeItems(
		[2]string{"run_script", "Runs Node.js scripts"},
		[2]string{"compile_code", "Compiles C++ sources"},
		[2]string{"list_invoices", "Lists unpaid invoices. Supports filters."},
		[2]string{"price_lookup", "look up a price for v1.2 products"},
		[2]string{"update_record", "update a record"},
		[2]string{"convert_document", "convert PDF files to text"},
		[2]string{"build_app", "Build an iOS or Android app"},
		[2]string{"get_location", "returns the device location"},
	)
	for query, want := range map[string][]string{
		"node.js":   {"run_script"},
		"C++":       {"compile_code"},
		"invoices.": {"list_invoices"},
		"v1.2":      {"price_lookup"},
		"(scripts)": {"run_script"},
		// A group on its own or after a word is still keywords.
		"node(js)":   {"run_script"},
		"invoice(s)": {"list_invoices"},
		"convert()":  {"convert_document"},
		// As a regex this would match the "pd" in update_record.
		"PDF?": {"convert_document"},
		// Each of these fails to parse the way punctuated keywords do.
		"*node":         {"run_script"},
		"(node":         {"run_script"},
		"node)":         {"run_script"},
		"node\\":        {"run_script"},
		"Objective-C++": {"compile_code"},
		"C++17":         {"compile_code"},
		"(C++)":         {"compile_code"},
		"(+js)":         {"run_script"},
		"node)(js":      {"run_script"},
		"node+\xdc":     {"run_script"},
		"convert())":    {"convert_document"},
		// A ] or a brace without a digit is text to Go's regexp.
		"node]":  {"run_script"},
		"{node}": {"run_script"},
		// A trailing ? or + on one character is still keywords.
		"sources?": {"compile_code"},
		"node+":    {"run_script"},
		"c+":       {"compile_code"},
		// So is one after a hyphen or a camelCase hump, which keywords have. As a
		// regex, iOS? would match the "io" in get_location.
		"iOS?":          {"build_app"},
		"nodeScripts?":  {"run_script"},
		"node-scripts?": {"run_script"},
		// Each of these is a pattern by its text alone (see patternMarkers).
		"node.+js":      nil,
		"node.+":        nil,
		"node.{2}js":    nil,
		"node.{,2}js":   nil,
		"node\\x2ejs":   nil,
		"node[a-z]js":   nil,
		"node[.]js":     nil,
		"node\\d":       nil,
		"node\\Djs":     nil,
		"node\\sjs":     nil,
		"node\\Sjs":     nil,
		"node\\Wjs":     nil,
		"node\\pLjs":    nil,
		"node\\PLjs":    nil,
		"node|deno":     nil,
		"node|node":     nil,
		"^node":         nil,
		"\\Anode":       nil,
		"js$":           nil,
		"js\\z":         nil,
		"no$de":         nil,
		"js\\b":         nil,
		"js\\B":         nil,
		"(?m)^node":     nil,
		"(?:run)?js":    nil,
		"node(?:s)?":    nil,
		"(?i)node":      nil,
		"(?:node)":      nil,
		"node\\.js":     nil,
		"\\Qnode\\E":    nil,
		"node_scripts?": nil,
		"node_script+":  nil,
		"(node)_js":     nil,
		// Each of these is a pattern only by a repeat in its parse tree.
		"node.*js":   nil,
		"node.?js":   nil,
		"nodes*js":   nil,
		"nodes*":     nil,
		"nodes+js":   nil,
		"node?js":    nil,
		"(run)?js":   nil,
		"node.(js)?": nil,
		"node.(js?)": nil,
		"nodes*j?":   nil,
	} {
		matches, note := Rank(items, query, nil, testCfg())
		if diff := cmp.Diff(want, names(matches), cmpopts.EquateEmpty()); diff != "" {
			t.Errorf("Rank(%q) names mismatch (-want +got):\n%s", query, diff)
		}
		wantNote := "showing keyword matches"
		if want == nil {
			wantNote = "patterns match names only"
		}
		if !strings.Contains(note, wantNote) {
			t.Errorf("Rank(%q) note = %q, want it to contain %q", query, note, wantNote)
		}
	}
}

// TestRank_UnsupportedPatternIsNotRun checks that a pattern Go's regexp
// rejects, for syntax it lacks such as lookaround or for a typo such as
// get_users**_record, is not ranked by its words, which would return the tools
// it excluded, and that the note says why. Prose with such syntax is still
// ranked by keywords.
func TestRank_UnsupportedPatternIsNotRun(t *testing.T) {
	items := makeItems(
		[2]string{"get_user_file", "fetch a user's file"},
		[2]string{"delete_user_record", "delete a record"},
	)
	for query, want := range map[string]string{
		"^(?!delete).*_record$": "invalid or unsupported Perl syntax",
		"_record\\Z":            "invalid escape sequence",
		"^get_[a-z]{1,2000}$":   "invalid repeat count",
		"[z-a]_record":          "invalid character class range",
		"(?<!delete)_record$":   "invalid named capture",
		"^get_[a-z_+_record$":   "missing closing ]",
		// Each of these fails with an error that punctuated keywords also
		// produce, but its text shows a pattern.
		"^get\\":                "trailing backslash at end of expression",
		"$\xdc":                 "invalid UTF-8",
		"^get_.*+_record$":      "invalid nested repetition operator",
		"get_users*+_record":    "invalid nested repetition operator",
		"^(get|list_.*_record$": "missing closing )",
		"^(get|delete)_.*)$":    "unexpected )",
		// Each of these shows one marker only.
		"^getuser)":        "unexpected )",
		"getuser)$":        "unexpected )",
		"(*FAIL)":          "missing argument to repetition operator",
		"getx.+*record":    "invalid nested repetition operator",
		"getusers*+record": "invalid nested repetition operator",
		"getuser?+record":  "invalid nested repetition operator",
		"getx\\w++record":  "invalid nested repetition operator",
		"getx{2}+":         "invalid nested repetition operator",
		// Each of these has no marker, and is a pattern by repeatsOnceParsed.
		"get_users**_record":  "invalid nested repetition operator",
		"getusers**record":    "invalid nested repetition operator",
		"(get_users*_record":  "missing closing )",
		"get_users*_record)":  "unexpected )",
		"*getusers*record":    "missing argument to repetition operator",
		"getusers*record\\":   "trailing backslash at end of expression",
		"getusers*record\xdc": "invalid UTF-8",
		"getx.**record":       "invalid nested repetition operator",
		"getx.?*record":       "invalid nested repetition operator",
		// Each of these lost a parenthesis. Adding one at the start or the end
		// makes it a pattern, and deleting one does not.
		"node.(js?":     "missing closing )",
		"node.js)?":     "unexpected )",
		"getx+)+record": "unexpected )",
		// Each of these hides a marker behind an invalid byte.
		"getx.\xff++record": "invalid UTF-8",
		"(\xff*FAIL)":       "invalid UTF-8",
		// Each of these is a pattern by its marker alone, since deleting what
		// breaks it leaves keywords.
		"getx.++record": "invalid nested repetition operator",
		"*+getusers":    "missing argument to repetition operator",
		"?+getusers":    "missing argument to repetition operator",
		// Deleting the first ( leaves keywords, and deleting the second leaves
		// a pattern, so this is a pattern.
		"(getusers(+record)": "missing argument to repetition operator",
		// Deleting both parentheses leaves a pattern, and adding their partners
		// does not.
		"getusers)(+record": "unexpected )",
		// Each of these looks like keywords, but one way of fixing it leaves a
		// pattern: deleting the first *, deleting the (, or closing the (.
		"*node*":  "missing argument to repetition operator",
		"tel(+48": "missing argument to repetition operator",
		"(beta?":  "missing closing )",
		// This still does not parse once fixed.
		"why???": "invalid nested repetition operator",
	} {
		matches, note := Rank(items, query, nil, testCfg())
		if len(matches) != 0 {
			t.Errorf("Rank(%q) = %v, want no matches", query, names(matches))
		}
		if !strings.HasPrefix(note, "pattern not run: ") || !strings.Contains(note, want) {
			t.Errorf("Rank(%q) note = %q, want it to say the pattern was not run because of %q", query, note, want)
		}
		if strings.Contains(note, "``") || !utf8.ValidString(note) {
			t.Errorf("Rank(%q) note = %q, want no empty quote and valid UTF-8", query, note)
		}
	}
	matches, note := Rank(items, "delete (?!x) record", nil, testCfg())
	if diff := cmp.Diff([]string{"delete_user_record"}, names(matches)); diff != "" || !strings.Contains(note, "showing keyword matches") {
		t.Errorf("Rank(prose) = %v, note %q, want delete_user_record ranked by keywords (-want +got):\n%s", names(matches), note, diff)
	}
}

// TestRank_BalancingSearchIsBounded checks that a query whose parentheses take
// more than maxBalancingSteps steps to balance counts as a pattern, which is
// not run, while one that takes no more is classified as usual.
func TestRank_BalancingSearchIsBounded(t *testing.T) {
	items := makeItems([2]string{"run_script", "Runs Node.js scripts"})
	// Nine ( and five ) take exactly maxBalancingSteps steps.
	matches, note := Rank(items, "(((((((((node)))))", nil, testCfg())
	if diff := cmp.Diff([]string{"run_script"}, names(matches)); diff != "" || !strings.Contains(note, "showing keyword matches") {
		t.Errorf("Rank(nine) = %v, note %q, want run_script by keywords (-want +got):\n%s", names(matches), note, diff)
	}
	// Ten take more.
	matches, note = Rank(items, "((((((((((node)))))", nil, testCfg())
	if len(matches) != 0 || !strings.HasPrefix(note, "pattern not run: missing closing )") {
		t.Errorf("Rank(ten) = %v, note %q, want no matches and a note that the pattern was not run", names(matches), note)
	}
}

// TestCompileQuery_BrokenPatternStaysPattern checks that a pattern with a typo
// of a kind punctuated keywords also have is still a pattern, and so is not
// ranked by its words. It tries every pattern of up to four characters from a
// small alphabet, with each typo at every place it fits. A doubled + is not
// tried, since a ++ is read as text (see deleteStrayRepeats). Nor is losing a )
// from a pattern with no *, + or ?, or with a ()+, since one that is a pattern
// only by its group reads as keywords once it loses its ) (see
// repeatsOnceParsed).
func TestCompileQuery_BrokenPatternStaysPattern(t *testing.T) {
	queries := []string{""}
	for i := 0; i < len(queries); i++ {
		if len(queries[i]) < 4 {
			for _, c := range "a_.*+?()" {
				queries = append(queries, queries[i]+string(c))
			}
		}
	}
	for _, p := range queries {
		if !looksLikeRegex(p) || compileQuery(p).keywordFallback {
			continue
		}
		if _, err := syntax.Parse(p, syntax.Perl); err != nil {
			continue
		}
		groupOnly := !strings.ContainsAny(p, "*+?") || strings.Contains(p, "()+")
		for _, typo := range typos(p) {
			if groupOnly && strings.Count(typo, ")") < strings.Count(p, ")") {
				continue
			}
			if compileQuery(typo).keywordFallback {
				t.Errorf("compileQuery(%q) is keywords, want a pattern like %q", typo, p)
			}
		}
	}
}

// typos returns p with each typo punctuated keywords also have: a (, a ) or an
// invalid byte added anywhere, one of its ( or ) deleted, a *, + or ? at the
// start or after a (, a doubled * or ?, and a trailing backslash.
func typos(p string) []string {
	var out []string
	insert := func(i int, s string) { out = append(out, p[:i]+s+p[i:]) }
	for i := 0; i <= len(p); i++ {
		for _, s := range []string{"(", ")", "\xff"} {
			insert(i, s)
		}
		if i == 0 || p[i-1] == '(' {
			for _, s := range []string{"*", "+", "?"} {
				insert(i, s)
			}
		}
		if i > 0 && (p[i-1] == '*' || p[i-1] == '?') {
			insert(i, p[i-1:i])
		}
		if i < len(p) && (p[i] == '(' || p[i] == ')') {
			out = append(out, p[:i]+p[i+1:])
		}
	}
	return append(out, p+`\`)
}

// TestRank_AlreadyAvailableNameMatches checks that a pattern whose only name
// match is already available says so, while punctuated keywords rank the other
// tools by keywords first, as plain keywords do, and report an already
// available name match only when nothing else matches.
func TestRank_AlreadyAvailableNameMatches(t *testing.T) {
	available := map[string]bool{"get_forecast": true, "go1.26_tool": true}
	items := makeItems(
		[2]string{"get_forecast", "returns the forecast"},
		[2]string{"weather_alerts", "weather alerts for a forecast region"},
	)
	for _, tc := range []struct {
		items      []Item
		query      string
		want       []string
		wantInNote string
	}{
		{items, "^get_forecast$", nil, "already available: get_forecast"},
		// As a regex, forecast? would match get_forecast and report it as
		// already available, so this row guards the classification.
		{items, "forecast?", []string{"weather_alerts"}, "showing keyword matches"},
		// Only the literal name match finds this one: BM25 sees "o1" and "2",
		// and the name's tokens are "go1", "26" and "tool".
		{makeItems([2]string{"go1.26_tool", "unrelated"}), "o1.2", nil, "already available: go1.26_tool"},
		// The same keyword ranks an eligible tool by keywords first.
		{makeItems([2]string{"go1.26_tool", "unrelated"}, [2]string{"release_notes", "notes for o1 2"}), "o1.2", []string{"release_notes"}, "showing keyword matches"},
	} {
		matches, note := Rank(tc.items, tc.query, available, testCfg())
		if diff := cmp.Diff(tc.want, names(matches), cmpopts.EquateEmpty()); diff != "" {
			t.Errorf("Rank(%q) names mismatch (-want +got):\n%s", tc.query, diff)
		}
		if !strings.Contains(note, tc.wantInNote) {
			t.Errorf("Rank(%q) note = %q, want it to contain %q", tc.query, note, tc.wantInNote)
		}
	}
}

// TestRank_KeywordsThatParseAsRegexSayHowToWriteOne checks that punctuated
// keywords that find nothing, but would also parse as a regex, are told how to
// write a pattern instead, while ones that do not parse, and prose, get the
// plain note.
func TestRank_KeywordsThatParseAsRegexSayHowToWriteOne(t *testing.T) {
	items := makeItems([2]string{"get_forecast", "returns the weather"})
	for query, want := range map[string]string{
		"fore.ast":         "add .* or a character class",
		"(forest":          noMatchNote(testCfg()),
		"how is fore.ast?": noMatchNote(testCfg()),
	} {
		matches, note := Rank(items, query, nil, testCfg())
		if len(matches) != 0 {
			t.Errorf("Rank(%q) = %v, want no matches", query, names(matches))
		}
		if !strings.Contains(note, want) {
			t.Errorf("Rank(%q) note = %q, want it to contain %q", query, note, want)
		}
	}
}

// TestCompileQuery_SizesPatternBeforeCompiling checks that a large pattern is
// rejected before regexp compiles it. Compiling this one allocates about 72,000
// times and takes tens of milliseconds on every search.
func TestCompileQuery_SizesPatternBeforeCompiling(t *testing.T) {
	query := strings.Repeat(".{0,999}", 24) + "QQQQQQQ"
	if allocs := testing.AllocsPerRun(1, func() { compileQuery(query) }); allocs > 1000 {
		t.Errorf("compileQuery() allocated %v times, want at most 1000", allocs)
	}
}

// TestRank_FoldsCaseAtMostOnce checks that a search folds case no more than once
// for a regex that does not set (?i) itself. Case folding a wide character class
// is the slow part, about 3 ms per class, and each extra case-insensitive parse
// pays it again.
func TestRank_FoldsCaseAtMostOnce(t *testing.T) {
	query := strings.Repeat("[B-\U0010ffff]", 10)
	// The names are shorter than the pattern, so nothing matches and Rank goes
	// on to the already-available y_tool.
	items := makeItems([2]string{"x_tool", "unrelated"}, [2]string{"y_tool", "unrelated"})
	available := map[string]bool{"y_tool": true}
	compile := testing.AllocsPerRun(1, func() { _, _ = regexp.Compile("(?i)" + query) })
	if got := testing.AllocsPerRun(1, func() { Rank(items, query, available, testCfg()) }); got > 1.5*compile {
		t.Errorf("Rank() allocated %v times, want at most 1.5 times the %v of one compile", got, compile)
	}
}

// TestRank_InvalidUTF8PatternDoesNotPanic checks the literal fallback for a
// query with invalid UTF-8, which regexp rejects even after QuoteMeta. Dropping
// the invalid byte instead of replacing it would match the "(" in the name.
func TestRank_InvalidUTF8PatternDoesNotPanic(t *testing.T) {
	items := makeItems([2]string{"price_(_tool", "returns a price"})
	matches, note := Rank(items, "(\xdc", nil, testCfg())
	if len(matches) != 0 {
		t.Errorf("Rank() = %v, want no matches", names(matches))
	}
	if want := noMatchNote(testCfg()); note != want {
		t.Errorf("Rank() note = %q, want %q", note, want)
	}
}

// TestRank_SelectWithNoNamesSaysNothingMatched checks that a select: query that
// names nothing gets the no-match note rather than an empty one.
func TestRank_SelectWithNoNamesSaysNothingMatched(t *testing.T) {
	items := makeItems([2]string{"list_books", "list books"})
	for _, query := range []string{"select:", "select: , "} {
		matches, note := Rank(items, query, nil, testCfg())
		if len(matches) != 0 {
			t.Errorf("Rank(%q) = %v, want no matches", query, names(matches))
		}
		if want := "no tools matched; try broader keywords or a different term"; note != want {
			t.Errorf("Rank(%q) note = %q, want %q", query, note, want)
		}
	}
}
