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
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/google/go-cmp/cmp"
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

// TestRank_InvalidRegexFallsBackToLiteral verifies an invalid regex pattern
// falls back to literal substring matching rather than erroring.
func TestRank_InvalidRegexFallsBackToLiteral(t *testing.T) {
	items := makeItems(
		[2]string{"list_books", "list all books"},
		[2]string{"get_author", "get author"},
	)
	// "(unclosed[bracket" is an invalid regex; the literal fallback matches nothing.
	matches, note := Rank(items, "(unclosed[bracket", nil, testCfg())
	if len(matches) != 0 {
		t.Errorf("Rank() = %v, want no matches", names(matches))
	}
	if note == "" {
		t.Error("Rank() note is empty, want a no-match note")
	}
}

// TestRank_RegexPattern verifies valid regex patterns are used as-is.
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

// TestRank_SelectSkipsAlreadyAvailable verifies select silently skips names that
// are already available rather than reporting them as not found.
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
	matches, _ := Rank([]Item{item("t", desc)}, "t", nil, testCfg())
	if len(matches) != 1 {
		t.Fatalf("Rank() returned %d matches, want 1", len(matches))
	}
	got := matches[0].Description
	if !utf8.ValidString(got) {
		t.Error("truncated description is not valid UTF-8")
	}
	if len(got) > testMaxDescLen {
		t.Errorf("len(description) = %d, want <= %d", len(got), testMaxDescLen)
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

// TestDescribeSearch_NoNamesIsBaseDescription verifies the description is
// unchanged when no names are provided.
func TestDescribeSearch_NoNamesIsBaseDescription(t *testing.T) {
	if got := DescribeSearch("base.", nil); got != "base." {
		t.Errorf("DescribeSearch() = %q, want %q", got, "base.")
	}
}

// TestRank_NaturalLanguageWithRegexCharacters verifies a natural-language query
// that happens to contain a regex metacharacter, such as a trailing question
// mark, still finds items when it matches nothing as a pattern.
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
	const fallbackNote = "no matches as a pattern"
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
	if _, note := Rank(items, "weather", available, testCfg()); strings.Contains(note, "already available") {
		t.Errorf("Rank(\"weather\") note = %q, want the no-match note", note)
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
