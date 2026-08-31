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

package main

import (
	"fmt"
	"strings"
)

// botCommentSignature is the leading text of the comment the bot posts when it
// asks for more context. It must stay in sync with the body written by
// buildContextComment so the bot can recognize its own past comment and never
// ask the same author twice.
const botCommentSignature = "**Automated pull request triage:**"

const (
	// maxSnippetRunes bounds how much of the title or body is forwarded to the
	// model. Long text is truncated; this keeps the prompt small and the run
	// cheap, and bounds the attacker-controlled share of it.
	maxSnippetRunes = 1500
	// maxPathRunes bounds one changed-file path. A path is attacker-controlled
	// (the author chose the filename), so it is both fenced and truncated.
	maxPathRunes = 200
)

// PullRequest is the normalized view of a pull request used for triage. It is
// deliberately small: only the fields the preconditions and the prompt need.
type PullRequest struct {
	Number int
	Title  string
	Body   string
	Author string
	// State is the GraphQL PullRequestState: OPEN, CLOSED or MERGED.
	State     string
	IsDraft   bool
	Assignees []string
	// Files are the changed-file paths, already bounded by the query.
	Files []string
	// Comments exist for the "have I already asked for context?" check only.
	// They are never shown to the model: they add attacker-controlled text
	// without improving either decision this bot makes.
	Comments []Comment
	// AssignedBy is the set of logins that have ever assigned someone to this
	// pull request, from its timeline. It is what makes assignment one-way: if
	// the bot is in here it has already had its one turn, so a maintainer's
	// later un-assignment is never undone.
	AssignedBy []string
}

// Comment is the normalized view of a single pull request comment.
type Comment struct {
	Author string
	Body   string
}

// contextItem is one allow-listed thing the model may ask a pull request author
// for. The model chooses keys; Go owns the prose. That is what keeps
// attacker-influenced text out of anything the bot posts.
type contextItem struct {
	key  string
	text string
}

// contextItems is the complete, fixed set of requests the bot can make. Adding
// an entry here is the only way to change what the bot can say, and the tool
// rejects any key that is not in this set.
var contextItems = []contextItem{
	{"problem", "What problem does this change solve? A sentence on the behavior before and after helps a lot."},
	{"summary", "A short summary of what changed, and why this approach rather than another."},
	{"linked_issue", "The issue this addresses, if there is one — linking it (for example `Fixes #123`) brings the background discussion along."},
	{"reproduction", "Steps to reproduce the original problem, so a reviewer can confirm the fix does what it should."},
	{"testing", "How this was tested: the tests that cover it, or the manual steps you ran."},
	{"breaking_change", "Whether this changes existing behavior or the public API, and what users would need to do about it."},
}

// contextItemText returns the prose for an allow-listed key.
func contextItemText(key string) (string, bool) {
	for _, it := range contextItems {
		if it.key == key {
			return it.text, true
		}
	}
	return "", false
}

// contextItemKeys returns the allow-listed keys in their declared order, for
// the prompt and for error messages.
func contextItemKeys() []string {
	keys := make([]string, 0, len(contextItems))
	for _, it := range contextItems {
		keys = append(keys, it.key)
	}
	return keys
}

// skipReason reports why this pull request must not be triaged, or "" when it
// is eligible. Every check reads state the bot computed from GitHub API
// metadata, never anything the model asserted, and all of it runs before the
// model is invoked — an ineligible pull request costs zero tokens.
//
// selfLogin may be empty when the bot could not resolve its own identity. The
// timeline check then falls back to refusing any pull request that has ever been
// assigned by anyone, which is the fail-safe direction: the bot does nothing
// rather than risk undoing a maintainer's decision.
func skipReason(pr PullRequest, selfLogin string, ownerMap map[string]string) string {
	if pr.State != "" && !strings.EqualFold(pr.State, "OPEN") {
		return "pull request is " + strings.ToLower(pr.State)
	}
	if pr.IsDraft {
		// Drafts are picked up later by the ready_for_review trigger, which is
		// why that event is in the workflow's trigger list.
		return "pull request is a draft"
	}
	if len(pr.Assignees) > 0 {
		return "pull request already has an assignee"
	}
	if reason := priorAssignmentReason(pr.AssignedBy, selfLogin); reason != "" {
		return reason
	}
	if isOwnerLogin(pr.Author, ownerMap) {
		return "pull request was opened by a component owner, who shepherds it themselves"
	}
	return ""
}

// priorAssignmentReason reports why a past assignment blocks this one.
// Assignment is one-way: the bot gets a single turn per pull request, ever.
func priorAssignmentReason(assignedBy []string, selfLogin string) string {
	for _, actor := range assignedBy {
		if selfLogin == "" {
			return "pull request has been assigned before and the bot's identity is unknown"
		}
		if strings.EqualFold(actor, selfLogin) {
			return "pull request was already assigned by this bot once"
		}
	}
	return ""
}

// isOwnerLogin reports whether login is one of the configured component owners
// (case-insensitively, as GitHub logins are case-insensitive).
func isOwnerLogin(login string, ownerMap map[string]string) bool {
	if login == "" {
		return false
	}
	for _, owner := range ownerMap {
		if strings.EqualFold(owner, login) {
			return true
		}
	}
	return false
}

// hasBotContextComment reports whether the bot has already asked this author for
// more context. It only counts comments authored by the bot itself, so nobody
// can suppress the request by pasting the signature into their own comment.
func hasBotContextComment(pr PullRequest, selfLogin string) bool {
	for _, c := range pr.Comments {
		if isSelfAuthor(c.Author, selfLogin) && strings.Contains(c.Body, botCommentSignature) {
			return true
		}
	}
	return false
}

// isSelfAuthor reports whether a login is the bot's own resolved identity
// (case-insensitively).
//
// It deliberately does NOT trust the generic "[bot]" suffix: anyone can create a
// GitHub App whose login ends in "[bot]", so trusting the suffix would let an
// attacker post a comment carrying botCommentSignature and suppress the bot's
// own request. When the identity could not be resolved (selfLogin == "") this
// returns false, and the within-run claim remains the only guard.
func isSelfAuthor(login, selfLogin string) bool {
	return selfLogin != "" && strings.EqualFold(login, selfLogin)
}

// assemblePRContext builds the text handed to the model for one pull request.
//
// Trust boundary: the headers ("Pull request #12 opened by @alice", "Title:",
// "Changed files") are TRUSTED scaffolding generated here from GitHub API
// metadata and are emitted OUTSIDE the fence. Every author-controlled blob --
// the title, the body, and the changed-file paths, which are attacker-chosen
// filenames from the fork's branch -- is wrapped in its own
// [UNTRUSTED:nonce] ... [/UNTRUSTED:nonce] fence. Because the nonce is
// unguessable, the author can neither close a fence to escape it nor forge a
// trusted header inside one: any such attempt stays trapped as inert data.
//
// It is pure so it can be exhaustively table-tested.
func assemblePRContext(pr PullRequest, maxFiles int, nonce string) string {
	open, closeTag := "[UNTRUSTED:"+nonce+"]", "[/UNTRUSTED:"+nonce+"]"
	fence := func(s string) string { return open + "\n" + s + "\n" + closeTag }

	var sections []string

	author := pr.Author
	if author == "" {
		author = "an unknown account"
	} else {
		author = "@" + author
	}
	sections = append(sections, fmt.Sprintf("Pull request #%d opened by %s.", pr.Number, author))

	if title := clean(pr.Title, maxSnippetRunes); title != "" {
		sections = append(sections, "Title (untrusted):\n"+fence(title))
	}
	if body := clean(pr.Body, maxSnippetRunes); body != "" {
		sections = append(sections, "Description (untrusted):\n"+fence(body))
	} else {
		// State the absence outside the fence. "The description is empty" is a
		// fact the bot established, and it is the single strongest signal for the
		// missing-context decision, so it must not be forgeable from inside one.
		sections = append(sections, "Description: (empty)")
	}
	if files := fileList(pr.Files, maxFiles); files != "" {
		sections = append(sections, fmt.Sprintf("Changed file paths (untrusted, %s):\n%s",
			fileCountNote(len(pr.Files), maxFiles), fence(files)))
	}

	return strings.Join(sections, "\n\n")
}

// fileList renders at most maxFiles changed-file paths, one per line, each
// truncated. Returns "" when there are none.
func fileList(paths []string, maxFiles int) string {
	if maxFiles < 0 {
		maxFiles = 0
	}
	var b strings.Builder
	for i, p := range paths {
		if i >= maxFiles {
			break
		}
		if p = clean(p, maxPathRunes); p == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(p)
	}
	return b.String()
}

// fileCountNote describes how many paths are shown, so a truncated list cannot
// be mistaken for the whole change.
func fileCountNote(total, maxFiles int) string {
	if total > maxFiles {
		return fmt.Sprintf("showing the first %d of %d", maxFiles, total)
	}
	return fmt.Sprintf("%d in total", total)
}

// clean trims surrounding whitespace and truncates to maxRunes.
//
// It deliberately does NOT strip fenced code blocks: text hidden inside a ```
// fence would then never be reviewed, and the nonce fence -- not the absence of
// backticks -- is what makes the content inert.
func clean(s string, maxRunes int) string {
	return truncateRunes(strings.TrimSpace(s), maxRunes)
}

// truncateRunes shortens s to at most n runes, appending a marker when it trims.
// It is rune-based so a multibyte character at the boundary is never split into
// invalid UTF-8.
func truncateRunes(s string, n int) string {
	if n < 0 {
		n = 0
	}
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + " …[truncated]"
}

// buildContextComment renders the comment asking the author for more context.
//
// Every word of it comes from constants in this file. The model supplies only
// allow-listed keys, so nothing derived from the (attacker-authored) pull
// request text is ever posted under the repository's identity. Unknown keys are
// dropped here as well as rejected by the tool, so a future caller cannot widen
// what the bot can say by bypassing the tool.
//
// It returns "" when no key resolves, which the caller treats as "nothing to
// ask" rather than posting an empty comment.
func buildContextComment(keys []string) string {
	var bullets []string
	seen := make(map[string]bool, len(keys))
	// Render in the declared order of contextItems rather than the model's
	// order, so the same request always reads the same way.
	for _, it := range contextItems {
		for _, k := range keys {
			if strings.EqualFold(strings.TrimSpace(k), it.key) && !seen[it.key] {
				seen[it.key] = true
				bullets = append(bullets, "- "+it.text)
			}
		}
	}
	if len(bullets) == 0 {
		return ""
	}
	return fmt.Sprintf(
		"%s thanks for the pull request. Before a reviewer picks this up, could you add a little more detail?\n\n%s\n\n"+
			"Editing the description is enough — no need to push anything.",
		botCommentSignature, strings.Join(bullets, "\n"),
	)
}
