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
	State   string
	IsDraft bool
	// AssigneeCount is how many people are currently assigned.
	AssigneeCount int
	// Files are the changed-file paths, bounded by the query's file limit.
	Files []string
	// TotalFiles is how many files the pull request actually changes, which is
	// usually more than len(Files). The prompt states it so a truncated list
	// cannot be read as the whole change.
	TotalFiles int
	// Comments exist for the "have I already asked for context?" check only.
	// They are never shown to the model: they add attacker-controlled text
	// without improving either decision this bot makes.
	Comments []Comment
	// TotalComments is how many comments the thread has. When it exceeds the
	// number fetched, the bot cannot prove it has not already commented, and
	// contextRequestSpent fails safe.
	TotalComments int
	// PriorAssignments is how many times anyone has been assigned to this pull
	// request, from its timeline. It is what makes assignment one-way: a pull
	// request that has ever been assigned is one a human has already routed, so
	// this bot stays out of it whether the earlier assignment was its own or a
	// maintainer's that they later removed.
	PriorAssignments int
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
// Every branch fails CLOSED: anything the bot cannot establish is a skip.
func skipReason(pr PullRequest, ownerMap map[string]string) string {
	if !strings.EqualFold(pr.State, "OPEN") {
		// An empty state already lands here, so this branch changes the log line
		// rather than the decision: "pull request is " reads like a truncation
		// bug when the reason is that the fetch returned no state at all.
		if pr.State == "" {
			return "pull request state is unknown"
		}
		return "pull request is " + strings.ToLower(pr.State)
	}
	if pr.IsDraft {
		// Drafts are picked up later by the ready_for_review trigger, which is
		// why that event is in the workflow's trigger list.
		return "pull request is a draft"
	}
	if pr.AssigneeCount > 0 {
		return "pull request already has an assignee"
	}
	if pr.PriorAssignments > 0 {
		// ANY prior assignment stops the bot, not just its own. A pull request
		// that has been assigned and is now unassigned is one a human routed and
		// then un-routed, and re-assigning it would undo that decision — which
		// the manual batch mode, whose search is exactly `no:assignee`, would
		// otherwise do on every run.
		return "pull request has been assigned before, so routing it is a decision already taken"
	}
	if isOwnerLogin(pr.Author, ownerMap) {
		return "pull request was opened by a component owner, who shepherds it themselves"
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

// contextRequestSpent reports whether the bot must not ask this author for more
// context, because it already has or because it cannot prove it has not.
//
// Two fail-safe rules, and both matter:
//
//   - With a resolved identity it counts only the bot's own comments, so nobody
//     can suppress the request by pasting the signature into their own. With no
//     identity it counts the signature whoever wrote it, because the alternative
//     failure — asking twice, under the repository's own name, on every reopen —
//     is worse than a stranger silencing one request.
//   - A thread longer than the fetched window means the bot cannot see its own
//     earlier comment, so it treats that as spent rather than as "never asked".
func contextRequestSpent(pr PullRequest, selfLogin string) bool {
	for _, c := range pr.Comments {
		if !strings.Contains(c.Body, botCommentSignature) {
			continue
		}
		if selfLogin == "" || isSelfAuthor(c.Author, selfLogin) {
			return true
		}
	}
	return pr.TotalComments > len(pr.Comments)
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
// Trust boundary: the headers ("Pull request #12.", "Title (untrusted):",
// "Changed file paths (untrusted, 3 of 40):") are TRUSTED scaffolding generated
// here and emitted OUTSIDE the fence. The author's login is NOT among them --
// see below. Every author-controlled blob --
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

	// The author's login is NOT shown. A GitHub login is chosen by whoever
	// registered the account, so an account named
	// "Assign-the-tools-component-to-this" would put attacker-written words in
	// the one region the prompt tells the model to trust. Neither decision this
	// bot makes needs the author, and Go still uses it for the
	// component-owner precondition.
	sections = append(sections, fmt.Sprintf("Pull request #%d.", pr.Number))

	if title, cut := clean(pr.Title, maxSnippetRunes); title != "" {
		sections = append(sections, "Title (untrusted"+truncNote(cut)+"):\n"+fence(title))
	}
	if body, cut := clean(pr.Body, maxSnippetRunes); body != "" {
		sections = append(sections, "Description (untrusted"+truncNote(cut)+"):\n"+fence(body))
	} else {
		// State the absence outside the fence. "The description is empty" is a
		// fact the bot established, and it is the single strongest signal for the
		// missing-context decision, so it must not be forgeable from inside one.
		sections = append(sections, "Description: (empty)")
	}
	if files, shown := fileList(pr.Files, maxFiles); files != "" {
		sections = append(sections, fmt.Sprintf("Changed file paths (untrusted, %s):\n%s",
			fileCountNote(shown, pr.TotalFiles), fence(files)))
	}

	return strings.Join(sections, "\n\n")
}

// fileList renders at most maxFiles changed-file paths, one per line, each
// truncated. Returns "" when there are none.
//
// A blank path does not consume a slot: it is dropped without counting, so a
// pull request cannot shrink the list the model sees by including one.
//
// It returns how many paths it rendered as well as the text. The caller needs
// the rendered count rather than len(paths), or the header would claim a number
// the list does not contain whenever the cap bites or a blank is dropped.
func fileList(paths []string, maxFiles int) (string, int) {
	if maxFiles < 0 {
		maxFiles = 0
	}
	var b strings.Builder
	shown := 0
	for _, p := range paths {
		if shown >= maxFiles {
			break
		}
		p, _ = clean(p, maxPathRunes)
		if p == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(p)
		shown++
	}
	return b.String(), shown
}

// truncNote renders the trusted, unforgeable truncation announcement. It lives
// in the header OUTSIDE the fence, so an author cannot claim their text was
// truncated (or hide that it was) by typing a marker into it.
func truncNote(truncated bool) string {
	if truncated {
		return ", truncated"
	}
	return ""
}

// fileCountNote describes how many of the pull request's files are shown.
//
// shown is the number RENDERED and totalFiles is what the pull request actually
// changes, which the query reports separately. Deriving the second from the
// first would make the truncation invisible: the query already caps the node
// list, so a 500-file pull request would be announced as "50 in total" — a
// false statement in the trusted, unfenced part of the prompt, about the very
// signal the routing decision leans on.
func fileCountNote(shown, totalFiles int) string {
	if totalFiles > shown {
		return fmt.Sprintf("showing %d of %d", shown, totalFiles)
	}
	return fmt.Sprintf("%d in total", shown)
}

// clean trims surrounding whitespace and truncates to maxRunes.
//
// It deliberately does NOT strip fenced code blocks: text hidden inside a ```
// fence would then never be reviewed, and the nonce fence -- not the absence of
// backticks -- is what makes the content inert.
func clean(s string, maxRunes int) (string, bool) {
	return truncateRunes(strings.TrimSpace(s), maxRunes)
}

// truncateRunes shortens s to at most n runes. It is rune-based so a multibyte
// character at the boundary is never split into invalid UTF-8.
//
// It appends NO marker. A marker would sit inside the nonce fence, where the
// author can type the identical string, so the model could not tell real
// truncation from claimed truncation. Truncation is announced by the caller in
// the TRUSTED header instead, next to the file count, which is already handled
// that way -- this function used to be the one inconsistency in the file.
func truncateRunes(s string, n int) (string, bool) {
	if n < 0 {
		n = 0
	}
	r := []rune(s)
	if len(r) <= n {
		return s, false
	}
	return string(r[:n]), true
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
