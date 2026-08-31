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
	"slices"
	"strings"
)

// botAlertSignature is the leading text of the comment the bot posts when it
// flags an issue. It is the human-readable header, not the identity check.
const botAlertSignature = "**Automated spam detection:**"

// botAlertMarker is what hasBotAlert actually matches on: an HTML comment,
// invisible in rendered Markdown, that only this bot emits.
//
// The visible signature is not enough on its own. In production the resolved
// identity is github-actions[bot], the login of every Actions workflow in the
// repository, so any sibling workflow that echoed issue-supplied text into a
// comment would let an attacker put the signature in an issue title and have
// hasBotAlert treat the issue as already handled. No such workflow exists here
// today -- this one holds the repository's only `issues: write` -- and matching
// on a string only this bot writes keeps that from becoming a suppression
// channel if one is added later.
const botAlertMarker = "<!-- adk-go-spam-detection-bot -->"

// maxSnippetRunes bounds how much of any single piece of user text (the issue
// body or one comment) is forwarded to the model. Long text is truncated; this
// keeps the prompt small and the run cheap.
const maxSnippetRunes = 1500

// maxSuspectRunes bounds the TOTAL text assembled for one issue, across every
// blob. maxSnippetRunes alone does not bound it: the fetch takes up to 100
// comments, so a thread bumped inside the freshness window could otherwise send
// on the order of 150k runes of attacker-authored text to the model, on every
// sweep, four times a day.
//
// It is set well above what a normal thread needs, because every rune of
// headroom is a comment an attacker cannot push out of the prompt by padding.
// When the bound does bite, assembleSuspectText says so in the prompt rather
// than letting the model judge a thread it was never told was partial.
const maxSuspectRunes = 40000

// truncNoteText is the TRUSTED annotation on a header whose blob was cut. The
// " …[truncated]" marker clean() leaves is inside the fence, and the prompt
// tells the model not to trust anything in there -- and an attacker can type it.
const truncNoteText = " [truncated for length]"

var truncNoteRunes = len([]rune(truncNoteText))

// maxReasonRunes bounds the model-authored detection_reason before it is posted
// under the bot's identity. The model is treated as attacker-influenced and Go
// does not control its output length, so an unbounded reason could exceed
// GitHub's comment-length limit -- which would fail the comment, leave the
// issue unlabeled, and at temperature 0 reproduce itself on every later sweep.
const maxReasonRunes = 500

// Comment is the normalized view of a single issue comment.
type Comment struct {
	Author string
	// Association is the commenter's GitHub author association (e.g.
	// FIRST_TIME_CONTRIBUTOR, NONE, MEMBER). It is a spam-likelihood prior fed to
	// the model, not a filter.
	Association string
	Body        string
}

// Issue is the normalized view of a GitHub issue used for spam review. It is
// deliberately small: only the fields needed to decide whether the content is
// spam.
type Issue struct {
	Number int
	Title  string
	Body   string
	Author string
	// Association is the issue author's GitHub author association (see Comment).
	Association string
	Labels      []string
	Comments    []Comment
	// UnfetchedComment counts comments the fetch window left behind. They are
	// content the model is never shown, so assembleSuspectText declares them
	// alongside the ones its own budget dropped -- otherwise a thread with more
	// than the window's worth of comments reaches the model looking complete.
	UnfetchedComment int
}

// maintainerSet builds a lowercased lookup set of maintainer logins. GitHub
// logins are case-insensitive, so normalizing here lets a maintainer configured
// as "Alice" match the API's "alice".
func maintainerSet(logins []string) map[string]bool {
	m := make(map[string]bool, len(logins))
	for _, l := range logins {
		if l = strings.ToLower(strings.TrimSpace(l)); l != "" {
			m[l] = true
		}
	}
	return m
}

// unknownAuthor is the placeholder header name for content whose author GitHub
// no longer resolves (a deleted account: GraphQL returns author: null).
const unknownAuthor = "(deleted account)"

// isIgnoredAuthor reports whether content from this login should be skipped when
// looking for spam: a configured maintainer. Their text is never sent to the
// model.
//
// The bot's OWN content is not covered here. In production the identity is
// github-actions[bot], which is the login of every Actions workflow in the
// repository, so skipping everything written under it would silently exempt any
// sibling workflow's comments from review too. isOwnAlert is the narrower test
// the callers use instead.
//
// An empty login is NOT ignored. GraphQL returns a null author for a deleted
// account, and treating that as ignorable let a spammer post and then delete
// the account: the comment stays visible on the issue and would never be
// reviewed again.
//
// A `[bot]` suffix is not ignored either. Only a GitHub App has such a login,
// and only an App a repository admin installed can comment here, so the set is
// small and known -- but it is not empty, and skipping it by suffix extends
// unconditional trust to every installed App. Naming the ones to skip in
// MAINTAINERS makes that trust explicit instead.
func isIgnoredAuthor(login string, maintainers map[string]bool) bool {
	// GitHub logins are case-insensitive; the set is lowercased to match.
	return login != "" && maintainers[strings.ToLower(login)]
}

// isOwnAlert reports whether a comment is one THIS bot posted: written under the
// bot's resolved identity AND carrying the marker only this bot emits. Both
// halves are needed -- the identity is shared with every other workflow in the
// repository, and the marker alone is public text anyone can paste.
func isOwnAlert(login, body, selfLogin string) bool {
	return isSelfAuthor(login, selfLogin) && strings.Contains(body, botAlertMarker)
}

// isSelfAuthor reports whether a login is the bot's own resolved identity
// (case-insensitive). Used to authenticate the bot's own alert comments.
//
// It deliberately does NOT trust the generic botSuffix suffix: anyone can create
// a GitHub App whose login ends in botSuffix, so trusting the suffix would let
// an attacker post a comment carrying the alert marker to make hasBotAlert treat
// the issue as already handled and suppress moderation.
//
// The selfLogin == "" guard is defence in depth rather than a live path:
// NewGitHubClient fails the run when the identity cannot be resolved, precisely
// because everything here would otherwise silently answer "not mine".
func isSelfAuthor(login, selfLogin string) bool {
	return selfLogin != "" && strings.EqualFold(login, selfLogin)
}

// hasLabel reports whether labels contains target (case-insensitive).
func hasLabel(labels []string, target string) bool {
	for _, l := range labels {
		if strings.EqualFold(strings.TrimSpace(l), target) {
			return true
		}
	}
	return false
}

// hasBotAlert reports whether the bot has already posted its alert comment on
// this issue. It requires both the bot's own resolved identity and the marker
// only this bot writes, so neither a spammer pasting the visible signature into
// their own comment nor another workflow running under the same
// github-actions[bot] login can make an issue look already handled.
func hasBotAlert(iss Issue, selfLogin string) bool {
	for _, c := range iss.Comments {
		if isOwnAlert(c.Author, c.Body, selfLogin) {
			return true
		}
	}
	return false
}

// alreadyHandled reports whether the issue has already been actioned and should
// be skipped before the model is invoked: it already carries the spam label, or
// the bot has already alerted on it. This is the idempotency gate that prevents
// duplicate labels and comments across runs (and in single-issue mode, where the
// search-time -label:spam filter does not apply).
func alreadyHandled(iss Issue, selfLogin, spamLabel string) bool {
	return hasLabel(iss.Labels, spamLabel) || hasBotAlert(iss, selfLogin)
}

// clean normalizes a piece of user text for review: it trims surrounding
// whitespace and truncates to maxRunes.
//
// It deliberately does NOT strip fenced code blocks (the Python original did):
// spam hidden inside a ``` fence would then never be reviewed. Keeping the text
// and bounding it with truncation closes that bypass while still capping tokens.
func clean(s string, maxRunes int) string {
	return truncateRunes(strings.TrimSpace(s), maxRunes)
}

// truncateRunes shortens s to at most n runes, appending a marker when it trims.
// A non-positive n yields the empty string rather than panicking on the slice.
func truncateRunes(s string, n int) string {
	if n <= 0 {
		return ""
	}
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + " …[truncated]"
}

// assembleSuspectText builds the text handed to the model for one issue: the
// issue's own title/body (when its author is reviewable) followed by each
// reviewable comment, with long text truncated. It returns "" when there is
// nothing to review (e.g. every author is a maintainer or a bot), which lets the
// caller skip the issue without invoking the model.
//
// Trust boundary: the authorship/association headers are TRUSTED scaffolding
// generated here from GitHub API metadata and are emitted OUTSIDE the fence.
// Each user-controlled blob (title+body, or a comment body) is wrapped in its
// own [UNTRUSTED:nonce] ... [/UNTRUSTED:nonce] fence. Because the nonce is
// unguessable, a spammer cannot close the fence to escape it, and because the
// headers live outside the fence they cannot forge a "Comment by @owner
// [author association: OWNER]" line inside their own text — any such attempt
// stays trapped inside the fence as inert data.
//
// It is pure so it can be exhaustively table-tested.
func assembleSuspectText(iss Issue, selfLogin string, maintainers map[string]bool, maxRunes int, nonce string) string {
	open, closeTag := "[UNTRUSTED:"+nonce+"]", "[/UNTRUSTED:"+nonce+"]"
	fence := func(s string) string { return open + "\n" + s + "\n" + closeTag }

	var sections []string
	// budget is the remaining allowance across the whole assembly, charged for
	// the trusted scaffolding as well as the text, so it bounds the assembled
	// prompt and not merely its untrusted parts. Each blob is still capped at
	// maxRunes; this stops a hundred capped blobs adding up.
	// Reserved up front for the omission note below, which is emitted after the
	// budget is spent and would otherwise push the assembly past the bound.
	budget := maxSuspectRunes - 256
	fenceCost := len([]rune(open)) + len([]rune(closeTag)) + len([]rune("\n\n---\n\n")) + 3

	// take returns as much of s as the budget allows, charged for the fence and
	// for the header the caller will put on it. It charges NOTHING when it
	// returns nothing, so a run of whitespace-only comments cannot drain the
	// allowance without contributing any text.
	// truncated records whether the last take() cut its input, so the caller can
	// say so in the TRUSTED header. The " …[truncated]" marker clean() leaves is
	// inside the fence, and the prompt tells the model not to trust anything in
	// there -- and an attacker can type the marker themselves.
	truncated := false
	take := func(s, header string) string {
		truncated = false
		// truncNoteRunes is charged unconditionally: the annotation is emitted
		// outside the fence when a blob is cut, and leaving it uncharged made
		// maxSuspectRunes a soft bound on exactly the threads that hit it.
		cost := fenceCost + len([]rune(header)) + truncNoteRunes
		if budget <= cost {
			return ""
		}
		allowed := min(maxRunes, budget-cost)
		trimmed := strings.TrimSpace(s)
		out := clean(s, allowed)
		if out == "" {
			return ""
		}
		// Against the ALLOWANCE, not against len(out): clean() appends a 13-rune
		// marker when it cuts, so comparing lengths reported "not truncated" for
		// any cut of 13 runes or fewer -- exactly the header signal the prompt
		// tells the model to rely on, silently absent.
		truncated = len([]rune(trimmed)) > allowed
		budget -= len([]rune(out)) + cost
		return out
	}

	// truncNote renders the trusted "this was cut" annotation.
	truncNote := func(cut bool) string {
		if cut {
			return truncNoteText
		}
		return ""
	}

	if !isIgnoredAuthor(iss.Author, maintainers) {
		header := fmt.Sprintf("Issue #%d opened by @%s%s", iss.Number, authorName(iss.Author), assocNote(iss.Association))
		var content strings.Builder
		cut := false
		if title := take(iss.Title, header); title != "" {
			content.WriteString("Title: " + title)
			cut = cut || truncated
		}
		if body := take(iss.Body, header); body != "" {
			if content.Len() > 0 {
				content.WriteString("\n")
			}
			content.WriteString("Body:\n" + body)
			cut = cut || truncated
		}
		if content.Len() > 0 {
			sections = append(sections, header+truncNote(cut)+"\n"+fence(content.String()))
		}
	}

	// The budget is spent NEWEST FIRST. FetchIssue asks for comments(last: N) and
	// GitHub returns those oldest-first, so spending it in slice order starved
	// the most recent comment — which on a thread the sweep selected precisely
	// because it was just updated is the one most likely to be new spam. The
	// sections are emitted back in thread order below, so the model still reads
	// the conversation the way it was written.
	var commentSections []string
	// Seeded with the comments the fetch never retrieved. A thread longer than
	// the fetch window would otherwise reach the model indistinguishable from a
	// complete one -- the same state the budget's own note exists to prevent,
	// reached by a different route and without any note at all.
	omitted := iss.UnfetchedComment
	for i := len(iss.Comments) - 1; i >= 0; i-- {
		c := iss.Comments[i]
		// The bot's own alerts are skipped by their marker, not by their author:
		// anything else written under the same shared Actions identity is
		// reviewed like any other comment.
		if isIgnoredAuthor(c.Author, maintainers) || isOwnAlert(c.Author, c.Body, selfLogin) {
			continue
		}
		header := fmt.Sprintf("Comment by @%s%s:", authorName(c.Author), assocNote(c.Association))
		body := take(c.Body, header)
		if body == "" {
			// Either the comment was empty, or the budget is gone. Only the
			// second is worth telling the model about.
			if strings.TrimSpace(c.Body) != "" {
				omitted++
			}
			continue
		}
		commentSections = append(commentSections, header+truncNote(truncated)+"\n"+fence(body))
	}
	slices.Reverse(commentSections)
	sections = append(sections, commentSections...)

	if len(sections) == 0 {
		return ""
	}
	if omitted > 0 {
		// A trusted line, outside every fence. Without it the model would judge
		// a partial thread as if it were the whole one, and answer "no spam"
		// over content it was never shown.
		sections = append(sections, fmt.Sprintf(
			"NOTE (trusted): %d comment(s) on this issue were omitted because the "+
				"review length limit was reached. Judge only what is shown; do not assume "+
				"the omitted comments were harmless.", omitted))
	}

	return strings.Join(sections, "\n\n---\n\n")
}

// authorName renders a login for a trusted header, naming a deleted account
// rather than emitting a bare "@".
func authorName(login string) string {
	if login == "" {
		return unknownAuthor
	}
	return login
}

// assocNote renders an author-association annotation for the prompt, e.g.
// " [author association: FIRST_TIME_CONTRIBUTOR]". It returns "" when the
// association is unknown so the prompt stays clean.
func assocNote(association string) string {
	if a := strings.TrimSpace(association); a != "" {
		return fmt.Sprintf(" [author association: %s]", a)
	}
	return ""
}

// buildAlertComment renders the maintainer-facing comment the bot posts when it
// flags an issue. It carries botAlertMarker (so the bot can recognize its own
// alerts) and embeds the model's reason as an inert fenced block so the reason
// text cannot break the comment's Markdown.
func buildAlertComment(reason string) string {
	// The reason is model-authored, so it is attacker-influenced. Bound it
	// before anything else: GitHub rejects a comment body over 65536 characters,
	// and a rejected comment leaves the issue unlabeled and un-alerted, which at
	// temperature 0 the next sweep would reproduce exactly.
	safe := truncateRunes(strings.TrimSpace(reason), maxReasonRunes)
	// Neutralize any fences in it so it cannot escape the code block below.
	safe = strings.ReplaceAll(safe, "```", "'''")
	if safe == "" {
		safe = "(no reason provided)"
	}
	return fmt.Sprintf(
		"%s a suspected spam comment was detected in this thread. "+
			"Maintainers, please review.\n\nReason:\n```text\n%s\n```\n%s",
		botAlertSignature, safe, botAlertMarker,
	)
}
