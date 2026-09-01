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
	"regexp"
	"strings"
	"unicode"
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

// truncNoteText is the TRUSTED annotation on a header whose text was cut. The
// " …[truncated]" marker clean() leaves is inside the fence, and the prompt
// tells the model not to trust anything in there -- and an attacker can type it.
const truncNoteText = " [truncated for length]"

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
	// Comments are fetched for the idempotency checks only -- hasBotAlert reads
	// them. They are never sent to the model; see assembleSuspectText.
	Comments []Comment
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

// fenceMarkerPattern matches anything in user text that could be read as one of
// the fence markers, in any case.
var fenceMarkerPattern = regexp.MustCompile(`(?i)\[/?UNTRUSTED:`)

// defangFenceMarkers rewrites text that could be mistaken for a fence marker.
//
// The nonce makes a real marker unguessable, so an attacker cannot forge one --
// but a model asked to respect a boundary is still swayed by text that merely
// LOOKS like the boundary. Measured against a real Gemini model, appending a
// fake close plus prose asserting the region had ended and the content was
// pre-approved suppressed detection of otherwise-obvious spam, where the same
// spam without it was flagged every time. Telling the prompt to ignore such
// text was already there and did not hold. The exact payload lives in
// TestE2EInstructionEvasionRate rather than here.
//
// So the capability is removed rather than argued with. Swapping the bracket
// for a parenthesis is the same length, so truncation and the assembly budget
// are unaffected, and the text stays legible to a human reading the prompt.
//
// This does NOT fix the evasion, because the persuasion is carried by the prose
// and not by the marker; it only takes one capability away from the attacker.
// See README.md for the measured residual, and for an inversion that reported
// the neutralized text to the model as evidence and was not shipped.
func defangFenceMarkers(s string) string {
	return fenceMarkerPattern.ReplaceAllStringFunc(s, func(m string) string {
		return "(" + m[1:]
	})
}

// clean normalizes a piece of user text for review: it trims surrounding
// whitespace, defangs anything resembling a fence marker, and truncates to
// maxRunes.
//
// It deliberately does NOT strip fenced code blocks (the Python original did):
// spam hidden inside a ``` fence would then never be reviewed. Keeping the text
// and bounding it with truncation closes that bypass while still capping tokens.
func clean(s string, maxRunes int) string {
	return truncateRunes(defangFenceMarkers(strings.TrimSpace(s)), maxRunes)
}

// stripInvisible removes characters that occupy no width but change what a
// reader sees: Unicode format characters (the bidirectional overrides, the
// zero-width space and joiners) and control characters other than newline and
// tab.
//
// This runs on text about to be posted publicly under the project's identity. A
// fenced code block stops a URL being clickable and stops an @mention pinging
// anyone, but it does NOT stop a bidirectional override reordering the rendered
// line, so the fence alone would let a reason display something other than what
// it says. Tab and newline are kept because they are the only invisible
// characters with a legitimate use in a reason.
func stripInvisible(s string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r == '\n' || r == '\t':
			return r
		case unicode.Is(unicode.Cf, r) || unicode.IsControl(r):
			return -1
		default:
			return r
		}
	}, s)
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

// assembleSuspectText builds the text handed to the model for one issue: its
// title and body, and nothing else. It returns "" when there is nothing to
// review -- the author is a maintainer, or the issue is empty -- which lets the
// caller skip the issue without invoking the model.
//
// COMMENTS ARE NOT INCLUDED, and that is the point rather than an omission.
// Flagging acts on the whole issue: it labels the thread and posts an alert on
// it. So any content the review is allowed to rest on is content a third party
// could use to get someone else's legitimate issue marked as spam -- a stranger
// leaves one promotional comment on a maintainer's bug report and the bot
// labels the bug report. The issue's author is answerable for the title and the
// body and for nothing else, so that is the whole of the input.
//
// This is enforced here rather than asked of the model. The prompt does say the
// bot judges only the issue's own text, but a prompt cannot be relied on to
// resist text that argues with it, and comment bodies are attacker-controlled.
// Not putting them in the prompt is the only version of this rule that holds.
//
// An issue OPENED under the bot's own identity is still reviewed. In production
// that identity is github-actions[bot], the login of every Actions workflow in
// the repository, so exempting it would exempt any sibling workflow that files
// an issue quoting user text. Only the maintainer set skips a review.
//
// Trust boundary: the authorship/association header is TRUSTED scaffolding
// generated here from GitHub API metadata and is emitted OUTSIDE the fence. The
// user-controlled title and body are wrapped in a single
// [UNTRUSTED:nonce] ... [/UNTRUSTED:nonce] fence. Because the nonce is
// unguessable, a spammer cannot close the fence to escape it, and because the
// header lives outside the fence they cannot forge a trusted authorship line --
// any such attempt stays trapped inside the fence as inert data.
//
// It is pure so it can be exhaustively table-tested.
func assembleSuspectText(iss Issue, maintainers map[string]bool, maxRunes int, nonce string) string {
	if isIgnoredAuthor(iss.Author, maintainers) {
		return ""
	}

	open, closeTag := "[UNTRUSTED:"+nonce+"]", "[/UNTRUSTED:"+nonce+"]"
	header := fmt.Sprintf("Issue #%d opened by @%s%s", iss.Number, authorName(iss.Author), assocNote(iss.Association))

	var content strings.Builder
	cut := false
	if title, trimmed := clean(iss.Title, maxRunes), strings.TrimSpace(iss.Title); title != "" {
		content.WriteString("Title: " + title)
		cut = cut || len([]rune(trimmed)) > maxRunes
	}
	if body, trimmed := clean(iss.Body, maxRunes), strings.TrimSpace(iss.Body); body != "" {
		if content.Len() > 0 {
			content.WriteString("\n")
		}
		content.WriteString("Body:\n" + body)
		cut = cut || len([]rune(trimmed)) > maxRunes
	}
	if content.Len() == 0 {
		return ""
	}

	// The cut is announced in the TRUSTED header. clean() leaves a marker inside
	// the fence, but the prompt tells the model not to trust anything in there --
	// and an attacker can type that marker themselves.
	return header + truncNote(cut) + "\n" + open + "\n" + content.String() + "\n" + closeTag
}

// truncNote renders the trusted "this was cut" annotation.
func truncNote(cut bool) string {
	if cut {
		return truncNoteText
	}
	return ""
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
	// The reason is model-authored, so it is attacker-influenced, and this is the
	// bot's only channel for putting text into a public place under the
	// project's identity. Everything below treats it as hostile.
	//
	// Strip invisible characters FIRST, before anything measures or matches on
	// the text. A bidirectional override reorders what a human sees without
	// changing what the bytes say, and a zero-width space splits a word so it
	// reads normally but matches nothing -- both survive inside a code block,
	// which is why the fence below is not sufficient on its own.
	safe := stripInvisible(strings.TrimSpace(reason))
	// Then neutralize the bot's own identity marker. A model talked into emitting
	// it would otherwise publish, under the shared github-actions[bot] identity,
	// a comment carrying the one string that makes this bot treat a comment as
	// its own prior alert. Doing this before the truncation below means a cut
	// cannot reassemble a marker out of a partial one.
	safe = strings.ReplaceAll(safe, botAlertMarker, "(marker removed)")
	// Bound the length: GitHub rejects a comment body over 65536 characters, and
	// a rejected comment leaves the issue unlabeled and un-alerted, which at
	// temperature 0 the next sweep would reproduce exactly.
	safe = truncateRunes(safe, maxReasonRunes)
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
