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
	"slices"
	"sort"
	"strings"
	"time"
)

// botAlertSignature is the leading text of the comment the bot posts when it
// detects a "silent" description edit. It must stay in sync with the body
// written by doAlertEdit so the bot can recognize its own alerts and avoid
// spamming the thread.
//
// It does not name the author: the alert also fires when someone else with write
// access edits the description, and the bot must not tell a public tracker that
// one person did something another person did.
const botAlertSignature = "**Notification:** The issue description was edited without a comment"

// botAlertBody is the COMPLETE body doAlertEdit posts, and the exact string the
// recognition below matches against.
//
// Matching the whole body rather than the signature prefix narrows the one route
// the identity check cannot close. Identity is shared: every ADK bot in this
// repository posts as github-actions[bot], so a sibling workflow's comment
// satisfies isSelfActor exactly as this bot's own does. Under a prefix match, a
// sibling that echoes user-supplied text at the START of a comment would carry
// whatever a commenter chose to put there — including this signature — and
// suppress a genuine alert. An exact match requires the sibling's entire comment
// to be this alert verbatim, which an echo of user text is not.
//
// This narrows the route rather than closing it. Closing it needs a marker a
// sibling cannot reproduce, and the per-issue nonce used for truncation cannot
// serve: recognition happens on a LATER run, and that nonce is not stable across
// runs. Recorded as a known residual rather than claimed as a fix.
const botAlertBody = botAlertSignature + ". Maintainers, please review."

// Role classifies the last human actor on an issue.
type Role string

const (
	roleAuthor     Role = "author"
	roleMaintainer Role = "maintainer"
	roleOther      Role = "other_user"
)

// eventType is the kind of a normalized history event.
type eventType string

const (
	eventCreated       eventType = "created"
	eventCommented     eventType = "commented"
	eventEditedComment eventType = "edited_comment"
	eventEditedDesc    eventType = "edited_description"
	eventRenamedTitle  eventType = "renamed_title"
	eventReopened      eventType = "reopened"
	eventUnlabeled     eventType = "unlabeled_stale"
)

// historyEvent is a single normalized, human-attributed event on the issue
// timeline.
type historyEvent struct {
	Type  eventType
	Actor string
	Time  time.Time
	Body  string // populated for comments only
}

// --- Raw GraphQL shapes -----------------------------------------------------
//
// These mirror the GraphQL response so the GitHub client can decode directly
// into them, while the pure functions below operate on this typed input
// (keeping them trivially unit-testable with struct literals).

type rawActor struct {
	Login string `json:"login"`
	// Typename is the GraphQL __typename ("Bot" for bot accounts). GitHub's
	// GraphQL API returns a bare bot login (e.g. "github-actions") without the
	// "[bot]" suffix that REST appends, so the type is the reliable bot signal.
	Typename string `json:"__typename"`
}

type rawComment struct {
	Author       *rawActor  `json:"author"`
	Body         string     `json:"body"`
	CreatedAt    time.Time  `json:"createdAt"`
	LastEditedAt *time.Time `json:"lastEditedAt"`
}

type rawEdit struct {
	Editor   *rawActor `json:"editor"`
	EditedAt time.Time `json:"editedAt"`
}

type rawTimelineItem struct {
	Typename  string    `json:"__typename"`
	CreatedAt time.Time `json:"createdAt"`
	Actor     *rawActor `json:"actor"`
	Label     *struct {
		Name string `json:"name"`
	} `json:"label"`
}

type rawIssue struct {
	Author    *rawActor `json:"author"`
	CreatedAt time.Time `json:"createdAt"`
	Labels    struct {
		Nodes []struct {
			Name string `json:"name"`
		} `json:"nodes"`
	} `json:"labels"`
	Comments struct {
		Nodes []rawComment `json:"nodes"`
	} `json:"comments"`
	UserContentEdits struct {
		Nodes []rawEdit `json:"nodes"`
	} `json:"userContentEdits"`
	TimelineItems struct {
		Nodes []rawTimelineItem `json:"nodes"`
	} `json:"timelineItems"`
}

// IssueState is the structured summary handed to the LLM. The JSON field names
// match the keys referenced by the prompt's decision tree.
type IssueState struct {
	Status string `json:"status"`
	// Error carries the reason when Status is not "success". It is separate from
	// the computed fields so a refusal never has to be squeezed into one of them
	// — last_action_type is documented to the model as an event kind, and a
	// free-form message there reads as a malformed enum value.
	Error                 string  `json:"error,omitempty"`
	LastActionRole        string  `json:"last_action_role"`
	LastActionType        string  `json:"last_action_type"`
	LastActorName         string  `json:"last_actor_name"`
	MaintainerAlertNeeded bool    `json:"maintainer_alert_needed"`
	IsStale               bool    `json:"is_stale"`
	DaysSinceActivity     float64 `json:"days_since_activity"`
	// DaysSinceLastActorAction ages the event that set last_action_role. It
	// equals days_since_activity except when the newest event is an edit of an
	// older comment by someone other than the last actor.
	DaysSinceLastActorAction float64 `json:"days_since_last_actor_action"`
	// DaysSinceAuthorAction ages the issue author's most recent action of any
	// kind, a comment edit included. It is how long the person the bot is waiting
	// on has been silent.
	DaysSinceAuthorAction float64  `json:"days_since_author_action"`
	DaysSinceStaleLabel   float64  `json:"days_since_stale_label"`
	LastCommentText       string   `json:"last_comment_text"`
	CurrentLabels         []string `json:"current_labels"`
	StaleThresholdDays    float64  `json:"stale_threshold_days"`
	CloseThresholdDays    float64  `json:"close_threshold_days"`
	Maintainers           []string `json:"maintainers"`
	IssueAuthor           string   `json:"issue_author"`
}

// sameLabel reports whether two label names denote the same GitHub label.
//
// GitHub label names are case-insensitively unique but preserve the casing they
// were created with, so a byte-exact comparison silently fails to recognize a
// label the repository actually carries — a repo whose label reads "Stale"
// against the default STALE_LABEL_NAME of "stale" reports is_stale=false
// forever, which kills the close and un-stale branches and re-posts the warning
// comment on every run. Every label comparison in this program goes through here
// so the read and write sides cannot drift apart.
func sameLabel(a, b string) bool { return strings.EqualFold(a, b) }

// isIgnoredActor reports whether events from this login should be ignored:
// empty actors, any "[bot]" account, and the bot's own identity.
func isIgnoredActor(login, selfLogin string) bool {
	// The suffix test folds case like every other login comparison here,
	// including the one in the next clause. GitHub emits the "[bot]" suffix
	// lowercase and actorLogin appends it lowercase, so no case-varying suffix
	// has been observed reaching this — the fold is for consistency in code
	// people copy, not for a demonstrated input.
	return login == "" || strings.HasSuffix(strings.ToLower(login), "[bot]") ||
		(selfLogin != "" && strings.EqualFold(login, selfLogin))
}

// isSelfActor reports whether a login is THIS bot's own resolved identity.
//
// It deliberately does not accept any "[bot]" login, which is what it used to
// do. The alert-suppression branch treats a matching comment as one this bot
// already posted, and botAlertSignature is a fixed literal in a public
// repository, so it is not a secret — identity is the only half of that pair
// an outsider cannot write. Admitting every bot account meant any GitHub App
// installed on the repository could permanently silence a genuine silent-edit
// alert by opening a comment with the signature. The route is not hypothetical
// here: all five ADK bots post as github-actions[bot], so a sibling workflow
// echoing user text that happened to start with the signature would suppress
// this bot's alert.
//
// The case fold stays: GitHub logins are case-insensitively unique and the API
// returns them in their registered casing, so a byte-exact match would let the
// bot fail to recognize its own comment.
//
// With no resolved identity the bot cannot tell its own comment from anyone
// else's, so it recognizes nothing and may repeat an alert. That is the safe
// direction to fail: a duplicate alert is noise, a suppressed one is the bot
// silenced on that issue for good.
func isSelfActor(login, selfLogin string) bool {
	return login != "" && selfLogin != "" && strings.EqualFold(login, selfLogin)
}

// buildTimeline normalizes the raw GraphQL data into a chronologically sorted
// list of human events. It also returns the times the stale label was applied
// and the most recent time the bot posted a silent-edit alert (used for spam
// prevention).
func buildTimeline(raw *rawIssue, selfLogin, staleLabel string) (events []historyEvent, staleLabelTimes []time.Time, lastBotAlert time.Time) {
	author := actorLogin(raw.Author)

	// Baseline: issue creation.
	events = append(events, historyEvent{Type: eventCreated, Actor: author, Time: raw.CreatedAt})

	// Comments.
	for _, c := range raw.Comments.Nodes {
		actor := actorLogin(c.Author)
		// Track the bot's own silent-edit alerts; never add them to history. The
		// identity check stops anyone else — a user, another GitHub App, or a
		// sibling workflow sharing this bot's login — from spoofing the
		// signature to suppress future genuine alerts.
		// An exact match on the whole body, not a prefix and not Contains: a
		// comment that merely quotes or begins with the alert must not suppress
		// the next genuine one. See botAlertBody for the shared-identity route
		// this narrows and the residual it leaves.
		if isSelfActor(actor, selfLogin) && c.Body == botAlertBody {
			if lastBotAlert.IsZero() || c.CreatedAt.After(lastBotAlert) {
				lastBotAlert = c.CreatedAt
			}
			continue
		}
		if isIgnoredActor(actor, selfLogin) {
			continue
		}
		// A comment occupies its position in the conversation at the time it was
		// POSTED. Ordering by the edit time instead let an author who tweaks an
		// old comment jump ahead of a maintainer who replied in between, so the
		// bot read the author as the last actor when the maintainer was.
		//
		// An edit is still activity, so it is recorded as its own event rather
		// than discarded: that keeps the staleness clock honest without
		// reordering the conversation.
		events = append(events, historyEvent{Type: eventCommented, Actor: actor, Time: c.CreatedAt, Body: c.Body})
		if c.LastEditedAt != nil && !c.LastEditedAt.IsZero() && c.LastEditedAt.After(c.CreatedAt) {
			events = append(events, historyEvent{Type: eventEditedComment, Actor: actor, Time: *c.LastEditedAt, Body: c.Body})
		}
	}

	// Description edits ("ghost edits").
	for _, e := range raw.UserContentEdits.Nodes {
		actor := actorLogin(e.Editor)
		if isIgnoredActor(actor, selfLogin) {
			continue
		}
		events = append(events, historyEvent{Type: eventEditedDesc, Actor: actor, Time: e.EditedAt})
	}

	// Timeline events: stale-label applications, title renames, reopens.
	for _, t := range raw.TimelineItems.Nodes {
		switch t.Typename {
		case "LabeledEvent":
			if t.Label != nil && sameLabel(t.Label.Name, staleLabel) {
				staleLabelTimes = append(staleLabelTimes, t.CreatedAt)
			}
			continue
		case "UnlabeledEvent":
			// A human removing the stale label is meaningful activity: record it
			// so the staleness clock resets and the bot does not immediately
			// re-mark the issue, overriding the person who cleared it. Removal of
			// other labels is ignored.
			if t.Label != nil && sameLabel(t.Label.Name, staleLabel) {
				actor := actorLogin(t.Actor)
				if !isIgnoredActor(actor, selfLogin) {
					events = append(events, historyEvent{Type: eventUnlabeled, Actor: actor, Time: t.CreatedAt})
				}
			}
			continue
		}
		actor := actorLogin(t.Actor)
		if isIgnoredActor(actor, selfLogin) {
			continue
		}
		var et eventType
		switch t.Typename {
		case "RenamedTitleEvent":
			et = eventRenamedTitle
		case "ReopenedEvent":
			et = eventReopened
		default:
			// Ignore unrequested/unknown timeline item types rather than
			// misattributing them (the query only asks for the four handled here).
			continue
		}
		events = append(events, historyEvent{Type: et, Actor: actor, Time: t.CreatedAt})
	}

	sort.SliceStable(events, func(i, j int) bool { return events[i].Time.Before(events[j].Time) })
	return events, staleLabelTimes, lastBotAlert
}

// replayResult captures the outcome of replaying the event history.
type replayResult struct {
	LastActorRole Role
	LastActivity  time.Time
	// LastActorAction is when the event that set LastActorRole happened. It
	// differs from LastActivity only for a comment edit, which advances the
	// clock without changing who spoke last, so a check that wants "did this
	// person act after X" has to read this rather than LastActivity.
	LastActorAction time.Time
	LastActionType  eventType
	LastActorName   string
	LastCommentText string
	LastCommentBy   string // login of LastCommentText's author
	// LastDescEdit is when the description was last edited by a human. The
	// silent-edit alert de-duplicates against this rather than against
	// LastActivity, because LastActivity is advanced by a comment edit from any
	// past commenter — which re-armed the alert and had the bot post the same
	// "the author has updated the description" comment once per run, for an edit
	// it had already reported.
	LastDescEdit time.Time
	// LastAuthorAction is the most recent event by the issue author, including a
	// comment edit. An edit does not change whose turn it is, so it must not move
	// LastActorAction — but an author who edits their comment to add the logs a
	// maintainer asked for HAS responded, and marking or closing the issue after
	// that breaks the warning comment's promise. This is the clock for "has the
	// person we are waiting on done anything".
	LastAuthorAction time.Time
}

// replay walks the sorted history to find the last human actor and their role.
//
// LastCommentText retains the most recent comment even when a later non-comment
// event (e.g. a title rename) becomes the last action, so the maintainer-intent
// analysis still has text to work with. LastCommentBy records who wrote it, so
// computeIssueState can avoid attributing one person's comment to another.
func replay(events []historyEvent, maintainers map[string]bool, author string) replayResult {
	st := replayResult{LastActorRole: roleAuthor, LastActionType: eventCreated, LastActorName: author}
	if len(events) > 0 {
		st.LastActivity = events[0].Time
	}
	for _, e := range events {
		// Editing an old comment is activity, so it advances the clock — but it
		// does not change who spoke LAST. Attributing authorship to the editor
		// let an author who tweaks a months-old comment displace a maintainer who
		// replied in between, flipping last_action_role from maintainer to author
		// and taking the issue down the wrong branch of the decision tree.
		if strings.EqualFold(e.Actor, author) {
			st.LastAuthorAction = e.Time
		}
		if e.Type == eventEditedComment {
			st.LastActivity = e.Time
			// An edit by whoever spoke last is that person still acting, so it
			// advances their clock. An edit by anyone else is activity on the
			// issue but not them coming back, and counting it as such let a
			// third party who once commented reset another person's turn.
			if strings.EqualFold(e.Actor, st.LastActorName) {
				st.LastActorAction = e.Time
			}
			continue
		}
		st.LastActorRole = classify(e.Actor, author, maintainers)
		st.LastActivity = e.Time
		st.LastActorAction = e.Time
		st.LastActionType = e.Type
		st.LastActorName = e.Actor
		if e.Type == eventEditedDesc {
			st.LastDescEdit = e.Time
		}
		if e.Type == eventCommented {
			st.LastCommentText = e.Body
			st.LastCommentBy = e.Actor
		}
	}
	return st
}

// classify decides whose turn it is, matching logins case-insensitively.
//
// GitHub logins are case-insensitive and the API returns them in their
// registered casing, so a maintainer configured as "wolo-lab" must still match
// an actor reported as "Wolo-Lab". Comparing verbatim silently demoted that
// maintainer to roleOther, which both suppressed STEP 3 and made STEP 1 strip
// the stale label the maintainer's own comment should have preserved. The
// sibling spam-detection bot already lowercases both sides; this matches it.
func classify(actor, author string, maintainers map[string]bool) Role {
	switch {
	case strings.EqualFold(actor, author):
		return roleAuthor
	case maintainers[strings.ToLower(actor)]:
		return roleMaintainer
	default:
		return roleOther
	}
}

// computeIssueState orchestrates timeline construction, replay, and the final
// staleness/alert calculations. It is pure: all inputs are explicit (including
// now) so it can be exhaustively unit-tested.
func computeIssueState(raw *rawIssue, selfLogin string, maintainers []string, staleLabel string, staleAfter, closeAfter time.Duration, now time.Time) IssueState {
	author := actorLogin(raw.Author)
	labels := make([]string, 0, len(raw.Labels.Nodes))
	for _, l := range raw.Labels.Nodes {
		labels = append(labels, l.Name)
	}

	events, staleLabelTimes, lastBotAlert := buildTimeline(raw, selfLogin, staleLabel)
	maintainerSet := toSet(maintainers)
	st := replay(events, maintainerSet, author)

	daysSinceActivity := now.Sub(st.LastActivity).Hours() / 24
	daysSinceLastActorAction := now.Sub(st.LastActorAction).Hours() / 24
	daysSinceAuthorAction := now.Sub(st.LastAuthorAction).Hours() / 24

	isStale := slices.ContainsFunc(labels, func(l string) bool { return sameLabel(l, staleLabel) })
	daysSinceStaleLabel := 0.0
	if isStale {
		if len(staleLabelTimes) > 0 {
			daysSinceStaleLabel = now.Sub(latest(staleLabelTimes)).Hours() / 24
		} else {
			// The stale LabeledEvent scrolled out of the bounded timeline window,
			// so the label's true age is unknown. The window holds the 50 most
			// recent items and the timeline only grows, so an event that has
			// scrolled out will not come back — this issue stays un-closeable
			// until a human intervenes.
			//
			// Substituting time-since-activity
			// was systematically biased toward closing early: the label is applied
			// AFTER the last activity, so that value is always >= the real age,
			// and the issue author can pad the window with title renames and
			// reopens to force this branch. Report -1 for "unknown"; the close
			// predicate refuses on it, so the issue is left alone rather than
			// closed on a guess.
			daysSinceStaleLabel = -1
		}
	}

	// Surface the last comment only when whoever wrote it holds the same role as
	// the last actor, so the maintainer-intent step never analyzes someone else's
	// words — the author's question, say — as if a maintainer had written them.
	//
	// The comparison is by ROLE, not by login. Requiring the same person meant one
	// maintainer asking for a repro and a second maintainer then retitling the
	// issue wiped the request, and stalePredicate refuses without a comment to
	// judge, so an ordinary two-maintainer triage could never go stale.
	lastCommentText := ""
	if st.LastCommentBy != "" && classify(st.LastCommentBy, author, maintainerSet) == st.LastActorRole {
		lastCommentText = st.LastCommentText
	}

	// Silent-edit alert: the author/user edited the description without
	// commenting, and we have not already alerted about an edit since then.
	alertNeeded := false
	if (st.LastActorRole == roleAuthor || st.LastActorRole == roleOther) && st.LastActionType == eventEditedDesc {
		// Compare against the edit itself, not against the activity clock. A
		// comment edit advances LastActivity without changing LastActionType, so
		// comparing there let anyone who had ever commented re-arm the alert and
		// make the bot re-report an edit it had already reported — one comment
		// per run, indefinitely.
		// One alert per edit, de-duplicated against the edit itself.
		//
		// A rate limit was tried here and reverted: capping alerts per window let
		// the author arm the suppression. A throwaway edit draws the alert, the
		// real edit lands inside the window and is never reported, and one
		// comment afterwards moves last_action_type off the description edit so
		// the branch can never be re-entered. Trading a bounded, attributable
		// spam vector for a deliberate way to hide an edit is the wrong side of
		// that trade: reporting the edit is the whole point of this branch.
		if lastBotAlert.IsZero() || !lastBotAlert.After(st.LastDescEdit) {
			alertNeeded = true
		}
	}

	return IssueState{
		Status:                   "success",
		LastActionRole:           string(st.LastActorRole),
		LastActionType:           string(st.LastActionType),
		LastActorName:            st.LastActorName,
		MaintainerAlertNeeded:    alertNeeded,
		IsStale:                  isStale,
		DaysSinceActivity:        daysSinceActivity,
		DaysSinceLastActorAction: daysSinceLastActorAction,
		DaysSinceAuthorAction:    daysSinceAuthorAction,
		DaysSinceStaleLabel:      daysSinceStaleLabel,
		LastCommentText:          lastCommentText,
		CurrentLabels:            labels,
		StaleThresholdDays:       staleAfter.Hours() / 24,
		CloseThresholdDays:       closeAfter.Hours() / 24,
		Maintainers:              maintainers,
		IssueAuthor:              author,
	}
}

func actorLogin(a *rawActor) string {
	if a == nil {
		return ""
	}
	// Canonicalize a bot login to the REST "[bot]" form. GraphQL returns the bare
	// login (e.g. "github-actions"), but selfLogin is resolved via REST
	// ("github-actions[bot]") and the ignore/self filters match on the "[bot]"
	// suffix; without this, bot activity would be counted as a human's and the
	// bot could fail to recognize its own alert comments.
	if a.Typename == "Bot" && !strings.HasSuffix(a.Login, "[bot]") {
		return a.Login + "[bot]"
	}
	return a.Login
}

// toSet builds the maintainer lookup set, lowercased and trimmed so classify
// can match a GitHub login regardless of its registered casing. Blank entries
// are dropped so a trailing comma in MAINTAINERS cannot admit the empty login.
func toSet(xs []string) map[string]bool {
	m := make(map[string]bool, len(xs))
	for _, x := range xs {
		// normalizeLogin, not just TrimSpace: a maintainer list may reach here
		// without passing through splitList — computeIssueState takes the slice
		// directly, and anyone copying this example may build it themselves.
		if x = strings.ToLower(normalizeLogin(x)); x != "" {
			m[x] = true
		}
	}
	return m
}

func latest(ts []time.Time) time.Time {
	var newest time.Time
	for _, t := range ts {
		if t.After(newest) {
			newest = t
		}
	}
	return newest
}
