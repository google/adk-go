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
	"context"
	"errors"
	"fmt"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// auditedIssueKey scopes a session to a single issue number. The runner builds
// the invocation context from the context passed to Run (which embeds it), so a
// value set here is visible to every tool via ctx.Value.
type auditedIssueKey struct{}

// withAuditedIssue binds the issue this session is allowed to mutate.
func withAuditedIssue(ctx context.Context, number int) context.Context {
	return context.WithValue(ctx, auditedIssueKey{}, number)
}

// authorizeIssue reports whether the tool may act on the requested issue. It is
// the defense against prompt injection: untrusted issue content cannot make the
// agent mutate an issue other than the one this session is auditing.
func authorizeIssue(ctx context.Context, requested int) (string, bool) {
	audited, ok := ctx.Value(auditedIssueKey{}).(int)
	if !ok {
		return "no issue is authorized for this session", false
	}
	if requested != audited {
		return fmt.Sprintf("session is scoped to issue #%d; refusing to act on issue #%d", audited, requested), false
	}
	return "", true
}

// isManagedLabel reports whether the bot is allowed to add/remove this label.
// It only ever manages the stale and request-clarification labels.
//
// The comparison is case-insensitive for the same reason every other label
// comparison is (see sameLabel). A byte-exact allow-list refuses in the safe
// direction, but it refuses the LEGITIMATE calls too: with the repository's label
// reading "Stale", removing it once the author came back was rejected as an
// unmanaged label, so the bot could mark an issue stale and never un-mark it.
// Widening this does not widen authority — the stale label is still refused just
// below, and every other label still falls through to the final refusal.
func (c *GitHubClient) isManagedLabel(label string) bool {
	if label == "" {
		return false
	}
	return sameLabel(label, c.cfg.StaleLabel) || sameLabel(label, c.cfg.RequestClarificationLabel)
}

func errResult(format string, a ...any) actionResult {
	return actionResult{Status: "error", Message: fmt.Sprintf(format, a...)}
}

// issueArg is the input for tools that operate on a single issue.
type issueArg struct {
	IssueNumber int `json:"issue_number"`
}

// labelArg is the input for label tools.
type labelArg struct {
	IssueNumber int    `json:"issue_number"`
	Label       string `json:"label"`
}

// actionResult is the typed result returned by mutating tools.
type actionResult struct {
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
}

var okResult = actionResult{Status: "success"}

// The do* methods below are the tool handler bodies, extracted so they are
// directly unit-testable (the functiontool closures are thin wrappers). Each
// enforces per-issue authorization first; validation failures return a
// model-readable errResult with a nil Go error, while infrastructure failures
// record a tool error (so the run fails loudly) and return the Go error.

// doGetIssueState fetches an issue's state. It is authorized like the mutating
// tools so untrusted content cannot make the model pull an out-of-scope issue's
// data into context.
func (c *GitHubClient) doGetIssueState(ctx context.Context, number int) (IssueState, error) {
	if msg, ok := authorizeIssue(ctx, number); !ok {
		// The message goes in its own field: last_action_type is documented to
		// the model as a computed event kind, and a free-form refusal there
		// reads as a malformed enum value.
		return IssueState{Status: "error", Error: msg}, nil
	}
	st, err := c.GetIssueState(ctx, number)
	if err != nil {
		c.recordToolError()
		return IssueState{}, err
	}
	// Draw the fence marker BEFORE recording the observation. Recording first
	// would leave a passing observation behind on the error path, which a later
	// destructive tool could claim against even though the model never saw the
	// state -- passing the mechanical gate while skipping the judgement the
	// prompt is there to make.
	nonce, err := newNonce()
	if err != nil {
		c.recordToolError()
		return IssueState{}, err
	}
	// Keep the unfenced state: the destructive tools re-check their mechanical
	// preconditions against this, not against the model's assertion, so those
	// checks must read plain values.
	c.recordObservation(number, st)
	// Fence the one attacker-controlled field before it reaches the model, under
	// a marker drawn for this issue alone.
	st.LastCommentText = fenceUntrusted(st.LastCommentText, nonce)
	return st, nil
}

// labelRaceGrace is how much older than the stale label a user's action may be
// and still count as happening "after" it, expressed in days.
//
// An audit reads the issue state and then writes the label a model round trip
// later, so a comment landing in that window genuinely predates the label by a
// few seconds while being, in every sense that matters, the user coming back.
// Without the grace such an issue is stuck: the removal is refused because the
// label postdates the comment, and the close is refused because the author acted
// last. An hour is far more than one audit (ISSUE_TIMEOUT defaults to five
// minutes) and far less than the gap in the case the check is really for, which
// is a maintainer labelling an issue that has been quiet for weeks.
const labelRaceGrace = 1.0 / 24

// checkStalePrecondition enforces, in Go, the mechanical half of STEP 3 of the
// decision tree: an issue may be marked stale only if it is not already stale,
// a maintainer acted last, and the author has been silent past the threshold.
//
// The judgement half of STEP 3 — whether the maintainer's comment is actually
// blocked on the author — genuinely requires the model and stays in the prompt.
// Splitting it this way means injected text in an issue comment can at worst
// make the bot decline to act, never make it act outside the threshold, which
// is the failure adk-python cited when it deleted its own triage agent.
func stalePredicate(number int) func(IssueState) (string, bool) {
	return func(st IssueState) (string, bool) {
		if st.Status != "success" {
			return fmt.Sprintf("issue #%d state was not retrieved successfully; refusing to act", number), false
		}
		if st.IsStale {
			return fmt.Sprintf("issue #%d is already stale", number), false
		}
		if st.LastActionRole != string(roleMaintainer) {
			return fmt.Sprintf("issue #%d was last acted on by %q, not a maintainer; only a maintainer-blocked issue can be marked stale", number, st.LastActionRole), false
		}
		// There must be a maintainer comment to judge. The role alone does not
		// mean anything was asked: reopening an issue, retitling it, or removing
		// the stale label all confer roleMaintainer just as a comment does, and
		// on any of them the bot would post "…after a maintainer requested
		// clarification" about a request that never happened, then close the
		// issue a week later. The prompt's STEP 3 already needs this text to
		// decide anything, so requiring it here costs nothing legitimate.
		if st.LastCommentText == "" {
			return fmt.Sprintf("issue #%d has no maintainer comment to judge; a rename, reopen or label change is not a request for clarification", number), false
		}
		// The actor clock, not the activity clock. The check above guarantees a
		// maintainer acted last, so this is exactly how long the author has been
		// silent since the request. days_since_activity is also advanced by a
		// comment edit from anyone who ever commented, which would let a stranger
		// keep any issue out of reach of the bot by re-saving an old comment.
		if st.DaysSinceLastActorAction <= st.StaleThresholdDays {
			return fmt.Sprintf("issue #%d has been waiting %.1f days, at or below the %.1f-day stale threshold", number, st.DaysSinceLastActorAction, st.StaleThresholdDays), false
		}
		// And the author must not have answered since the request. A comment EDIT
		// by the author does not change whose turn it is, so it leaves the
		// maintainer as the last actor and slips past the check above — but an
		// author who edits their comment to add the logs that were asked for has
		// answered, and marking that stale is the mistake this bot exists to
		// avoid.
		//
		// The test is ORDERING, not a threshold. Asking whether the author has
		// been quiet for StaleThresholdDays marks the issue stale once their
		// answer ages past the threshold, which is a wrong write on an issue
		// whose ball is in the maintainers' court. Asking whether they acted
		// after the maintainer refuses for as long as that stays true, and a
		// fresh maintainer action clears it — which is exactly the intent.
		if st.DaysSinceAuthorAction < st.DaysSinceLastActorAction {
			return fmt.Sprintf("issue #%d: its author acted %.1f days ago, after the maintainer's %.1f-day-old request; the ball is in the maintainers' court", number, st.DaysSinceAuthorAction, st.DaysSinceLastActorAction), false
		}
		return "", true
	}
}

// checkClosePrecondition enforces STEP 1's close branch in Go. Every condition
// there is mechanical, so all of it is checked here.
func closePredicate(number int) func(IssueState) (string, bool) {
	return func(st IssueState) (string, bool) {
		if st.Status != "success" {
			return fmt.Sprintf("issue #%d state was not retrieved successfully; refusing to act", number), false
		}
		if !st.IsStale {
			return fmt.Sprintf("issue #%d is not marked stale; it cannot be closed as stale", number), false
		}
		if st.LastActionRole != string(roleMaintainer) {
			return fmt.Sprintf("issue #%d was last acted on by %q; the author or another user responded, so it must not be closed", number, st.LastActionRole), false
		}
		if st.DaysSinceStaleLabel < 0 {
			return fmt.Sprintf("issue #%d has been stale for an unknown length of time (the label event is outside the timeline window); refusing to close on a guess", number), false
		}
		if st.DaysSinceStaleLabel <= st.CloseThresholdDays {
			return fmt.Sprintf("issue #%d has been stale %.1f days, at or below the %.1f-day close threshold", number, st.DaysSinceStaleLabel, st.CloseThresholdDays), false
		}
		// The warning comment promises the issue closes only "if no further
		// activity occurs within N days", so recent activity has to stop it —
		// not just activity that changes whose turn it is. The role check above
		// does not cover this: an author who answers and a maintainer who then
		// replies leave the role at "maintainer" while the label keeps ageing,
		// so the issue was closed as not-planned days after it was answered.
		//
		// Activity RESTARTS the close clock, it does not cancel it. Comparing
		// against the label age instead would be permanent: both values are
		// "days since", so they grow in lockstep and their difference never
		// changes, which left an issue that anyone had touched once un-closeable
		// forever while every other tool also refused it.
		//
		// The actor clock again. days_since_activity is advanced by a comment
		// edit from anyone who ever commented on the issue — a change nobody is
		// notified of and nobody can see — so gating on it let a stranger hold
		// any issue open indefinitely by re-saving an old comment once a week.
		// An edit by whoever spoke last does advance this clock, so the promise
		// still holds for everyone with a stake in the conversation.
		if st.DaysSinceLastActorAction <= st.CloseThresholdDays {
			return fmt.Sprintf("issue #%d was last acted on %.1f days ago, inside the %.1f-day close window; the warning promised no close while activity continues", number, st.DaysSinceLastActorAction, st.CloseThresholdDays), false
		}
		// Same gap as in stalePredicate: an author answering by editing their own
		// comment leaves the maintainer as the last actor, so without this the bot
		// closes an issue that was answered. Ordering again, not a threshold —
		// their answer does not stop mattering once it is a week old.
		if st.DaysSinceAuthorAction < st.DaysSinceLastActorAction {
			return fmt.Sprintf("issue #%d: its author acted %.1f days ago, after the last maintainer action %.1f days ago; it is not waiting on them", number, st.DaysSinceAuthorAction, st.DaysSinceLastActorAction), false
		}
		return "", true
	}
}

// removeStalePredicate enforces STEP 1's "user came back" branch: the stale
// label may be stripped only from an issue that is stale and whose last actor
// was the author or another user, never a maintainer.
func removeStalePredicate(number int) func(IssueState) (string, bool) {
	return func(st IssueState) (string, bool) {
		if st.Status != "success" {
			return fmt.Sprintf("issue #%d state was not retrieved successfully; refusing to act", number), false
		}
		if !st.IsStale {
			return fmt.Sprintf("issue #%d is not marked stale; there is no stale label to remove", number), false
		}
		// The author answering by editing their own comment does not change whose
		// turn it is, so the role stays "maintainer" and the branch below refuses
		// — leaving the issue labelled stale, unclosable (the close refuses for
		// the same reason) and unclearable. Treat the author acting after the
		// label as them coming back, whatever it did to the turn order.
		// "and nobody has spoken since" is the second half. Without it, a
		// maintainer who re-affirms the label after the author's edit — "still
		// need the repro, leaving this stale" — is overridden anyway, because the
		// escape looked only at the label and never at the last actor.
		authorCameBack := st.DaysSinceStaleLabel >= 0 &&
			st.DaysSinceAuthorAction < st.DaysSinceStaleLabel+labelRaceGrace &&
			st.DaysSinceAuthorAction <= st.DaysSinceLastActorAction
		if !authorCameBack && st.LastActionRole == string(roleMaintainer) {
			return fmt.Sprintf("issue #%d was last acted on by a maintainer, so it is still waiting on the author; the stale label must stay", number), false
		}
		// "The user came back" means they acted AFTER the label went on. Without
		// this, a maintainer who hand-labels an issue whose newest event is a
		// months-old author comment has their triage reversed on the next sweep:
		// the role reads "author", both branches above pass, and the bot strips
		// the label.
		//
		// The comparison is against the last ROLE-BEARING action, not against
		// days_since_activity: editing an old comment advances the clock without
		// changing who spoke last, so a third party touching their own ancient
		// comment would otherwise read as the author returning.
		//
		// An unknown label age (-1) is left alone rather than guessed at —
		// removal posts no comment and a maintainer can redo it, whereas
		// refusing would strand the issue.
		if !authorCameBack && st.DaysSinceStaleLabel >= 0 && st.DaysSinceLastActorAction > st.DaysSinceStaleLabel+labelRaceGrace {
			return fmt.Sprintf("issue #%d last heard from %q %.1f days ago but was labelled stale %.1f days ago, so nobody has come back since the label; only a maintainer should clear it", number, st.LastActorName, st.DaysSinceLastActorAction, st.DaysSinceStaleLabel), false
		}
		return "", true
	}
}

// alertPredicate enforces that a maintainer alert is posted only when the bot
// itself computed that an unannounced description edit needs one.
func alertPredicate(number int) func(IssueState) (string, bool) {
	return func(st IssueState) (string, bool) {
		if st.Status != "success" {
			return fmt.Sprintf("issue #%d state was not retrieved successfully; refusing to act", number), false
		}
		if !st.MaintainerAlertNeeded {
			return fmt.Sprintf("issue #%d does not need a maintainer alert", number), false
		}
		return "", true
	}
}

func (c *GitHubClient) doAddLabel(ctx context.Context, number int, label string) (actionResult, error) {
	if msg, ok := authorizeIssue(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	if !c.isManagedLabel(label) {
		return errResult("label %q is not managed by this bot", label), nil
	}
	// The stale label must never be applied through this tool. Marking an issue
	// stale is gated on the thresholds via add_stale_label_and_comment, and this
	// tool is not, so allowing it here would be a way around that gate: an issue
	// would become is_stale with no warning comment posted and its close clock
	// already running, so a later run could close it after CloseAfter days
	// instead of StaleAfter + CloseAfter, with the author never warned. The
	// legitimate path calls AddLabel directly from MarkStale, not through here.
	if sameLabel(label, c.cfg.StaleLabel) {
		return errResult("use add_stale_label_and_comment to mark issue #%d stale; %q cannot be applied with this tool", number, label), nil
	}
	// The clarification label is STEP 3's follow-up to marking stale, so it earns
	// the same precondition rather than being writable on any in-scope issue. Its
	// own action key keeps it from contending with the mark-stale claim taken
	// moments earlier against the same observation.
	// The check is shared with marking stale, but its refusal text is not: say
	// which action was refused, or the model reads it as a report about a tool it
	// did not call.
	if msg, ok := c.claimAction(number, actionAddClarify, stalePredicate(number)); !ok {
		return errResult("cannot flag issue #%d as waiting on its author: %s", number, msg), nil
	}
	// Write the CONFIGURED name, never the model's argument. The allow-list
	// matches case-insensitively so the bot can recognize a label the repository
	// stored under different casing, and Unicode simple folding makes that
	// equivalence class wider than ASCII case: "requeſt clarification" folds
	// equal to "request clarification" but is a different label to GitHub.
	// Passing the argument through would let the model choose the bytes that
	// reach the API, which is exactly what the allow-list exists to prevent.
	if err := c.AddLabel(ctx, number, c.cfg.RequestClarificationLabel); err != nil {
		c.recordToolError()
		return actionResult{}, err
	}
	return okResult, nil
}

func (c *GitHubClient) doRemoveLabel(ctx context.Context, number int, label string) (actionResult, error) {
	if msg, ok := authorizeIssue(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	if !c.isManagedLabel(label) {
		return errResult("label %q is not managed by this bot", label), nil
	}
	// Removing the stale label is destructive: it resets days_since_stale_label
	// to zero, so an issue steered here can never reach the close branch. STEP 1
	// permits it only when the author or another user came back.
	if sameLabel(label, c.cfg.StaleLabel) {
		if msg, ok := c.claimAction(number, actionRemoveStale, removeStalePredicate(number)); !ok {
			return errResult("%s", msg), nil
		}
	} else {
		// The decision tree removes only the stale label. Refusing anything else
		// keeps the code's authority and the prompt's instructions in step: if a
		// future revision needs this, both change together.
		return errResult("this bot only removes %q; %q must be removed by a maintainer", c.cfg.StaleLabel, label), nil
	}
	// The configured name again, for the reason given in doAddLabel.
	if err := c.RemoveLabel(ctx, number, c.cfg.StaleLabel); err != nil {
		c.recordToolError()
		return actionResult{}, err
	}
	return okResult, nil
}

// staleCommentBody and closeCommentBody build the two bodies this bot publishes.
// They are named functions so a test asserts on the text production actually
// posts rather than on a copy of it, which would keep passing after this one
// changed.
func staleCommentBody(c *GitHubClient) string {
	return fmt.Sprintf(
		"This issue has been marked as stale: a maintainer commented more than %s ago "+
			"and there has been no reply since. It will be closed if there is no "+
			"further activity in the next %s.",
		humanDays(c.cfg.StaleAfter), humanDays(c.cfg.CloseAfter),
	)
}

func closeCommentBody(c *GitHubClient) string {
	return fmt.Sprintf(
		"This has been closed automatically: it was marked as stale more than %s ago "+
			"and there has been no activity since.",
		humanDays(c.cfg.CloseAfter),
	)
}

func (c *GitHubClient) doMarkStale(ctx context.Context, number int) (actionResult, error) {
	if msg, ok := authorizeIssue(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	if msg, ok := c.claimAction(number, actionMarkStale, stalePredicate(number)); !ok {
		return errResult("%s", msg), nil
	}
	// Every claim in this sentence is one the Go gate above established.
	//
	// It used to say "after a maintainer requested clarification". Nothing
	// verifies that. stalePredicate checks that the last actor was a maintainer
	// and that they left a comment, but whether that comment ASKED the author
	// for anything is the model's judgement alone — the one decision in this
	// program no gate can make. Publishing it as fact meant that whenever the
	// model was wrong, the bot told a stranger, in public and under the company's
	// name, that a maintainer had asked them something that was never asked.
	//
	// What is left is what the predicate proves: a maintainer commented, it was
	// longer ago than the threshold, and the author has not acted since.
	comment := staleCommentBody(c)
	if err := c.MarkStale(ctx, number, comment); err != nil {
		c.recordToolError()
		return actionResult{}, err
	}
	return okResult, nil
}

func (c *GitHubClient) doAlertEdit(ctx context.Context, number int) (actionResult, error) {
	if msg, ok := authorizeIssue(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	// The alert posts a comment, so it needs the same Go-side gate: the bot
	// computed whether an unannounced description edit actually happened.
	if msg, ok := c.claimAction(number, actionAlertEdit, alertPredicate(number)); !ok {
		return errResult("%s", msg), nil
	}
	// botAlertBody is the exact string the recognition in buildTimeline matches,
	// so the two cannot drift: an alert written in any other shape would not be
	// recognized on the next run and would be posted again on every sweep.
	if err := c.Comment(ctx, number, botAlertBody); err != nil {
		c.recordToolError()
		return actionResult{}, err
	}
	return okResult, nil
}

func (c *GitHubClient) doClose(ctx context.Context, number int) (actionResult, error) {
	if msg, ok := authorizeIssue(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	if msg, ok := c.claimAction(number, actionClose, closePredicate(number)); !ok {
		return errResult("%s", msg), nil
	}
	comment := closeCommentBody(c)
	if err := c.CloseAsStale(ctx, number, comment); err != nil {
		c.recordToolError()
		return actionResult{}, err
	}
	return okResult, nil
}

// tools builds the function tools the agent uses. The names match those
// referenced by the prompt's decision tree. Each handler closes over the
// GitHub client; agent.Context embeds context.Context, so it is passed
// directly to the do* methods.
func (c *GitHubClient) tools() ([]tool.Tool, error) {
	var (
		tools []tool.Tool
		errs  []error
	)
	add := func(t tool.Tool, err error) {
		if err != nil {
			errs = append(errs, err)
			return
		}
		tools = append(tools, t)
	}

	add(functiontool.New(functiontool.Config{
		Name:        "get_issue_state",
		Description: "Fetches and analyzes the full state of a GitHub issue, returning its staleness, last actor role, labels, and timing.",
	}, func(ctx agent.Context, a issueArg) (IssueState, error) {
		return c.doGetIssueState(ctx, a.IssueNumber)
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "add_label_to_issue",
		Description: "Adds the specified label to the issue.",
	}, func(ctx agent.Context, a labelArg) (actionResult, error) {
		return c.doAddLabel(ctx, a.IssueNumber, a.Label)
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "remove_label_from_issue",
		Description: "Removes the specified label from the issue.",
	}, func(ctx agent.Context, a labelArg) (actionResult, error) {
		return c.doRemoveLabel(ctx, a.IssueNumber, a.Label)
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "add_stale_label_and_comment",
		Description: "Marks the issue as stale by adding the stale label and posting an explanatory comment.",
	}, func(ctx agent.Context, a issueArg) (actionResult, error) {
		return c.doMarkStale(ctx, a.IssueNumber)
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "alert_maintainer_of_edit",
		Description: "Posts a comment alerting maintainers that the author silently edited the issue description.",
	}, func(ctx agent.Context, a issueArg) (actionResult, error) {
		return c.doAlertEdit(ctx, a.IssueNumber)
	}))

	add(functiontool.New(functiontool.Config{
		Name:        "close_as_stale",
		Description: "Closes the issue as not planned after it has remained stale past the close threshold.",
	}, func(ctx agent.Context, a issueArg) (actionResult, error) {
		return c.doClose(ctx, a.IssueNumber)
	}))

	if len(errs) > 0 {
		return nil, fmt.Errorf("create tools: %w", errors.Join(errs...))
	}
	// Cap the slice at its length before handing it out. One []tool.Tool is
	// shared by every per-issue agent, and append leaves spare capacity, so a
	// consumer appending to it would write one backing array from every
	// goroutine at once. Whether the agent appends is not this package's to
	// know: a full slice expression makes the question moot, because an append
	// then copies.
	return tools[:len(tools):len(tools)], nil
}
