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
	"fmt"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// auditedPRKey scopes a session to a single pull request number. The runner
// builds the invocation context from the context passed to Run (which embeds
// it), so a value set here is visible to every tool via ctx.Value.
type auditedPRKey struct{}

// withAuditedPR binds the pull request this session is allowed to mutate.
func withAuditedPR(ctx context.Context, number int) context.Context {
	return context.WithValue(ctx, auditedPRKey{}, number)
}

// authorizePR reports whether the tool may act on the requested pull request.
// It is the defense against prompt injection: untrusted title, description or
// filename text cannot make the agent act on a pull request other than the one
// this session is triaging.
//
// This relies on ADK propagating the context passed to runner.Run (which carries
// the withAuditedPR value) through to the agent.Context seen here. If a future
// ADK release stops embedding context in its tool context, the lookup below
// simply misses and every call is rejected — the bot stops acting (fail-safe)
// rather than acting on the wrong pull request.
func authorizePR(ctx context.Context, requested int) (string, bool) {
	audited, ok := ctx.Value(auditedPRKey{}).(int)
	if !ok {
		return "no pull request is authorized for this session", false
	}
	if requested != audited {
		return fmt.Sprintf("session is scoped to pull request #%d; refusing to act on #%d", audited, requested), false
	}
	return "", true
}

// assignArgs is the input for the assign_owner_to_pull_request tool.
// functiontool.New reflects over this struct to build the JSON schema the model
// fills, so the json tags name the fields the model produces.
//
// Note what is absent: there is no assignee field. The model names a component
// and Go resolves the login, so no login the operator did not configure can
// ever be assigned, and no login appears in the prompt for an injected
// instruction to aim at.
type assignArgs struct {
	PullRequestNumber int    `json:"pull_request_number"`
	Component         string `json:"component"`
}

// contextArgs is the input for the request_more_context tool. missing_items are
// keys from a fixed allow-list; Go owns every word that reaches the pull
// request, so no model-authored prose is ever posted.
type contextArgs struct {
	PullRequestNumber int      `json:"pull_request_number"`
	MissingItems      []string `json:"missing_items"`
}

// actionResult is the typed result returned by both tools.
type actionResult struct {
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
}

// errResult is a model-readable failure: the tool ran as a Go call but the
// request was rejected (wrong pull request, unknown component, claim already
// spent). It is returned with a nil Go error so the model receives it as data.
// Real I/O failures return a Go error instead.
func errResult(format string, a ...any) actionResult {
	return actionResult{Status: "error", Message: fmt.Sprintf(format, a...)}
}

// claimRejection maps a non-granted claim onto a model-readable result. It never
// invites a retry: every one of these states is final for this run.
func claimRejection(res claimResult, number int, act action) actionResult {
	switch res {
	case claimNotEligible:
		return errResult("pull request #%d was not cleared for triage; refusing to %s", number, act)
	case claimAlreadyFailed:
		return errResult("the %s action on pull request #%d already failed this run and will not be retried", act, number)
	default:
		return actionResult{
			Status:  "skipped",
			Message: fmt.Sprintf("the %s action has already been taken on pull request #%d; nothing more to do", act, number),
		}
	}
}

// assignOwner is the body of the assign_owner_to_pull_request tool, factored out
// so it can be unit-tested without going through the agent.
//
// The order matters. The claim is taken BEFORE the assignability check, so an
// owner who turns out not to be assignable ends the pull request's single
// assignment attempt. Checking first would let the model walk the component map
// until it found an assignable login — exactly the retry loop the design
// forbids, and a way for injected text to steer who gets assigned.
func (c *GitHubClient) assignOwner(ctx context.Context, number int, component string) (actionResult, error) {
	if msg, ok := authorizePR(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	key := strings.ToLower(strings.TrimSpace(component))
	owner, known := c.cfg.OwnerMap[key]
	if !known {
		// Reject before the claim: an unknown component is the model getting the
		// vocabulary wrong, not an attempt at an action, so it may name a real
		// component instead. It cannot reach a login this way — only a key of the
		// configured map resolves.
		return errResult("unknown component %q; choose one of: %s",
			component, strings.Join(c.cfg.components(), ", ")), nil
	}
	if res := c.claim(number, actionAssign); res != claimGranted {
		return claimRejection(res, number, actionAssign), nil
	}

	// Re-read the preconditions from GitHub immediately before writing. The claim
	// is per process, and a manual batch run and an event-driven run for the same
	// pull request are in different workflow concurrency groups: both can fetch
	// before either writes, and AddAssignees APPENDS, so without this the pull
	// request ends up with two owners. This does not make the write atomic — it
	// narrows the window from a whole model turn to one round trip.
	if stale, err := c.assignmentStillWanted(ctx, number); err != nil {
		c.recordFailure(number, actionAssign)
		c.recordError()
		return actionResult{}, err
	} else if stale != "" {
		c.log.Info("assignment no longer wanted", "pr", number, "reason", stale)
		return actionResult{
			Status:  "skipped",
			Message: fmt.Sprintf("pull request #%d no longer needs an owner: %s", number, stale),
		}, nil
	}

	assignable, err := c.IsAssignable(ctx, key, owner)
	if err != nil {
		c.recordFailure(number, actionAssign)
		c.recordError()
		return actionResult{}, err
	}
	if !assignable {
		// GitHub silently drops an assignee without write or triage access, so
		// report the skip rather than claiming a success that did not happen. The
		// claim stays spent: this is a configuration problem for a human to fix,
		// not something to retry against another component.
		c.log.Warn("configured owner is not assignable", "pr", number, "component", key, "owner", owner)
		return actionResult{
			Status: "skipped",
			Message: fmt.Sprintf("the owner configured for component %q cannot be assigned in this repository "+
				"(GitHub requires write or triage access); leaving pull request #%d unassigned", key, number),
		}, nil
	}

	if err := c.AssignOwner(ctx, number, key, owner); err != nil {
		c.recordFailure(number, actionAssign)
		c.recordError()
		return actionResult{}, err
	}
	return actionResult{
		Status:  "success",
		Message: fmt.Sprintf("assigned the owner of component %q to pull request #%d", key, number),
	}, nil
}

// maxContextItems bounds the array the model may pass. buildContextComment
// dedups anyway, so anything longer than the allow-list is malformed input.
var maxContextItems = len(contextItems)

// requestMoreContext is the body of the request_more_context tool. It posts one
// comment, built entirely from constants, asking the author for the named
// missing pieces.
func (c *GitHubClient) requestMoreContext(ctx context.Context, number int, items []string) (actionResult, error) {
	if msg, ok := authorizePR(ctx, number); !ok {
		return errResult("%s", msg), nil
	}
	if len(items) == 0 {
		return errResult("missing_items is empty; name at least one of: %s",
			strings.Join(contextItemKeys(), ", ")), nil
	}
	if len(items) > maxContextItems {
		return errResult("missing_items has %d entries; at most %d are allowed", len(items), maxContextItems), nil
	}
	// Validate every key before claiming, so a typo does not burn the pull
	// request's single comment. The allow-list is the value bound: nothing the
	// model writes reaches GitHub, only which of these fixed sentences to use.
	for _, item := range items {
		if _, ok := contextItemText(strings.ToLower(strings.TrimSpace(item))); !ok {
			return errResult("unknown missing_items entry %q; choose from: %s",
				item, strings.Join(contextItemKeys(), ", ")), nil
		}
	}
	if res := c.claim(number, actionComment); res != claimGranted {
		return claimRejection(res, number, actionComment), nil
	}

	body := buildContextComment(items)
	if body == "" {
		// Unreachable given the validation above; if it ever is reached, an empty
		// comment must not be posted.
		c.recordFailure(number, actionComment)
		return errResult("no renderable request for pull request #%d", number), nil
	}
	if err := c.PostComment(ctx, number, body); err != nil {
		c.recordFailure(number, actionComment)
		c.recordError()
		return actionResult{}, err
	}
	return actionResult{
		Status:  "success",
		Message: fmt.Sprintf("asked the author of pull request #%d for the missing context", number),
	}, nil
}

// tools builds the agent's toolset: assignment always, and the context request
// only when it is enabled. Everything else — which pull requests to look at,
// whether one is eligible at all, which login a component resolves to, and every
// word of the comment — is decided in code.
//
// A test pins this inventory in both configurations, so a new tool cannot be
// added without that test failing.
func (c *GitHubClient) tools() ([]tool.Tool, error) {
	assign, err := functiontool.New(functiontool.Config{
		Name: "assign_owner_to_pull_request",
		Description: "Assigns the owner of a component to the pull request, as its shepherd. " +
			"Call this exactly once, with the component the change belongs to. The component " +
			"must be one of the components listed in your instructions; you do not choose the " +
			"person, only the component.",
	}, func(ctx agent.Context, a assignArgs) (actionResult, error) {
		return c.assignOwner(ctx, a.PullRequestNumber, a.Component)
	})
	if err != nil {
		return nil, fmt.Errorf("create assign tool: %w", err)
	}
	tools := []tool.Tool{assign}

	if !c.cfg.contextRequestsEnabled() {
		return tools, nil
	}
	request, err := functiontool.New(functiontool.Config{
		Name: "request_more_context",
		Description: "Posts a single comment asking the pull request author for missing context. " +
			"missing_items names which pieces are missing, chosen from: " +
			strings.Join(contextItemKeys(), ", ") + ". The wording of the comment is fixed; " +
			"you only choose which pieces to ask for. Call this at most once, and only when the " +
			"description genuinely lacks what a reviewer would need.",
	}, func(ctx agent.Context, a contextArgs) (actionResult, error) {
		return c.requestMoreContext(ctx, a.PullRequestNumber, a.MissingItems)
	})
	if err != nil {
		return nil, fmt.Errorf("create context-request tool: %w", err)
	}
	return append(tools, request), nil
}
