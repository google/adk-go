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
	"slices"
	"sync"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// auditedGroupKey scopes a session to one file group of one release. The runner
// builds the invocation context from the context passed to Run (which embeds
// it), so a value set here is visible to every tool via ctx.Value.
type auditedGroupKey struct{}

// groupScope is the authority a session holds: it may record findings for
// exactly this release and exactly this group index, and nothing else.
type groupScope struct {
	release string
	index   int
}

// withAuditedGroup binds the release and file group this session may record for.
//
// This is the release-docs analogue of a per-issue scope. The bot has no
// per-issue session to bind -- it files one issue at the end of the run -- so
// the binding is per (release, file group) instead, which is the unit a model
// call actually sees.
func withAuditedGroup(ctx context.Context, release string, index int) context.Context {
	return context.WithValue(ctx, auditedGroupKey{}, groupScope{release: release, index: index})
}

// authorizeGroup reports whether the tool may record for the requested release
// and group. It is the defense against prompt injection: contributor-authored
// diff text cannot make the agent attribute its output to another group, and an
// unscoped session can record nothing at all.
//
// This relies on ADK propagating the context passed to runner.Run through to the
// agent.Context seen here. If a future ADK release stops embedding context in
// its tool context, the lookup below simply misses and every call is rejected --
// the bot records nothing (fail-safe) rather than recording under the wrong key.
func authorizeGroup(ctx context.Context, release string, index int) (string, bool) {
	scope, ok := ctx.Value(auditedGroupKey{}).(groupScope)
	if !ok {
		return "no file group is authorized for this session", false
	}
	if scope.release != release {
		return fmt.Sprintf("session is scoped to release %s; refusing to record for %s", scope.release, release), false
	}
	if scope.index != index {
		return fmt.Sprintf("session is scoped to group %d; refusing to record for group %d", scope.index, index), false
	}
	return "", true
}

// recordArgs is the input for the record_documentation_findings tool.
// functiontool.New reflects over this struct to build the JSON schema the model
// fills, so the json tags name the fields the model produces.
type recordArgs struct {
	Release    string    `json:"release"`
	GroupIndex int       `json:"group_index"`
	Findings   []Finding `json:"findings"`
}

// actionResult is the typed result returned by the tool.
type actionResult struct {
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
}

// errResult is a model-readable failure: the tool ran as a Go call but the
// request was rejected (e.g. the wrong group). It is returned with a nil Go
// error so the model receives it as data.
func errResult(format string, a ...any) actionResult {
	return actionResult{Status: "error", Message: fmt.Sprintf(format, a...)}
}

// recorder collects the model's findings for one run. It holds no GitHub client
// and can perform no mutation: the model's entire authority is to add sanitized
// entries to this map, and code decides afterwards whether anything is filed.
type recorder struct {
	cfg *Config

	mu sync.Mutex
	// byGroup is keyed by group index so a duplicate call for a group can be
	// rejected rather than appending the same findings twice.
	byGroup map[int][]Finding
	// discarded counts findings sanitization emptied completely, so a model
	// whose output is entirely unrenderable is reported rather than silently
	// producing an issue that says it found nothing.
	discarded int
}

func newRecorder(cfg *Config) *recorder {
	return &recorder{cfg: cfg, byGroup: make(map[int][]Finding)}
}

// record stores the sanitized findings for one group and reports whether this
// call won the group's single slot.
//
// The claim and the write happen in one critical section: a check-then-act would
// let two calls for one group both observe "not yet recorded" and both append.
func (r *recorder) record(index int, findings []Finding) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, done := r.byGroup[index]; done {
		return false
	}
	r.byGroup[index] = findings
	return true
}

// addDiscarded records findings that sanitization emptied completely.
func (r *recorder) addDiscarded(n int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.discarded += n
}

// discardedCount returns how many recorded findings sanitization emptied.
func (r *recorder) discardedCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.discarded
}

// unreported reports how many of the first n group indexes recorded nothing at
// all. Recording an empty list still occupies the slot, so this counts only the
// groups whose tool never fired -- the shape a model steered into silence, or a
// group that died before it could report, both produce.
func (r *recorder) unreported(n int) int {
	return r.unreportedExcept(n, nil)
}

// unreportedExcept counts the first n group indexes that recorded nothing,
// ignoring the ones named in skip.
//
// The skip set is the groups that failed: a group can record its findings and
// then hit an error, so "failed" and "recorded nothing" are independent facts
// and neither can be derived from the other by arithmetic.
func (r *recorder) unreportedExcept(n int, skip []int) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	skipped := make(map[int]bool, len(skip))
	for _, i := range skip {
		skipped[i] = true
	}
	missing := 0
	for i := range n {
		if skipped[i] {
			continue
		}
		if _, done := r.byGroup[i]; !done {
			missing++
		}
	}
	return missing
}

// findings returns every recorded finding, ordered by group index so the issue
// body is deterministic regardless of the order the groups completed in.
func (r *recorder) findings() []Finding {
	r.mu.Lock()
	defer r.mu.Unlock()
	indexes := make([]int, 0, len(r.byGroup))
	for i := range r.byGroup {
		indexes = append(indexes, i)
	}
	slices.Sort(indexes)
	var out []Finding
	for _, i := range indexes {
		out = append(out, r.byGroup[i]...)
	}
	return out
}

// recordFindings is the body of the record_documentation_findings tool, factored
// out so it can be unit-tested without going through the agent.
//
// Every rejection is returned as a model-readable result with a nil Go error:
// there is no I/O here to fail, because the tool touches nothing but this run's
// in-memory collector.
func (r *recorder) recordFindings(ctx context.Context, a recordArgs) actionResult {
	if msg, ok := authorizeGroup(ctx, a.Release, a.GroupIndex); !ok {
		return errResult("%s", msg)
	}
	// Bound the volume before anything is stored: a steered model must not be
	// able to make the issue body arbitrarily large.
	raw := a.Findings
	dropped := 0
	if len(raw) > r.cfg.MaxFindingsPerGroup {
		dropped = len(raw) - r.cfg.MaxFindingsPerGroup
		raw = raw[:r.cfg.MaxFindingsPerGroup]
	}
	clean := make([]Finding, 0, len(raw))
	emptied := 0
	for _, f := range raw {
		if s := sanitizeFinding(f); !s.empty() {
			clean = append(clean, s)
		} else {
			emptied++
		}
	}
	if !r.record(a.GroupIndex, clean) {
		return errResult("group %d has already recorded its findings", a.GroupIndex)
	}
	r.addDiscarded(emptied)
	msg := fmt.Sprintf("recorded %d findings for group %d", len(clean), a.GroupIndex)
	if dropped > 0 {
		msg += fmt.Sprintf(" (%d beyond the per-group cap were dropped)", dropped)
	}
	if emptied > 0 {
		msg += fmt.Sprintf(" (%d were discarded: nothing in them survived sanitization)", emptied)
	}
	return actionResult{Status: "success", Message: msg}
}

// tools builds the agent's toolset.
//
// There is exactly one tool and it mutates nothing: it appends sanitized
// findings to this run's in-memory collector. Creating the issue is done by Go
// after the analysis loop, with the target repository, the title and the marker
// all computed from configuration and API metadata -- so a fully steered model
// cannot file an issue, cannot choose where one is filed, and cannot decide that
// one is filed at all.
func (r *recorder) tools() ([]tool.Tool, error) {
	t, err := functiontool.New(functiontool.Config{
		Name: "record_documentation_findings",
		Description: "Records the documentation updates suggested for the file group you were given. " +
			"Call this exactly once, with the release and group_index from the message. " +
			"Pass an empty findings list if the group needs no documentation change.",
	}, func(ctx agent.Context, a recordArgs) (actionResult, error) {
		return r.recordFindings(ctx, a), nil
	})
	if err != nil {
		return nil, fmt.Errorf("create tools: %w", err)
	}
	return []tool.Tool{t}, nil
}
