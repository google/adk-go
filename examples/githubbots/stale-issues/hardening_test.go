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

// Pins for the hardening added after the pre-release security audit. Each test
// here corresponds to a finding, and each was checked against a mutant that
// reverts the fix.

import (
	"context"
	"encoding/json"
	"io"
	"iter"
	"net/http"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// --- Finding 2: alert recognition must be bound to this bot's identity -------

// botAlertSignature is a fixed literal in a public repository, so it is not a
// secret and anyone can type it. Identity is the only half of the pair an
// outsider cannot forge — and "any [bot] account" is not identity.
//
// The concrete route: every ADK bot in this repository posts as
// github-actions[bot], and any third party can install a GitHub App whose login
// ends in [bot]. Under the old suffix test, one comment opening with the
// signature permanently silenced this bot's silent-edit alert on that issue,
// because the branch is only re-entered while last_action_type is still the
// description edit.
func TestOnlyThisBotsOwnIdentitySuppressesItsAlert(t *testing.T) {
	cfg := e2eCfg()
	alert := botAlertSignature + ". Maintainers, please review."

	// A bot account that is not THIS bot must not suppress the alert. Its
	// comment is dropped from history by isIgnoredActor, so the description edit
	// stays the newest event and the alert is still owed — unless the comment is
	// wrongly recognized as this bot's own, which is the finding.
	//
	// Any third party can install a GitHub App whose login ends in "[bot]", and
	// every ADK bot in this repository already posts as github-actions[bot].
	for _, impostor := range []struct{ name, login string }{
		{"another GitHub App", "friendly-scanner[bot]"},
		{"a sibling workflow's app", "dependabot[bot]"},
	} {
		t.Run(impostor.name, func(t *testing.T) {
			// The impostor's comment must POSTDATE the description edit. The
			// suppression compares the last alert against the edit, so an
			// impostor comment older than the edit could not suppress anything
			// even if wrongly recognized, and this test would pass proving
			// nothing. Measured: with the ordering reversed, the mutant that
			// accepts any "[bot]" login survived.
			raw := newIssue(60).comment(e2eMaint, 20, "Could you share a repro?").build()
			raw.UserContentEdits.Nodes = append(raw.UserContentEdits.Nodes, rawEdit{
				Editor: &rawActor{Login: e2eAuthor}, EditedAt: ago(10),
			})
			raw.Comments.Nodes = append(raw.Comments.Nodes, rawComment{
				Author:    &rawActor{Login: impostor.login, Typename: "Bot"},
				Body:      alert,
				CreatedAt: ago(1),
			})

			got := computeIssueState(raw, e2eSelf, cfg.Maintainers, cfg.StaleLabel, cfg.StaleAfter, cfg.CloseAfter, timeNowUTC())
			if !got.MaintainerAlertNeeded {
				t.Errorf("%s posted the signature and silenced a genuine silent-edit alert: recognition is not bound to this bot's identity, so anyone who can install a GitHub App can switch this alert off on any issue", impostor.name)
			}
		})
	}

	// An ordinary user is a different case and must not be conflated with the
	// one above. Their comment is real history, so posting the signature does
	// not suppress anything — it supersedes the edit as the newest event, which
	// is correct. What must hold for them is that the comment stays VISIBLE:
	// if the signature let a user's comment be swallowed as the bot's own, they
	// could act on an issue without the bot ever seeing that they had.
	t.Run("an ordinary user's comment is not swallowed", func(t *testing.T) {
		raw := newIssue(60).
			comment(e2eMaint, 20, "Could you share a repro?").
			comment(e2eOther, 1, alert).
			build()
		got := computeIssueState(raw, e2eSelf, cfg.Maintainers, cfg.StaleLabel, cfg.StaleAfter, cfg.CloseAfter, timeNowUTC())
		if !strings.EqualFold(got.LastActorName, e2eOther) {
			t.Errorf("last_actor_name = %q, want %q: a user who opens their comment with the bot's signature made it invisible to the history", got.LastActorName, e2eOther)
		}
		if got.LastActionRole != string(roleOther) {
			t.Errorf("last_action_role = %q, want %q", got.LastActionRole, roleOther)
		}
	})

	// And the bot's own alert still suppresses, or the cases above would pass
	// simply because suppression is broken for everyone.
	t.Run("the bot itself", func(t *testing.T) {
		raw := newIssue(60).
			comment(e2eMaint, 20, "Could you share a repro?").
			descEdit(e2eAuthor, 10).
			botComment(e2eSelf, 1, alert).
			build()
		got := computeIssueState(raw, e2eSelf, cfg.Maintainers, cfg.StaleLabel, cfg.StaleAfter, cfg.CloseAfter, timeNowUTC())
		if got.MaintainerAlertNeeded {
			t.Error("the bot's own alert no longer suppresses a repeat, so it will re-post the same notice on every run")
		}
	})
}

// Identity is SHARED: every ADK bot in this repository posts as
// github-actions[bot], so a sibling workflow's comment satisfies the identity
// check exactly as this bot's own does. The body match is the only thing left
// between a sibling and this bot's alert-suppression state.
//
// A prefix match is not enough for that job. A sibling that echoes user-supplied
// text at the start of a comment carries whatever a commenter put there, so a
// commenter who writes this signature gets it posted under the shared identity
// and silences a genuine alert. Matching the whole body requires the sibling's
// entire comment to be this alert verbatim, which an echo of user text is not.
//
// This narrows the route; it does not close it. See botAlertBody.
func TestASiblingBotCommentMerelyStartingWithTheAlertDoesNotSuppressIt(t *testing.T) {
	cfg := e2eCfg()
	raw := newIssue(60).comment(e2eMaint, 20, "Could you share a repro?").build()
	raw.UserContentEdits.Nodes = append(raw.UserContentEdits.Nodes, rawEdit{
		Editor: &rawActor{Login: e2eAuthor}, EditedAt: ago(10),
	})
	// Posted under the shared bot identity, opening with the alert but carrying
	// echoed user text after it — the shape a sibling workflow would produce.
	raw.Comments.Nodes = append(raw.Comments.Nodes, rawComment{
		Author:    &rawActor{Login: e2eSelf},
		Body:      botAlertBody + "\n\n> user-supplied text echoed by another workflow",
		CreatedAt: ago(1),
	})

	got := computeIssueState(raw, e2eSelf, cfg.Maintainers, cfg.StaleLabel, cfg.StaleAfter, cfg.CloseAfter, timeNowUTC())
	if !got.MaintainerAlertNeeded {
		t.Error("a comment that merely STARTS WITH the alert suppressed a genuine one: under the shared github-actions[bot] identity that is reachable by anyone whose text a sibling workflow echoes")
	}
}

// GraphQL returns a bot's login bare, without the REST "[bot]" suffix, so the
// identity match has to survive that normalization or the bot stops recognizing
// its own alerts the moment the suffix test is removed.
func TestTheBotRecognizesItsOwnAlertThroughGraphQLsBareLogin(t *testing.T) {
	cfg := e2eCfg()
	raw := newIssue(60).
		comment(e2eMaint, 20, "Could you share a repro?").
		descEdit(e2eAuthor, 10).
		build()
	// __typename "Bot" with the bare login, exactly as GraphQL sends it.
	raw.Comments.Nodes = append(raw.Comments.Nodes, rawComment{
		Author:    &rawActor{Login: "github-actions", Typename: "Bot"},
		Body:      botAlertSignature + ". Maintainers, please review.",
		CreatedAt: ago(1),
	})
	got := computeIssueState(raw, "github-actions[bot]", cfg.Maintainers, cfg.StaleLabel, cfg.StaleAfter, cfg.CloseAfter, timeNowUTC())
	if got.MaintainerAlertNeeded {
		t.Error("the bot did not recognize its own alert when GraphQL sent the bare login: actorLogin's [bot] normalization is what makes the identity match work, and without it the bot re-alerts every run")
	}
}

// --- Finding 1a: the sweep must be bounded -----------------------------------

// The search paginated to exhaustion, so the size of a run was a property of
// the backlog rather than of the configuration. Every issue in that list is a
// candidate for a public comment and, a week later, a close.
func TestSearchStopsAtTheIssueCap(t *testing.T) {
	var pages int
	cfg := baseCfg()
	cfg.MaxIssues = 7

	// Always report another page, so only the cap can end the loop.
	c := testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pages++
		if pages > 50 {
			t.Fatal("the search never stopped: MAX_ISSUES is not bounding the sweep")
		}
		w.Header().Set("Link", `<https://api.github.com/search/issues?page=99>; rel="next"`)
		var b strings.Builder
		b.WriteString(`{"total_count":1000,"incomplete_results":false,"items":[`)
		for i := range 5 {
			if i > 0 {
				b.WriteString(",")
			}
			b.WriteString(`{"number":`)
			b.WriteString(itoa(pages*100 + i))
			b.WriteString(`}`)
		}
		b.WriteString(`]}`)
		_, _ = io.WriteString(w, b.String())
	}))

	got, err := c.SearchOldOpenIssues(context.Background())
	if err != nil {
		t.Fatalf("SearchOldOpenIssues: %v", err)
	}
	if len(got) != cfg.MaxIssues {
		t.Errorf("returned %d candidates, want exactly %d: an unbounded sweep puts the whole backlog one model misjudgement away from a mass stale-and-close", len(got), cfg.MaxIssues)
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// --- Finding 1b: a per-run ceiling on harm -----------------------------------

// The ceiling bounds what one run can damage if the judgement behind its writes
// goes wrong at scale. Closing is the only irreversible action this bot has, and
// whether a maintainer comment is really a request is decided by the prompt, so
// the ceiling is the mechanical backstop under a judgement that is not.
func TestDestructiveActionsAreCappedPerRun(t *testing.T) {
	c := newTestClient(t)
	c.cfg.MaxDestructiveActions = 3

	var allowed int
	for i := 1; i <= 10; i++ {
		c.recordObservation(i, okState())
		if _, ok := c.claimAction(i, actionMarkStale, stalePredicate(i)); ok {
			allowed++
		}
	}
	if allowed != 3 {
		t.Errorf("%d issues were marked stale against a ceiling of 3: the cap is not enforced, so one bad run reaches the whole candidate list", allowed)
	}
	if !c.hitDestructiveCeiling() {
		t.Error("hitDestructiveCeiling() = false after the cap was reached, so the run would exit 0 and nobody would look at it")
	}
}

// Corrective actions must survive the ceiling. A run that has stopped marking
// issues stale must still be able to take a label back off and to alert a
// maintainer, or hitting the cap would also disable the repairs.
func TestTheCeilingDoesNotBlockCorrectiveActions(t *testing.T) {
	c := newTestClient(t)
	c.cfg.MaxDestructiveActions = 1

	c.recordObservation(1, okState())
	if _, ok := c.claimAction(1, actionMarkStale, stalePredicate(1)); !ok {
		t.Fatal("the first mark-stale was refused, so this test cannot reach the ceiling")
	}

	// Ceiling reached. A removal on a different issue must still go through.
	st := okState()
	st.IsStale = true
	st.LastActionRole = string(roleAuthor)
	st.DaysSinceStaleLabel = 10
	st.DaysSinceLastActorAction = 2
	c.recordObservation(2, st)
	if _, ok := c.claimAction(2, actionRemoveStale, func(IssueState) (string, bool) { return "", true }); !ok {
		t.Error("removing a stale label was refused because the destructive ceiling was reached: the cap must bound harm, not repairs")
	}
}

// --- Finding 3: a panic must not take the sweep down -------------------------

// MarkStale writes the label and then the comment, and compensates by removing
// the label if the comment fails. None of that runs if the process dies, so a
// panic mid-sweep can leave an issue labelled stale with no warning posted —
// which the next run refuses to re-mark and closes on schedule, having warned
// nobody.
func TestAPanicInOneAuditFailsTheRunInsteadOfKillingIt(t *testing.T) {
	cfg := baseCfg()
	cfg.Concurrency = 2
	cfg.IssueTimeout = time.Minute
	log := discardLogger()

	var completed int32
	err := auditAll(context.Background(), cfg, log, []int{1, 2, 3}, func(_ context.Context, n int) error {
		if n == 2 {
			panic("malformed model response")
		}
		completed++
		return nil
	})
	if err == nil {
		t.Fatal("auditAll returned nil after a goroutine panicked: the panic was swallowed, and a run that died mid-write would report success")
	}
	if !strings.Contains(err.Error(), "panic") || !strings.Contains(err.Error(), "#2") {
		t.Errorf("auditAll error = %v, want it to name the panic and the issue it happened on", err)
	}
	if completed != 2 {
		t.Errorf("%d of the 2 healthy issues finished, want both: one bad issue must not abort the others", completed)
	}
}

// nilPartLLM yields a response whose Parts slice contains a nil element, which
// the genai type permits and a malformed response can produce.
type nilPartLLM struct{}

func (nilPartLLM) Name() string { return "nil-part" }

func (nilPartLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Role:  genai.RoleModel,
				Parts: []*genai.Part{nil, {Text: "Analysis for Issue #7: ACTIVE. No action."}},
			},
		}, nil)
	}
}

// A malformed model response must fail the run rather than kill the process.
//
// Measured, and it corrects the premise this was added under. The audit read
// main.go's unguarded p.Text as "one malformed model response kills the run
// mid-write". It does not: ADK recovers a panic inside an agent node itself and
// surfaces it as a stream error, so this input never reaches that loop. With the
// recover in auditAll deliberately removed, this case still produced an error
// rather than a crash.
//
// The nil guard on p.Text therefore stays as cheap defence but is UNMEASURED —
// no input was found that reaches it, because ADK panics upstream first. It is
// not classified as pinned, and this test does not pretend to pin it.
//
// What IS load-bearing is the recover, for panics raised in this program's own
// code rather than inside an ADK node: see the test above, where removing the
// recover crashes the process outright.
func TestAMalformedModelResponseFailsTheRunRatherThanCrashing(t *testing.T) {
	cfg := baseCfg()
	cfg.Concurrency = 1
	cfg.IssueTimeout = time.Minute
	log := discardLogger()

	tools, err := (&GitHubClient{cfg: cfg, log: log}).tools()
	if err != nil {
		t.Fatalf("tools: %v", err)
	}
	err = auditAll(context.Background(), cfg, log, []int{7}, perIssueRunFn(cfg, nilPartLLM{}, tools, log))
	if err == nil {
		t.Fatal("a nil Content.Part produced no error: a malformed response must fail the run loudly, not be treated as a completed audit")
	}
	if !strings.Contains(err.Error(), "#7") {
		t.Errorf("error = %v, want it to name the issue the malformed response arrived on", err)
	}
}

// --- The megaphone property, asserted structurally ---------------------------

// The adversarial e2e scenarios assert that attacker text does not appear in
// what the bot writes. That assertion passes identically whether the model
// refused the injection or the Go layer made it impossible, so on its own it
// cannot show which one is load-bearing — and the model's compliance rate with
// output injection has been measured elsewhere in this fleet at 2 in 5.
//
// This is the structural half: every byte the bot can publish is templated in
// Go from configuration, and no tool exposes a parameter through which the model
// could supply prose. That is why the megaphone class is closed here regardless
// of what the model decides to do.
func TestNoToolAcceptsFreeTextForAPublishedBody(t *testing.T) {
	c := newTestClient(t)
	tools, err := c.tools()
	if err != nil {
		t.Fatalf("tools: %v", err)
	}

	// The only string the model may supply anywhere is a label name, and that is
	// checked against the allow-list and then discarded in favour of the
	// configured spelling before anything is written.
	allowedStringParams := map[string]bool{"label": true}

	// ADK publishes the tool schema in ParametersJsonSchema, not in Parameters,
	// which is nil for every one of these tools. An earlier version of this test
	// read Parameters, iterated nothing and passed unconditionally — so it
	// asserted the megaphone property while being incapable of failing. The
	// counter below makes that failure mode loud: if no property is ever seen,
	// the test fails instead of passing silently.
	var propertiesSeen int

	for _, tl := range tools {
		declarer, ok := tl.(interface {
			Declaration() *genai.FunctionDeclaration
		})
		if !ok {
			t.Errorf("tool %q exposes no declaration, so its parameter surface cannot be checked", tl.Name())
			continue
		}
		decl := declarer.Declaration()
		if decl == nil {
			t.Errorf("tool %q has a nil declaration", tl.Name())
			continue
		}
		raw, err := json.Marshal(decl.ParametersJsonSchema)
		if err != nil {
			t.Errorf("tool %q: marshal schema: %v", decl.Name, err)
			continue
		}
		var schema struct {
			Properties map[string]struct {
				Type string `json:"type"`
			} `json:"properties"`
		}
		if err := json.Unmarshal(raw, &schema); err != nil {
			t.Errorf("tool %q: parse schema %s: %v", decl.Name, raw, err)
			continue
		}
		for name, p := range schema.Properties {
			propertiesSeen++
			if p.Type != "string" {
				continue
			}
			if !allowedStringParams[strings.ToLower(name)] {
				t.Errorf("tool %q exposes a free-text string parameter %q. Every body this bot publishes is templated in Go from configuration, and that is the whole reason attacker text cannot reach a public comment under our identity. A string parameter is a channel through which the model — and therefore anyone who can steer it — could supply prose that gets written out.", decl.Name, name)
			}
		}
	}

	if propertiesSeen == 0 {
		t.Fatal("no tool parameters were inspected, so this test proved nothing: ADK has moved the schema again and the check needs re-pointing, not deleting")
	}
}
