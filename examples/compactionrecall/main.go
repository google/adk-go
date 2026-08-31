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

// Package main measures whether a compaction config still lets the agent
// answer.
//
// Compaction is normally judged by prompt size, which is easy to measure and
// says nothing about whether the conversation survived. This does the other
// half: it plants checkable facts, buries them under filler turns until
// compaction has summarized them away, then asks about each one. A fact counts
// as recalled only if the answer carries the value.
//
// Point it at your own settings to find out what they cost you:
//
//	GOOGLE_API_KEY=... go run ./examples/compactionrecall \
//	    -threshold=32000 -retention=10 -runs=5
//
// # Read the per-run numbers, not the average
//
// Recall here is close to all-or-nothing. Each compaction pass either copies a
// value forward or generalizes it to its label -- "the deployment region" in
// place of "europe-west4" -- and a value replaced by its label cannot come
// back, because the next pass sees only the previous summary and the retained
// tail. So a config tends to score 8/8 or 0/8 rather than something in between,
// and two configs that both average 50% can mean "always half-remembers" and
// "works perfectly half the time".
//
// That is why -runs defaults to 3 and the per-run scores are printed
// individually. A single run tells you very little; it is one draw from a coin
// whose bias is what you actually want to know.
//
// # The arms
//
//	default   the shipped summarizer prompt
//	terse     the same prompt without its fact-retention instruction
//
// The terse arm exists to show what that instruction is worth, and to stand in
// for any custom PromptTemplate that asks only for a "concise" summary. If you
// are writing your own template, run it here before trusting it.
//
// This calls a real model. One run of one arm is roughly 70 model calls, so the
// default of two arms over three runs is a few hundred. A single transient
// error aborts the run, and a busy model returns 503 often enough to matter at
// this call count; -model pins a different one if the default is saturated.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"regexp"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// fact is something the user states early and is asked about later. accept
// lists the forms of the answer that count as recall: a model that reads back
// "March 14" for "14 March" has remembered, and scoring it as a miss would
// overstate the loss.
type fact struct {
	tell   string
	ask    string
	label  string
	accept []string

	// matchers are accept compiled with word boundaries, so a short value
	// cannot be found inside a longer one. Without this "7" is recalled by any
	// answer that happens to mention 17, and the weakest probe decides the
	// score.
	matchers []*regexp.Regexp
}

func init() {
	for i := range facts {
		for _, form := range facts[i].accept {
			facts[i].matchers = append(facts[i].matchers,
				regexp.MustCompile(`(?i)\b`+regexp.QuoteMeta(form)+`\b`))
		}
	}
}

// facts are deliberately arbitrary. Nothing here can be answered from world
// knowledge or guessed from the question, so a hit means the detail genuinely
// survived into the prompt.
var facts = []fact{
	{
		tell:  "For this project the deployment region is europe-west4. Just note it.",
		ask:   "Which deployment region did I say we are using?",
		label: "europe-west4", accept: []string{"europe-west4"},
	},
	{
		tell:  "The incident we are tracking is INC-77312. Note it.",
		ask:   "What is the incident ID I mentioned?",
		label: "INC-77312", accept: []string{"INC-77312"},
	},
	{
		tell:  "We picked Postgres over MySQL, specifically because of PostGIS. Note it.",
		ask:   "Which database did we pick, and what was the specific reason?",
		label: "PostGIS", accept: []string{"PostGIS"},
	},
	{
		tell:  "The cutover deadline is 14 March. Note it.",
		ask:   "What is the cutover deadline?",
		label: "14 March", accept: []string{"14 March", "March 14", "14th March", "March 14th"},
	},
	{
		tell:  "The on-call rotation owner is Priya. Note it.",
		ask:   "Who owns the on-call rotation?",
		label: "Priya", accept: []string{"Priya"},
	},
	{
		tell:  "Our error budget for the quarter is 0.3 percent. Note it.",
		ask:   "What is our error budget for the quarter?",
		label: "0.3", accept: []string{"0.3"},
	},
	{
		tell:  "The staging bucket is gs://tarn-staging-91. Note it.",
		ask:   "What is the staging bucket?",
		label: "tarn-staging-91", accept: []string{"tarn-staging-91"},
	},
	{
		tell:  "We agreed to cap retries at 7. Note it.",
		ask:   "What did we agree to cap retries at?",
		label: "7", accept: []string{"7", "seven"},
	},
}

// noRecordSentinel is what the agent is instructed to reply when it cannot
// answer. Scoring a denial by matching English phrasing is guesswork -- the
// model can always word it a way the list does not have -- so the agent is
// asked for a token instead, and the token decides.
const noRecordSentinel = "NO_RECORD"

// disclaimers are a fallback for a model that ignores the sentinel and
// disclaims in prose anyway. They are not relied on: the risk they cover is an
// answer that both denies knowledge and happens to contain a value, which for
// the short values here is the difference between a sound score and a lucky
// one.
var disclaimers = []string{
	"do not have that", "don't have that", "not mentioned",
	"you have not mentioned", "no record of", "cannot recall",
	"do not have", "don't have", "unable to", "no information",
	"not been shared", "not been provided", "do not see", "don't see",
}

// filler pushes the prompt up so compaction fires, carrying nothing the probes
// ask about.
var filler = []string{
	"Explain in two sentences why connection pooling matters.",
	"Give me two sentences on the tradeoff between read replicas and sharding.",
	"In two sentences, when is a circuit breaker worth adding?",
	"Two sentences: what is the point of a canary deploy?",
	"Briefly, why do idempotency keys matter for retries?",
	"Two sentences on structured logging versus plain text.",
	"Why might p99 latency matter more than the mean? Two sentences.",
	"Two sentences: what does backpressure mean in a queue?",
	"Briefly explain the difference between a liveness and a readiness probe.",
	"Two sentences on why schema migrations should be backwards compatible.",
	"What is a bulkhead pattern? Two sentences.",
	"Two sentences: why cap the size of a retry queue?",
	"Briefly, what is the risk of a thundering herd?",
	"Two sentences on why clock skew breaks distributed tracing.",
	"What does exactly-once delivery really cost? Two sentences.",
	"Two sentences on when to prefer a pull over a push model.",
}

// tersePromptTemplate is the summarizer prompt without the instruction that
// keeps concrete values alive across re-summarization. It is what the default
// used to be, and it stands in for any custom template that asks only for a
// short summary.
const tersePromptTemplate = "The following is a conversation history between a user and an AI agent." +
	" It may or may not start from a compacted history. Please identify and" +
	" reiterate the user request, summarize the context so far, focusing on" +
	" key decisions made and information obtained, as well as any unresolved" +
	" questions or tasks. The summary should be concise and capture the" +
	" essence of the interaction.\n\n" + compaction.ConversationHistoryPlaceholder

type armResult struct {
	hits        int
	buried      int
	lostBuried  int
	compactions int
	missed      []string
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelName := flag.String("model", "gemini-flash-latest", "model to run the conversation and the summarizer on")
	threshold := flag.Int("threshold", 700, "compaction.Config.TokenThreshold")
	retention := flag.Int("retention", 3, "compaction.Config.EventRetentionSize")
	interval := flag.Int("interval", 0, "compaction.Config.CompactionInterval, 0 to leave the sliding window off")
	fillerx := flag.Int("fillerx", 3, "repeat the filler block this many times, to force more compaction passes")
	runs := flag.Int("runs", 3, "repeat each arm this many times; recall is near-binary, so one run proves little")
	armList := flag.String("arms", "default,terse", "comma-separated: default, terse")
	verbose := flag.Bool("v", false, "print every summary and every probe answer")
	flag.Parse()

	if os.Getenv("GOOGLE_API_KEY") == "" {
		return fmt.Errorf("GOOGLE_API_KEY is not set")
	}

	want := map[string]bool{}
	for _, n := range strings.Split(*armList, ",") {
		want[strings.TrimSpace(n)] = true
	}
	for n := range want {
		if n != "default" && n != "terse" {
			return fmt.Errorf("unknown arm %q, want default or terse", n)
		}
	}

	ctx := context.Background()
	fmt.Printf("model=%s threshold=%d retention=%d interval=%d facts=%d fillerTurns=%d runs=%d\n\n",
		*modelName, *threshold, *retention, *interval, len(facts), len(filler)**fillerx, *runs)

	for _, name := range []string{"default", "terse"} {
		if !want[name] {
			continue
		}
		var scores []string
		totalHits, totalProbes, wipeouts := 0, 0, 0

		for i := range *runs {
			cfg := &compaction.Config{
				TokenThreshold:     *threshold,
				EventRetentionSize: *retention,
				CompactionInterval: *interval,
			}
			res, err := runArm(ctx, name, cfg, *modelName, *fillerx, i, *verbose)
			if err != nil {
				return fmt.Errorf("arm %s run %d: %w", name, i+1, err)
			}
			scores = append(scores, fmt.Sprintf("%d/%d", res.hits, len(facts)))
			totalHits += res.hits
			totalProbes += len(facts)
			if res.hits == 0 {
				wipeouts++
			}
			fmt.Printf("  %-8s run %d: recalled %d/%d  buried=%d  lostWhileBuried=%d  compactions=%d\n",
				name, i+1, res.hits, len(facts), res.buried, res.lostBuried, res.compactions)
			if len(res.missed) > 0 {
				fmt.Printf("           lost: %s\n", strings.Join(res.missed, ", "))
			}
		}

		fmt.Printf("  %-8s TOTAL %d/%d probes, per-run %s, runs losing everything: %d/%d\n\n",
			name, totalHits, totalProbes, strings.Join(scores, " "), wipeouts, *runs)
	}

	fmt.Println("A run that lost everything is the failure that matters: the summary kept")
	fmt.Println("the shape of the conversation and dropped the values. Compare the count of")
	fmt.Println("those, not the averages.")
	return nil
}

func runArm(ctx context.Context, name string, cfg *compaction.Config, modelName string, fillerx, runIdx int, verbose bool) (*armResult, error) {
	m, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{APIKey: os.Getenv("GOOGLE_API_KEY")})
	if err != nil {
		return nil, fmt.Errorf("model: %w", err)
	}

	if name == "terse" {
		sum, err := compaction.NewLLMSummarizer(compaction.LLMSummarizerConfig{
			Model:          m,
			PromptTemplate: tersePromptTemplate,
		})
		if err != nil {
			return nil, fmt.Errorf("summarizer: %w", err)
		}
		cfg.Summarizer = sum
	}

	root, err := llmagent.New(llmagent.Config{
		Name:  "assistant",
		Model: m,
		Instruction: "You are a helpful assistant in a long running conversation. " +
			"When the user states a fact and asks you to note it, acknowledge it briefly. " +
			"When the user asks about something stated earlier, answer from the conversation. " +
			"If you genuinely do not have the information, reply with exactly " +
			noRecordSentinel + " and nothing else. Never guess.",
	})
	if err != nil {
		return nil, fmt.Errorf("agent: %w", err)
	}

	const appName = "compaction-recall"
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           appName,
		Agent:             root,
		SessionService:    svc,
		AutoCreateSession: true,
		Compaction:        cfg,
	})
	if err != nil {
		return nil, fmt.Errorf("runner: %w", err)
	}

	userID := "u"
	sessionID := fmt.Sprintf("s-%s-%d", name, runIdx)

	say := func(text string) (string, error) {
		var sb strings.Builder
		msg := genai.NewContentFromText(text, genai.RoleUser)
		for ev, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
			if err != nil {
				return "", err
			}
			if ev == nil || ev.LLMResponse.Content == nil {
				continue
			}
			for _, p := range ev.LLMResponse.Content.Parts {
				if p != nil && !p.Thought {
					sb.WriteString(p.Text)
				}
			}
		}
		return sb.String(), nil
	}

	for i, f := range facts {
		if _, err := say(f.tell); err != nil {
			return nil, fmt.Errorf("planting fact %d: %w", i, err)
		}
	}
	for pass := range fillerx {
		for i, q := range filler {
			if _, err := say(q); err != nil {
				return nil, fmt.Errorf("filler %d/%d: %w", pass, i, err)
			}
		}
	}

	res := &armResult{}
	buried, err := buriedFacts(ctx, svc, appName, userID, sessionID, verbose, name)
	if err != nil {
		return nil, err
	}

	for _, f := range facts {
		ans, err := say(f.ask)
		if err != nil {
			return nil, fmt.Errorf("probing %q: %w", f.label, err)
		}
		if verbose {
			fmt.Printf("  [%s] Q=%q\n         A=%q\n", name, f.ask, truncate(ans, 200))
		}
		if buried[f.label] {
			res.buried++
		}
		if recalled(ans, f) {
			res.hits++
			continue
		}
		res.missed = append(res.missed, f.label)
		if buried[f.label] {
			res.lostBuried++
		}
	}

	got, err := svc.Get(ctx, &session.GetRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err != nil {
		return nil, fmt.Errorf("reading session: %w", err)
	}
	for ev := range got.Session.Events().All() {
		if ev.Actions.Compaction != nil {
			res.compactions++
		}
	}
	return res, nil
}

// recalled reports whether the answer carries the planted value.
//
// A denial is a miss even if a value appears somewhere in it, so the checks are
// ordered: the sentinel the agent was told to use, then prose denials, then the
// value itself. Matching a denial by phrasing alone would be unreliable, which
// is why the agent is asked for the sentinel in the first place.
func recalled(answer string, f fact) bool {
	if strings.Contains(strings.ToUpper(answer), noRecordSentinel) {
		return false
	}
	a := strings.ToLower(answer)
	for _, d := range disclaimers {
		if strings.Contains(a, d) {
			return false
		}
	}
	for _, m := range f.matchers {
		if m.MatchString(answer) {
			return true
		}
	}
	return false
}

// buriedFacts reports which planted facts sit behind a compaction boundary at
// probe time. A fact still present as a raw event proves nothing about the
// summary, so only the buried ones say anything about what compaction cost.
func buriedFacts(ctx context.Context, svc session.Service, appName, userID, sessionID string, verbose bool, arm string) (map[string]bool, error) {
	got, err := svc.Get(ctx, &session.GetRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err != nil {
		return nil, fmt.Errorf("reading session: %w", err)
	}

	type span struct{ start, end int64 }
	var covered []span
	n := 0
	for ev := range got.Session.Events().All() {
		c := ev.Actions.Compaction
		if c == nil || c.CompactedContent == nil {
			continue
		}
		covered = append(covered, span{c.StartTimestamp.UnixNano(), c.EndTimestamp.UnixNano()})
		if verbose {
			n++
			var text string
			for _, p := range c.CompactedContent.Parts {
				if p != nil {
					text += p.Text
				}
			}
			fmt.Printf("  [%s] summary %d (%d chars):\n%s\n\n", arm, n, len(text), text)
		}
	}

	buried := map[string]bool{}
	for ev := range got.Session.Events().All() {
		if ev.Actions.Compaction != nil || ev.LLMResponse.Content == nil {
			continue
		}
		var text string
		for _, p := range ev.LLMResponse.Content.Parts {
			if p != nil {
				text += p.Text
			}
		}
		at := ev.Timestamp.UnixNano()
		for _, f := range facts {
			if !strings.Contains(text, f.label) {
				continue
			}
			for _, s := range covered {
				if at >= s.start && at <= s.end {
					buried[f.label] = true
				}
			}
		}
	}
	return buried, nil
}

func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
