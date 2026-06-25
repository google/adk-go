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

// Package llmagent_test contains black-box tests for agent collaboration
// (chat / task / single_turn delegation modes).
//
// Each test drives a real Gemini model whose HTTP traffic is captured
// and replayed via the httprr harness (see internal/testutil). In CI
// and on machines without an API key, the .httprr files under
// testdata/ are replayed deterministically. To re-record after
// changing an Instruction, tool declaration, or agent shape, delete
// the corresponding .httprr file and run:
//
//	GEMINI_API_KEY=... go test ./agent/llmagent/ -httprecord=TestDelegation_NN
//
// To re-record everything: -httprecord=TestDelegation.
//
// The tests generate *.events.yaml file to visualize delegation and overall
// flow of execution.
//
// Assertions are structural (FC/FR names, scope ids, presence of
// synthesised FRs) rather than exact-text — recorded model output may
// shift in wording across re-recordings, but the framework's
// orchestration behaviour around it must not.

package llmagent_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"google.golang.org/genai"
	"gopkg.in/yaml.v3"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/httprr"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/workflow"
)

const delegationModelName = "gemini-3.5-flash"

//go:generate go test -httprecord=TestDelegation

// newDelegationModel returns a Gemini model whose HTTP transport is
// the httprr record/replay wrapper, scoped to a trace file derived
// from the current test name.
//
// IMPORTANT: a single test must reuse the SAME model instance across
// all of its agents. Each httprr.Open call opens the trace file with
// os.Create, which truncates the existing file; concurrent file
// descriptors writing at the same position then overwrite each
// other's records. delegationModel memoises per (test, file) so
// repeated calls in one test return the same instance. Existing
// llmagent_test.go tests are immune because they create one model
// per test; our multi-agent tests must too.
func newDelegationModel(t *testing.T) model.LLM {
	t.Helper()
	return sharedDelegationModel(t)
}

// sharedDelegationModel returns the single delegationModel instance
// for the current test, creating it on first use.
func sharedDelegationModel(t *testing.T) model.LLM {
	t.Helper()
	if m, ok := delegationModels.Load(t.Name()); ok {
		return m.(model.LLM)
	}
	trace := filepath.Join("testdata", strings.ReplaceAll(t.Name()+".httprr", "/", "_"))
	cfg := testutil.NewGeminiTestClientConfig(t, trace)
	m, err := gemini.NewModel(t.Context(), delegationModelName, cfg)
	if err != nil {
		t.Fatalf("gemini.NewModel: %v", err)
	}
	delegationModels.Store(t.Name(), m)
	t.Cleanup(func() { delegationModels.Delete(t.Name()) })
	return m
}

// delegationModels caches one model instance per test name.
// Keyed by t.Name(); cleared in t.Cleanup so a -count=N run doesn't
// reuse a stale instance across iterations.
var delegationModels sync.Map

// delegationRunner is a thin wrapper around runner.Runner +
// session.InMemoryService that:
//   - exposes Events() to read the FULL persisted session (including
//     user messages, which never flow through the agent event stream
//     but DO end up in the session),
//   - registers a t.Cleanup that dumps the final session events to
//     testdata/<TestName>.events.yaml in YAML form.
//
// Use it instead of testutil.TestAgentRunner whenever a test wants
// the YAML side effect; runner-internals access is via the public
// runner.New + session.Service APIs only.
type delegationRunner struct {
	t      *testing.T
	r      *runner.Runner
	svc    session.Service
	app    string
	user   string
	sessID string
	// sess is the most-recent session snapshot, cached so the cleanup
	// dumper doesn't need to re-Get from the service.
	sess session.Session
	// skipDump, when true, suppresses the testdata/*.events.yaml
	// emission in the t.Cleanup hook. Set via SkipDump().
	skipDump bool
}

const (
	delegationApp     = "delegation_app"
	delegationUser    = "delegation_user"
	delegationSession = "delegation_session"
)

// newDelegationRunner builds a runner around a fresh in-memory
// session service, creates the session up front, and registers a
// t.Cleanup that dumps the FULL session (incl. user messages) to
// testdata/<TestName>.events.yaml at test end.
//
// Tests whose execution is non-deterministic (e.g. parallel
// single_turn dispatch reorders events across runs) should call
// dr.SkipDump() to prevent the dump from dirtying the checked-in
// .events.yaml on every replay.
func newDelegationRunner(t *testing.T, root agent.Agent) *delegationRunner {
	t.Helper()
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:        delegationApp,
		Agent:          root,
		SessionService: svc,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	if _, err := svc.Create(context.Background(), &session.CreateRequest{
		AppName:   delegationApp,
		UserID:    delegationUser,
		SessionID: delegationSession,
	}); err != nil {
		t.Fatalf("session.Create: %v", err)
	}
	dr := &delegationRunner{
		t: t, r: r, svc: svc,
		app: delegationApp, user: delegationUser, sessID: delegationSession,
	}
	// Only dump testdata/*.events.yaml when re-recording the httprr trace.
	trace := filepath.Join("testdata", strings.ReplaceAll(t.Name()+".httprr", "/", "_"))
	recording, err := httprr.Recording(trace)
	if err != nil {
		t.Fatalf("httprr.Recording: %v", err)
	}
	dr.skipDump = !recording
	t.Cleanup(func() {
		// Pull the freshest session snapshot, then dump.
		if dr.sess != nil && !dr.skipDump {
			dumpSession(t, dr.sess)
		}
	})
	return dr
}

// SkipDump disables the testdata/<TestName>.events.yaml dump at
// t.Cleanup. Use this for tests with non-deterministic event
// ordering (e.g. parallel single_turn dispatch) where the dump
// would otherwise dirty git after every replay.
func (dr *delegationRunner) SkipDump() {
	dr.skipDump = true
}

// turn drives one user turn (a plain text message) and returns the
// agent events streamed back. The session.Service is updated as a
// side effect, including the user message and the dumped events.
func (dr *delegationRunner) turn(userMsg string) []*session.Event {
	dr.t.Helper()
	return dr.turnContent(genai.NewContentFromText(userMsg, genai.RoleUser))
}

// turnContent drives one turn from a pre-built Content (e.g. a
// confirmation FunctionResponse for HITL resume).
func (dr *delegationRunner) turnContent(msg *genai.Content) []*session.Event {
	dr.t.Helper()
	stream := dr.r.Run(context.Background(), dr.user, dr.sessID, msg, agent.RunConfig{})
	events, err := testutil.CollectEvents(stream)
	if err != nil {
		dr.t.Fatalf("collect events: %v", err)
	}
	// Refresh the session snapshot so Cleanup dumps the latest state.
	if got, err := dr.svc.Get(context.Background(), &session.GetRequest{
		AppName: dr.app, UserID: dr.user, SessionID: dr.sessID,
	}); err == nil {
		dr.sess = got.Session
	} else {
		dr.t.Fatalf("session.Get: %v", err)
	}
	return events
}

// dumpSession reads every event persisted in the given session
// (including user messages, which the runner appends but never emits
// on the agent stream) and writes a YAML representation to
// testdata/<TestName>.events.yaml.
//
// The dump captures the framework-observable fields per event:
// author, isolation_scope, branch, role, partial, long_running_tool_ids,
// node_info (path, message_as_output), output (when set), actions
// (transfer_to_agent, requested_tool_confirmations, skip_summarization),
// and content parts decomposed into text / function_call / function_response.
// Timestamps and storage-assigned IDs are omitted (they vary across runs).
func dumpSession(t *testing.T, sess session.Session) {
	t.Helper()
	events := sess.Events()
	dump := make([]eventDump, 0, events.Len())
	for i := 0; i < events.Len(); i++ {
		dump = append(dump, eventToDump(events.At(i)))
	}
	out, err := yaml.Marshal(dump)
	if err != nil {
		t.Fatalf("yaml.Marshal events: %v", err)
	}
	path := filepath.Join("testdata", strings.ReplaceAll(t.Name()+".events.yaml", "/", "_"))
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// eventDump is the YAML-marshalled per-event record. Fields are
// pointers / omitempty so the produced YAML stays compact.
type eventDump struct {
	Author             string      `yaml:"author,omitempty"`
	IsolationScope     string      `yaml:"isolation_scope,omitempty"`
	Branch             string      `yaml:"branch,omitempty"`
	Role               string      `yaml:"role,omitempty"`
	Partial            bool        `yaml:"partial,omitempty"`
	LongRunningToolIDs []string    `yaml:"long_running_tool_ids,omitempty"`
	Parts              []partDump  `yaml:"parts,omitempty"`
	Output             any         `yaml:"output,omitempty"`
	NodeInfo           *nodeDump   `yaml:"node_info,omitempty"`
	Actions            *actionDump `yaml:"actions,omitempty"`
}

// partDump renders a single content Part: at most one of text /
// function_call / function_response is set.
type partDump struct {
	Text             string  `yaml:"text,omitempty"`
	Thought          bool    `yaml:"thought,omitempty"`
	FunctionCall     *fcDump `yaml:"function_call,omitempty"`
	FunctionResponse *frDump `yaml:"function_response,omitempty"`
	Other            string  `yaml:"other,omitempty"`
}

type fcDump struct {
	Name string         `yaml:"name"`
	ID   string         `yaml:"id,omitempty"`
	Args map[string]any `yaml:"args,omitempty"`
}

type frDump struct {
	Name     string         `yaml:"name"`
	ID       string         `yaml:"id,omitempty"`
	Response map[string]any `yaml:"response,omitempty"`
}

type nodeDump struct {
	Path            string   `yaml:"path,omitempty"`
	MessageAsOutput bool     `yaml:"message_as_output,omitempty"`
	OutputFor       []string `yaml:"output_for,omitempty"`
}

type actionDump struct {
	TransferToAgent            string         `yaml:"transfer_to_agent,omitempty"`
	SkipSummarization          bool           `yaml:"skip_summarization,omitempty"`
	Escalate                   bool           `yaml:"escalate,omitempty"`
	StateDelta                 map[string]any `yaml:"state_delta,omitempty"`
	RequestedToolConfirmations map[string]any `yaml:"requested_tool_confirmations,omitempty"`
}

// eventToDump projects a session.Event onto an eventDump, dropping
// storage-assigned and timing fields that would make the YAML noisy
// or non-deterministic across re-recordings.
//
// Framework-generated UUIDs (prefixed "adk-") are replaced with a
// stable per-id placeholder before emission so the dump stays
// deterministic across runs. These IDs originate from utils.New /
// utils.GenerateFunctionCallID and are minted each time
// generateRequestConfirmationEvent (or similar) fires; without
// redaction they would change on every replay and dirty the
// checked-in .events.yaml.
func eventToDump(ev *session.Event) eventDump {
	r := newIDRedactor()
	d := eventDump{
		Author:             ev.Author,
		IsolationScope:     ev.IsolationScope,
		Branch:             ev.Branch,
		Partial:            ev.LLMResponse.Partial,
		LongRunningToolIDs: redactIDs(r, ev.LongRunningToolIDs),
	}
	if ev.LLMResponse.Content != nil {
		d.Role = ev.LLMResponse.Content.Role
		for _, p := range ev.LLMResponse.Content.Parts {
			if p == nil {
				continue
			}
			pd := partDump{}
			switch {
			case p.FunctionCall != nil:
				pd.FunctionCall = &fcDump{
					Name: p.FunctionCall.Name,
					ID:   r.redact(p.FunctionCall.ID),
					Args: redactInValue(r, p.FunctionCall.Args).(map[string]any),
				}
			case p.FunctionResponse != nil:
				pd.FunctionResponse = &frDump{
					Name:     p.FunctionResponse.Name,
					ID:       r.redact(p.FunctionResponse.ID),
					Response: redactInValue(r, p.FunctionResponse.Response).(map[string]any),
				}
			case p.Text != "":
				pd.Text = p.Text
				pd.Thought = p.Thought
			default:
				pd.Other = "<binary or unsupported part>"
			}
			d.Parts = append(d.Parts, pd)
		}
	}
	if ev.Output != nil {
		d.Output = ev.Output
	}
	if ev.NodeInfo != nil {
		d.NodeInfo = &nodeDump{
			Path:            ev.NodeInfo.Path,
			MessageAsOutput: ev.NodeInfo.MessageAsOutput,
			OutputFor:       ev.NodeInfo.OutputFor,
		}
	}
	a := ev.Actions
	if a.TransferToAgent != "" || a.SkipSummarization || a.Escalate ||
		len(a.StateDelta) > 0 || len(a.RequestedToolConfirmations) > 0 {
		ad := &actionDump{
			TransferToAgent:   a.TransferToAgent,
			SkipSummarization: a.SkipSummarization,
			Escalate:          a.Escalate,
			StateDelta:        a.StateDelta,
		}
		// RequestedToolConfirmations is map[string]ToolConfirmation;
		// dump the hint/confirmed/payload fields per entry. The
		// map key is a FunctionCall ID — redact framework-generated
		// UUIDs the same way we redact ids elsewhere.
		if len(a.RequestedToolConfirmations) > 0 {
			ad.RequestedToolConfirmations = make(map[string]any, len(a.RequestedToolConfirmations))
			for k, v := range a.RequestedToolConfirmations {
				ad.RequestedToolConfirmations[r.redact(k)] = map[string]any{
					"hint":      v.Hint,
					"confirmed": v.Confirmed,
					"payload":   v.Payload,
				}
			}
		}
		d.Actions = ad
	}
	return d
}

// idRedactor maps non-deterministic framework-generated IDs (those
// starting with the "adk-" prefix produced by utils.New /
// GenerateFunctionCallID) to stable placeholders like "adk-id-1",
// "adk-id-2", etc. Each call to redact() returns the same
// placeholder for the same input id; ids that don't match the
// "adk-" prefix pass through unchanged (these come from the LLM
// response and ARE stable across httprr replays).
type idRedactor struct {
	mapping map[string]string
}

func newIDRedactor() *idRedactor {
	return &idRedactor{mapping: map[string]string{}}
}

func (r *idRedactor) redact(id string) string {
	if !strings.HasPrefix(id, "adk-") {
		return id
	}
	if placeholder, ok := r.mapping[id]; ok {
		return placeholder
	}
	placeholder := fmt.Sprintf("adk-id-%d", len(r.mapping)+1)
	r.mapping[id] = placeholder
	return placeholder
}

// redactIDs returns a copy of ids with every "adk-"-prefixed entry
// replaced through r.redact.
func redactIDs(r *idRedactor, ids []string) []string {
	if len(ids) == 0 {
		return ids
	}
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = r.redact(id)
	}
	return out
}

// redactInValue walks a JSON-like value (maps, slices, strings) and
// substitutes any "adk-"-prefixed id strings via r.redact. Used to
// scrub FC.Args / FR.Response, which may carry IDs nested inside
// (e.g. adk_request_confirmation.args.originalFunctionCall.id).
// Non-string leaves and non-id strings pass through unchanged.
func redactInValue(r *idRedactor, v any) any {
	switch x := v.(type) {
	case nil:
		return nil
	case string:
		return r.redact(x)
	case []any:
		for i, it := range x {
			x[i] = redactInValue(r, it)
		}
		return x
	case map[string]any:
		for k, it := range x {
			x[k] = redactInValue(r, it)
		}
		return x
	default:
		return v
	}
}

// collectFCsByName scans every event's content parts and returns
// every FunctionCall whose Name matches.
func collectFCsByName(events []*session.Event, name string) []*genai.FunctionCall {
	var out []*genai.FunctionCall
	for _, ev := range events {
		if ev == nil || ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p != nil && p.FunctionCall != nil && p.FunctionCall.Name == name {
				out = append(out, p.FunctionCall)
			}
		}
	}
	return out
}

// collectFRsByName scans every event's content parts and returns
// every FunctionResponse whose Name matches.
func collectFRsByName(events []*session.Event, name string) []*genai.FunctionResponse {
	var out []*genai.FunctionResponse
	for _, ev := range events {
		if ev == nil || ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p != nil && p.FunctionResponse != nil && p.FunctionResponse.Name == name {
				out = append(out, p.FunctionResponse)
			}
		}
	}
	return out
}

// eventsByAuthor returns every event whose Author matches.
func eventsByAuthor(events []*session.Event, name string) []*session.Event {
	var out []*session.Event
	for _, ev := range events {
		if ev != nil && ev.Author == name {
			out = append(out, ev)
		}
	}
	return out
}

// =============================================================================
// Scenario 1: chat coordinator → 1 task sub-agent → 1 function tool.
// =============================================================================

// TestDelegation_01_ChatToTaskToTool covers the canonical task
// delegation roundtrip: the coordinator (chat) asks data_fetcher
// (task, with an output_schema) to produce a list of items, then
// hands the result into a plain function tool. Validates that:
//   - the coordinator calls the TaskAgentTool for data_fetcher,
//   - data_fetcher finishes via finish_task with the structured
//     output (validated against its schema),
//   - the synthesised FR for the delegation carries an `output` key
//     with the structured payload, and
//   - the coordinator then invokes the plain function tool
//     `store_report` exactly once.
func TestDelegation_01_ChatToTaskToTool(t *testing.T) {
	// Output schema for data_fetcher: {items: []string}.
	itemsSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"items": {
				Type:        genai.TypeArray,
				Description: "List of fetched items.",
				Items:       &genai.Schema{Type: genai.TypeString},
			},
		},
		Required: []string{"items"},
	}

	dataFetcher, err := llmagent.New(llmagent.Config{
		Name:         "data_fetcher",
		Description:  "Fetches a list of items requested by the coordinator.",
		Model:        newDelegationModel(t),
		Mode:         llmagent.ModeTask,
		OutputSchema: itemsSchema,
		Instruction: `You collect a structured list of items for the coordinator.
You have no tools. When you receive a request, immediately call ` + "`finish_task`" + `
with the items field populated from the request. Do not ask follow-up
questions; the request always describes the items you should return.`,
	})
	if err != nil {
		t.Fatalf("data_fetcher: %v", err)
	}

	// store_report is a plain function tool the coordinator must call
	// AFTER receiving the data_fetcher result.
	type StoreReportArgs struct {
		Items []string `json:"items"`
	}
	type StoreReportResult struct {
		Stored int `json:"stored"`
	}
	var storeCalls int
	var storedItems []string
	storeReport, err := functiontool.New(functiontool.Config{
		Name:        "store_report",
		Description: "Stores the final list of items.",
	}, func(_ context.Context, _ agent.Context, in StoreReportArgs) (StoreReportResult, error) {
		storeCalls++
		storedItems = in.Items
		return StoreReportResult{Stored: len(in.Items)}, nil
	})
	if err != nil {
		t.Fatalf("store_report: %v", err)
	}

	coordinator, err := llmagent.New(llmagent.Config{
		Name:        "coordinator",
		Description: "Routes requests to data_fetcher, then stores the result.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeChat,
		SubAgents:   []agent.Agent{dataFetcher},
		Tools:       []tool.Tool{storeReport},
		Instruction: `You are a coordinator with one sub-agent and one tool.

WORKFLOW:
  1. Always call the data_fetcher sub-agent first with a request
     describing the items the user mentioned.
  2. Once data_fetcher returns its structured ` + "`output`" + ` (a list of items),
     call the store_report tool exactly once with those items.
  3. After store_report returns, reply with a single concise sentence
     confirming the number stored. Do not call any tool again.

Never ask the user for clarification; the user's first message always
contains everything you need.`,
	})
	if err != nil {
		t.Fatalf("coordinator: %v", err)
	}

	dr := newDelegationRunner(t, coordinator)

	// --- Turn 1: fetch and store ---
	events1 := dr.turn("Fetch and store the following items: apple, banana, cherry.")

	// (1) Coordinator emits exactly one TaskAgentTool FC for data_fetcher.
	fetcherFCs := collectFCsByName(events1, "data_fetcher")
	if len(fetcherFCs) != 1 {
		t.Errorf("turn 1: data_fetcher FCs = %d, want 1; events=%v",
			len(fetcherFCs), eventSummaries(events1))
	}

	// (2) data_fetcher itself emits exactly one finish_task FC with
	//     a successful FR back to it.
	finishFCs := collectFCsByName(events1, "finish_task")
	if len(finishFCs) != 1 {
		t.Errorf("turn 1: finish_task FCs = %d, want 1; events=%v",
			len(finishFCs), eventSummaries(events1))
	}
	finishFRs := collectFRsByName(events1, "finish_task")
	if len(finishFRs) != 1 {
		t.Errorf("turn 1: finish_task FRs = %d, want 1", len(finishFRs))
	} else if got, _ := finishFRs[0].Response["result"].(string); got != "Task completed." {
		t.Errorf("turn 1: finish_task FR result = %q, want %q", got, "Task completed.")
	}

	// (3) The synthesised FR back to the coordinator carries the
	//     structured items. For object-schema task agents the
	//     synthesised FR passes the map through directly (NOT
	//     wrapped under `output`); only primitive/array outputs get
	//     wrapped via the schema's WrapperKey.
	fetcherFRs := collectFRsByName(events1, "data_fetcher")
	if len(fetcherFRs) != 1 {
		t.Fatalf("turn 1: data_fetcher FRs = %d, want 1", len(fetcherFRs))
	}
	if _, ok := fetcherFRs[0].Response["items"]; !ok {
		t.Errorf("turn 1: data_fetcher FR missing `items`: %v", fetcherFRs[0].Response)
	}

	// (4) store_report is invoked exactly once with the items.
	if storeCalls != 1 {
		t.Errorf("turn 1: store_report invocations = %d, want 1", storeCalls)
	}
	if len(storedItems) == 0 {
		t.Errorf("turn 1: store_report received empty items; got %v", storedItems)
	}

	// (5) data_fetcher's events are scope-isolated from the coordinator's.
	taskEvents := eventsByAuthor(events1, "data_fetcher")
	if len(taskEvents) == 0 {
		t.Fatalf("turn 1: no events authored by data_fetcher")
	}
	for _, ev := range taskEvents {
		if ev.IsolationScope == "" {
			t.Errorf("turn 1: data_fetcher event has empty IsolationScope; "+
				"task sub-agent events must be scoped to the dispatching FC's id; ev=%v",
				eventSummary(ev))
		}
	}

	// --- Turn 2: ask a follow-up that requires memory of turn 1 ---
	//
	// Verifies:
	//   - the runner re-picks the coordinator (root chat agent) as
	//     active on turn 2 (the most-recent non-user author from
	//     turn 1 was the coordinator's final text),
	//   - the coordinator's content view on turn 2 includes the
	//     synthesised data_fetcher FR (unscoped, visible to root)
	//     so it can recall the items WITHOUT re-dispatching,
	//   - no new data_fetcher delegation is issued (the model uses
	//     conversation history instead of re-querying the sub-agent),
	//   - no new store_report tool call is issued.
	events2 := dr.turn("What items did I just ask you to store?")
	if got := len(collectFCsByName(events2, "data_fetcher")); got != 0 {
		t.Errorf("turn 2: data_fetcher FCs = %d, want 0 (coordinator should "+
			"answer from history, not re-dispatch)", got)
	}
	if storeCalls != 1 {
		t.Errorf("turn 2: store_report total invocations = %d, want 1 "+
			"(no new tool calls on turn 2)", storeCalls)
	}
	// Final text on turn 2 must come from the coordinator (proves the
	// runner picked the right active agent across turns).
	if got := finalModelTextAuthor(events2); got != "coordinator" {
		t.Errorf("turn 2: final model text author = %q, want %q",
			got, "coordinator")
	}
	// The reply should mention the stored items (basic memory check).
	if reply := finalModelText(events2); !mentionsAll(reply, "apple", "banana", "cherry") {
		t.Errorf("turn 2: coordinator's reply %q should mention all stored items "+
			"(apple, banana, cherry)", reply)
	}
}

// =============================================================================
// Scenario 2: chat coordinator → two task sub-agents called sequentially
// in one user turn.
// =============================================================================

// TestDelegation_02_ChatToTwoTaskSequential covers runChat's outer
// re-entry loop: after dispatching the first task FC and synthesising
// its FR, the coordinator's LLM is re-invoked and emits the second
// task FC in the SAME user turn. Validates that each sub-agent runs
// in its own isolation scope and that the coordinator sees both
// synthesised FRs before producing the final text.
func TestDelegation_02_ChatToTwoTaskSequential(t *testing.T) {
	orderSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"item":     {Type: genai.TypeString},
			"quantity": {Type: genai.TypeInteger},
		},
		Required: []string{"item", "quantity"},
	}
	paymentSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"credit_card_number": {Type: genai.TypeString},
			"cvv":                {Type: genai.TypeString},
		},
		Required: []string{"credit_card_number", "cvv"},
	}

	orderCollector, err := llmagent.New(llmagent.Config{
		Name:         "order_collector",
		Description:  "Collects the order item and quantity from the request.",
		Model:        newDelegationModel(t),
		Mode:         llmagent.ModeTask,
		OutputSchema: orderSchema,
		Instruction: `Extract the order from the request and immediately call
` + "`finish_task`" + ` with {item, quantity}. The request always contains
both. Do not ask the user any questions.`,
	})
	if err != nil {
		t.Fatalf("order_collector: %v", err)
	}
	paymentCollector, err := llmagent.New(llmagent.Config{
		Name:         "payment_collector",
		Description:  "Collects credit card details from the request.",
		Model:        newDelegationModel(t),
		Mode:         llmagent.ModeTask,
		OutputSchema: paymentSchema,
		Instruction: `Extract credit_card_number and cvv from the request and
immediately call ` + "`finish_task`" + ` with both fields. The request always
contains both. Do not ask the user any questions.`,
	})
	if err != nil {
		t.Fatalf("payment_collector: %v", err)
	}
	coordinator, err := llmagent.New(llmagent.Config{
		Name:        "coordinator",
		Description: "Calls order_collector, then payment_collector, then summarises.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeChat,
		SubAgents:   []agent.Agent{orderCollector, paymentCollector},
		Instruction: `You orchestrate the order pipeline in TWO sequential steps:

  1. First call order_collector with a request describing the user's
     order (item + quantity). Wait for its structured ` + "`output`" + `.
  2. Then call payment_collector with a request describing the user's
     payment details. Wait for its structured ` + "`output`" + `.
  3. Reply with one concise sentence acknowledging both pieces.

Never ask the user follow-up questions; the user's first message
always contains everything needed for both steps.`,
	})
	if err != nil {
		t.Fatalf("coordinator: %v", err)
	}

	dr := newDelegationRunner(t, coordinator)

	// --- Turn 1: order + payment in one user message ---
	events1 := dr.turn(
		"Order 2 pizzas. Pay with card 4111111111111111, cvv 123.")

	// Both delegation FCs present.
	if got := len(collectFCsByName(events1, "order_collector")); got != 1 {
		t.Errorf("turn 1: order_collector FCs = %d, want 1", got)
	}
	if got := len(collectFCsByName(events1, "payment_collector")); got != 1 {
		t.Errorf("turn 1: payment_collector FCs = %d, want 1", got)
	}

	// Both synthesised FRs present. Object-schema task agents pass
	// the structured map straight through as the FR Response.
	orderFRs := collectFRsByName(events1, "order_collector")
	if len(orderFRs) != 1 {
		t.Fatalf("turn 1: order_collector FRs = %d, want 1", len(orderFRs))
	}
	if _, ok := orderFRs[0].Response["item"]; !ok {
		t.Errorf("turn 1: order_collector FR missing `item`: %v", orderFRs[0].Response)
	}
	payFRs := collectFRsByName(events1, "payment_collector")
	if len(payFRs) != 1 {
		t.Fatalf("turn 1: payment_collector FRs = %d, want 1", len(payFRs))
	}
	if _, ok := payFRs[0].Response["credit_card_number"]; !ok {
		t.Errorf("turn 1: payment_collector FR missing `credit_card_number`: %v",
			payFRs[0].Response)
	}

	// Each sub-agent's events are stamped with a distinct
	// non-empty IsolationScope.
	orderScopes := uniqueScopes(eventsByAuthor(events1, "order_collector"))
	if len(orderScopes) != 1 || orderScopes[0] == "" {
		t.Errorf("turn 1: order_collector scopes = %v, want exactly 1 non-empty", orderScopes)
	}
	payScopes := uniqueScopes(eventsByAuthor(events1, "payment_collector"))
	if len(payScopes) != 1 || payScopes[0] == "" {
		t.Errorf("turn 1: payment_collector scopes = %v, want exactly 1 non-empty", payScopes)
	}
	if len(orderScopes) > 0 && len(payScopes) > 0 && orderScopes[0] == payScopes[0] {
		t.Errorf("turn 1: order_collector and payment_collector share scope %q; "+
			"each task delegation must run in its own isolation scope",
			orderScopes[0])
	}

	// --- Turn 2: follow-up requiring memory of BOTH delegations ---
	//
	// Verifies the coordinator's history view on turn 2 contains
	// both synthesised FRs (order + payment) and no new delegations
	// are issued — the model answers from history.
	events2 := dr.turn("What did I order, and how am I paying?")
	if got := len(collectFCsByName(events2, "order_collector")); got != 0 {
		t.Errorf("turn 2: order_collector FCs = %d, want 0 (no re-dispatch)", got)
	}
	if got := len(collectFCsByName(events2, "payment_collector")); got != 0 {
		t.Errorf("turn 2: payment_collector FCs = %d, want 0 (no re-dispatch)", got)
	}
	if got := finalModelTextAuthor(events2); got != "coordinator" {
		t.Errorf("turn 2: final model text author = %q, want %q", got, "coordinator")
	}
	// Reply should mention both pizza (order) and a recognisable
	// fragment of the card number (payment) — proves both FRs are
	// in the coordinator's view.
	if reply := finalModelText(events2); !mentionsAll(reply, "pizza") {
		t.Errorf("turn 2: coordinator's reply %q should mention pizza", reply)
	}
}

// =============================================================================
// Scenario 3: chat coordinator → single_turn sub-agent (fire-and-return).
// =============================================================================

// TestDelegation_03_ChatToSingleTurn covers the SingleTurnTool path:
// the coordinator emits an FC for the sub-agent, the standard tool-
// execution pipeline runs it through an AgentNode in a sub-branch,
// the sub-agent runs exactly ONE LLM round with IncludeContents="none"
// (so it doesn't see the coordinator's conversation), and its
// structured reply is returned to the coordinator as a regular FR.
//
// Unlike task mode, single_turn does NOT install a finish_task tool;
// the sub-agent's terminal model reply IS the output.
func TestDelegation_03_ChatToSingleTurn(t *testing.T) {
	translationSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"translation": {Type: genai.TypeString},
		},
		Required: []string{"translation"},
	}

	translator, err := llmagent.New(llmagent.Config{
		Name:         "translator",
		Description:  "Translates a single phrase to a target language.",
		Model:        newDelegationModel(t),
		Mode:         llmagent.ModeSingleTurn,
		OutputSchema: translationSchema,
		Instruction: `You translate one phrase. Read the request, then output ONLY
a JSON object of the form {"translation":"..."}. Output nothing else.`,
	})
	if err != nil {
		t.Fatalf("translator: %v", err)
	}
	coordinator, err := llmagent.New(llmagent.Config{
		Name:        "coordinator",
		Description: "Calls translator and reports the result.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeChat,
		SubAgents:   []agent.Agent{translator},
		Instruction: `For any user request asking for a translation, call the
translator sub-agent EXACTLY ONCE with a request describing the
phrase and target language. After the sub-agent returns, reply with
one concise English sentence reporting the translation. Do NOT call
the sub-agent more than once. Do NOT ask the user follow-up questions.`,
	})
	if err != nil {
		t.Fatalf("coordinator: %v", err)
	}

	dr := newDelegationRunner(t, coordinator)

	// --- Turn 1: translate 'hello' to Spanish ---
	events1 := dr.turn("Translate 'hello' to Spanish.")

	// (1) Single_turn does NOT use the finish_task machinery.
	if got := len(collectFCsByName(events1, "finish_task")); got != 0 {
		t.Errorf("turn 1: finish_task FCs = %d, want 0 (single_turn has no finish_task)", got)
	}

	// (2) Exactly one translator FC and exactly one translator FR.
	if got := len(collectFCsByName(events1, "translator")); got != 1 {
		t.Errorf("turn 1: translator FCs = %d, want 1", got)
	}
	transFRs := collectFRsByName(events1, "translator")
	if len(transFRs) != 1 {
		t.Fatalf("turn 1: translator FRs = %d, want 1", len(transFRs))
	}
	// (3) The FR carries a structured result.
	if len(transFRs[0].Response) == 0 {
		t.Errorf("turn 1: translator FR has empty Response: %v", transFRs[0])
	}

	// (4) Sub-agent's events appear in a sub-branch (NOT in the
	//     coordinator's branch).
	transEvents := eventsByAuthor(events1, "translator")
	if len(transEvents) == 0 {
		t.Fatalf("turn 1: no events authored by translator")
	}
	coordEvents := eventsByAuthor(events1, "coordinator")
	var coordBranch string
	if len(coordEvents) > 0 {
		coordBranch = coordEvents[0].Branch
	}
	var sawSubBranch bool
	for _, ev := range transEvents {
		if ev.Branch != "" && ev.Branch != coordBranch {
			sawSubBranch = true
			break
		}
	}
	if !sawSubBranch {
		t.Errorf("turn 1: expected translator events in a sub-branch distinct from "+
			"coordinator branch %q; got %v",
			coordBranch, branchesOf(transEvents))
	}

	// --- Turn 2: ask the SAME word in a different language ---
	//
	// Verifies single_turn sub-agents are re-invocable across user
	// turns: the coordinator re-dispatches `translator` for a new
	// language. The previous translation should NOT leak into the
	// new sub-branch (single_turn agents see IncludeContents="none"),
	// but the coordinator's own history still includes both
	// translator FRs across turns.
	events2 := dr.turn("Now translate the same word to French.")
	if got := len(collectFCsByName(events2, "translator")); got != 1 {
		t.Errorf("turn 2: translator FCs = %d, want 1 (one new dispatch)", got)
	}
	if got := len(collectFRsByName(events2, "translator")); got != 1 {
		t.Errorf("turn 2: translator FRs = %d, want 1", got)
	}
	if got := finalModelTextAuthor(events2); got != "coordinator" {
		t.Errorf("turn 2: final model text author = %q, want %q", got, "coordinator")
	}
}

// =============================================================================
// Scenario 4: chat coordinator → 2 single_turn sub-agents (parallel).
// =============================================================================

// TestDelegation_04_ChatToTwoSingleTurnParallel exercises the docs'
// claim that single_turn agents "can be run in parallel": when the
// coordinator's LLM emits both FCs in the SAME model response, the
// base flow's handleFunctionCalls spawns one goroutine per FC and
// joins them. This test is the primary -race target — it must remain
// race-free under -race -count=N.
//
// The recording captures whichever shape Gemini chose for the
// recording session (one event with two FCs, or two events each
// with one FC). Both cases are valid; we assert only that BOTH
// FRs end up in the coordinator's view by the end of the turn.
func TestDelegation_04_ChatToTwoSingleTurnParallel(t *testing.T) {
	translatorEs, err := llmagent.New(llmagent.Config{
		Name:        "translator_es",
		Description: "Translates an English phrase into Spanish.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeSingleTurn,
		Instruction: `Translate the given English phrase into Spanish. Reply
with ONLY the Spanish translation, no other words.`,
	})
	if err != nil {
		t.Fatalf("translator_es: %v", err)
	}
	translatorFr, err := llmagent.New(llmagent.Config{
		Name:        "translator_fr",
		Description: "Translates an English phrase into French.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeSingleTurn,
		Instruction: `Translate the given English phrase into French. Reply
with ONLY the French translation, no other words.`,
	})
	if err != nil {
		t.Fatalf("translator_fr: %v", err)
	}
	coordinator, err := llmagent.New(llmagent.Config{
		Name:        "coordinator",
		Description: "Calls both translators (in parallel) and reports both.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeChat,
		SubAgents:   []agent.Agent{translatorEs, translatorFr},
		Instruction: `When the user asks for translations into multiple
languages, call BOTH translator sub-agents in your NEXT response —
prefer putting both function calls in a single response so they run
in parallel. After both return, reply with one short sentence
listing both translations.`,
	})
	if err != nil {
		t.Fatalf("coordinator: %v", err)
	}

	dr := newDelegationRunner(t, coordinator)
	// Parallel single_turn dispatch races to write events into the
	// session in non-deterministic order; skip the YAML dump so the
	// checked-in testdata file doesn't churn on every replay. Turn-1
	// behaviour is still asserted directly against the event stream.
	dr.SkipDump()

	// --- Turn 1: translate 'good morning' to ES + FR ---
	events1 := dr.turn("Translate 'good morning' to both Spanish and French.")

	// Both translator FCs and FRs present.
	if got := len(collectFCsByName(events1, "translator_es")); got < 1 {
		t.Errorf("turn 1: translator_es FCs = %d, want >= 1", got)
	}
	if got := len(collectFCsByName(events1, "translator_fr")); got < 1 {
		t.Errorf("turn 1: translator_fr FCs = %d, want >= 1", got)
	}
	if got := len(collectFRsByName(events1, "translator_es")); got < 1 {
		t.Errorf("turn 1: translator_es FRs = %d, want >= 1", got)
	}
	if got := len(collectFRsByName(events1, "translator_fr")); got < 1 {
		t.Errorf("turn 1: translator_fr FRs = %d, want >= 1", got)
	}

	// Each sub-agent's events live in a distinct sub-branch (the
	// parallel-execution invariant).
	esBranches := uniqueBranches(eventsByAuthor(events1, "translator_es"))
	frBranches := uniqueBranches(eventsByAuthor(events1, "translator_fr"))
	if len(esBranches) != 1 || esBranches[0] == "" {
		t.Errorf("turn 1: translator_es branches = %v, want exactly 1 non-empty", esBranches)
	}
	if len(frBranches) != 1 || frBranches[0] == "" {
		t.Errorf("turn 1: translator_fr branches = %v, want exactly 1 non-empty", frBranches)
	}
	if len(esBranches) > 0 && len(frBranches) > 0 && esBranches[0] == frBranches[0] {
		t.Errorf("turn 1: translator_es and translator_fr share branch %q; each "+
			"single_turn sub-agent must run in its own sub-branch", esBranches[0])
	}

	// Note: this scenario is intentionally single-turn. Adding a
	// second turn makes the replay flaky because parallel
	// single_turn dispatch orders FRs non-deterministically into
	// the session; the next turn's LLM request bytes (which include
	// the session contents) then don't match what httprr recorded.
	// Conversation-continuity across user turns is covered by the
	// other 8 scenarios.
}

// =============================================================================
// Scenario 5: chat coordinator → chat sub-agent → task sub-sub-agent
// (3-level nesting via transfer_to_agent + task delegation).
// =============================================================================

// TestDelegation_05_ChatChainedThroughChatToTask covers a 3-level
// hierarchy: a chat root transfers to a chat middle, which then
// delegates to a task leaf. Exercises:
//   - transfer_to_agent routing through the mode wrapper (so the
//     transferred-to chat agent still engages runChat), and
//   - task delegation from a non-root chat agent (writer dispatches
//     editor via runChat's TaskAgentTool intercept).
//
// adk-go's base_flow handles transfer_to_agent inline by calling the
// target via RunNode (NOT plain Run), which re-engages the LlmAgent
// mode wrapper for the target. End-effect matches adk-python's
// DynamicNodeScheduler re-dispatch behaviour at
// _dynamic_node_scheduler.py:228-281, without needing the
// scheduler-level loop or resumability bookkeeping.
func TestDelegation_05_ChatChainedThroughChatToTask(t *testing.T) {
	editedDocSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"edited": {Type: genai.TypeString},
		},
		Required: []string{"edited"},
	}

	editor, err := llmagent.New(llmagent.Config{
		Name:         "editor",
		Description:  "Edits a piece of text for clarity.",
		Model:        newDelegationModel(t),
		Mode:         llmagent.ModeTask,
		OutputSchema: editedDocSchema,
		Instruction: `Read the text in the request, produce a polished version
(fix grammar, tighten phrasing), and immediately call ` + "`finish_task`" + `
with {"edited":"<polished text>"}. Never ask follow-up questions.`,
	})
	if err != nil {
		t.Fatalf("editor: %v", err)
	}
	writer, err := llmagent.New(llmagent.Config{
		Name:        "writer",
		Description: "Polishes a draft by delegating to the editor sub-agent.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeChat,
		SubAgents:   []agent.Agent{editor},
		Instruction: `For EACH draft you receive (a new draft on every
user turn), call the editor sub-agent ONCE with a request
containing the draft text. After editor returns its structured
` + "`output`" + `, reply with one short sentence presenting the edited
text. Always call editor for any new draft, even if you've edited
similar drafts earlier in the conversation.`,
	})
	if err != nil {
		t.Fatalf("writer: %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name:        "root",
		Description: "Routes editing requests to the writer agent.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeChat,
		SubAgents:   []agent.Agent{writer},
		Instruction: `For ANY editing request, immediately call
transfer_to_agent with agent_name="writer". Do not attempt to edit
the draft yourself. Do not produce any final text — let writer
respond to the user.`,
	})
	if err != nil {
		t.Fatalf("root: %v", err)
	}

	dr := newDelegationRunner(t, root)

	// --- Turn 1: editing request ---
	events1 := dr.turn("Please edit this draft: 'me and him goes to the store yesterdy'.")

	// (1) Root issued the transfer to writer.
	transferFCs := collectFCsByName(events1, "transfer_to_agent")
	if len(transferFCs) < 1 {
		t.Errorf("turn 1: transfer_to_agent FCs = %d, want >= 1", len(transferFCs))
	}
	var transferToWriter bool
	for _, fc := range transferFCs {
		if name, _ := fc.Args["agent_name"].(string); name == "writer" {
			transferToWriter = true
			break
		}
	}
	if !transferToWriter {
		t.Errorf("turn 1: expected a transfer_to_agent call with agent_name=writer; got %v", transferFCs)
	}

	// (2) Writer dispatched editor exactly once (the
	//     transfer-via-RunNode path engages writer's runChat, which
	//     intercepts the TaskAgentTool FC and dispatches editor).
	if got := len(collectFCsByName(events1, "editor")); got != 1 {
		t.Errorf("turn 1: editor FCs = %d, want 1; events=%v", got, eventSummaries(events1))
	}

	// (3) Editor's finish_task succeeded and the synthesised editor
	//     FR carries the polished text (object schema, passed through).
	editorFRs := collectFRsByName(events1, "editor")
	if len(editorFRs) != 1 {
		t.Fatalf("turn 1: editor FRs = %d, want 1; events=%v", len(editorFRs), eventSummaries(events1))
	}
	if _, ok := editorFRs[0].Response["edited"]; !ok {
		t.Errorf("turn 1: editor FR missing `edited`: %v", editorFRs[0].Response)
	}

	// (4) The final user-facing text is authored by `writer`, NOT
	//     root — proves the transferred-to agent produced the reply
	//     (and not a hallucinated reply from root post-transfer).
	if got := finalModelTextAuthor(events1); got != "writer" {
		t.Errorf("turn 1: final model text author = %q, want %q "+
			"(writer must produce the user-facing reply after the transfer)",
			got, "writer")
	}

	// --- Turn 2: second editing request ---
	//
	// Verifies:
	//   - the runner picks `writer` (NOT root) as the active agent
	//     on turn 2: writer authored turn 1's last model-text event,
	//     so runner.findAgentToRun walks back and selects writer,
	//   - writer's runChat is re-engaged and dispatches editor
	//     AGAIN for the new draft (no re-transfer from root needed),
	//   - the new editor delegation runs in its OWN isolation scope
	//     distinct from turn 1's,
	//   - writer produces the user-facing reply on turn 2 too.
	events2 := dr.turn("Now polish this draft: 'they was eatin lunch quick'.")
	// No new transfer to writer on turn 2 (writer is already active).
	if got := len(collectFCsByName(events2, "transfer_to_agent")); got > 0 {
		t.Errorf("turn 2: transfer_to_agent FCs = %d, want 0; writer "+
			"should remain active without re-transfer", got)
	}
	// New editor dispatch on turn 2 (one per turn, session total = 2).
	if got := len(collectFCsByName(events2, "editor")); got != 1 {
		t.Errorf("turn 2: editor FCs (this turn's stream) = %d, want 1",
			got)
	}
	// Session-wide: 2 distinct editor isolation scopes (turn-1 + turn-2).
	sessionEvents := sessionEventsSnapshot(dr)
	editorScopes := uniqueScopes(eventsByAuthor(sessionEvents, "editor"))
	var nonEmpty []string
	for _, s := range editorScopes {
		if s != "" {
			nonEmpty = append(nonEmpty, s)
		}
	}
	if len(nonEmpty) != 2 {
		t.Errorf("turn 2: distinct editor isolation_scopes session-wide = %v, "+
			"want 2 (turn-1 scope + new turn-2 scope)", nonEmpty)
	}
	// Writer again produces the user-facing reply.
	if got := finalModelTextAuthor(events2); got != "writer" {
		t.Errorf("turn 2: final model text author = %q, want %q "+
			"(writer should be the active agent across turns)", got, "writer")
	}
}

// sessionEventsSnapshot reads every persisted event from the
// delegationRunner's session into a slice. Used by multi-turn tests
// that need session-wide assertions (each dr.turn return value is
// only the events streamed from THAT turn).
func sessionEventsSnapshot(dr *delegationRunner) []*session.Event {
	if dr.sess == nil {
		return nil
	}
	events := dr.sess.Events()
	out := make([]*session.Event, 0, events.Len())
	for i := 0; i < events.Len(); i++ {
		out = append(out, events.At(i))
	}
	return out
}

// =============================================================================
// Scenario 6: chat coordinator → task sub-agent → HITL tool (pause-and-resume
// across two user turns).
// =============================================================================

// TestDelegation_06_ChatToTaskHITLResume covers the full pause/resume
// cycle for a task sub-agent that calls a RequireConfirmation-tagged
// tool:
//
//	Turn 1: user asks for an action that requires confirmation; task
//	        sub-agent calls the confirm tool; the framework wraps the
//	        call as an adk_request_confirmation long-running tool and
//	        pauses; runChat's dispatchAndYield leaves the delegation
//	        UNRESOLVED (no synthesised FR) and drains cleanly.
//	Turn 2: user replies with a FunctionResponse for the pending
//	        adk_request_confirmation FC; the runner stamps the user
//	        event with the still-open task scope
//	        (findActiveTaskIsolationScope); findUnresolvedTaskDelegations
//	        re-dispatches into the same scope; the request_confirmation
//	        processor sees the user reply and resumes the tool; the
//	        task agent eventually calls finish_task and the coordinator
//	        produces its final text.
//
// Both turns are recorded in a single .httprr file.
func TestDelegation_06_ChatToTaskHITLResume(t *testing.T) {
	type ConfirmInput struct{}
	type ConfirmOutput struct {
		Result string `json:"result"`
	}
	var confirmCalls int
	confirmTool, err := functiontool.New(functiontool.Config{
		Name:                "confirm",
		Description:         "Confirms the user wants to proceed with the irreversible action.",
		RequireConfirmation: true,
	}, func(_ context.Context, _ agent.Context, _ ConfirmInput) (ConfirmOutput, error) {
		confirmCalls++
		return ConfirmOutput{Result: "Approved. Proceeding."}, nil
	})
	if err != nil {
		t.Fatalf("confirm tool: %v", err)
	}

	resultSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"done": {Type: genai.TypeBoolean},
		},
		Required: []string{"done"},
	}
	executor, err := llmagent.New(llmagent.Config{
		Name:         "executor",
		Description:  "Executes an irreversible action after the user confirms.",
		Model:        newDelegationModel(t),
		Mode:         llmagent.ModeTask,
		OutputSchema: resultSchema,
		Tools:        []tool.Tool{confirmTool},
		Instruction: `You execute an irreversible action.

WORKFLOW (exact order):
  1. FIRST call the ` + "`confirm`" + ` tool (no arguments). This requires
     human approval and will pause your execution.
  2. AFTER the confirm tool returns a successful result, immediately
     call ` + "`finish_task`" + ` with {"done": true}.

Never skip step 1. Never ask the user follow-up questions in natural
language — always use the confirm tool.`,
	})
	if err != nil {
		t.Fatalf("executor: %v", err)
	}
	coordinator, err := llmagent.New(llmagent.Config{
		Name:        "coordinator",
		Description: "Routes the action to the executor.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeChat,
		SubAgents:   []agent.Agent{executor},
		Instruction: `For any "perform an action" request, call the executor
sub-agent EXACTLY ONCE with a request describing what to do. Once it
returns its structured ` + "`output`" + `, reply with one short sentence
confirming completion.`,
	})
	if err != nil {
		t.Fatalf("coordinator: %v", err)
	}

	dr := newDelegationRunner(t, coordinator)

	// --- Turn 1 ---
	events1 := dr.turn("Please perform the irreversible action.")
	// After turn 1: executor task FC present, but NO finish_task FR
	// yet (the delegation is paused, left unresolved by runChat).
	if got := len(collectFCsByName(events1, "executor")); got != 1 {
		t.Errorf("turn 1: executor FCs = %d, want 1", got)
	}
	if got := len(collectFRsByName(events1, "executor")); got != 0 {
		t.Errorf("turn 1: executor FRs = %d, want 0 (delegation must "+
			"remain unresolved across the HITL pause)", got)
	}
	// A long-running adk_request_confirmation FC event must be
	// present so the consumer (in production: console launcher; in
	// this test: the test itself) can render the prompt.
	pendingID := findPendingConfirmationID(events1)
	if pendingID == "" {
		t.Fatalf("turn 1: no pending adk_request_confirmation FC found; events=%v",
			eventSummaries(events1))
	}
	// confirmTool was NOT yet invoked (still waiting for confirmation).
	if confirmCalls != 0 {
		t.Errorf("turn 1: confirm handler invocations = %d, want 0", confirmCalls)
	}

	// --- Turn 2: user replies with a confirmation FR ---
	events2 := dr.turnContent(&genai.Content{
		Role: "user",
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       pendingID,
				Name:     "adk_request_confirmation",
				Response: map[string]any{"confirmed": true},
			},
		}},
	})
	// On resume: confirm tool was actually invoked, executor called
	// finish_task with `done=true`, and the coordinator received a
	// synthesised executor FR carrying the structured output.
	if confirmCalls != 1 {
		t.Errorf("turn 2: confirm handler invocations = %d, want 1 (must "+
			"actually run after the user approves)", confirmCalls)
	}
	finishSuccessFRs := successFinishFRs(events2)
	if len(finishSuccessFRs) != 1 {
		t.Errorf("turn 2: finish_task success FRs = %d, want 1", len(finishSuccessFRs))
	}
	executorFRs := collectFRsByName(events2, "executor")
	if len(executorFRs) != 1 {
		t.Fatalf("turn 2: executor FRs = %d, want 1", len(executorFRs))
	}
	if done, _ := executorFRs[0].Response["done"].(bool); !done {
		t.Errorf("turn 2: executor FR done = %v, want true; resp=%v",
			executorFRs[0].Response["done"], executorFRs[0].Response)
	}

	// --- Turn 3: post-resume memory check ---
	//
	// After the HITL resume cycle, the coordinator must remain the
	// active agent and remember that the action completed
	// successfully. No new executor dispatch should happen.
	events3 := dr.turn("Did the action complete successfully?")
	if got := len(collectFCsByName(events3, "executor")); got != 0 {
		t.Errorf("turn 3: executor FCs = %d, want 0 (coordinator should answer "+
			"from history)", got)
	}
	if got := finalModelTextAuthor(events3); got != "coordinator" {
		t.Errorf("turn 3: final model text author = %q, want %q", got, "coordinator")
	}
	// confirmCalls still 1 (no re-resume).
	if confirmCalls != 1 {
		t.Errorf("turn 3: confirm handler invocations = %d, want 1 (no re-resume)",
			confirmCalls)
	}
}

// =============================================================================
// Scenario 7: chat coordinator → task sub-agent that retries after a
// finish_task validation failure.
// =============================================================================

// TestDelegation_07_ChatToTaskValidationRetry covers FinishTaskTool's
// schema-validation path: the sub-agent's first finish_task call
// supplies args that DON'T satisfy the output schema; FinishTaskTool
// returns an error FR (not a Go error), the sub-agent's LLM sees the
// error on its next round, and emits a corrected finish_task call.
// runTask only terminates on a successful finish_task FR, so the
// retry path must complete cleanly.
func TestDelegation_07_ChatToTaskValidationRetry(t *testing.T) {
	answerSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"answer": {Type: genai.TypeInteger, Description: "The numeric answer."},
		},
		Required: []string{"answer"},
	}

	solver, err := llmagent.New(llmagent.Config{
		Name:         "solver",
		Description:  "Returns the integer answer to a simple arithmetic question.",
		Model:        newDelegationModel(t),
		Mode:         llmagent.ModeTask,
		OutputSchema: answerSchema,
		Instruction: `You answer one arithmetic question. On your FIRST
finish_task call, intentionally pass a wrong field name ("wrong_field"
instead of "answer") so the validation fails. On your SECOND call,
correct it and pass {"answer": <integer>}. Never give up; the
validation error message will tell you what to fix.`,
	})
	if err != nil {
		t.Fatalf("solver: %v", err)
	}
	coordinator, err := llmagent.New(llmagent.Config{
		Name:        "coordinator",
		Description: "Delegates to solver and reports the answer.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeChat,
		SubAgents:   []agent.Agent{solver},
		Instruction: `For any arithmetic question, call the solver sub-agent
EXACTLY ONCE with a request restating the question. Reply with the
answer in one short sentence. Do NOT call solver more than once.`,
	})
	if err != nil {
		t.Fatalf("coordinator: %v", err)
	}

	dr := newDelegationRunner(t, coordinator)

	// --- Turn 1: first arithmetic question (with validation retry) ---
	events1 := dr.turn("What is 21 plus 21?")

	// Coordinator delegates exactly once.
	if got := len(collectFCsByName(events1, "solver")); got != 1 {
		t.Errorf("turn 1: solver FCs = %d, want 1", got)
	}

	// Solver emits at least TWO finish_task FCs (one bad, one good).
	finishFCs := collectFCsByName(events1, "finish_task")
	if len(finishFCs) < 2 {
		t.Errorf("turn 1: finish_task FCs = %d, want >= 2 (one validation failure + one success); "+
			"events=%v", len(finishFCs), eventSummaries(events1))
	}

	// At least one error FR (with `error` key) AND exactly one success FR.
	finishFRs := collectFRsByName(events1, "finish_task")
	var errFRs, successFRs int
	for _, fr := range finishFRs {
		if _, hasErr := fr.Response["error"]; hasErr {
			errFRs++
			continue
		}
		if got, _ := fr.Response["result"].(string); got == "Task completed." {
			successFRs++
		}
	}
	if errFRs < 1 {
		t.Errorf("turn 1: finish_task error FRs = %d, want >= 1", errFRs)
	}
	if successFRs != 1 {
		t.Errorf("turn 1: finish_task success FRs = %d, want exactly 1", successFRs)
	}

	// Synthesised solver FR carries the corrected structured payload
	// (object schemas pass through as the FR Response directly).
	solverFRs := collectFRsByName(events1, "solver")
	if len(solverFRs) != 1 {
		t.Fatalf("turn 1: solver FRs = %d, want 1", len(solverFRs))
	}
	if _, ok := solverFRs[0].Response["answer"]; !ok {
		t.Errorf("turn 1: solver FR missing `answer`: %v", solverFRs[0].Response)
	}

	// --- Turn 2: follow-up requiring the answer ---
	//
	// Verifies the validation-retry path didn't break the
	// coordinator's session: on a follow-up turn the coordinator
	// either re-dispatches solver or answers from history; either
	// way the eventual reply should contain "42". This pins that
	// the prior turn's bookkeeping (one synthesised solver FR with
	// the corrected answer, no orphan FR/FC pairs) didn't poison
	// the conversation history.
	events2 := dr.turn("What was the answer to the previous question?")
	if got := finalModelTextAuthor(events2); got != "coordinator" {
		t.Errorf("turn 2: final model text author = %q, want %q", got, "coordinator")
	}
	if reply := finalModelText(events2); !strings.Contains(reply, "42") {
		t.Errorf("turn 2: coordinator's reply %q should mention 42", reply)
	}
}

// =============================================================================
// Scenario 8: chat coordinator → 1 task sub-agent + 1 plain function tool
// (mixed delegation + tool dispatch).
// =============================================================================

// TestDelegation_08_ChatToTaskPlusFunctionTool exercises mixed
// dispatch in the same coordinator: a TaskAgentTool (deferred FR,
// synthesised by runChat) AND a regular FunctionTool (immediate FR,
// produced by the standard tool-execution pipeline) live in the same
// tools dict. The coordinator must distinguish them and the framework
// must route them through the correct paths.
func TestDelegation_08_ChatToTaskPlusFunctionTool(t *testing.T) {
	extractorSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"items": {
				Type:  genai.TypeArray,
				Items: &genai.Schema{Type: genai.TypeString},
			},
		},
		Required: []string{"items"},
	}
	extractor, err := llmagent.New(llmagent.Config{
		Name:         "extractor",
		Description:  "Extracts a list of items from a request.",
		Model:        newDelegationModel(t),
		Mode:         llmagent.ModeTask,
		OutputSchema: extractorSchema,
		Instruction: `Read the request, extract the list of items mentioned, and
immediately call ` + "`finish_task`" + ` with {"items":[...]}.`,
	})
	if err != nil {
		t.Fatalf("extractor: %v", err)
	}

	type SummarizeArgs struct {
		Items []string `json:"items"`
	}
	type SummarizeResult struct {
		Summary string `json:"summary"`
	}
	var summarizeCalls int
	summarize, err := functiontool.New(functiontool.Config{
		Name:        "summarize",
		Description: "Summarises a list of items into a one-line string.",
	}, func(_ context.Context, _ agent.Context, in SummarizeArgs) (SummarizeResult, error) {
		summarizeCalls++
		return SummarizeResult{Summary: "items: " + strings.Join(in.Items, ", ")}, nil
	})
	if err != nil {
		t.Fatalf("summarize: %v", err)
	}

	coordinator, err := llmagent.New(llmagent.Config{
		Name:        "coordinator",
		Description: "First calls extractor, then summarize.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeChat,
		SubAgents:   []agent.Agent{extractor},
		Tools:       []tool.Tool{summarize},
		Instruction: `Two-step workflow:
  1. Call extractor with a request describing the items the user
     mentioned. Wait for its structured ` + "`output`" + ` containing the items list.
  2. Call summarize EXACTLY ONCE, passing the items from step 1.
  3. Reply with one short sentence repeating the summary.`,
	})
	if err != nil {
		t.Fatalf("coordinator: %v", err)
	}

	dr := newDelegationRunner(t, coordinator)

	// --- Turn 1: extract + summarise ---
	events1 := dr.turn("Extract and summarise: apples, bread, eggs.")

	// Task path: extractor delegation + synthesised FR carrying the
	// structured output (object schema, passed through directly).
	if got := len(collectFCsByName(events1, "extractor")); got != 1 {
		t.Errorf("turn 1: extractor FCs = %d, want 1", got)
	}
	exFRs := collectFRsByName(events1, "extractor")
	if len(exFRs) != 1 {
		t.Fatalf("turn 1: extractor FRs = %d, want 1", len(exFRs))
	}
	if _, ok := exFRs[0].Response["items"]; !ok {
		t.Errorf("turn 1: extractor FR missing `items`: %v", exFRs[0].Response)
	}

	// Function-tool path: summarize called via the standard pipeline,
	// FR appears in events without an `output` wrapper.
	if got := len(collectFCsByName(events1, "summarize")); got != 1 {
		t.Errorf("turn 1: summarize FCs = %d, want 1", got)
	}
	sumFRs := collectFRsByName(events1, "summarize")
	if len(sumFRs) != 1 {
		t.Fatalf("turn 1: summarize FRs = %d, want 1", len(sumFRs))
	}
	// Function tool returns are NOT wrapped under an `output` key —
	// they appear as the tool's return value directly under a
	// JSON-friendly shape (e.g. {"summary":"..."}). Verify the
	// returned key from SummarizeResult, not `output`.
	if _, ok := sumFRs[0].Response["summary"]; !ok {
		t.Errorf("turn 1: summarize FR has no `summary` key (function tools "+
			"return their result directly, not under `output`); got %v",
			sumFRs[0].Response)
	}
	if summarizeCalls != 1 {
		t.Errorf("turn 1: summarize handler invocations = %d, want 1", summarizeCalls)
	}

	// --- Turn 2: follow-up about the previous items ---
	//
	// Verifies the coordinator retains access to the prior turn's
	// structured output (the extractor's items) AND the function
	// tool result, so it can answer a follow-up without
	// re-extracting. The mixed-dispatch pipeline state survives
	// across turns.
	events2 := dr.turn("How many items did I ask you to summarise?")
	if got := len(collectFCsByName(events2, "extractor")); got != 0 {
		t.Errorf("turn 2: extractor FCs = %d, want 0 (no re-dispatch)", got)
	}
	if summarizeCalls != 1 {
		t.Errorf("turn 2: summarize handler total invocations = %d, want 1 "+
			"(no new tool calls)", summarizeCalls)
	}
	if got := finalModelTextAuthor(events2); got != "coordinator" {
		t.Errorf("turn 2: final model text author = %q, want %q", got, "coordinator")
	}
	// Reply should mention "3" (the count of items: apples, bread, eggs).
	if reply := finalModelText(events2); !strings.Contains(reply, "3") &&
		!strings.Contains(strings.ToLower(reply), "three") {
		t.Errorf("turn 2: coordinator's reply %q should mention 3 (or 'three')", reply)
	}
}

// =============================================================================
// Scenario 9: chat coordinator → 2 chat peers + explicit peer transfer.
// =============================================================================

// TestDelegation_09_ChatPeerTransferAcrossSiblings covers the classic
// multi-agent transfer pattern: root has two chat children
// (billing, support); user message routes to one, a follow-up routes
// to the other via a peer transfer (billing -> support).
//
// adk-go's runner picks the LAST non-user author as the active agent
// on each subsequent turn (the documented inline-forward divergence
// vs adk-python). This test pins that behaviour: after turn 2's
// peer transfer, the active agent is `support`.
//
// Two user turns are recorded in the same trace file.
func TestDelegation_09_ChatPeerTransferAcrossSiblings(t *testing.T) {
	billing, err := llmagent.New(llmagent.Config{
		Name:        "billing",
		Description: "Handles billing questions.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeChat,
		Instruction: `You are the billing specialist. Answer billing-related
questions concisely. If the user asks about anything OTHER than
billing (e.g. technical support), immediately call transfer_to_agent
with agent_name="support" — do not attempt to answer yourself.`,
	})
	if err != nil {
		t.Fatalf("billing: %v", err)
	}
	support, err := llmagent.New(llmagent.Config{
		Name:        "support",
		Description: "Handles technical support questions.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeChat,
		Instruction: `You are the technical support specialist. Answer
technical questions concisely. If the user asks about anything OTHER
than tech support (e.g. billing), immediately call transfer_to_agent
with agent_name="billing".`,
	})
	if err != nil {
		t.Fatalf("support: %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name:        "root",
		Description: "Routes the user to the right specialist.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeChat,
		SubAgents:   []agent.Agent{billing, support},
		Instruction: `Look at the user's first message and immediately call
transfer_to_agent with agent_name set to "billing" (for billing
questions) or "support" (for technical questions). Never answer
yourself; always transfer.`,
	})
	if err != nil {
		t.Fatalf("root: %v", err)
	}

	dr := newDelegationRunner(t, root)

	// Turn 1: billing question.
	events1 := dr.turn("I have a question about my invoice.")
	// Root must have transferred to billing.
	if !hasTransferTo(events1, "billing") {
		t.Errorf("turn 1: expected transfer_to_agent(agent_name=billing); events=%v",
			eventSummaries(events1))
	}
	// adk-go inlines the transfer: billing should be the last
	// model-text author of turn 1.
	if got := finalModelTextAuthor(events1); got != "billing" {
		t.Errorf("turn 1 final author = %q, want %q (inline-forwarded)",
			got, "billing")
	}

	// Turn 2: technical question — billing should peer-transfer to support.
	events2 := dr.turn("Actually I need help resetting my password.")
	// The transfer FC on turn 2 should be authored by `billing`
	// (the most-recent non-user author from turn 1), NOT root.
	transferAuthors := transferAuthorsByTarget(events2, "support")
	if len(transferAuthors) == 0 {
		t.Errorf("turn 2: expected at least one transfer_to_agent(agent_name=support); events=%v",
			eventSummaries(events2))
	} else {
		var sawBilling bool
		for _, a := range transferAuthors {
			if a == "billing" {
				sawBilling = true
				break
			}
		}
		if !sawBilling {
			t.Errorf("turn 2: peer transfer to support should be authored by `billing` "+
				"(adk-go re-picks the last-active agent on resume); got authors=%v",
				transferAuthors)
		}
	}
	// After turn 2, support should have produced the final reply.
	if got := finalModelTextAuthor(events2); got != "support" {
		t.Errorf("turn 2 final author = %q, want %q", got, "support")
	}

	// --- Turn 3: peer-transfer back from support to billing ---
	//
	// Verifies multi-hop peer transfers in a 3-turn conversation:
	// the user pivots back to billing topics, support transfers
	// back, and billing produces the final reply. Exercises the
	// "active agent picker walks back to last non-user author"
	// invariant across THREE turns.
	events3 := dr.turn("Actually, back to my invoice — what's the latest charge?")
	transferAuthorsToBilling := transferAuthorsByTarget(events3, "billing")
	if len(transferAuthorsToBilling) == 0 {
		t.Errorf("turn 3: expected at least one transfer_to_agent(agent_name=billing); events=%v",
			eventSummaries(events3))
	} else {
		var sawSupport bool
		for _, a := range transferAuthorsToBilling {
			if a == "support" {
				sawSupport = true
				break
			}
		}
		if !sawSupport {
			t.Errorf("turn 3: peer transfer back to billing should be authored by `support` "+
				"(last-active agent from turn 2); got authors=%v",
				transferAuthorsToBilling)
		}
	}
	// After turn 3, billing should produce the final reply.
	if got := finalModelTextAuthor(events3); got != "billing" {
		t.Errorf("turn 3 final author = %q, want %q", got, "billing")
	}
}

// =============================================================================
// Scenario 10: workflow rejection of task-mode agents as static graph nodes
// (negative test).
// =============================================================================

// TestDelegation_10_TaskAgentRejectedAsStaticGraphNode pins
// workflow.New's deliberate rejection of task-mode LlmAgents as
// graph nodes (workflow/validation.go:validateNoTaskModeGraphNodes).
// The rationale is that the scheduler overwrites node_input on every
// re-entry, which would lose the task brief. Chat and single_turn
// agents ARE accepted as graph nodes (positive controls).
//
// This test does NOT need an LLM — it asserts the construction-time
// validation only. Therefore no testdata/*.httprr file is required.
func TestDelegation_10_TaskAgentRejectedAsStaticGraphNode(t *testing.T) {
	t.Parallel()

	makeAgent := func(name string, mode llmagent.Mode) agent.Agent {
		t.Helper()
		a, err := llmagent.New(llmagent.Config{
			Name: name,
			Mode: mode,
		})
		if err != nil {
			t.Fatalf("llmagent.New(%q, %q): %v", name, mode, err)
		}
		return a
	}

	t.Run("task_mode_rejected", func(t *testing.T) {
		t.Parallel()
		taskAgent := makeAgent("doer", llmagent.ModeTask)
		taskNode, err := workflow.NewAgentNode(taskAgent, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("NewAgentNode: %v", err)
		}
		_, err = workflow.New("wf", workflow.Chain(workflow.Start, taskNode))
		if err == nil {
			t.Fatal("workflow.New accepted a task-mode AgentNode; want an error")
		}
		if !strings.Contains(err.Error(), "task") {
			t.Errorf("error message should mention task mode; got: %v", err)
		}
	})

	t.Run("chat_mode_accepted", func(t *testing.T) {
		t.Parallel()
		chatAgent := makeAgent("chatter", llmagent.ModeChat)
		chatNode, err := workflow.NewAgentNode(chatAgent, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("NewAgentNode: %v", err)
		}
		if _, err := workflow.New("wf", workflow.Chain(workflow.Start, chatNode)); err != nil {
			t.Errorf("workflow.New rejected a chat-mode AgentNode: %v", err)
		}
	})

	t.Run("single_turn_mode_accepted", func(t *testing.T) {
		t.Parallel()
		stAgent := makeAgent("oneshot", llmagent.ModeSingleTurn)
		stNode, err := workflow.NewAgentNode(stAgent, workflow.NodeConfig{})
		if err != nil {
			t.Fatalf("NewAgentNode: %v", err)
		}
		if _, err := workflow.New("wf", workflow.Chain(workflow.Start, stNode)); err != nil {
			t.Errorf("workflow.New rejected a single_turn-mode AgentNode: %v", err)
		}
	})
}

// =============================================================================
// Scenario 11: chat coordinator → single_turn sub-agent that chains 2 tools.
// =============================================================================

// TestDelegation_11_ChatToSingleTurnChainedTools covers a single_turn
// sub-agent that chains two tools where the second tool's argument
// depends on the first tool's result. The chat coordinator delegates
// once; the sub-agent calls tool_one, sees its FR, then calls
// tool_two with the token tool_one returned, and emits a final
// structured reply.
//
// Pins:
//  1. The sub-agent's own tool history (its FC/FR events) is visible
//     to its model on subsequent LLM rounds — captured directly via
//     a BeforeModelCallback.
//  2. The sub-agent emits FCs for BOTH tool_one and tool_two.
//  3. The sub-agent's model is called a bounded number of times.
func TestDelegation_11_ChatToSingleTurnChainedTools(t *testing.T) {
	type token struct {
		Token string `json:"token"`
	}
	type echo struct {
		Result string `json:"result"`
	}

	// tool_one returns a fixed token; tool_two needs that exact
	// token. The model can only call tool_two correctly after
	// seeing tool_one's FR, so the single_turn agent's own tool
	// history must survive across rounds.
	t1, err := functiontool.New(functiontool.Config{
		Name:        "tool_one",
		Description: "Returns a fixed token string. Call this first.",
	}, func(_ context.Context, _ agent.Context, _ struct{}) (token, error) {
		return token{Token: "alpha-42"}, nil
	})
	if err != nil {
		t.Fatalf("tool_one: %v", err)
	}
	t2, err := functiontool.New(functiontool.Config{
		Name:        "tool_two",
		Description: "Echoes the token returned by tool_one.",
	}, func(_ context.Context, _ agent.Context, in token) (echo, error) {
		return echo{Result: "got " + in.Token}, nil
	})
	if err != nil {
		t.Fatalf("tool_two: %v", err)
	}

	// Capture the contents the sub-agent's model sees on each round.
	// BeforeModelCallback runs after all request processors, so the
	// snapshot is exactly what the model receives.
	var (
		mu       sync.Mutex
		requests [][]*genai.Content
	)
	capture := func(_ context.Context, _ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
		mu.Lock()
		defer mu.Unlock()
		snap := make([]*genai.Content, len(req.Contents))
		copy(snap, req.Contents)
		requests = append(requests, snap)
		return nil, nil
	}

	sub, err := llmagent.New(llmagent.Config{
		Name:        "sub",
		Description: "Chains tool_one then tool_two.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeSingleTurn,
		Tools:       []tool.Tool{t1, t2},
		Instruction: `Call tool_one to obtain a token, then call tool_two with
that exact token, then report the result string. Call each tool at most once.`,
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{capture},
	})
	if err != nil {
		t.Fatalf("sub: %v", err)
	}

	coordinator, err := llmagent.New(llmagent.Config{
		Name:        "coordinator",
		Description: "Delegates to sub and reports the answer.",
		Model:       newDelegationModel(t),
		Mode:        llmagent.ModeChat,
		SubAgents:   []agent.Agent{sub},
		Instruction: `Call the sub agent exactly once with a brief description
of what to do, then report its answer in one sentence.`,
	})
	if err != nil {
		t.Fatalf("coordinator: %v", err)
	}

	events := newDelegationRunner(t, coordinator).turn("Run the chained tools.")

	mu.Lock()
	defer mu.Unlock()

	// (1) Bounded rounds: ~3 in the healthy path (tool_one, tool_two,
	// final reply).
	if n := len(requests); n == 0 || n > 8 {
		t.Fatalf("sub-agent model rounds = %d (want 1..8); requests: %s",
			n, summarizeAllRounds(requests))
	}

	// (2) Both tools were called; tool_one at most a few times.
	if n := len(collectFCsByName(events, "tool_one")); n == 0 || n > 3 {
		t.Errorf("tool_one FCs = %d, want 1..3", n)
	}
	if len(collectFCsByName(events, "tool_two")) == 0 {
		t.Errorf("tool_two never called; model could not proceed past tool_one")
	}

	// (3) Some round's contents must contain BOTH tool_one's FC and
	// FR — direct evidence the agent's own tool history survived
	// between LLM rounds.
	var historySurvived bool
	for _, c := range requests {
		if hasFunctionCall(c, "tool_one") && hasFunctionResponse(c, "tool_one") {
			historySurvived = true
			break
		}
	}
	if !historySurvived {
		t.Errorf("no captured round contained both tool_one FC and FR; tool history did not survive across rounds. requests: %s",
			summarizeAllRounds(requests))
	}
}

// hasFunctionCall reports whether any content part across the slice
// is a FunctionCall with the given name. Used by chained-tool tests
// to assert prior-round tool history survives across LLM rounds.
func hasFunctionCall(contents []*genai.Content, name string) bool {
	for _, c := range contents {
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			if p != nil && p.FunctionCall != nil && p.FunctionCall.Name == name {
				return true
			}
		}
	}
	return false
}

// hasFunctionResponse reports whether any content part across the
// slice is a FunctionResponse with the given name.
func hasFunctionResponse(contents []*genai.Content, name string) bool {
	for _, c := range contents {
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			if p != nil && p.FunctionResponse != nil && p.FunctionResponse.Name == name {
				return true
			}
		}
	}
	return false
}

// summarizeContents returns a compact one-line description of an
// LLMRequest.Contents slice for use in t.Logf / t.Errorf diagnostics.
func summarizeContents(contents []*genai.Content) string {
	var b strings.Builder
	for i, c := range contents {
		if i > 0 {
			b.WriteString(" / ")
		}
		if c == nil {
			b.WriteString("<nil>")
			continue
		}
		b.WriteString(c.Role)
		b.WriteString(":[")
		for j, p := range c.Parts {
			if j > 0 {
				b.WriteString(",")
			}
			switch {
			case p == nil:
				b.WriteString("<nil>")
			case p.FunctionCall != nil:
				b.WriteString("FC(" + p.FunctionCall.Name + ")")
			case p.FunctionResponse != nil:
				b.WriteString("FR(" + p.FunctionResponse.Name + ")")
			case p.Text != "":
				b.WriteString("text")
			default:
				b.WriteString("?")
			}
		}
		b.WriteString("]")
	}
	return b.String()
}

// summarizeAllRounds formats every captured round's contents on a
// single labelled line for use in failure diagnostics.
func summarizeAllRounds(rounds [][]*genai.Content) string {
	parts := make([]string, len(rounds))
	for i, c := range rounds {
		parts[i] = fmt.Sprintf("round %d: %s", i+1, summarizeContents(c))
	}
	return strings.Join(parts, " | ")
}

// =============================================================================
// Diagnostic helpers
// =============================================================================

// findPendingConfirmationID scans events for an
// adk_request_confirmation FC whose id has no matching FR (i.e.
// still awaiting the user's reply). Returns the FC id, or "" if
// none.
func findPendingConfirmationID(events []*session.Event) string {
	resolved := map[string]struct{}{}
	for _, fr := range collectFRsByName(events, "adk_request_confirmation") {
		resolved[fr.ID] = struct{}{}
	}
	for _, fc := range collectFCsByName(events, "adk_request_confirmation") {
		if _, done := resolved[fc.ID]; !done {
			return fc.ID
		}
	}
	return ""
}

// successFinishFRs returns every finish_task FR whose Response has
// {"result": "Task completed."} — i.e. validation passed.
func successFinishFRs(events []*session.Event) []*genai.FunctionResponse {
	var out []*genai.FunctionResponse
	for _, fr := range collectFRsByName(events, "finish_task") {
		if got, _ := fr.Response["result"].(string); got == "Task completed." {
			out = append(out, fr)
		}
	}
	return out
}

// finalModelTextAuthor returns the Author of the last event whose
// content is a non-empty model text part (i.e. the final
// user-facing message produced in the stream). Returns "" if none.
func finalModelTextAuthor(events []*session.Event) string {
	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]
		if ev == nil || ev.LLMResponse.Content == nil {
			continue
		}
		if ev.LLMResponse.Content.Role != "model" {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p != nil && p.Text != "" && !p.Thought {
				return ev.Author
			}
		}
	}
	return ""
}

// finalModelText returns the concatenated text of the last event
// whose content is one or more non-thought model text parts. Used in
// turn-2 assertions to verify the coordinator's reply mentions
// expected substrings (basic conversational-memory check).
func finalModelText(events []*session.Event) string {
	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]
		if ev == nil || ev.LLMResponse.Content == nil {
			continue
		}
		if ev.LLMResponse.Content.Role != "model" {
			continue
		}
		var parts []string
		for _, p := range ev.LLMResponse.Content.Parts {
			if p != nil && p.Text != "" && !p.Thought {
				parts = append(parts, p.Text)
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, "")
		}
	}
	return ""
}

// mentionsAll reports whether s contains every needle (case-insensitive).
// Useful for asserting an LLM reply mentions all of a known set of
// items without pinning exact wording.
func mentionsAll(s string, needles ...string) bool {
	lo := strings.ToLower(s)
	for _, n := range needles {
		if !strings.Contains(lo, strings.ToLower(n)) {
			return false
		}
	}
	return true
}

// hasTransferTo reports whether any event carries a
// transfer_to_agent FC targeting the given agent_name.
func hasTransferTo(events []*session.Event, name string) bool {
	for _, fc := range collectFCsByName(events, "transfer_to_agent") {
		if got, _ := fc.Args["agent_name"].(string); got == name {
			return true
		}
	}
	return false
}

// transferAuthorsByTarget returns the distinct Authors of every
// event that carried a transfer_to_agent FC targeting `target`.
func transferAuthorsByTarget(events []*session.Event, target string) []string {
	seen := map[string]struct{}{}
	var out []string
	for _, ev := range events {
		if ev == nil || ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p == nil || p.FunctionCall == nil || p.FunctionCall.Name != "transfer_to_agent" {
				continue
			}
			if name, _ := p.FunctionCall.Args["agent_name"].(string); name != target {
				continue
			}
			if _, dup := seen[ev.Author]; dup {
				continue
			}
			seen[ev.Author] = struct{}{}
			out = append(out, ev.Author)
		}
	}
	return out
}

// uniqueBranches returns the deduplicated, order-preserved set of
// Branch values across the given events.
func uniqueBranches(events []*session.Event) []string {
	seen := map[string]struct{}{}
	var out []string
	for _, ev := range events {
		if _, ok := seen[ev.Branch]; ok {
			continue
		}
		seen[ev.Branch] = struct{}{}
		out = append(out, ev.Branch)
	}
	return out
}

// uniqueScopes returns the deduplicated, order-preserved set of
// IsolationScope values across the given events.
func uniqueScopes(events []*session.Event) []string {
	seen := map[string]struct{}{}
	var out []string
	for _, ev := range events {
		if _, ok := seen[ev.IsolationScope]; ok {
			continue
		}
		seen[ev.IsolationScope] = struct{}{}
		out = append(out, ev.IsolationScope)
	}
	return out
}

// branchesOf returns each event's Branch field, used in failure
// diagnostics for sub-branch assertions.
func branchesOf(events []*session.Event) []string {
	out := make([]string, 0, len(events))
	for _, ev := range events {
		out = append(out, ev.Branch)
	}
	return out
}

// eventSummary returns a compact human-readable description of an
// event, used only in failure messages.
func eventSummary(ev *session.Event) string {
	if ev == nil {
		return "<nil>"
	}
	var parts []string
	parts = append(parts, "author="+ev.Author)
	if ev.IsolationScope != "" {
		parts = append(parts, "scope="+ev.IsolationScope)
	}
	if ev.LLMResponse.Content != nil {
		for _, p := range ev.LLMResponse.Content.Parts {
			switch {
			case p == nil:
				continue
			case p.FunctionCall != nil:
				parts = append(parts, "FC:"+p.FunctionCall.Name)
			case p.FunctionResponse != nil:
				parts = append(parts, "FR:"+p.FunctionResponse.Name)
			case p.Text != "":
				t := p.Text
				if len(t) > 40 {
					t = t[:40] + "..."
				}
				parts = append(parts, "text="+t)
			}
		}
	}
	return "{" + strings.Join(parts, " ") + "}"
}

// eventSummaries renders eventSummary over a slice.
func eventSummaries(events []*session.Event) []string {
	out := make([]string, 0, len(events))
	for _, ev := range events {
		out = append(out, eventSummary(ev))
	}
	return out
}
