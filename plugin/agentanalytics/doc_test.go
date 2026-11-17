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

package agentanalytics

import (
	"encoding/json"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestDocGo_ParityBulletsReferenceExistingTests(t *testing.T) {
	fset := token.NewFileSet()
	docAST, err := parser.ParseFile(fset, "doc.go", nil, parser.ParseComments)
	if err != nil {
		t.Fatalf("parser.ParseFile(doc.go): %v", err)
	}
	if docAST.Doc == nil {
		t.Fatal("doc.go has no package doc comment")
	}

	testFiles, err := filepath.Glob("*_test.go")
	if err != nil {
		t.Fatalf("filepath.Glob(*_test.go): %v", err)
	}
	testsByFile := make(map[string]map[string]bool, len(testFiles))
	for _, tf := range testFiles {
		f, err := parser.ParseFile(fset, tf, nil, 0)
		if err != nil {
			t.Fatalf("parser.ParseFile(%s): %v", tf, err)
		}
		funcs := make(map[string]bool)
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv != nil || !strings.HasPrefix(fn.Name.Name, "Test") {
				continue
			}
			funcs[fn.Name.Name] = true
		}
		testsByFile[tf] = funcs
	}

	var inSection bool
	var bullets []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			bullets = append(bullets, cur.String())
			cur.Reset()
		}
	}
	for _, c := range docAST.Doc.List {
		line := strings.TrimPrefix(c.Text, "//")
		if strings.Contains(line, "# Behavioral Differences from adk-python") {
			inSection = true
			continue
		}
		if !inSection {
			continue
		}
		if strings.HasPrefix(line, "   - ") {
			flush()
			cur.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "   - ")))
			continue
		}
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || !strings.HasPrefix(line, "     ") {
			flush()
			continue
		}
		if cur.Len() > 0 {
			cur.WriteByte(' ')
			cur.WriteString(trimmed)
		}
	}
	flush()

	const wantBulletCount = 24
	if len(bullets) != wantBulletCount {
		t.Fatalf("parsed %d behavioral-differences bullets in doc.go, want %d", len(bullets), wantBulletCount)
	}

	pinnedRe := regexp.MustCompile(`\(Pinned by ([^()]+)\)\.$`)
	clauseRe := regexp.MustCompile(`([A-Za-z0-9_,\s]+?)\s+in\s+([a-z0-9_]+_test\.go)`)
	sepOnlyRe := regexp.MustCompile(`^[\s,]*(?:and[\s,]*)?$`)
	testNameRe := regexp.MustCompile(`^Test[A-Za-z0-9_]+$`)

	for i, bullet := range bullets {
		m := pinnedRe.FindStringSubmatch(bullet)
		if m == nil {
			t.Errorf("bullet %d missing trailing \"(Pinned by ...).\" citation: %q", i+1, bullet)
			continue
		}
		inner := m[1]
		locs := clauseRe.FindAllStringSubmatchIndex(inner, -1)
		if len(locs) == 0 {
			t.Errorf("bullet %d has no \"<Test> in <file>_test.go\" clause in %q", i+1, inner)
			continue
		}
		pos := 0
		citedCount := 0
		for _, loc := range locs {
			gap := inner[pos:loc[0]]
			if !sepOnlyRe.MatchString(gap) {
				t.Errorf("bullet %d has unexpected text %q between citations in %q", i+1, gap, inner)
			}
			pos = loc[1]
			namesRaw := inner[loc[2]:loc[3]]
			fileName := inner[loc[4]:loc[5]]
			funcs, ok := testsByFile[fileName]
			if !ok {
				t.Errorf("bullet %d references unknown test file %q", i+1, fileName)
				continue
			}
			cleaned := strings.NewReplacer(",", " ", " and ", " ").Replace(namesRaw)
			tokens := strings.Fields(cleaned)
			if len(tokens) == 0 {
				t.Errorf("bullet %d has empty test list before %q", i+1, fileName)
				continue
			}
			for _, tok := range tokens {
				if tok == "and" {
					continue
				}
				if !testNameRe.MatchString(tok) {
					t.Errorf("bullet %d has invalid test identifier %q for %s", i+1, tok, fileName)
					continue
				}
				if !funcs[tok] {
					t.Errorf("bullet %d references %s in %s, which does not exist", i+1, tok, fileName)
					continue
				}
				citedCount++
			}
		}
		if tail := inner[pos:]; !sepOnlyRe.MatchString(tail) {
			t.Errorf("bullet %d has unexpected trailing text %q in %q", i+1, tail, inner)
		}
		if citedCount == 0 {
			t.Errorf("bullet %d cited 0 existing test functions", i+1)
		}
	}
}

func TestDocParity_ErrorTracebackAndUnclosedSpanSyntheticError(t *testing.T) {
	t.Run("recorded error populates content.error_traceback with Error() text on AGENT_ERROR and INVOCATION_ERROR", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors
		errs.add(p.BeforeRunCallback()(mCtx))
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeModelCallback()(mCtx, &model.LLMRequest{Model: "gemini-test"}))
		if res, err := p.OnModelErrorCallback()(mCtx, &model.LLMRequest{Model: "gemini-test"}, errors.New("model exploded")); res != nil || err != nil {
			t.Fatalf("OnModelErrorCallback() = %v, %v; want nil, nil", res, err)
		}
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationError))
		agentErrRow := onlyRowOfType(t, rows, EventAgentError)
		if got := agentErrRow["error_message"]; got != "model exploded" {
			t.Errorf("AGENT_ERROR error_message = %v, want %q", got, "model exploded")
		}
		checkJSON(t, "AGENT_ERROR content", jsonObjectColumn(t, agentErrRow, columnContent),
			`{"error_traceback":"model exploded"}`)

		invErrRow := onlyRowOfType(t, rows, EventInvocationError)
		if got := invErrRow["error_message"]; got != "model exploded" {
			t.Errorf("INVOCATION_ERROR error_message = %v, want %q", got, "model exploded")
		}
		checkJSON(t, "INVOCATION_ERROR content", jsonObjectColumn(t, invErrRow, columnContent),
			`{"error_traceback":"model exploded"}`)
	})

	t.Run("unclosed agent span with no recorded error synthesizes error_message and content.error_traceback", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors
		errs.add(p.BeforeRunCallback()(mCtx))
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		const wantMsg = `agent "test-agent" execution failed before completion`
		rows := collectRows(t, requests, untilEventType(EventInvocationError))
		agentErrRow := onlyRowOfType(t, rows, EventAgentError)
		if got := agentErrRow["status"]; got != "ERROR" {
			t.Errorf("AGENT_ERROR status = %v, want ERROR", got)
		}
		if got := agentErrRow["error_message"]; got != wantMsg {
			t.Errorf("AGENT_ERROR error_message = %v, want %q", got, wantMsg)
		}
		checkJSON(t, "AGENT_ERROR content", jsonObjectColumn(t, agentErrRow, columnContent),
			`{"error_traceback":"agent \"test-agent\" execution failed before completion"}`)

		invErrRow := onlyRowOfType(t, rows, EventInvocationError)
		if got := invErrRow["error_message"]; got != wantMsg {
			t.Errorf("INVOCATION_ERROR error_message = %v, want %q", got, wantMsg)
		}
		checkJSON(t, "INVOCATION_ERROR content", jsonObjectColumn(t, invErrRow, columnContent),
			`{"error_traceback":"agent \"test-agent\" execution failed before completion"}`)
	})
}

func TestDocParity_PausedLongRunningToolWritesTwoToolCompletedRows(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	toolCtx := withFunctionCall(mCtx, "call-lr-1")
	deployTool := &fakeTool{name: "deploy_job", longRunning: true}
	args := map[string]any{"region": "us-central1"}

	var errs callbackErrors
	errs.add(p.BeforeToolCallback()(toolCtx, deployTool, args))
	errs.add(p.AfterToolCallback()(toolCtx, deployTool, args, map[string]any{"status": "pending", "ticket": "t-1"}, nil))
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:                 "ev-lr-pause",
		Author:             "test-agent",
		LongRunningToolIDs: []string{"call-lr-1"},
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "call-lr-1", Name: "deploy_job", Args: args}},
				},
			},
		},
	}))
	errs.add(p.OnUserMessageCallback()(mCtx, &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{
				ID:       "call-lr-1",
				Name:     "deploy_job",
				Response: map[string]any{"status": "done", "job_id": "j-99"},
			}},
		},
	}))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	wantTypes := []string{
		EventToolStarting,
		EventToolCompleted,
		EventToolPaused,
		EventUserMessageReceived,
		EventToolCompleted,
		EventInvocationCompleted,
	}
	if diff := cmp.Diff(wantTypes, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}

	pausedRow := onlyRowOfType(t, rows, EventToolPaused)
	pausedAttrs := jsonObjectColumn(t, pausedRow, "attributes")
	checkJSONAt(t, pausedAttrs, `"tool"`, "adk", "pause_kind")
	checkJSONAt(t, pausedAttrs, `"call-lr-1"`, "adk", "function_call_id")

	completedRows := rowsOfType(rows, EventToolCompleted)
	if len(completedRows) != 2 {
		t.Fatalf("got %d TOOL_COMPLETED rows, want 2", len(completedRows))
	}
	initialAttrs := jsonObjectColumn(t, completedRows[0], "attributes")
	checkJSONAt(t, initialAttrs, `"call-lr-1"`, "adk", "function_call_id")
	if v, ok := lookup(initialAttrs, "adk", "pause_kind"); ok {
		t.Errorf("initial TOOL_COMPLETED adk.pause_kind = %v, want absent", v)
	}
	checkJSONAt(t, jsonObjectColumn(t, completedRows[0], columnContent), `"pending"`, "result", "status")

	resumedAttrs := jsonObjectColumn(t, completedRows[1], "attributes")
	checkJSONAt(t, resumedAttrs, `"call-lr-1"`, "adk", "function_call_id")
	checkJSONAt(t, resumedAttrs, `"tool"`, "adk", "pause_kind")
	checkJSONAt(t, jsonObjectColumn(t, completedRows[1], columnContent), `"done"`, "result", "status")
}

func TestDocParity_AgentResponseExcludesLongRunningToolsAndCompaction(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	var errs callbackErrors

	// Non-empty LongRunningToolIDs suppresses AGENT_RESPONSE even when text is present.
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:                 "ev-lr-text",
		Author:             "test-agent",
		LongRunningToolIDs: []string{"call-wait-1"},
		LLMResponse:        model.LLMResponse{Content: textContent("model", "waiting for external job")},
	}))

	// Actions.Compaction suppresses AGENT_RESPONSE even when text is present (verified directly
	// against hasActionsCompaction and via OnEventCallback when EventActions.Compaction exists).
	type fakeCompaction struct {
		StartTimestamp   time.Time
		EndTimestamp     time.Time
		CompactedContent *genai.Content
	}
	if !hasActionsCompaction(struct{ Compaction *fakeCompaction }{
		Compaction: &fakeCompaction{
			StartTimestamp:   time.Date(2026, 10, 10, 1, 0, 0, 0, time.UTC),
			EndTimestamp:     time.Date(2026, 10, 10, 1, 5, 0, 0, time.UTC),
			CompactedContent: textContent("model", "compacted summary"),
		},
	}) {
		t.Error("hasActionsCompaction() = false, want true when Compaction is non-nil")
	}
	if hasActionsCompaction(struct{ Compaction *fakeCompaction }{Compaction: nil}) {
		t.Error("hasActionsCompaction() = true, want false when Compaction is nil")
	}

	evCompact := &session.Event{
		ID:          "ev-compact-text",
		Author:      "test-agent",
		LLMResponse: model.LLMResponse{Content: textContent("model", "compacted summary")},
	}
	hasTypedCompaction := false
	if cf := reflect.ValueOf(&evCompact.Actions).Elem().FieldByName("Compaction"); cf.IsValid() && cf.CanSet() {
		hasTypedCompaction = true
		comp := reflect.New(cf.Type().Elem())
		comp.Elem().FieldByName("StartTimestamp").Set(reflect.ValueOf(time.Date(2026, 10, 10, 1, 0, 0, 0, time.UTC)))
		comp.Elem().FieldByName("EndTimestamp").Set(reflect.ValueOf(time.Date(2026, 10, 10, 1, 5, 0, 0, time.UTC)))
		comp.Elem().FieldByName("CompactedContent").Set(reflect.ValueOf(textContent("model", "compacted summary")))
		cf.Set(comp)
		errs.add(p.OnEventCallback()(mCtx, evCompact))
	}

	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	if n := len(rowsOfType(rows, EventAgentResponse)); n != 0 {
		t.Errorf("got %d AGENT_RESPONSE rows, want 0; event types: %v", n, eventTypes(rows))
	}
	if hasTypedCompaction {
		onlyRowOfType(t, rows, EventEventCompaction)
	}
}

func TestDocParity_OffloadedSingleTextLLMResponseWritesEmptyResponseString(t *testing.T) {
	off := &fakeOffloader{}
	p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
		c.MaxInlineTextSize = 20
	})
	longResp := "OFFLOADED-SINGLE-" + strings.Repeat("Z", 80)
	if _, err := p.AfterModelCallback()(mCtx, &model.LLMResponse{Content: textContent("model", longResp)}, nil); err != nil {
		t.Fatalf("AfterModelCallback() error: %v", err)
	}
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	row := onlyRowOfType(t, rows, EventLLMResponse)
	checkJSON(t, "single offloaded LLM_RESPONSE content", jsonObjectColumn(t, row, columnContent), `{"response":""}`)

	parts := contentPartsOf(t, row)
	if len(parts) != 1 {
		t.Fatalf("got %d content_parts, want 1", len(parts))
	}
	if parts[0]["storage_mode"] != "GCS_REFERENCE" {
		t.Errorf("storage_mode = %v, want GCS_REFERENCE", parts[0]["storage_mode"])
	}
	uri, _ := parts[0]["uri"].(string)
	if !strings.HasPrefix(uri, "gs://") {
		t.Errorf("uri = %v, want gs:// URI", parts[0]["uri"])
	}
	objRef, ok := parts[0]["object_ref"].(map[string]any)
	if !ok || objRef["uri"] != uri {
		t.Errorf("object_ref = %v, want map with uri %q", parts[0]["object_ref"], uri)
	}
	if preview, _ := parts[0]["text"].(string); !strings.HasPrefix(preview, "OFFLOADED-SINGLE-") {
		t.Errorf("preview text = %q, want prefix OFFLOADED-SINGLE-", preview)
	}
}

func TestDocParity_StateDeltaTrimsLeadingWhitespaceBeforeTempPrefix(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	var errs callbackErrors

	// Leading whitespace before "temp:" must still be scrubbed; temp-only delta emits no row.
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:     "ev-ws-temp-only",
		Author: "test-agent",
		Actions: session.EventActions{
			StateDelta: map[string]any{"  \ttemp:scratch": "ephemeral"},
		},
	}))

	// Mixed delta strips whitespace-padded "temp:" key and preserves durable key.
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:     "ev-ws-temp-mixed",
		Author: "test-agent",
		Actions: session.EventActions{
			StateDelta: map[string]any{
				"  temp:ephemeral": "drop-me",
				"durable":          "keep-me",
			},
		},
	}))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	row := onlyRowOfType(t, rows, EventStateDelta)
	checkJSONAt(t, jsonObjectColumn(t, row, "attributes"), `{"durable":"keep-me"}`, "state_delta")
}

func TestDocParity_HITLInputRequestFromRequestedInput(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	if _, err := p.OnEventCallback()(mCtx, &session.Event{
		ID:     "ev-req-input-only",
		Author: "workflow-agent",
		RequestedInput: &session.RequestInput{
			InterruptID: "int-only-1",
			Message:     "Provide target region",
		},
	}); err != nil {
		t.Fatalf("OnEventCallback() error: %v", err)
	}
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	row := onlyRowOfType(t, rows, EventHITLInputRequest)
	checkJSON(t, "HITL_INPUT_REQUEST content", jsonObjectColumn(t, row, columnContent),
		`{"tool":"adk_request_input","args":{"interruptId":"int-only-1","message":"Provide target region"},"requested_input":{"interruptId":"int-only-1","message":"Provide target region"}}`)
}

func TestDocParity_ADKNodeAttributesAndOmissions(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	var errs callbackErrors

	// Non-workflow event (NodeInfo == nil) must omit attributes.adk.node completely,
	// and render_ui_widgets / rewind_before_invocation_id must never appear.
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:          "ev-no-node",
		Author:      "test-agent",
		LLMResponse: model.LLMResponse{Content: textContent("model", "plain answer")},
	}))

	// Workflow node event with MessageAsOutput and SkipSummarization must record
	// adk.node.message_as_output = true and adk.skip_summarization = true.
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:          "ev-msg-as-output",
		Author:      "workflow-node",
		NodeInfo:    &session.NodeInfo{Path: "wf@0/msg@1", MessageAsOutput: true},
		Actions:     session.EventActions{SkipSummarization: true},
		LLMResponse: model.LLMResponse{Content: textContent("model", "node text answer")},
	}))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	respRows := rowsOfType(rows, EventAgentResponse)
	if len(respRows) != 2 {
		t.Fatalf("got %d AGENT_RESPONSE rows, want 2", len(respRows))
	}

	plainAttrs := jsonObjectColumn(t, respRows[0], "attributes")
	if v, ok := lookup(plainAttrs, "adk", "node"); ok {
		t.Errorf("attributes.adk.node = %v when NodeInfo == nil, want absent", v)
	}
	if v, ok := lookup(plainAttrs, "adk", "skip_summarization"); ok {
		t.Errorf("attributes.adk.skip_summarization = %v when unset, want absent", v)
	}

	msgOutAttrs := jsonObjectColumn(t, respRows[1], "attributes")
	checkJSONAt(t, msgOutAttrs, `{"path":"wf@0/msg@1","run_id":"1","parent_run_id":"0","message_as_output":true}`, "adk", "node")
	checkJSONAt(t, msgOutAttrs, "true", "adk", "skip_summarization")

	for _, row := range rows {
		attrs := jsonObjectColumn(t, row, "attributes")
		for _, omittedKey := range []string{"render_ui_widgets", "rewind_before_invocation_id"} {
			if v, ok := lookup(attrs, "adk", omittedKey); ok {
				t.Errorf("%s attributes.adk.%s = %v, want absent", row["event_type"], omittedKey, v)
			}
		}
	}

	eventType := reflect.TypeFor[session.Event]()
	actionsType := reflect.TypeFor[session.EventActions]()
	for _, field := range []string{"RenderUIWidgets", "RewindBeforeInvocationID"} {
		if _, ok := eventType.FieldByName(field); ok {
			t.Errorf("session.Event unexpectedly defines %s", field)
		}
		if _, ok := actionsType.FieldByName(field); ok {
			t.Errorf("session.EventActions unexpectedly defines %s", field)
		}
	}
}

func TestDocParity_UsageMetadataAndContentUsageShape(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	var errs callbackErrors

	// 1. Non-zero optional counts + modality breakdown arrays: optional usage_metadata and
	// content.usage fields are emitted, while modality details and nil fields are omitted.
	errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{
		Content: textContent("model", "detailed"),
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:        12,
			CandidatesTokenCount:    8,
			TotalTokenCount:         29,
			ThoughtsTokenCount:      4,
			CachedContentTokenCount: 3,
			ToolUsePromptTokenCount: 2,
			PromptTokensDetails:     []*genai.ModalityTokenCount{{Modality: genai.MediaModalityText, TokenCount: 12}},
			CandidatesTokensDetails: []*genai.ModalityTokenCount{{Modality: genai.MediaModalityText, TokenCount: 8}},
		},
	}, nil))

	// 2. Zero/omitted counts: content.usage writes integer 0 for core counts and omits
	// thoughts/cached; usage_metadata omits zero optional keys instead of writing null.
	errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{
		Content:       textContent("model", "zeros"),
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{},
	}, nil))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	respRows := rowsOfType(rows, EventLLMResponse)
	if len(respRows) != 2 {
		t.Fatalf("got %d LLM_RESPONSE rows, want 2", len(respRows))
	}

	// Row 1 checks: full positive optional fields, no modality breakdowns.
	c1 := jsonObjectColumn(t, respRows[0], columnContent)
	checkJSONAt(t, c1, `{"prompt":12,"completion":8,"total":29,"thoughts":4,"cached":3}`, "usage")
	a1 := jsonObjectColumn(t, respRows[0], "attributes")
	checkJSONAt(t, a1, `{"prompt_token_count":12,"candidates_token_count":8,"total_token_count":29,"thoughts_token_count":4,"cached_content_token_count":3,"tool_use_prompt_token_count":2}`, "usage_metadata")

	// Row 2 checks: integer 0 counts and strict omission of zero optional fields / nulls.
	c2 := jsonObjectColumn(t, respRows[1], columnContent)
	checkJSONAt(t, c2, `{"prompt":0,"completion":0,"total":0}`, "usage")
	a2 := jsonObjectColumn(t, respRows[1], "attributes")
	checkJSONAt(t, a2, `{"prompt_token_count":0,"candidates_token_count":0,"total_token_count":0}`, "usage_metadata")
}

func TestDocParity_LLMRequestToolSchemaCamelCaseAndDeterministicOrder(t *testing.T) {
	req := &model.LLMRequest{
		Tools: map[string]any{
			"zebra": &declaredTool{
				fakeTool: fakeTool{name: "zebra", description: "Last alphabetically"},
				decl: &genai.FunctionDeclaration{
					Name: "zebra",
					Parameters: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"tags": {
								Type:     genai.TypeArray,
								MaxItems: new(int64(3)),
								Items:    &genai.Schema{Type: genai.TypeString},
							},
							"target": {
								AnyOf: []*genai.Schema{
									{Type: genai.TypeString},
									{Type: genai.TypeInteger},
								},
							},
						},
					},
				},
			},
			"alpha":  &fakeTool{name: "alpha", description: "First alphabetically"},
			"middle": &fakeTool{name: "middle", description: "Second alphabetically"},
		},
	}

	var prevRaw string
	for run := range 10 {
		tools, ok := llmRequestAttributes(req)["tools"]
		if !ok {
			t.Fatal("llmRequestAttributes() missing tools")
		}
		b, _, err := smartTruncate(tools, -1, true)
		if err != nil {
			t.Fatalf("run %d smartTruncate() error: %v", run, err)
		}
		gotRaw := string(b)
		if prevRaw != "" && gotRaw != prevRaw {
			t.Fatalf("run %d tool order drifted:\ngot:  %s\nwant: %s", run, gotRaw, prevRaw)
		}
		prevRaw = gotRaw

		var parsed []map[string]any
		if err := json.Unmarshal(b, &parsed); err != nil {
			t.Fatalf("json.Unmarshal(%s): %v", b, err)
		}
		var names []string
		for _, item := range parsed {
			names = append(names, item["name"].(string))
		}
		if diff := cmp.Diff([]string{"alpha", "middle", "zebra"}, names); diff != "" {
			t.Fatalf("tool ordering mismatch (-want +got):\n%s", diff)
		}

		zebraParams := parsed[2]["parameters"].(map[string]any)
		props := zebraParams["properties"].(map[string]any)
		tags := props["tags"].(map[string]any)
		if got := tags["maxItems"]; got != float64(3) {
			t.Errorf("tags.maxItems = %v, want camelCase maxItems=3; raw=%v", got, tags)
		}
		if _, ok := tags["max_items"]; ok {
			t.Errorf("unexpected snake_case max_items in %v", tags)
		}
		target := props["target"].(map[string]any)
		if anyOf, ok := target["anyOf"].([]any); !ok || len(anyOf) != 2 {
			t.Errorf("target.anyOf = %v, want camelCase 2-element slice", target["anyOf"])
		}
		if _, ok := target["any_of"]; ok {
			t.Errorf("unexpected snake_case any_of in %v", target)
		}
	}
}

func TestDocParity_ContentPartsFunctionCallArgsTruncated(t *testing.T) {
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.MaxContentLen = 30
	})
	resp := &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						ID:   "call-trunc-1",
						Name: "search_docs",
						Args: map[string]any{
							"api_key": "sk-super-secret",
							"query":   strings.Repeat("word-", 20),
						},
					},
				},
			},
		},
	}
	if _, err := p.AfterModelCallback()(mCtx, resp, nil); err != nil {
		t.Fatalf("AfterModelCallback() error: %v", err)
	}
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	row := onlyRowOfType(t, rows, EventLLMResponse)
	if row["is_truncated"] != true {
		t.Errorf("is_truncated = %v, want true for oversized function_call args", row["is_truncated"])
	}
	parts := contentPartsOf(t, row)
	if len(parts) != 1 {
		t.Fatalf("got %d content_parts, want 1", len(parts))
	}
	attrsStr, _ := parts[0]["part_attributes"].(string)
	if strings.Contains(attrsStr, "sk-super-secret") {
		t.Errorf("part_attributes leaked secret: %s", attrsStr)
	}
	var attrs map[string]any
	if err := json.Unmarshal([]byte(attrsStr), &attrs); err != nil {
		t.Fatalf("json.Unmarshal(part_attributes): %v", err)
	}
	if attrs["function_name"] != "search_docs" {
		t.Errorf("function_name = %v, want %q", attrs["function_name"], "search_docs")
	}
	queryArg, ok := lookup(attrs, "function_call", "args", "query")
	if !ok {
		t.Fatalf("missing function_call.args.query in %s", attrsStr)
	}
	if s, _ := queryArg.(string); !strings.HasSuffix(s, "...[TRUNCATED]") {
		t.Errorf("function_call.args.query = %q, want ...[TRUNCATED] suffix", s)
	}
}

func TestDocParity_ContentFormatterReceivesTypedModelPayloadsAndSkipsNilContent(t *testing.T) {
	var calls []string
	var sawReqType, sawRespType bool
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.ContentFormatter = func(payload any, eventType string) (any, error) {
			calls = append(calls, eventType)
			switch eventType {
			case EventLLMRequest:
				req, ok := payload.(*model.LLMRequest)
				sawReqType = ok
				if ok {
					return map[string]any{"custom_req_model": req.Model}, nil
				}
			case EventLLMResponse:
				resp, ok := payload.(*model.LLMResponse)
				sawRespType = ok
				if ok && resp.Content != nil {
					return map[string]any{"custom_resp": "formatted"}, nil
				}
			}
			return payload, nil
		}
	})

	var errs callbackErrors
	errs.add(p.BeforeRunCallback()(mCtx))
	errs.add(p.BeforeModelCallback()(mCtx, &model.LLMRequest{
		Model:    "gemini-custom",
		Contents: []*genai.Content{textContent("user", "hi")},
	}))
	errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{
		Content: textContent("model", "hello"),
	}, nil))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	if !sawReqType {
		t.Error("ContentFormatter did not receive *model.LLMRequest on EventLLMRequest")
	}
	if !sawRespType {
		t.Error("ContentFormatter did not receive *model.LLMResponse on EventLLMResponse")
	}
	wantCalls := []string{EventLLMRequest, EventLLMResponse}
	if diff := cmp.Diff(wantCalls, calls); diff != "" {
		t.Errorf("ContentFormatter calls mismatch (-want +got):\n%s (nil-content rows like INVOCATION_STARTING/INVOCATION_COMPLETED must be skipped)", diff)
	}
	checkJSON(t, "formatted LLM_REQUEST content", jsonObjectColumn(t, onlyRowOfType(t, rows, EventLLMRequest), columnContent),
		`{"custom_req_model":"gemini-custom"}`)
	checkJSON(t, "formatted LLM_RESPONSE content", jsonObjectColumn(t, onlyRowOfType(t, rows, EventLLMResponse), columnContent),
		`{"custom_resp":"formatted"}`)
}

func TestDocParity_DefaultConfigBatchSizeAndRetryDelay(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.BatchSize != 100 {
		t.Errorf("DefaultConfig().BatchSize = %d, want 100", cfg.BatchSize)
	}
	if cfg.RetryConfig.InitialDelay != 100*time.Millisecond {
		t.Errorf("DefaultConfig().RetryConfig.InitialDelay = %v, want 100ms", cfg.RetryConfig.InitialDelay)
	}

	applied := applyDefaults(Config{})
	if applied.BatchSize != 100 {
		t.Errorf("applyDefaults(Config{}).BatchSize = %d, want 100", applied.BatchSize)
	}
	if applied.RetryConfig.InitialDelay != 100*time.Millisecond {
		t.Errorf("applyDefaults(Config{}).RetryConfig.InitialDelay = %v, want 100ms", applied.RetryConfig.InitialDelay)
	}
}

func TestDocParity_ResponseSummaryAndFunctionNameBounds(t *testing.T) {
	atCallCap := strings.Repeat("c", 256)
	atRespCap := strings.Repeat("r", 256)
	overCapCall := strings.Repeat("c", 257)
	overCapResp := strings.Repeat("r", 257)
	manyParts := responseTextParts(40, 100, "a")
	fortySmall := joinedResponsePieces(responseTextPiece(strings.Repeat("a", 100)), 40)

	t.Run("finite MaxContentLen caps function names at 256 runes and joined summaries at MaxContentLen plus 4096", func(t *testing.T) {
		const maxLen = 100
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.MaxContentLen = maxLen
			c.CustomMetadataAllowlist = []string{"adk_compaction"}
		})
		var errs callbackErrors
		errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{Name: overCapCall}},
					{FunctionResponse: &genai.FunctionResponse{Name: overCapResp}},
				},
			},
		}, nil))
		errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{
			Content: &genai.Content{Role: "model", Parts: manyParts},
		}, nil))
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:          "ev-bounded-agent-resp",
			Author:      "test-agent",
			LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "model", Parts: manyParts}},
		}))
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-bounded-compaction",
			Author: "test-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"adk_compaction": map[string]any{
						"start_timestamp":   1700000000.0,
						"end_timestamp":     1700000100.0,
						"compacted_content": &genai.Content{Role: "model", Parts: manyParts},
					},
				},
			},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		llmRows := rowsOfType(rows, EventLLMResponse)
		if len(llmRows) != 2 {
			t.Fatalf("got %d LLM_RESPONSE rows, want 2", len(llmRows))
		}

		wantNames := "call: " + atCallCap + truncationSuffix + " | resp: " + atRespCap + truncationSuffix
		if got, _ := jsonObjectColumn(t, llmRows[0], columnContent)["response"].(string); got != wantNames {
			t.Errorf("LLM_RESPONSE[0] content.response = %q, want %q", got, wantNames)
		}
		if llmRows[0]["is_truncated"] != true {
			t.Errorf("LLM_RESPONSE[0] is_truncated = %v, want true", llmRows[0]["is_truncated"])
		}

		wantJoined := string([]rune(fortySmall)[:maxLen+4096]) + truncationSuffix
		gotJoined, _ := jsonObjectColumn(t, llmRows[1], columnContent)["response"].(string)
		if gotJoined != wantJoined || utf8.RuneCountInString(gotJoined) != maxLen+4096+len(truncationSuffix) {
			t.Errorf("LLM_RESPONSE[1] content.response = %d runes, want %d runes",
				utf8.RuneCountInString(gotJoined), maxLen+4096+len(truncationSuffix))
		}
		if llmRows[1]["is_truncated"] != true {
			t.Errorf("LLM_RESPONSE[1] is_truncated = %v, want true", llmRows[1]["is_truncated"])
		}

		agentRow := onlyRowOfType(t, rows, EventAgentResponse)
		wantAgent := string([]rune(fortySmall)[:maxLen]) + truncationSuffix
		if got, _ := jsonObjectColumn(t, agentRow, columnContent)["response"].(string); got != wantAgent {
			t.Errorf("AGENT_RESPONSE content.response = %q, want %q", got, wantAgent)
		}
		if agentRow["is_truncated"] != true {
			t.Errorf("AGENT_RESPONSE is_truncated = %v, want true", agentRow["is_truncated"])
		}

		compRow := onlyRowOfType(t, rows, EventEventCompaction)
		if got, _ := jsonObjectColumn(t, compRow, columnContent)["compacted_content"].(string); got != wantAgent {
			t.Errorf("EVENT_COMPACTION content.compacted_content = %q, want %q", got, wantAgent)
		}
		if compRow["is_truncated"] != true {
			t.Errorf("EVENT_COMPACTION is_truncated = %v, want true", compRow["is_truncated"])
		}

		// When MaxContentLen exceeds a capped function-name summary, EVENT_COMPACTION
		// retains the 256-rune function-name cap from formatResponseContent.
		pComp, reqComp, ctxComp := setupTestPluginWithConfig(t, func(c *Config) {
			c.MaxContentLen = 600
			c.CustomMetadataAllowlist = []string{"adk_compaction"}
		})
		if _, err := pComp.OnEventCallback()(ctxComp, &session.Event{
			ID:     "ev-compaction-long-fn",
			Author: "test-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"adk_compaction": map[string]any{
						"start_timestamp": 1700000000.0,
						"end_timestamp":   1700000100.0,
						"compacted_content": &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{FunctionCall: &genai.FunctionCall{Name: overCapCall}},
								{FunctionResponse: &genai.FunctionResponse{Name: overCapResp}},
							},
						},
					},
				},
			},
		}); err != nil {
			t.Fatalf("OnEventCallback(compaction) error: %v", err)
		}
		pComp.AfterRunCallback()(ctxComp)
		compNameRow := onlyRowOfType(t, collectRows(t, reqComp, untilEventType(EventInvocationCompleted)), EventEventCompaction)
		if got, _ := jsonObjectColumn(t, compNameRow, columnContent)["compacted_content"].(string); got != wantNames {
			t.Errorf("EVENT_COMPACTION long-name content.compacted_content = %q, want %q", got, wantNames)
		}
		if compNameRow["is_truncated"] != true {
			t.Errorf("EVENT_COMPACTION long-name is_truncated = %v, want true", compNameRow["is_truncated"])
		}
	})

	t.Run("CaptureFullMessageContent preserves long function names and multi-part summaries while redacting credentials", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.MaxContentLen = 1
			c.CaptureFullMessageContent = true
			c.CustomMetadataAllowlist = []string{"adk_compaction"}
		})
		fullParts := append([]*genai.Part{
			{FunctionCall: &genai.FunctionCall{Name: overCapCall}},
			{FunctionResponse: &genai.FunctionResponse{Name: overCapResp}},
		}, manyParts...)
		const secret = "SYNTHETIC_DOC_SECRET_0123456789"

		var errs callbackErrors
		errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{
			Content: &genai.Content{Role: "model", Parts: fullParts},
		}, nil))
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:          "ev-full-doc-agent-resp",
			Author:      "test-agent",
			LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "model", Parts: manyParts}},
		}))
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-full-doc-compaction",
			Author: "test-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"adk_compaction": map[string]any{
						"start_timestamp":   1700000000.0,
						"end_timestamp":     1700000100.0,
						"compacted_content": &genai.Content{Role: "model", Parts: fullParts},
					},
				},
			},
		}))
		errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{Name: overCapCall + " api_key=" + secret}},
				},
			},
		}, nil))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		llmRows := rowsOfType(rows, EventLLMResponse)
		if len(llmRows) != 2 {
			t.Fatalf("got %d LLM_RESPONSE rows, want 2", len(llmRows))
		}

		wantFull := "call: " + overCapCall + " | resp: " + overCapResp + " | " + fortySmall
		if got, _ := jsonObjectColumn(t, llmRows[0], columnContent)["response"].(string); got != wantFull {
			t.Errorf("LLM_RESPONSE[0] content.response = %d runes, want %d runes",
				utf8.RuneCountInString(got), utf8.RuneCountInString(wantFull))
		}
		if llmRows[0]["is_truncated"] != false {
			t.Errorf("LLM_RESPONSE[0] is_truncated = %v, want false", llmRows[0]["is_truncated"])
		}

		agentRow := onlyRowOfType(t, rows, EventAgentResponse)
		if got, _ := jsonObjectColumn(t, agentRow, columnContent)["response"].(string); got != fortySmall {
			t.Errorf("AGENT_RESPONSE content.response = %d runes, want %d runes",
				utf8.RuneCountInString(got), utf8.RuneCountInString(fortySmall))
		}
		if agentRow["is_truncated"] != false {
			t.Errorf("AGENT_RESPONSE is_truncated = %v, want false", agentRow["is_truncated"])
		}

		compRow := onlyRowOfType(t, rows, EventEventCompaction)
		if got, _ := jsonObjectColumn(t, compRow, columnContent)["compacted_content"].(string); got != wantFull {
			t.Errorf("EVENT_COMPACTION content.compacted_content = %d runes, want %d runes",
				utf8.RuneCountInString(got), utf8.RuneCountInString(wantFull))
		}
		if compRow["is_truncated"] != false {
			t.Errorf("EVENT_COMPACTION is_truncated = %v, want false", compRow["is_truncated"])
		}

		gotRedacted, _ := jsonObjectColumn(t, llmRows[1], columnContent)["response"].(string)
		wantRedacted := "call: " + overCapCall + " api_key=" + redactedValue
		if gotRedacted != wantRedacted {
			t.Errorf("LLM_RESPONSE[1] content.response = %q, want %q", gotRedacted, wantRedacted)
		}
		if strings.Contains(gotRedacted, secret) || strings.Contains(gotRedacted, truncationSuffix) {
			t.Errorf("LLM_RESPONSE[1] content.response = %q, want secret redacted without length truncation", gotRedacted)
		}
		if llmRows[1]["is_truncated"] != true {
			t.Errorf("LLM_RESPONSE[1] is_truncated = %v, want true when redaction scrubs a credential", llmRows[1]["is_truncated"])
		}
	})

	t.Run("MaxContentLen -1 without CaptureFullMessageContent lifts both length caps while preserving redaction when enabled", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.MaxContentLen = -1
			c.CaptureFullMessageContent = false
			c.CustomMetadataAllowlist = []string{"adk_compaction"}
		})
		fullParts := append([]*genai.Part{
			{FunctionCall: &genai.FunctionCall{Name: overCapCall}},
			{FunctionResponse: &genai.FunctionResponse{Name: overCapResp}},
		}, manyParts...)
		const secret = "SYNTHETIC_DOC_NEG_SECRET_0123456789"

		var errs callbackErrors
		errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{
			Content: &genai.Content{Role: "model", Parts: fullParts},
		}, nil))
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:          "ev-neg-doc-agent-resp",
			Author:      "test-agent",
			LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "model", Parts: manyParts}},
		}))
		errs.add(p.OnEventCallback()(mCtx, &session.Event{
			ID:     "ev-neg-doc-compaction",
			Author: "test-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"adk_compaction": map[string]any{
						"start_timestamp":   1700000000.0,
						"end_timestamp":     1700000100.0,
						"compacted_content": &genai.Content{Role: "model", Parts: fullParts},
					},
				},
			},
		}))
		errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{Name: overCapCall + " api_key=" + secret}},
				},
			},
		}, nil))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		llmRows := rowsOfType(rows, EventLLMResponse)
		if len(llmRows) != 2 {
			t.Fatalf("got %d LLM_RESPONSE rows, want 2", len(llmRows))
		}

		wantFull := "call: " + overCapCall + " | resp: " + overCapResp + " | " + fortySmall
		if got, _ := jsonObjectColumn(t, llmRows[0], columnContent)["response"].(string); got != wantFull {
			t.Errorf("LLM_RESPONSE[0] content.response = %d runes, want %d runes",
				utf8.RuneCountInString(got), utf8.RuneCountInString(wantFull))
		}
		if llmRows[0]["is_truncated"] != false {
			t.Errorf("LLM_RESPONSE[0] is_truncated = %v, want false", llmRows[0]["is_truncated"])
		}

		agentRow := onlyRowOfType(t, rows, EventAgentResponse)
		if got, _ := jsonObjectColumn(t, agentRow, columnContent)["response"].(string); got != fortySmall {
			t.Errorf("AGENT_RESPONSE content.response = %d runes, want %d runes",
				utf8.RuneCountInString(got), utf8.RuneCountInString(fortySmall))
		}
		if agentRow["is_truncated"] != false {
			t.Errorf("AGENT_RESPONSE is_truncated = %v, want false", agentRow["is_truncated"])
		}

		compRow := onlyRowOfType(t, rows, EventEventCompaction)
		if got, _ := jsonObjectColumn(t, compRow, columnContent)["compacted_content"].(string); got != wantFull {
			t.Errorf("EVENT_COMPACTION content.compacted_content = %d runes, want %d runes",
				utf8.RuneCountInString(got), utf8.RuneCountInString(wantFull))
		}
		if compRow["is_truncated"] != false {
			t.Errorf("EVENT_COMPACTION is_truncated = %v, want false", compRow["is_truncated"])
		}

		gotRedacted, _ := jsonObjectColumn(t, llmRows[1], columnContent)["response"].(string)
		wantRedacted := "call: " + overCapCall + " api_key=" + redactedValue
		if gotRedacted != wantRedacted {
			t.Errorf("LLM_RESPONSE[1] content.response = %q, want %q", gotRedacted, wantRedacted)
		}
		if llmRows[1]["is_truncated"] != true {
			t.Errorf("LLM_RESPONSE[1] is_truncated = %v, want true when redaction scrubs a credential", llmRows[1]["is_truncated"])
		}

		pUnredacted, reqUnredacted, ctxUnredacted := setupTestPluginWithConfig(t, func(c *Config) {
			c.MaxContentLen = -1
			c.DisableSensitiveKeyRedaction = true
		})
		if _, err := pUnredacted.AfterModelCallback()(ctxUnredacted, &model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{Name: overCapCall + " api_key=" + secret}},
				},
			},
		}, nil); err != nil {
			t.Fatalf("AfterModelCallback(unredacted) error: %v", err)
		}
		pUnredacted.AfterRunCallback()(ctxUnredacted)
		unredactedRow := onlyRowOfType(t, collectRows(t, reqUnredacted, untilEventType(EventInvocationCompleted)), EventLLMResponse)
		wantRaw := "call: " + overCapCall + " api_key=" + secret
		if got, _ := jsonObjectColumn(t, unredactedRow, columnContent)["response"].(string); got != wantRaw {
			t.Errorf("unredacted LLM_RESPONSE content.response = %q, want %q", got, wantRaw)
		}
		if unredactedRow["is_truncated"] != false {
			t.Errorf("unredacted LLM_RESPONSE is_truncated = %v, want false", unredactedRow["is_truncated"])
		}
	})
}
