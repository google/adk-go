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

package llmagent_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/codeexecutor"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// The load and explore lines adk-python v2.12.0 writes at the end of the
// exploration code of data_1_1.csv (_get_data_file_preprocessing_code).
const exploreDataTail11 = "\n\n# Load the dataframe.\ndata_1_1 = pd.read_csv('data_1_1.csv')\n\n# Use `explore_df` to guide my analysis.\nexplore_df(data_1_1)\n"

// runTurns runs a in session s1 once per message, on one runner, and returns
// what each run yielded up to its first error. opts.ctx and opts.artifacts are
// honored as by runFlow.
func runTurns(t *testing.T, a agent.Agent, opts flowOpts, msgs ...*genai.Content) []flowRun {
	t.Helper()
	ctx := opts.ctx
	if ctx == nil {
		ctx = t.Context()
	}
	svc := opts.sessions
	if svc == nil {
		svc = session.InMemoryService()
	}
	key := session.GetRequest{AppName: "test_app", UserID: "test_user", SessionID: "s1"}
	created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: key.AppName, UserID: key.UserID, SessionID: key.SessionID, State: opts.state})
	if err != nil {
		t.Fatalf("session Create: %v", err)
	}
	for _, ev := range opts.history {
		if err := svc.AppendEvent(t.Context(), created.Session, ev); err != nil {
			t.Fatalf("session AppendEvent: %v", err)
		}
	}
	cfg := runner.Config{AppName: key.AppName, Agent: a, SessionService: svc}
	arts := opts.artifacts
	if !opts.noArtifacts {
		if arts == nil {
			arts = newFlowArtifacts()
		}
		cfg.ArtifactService = arts
	}
	r, err := runner.New(cfg)
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	var runs []flowRun
	for _, msg := range msgs {
		run := flowRun{artifacts: arts}
		for ev, err := range r.Run(ctx, key.UserID, key.SessionID, msg, agent.RunConfig{}) {
			if err != nil {
				run.err = err
				break
			}
			run.events = append(run.events, ev)
		}
		got, err := svc.Get(t.Context(), &key)
		if err != nil {
			t.Fatalf("session Get: %v", err)
		}
		run.sess = got.Session
		runs = append(runs, run)
	}
	return runs
}

func csvPart(data string) *genai.Part {
	return &genai.Part{InlineData: &genai.Blob{MIMEType: "text/csv", Data: []byte(data)}}
}

func userParts(parts ...*genai.Part) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: parts}
}

// instructionText returns the text of the request's system instruction, or ""
// when it has none.
func instructionText(req *model.LLMRequest) string {
	if req.Config == nil || req.Config.SystemInstruction == nil {
		return ""
	}
	var texts []string
	for _, p := range req.Config.SystemInstruction.Parts {
		texts = append(texts, p.Text)
	}
	return strings.Join(texts, "")
}

func TestLLMAgent_CodeExecutionFencedInstructionAppendsAndSurvivesErrorGate(t *testing.T) {
	// The instruction text is adk-python v2.12.0's
	// _NON_BUILTIN_EXECUTOR_INSTRUCTION formatted with the first pair, as its
	// append_instructions joins it; the identity sentence sits between.
	const wantPyPair = "Be concise.\n\nYou are an agent. Your internal name is \"test_agent\".\n\n" +
		"# CRITICAL: Code execution format\n\nYou have access to an external Python sandbox managed by the host\napplication. To run Python code, output it inside a fenced markdown\nblock exactly like this:\n\n<py>\nprint(\"hello\")\n\n</py>\n\nDO NOT emit native executable_code parts.\nDO NOT attempt to call a code_execution tool — no such tool is\nregistered for this request and the API will reject the response with\nUNEXPECTED_TOOL_CALL or MALFORMED_FUNCTION_CALL.\n\nAlways wrap Python code in the markdown fence shown above.\n"
	pairs := [][2]string{{"<py>\n", "\n</py>"}, {"```python\n", "\n```"}}

	t.Run("joins_onto_existing_instruction_with_first_delimiter", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("done")}}
		exec := newFlowExecutor(codeexecutor.Settings{OptimizeDataFile: true, CodeBlockDelimiters: pairs})
		a := newPlannerAgent(t, llmagent.Config{Name: "test_agent", Instruction: "Be concise.", Model: llm, CodeExecutor: exec})

		runs := runTurns(t, a, flowOpts{}, genai.NewContentFromText("go", genai.RoleUser))

		if runs[0].err != nil {
			t.Fatalf("run error: %v", runs[0].err)
		}
		if got := instructionText(llm.recorded()[0]); got != wantPyPair {
			t.Errorf("system instruction mismatch (-want +got):\n%s", cmp.Diff(wantPyPair, got))
		}
	})

	t.Run("survives_closed_error_gate", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("done")}}
		exec := newFlowExecutor(codeexecutor.Settings{OptimizeDataFile: true, CodeBlockDelimiters: pairs, ErrorRetryAttempts: new(0)})
		a := newPlannerAgent(t, llmagent.Config{Name: "test_agent", Instruction: "Be concise.", Model: llm, CodeExecutor: exec})

		runs := runTurns(t, a, flowOpts{}, userParts(csvPart("a,b\n1,2\n")))

		if runs[0].err != nil {
			t.Fatalf("run error: %v", runs[0].err)
		}
		req := llm.recorded()[0]
		if got := instructionText(req); got != wantPyPair {
			t.Errorf("system instruction mismatch (-want +got):\n%s", cmp.Diff(wantPyPair, got))
		}
		if got := len(exec.recorded()); got != 0 {
			t.Errorf("executor calls = %d, want 0", got)
		}
		// A closed gate returns before the data files are extracted.
		want := []*genai.Content{userParts(csvPart("a,b\n1,2\n"))}
		if diff := cmp.Diff(want, req.Contents); diff != "" {
			t.Errorf("request contents mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("empty_delimiters_omits_fence_and_keeps_warning", func(t *testing.T) {
		const want = "Be brief.\n\nYou are an agent. Your internal name is \"test_agent\".\n\n" +
			"# CRITICAL: Code execution format\n\nDO NOT emit native executable_code parts.\nDO NOT attempt to call a code_execution tool — no such tool is\nregistered for this request and the API will reject the response with\nUNEXPECTED_TOOL_CALL or MALFORMED_FUNCTION_CALL.\n"
		llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("done")}}
		exec := newFlowExecutor(codeexecutor.Settings{OptimizeDataFile: true, CodeBlockDelimiters: [][2]string{}})
		a := newPlannerAgent(t, llmagent.Config{Name: "test_agent", Instruction: "Be brief.", Model: llm, CodeExecutor: exec})

		runs := runTurns(t, a, flowOpts{}, genai.NewContentFromText("go", genai.RoleUser))

		if runs[0].err != nil {
			t.Fatalf("run error: %v", runs[0].err)
		}
		if got := instructionText(llm.recorded()[0]); got != want {
			t.Errorf("system instruction mismatch (-want +got):\n%s", cmp.Diff(want, got))
		}
	})

	noPreProcessor := []struct {
		name  string
		model string
		exec  func() codeexecutor.CodeExecutor
	}{
		{name: "nil_executor", exec: func() codeexecutor.CodeExecutor { return nil }},
		{name: "optimize_data_file_false", exec: func() codeexecutor.CodeExecutor {
			return newFlowExecutor(codeexecutor.Settings{CodeBlockDelimiters: pairs})
		}},
		{name: "built_in", model: "gemini-2.0-flash", exec: func() codeexecutor.CodeExecutor { return codeexecutor.NewBuiltInExecutor() }},
	}
	for _, tc := range noPreProcessor {
		t.Run("no_instruction_or_extraction/"+tc.name, func(t *testing.T) {
			llm := newNamedModel(tc.model, textResponse("done"))
			a := newPlannerAgent(t, llmagent.Config{Name: "test_agent", Model: llm, CodeExecutor: tc.exec()})

			runs := runTurns(t, a, flowOpts{}, userParts(csvPart("a,b\n1,2\n")))

			if runs[0].err != nil {
				t.Fatalf("run error: %v", runs[0].err)
			}
			req := llm.recorded()[0]
			if got := instructionText(req); strings.Contains(got, "CRITICAL: Code execution format") {
				t.Errorf("system instruction = %q, want no code execution format instruction", got)
			}
			want := []*genai.Content{userParts(csvPart("a,b\n1,2\n"))}
			if diff := cmp.Diff(want, req.Contents); diff != "" {
				t.Errorf("request contents mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestLLMAgent_CodeExecutionExtractsOnlyUserCSV(t *testing.T) {
	llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("done")}}
	exec := newFlowExecutor(codeexecutor.Settings{OptimizeDataFile: true}, flowStep{res: &codeexecutor.Result{Stdout: "summary\n"}})
	seed := session.NewEvent(t.Context(), "inv-old")
	seed.Author = "test_agent"
	seed.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{csvPart("model,csv")}}}
	png := &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("png_bytes")}}
	nilCSV := &genai.Part{InlineData: &genai.Blob{MIMEType: "text/csv"}}

	runs := runTurns(t, newFlowAgent(t, llm, exec), flowOpts{history: []*session.Event{seed}},
		userParts(png, nilCSV, csvPart("x,y\n1,2\n"), genai.NewPartFromText("analyze")))

	if runs[0].err != nil {
		t.Fatalf("run error: %v", runs[0].err)
	}
	contents := llm.recorded()[0].Contents
	if len(contents) < 2 {
		t.Fatalf("request has %d contents, want at least 2", len(contents))
	}
	wantHead := []*genai.Content{
		{Role: genai.RoleModel, Parts: []*genai.Part{csvPart("model,csv")}},
		userParts(
			&genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("png_bytes")}},
			&genai.Part{InlineData: &genai.Blob{MIMEType: "text/csv"}},
			&genai.Part{Text: "\nAvailable file: `data_2_3.csv`\n"},
			&genai.Part{Text: "analyze"},
		),
	}
	if diff := cmp.Diff(wantHead, contents[:2]); diff != "" {
		t.Errorf("request contents mismatch (-want +got):\n%s", diff)
	}
	calls := exec.recorded()
	if len(calls) != 1 {
		t.Fatalf("executor calls = %d, want 1", len(calls))
	}
	wantFiles := []*codeexecutor.File{{Name: "data_2_3.csv", Content: []byte("x,y\n1,2\n"), MimeType: "text/csv"}}
	if diff := cmp.Diff(wantFiles, calls[0].inputFiles); diff != "" {
		t.Errorf("executor InputFiles mismatch (-want +got):\n%s", diff)
	}
}

func TestLLMAgent_CodeExecutionDataFileExplorationEventsAndRequest(t *testing.T) {
	llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("Analysis complete.")}}
	exec := newFlowExecutor(codeexecutor.Settings{OptimizeDataFile: true, Stateful: true}, flowStep{res: &codeexecutor.Result{
		Stdout:      "Total rows: 1\n",
		OutputFiles: []*codeexecutor.File{{Name: "schema.txt", Content: []byte("col:a"), MimeType: "text/plain"}},
	}})

	runs := runTurns(t, newFlowAgent(t, llm, exec), flowOpts{}, userParts(csvPart("a\n1\n")))

	run := runs[0]
	if run.err != nil {
		t.Fatalf("run error: %v", run.err)
	}
	if len(run.events) != 3 {
		t.Fatalf("run yielded %d events, want 3", len(run.events))
	}
	for i, ev := range run.events {
		if ev.Author != "test_agent" || ev.LLMResponse.Content.Role != genai.RoleModel {
			t.Errorf("event %d Author, Role = %q, %q, want %q, %q", i, ev.Author, ev.LLMResponse.Content.Role, "test_agent", genai.RoleModel)
		}
	}
	codeParts := eventParts(t, run.events[0])
	if len(codeParts) != 2 || codeParts[1].ExecutableCode == nil {
		t.Fatalf("code event parts = %+v, want a text part and an executable code part", codeParts)
	}
	code := codeParts[1].ExecutableCode.Code
	if !strings.HasPrefix(code, "\n\nimport pandas as pd\n") || !strings.HasSuffix(code, exploreDataTail11) {
		t.Errorf("exploration code = %q, want adk-python's code for data_1_1.csv", code)
	}
	wantCodeParts := []*genai.Part{{Text: "Processing input file: `data_1_1.csv`"}, execPart(code)}
	if diff := cmp.Diff(wantCodeParts, codeParts); diff != "" {
		t.Errorf("code event parts mismatch (-want +got):\n%s", diff)
	}
	if len(run.events[0].Actions.StateDelta) != 0 {
		t.Errorf("code event StateDelta = %v, want empty", run.events[0].Actions.StateDelta)
	}
	const resultText = "Code execution result:\nTotal rows: 1\n\n\n\nSaved artifacts:\n`schema.txt`"
	wantResult := []*genai.Part{genai.NewPartFromCodeExecutionResult(genai.OutcomeOK, resultText)}
	if diff := cmp.Diff(wantResult, eventParts(t, run.events[1])); diff != "" {
		t.Errorf("result event parts mismatch (-want +got):\n%s", diff)
	}
	// adk-go's in-memory artifact service numbers versions from 1.
	if diff := cmp.Diff(map[string]int64{"schema.txt": 1}, run.events[1].Actions.ArtifactDelta); diff != "" {
		t.Errorf("result event ArtifactDelta mismatch (-want +got):\n%s", diff)
	}
	wantDelta := map[string]any{codeExecutionContextKey: map[string]any{
		"execution_session_id":  "s1",
		"processed_input_files": []string{"data_1_1.csv"},
	}}
	if diff := cmp.Diff(wantDelta, run.events[1].Actions.StateDelta); diff != "" {
		t.Errorf("result event StateDelta mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]*genai.Part{{Text: "Analysis complete."}}, eventParts(t, run.events[2])); diff != "" {
		t.Errorf("model event parts mismatch (-want +got):\n%s", diff)
	}
	if got := exec.recorded()[0].executionID; got != "s1" {
		t.Errorf("executor ExecutionID = %q, want %q", got, "s1")
	}
	want := []roleText{
		{Role: genai.RoleUser, Texts: []string{"\nAvailable file: `data_1_1.csv`\n"}},
		{Role: genai.RoleModel, Texts: []string{"Processing input file: `data_1_1.csv`", "```tool_code\n" + code + "\n```"}},
		{Role: genai.RoleUser, Texts: []string{"```tool_output\n" + resultText + "\n```"}},
	}
	if diff := cmp.Diff(want, roleTexts(llm.recorded()[0].Contents)); diff != "" {
		t.Errorf("request contents mismatch (-want +got):\n%s", diff)
	}
}

func TestLLMAgent_CodeExecutionSharedStateAcrossPreAndPostProcessor(t *testing.T) {
	file := func(name, data string) *codeexecutor.File {
		return &codeexecutor.File{Name: name, Content: []byte(data), MimeType: "text/csv"}
	}

	t.Run("single_file_per_exploration_and_all_files_in_post_processor", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{
			textResponse("```tool_code\nprint('both')\n```"),
			textResponse("done"),
			textResponse("all done"),
		}}
		ok := flowStep{res: &codeexecutor.Result{Stdout: "ok\n"}}
		exec := newFlowExecutor(codeexecutor.Settings{OptimizeDataFile: true}, ok, ok, ok, ok)

		runs := runTurns(t, newFlowAgent(t, llm, exec), flowOpts{},
			userParts(csvPart("a\n1\n"), csvPart("b\n2\n")),
			userParts(genai.NewPartFromText("follow up"), csvPart("c\n3\n")),
		)

		for i, run := range runs {
			if run.err != nil {
				t.Fatalf("run %d error: %v", i, run.err)
			}
		}
		var gotFiles [][]*codeexecutor.File
		for _, c := range exec.recorded() {
			gotFiles = append(gotFiles, c.inputFiles)
		}
		// Turn 2's user content is content 9 of its request: turn 1 left eight
		// contents (user, two code and result pairs, the model's code and its
		// result, "done").
		wantFiles := [][]*codeexecutor.File{
			{file("data_1_1.csv", "a\n1\n")},
			{file("data_1_2.csv", "b\n2\n")},
			{file("data_1_1.csv", "a\n1\n"), file("data_1_2.csv", "b\n2\n")},
			{file("data_9_2.csv", "c\n3\n")},
		}
		if diff := cmp.Diff(wantFiles, gotFiles); diff != "" {
			t.Errorf("executor InputFiles per call mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("failed_explorations_close_the_post_processor_gate", func(t *testing.T) {
		const reply = "```tool_code\nprint('should not run')\n```"
		llm := &scriptedModel{responses: []*model.LLMResponse{textResponse(reply)}}
		exec := newFlowExecutor(codeexecutor.Settings{OptimizeDataFile: true, ErrorRetryAttempts: new(2)},
			flowStep{res: &codeexecutor.Result{Stderr: "err1"}},
			flowStep{res: &codeexecutor.Result{Stderr: "err2"}},
		)

		runs := runTurns(t, newFlowAgent(t, llm, exec), flowOpts{}, userParts(csvPart("a\n1\n"), csvPart("b\n2\n")))

		run := runs[0]
		if run.err != nil {
			t.Fatalf("run error: %v", run.err)
		}
		if got := len(exec.recorded()); got != 2 {
			t.Errorf("executor calls = %d, want 2", got)
		}
		if len(run.events) != 5 {
			t.Fatalf("run yielded %d events, want 5", len(run.events))
		}
		if diff := cmp.Diff([]*genai.Part{{Text: reply}}, eventParts(t, run.events[4])); diff != "" {
			t.Errorf("last event parts mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestLLMAgent_CodeExecutionPreProcessFailures(t *testing.T) {
	t.Run("executor_error_leaves_file_unexplored", func(t *testing.T) {
		errSandbox := errors.New("sandbox down")
		llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("done")}}
		exec := newFlowExecutor(codeexecutor.Settings{OptimizeDataFile: true},
			flowStep{err: errSandbox},
			flowStep{res: &codeexecutor.Result{Stdout: "recovered\n"}},
		)

		runs := runTurns(t, newFlowAgent(t, llm, exec), flowOpts{},
			userParts(csvPart("a\n1\n")),
			genai.NewContentFromText("retry", genai.RoleUser),
		)

		if !errors.Is(runs[0].err, errSandbox) || runs[0].err.Error() != "execute code: sandbox down" {
			t.Errorf("run 1 error = %v, want %q wrapping the executor's error", runs[0].err, "execute code: sandbox down")
		}
		if len(runs[0].events) != 1 || eventParts(t, runs[0].events[0])[0].Text != "Processing input file: `data_1_1.csv`" {
			t.Errorf("run 1 yielded %d events, want only the code event", len(runs[0].events))
		}
		if runs[1].err != nil {
			t.Fatalf("run 2 error: %v", runs[1].err)
		}
		calls := exec.recorded()
		if len(calls) != 2 {
			t.Fatalf("executor calls = %d, want 2", len(calls))
		}
		if !strings.HasSuffix(calls[1].code, exploreDataTail11) {
			t.Errorf("executor call 2 code = %q, want the exploration of data_1_1.csv", calls[1].code)
		}
	})

	t.Run("missing_artifact_service", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("done")}}
		exec := newFlowExecutor(codeexecutor.Settings{OptimizeDataFile: true}, flowStep{res: &codeexecutor.Result{Stdout: "ok\n"}})

		runs := runTurns(t, newFlowAgent(t, llm, exec), flowOpts{noArtifacts: true}, userParts(csvPart("a\n1\n")))

		if runs[0].err == nil || runs[0].err.Error() != "artifact service is not initialized" {
			t.Errorf("run error = %v, want %q", runs[0].err, "artifact service is not initialized")
		}
		if len(runs[0].events) != 1 {
			t.Errorf("run yielded %d events, want only the code event", len(runs[0].events))
		}
		if got := len(llm.recorded()); got != 0 {
			t.Errorf("model calls = %d, want 0", got)
		}
	})

	t.Run("unsupported_stored_input_file_stops_exploration", func(t *testing.T) {
		llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("done")}}
		exec := newFlowExecutor(codeexecutor.Settings{OptimizeDataFile: true})
		state := map[string]any{inputFilesKey: []*codeexecutor.File{{Name: "raw.bin", Content: []byte("x"), MimeType: "application/octet-stream"}}}

		runs := runTurns(t, newFlowAgent(t, llm, exec), flowOpts{state: state}, userParts(csvPart("a\n1\n")))

		if runs[0].err != nil {
			t.Fatalf("run error: %v", runs[0].err)
		}
		if got := len(exec.recorded()); got != 0 {
			t.Errorf("executor calls = %d, want 0", got)
		}
		if len(runs[0].events) != 1 {
			t.Fatalf("run yielded %d events, want only the model event", len(runs[0].events))
		}
		want := []roleText{{Role: genai.RoleUser, Texts: []string{"\nAvailable file: `data_1_1.csv`\n"}}}
		if diff := cmp.Diff(want, roleTexts(llm.recorded()[0].Contents)); diff != "" {
			t.Errorf("request contents mismatch (-want +got):\n%s", diff)
		}
	})
}

// A stored input file whose name comes back with other bytes is replaced and
// explored again; with the same bytes it is not explored again. The session
// is created with the stored file, because the input files are written
// straight into the state and do not outlive an invocation otherwise.
func TestLLMAgent_CodeExecutionDataFileNameCollision(t *testing.T) {
	tests := []struct {
		name      string
		stored    string
		wantFiles [][]*codeexecutor.File
	}{
		{
			name:      "other_bytes_are_explored",
			stored:    "a,b\n1,2\n",
			wantFiles: [][]*codeexecutor.File{{{Name: "data_1_1.csv", Content: []byte("c,d\n3,4\n"), MimeType: "text/csv"}}},
		},
		{name: "same_bytes_are_not_explored", stored: "c,d\n3,4\n"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			llm := &scriptedModel{responses: []*model.LLMResponse{textResponse("done")}}
			exec := newFlowExecutor(codeexecutor.Settings{OptimizeDataFile: true}, flowStep{res: &codeexecutor.Result{Stdout: "ok\n"}})
			state := map[string]any{
				inputFilesKey:           []*codeexecutor.File{{Name: "data_1_1.csv", Content: []byte(tc.stored), MimeType: "text/csv"}},
				codeExecutionContextKey: map[string]any{"processed_input_files": []any{"data_1_1.csv"}},
			}

			runs := runTurns(t, newFlowAgent(t, llm, exec), flowOpts{state: state}, userParts(csvPart("c,d\n3,4\n")))

			if runs[0].err != nil {
				t.Fatalf("run error: %v", runs[0].err)
			}
			var gotFiles [][]*codeexecutor.File
			for _, c := range exec.recorded() {
				gotFiles = append(gotFiles, c.inputFiles)
				if !strings.HasSuffix(c.code, exploreDataTail11) {
					t.Errorf("executor code = %q, want the exploration of data_1_1.csv", c.code)
				}
			}
			if diff := cmp.Diff(tc.wantFiles, gotFiles); diff != "" {
				t.Errorf("executor InputFiles per call mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// After a code turn the request ends on the converted result, a user turn, so
// no continuation turn follows it (adk-python appends that turn in the model
// layer, after every request processor).
func TestLLMAgent_CodeExecutionHistoryEndsWithResult(t *testing.T) {
	llm := &scriptedModel{responses: []*model.LLMResponse{
		textResponse("```python\nprint(6 * 7)\n```"),
		textResponse("The answer is 42."),
	}}
	exec := newFlowExecutor(codeexecutor.DefaultSettings(), flowStep{res: &codeexecutor.Result{Stdout: "42\n"}})

	run := runFlow(t, newFlowAgent(t, llm, exec), flowOpts{})

	if run.err != nil {
		t.Fatalf("run error: %v", run.err)
	}
	reqs := llm.recorded()
	if len(reqs) != 2 {
		t.Fatalf("model calls = %d, want 2", len(reqs))
	}
	want := []roleText{
		{Role: genai.RoleUser, Texts: []string{"go"}},
		{Role: genai.RoleModel, Texts: []string{"```tool_code\nprint(6 * 7)\n```"}},
		{Role: genai.RoleUser, Texts: []string{"```tool_output\nCode execution result:\n42\n\n\n```"}},
	}
	if diff := cmp.Diff(want, roleTexts(reqs[1].Contents)); diff != "" {
		t.Errorf("second request contents mismatch (-want +got):\n%s", diff)
	}
}
