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

package llminternal

import (
	"iter"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/codeexecutor"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// wantDataFileCode21 is the exploration code adk-python v2.12.0 writes for
// data_2_1.csv (_get_data_file_preprocessing_code), one source line per
// literal.
const wantDataFileCode21 = "" +
	"\n" +
	"\n" +
	"import pandas as pd\n" +
	"\n" +
	"def crop(s: str, max_chars: int = 64) -> str:\n" +
	"  \"\"\"Crops a string to max_chars characters.\"\"\"\n" +
	"  if len(s) <= max_chars:\n" +
	"    return s\n" +
	"  if max_chars >= 3:\n" +
	"    return s[:max_chars - 3] + '...'\n" +
	"  return s[:max_chars]\n" +
	"\n" +
	"def explore_df(df: pd.DataFrame) -> None:\n" +
	"  \"\"\"Prints some information about a pandas DataFrame.\"\"\"\n" +
	"\n" +
	"  with pd.option_context(\n" +
	"      'display.max_columns', None, 'display.expand_frame_repr', False\n" +
	"  ):\n" +
	"    # Print the column names to never encounter KeyError when selecting one.\n" +
	"    df_dtypes = df.dtypes\n" +
	"\n" +
	"    # Obtain information about data types and missing values.\n" +
	"    df_nulls = (len(df) - df.isnull().sum()).apply(\n" +
	"        lambda x: f'{x} / {df.shape[0]} non-null'\n" +
	"    )\n" +
	"\n" +
	"    # Explore unique total values in columns using `.unique()`.\n" +
	"    df_unique_count = df.apply(lambda x: len(x.unique()))\n" +
	"\n" +
	"    # Explore unique values in columns using `.unique()`.\n" +
	"    df_unique = df.apply(lambda x: crop(str(list(x.unique()))))\n" +
	"\n" +
	"    df_info = pd.concat(\n" +
	"        (\n" +
	"            df_dtypes.rename('Dtype'),\n" +
	"            df_nulls.rename('Non-Null Count'),\n" +
	"            df_unique_count.rename('Unique Values Count'),\n" +
	"            df_unique.rename('Unique Values'),\n" +
	"        ),\n" +
	"        axis=1,\n" +
	"    )\n" +
	"    df_info.index.name = 'Columns'\n" +
	"    print(f\"\"\"Total rows: {df.shape[0]}\n" +
	"Total columns: {df.shape[1]}\n" +
	"\n" +
	"{df_info}\"\"\")\n" +
	"\n" +
	"\n" +
	"# Load the dataframe.\n" +
	"data_2_1 = pd.read_csv('data_2_1.csv')\n" +
	"\n" +
	"# Use `explore_df` to guide my analysis.\n" +
	"explore_df(data_2_1)\n"

func TestDataFilePreprocessingCode(t *testing.T) {
	code, ok := dataFilePreprocessingCode(&codeexecutor.File{Name: "data_2_1.csv", MimeType: "text/csv"})
	if !ok {
		t.Fatalf("dataFilePreprocessingCode(data_2_1.csv) ok = false, want true")
	}
	if code != wantDataFileCode21 {
		t.Errorf("dataFilePreprocessingCode(data_2_1.csv) mismatch (-want +got):\n%s", cmp.Diff(wantDataFileCode21, code))
	}

	// The variable name and the loader of each name are what adk-python
	// v2.12.0 gives (os.path.splitext, re.sub and repr).
	tests := []struct {
		name       string
		file       string
		wantVar    string
		wantLoader string
	}{
		{name: "digit_prefix", file: "123file.csv", wantVar: "_123file", wantLoader: `pd.read_csv('123file.csv')`},
		{name: "punctuation_only_stem", file: "---.csv", wantVar: "___", wantLoader: `pd.read_csv('---.csv')`},
		{name: "leading_dot_is_stem", file: ".csv", wantVar: "_csv", wantLoader: `pd.read_csv('.csv')`},
		{name: "leading_dots_without_extension", file: "..x", wantVar: "__x", wantLoader: `pd.read_csv('..x')`},
		{name: "dot_after_separator", file: "dir/.csv", wantVar: "dir__csv", wantLoader: `pd.read_csv('dir/.csv')`},
		{name: "last_extension_only", file: "a.b.csv", wantVar: "a_b", wantLoader: `pd.read_csv('a.b.csv')`},
		{name: "empty", file: "", wantVar: "_data", wantLoader: `pd.read_csv('')`},
		{name: "injection", file: "'); print('PWNED')#.csv", wantVar: "____print__PWNED___", wantLoader: `pd.read_csv("'); print('PWNED')#.csv")`},
		{name: "both_quotes_and_controls", file: "a'b\"c\n\t\x00.csv", wantVar: "a_b_c___", wantLoader: `pd.read_csv('a\'b"c\n\t\x00.csv')`},
		{name: "double_quote_only", file: `a"b.csv`, wantVar: "a_b", wantLoader: `pd.read_csv('a"b.csv')`},
		{name: "backslash", file: `back\slash.csv`, wantVar: "back_slash", wantLoader: `pd.read_csv('back\\slash.csv')`},
		{name: "delete_char", file: "q\x7f.csv", wantVar: "q_", wantLoader: `pd.read_csv('q\x7f.csv')`},
		{name: "non_printable_below_0x100", file: "a\u00a0b.csv", wantVar: "a_b", wantLoader: `pd.read_csv('a\xa0b.csv')`},
		{name: "format_char_below_0x100", file: "a\u00adb.csv", wantVar: "a_b", wantLoader: `pd.read_csv('a\xadb.csv')`},
		{name: "c1_control", file: "\u0085.csv", wantVar: "_", wantLoader: `pd.read_csv('\x85.csv')`},
		{name: "line_separator", file: "x\u2028y.csv", wantVar: "x_y", wantLoader: `pd.read_csv('x\u2028y.csv')`},
		{name: "printable_non_ascii", file: "é.csv", wantVar: "_", wantLoader: `pd.read_csv('é.csv')`},
		{name: "astral", file: "\U0001F600.csv", wantVar: "_", wantLoader: "pd.read_csv('\U0001F600.csv')"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			code, ok := dataFilePreprocessingCode(&codeexecutor.File{Name: tc.file, MimeType: "text/csv"})
			if !ok {
				t.Fatalf("ok = false, want true")
			}
			wantTail := "\n\n# Load the dataframe.\n" + tc.wantVar + " = " + tc.wantLoader + "\n\n# Use `explore_df` to guide my analysis.\nexplore_df(" + tc.wantVar + ")\n"
			if !strings.HasSuffix(code, wantTail) {
				t.Errorf("code ends %q, want %q", code[strings.LastIndex(code, "# Load"):], wantTail[2:])
			}
		})
	}

	t.Run("no_loader_for_mime_type", func(t *testing.T) {
		code, ok := dataFilePreprocessingCode(&codeexecutor.File{Name: "image.png", MimeType: "image/png"})
		if ok || code != "" {
			t.Errorf("dataFilePreprocessingCode(image/png) = %q, %v, want \"\", false", code, ok)
		}
	})
}

func TestCodeExecutorContext_ProcessedFileNames(t *testing.T) {
	t.Run("json_round_trip_shape_and_no_duplicates", func(t *testing.T) {
		state := newTestState(t, map[string]any{codeExecutionContextKey: map[string]any{
			"processed_input_files": []any{"data_1_1.csv", "data_1_2.csv"},
		}})
		ec, err := newCodeExecutorContext(state)
		if err != nil {
			t.Fatal(err)
		}

		got, err := ec.processedFileNames()
		if err != nil {
			t.Fatalf("processedFileNames: %v", err)
		}
		if diff := cmp.Diff([]string{"data_1_1.csv", "data_1_2.csv"}, got); diff != "" {
			t.Errorf("processedFileNames mismatch (-want +got):\n%s", diff)
		}
		for _, name := range []string{"data_1_2.csv", "data_2_1.csv"} {
			if err := ec.addProcessedFileName(name); err != nil {
				t.Fatalf("addProcessedFileName(%q): %v", name, err)
			}
		}
		got, err = ec.processedFileNames()
		if err != nil {
			t.Fatalf("processedFileNames: %v", err)
		}
		if diff := cmp.Diff([]string{"data_1_1.csv", "data_1_2.csv", "data_2_1.csv"}, got); diff != "" {
			t.Errorf("processedFileNames after add mismatch (-want +got):\n%s", diff)
		}
	})

	for _, tc := range []struct {
		name     string
		value    any
		wantType string
	}{
		{name: "not_a_list", value: 123, wantType: "int"},
		{name: "list_of_non_strings", value: []any{"secret_name.csv", 123}, wantType: "int"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state := newTestState(t, map[string]any{codeExecutionContextKey: map[string]any{"processed_input_files": tc.value}})
			ec, err := newCodeExecutorContext(state)
			if err != nil {
				t.Fatal(err)
			}

			_, err = ec.processedFileNames()

			if err == nil || !strings.Contains(err.Error(), codeExecutionContextKey) || !strings.Contains(err.Error(), tc.wantType) {
				t.Fatalf("processedFileNames error = %v, want one naming %s and %s", err, codeExecutionContextKey, tc.wantType)
			}
			if strings.Contains(err.Error(), "secret_name") {
				t.Errorf("processedFileNames error names a stored value")
			}
		})
	}
}

// newTestState returns the state of a new in-memory session created with
// initial.
func newTestState(t *testing.T, initial map[string]any) session.State {
	t.Helper()
	created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s1", State: initial})
	if err != nil {
		t.Fatal(err)
	}
	return created.Session.State()
}

// newContinuationContext returns the invocation context of an LLM agent named
// test_agent whose session holds one event of its own with each of contents.
func newContinuationContext(t *testing.T, contents ...*genai.Content) agent.InvocationContext {
	t.Helper()
	svc := session.InMemoryService()
	created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range contents {
		ev := session.NewEvent(t.Context(), "inv-old")
		ev.Author = "test_agent"
		ev.LLMResponse = model.LLMResponse{Content: c}
		if err := svc.AppendEvent(t.Context(), created.Session, ev); err != nil {
			t.Fatal(err)
		}
	}
	base, err := agent.New(agent.Config{Name: "test_agent"})
	if err != nil {
		t.Fatal(err)
	}
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:   &mockLLMAgent{Agent: base, s: &State{}},
		Session: created.Session,
	})
}

// The continuation turn follows a last content of any role but user
// (adk-python models/base_llm.py:260, _maybe_append_user_content). Each want
// is the contents adk-python v2.12.0 sends for the same case.
func TestDefaultRequestProcessors_ContinuationTurnAfterModelTurn(t *testing.T) {
	const continuation = "Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed."
	tests := []struct {
		name string
		role genai.Role
		want []*genai.Content
	}{
		{
			name: "model",
			role: genai.RoleModel,
			want: []*genai.Content{
				genai.NewContentFromText("I am thinking", genai.RoleModel),
				genai.NewContentFromText(continuation, genai.RoleUser),
			},
		},
		{
			name: "function",
			role: "function",
			want: []*genai.Content{
				genai.NewContentFromText("I am thinking", "function"),
				genai.NewContentFromText(continuation, genai.RoleUser),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := newContinuationContext(t, genai.NewContentFromText("I am thinking", tc.role))
			req := &model.LLMRequest{}

			for _, err := range (&Flow{RequestProcessors: DefaultRequestProcessors}).preprocess(ctx, req) {
				if err != nil {
					t.Fatalf("preprocess: %v", err)
				}
			}

			if diff := cmp.Diff(tc.want, req.Contents); diff != "" {
				t.Errorf("request contents mismatch (-want +got):\n%s", diff)
			}
		})
	}

	// No stored event gives a content without a role: the history skips it,
	// as adk-python's does (flows/llm_flows/context/_contents.py:279). So
	// this case starts from a request that already ends on one and runs the
	// continuation step alone.
	t.Run("empty role", func(t *testing.T) {
		ctx := newContinuationContext(t)
		// genai.NewContentFromText turns an empty role into user, so the
		// contents are written out.
		req := &model.LLMRequest{Contents: []*genai.Content{{Parts: []*genai.Part{{Text: "I am thinking"}}}}}

		processors := []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{continuationTurnRequestProcessor}
		for _, err := range (&Flow{RequestProcessors: processors}).preprocess(ctx, req) {
			if err != nil {
				t.Fatalf("preprocess: %v", err)
			}
		}

		want := []*genai.Content{
			{Parts: []*genai.Part{{Text: "I am thinking"}}},
			genai.NewContentFromText(continuation, genai.RoleUser),
		}
		if diff := cmp.Diff(want, req.Contents); diff != "" {
			t.Errorf("request contents mismatch (-want +got):\n%s", diff)
		}
	})
}

// builtInWrapper embeds *codeexecutor.BuiltInExecutor, so it is a
// codeexecutor.BuiltIn of a user type.
type builtInWrapper struct {
	*codeexecutor.BuiltInExecutor
}

func TestResolveCodeExecutor(t *testing.T) {
	custom := &countingCodeExecutor{}
	builtIn := codeexecutor.NewBuiltInExecutor()
	wrapped := builtInWrapper{codeexecutor.NewBuiltInExecutor()}
	cfc := &agent.RunConfig{SupportCFC: true}
	type agentKind int
	const (
		root agentKind = iota
		child
		notLLM
	)
	tests := []struct {
		name        string
		kind        agentKind
		exec        codeexecutor.CodeExecutor
		runConfig   *agent.RunConfig
		want        codeexecutor.CodeExecutor
		wantBuiltIn bool
	}{
		{name: "not_an_llm_agent", kind: notLLM, runConfig: cfc, want: nil},
		{name: "no_cfc_nil", kind: root, runConfig: &agent.RunConfig{}, want: nil},
		{name: "no_cfc_custom", kind: root, exec: custom, runConfig: &agent.RunConfig{}, want: custom},
		{name: "no_cfc_built_in", kind: root, exec: builtIn, runConfig: &agent.RunConfig{}, want: builtIn},
		{name: "no_run_config", kind: root, exec: custom, want: custom},
		{name: "cfc_root_nil", kind: root, runConfig: cfc, wantBuiltIn: true},
		{name: "cfc_root_custom", kind: root, exec: custom, runConfig: cfc, wantBuiltIn: true},
		{name: "cfc_root_built_in_kept", kind: root, exec: builtIn, runConfig: cfc, want: builtIn},
		{name: "cfc_root_user_built_in_kept", kind: root, exec: wrapped, runConfig: cfc, want: wrapped},
		{name: "cfc_child_nil", kind: child, runConfig: cfc, want: nil},
		{name: "cfc_child_custom", kind: child, exec: custom, runConfig: cfc, want: custom},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var a agent.Agent
			switch tc.kind {
			case notLLM:
				a = newCustomAgent(t, "custom")
			case root:
				a = newLinkedLLMAgent(t, "test_agent", nil)
			case child:
				a = newLinkedLLMAgent(t, "test_agent", nil)
				// The parent links a; the invocation has no parent map, as
				// when a runner has a as its root.
				newCustomAgent(t, "parent", a)
			}
			if tc.kind != notLLM {
				a.(*mockLLMAgent).s.CodeExecutor = tc.exec
			}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a, RunConfig: tc.runConfig})

			got := resolveCodeExecutor(ctx)

			if tc.wantBuiltIn {
				if _, ok := got.(*codeexecutor.BuiltInExecutor); !ok {
					t.Errorf("resolveCodeExecutor = %T, want *codeexecutor.BuiltInExecutor", got)
				}
			} else if got != tc.want {
				t.Errorf("resolveCodeExecutor = %v, want %v", got, tc.want)
			}
			if tc.kind != notLLM && getCodeExecutor(ctx) != tc.exec {
				t.Errorf("agent's code executor = %v after resolveCodeExecutor, want %v", getCodeExecutor(ctx), tc.exec)
			}
		})
	}
}

// Under SupportCFC both code execution processors of a root agent with an
// executor that is not built in use the built-in executor: the request gets
// the model's code-execution tool and the response's image is saved, while the
// agent's executor runs nothing.
func TestCodeExecutionProcessorsUseResolvedExecutor(t *testing.T) {
	for _, supportCFC := range []bool{true, false} {
		t.Run(map[bool]string{true: "cfc", false: "no_cfc"}[supportCFC], func(t *testing.T) {
			exec := &countingCodeExecutor{}
			base, err := agent.New(agent.Config{Name: "test_agent"})
			if err != nil {
				t.Fatal(err)
			}
			created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s1"})
			if err != nil {
				t.Fatal(err)
			}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Agent:     &mockLLMAgent{Agent: base, s: &State{CodeExecutor: exec}},
				Session:   created.Session,
				Artifacts: &artifactinternal.Artifacts{Service: artifact.InMemoryService(), AppName: "app", UserID: "user", SessionID: "s1"},
				RunConfig: &agent.RunConfig{SupportCFC: supportCFC},
			})
			req := &model.LLMRequest{Model: "gemini-2.0-flash", Config: &genai.GenerateContentConfig{}}

			for _, err := range codeExecutionRequestProcessor(ctx, req, &Flow{}) {
				if err != nil {
					t.Fatalf("codeExecutionRequestProcessor: %v", err)
				}
			}
			resp := &model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
				{Text: "```python\nx = 1\n```"},
				{InlineData: &genai.Blob{Data: []byte("png"), MIMEType: "image/png", DisplayName: "plot.png"}},
			}}}
			var deltas []map[string]int64
			for ev, err := range codeExecutionResponseProcessor(ctx, req, resp) {
				if err != nil {
					break
				}
				deltas = append(deltas, ev.Actions.ArtifactDelta)
			}

			var wantTools []*genai.Tool
			wantDeltas := []map[string]int64{{}}
			var wantCalls int32 = 1
			if supportCFC {
				wantTools = []*genai.Tool{{CodeExecution: &genai.ToolCodeExecution{}}}
				wantDeltas = []map[string]int64{{"plot.png": 1}}
				wantCalls = 0
			}
			if diff := cmp.Diff(wantTools, req.Config.Tools); diff != "" {
				t.Errorf("request tools mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(wantDeltas, deltas[:1]); diff != "" {
				t.Errorf("first response processor event ArtifactDelta mismatch (-want +got):\n%s", diff)
			}
			if got := exec.calls.Load(); got != wantCalls {
				t.Errorf("agent's executor calls = %d, want %d", got, wantCalls)
			}
		})
	}
}
