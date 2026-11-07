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
	"context"
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// classifierCall records one call of a ToolResultClassifier under test.
type classifierCall struct {
	callID   string
	toolName string
	args     map[string]any
	result   map[string]any
}

// recordingClassifier returns a ToolResultClassifier that records its calls
// in calls and returns verdict for every call.
func recordingClassifier(calls *[]classifierCall, mu *sync.Mutex, verdict ToolResultVerdict) ToolResultClassifier {
	return func(ctx agent.Context, t tool.Tool, args, result map[string]any) ToolResultVerdict {
		mu.Lock()
		defer mu.Unlock()
		*calls = append(*calls, classifierCall{
			callID:   ctx.FunctionCallID(),
			toolName: toolName(t),
			args:     args,
			result:   result,
		})
		return verdict
	}
}

// runClassifiedToolCall runs one tool call of a FinalResponseToolNames tool
// named submit_answer through a plugin whose classifier returns verdict, with
// the result, and returns the rows written up to INVOCATION_COMPLETED, the
// recorded classifier calls and the logger.
func runClassifiedToolCall(t *testing.T, verdict ToolResultVerdict, result map[string]any, mutate func(*Config)) ([]map[string]any, []classifierCall, *recordingLogger) {
	t.Helper()
	var (
		calls []classifierCall
		mu    sync.Mutex
	)
	logger := &recordingLogger{}
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.FinalResponseToolNames = []string{"submit_answer"}
		c.ToolResultClassifier = recordingClassifier(&calls, &mu, verdict)
		c.Logger = logger
		if mutate != nil {
			mutate(c)
		}
	})
	mCtx.functionCallID = "call-classified"
	submitTool := &fakeTool{name: "submit_answer"}
	args := map[string]any{"answer": "draft"}

	var errs callbackErrors
	errs.add(p.BeforeToolCallback()(mCtx, submitTool, args))
	errs.add(p.AfterToolCallback()(mCtx, submitTool, args, result, nil))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilInvocationEnd)
	mu.Lock()
	defer mu.Unlock()
	return rows, calls, logger
}

// requireToolErrorRow checks that rows hold the TOOL_STARTING row and one
// TOOL_ERROR row of a tool call that the classifier recorded as a failure,
// with no TOOL_COMPLETED or AGENT_RESPONSE row, and returns the TOOL_ERROR
// row.
func requireToolErrorRow(t *testing.T, rows []map[string]any) map[string]any {
	t.Helper()
	wantTypes := []string{EventToolStarting, EventToolError, EventInvocationCompleted}
	if diff := cmp.Diff(wantTypes, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}
	row := onlyRowOfType(t, rows, EventToolError)
	if got := row["status"]; got != "ERROR" {
		t.Errorf("TOOL_ERROR status = %v, want ERROR", got)
	}
	content := jsonObjectColumn(t, row, "content")
	checkJSON(t, "TOOL_ERROR content", content,
		`{"tool":"submit_answer","args":{"answer":"draft"},"tool_origin":"UNKNOWN"}`)
	if _, hasResult := content["result"]; hasResult {
		t.Errorf("TOOL_ERROR content unexpectedly has a result field: %v", content)
	}
	checkJSONAt(t, jsonObjectColumn(t, row, "attributes"), `"call-classified"`, "adk", "function_call_id")
	if got, want := row["span_id"], onlyRowOfType(t, rows, EventToolStarting)["span_id"]; got != want {
		t.Errorf("TOOL_ERROR span_id = %v, want TOOL_STARTING span_id %v", got, want)
	}
	if _, ok := row[columnLatencyMS]; !ok {
		t.Error("TOOL_ERROR row missing latency_ms")
	}
	return row
}

// requireToolCompletedRow checks that rows hold the TOOL_STARTING,
// TOOL_COMPLETED and AGENT_RESPONSE rows of a successful call of a
// FinalResponseToolNames tool and returns the TOOL_COMPLETED row.
func requireToolCompletedRow(t *testing.T, rows []map[string]any) map[string]any {
	t.Helper()
	wantTypes := []string{EventToolStarting, EventToolCompleted, EventAgentResponse, EventInvocationCompleted}
	if diff := cmp.Diff(wantTypes, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}
	row := onlyRowOfType(t, rows, EventToolCompleted)
	if got := row["status"]; got != "OK" {
		t.Errorf("TOOL_COMPLETED status = %v, want OK", got)
	}
	if got, ok := row["error_message"]; ok {
		t.Errorf("TOOL_COMPLETED error_message = %v, want none", got)
	}
	return row
}

func TestToolResultDecision_String(t *testing.T) {
	for _, tc := range []struct {
		decision ToolResultDecision
		want     string
	}{
		{ToolResultAbstain, "ToolResultAbstain"},
		{ToolResultOK, "ToolResultOK"},
		{ToolResultError, "ToolResultError"},
		{ToolResultDecision(7), "ToolResultDecision(7)"},
		{ToolResultDecision(-1), "ToolResultDecision(-1)"},
	} {
		if got := tc.decision.String(); got != tc.want {
			t.Errorf("ToolResultDecision(%d).String() = %q, want %q", int(tc.decision), got, tc.want)
		}
	}
	if (ToolResultVerdict{}).Decision != ToolResultAbstain {
		t.Error("the zero ToolResultVerdict does not abstain")
	}
}

func TestPlugin_ToolResultClassifier_Verdicts(t *testing.T) {
	t.Run("OK records TOOL_COMPLETED and the AGENT_RESPONSE of a final response tool", func(t *testing.T) {
		result := map[string]any{"status": "ERROR", "details": "handled upstream"}
		rows, calls, logger := runClassifiedToolCall(t, ToolResultVerdict{Decision: ToolResultOK}, result, nil)
		row := requireToolCompletedRow(t, rows)
		checkJSON(t, "TOOL_COMPLETED content", jsonObjectColumn(t, row, "content"),
			`{"tool":"submit_answer","result":{"status":"ERROR","details":"handled upstream"},"tool_origin":"UNKNOWN"}`)
		checkJSON(t, "AGENT_RESPONSE content", jsonObjectColumn(t, onlyRowOfType(t, rows, EventAgentResponse), "content"),
			`{"response":{"answer":"draft"}}`)
		if len(calls) != 1 {
			t.Fatalf("classifier called %d times, want 1", len(calls))
		}
		if logger.contains("ToolResultClassifier") {
			t.Errorf("unexpected classifier warning:\n%s", logger)
		}
	})

	t.Run("OK with an ErrorMessage ignores the message", func(t *testing.T) {
		rows, _, _ := runClassifiedToolCall(t, ToolResultVerdict{Decision: ToolResultOK, ErrorMessage: "ignored"}, map[string]any{"status": "ERROR"}, nil)
		requireToolCompletedRow(t, rows)
	})

	t.Run("ERROR records TOOL_ERROR with the classifier's message and no AGENT_RESPONSE", func(t *testing.T) {
		result := map[string]any{"status": "ERROR", "error_details": "quota exceeded"}
		rows, calls, logger := runClassifiedToolCall(t, ToolResultVerdict{Decision: ToolResultError, ErrorMessage: "quota exceeded"}, result, nil)
		row := requireToolErrorRow(t, rows)
		if got := row["error_message"]; got != "quota exceeded" {
			t.Errorf("error_message = %v, want %q", got, "quota exceeded")
		}
		if len(calls) != 1 {
			t.Fatalf("classifier called %d times, want 1", len(calls))
		}
		if logger.contains("ToolResultClassifier") {
			t.Errorf("unexpected classifier warning:\n%s", logger)
		}
	})

	t.Run("ERROR with an empty message writes the adk-python default message", func(t *testing.T) {
		rows, _, _ := runClassifiedToolCall(t, ToolResultVerdict{Decision: ToolResultError}, map[string]any{"status": "ERROR"}, nil)
		row := requireToolErrorRow(t, rows)
		const want = "tool_result_classifier classified the tool result as an error."
		if got := row["error_message"]; got != want {
			t.Errorf("error_message = %v, want %q", got, want)
		}
	})

	t.Run("ERROR records TOOL_ERROR for a result that reports success", func(t *testing.T) {
		rows, _, _ := runClassifiedToolCall(t, ToolResultVerdict{Decision: ToolResultError, ErrorMessage: "not ok"}, map[string]any{"ok": true}, nil)
		row := requireToolErrorRow(t, rows)
		if got := row["error_message"]; got != "not ok" {
			t.Errorf("error_message = %v, want %q", got, "not ok")
		}
	})

	t.Run("abstain records TOOL_COMPLETED without a warning", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			result map[string]any
		}{
			{name: "plain result", result: map[string]any{"answer": 42}},
			{name: "result that reports an error", result: map[string]any{"status": "ERROR", "error_details": "quota exceeded"}},
			{name: "nil result", result: nil},
		} {
			t.Run(tc.name, func(t *testing.T) {
				rows, calls, logger := runClassifiedToolCall(t, ToolResultVerdict{}, tc.result, nil)
				requireToolCompletedRow(t, rows)
				if len(calls) != 1 {
					t.Fatalf("classifier called %d times, want 1", len(calls))
				}
				if logger.contains("ToolResultClassifier") {
					t.Errorf("abstaining is not a failure, but a warning was logged:\n%s", logger)
				}
			})
		}
	})

	t.Run("the classifier receives the callback's context, tool, args and result", func(t *testing.T) {
		result := map[string]any{"answer": 42, "nested": map[string]any{"ok": true}}
		_, calls, _ := runClassifiedToolCall(t, ToolResultVerdict{}, result, nil)
		if len(calls) != 1 {
			t.Fatalf("classifier called %d times, want 1", len(calls))
		}
		call := calls[0]
		if call.callID != "call-classified" {
			t.Errorf("classifier ctx.FunctionCallID() = %q, want %q", call.callID, "call-classified")
		}
		if call.toolName != "submit_answer" {
			t.Errorf("classifier tool name = %q, want %q", call.toolName, "submit_answer")
		}
		if diff := cmp.Diff(map[string]any{"answer": "draft"}, call.args); diff != "" {
			t.Errorf("classifier args mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(result, call.result); diff != "" {
			t.Errorf("classifier result mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestPlugin_ToolResultClassifier_ErrorMessageRedaction(t *testing.T) {
	const secret = "sk-live-SECRET"
	message := "upstream failed: Bearer " + secret
	redacted := redactErrorMessage(message)
	if redacted == message || strings.Contains(redacted, secret) {
		t.Fatalf("redactErrorMessage(%q) = %q, want the credential redacted", message, redacted)
	}

	t.Run("the classifier's message is scrubbed like every other error_message", func(t *testing.T) {
		rows, _, _ := runClassifiedToolCall(t, ToolResultVerdict{Decision: ToolResultError, ErrorMessage: message}, map[string]any{"status": "ERROR"}, nil)
		row := requireToolErrorRow(t, rows)
		got, _ := row["error_message"].(string)
		if got != redacted {
			t.Errorf("error_message = %q, want %q", got, redacted)
		}
		if strings.Contains(got, secret) {
			t.Errorf("error_message %q leaks the credential", got)
		}
	})

	t.Run("DisableSensitiveKeyRedaction writes the classifier's message as is", func(t *testing.T) {
		rows, _, _ := runClassifiedToolCall(t, ToolResultVerdict{Decision: ToolResultError, ErrorMessage: message}, map[string]any{"status": "ERROR"}, func(c *Config) {
			c.DisableSensitiveKeyRedaction = true
		})
		row := requireToolErrorRow(t, rows)
		if got := row["error_message"]; got != message {
			t.Errorf("error_message = %v, want %q", got, message)
		}
	})

	t.Run("the classifier's message is truncated to MaxContentLen", func(t *testing.T) {
		long := strings.Repeat("x", 200)
		rows, _, _ := runClassifiedToolCall(t, ToolResultVerdict{Decision: ToolResultError, ErrorMessage: long}, map[string]any{"status": "ERROR"}, func(c *Config) {
			c.MaxContentLen = 50
		})
		row := requireToolErrorRow(t, rows)
		got, _ := row["error_message"].(string)
		want, _ := truncateString(long, 50)
		if got != want {
			t.Errorf("error_message = %q, want %q", got, want)
		}
		if row["is_truncated"] != true {
			t.Errorf("is_truncated = %v, want true", row["is_truncated"])
		}
	})
}

func TestPlugin_ToolResultClassifier_FailuresRecordToolCompleted(t *testing.T) {
	// secret stands for tool output that must not reach the log.
	const secret = "result-secret-7f3a"
	// A failing classifier is skipped: the call is recorded as
	// TOOL_COMPLETED whatever the result holds, as it is without a
	// classifier.
	results := []struct {
		name   string
		result map[string]any
	}{
		{name: "plain result", result: map[string]any{"detail": secret}},
		{name: "result that reports an error", result: map[string]any{"status": "ERROR", "detail": secret}},
	}

	t.Run("a classifier that panics is skipped and the warning names only the tool", func(t *testing.T) {
		for _, tc := range results {
			t.Run(tc.name, func(t *testing.T) {
				logger := &recordingLogger{}
				var calls atomic.Int32
				p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
					c.FinalResponseToolNames = []string{"submit_answer"}
					c.Logger = logger
					c.ToolResultClassifier = func(_ agent.Context, _ tool.Tool, _, result map[string]any) ToolResultVerdict {
						calls.Add(1)
						panic(fmt.Errorf("classifier saw %v: %w", result, errors.New("cause")))
					}
				})
				mCtx.functionCallID = "call-classified"
				submitTool := &fakeTool{name: "submit_answer"}
				args := map[string]any{"answer": "draft"}
				var errs callbackErrors
				errs.add(p.BeforeToolCallback()(mCtx, submitTool, args))
				errs.add(p.AfterToolCallback()(mCtx, submitTool, args, tc.result, nil))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireToolCompletedRow(t, rows)
				if got := calls.Load(); got != 1 {
					t.Errorf("classifier called %d times, want 1", got)
				}
				const wantLine = `agentanalytics: ToolResultClassifier failed for tool "submit_answer": recovered panic of type *fmt.wrapError; recording the call as TOOL_COMPLETED`
				if !logger.contains(wantLine) {
					t.Errorf("logger lacks %q; got:\n%s", wantLine, logger)
				}
				if logger.contains(secret) {
					t.Errorf("the warning leaks the tool result:\n%s", logger)
				}
				// The row was written, so the panic is not a dropped event.
				bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
				if !ok {
					t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) failed")
				}
				if stats := bqaap.DropStats(); stats.AppendFailureDrops != 0 {
					t.Errorf("AppendFailureDrops = %d, want 0", stats.AppendFailureDrops)
				}
			})
		}
	})

	t.Run("a decision that is not defined is skipped and logged", func(t *testing.T) {
		for _, tc := range results {
			t.Run(tc.name, func(t *testing.T) {
				rows, calls, logger := runClassifiedToolCall(t, ToolResultVerdict{Decision: ToolResultDecision(7), ErrorMessage: "ignored"}, tc.result, nil)
				requireToolCompletedRow(t, rows)
				if len(calls) != 1 {
					t.Errorf("classifier called %d times, want 1", len(calls))
				}
				const wantLine = `agentanalytics: ToolResultClassifier returned an unusable verdict ToolResultDecision(7) for tool "submit_answer"; recording the call as TOOL_COMPLETED`
				if !logger.contains(wantLine) {
					t.Errorf("logger lacks %q; got:\n%s", wantLine, logger)
				}
				if logger.contains(secret) || logger.contains("ignored") {
					t.Errorf("the warning leaks the result or the verdict:\n%s", logger)
				}
			})
		}
	})
}

func TestPlugin_ToolResultClassifier_Precedence(t *testing.T) {
	t.Run("a tool error that OnToolErrorCallback recorded is not classified", func(t *testing.T) {
		var (
			calls []classifierCall
			mu    sync.Mutex
		)
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"submit_answer"}
			c.ToolResultClassifier = recordingClassifier(&calls, &mu, ToolResultVerdict{Decision: ToolResultOK})
		})
		mCtx.functionCallID = "call-classified"
		submitTool := &fakeTool{name: "submit_answer"}
		args := map[string]any{"answer": "draft"}
		var errs callbackErrors
		errs.add(p.BeforeToolCallback()(mCtx, submitTool, args))
		errs.add(p.OnToolErrorCallback()(mCtx, submitTool, args, errors.New("boom")))
		// A later plugin recovered the error into a result.
		errs.add(p.AfterToolCallback()(mCtx, submitTool, args, map[string]any{"answer": "recovered"}, nil))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		// The TOOL_ERROR row that OnToolErrorCallback wrote is the call's only
		// one: the classifier is not asked about the recovered result, so it
		// cannot add a second.
		row := onlyRowOfType(t, rows, EventToolError)
		if got := row["error_message"]; got != "boom" {
			t.Errorf("error_message = %v, want %q", got, "boom")
		}
		mu.Lock()
		defer mu.Unlock()
		if len(calls) != 0 {
			t.Errorf("classifier called %d times for a recorded tool error, want 0", len(calls))
		}
	})

	t.Run("an error that reaches AfterToolCallback unrecovered is not classified", func(t *testing.T) {
		var (
			calls []classifierCall
			mu    sync.Mutex
		)
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"submit_answer"}
			c.ToolResultClassifier = recordingClassifier(&calls, &mu, ToolResultVerdict{Decision: ToolResultOK})
		})
		mCtx.functionCallID = "call-classified"
		submitTool := &fakeTool{name: "submit_answer"}
		args := map[string]any{"answer": "draft"}
		var errs callbackErrors
		errs.add(p.BeforeToolCallback()(mCtx, submitTool, args))
		errs.add(p.AfterToolCallback()(mCtx, submitTool, args, nil, errors.New("passed through")))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		row := requireToolErrorRow(t, rows)
		if got := row["error_message"]; got != "passed through" {
			t.Errorf("error_message = %v, want %q", got, "passed through")
		}
		mu.Lock()
		defer mu.Unlock()
		if len(calls) != 0 {
			t.Errorf("classifier called %d times for an unrecovered error, want 0", len(calls))
		}
	})

	t.Run("a confirmation outcome reaches the classifier as the result ADK sends the model", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			verdict ToolResultVerdict
		}{
			{name: "abstain completes the call", verdict: ToolResultVerdict{}},
			{name: "ERROR records TOOL_ERROR", verdict: ToolResultVerdict{Decision: ToolResultError, ErrorMessage: "needs approval"}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var (
					calls []classifierCall
					mu    sync.Mutex
				)
				p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
					c.FinalResponseToolNames = []string{"deploy"}
					c.ToolResultClassifier = recordingClassifier(&calls, &mu, tc.verdict)
				})
				mCtx.functionCallID = "call-confirm"
				deployTool := &fakeTool{name: "deploy"}
				args := map[string]any{"region": "us-central1"}
				toolErr := fmt.Errorf("error tool %q %w", "deploy", tool.ErrConfirmationRequired)
				var errs callbackErrors
				errs.add(p.BeforeToolCallback()(mCtx, deployTool, args))
				errs.add(p.OnToolErrorCallback()(mCtx, deployTool, args, toolErr))
				errs.add(p.AfterToolCallback()(mCtx, deployTool, args, nil, toolErr))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				mu.Lock()
				gotCalls := append([]classifierCall(nil), calls...)
				mu.Unlock()
				if len(gotCalls) != 1 {
					t.Fatalf("classifier called %d times, want 1", len(gotCalls))
				}
				wantResult := map[string]any{"error": toolErr.Error()}
				if diff := cmp.Diff(wantResult, gotCalls[0].result); diff != "" {
					t.Errorf("classifier result mismatch (-want +got):\n%s", diff)
				}
				if tc.verdict.Decision == ToolResultError {
					wantTypes := []string{EventToolStarting, EventToolError, EventInvocationCompleted}
					if diff := cmp.Diff(wantTypes, eventTypes(rows)); diff != "" {
						t.Fatalf("event types mismatch (-want +got):\n%s", diff)
					}
					if got := onlyRowOfType(t, rows, EventToolError)["error_message"]; got != "needs approval" {
						t.Errorf("error_message = %v, want %q", got, "needs approval")
					}
					return
				}
				// A call that waits for confirmation has not run, so the
				// FinalResponseToolNames AGENT_RESPONSE row waits too.
				wantTypes := []string{EventToolStarting, EventToolCompleted, EventInvocationCompleted}
				if diff := cmp.Diff(wantTypes, eventTypes(rows)); diff != "" {
					t.Fatalf("event types mismatch (-want +got):\n%s", diff)
				}
				checkJSON(t, "TOOL_COMPLETED content", jsonObjectColumn(t, onlyRowOfType(t, rows, EventToolCompleted), "content"),
					`{"tool":"deploy","result":{"error":"error tool \"deploy\" requires confirmation, please approve or reject"},"tool_origin":"UNKNOWN"}`)
			})
		}
	})
}

func TestPlugin_ToolResultClassifier_Runner(t *testing.T) {
	t.Run("ERROR verdict on a FinalResponseToolNames tool suppresses its AGENT_RESPONSE", func(t *testing.T) {
		type toolArgs struct {
			Query string `json:"query"`
		}
		fnTool, err := functiontool.New(
			functiontool.Config{Name: "final_lookup", Description: "Looks up final answer"},
			func(_ agent.Context, in toolArgs) (map[string]any, error) {
				return map[string]any{"status": "ERROR", "error_details": "no such " + in.Query}, nil
			},
		)
		if err != nil {
			t.Fatalf("functiontool.New: %v", err)
		}
		p, requests, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"final_lookup"}
			c.ToolResultClassifier = func(_ agent.Context, _ tool.Tool, _, result map[string]any) ToolResultVerdict {
				if result["status"] == "ERROR" {
					details, _ := result["error_details"].(string)
					return ToolResultVerdict{Decision: ToolResultError, ErrorMessage: details}
				}
				return ToolResultVerdict{}
			}
		})
		bot, err := llmagent.New(llmagent.Config{
			Name:  "bot",
			Tools: []tool.Tool{fnTool},
			Model: &scriptedLLM{
				responses: []*model.LLMResponse{
					{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{
									ID:   "call-fn-1",
									Name: "final_lookup",
									Args: map[string]any{"query": "status"},
								},
							}},
						},
					},
					{Content: textContent("model", "handled failure"), TurnComplete: true},
				},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New: %v", err)
		}
		for _, runErr := range newTestRunner(t, bot, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error: %v", runErr)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		if got := len(rowsOfType(rows, EventToolCompleted)); got != 0 {
			t.Fatalf("got %d TOOL_COMPLETED rows, want 0; events: %v", got, eventTypes(rows))
		}
		errRow := onlyRowOfType(t, rows, EventToolError)
		if got := errRow["error_message"]; got != "no such status" {
			t.Errorf("error_message = %v, want %q", got, "no such status")
		}
		// Only the final model turn text emits AGENT_RESPONSE, not final_lookup.
		agentResp := onlyRowOfType(t, rows, EventAgentResponse)
		if _, hasSourceTool := lookup(jsonObjectColumn(t, agentResp, "attributes"), "source_tool"); hasSourceTool {
			t.Errorf("AGENT_RESPONSE unexpectedly emitted for the failed FinalResponseToolNames tool: %v", agentResp["attributes"])
		}
	})

	t.Run("ERROR verdict on finish_task does not note the task as finished", func(t *testing.T) {
		var runCount atomic.Int32
		p, requests, _ := setupTestPluginWithConfig(t, func(c *Config) {
			// Only the application knows that {"status": "ERROR"} is a failure.
			c.ToolResultClassifier = func(_ agent.Context, _ tool.Tool, _, result map[string]any) ToolResultVerdict {
				if result["status"] == "ERROR" {
					return ToolResultVerdict{Decision: ToolResultError}
				}
				return ToolResultVerdict{}
			}
		})
		// An earlier plugin turns Run 2's first finish_task answer into an
		// application error envelope, which is not a task completion, so the
		// task agent calls the model again.
		rewritePlugin, err := baseplugin.New(baseplugin.Config{
			Name: "rewrite_finish_task_result",
			AfterToolCallback: func(ctx agent.Context, tl tool.Tool, _, result map[string]any, err error) (map[string]any, error) {
				if err == nil && tl != nil && tl.Name() == "finish_task" && ctx.FunctionCallID() == "call-shared" && runCount.Load() == 2 {
					delete(result, "result")
					result["status"] = "ERROR"
				}
				return nil, nil
			},
		})
		if err != nil {
			t.Fatalf("baseplugin.New: %v", err)
		}

		// Run 1 calls finish_task("call-shared") and succeeds.
		// Run 2 (started while Run 1's finish_task response event is still held
		// back, so Run 2 sits above Run 1 on agentStack) first calls
		// finish_task("call-shared"), which rewritePlugin turns into
		// {"status": "ERROR"} and the classifier records as a failure, and then
		// calls finish_task("call-second") which succeeds. If Run 2's failed
		// "call-shared" wrongly called noteTaskFinished, both activations would
		// claim "call-shared" and finishedTaskLocked would fall back to the
		// top-of-stack activation (Run 2) instead of closing Run 1's span on
		// Run 1's event.
		var turn atomic.Int32
		taskModel := &funcStreamLLM{
			fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				n := turn.Add(1)
				callID := "call-shared"
				if n == 3 {
					callID = "call-second"
				}
				yield(&model.LLMResponse{
					Content: &genai.Content{
						Role: "model",
						Parts: []*genai.Part{{
							FunctionCall: &genai.FunctionCall{
								ID:   callID,
								Name: "finish_task",
								Args: map[string]any{"result": "done"},
							},
						}},
					},
				}, nil)
			},
		}
		taskAgent, err := llmagent.New(llmagent.Config{
			Name:  "task",
			Mode:  llmagent.ModeTask,
			Model: taskModel,
		})
		if err != nil {
			t.Fatalf("llmagent.New(task): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name: "root",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var held []*session.Event
					nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
					for i := 0; i < 2; i++ {
						runCount.Add(1)
						for ev, err := range llmagent.RunLLMAgentAsNode(taskAgent, nodeCtx, "work") {
							if err != nil {
								yield(nil, err)
								return
							}
							if hasTaskCompletedResponse(ev.Content) {
								held = append(held, ev)
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
					}
					for _, ev := range held {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, rewritePlugin, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error: %v", runErr)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		toolErrRow := onlyRowOfType(t, rows, EventToolError)
		if got := toolErrRow["error_message"]; got != classifiedToolErrorMessage {
			t.Errorf("TOOL_ERROR error_message = %v, want %q", got, classifiedToolErrorMessage)
		}
		if got := len(rowsOfType(rows, EventToolCompleted)); got != 2 {
			t.Fatalf("got %d TOOL_COMPLETED rows, want 2; events: %v", got, eventTypes(rows))
		}
		starts := agentRowsOfType(rows, EventAgentStarting, "task")
		ends := agentRowsOfType(rows, EventAgentCompleted, "task")
		if len(starts) != 2 || len(ends) != 2 {
			t.Fatalf("task rows: %d AGENT_STARTING and %d AGENT_COMPLETED, want 2 of each", len(starts), len(ends))
		}
		if got, want := ends[0]["span_id"], starts[0]["span_id"]; got != want {
			t.Errorf("first AGENT_COMPLETED span_id = %v, want Run 1 span_id %v (Run 2 span_id = %v)", got, want, starts[1]["span_id"])
		}
		if got, want := ends[1]["span_id"], starts[1]["span_id"]; got != want {
			t.Errorf("second AGENT_COMPLETED span_id = %v, want Run 2 span_id %v", got, want)
		}
	})

	t.Run("ERROR verdict on runtime-successful finish_task preserves activation attribution", func(t *testing.T) {
		p, requests, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.ToolResultClassifier = func(_ agent.Context, tl tool.Tool, args, result map[string]any) ToolResultVerdict {
				if tl != nil && tl.Name() == finishTaskToolName && args["result"] == "bad" && result["result"] == "Task completed." {
					return ToolResultVerdict{Decision: ToolResultError}
				}
				return ToolResultVerdict{}
			}
		})

		var turn atomic.Int32
		taskModel := &funcStreamLLM{
			fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				callID := "call-first"
				resArg := "bad"
				if turn.Add(1) == 2 {
					callID = "call-second"
					resArg = "good"
				}
				yield(&model.LLMResponse{
					Content: &genai.Content{
						Role: "model",
						Parts: []*genai.Part{{
							FunctionCall: &genai.FunctionCall{
								ID:   callID,
								Name: finishTaskToolName,
								Args: map[string]any{"result": resArg},
							},
						}},
					},
				}, nil)
			},
		}
		taskAgent, err := llmagent.New(llmagent.Config{
			Name:  "task",
			Mode:  llmagent.ModeTask,
			Model: taskModel,
		})
		if err != nil {
			t.Fatalf("llmagent.New(task): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name: "root",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var held []*session.Event
					nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
					for i := 0; i < 2; i++ {
						for ev, runErr := range llmagent.RunLLMAgentAsNode(taskAgent, nodeCtx, "work") {
							if runErr != nil {
								yield(nil, runErr)
								return
							}
							if hasTaskCompletedResponse(ev.Content) {
								held = append(held, ev)
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
					}
					for _, ev := range held {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error: %v", runErr)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		toolErrRow := onlyRowOfType(t, rows, EventToolError)
		if got := toolErrRow["error_message"]; got != classifiedToolErrorMessage {
			t.Errorf("TOOL_ERROR error_message = %v, want %q", got, classifiedToolErrorMessage)
		}
		if got := len(rowsOfType(rows, EventToolCompleted)); got != 1 {
			t.Fatalf("got %d TOOL_COMPLETED rows, want 1; events: %v", got, eventTypes(rows))
		}
		starts := agentRowsOfType(rows, EventAgentStarting, "task")
		ends := agentRowsOfType(rows, EventAgentCompleted, "task")
		if len(starts) != 2 || len(ends) != 2 {
			t.Fatalf("task rows: %d AGENT_STARTING and %d AGENT_COMPLETED, want 2 of each", len(starts), len(ends))
		}
		if got, want := ends[0]["span_id"], starts[0]["span_id"]; got != want {
			t.Errorf("first AGENT_COMPLETED span_id = %v, want Run 1 span_id %v (Run 2 span_id = %v)", got, want, starts[1]["span_id"])
		}
		if got, want := ends[1]["span_id"], starts[1]["span_id"]; got != want {
			t.Errorf("second AGENT_COMPLETED span_id = %v, want Run 2 span_id %v (Run 1 span_id = %v)", got, want, starts[0]["span_id"])
		}
	})
}

// TestPlugin_ToolResultClassifier_ConcurrentCalls classifies the parallel
// tool calls of one agent, as ADK runs the function calls of a model
// response, each in a goroutine of its own. Under -race, the test also checks
// that the classifier path shares no unsynchronized state.
func TestPlugin_ToolResultClassifier_ConcurrentCalls(t *testing.T) {
	const calls = 16
	var classified atomic.Int32
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.ToolResultClassifier = func(ctx agent.Context, _ tool.Tool, _, result map[string]any) ToolResultVerdict {
			classified.Add(1)
			if result["fail"] == true {
				return ToolResultVerdict{Decision: ToolResultError, ErrorMessage: "failed " + ctx.FunctionCallID()}
			}
			return ToolResultVerdict{Decision: ToolResultOK}
		}
	})
	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(mCtx))
	errs.check(t)
	var wg sync.WaitGroup
	errCh := make(chan error, 2*calls)
	for i := 0; i < calls; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			callCtx := withFunctionCall(mCtx, fmt.Sprintf("call-%02d", i))
			tl := &fakeTool{name: fmt.Sprintf("tool%02d", i)}
			args := map[string]any{"i": i}
			if _, err := p.BeforeToolCallback()(callCtx, tl, args); err != nil {
				errCh <- err
			}
			// Even calls fail; odd calls succeed.
			result := map[string]any{"fail": i%2 == 0}
			if _, err := p.AfterToolCallback()(callCtx, tl, args, result, nil); err != nil {
				errCh <- err
			}
		}(i)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Errorf("callback error: %v", err)
	}
	errs.add(p.AfterAgentCallback()(mCtx))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilInvocationEnd)
	if got := classified.Load(); got != calls {
		t.Errorf("classifier called %d times, want %d", got, calls)
	}
	errRows := rowsOfType(rows, EventToolError)
	okRows := rowsOfType(rows, EventToolCompleted)
	if len(errRows) != calls/2 || len(okRows) != calls/2 {
		t.Fatalf("got %d TOOL_ERROR and %d TOOL_COMPLETED rows, want %d of each; events: %v", len(errRows), len(okRows), calls/2, eventTypes(rows))
	}
	for _, row := range errRows {
		if got, want := row["error_message"], "failed "+functionCallIDOf(t, row); got != want {
			t.Errorf("TOOL_ERROR error_message = %v, want %q", got, want)
		}
	}
}
