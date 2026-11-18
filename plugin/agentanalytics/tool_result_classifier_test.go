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
	"slices"
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
	"google.golang.org/adk/v2/plugin/retryandreflect"
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

	t.Run("abstain leaves a top-level isError or is_error of true to the built-in rule", func(t *testing.T) {
		// As in adk-python, which records these two as tool errors when its
		// classifier abstains.
		for _, tc := range []struct {
			name   string
			result map[string]any
		}{
			{name: "result with a top-level isError of true", result: map[string]any{"isError": true}},
			{name: "result with a top-level is_error of true", result: map[string]any{"is_error": true}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				rows, calls, logger := runClassifiedToolCall(t, ToolResultVerdict{}, tc.result, nil)
				if got := requireToolErrorRow(t, rows)["error_message"]; got != mcpToolErrorMessage {
					t.Errorf("error_message = %v, want %q", got, mcpToolErrorMessage)
				}
				if len(calls) != 1 {
					t.Fatalf("classifier called %d times, want 1", len(calls))
				}
				if logger.contains("ToolResultClassifier") {
					t.Errorf("abstaining is not a failure, but a warning was logged:\n%s", logger)
				}
			})
		}
	})

	t.Run("nil classifier records TOOL_ERROR for a result with a top-level isError of true", func(t *testing.T) {
		// As in adk-python, which records this result as a tool error when it
		// has no classifier.
		rows, calls, _ := runClassifiedToolCall(t, ToolResultVerdict{}, map[string]any{"isError": true}, func(c *Config) {
			c.ToolResultClassifier = nil
		})
		if got := requireToolErrorRow(t, rows)["error_message"]; got != mcpToolErrorMessage {
			t.Errorf("error_message = %v, want %q", got, mcpToolErrorMessage)
		}
		if len(calls) != 0 {
			t.Fatalf("classifier called %d times, want 0", len(calls))
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

	t.Run("CaptureFullMessageContent writes the classifier's message in full, still scrubbed", func(t *testing.T) {
		long := strings.Repeat("x", 200) + " " + message
		rows, _, _ := runClassifiedToolCall(t, ToolResultVerdict{Decision: ToolResultError, ErrorMessage: long}, map[string]any{"status": "ERROR"}, func(c *Config) {
			c.MaxContentLen = 50
			c.CaptureFullMessageContent = true
		})
		row := requireToolErrorRow(t, rows)
		got, _ := row["error_message"].(string)
		if want := redactErrorMessage(long); got != want {
			t.Errorf("error_message = %q, want %q", got, want)
		}
		if strings.Contains(got, secret) {
			t.Errorf("error_message %q leaks the credential", got)
		}
		if row["is_truncated"] != false {
			t.Errorf("is_truncated = %v, want false", row["is_truncated"])
		}
	})
}

// TestPlugin_ToolResultClassifier_EventFilters checks that the classifier's
// verdict decides the AGENT_RESPONSE row of a FinalResponseToolNames tool
// when the event filters drop the TOOL_COMPLETED and TOOL_ERROR rows: the
// classifier runs before the filters are consulted.
func TestPlugin_ToolResultClassifier_EventFilters(t *testing.T) {
	for _, filter := range []struct {
		name   string
		mutate func(*Config)
	}{
		{
			name: "EventDenylist drops TOOL_COMPLETED and TOOL_ERROR",
			mutate: func(c *Config) {
				c.EventDenylist = []string{EventToolCompleted, EventToolError}
			},
		},
		{
			name: "EventAllowlist leaves out TOOL_COMPLETED and TOOL_ERROR",
			mutate: func(c *Config) {
				c.EventAllowlist = []string{EventToolStarting, EventAgentResponse, EventInvocationCompleted}
			},
		},
	} {
		t.Run(filter.name, func(t *testing.T) {
			for _, tc := range []struct {
				name      string
				verdict   ToolResultVerdict
				wantTypes []string
			}{
				{
					name:      "ERROR writes no AGENT_RESPONSE",
					verdict:   ToolResultVerdict{Decision: ToolResultError, ErrorMessage: "quota exceeded"},
					wantTypes: []string{EventToolStarting, EventInvocationCompleted},
				},
				{
					name:      "OK writes the AGENT_RESPONSE",
					verdict:   ToolResultVerdict{Decision: ToolResultOK},
					wantTypes: []string{EventToolStarting, EventAgentResponse, EventInvocationCompleted},
				},
				{
					name:      "abstain writes the AGENT_RESPONSE",
					verdict:   ToolResultVerdict{},
					wantTypes: []string{EventToolStarting, EventAgentResponse, EventInvocationCompleted},
				},
			} {
				t.Run(tc.name, func(t *testing.T) {
					rows, calls, _ := runClassifiedToolCall(t, tc.verdict, map[string]any{"status": "ERROR"}, filter.mutate)
					if diff := cmp.Diff(tc.wantTypes, eventTypes(rows)); diff != "" {
						t.Errorf("event types mismatch (-want +got):\n%s", diff)
					}
					if len(calls) != 1 {
						t.Errorf("classifier called %d times, want 1", len(calls))
					}
				})
			}
		})
	}
}

func TestPlugin_ToolResultClassifier_FailuresAreSkipped(t *testing.T) {
	// secret stands for tool output that must not reach the log.
	const secret = "result-secret-7f3a"
	// A failing classifier is skipped: the call is recorded as it is without
	// a classifier, which records it as TOOL_COMPLETED unless the built-in
	// rule records it as TOOL_ERROR.
	results := []struct {
		name   string
		result map[string]any
		// wantToolError reports that the built-in rule records the call as
		// TOOL_ERROR.
		wantToolError bool
	}{
		{name: "plain result", result: map[string]any{"detail": secret}},
		{name: "result that reports an error", result: map[string]any{"status": "ERROR", "detail": secret}},
		// As in adk-python, which records this one as a tool error when its
		// classifier fails.
		{name: "result with a top-level isError of true", result: map[string]any{"isError": true, "detail": secret}, wantToolError: true},
	}
	requireRow := func(t *testing.T, rows []map[string]any, wantToolError bool) {
		t.Helper()
		if !wantToolError {
			requireToolCompletedRow(t, rows)
			return
		}
		if got := requireToolErrorRow(t, rows)["error_message"]; got != mcpToolErrorMessage {
			t.Errorf("error_message = %v, want %q", got, mcpToolErrorMessage)
		}
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
				requireRow(t, rows, tc.wantToolError)
				if got := calls.Load(); got != 1 {
					t.Errorf("classifier called %d times, want 1", got)
				}
				const wantLine = `agentanalytics: ToolResultClassifier failed for tool "submit_answer": recovered panic of type *fmt.wrapError; recording the call as if the classifier abstained`
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
				requireRow(t, rows, tc.wantToolError)
				if len(calls) != 1 {
					t.Errorf("classifier called %d times, want 1", len(calls))
				}
				const wantLine = `agentanalytics: ToolResultClassifier returned an unusable verdict ToolResultDecision(7) for tool "submit_answer"; recording the call as if the classifier abstained`
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

	// afterTool records every finish_task attempt before classification, and
	// onEvent retires attempts whose final FunctionResponse is not a
	// completion envelope, so the rewritten result below leaves the task open
	// and cannot claim Run 1's completion whether or not a classifier records
	// the call as a failure.
	t.Run("finish_task result that is not a completion envelope does not note the task as finished", func(t *testing.T) {
		// Only the application knows that {"status": "ERROR"} is a failure.
		errorOnStatus := func(_ agent.Context, _ tool.Tool, _, result map[string]any) ToolResultVerdict {
			if result["status"] == "ERROR" {
				return ToolResultVerdict{Decision: ToolResultError}
			}
			return ToolResultVerdict{}
		}
		for _, tc := range []struct {
			name       string
			classifier ToolResultClassifier
			// wantToolErrors and wantToolCompleted count the rows of the three
			// finish_task calls.
			wantToolErrors    int
			wantToolCompleted int
		}{
			{
				name:              "a classifier records the result as TOOL_ERROR",
				classifier:        errorOnStatus,
				wantToolErrors:    1,
				wantToolCompleted: 2,
			},
			{
				// Without a classifier the rewritten result is a completed
				// call, and it still does not finish the task.
				name:              "a nil classifier records the result as TOOL_COMPLETED",
				classifier:        nil,
				wantToolErrors:    0,
				wantToolCompleted: 3,
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var runCount atomic.Int32
				p, requests, _ := setupTestPluginWithConfig(t, func(c *Config) {
					c.ToolResultClassifier = tc.classifier
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
				// {"status": "ERROR"}, a result that is not a completion envelope
				// whether a classifier records it as a failure or not, and then calls
				// finish_task("call-second") which succeeds. If Run 2's rewritten
				// "call-shared" were wrongly noted as an attempt that ended the task, both
				// activations would claim "call-shared" and finishedTaskLocked would fall back to the
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
							for range 2 {
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
				toolErrRows := rowsOfType(rows, EventToolError)
				if got := len(toolErrRows); got != tc.wantToolErrors {
					t.Fatalf("got %d TOOL_ERROR rows, want %d; events: %v", got, tc.wantToolErrors, eventTypes(rows))
				}
				for _, row := range toolErrRows {
					if got := row["error_message"]; got != classifiedToolErrorMessage {
						t.Errorf("TOOL_ERROR error_message = %v, want %q", got, classifiedToolErrorMessage)
					}
				}
				if got := len(rowsOfType(rows, EventToolCompleted)); got != tc.wantToolCompleted {
					t.Fatalf("got %d TOOL_COMPLETED rows, want %d; events: %v", got, tc.wantToolCompleted, eventTypes(rows))
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
		}
	})

	t.Run("later AfterToolCallback promoting finish_task failure to completion preserves activation attribution", func(t *testing.T) {
		for _, tc := range []struct {
			name               string
			reusedFailedCallID bool
		}{
			{
				name: "distinct call IDs with first run promoted after analytics afterTool",
			},
			{
				name:               "reused call ID on second run fails before second run completes",
				reusedFailedCallID: true,
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var runCount atomic.Int32
				// Keep every optional hook (ToolResultClassifier, OnSchemaReady,
				// OnSchemaError) unset.
				p, requests, _ := setupTestPluginWithConfig(t, nil)

				// Simulate a finish_task validation failure arriving at p.afterTool
				// with a nil error (for Run 1's promoted call, and optionally for
				// Run 2's first failed attempt reusing the same call ID).
				validationFailurePlugin, err := baseplugin.New(baseplugin.Config{
					Name: "validation_failure_before_analytics",
					AfterToolCallback: func(ctx agent.Context, tl tool.Tool, _, result map[string]any, err error) (map[string]any, error) {
						if err == nil && tl != nil && tl.Name() == finishTaskToolName && ctx.FunctionCallID() == "call-first" {
							delete(result, "result")
							result["error"] = "missing validation token"
						}
						return nil, nil
					},
				})
				if err != nil {
					t.Fatalf("baseplugin.New(validation_failure_before_analytics): %v", err)
				}

				// Registered after p so it runs after p.afterTool, matching a later
				// plugin or agent-level AfterToolCallback that turns Run 1's
				// non-completion map into a task completion before ADK builds the
				// FunctionResponse event.
				promoteAfterAnalyticsPlugin, err := baseplugin.New(baseplugin.Config{
					Name: "promote_after_analytics",
					AfterToolCallback: func(ctx agent.Context, tl tool.Tool, _, _ map[string]any, err error) (map[string]any, error) {
						if err == nil && tl != nil && tl.Name() == finishTaskToolName &&
							ctx.FunctionCallID() == "call-first" && runCount.Load() == 1 {
							return map[string]any{"result": "Task completed."}, nil
						}
						return nil, nil
					},
				})
				if err != nil {
					t.Fatalf("baseplugin.New(promote_after_analytics): %v", err)
				}

				var turn atomic.Int32
				taskModel := &funcStreamLLM{
					fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
						n := turn.Add(1)
						callID := "call-first"
						if (!tc.reusedFailedCallID && n == 2) || (tc.reusedFailedCallID && n == 3) {
							callID = "call-second"
						}
						yield(&model.LLMResponse{
							Content: &genai.Content{
								Role: "model",
								Parts: []*genai.Part{{
									FunctionCall: &genai.FunctionCall{
										ID:   callID,
										Name: finishTaskToolName,
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
							for range 2 {
								runCount.Add(1)
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

				runner := newTestRunner(t, root, validationFailurePlugin, p, promoteAfterAnalyticsPlugin)
				for _, runErr := range runner.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
					if runErr != nil {
						t.Fatalf("runner.Run error: %v", runErr)
					}
				}

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
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
					for range 2 {
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

	// adk-python records the answer of its ReflectAndRetryToolPlugin as a tool
	// error before its classifier runs. This plugin has no such rule: when a
	// retryandreflect plugin registered before it answers a tool error, the
	// answer reaches the classifier as the call's result.
	t.Run("retryandreflect answer reaches the classifier when that plugin is registered first", func(t *testing.T) {
		const retryAnswerType = "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN"
		for _, tc := range []struct {
			name              string
			nilClassifier     bool
			verdict           ToolResultVerdict
			wantToolCompleted int
			wantToolErrors    int
		}{
			{name: "nil classifier records TOOL_COMPLETED", nilClassifier: true, wantToolCompleted: 1},
			{name: "abstain records TOOL_COMPLETED", verdict: ToolResultVerdict{}, wantToolCompleted: 1},
			{name: "OK records TOOL_COMPLETED", verdict: ToolResultVerdict{Decision: ToolResultOK}, wantToolCompleted: 1},
			{name: "ERROR records TOOL_ERROR", verdict: ToolResultVerdict{Decision: ToolResultError, ErrorMessage: "lookup failed"}, wantToolErrors: 1},
		} {
			t.Run(tc.name, func(t *testing.T) {
				type toolArgs struct {
					Query string `json:"query"`
				}
				fnTool, err := functiontool.New(
					functiontool.Config{Name: "lookup", Description: "Looks up a value"},
					func(_ agent.Context, _ toolArgs) (map[string]any, error) {
						return nil, errors.New("boom")
					},
				)
				if err != nil {
					t.Fatalf("functiontool.New: %v", err)
				}
				var (
					calls []classifierCall
					mu    sync.Mutex
				)
				p, requests, _ := setupTestPluginWithConfig(t, func(c *Config) {
					if !tc.nilClassifier {
						c.ToolResultClassifier = recordingClassifier(&calls, &mu, tc.verdict)
					}
				})
				retryPlugin, err := retryandreflect.New()
				if err != nil {
					t.Fatalf("retryandreflect.New: %v", err)
				}
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
											Name: "lookup",
											Args: map[string]any{"query": "status"},
										},
									}},
								},
							},
							{Content: textContent("model", "gave up"), TurnComplete: true},
						},
					},
				})
				if err != nil {
					t.Fatalf("llmagent.New: %v", err)
				}
				for _, runErr := range newTestRunner(t, bot, retryPlugin, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
					if runErr != nil {
						t.Fatalf("runner.Run error: %v", runErr)
					}
				}

				rows := collectRows(t, requests, untilInvocationEnd)
				mu.Lock()
				gotCalls := append([]classifierCall(nil), calls...)
				mu.Unlock()
				wantCalls := 1
				if tc.nilClassifier {
					wantCalls = 0
				}
				if len(gotCalls) != wantCalls {
					t.Fatalf("classifier called %d times, want %d", len(gotCalls), wantCalls)
				}
				for _, call := range gotCalls {
					if got := call.result["response_type"]; got != retryAnswerType {
						t.Errorf("classifier result response_type = %v, want %q", got, retryAnswerType)
					}
				}
				completedRows := rowsOfType(rows, EventToolCompleted)
				if got := len(completedRows); got != tc.wantToolCompleted {
					t.Errorf("got %d TOOL_COMPLETED rows, want %d; events: %v", got, tc.wantToolCompleted, eventTypes(rows))
				}
				for _, row := range completedRows {
					checkJSONAt(t, jsonObjectColumn(t, row, "content"), `"`+retryAnswerType+`"`, "result", "response_type")
				}
				toolErrRows := rowsOfType(rows, EventToolError)
				if got := len(toolErrRows); got != tc.wantToolErrors {
					t.Errorf("got %d TOOL_ERROR rows, want %d; events: %v", got, tc.wantToolErrors, eventTypes(rows))
				}
				for _, row := range toolErrRows {
					if got := row["error_message"]; got != "lookup failed" {
						t.Errorf("TOOL_ERROR error_message = %v, want %q", got, "lookup failed")
					}
				}
			})
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
	for i := range calls {
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

// finishTaskCall returns a model response that calls finish_task once for
// each call ID in ids, in order, with args.
func finishTaskCall(args map[string]any, ids ...string) *model.LLMResponse {
	parts := make([]*genai.Part, 0, len(ids))
	for _, id := range ids {
		parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: finishTaskToolName, Args: args}})
	}
	return &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: parts}}
}

// runHeldTaskCompletions runs taskAgent twice, as two activations of the
// "work" node of a custom root agent, through a real runner with plugins. The
// root holds back every event that carries a completed finish_task response,
// as a parent that forwards a sub-agent's events late does, and forwards the
// held events in order once release returns true for an event of the second
// activation, after forwarding that event, so the second activation is still
// running, or else once both activations returned. run holds the number of the
// activation that is running.
func runHeldTaskCompletions(t *testing.T, taskAgent agent.Agent, run *atomic.Int32, release func(*session.Event) bool, plugins ...*baseplugin.Plugin) {
	t.Helper()
	root, err := agent.New(agent.Config{
		Name: "coordinator",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				var held []*session.Event
				flush := func() bool {
					for _, ev := range held {
						if !yield(ev, nil) {
							return false
						}
					}
					held = nil
					return true
				}
				nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
				for range 2 {
					n := run.Add(1)
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
						if n == 2 && len(held) > 0 && release != nil && release(ev) && !flush() {
							return
						}
					}
				}
				flush()
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(coordinator): %v", err)
	}
	for _, runErr := range newTestRunner(t, root, plugins...).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("runner.Run error: %v", runErr)
		}
	}
}

// requireTaskSpansInOrder checks that the two activations of the "worker"
// task agent each wrote one AGENT_STARTING and one AGENT_COMPLETED row, and
// that the first AGENT_COMPLETED row ends the first activation's span and the
// second ends the second's.
func requireTaskSpansInOrder(t *testing.T, rows []map[string]any) {
	t.Helper()
	starts := agentRowsOfType(rows, EventAgentStarting, "worker")
	ends := agentRowsOfType(rows, EventAgentCompleted, "worker")
	if len(starts) != 2 || len(ends) != 2 {
		t.Fatalf("got %d worker AGENT_STARTING and %d worker AGENT_COMPLETED rows, want 2 of each; events: %v", len(starts), len(ends), eventTypes(rows))
	}
	run1, run2 := starts[0]["span_id"], starts[1]["span_id"]
	if run1 == nil || run1 == "" || run1 == run2 {
		t.Fatalf("invalid worker AGENT_STARTING span IDs: run 1 = %v, run 2 = %v", run1, run2)
	}
	if got := ends[0]["span_id"]; got != run1 {
		t.Errorf("first worker AGENT_COMPLETED span_id = %v, want run 1 span %v (run 2 span = %v)", got, run1, run2)
	}
	if got := ends[1]["span_id"]; got != run2 {
		t.Errorf("second worker AGENT_COMPLETED span_id = %v, want run 2 span %v (run 1 span = %v)", got, run2, run1)
	}
}

// isFailedTaskResponse reports whether ev carries a finish_task function
// response that does not end the task.
func isFailedTaskResponse(ev *session.Event) bool {
	return len(nonEndingTaskResponseCalls(ev.Content)) > 0
}

// TestPlugin_FinishTask_OverlappingErrorDoesNotStealCompletion runs a
// task-mode agent twice through a real runner, with both activations calling
// finish_task with the same call ID, as providers that number calls per turn
// do. The first activation completes, but its parent forwards the completed
// response only after the second activation's call failed in ADK's real tool
// dispatch and while that activation is still running. The failed call must
// not take over the first activation's completion: the held response ends the
// first activation, and the second ends on its own AfterAgentCallback.
func TestPlugin_FinishTask_OverlappingErrorDoesNotStealCompletion(t *testing.T) {
	const sharedCall = "call_0"
	toolErr := errors.New("tool execution failed")
	// failRun2 fails the second activation's finish_task call in an agent
	// before-tool callback, which ADK runs after the plugin's, with err.
	failRun2 := func(run *atomic.Int32, err func(agent.Context, tool.Tool) error) llmagent.BeforeToolCallback {
		return func(ctx agent.Context, tl tool.Tool, _ map[string]any) (map[string]any, error) {
			if run.Load() == 2 && tl.Name() == finishTaskToolName {
				return nil, err(ctx, tl)
			}
			return nil, nil
		}
	}
	for _, tc := range []struct {
		name string
		// configure sets up the agent and the plugins around the analytics
		// plugin p, which logs the rows.
		configure      func(run *atomic.Int32, cfg *llmagent.Config, p *baseplugin.Plugin) []*baseplugin.Plugin
		wantToolErrors int
		wantErrMessage string
	}{
		{
			// ADK passes the error to the on-tool-error callbacks, where the
			// plugin records TOOL_ERROR, and then to the after-tool callbacks.
			name: "tool error",
			configure: func(run *atomic.Int32, cfg *llmagent.Config, p *baseplugin.Plugin) []*baseplugin.Plugin {
				cfg.BeforeToolCallbacks = []llmagent.BeforeToolCallback{failRun2(run, func(agent.Context, tool.Tool) error { return toolErr })}
				return []*baseplugin.Plugin{p}
			},
			wantToolErrors: 1,
			wantErrMessage: toolErr.Error(),
		},
		{
			// A plugin registered before the analytics plugin replaces the
			// error in its on-tool-error callback, so the plugin's
			// OnToolErrorCallback does not run and its AfterToolCallback
			// records the TOOL_ERROR row of the replaced error.
			name: "on-tool-error callback of an earlier plugin replaces the error",
			configure: func(run *atomic.Int32, cfg *llmagent.Config, p *baseplugin.Plugin) []*baseplugin.Plugin {
				cfg.BeforeToolCallbacks = []llmagent.BeforeToolCallback{failRun2(run, func(agent.Context, tool.Tool) error { return toolErr })}
				rewrap, err := baseplugin.New(baseplugin.Config{
					Name: "rewrap_tool_error",
					OnToolErrorCallback: func(_ agent.Context, _ tool.Tool, _ map[string]any, err error) (map[string]any, error) {
						return nil, fmt.Errorf("rewrapped: %w", err)
					},
				})
				if err != nil {
					t.Fatalf("baseplugin.New(rewrap_tool_error): %v", err)
				}
				return []*baseplugin.Plugin{rewrap, p}
			},
			wantToolErrors: 1,
			wantErrMessage: "rewrapped: " + toolErr.Error(),
		},
		{
			// An agent after-tool callback, which ADK runs after the
			// plugin's, turns the second activation's completion into an
			// error, so the plugin saw a completed attempt that the
			// function response then fails.
			name: "agent after-tool callback turns a completion into an error",
			configure: func(run *atomic.Int32, cfg *llmagent.Config, p *baseplugin.Plugin) []*baseplugin.Plugin {
				cfg.AfterToolCallbacks = []llmagent.AfterToolCallback{
					func(_ agent.Context, tl tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
						if run.Load() == 2 && tl.Name() == finishTaskToolName {
							return nil, errors.New("after_tool error")
						}
						return nil, nil
					},
				}
				return []*baseplugin.Plugin{p}
			},
		},
		{
			// The call asks for the user's confirmation, as
			// tool.WithConfirmation does, and the activation pauses.
			name: "confirmation required",
			configure: func(run *atomic.Int32, cfg *llmagent.Config, p *baseplugin.Plugin) []*baseplugin.Plugin {
				cfg.BeforeToolCallbacks = []llmagent.BeforeToolCallback{failRun2(run, func(ctx agent.Context, tl tool.Tool) error {
					if err := ctx.RequestConfirmation("approve finish_task", nil); err != nil {
						return err
					}
					ctx.Actions().SkipSummarization = true
					return fmt.Errorf("error tool %q %w", tl.Name(), tool.ErrConfirmationRequired)
				})}
				return []*baseplugin.Plugin{p}
			},
		},
		{
			// The call is rejected, as tool.WithConfirmation rejects a call
			// that the user declined.
			name: "confirmation rejected",
			configure: func(run *atomic.Int32, cfg *llmagent.Config, p *baseplugin.Plugin) []*baseplugin.Plugin {
				cfg.BeforeToolCallbacks = []llmagent.BeforeToolCallback{failRun2(run, func(_ agent.Context, tl tool.Tool) error {
					return fmt.Errorf("error tool %q %w", tl.Name(), tool.ErrConfirmationRejected)
				})}
				return []*baseplugin.Plugin{p}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPluginWithConfig(t, nil)
			var run atomic.Int32
			cfg := llmagent.Config{
				Name: "worker",
				Mode: llmagent.ModeTask,
				Model: &scriptedLLM{responses: []*model.LLMResponse{
					finishTaskCall(map[string]any{"result": "done"}, sharedCall),
					finishTaskCall(map[string]any{"result": "done"}, sharedCall),
					{Content: textContent("model", "could not finish"), TurnComplete: true},
				}},
			}
			plugins := tc.configure(&run, &cfg, p)
			worker, err := llmagent.New(cfg)
			if err != nil {
				t.Fatalf("llmagent.New(worker): %v", err)
			}
			runHeldTaskCompletions(t, worker, &run, isFailedTaskResponse, plugins...)

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			requireTaskSpansInOrder(t, rows)
			toolErrRows := rowsOfType(rows, EventToolError)
			if got := len(toolErrRows); got != tc.wantToolErrors {
				t.Fatalf("got %d TOOL_ERROR rows, want %d; events: %v", got, tc.wantToolErrors, eventTypes(rows))
			}
			for _, row := range toolErrRows {
				if got := row["error_message"]; got != tc.wantErrMessage {
					t.Errorf("TOOL_ERROR error_message = %v, want %q", got, tc.wantErrMessage)
				}
			}
		})
	}
}

// TestPlugin_FinishTask_PromotedErrorClosesOriginatingSpan runs a task-mode
// agent twice through a real runner. The first activation's finish_task call
// fails in ADK's real tool dispatch, and a plugin registered after the
// analytics plugin turns the failure into a completion in its after-tool
// callback, which ends the activation. Its parent forwards that response
// while the second activation is running: the response must end the first
// activation, which made the call, not the second, which is newer.
func TestPlugin_FinishTask_PromotedErrorClosesOriginatingSpan(t *testing.T) {
	for _, tc := range []struct {
		name           string
		err            func(tool.Tool) error
		wantToolErrors int
	}{
		{
			name:           "tool error",
			err:            func(tool.Tool) error { return errors.New("recoverable validation error") },
			wantToolErrors: 1,
		},
		{
			name: "confirmation required",
			err: func(tl tool.Tool) error {
				return fmt.Errorf("error tool %q %w", tl.Name(), tool.ErrConfirmationRequired)
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPluginWithConfig(t, nil)
			var run atomic.Int32
			promote, err := baseplugin.New(baseplugin.Config{
				Name: "promote_after_analytics",
				AfterToolCallback: func(_ agent.Context, tl tool.Tool, _, _ map[string]any, err error) (map[string]any, error) {
					if err != nil && run.Load() == 1 && tl.Name() == finishTaskToolName {
						return map[string]any{"result": "Task completed."}, nil
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New(promote_after_analytics): %v", err)
			}
			worker, err := llmagent.New(llmagent.Config{
				Name: "worker",
				Mode: llmagent.ModeTask,
				Model: &scriptedLLM{responses: []*model.LLMResponse{
					finishTaskCall(map[string]any{"result": "done"}, "call_0"),
					{Content: textContent("model", "nothing left to do"), TurnComplete: true},
				}},
				BeforeToolCallbacks: []llmagent.BeforeToolCallback{
					func(_ agent.Context, tl tool.Tool, _ map[string]any) (map[string]any, error) {
						if run.Load() == 1 && tl.Name() == finishTaskToolName {
							return nil, tc.err(tl)
						}
						return nil, nil
					},
				},
			})
			if err != nil {
				t.Fatalf("llmagent.New(worker): %v", err)
			}
			// Forward the held completion on the second activation's first
			// event, before that activation ends.
			runHeldTaskCompletions(t, worker, &run, func(*session.Event) bool { return true }, p, promote)

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			requireTaskSpansInOrder(t, rows)
			if got := len(rowsOfType(rows, EventToolError)); got != tc.wantToolErrors {
				t.Fatalf("got %d TOOL_ERROR rows, want %d; events: %v", got, tc.wantToolErrors, eventTypes(rows))
			}
		})
	}
}

// TestPlugin_FinishTask_MixedResponseEventEndsOwnActivation runs a task-mode
// agent twice through a real runner. The first activation calls finish_task
// twice in one model response: "call_x" fails validation and "call_y"
// completes, so ADK merges a failed and a completed finish_task response into
// one event. The second activation then completes with a call that reuses
// "call_x". The parent forwards both completed responses once both
// activations returned. The merged event must end the activation whose call
// completed, not every activation whose attempts share one of its call IDs.
func TestPlugin_FinishTask_MixedResponseEventEndsOwnActivation(t *testing.T) {
	p, requests, _ := setupTestPluginWithConfig(t, nil)
	var run atomic.Int32
	worker, err := llmagent.New(llmagent.Config{
		Name: "worker",
		Mode: llmagent.ModeTask,
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call_y", Name: finishTaskToolName, Args: map[string]any{"result": "done"}}},
				// Missing the required result, so finish_task returns a
				// validation error.
				{FunctionCall: &genai.FunctionCall{ID: "call_x", Name: finishTaskToolName, Args: map[string]any{}}},
			}}},
			finishTaskCall(map[string]any{"result": "done"}, "call_x"),
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(worker): %v", err)
	}
	var mixedEvents atomic.Int32
	inspect, err := baseplugin.New(baseplugin.Config{
		Name: "inspect_events",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if slices.Equal(endingTaskResponseCalls(ev.Content), []string{"call_y"}) &&
				slices.Equal(nonEndingTaskResponseCalls(ev.Content), []string{"call_x"}) {
				mixedEvents.Add(1)
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("baseplugin.New(inspect_events): %v", err)
	}
	runHeldTaskCompletions(t, worker, &run, nil, p, inspect)

	rows := collectRows(t, requests, untilInvocationEnd)
	if got := mixedEvents.Load(); got != 1 {
		t.Fatalf("got %d events with a completed call_y and a failed call_x finish_task response, want 1", got)
	}
	requireInvocationCompletedOK(t, rows)
	requireTaskSpansInOrder(t, rows)
}
