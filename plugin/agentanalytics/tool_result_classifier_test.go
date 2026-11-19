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
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
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

	t.Run("a classifier that panics is skipped and the warning logs only the panic type", func(t *testing.T) {
		for _, tc := range results {
			t.Run(tc.name, func(t *testing.T) {
				logger := &printfArgsLogger{recordingLogger: &recordingLogger{}}
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
				const wantLine = `agentanalytics: ToolResultClassifier failed: recovered panic of type *fmt.wrapError; recording the call as if the classifier abstained`
				if !logger.contains(wantLine) {
					t.Errorf("logger lacks %q; got:\n%s", wantLine, logger.recordingLogger)
				}
				if logger.contains(secret) || logger.contains("submit_answer") {
					t.Errorf("the warning leaks the tool result or tool name:\n%s", logger.recordingLogger)
				}
				for _, callArgs := range logger.recordedArgs() {
					for _, arg := range callArgs {
						if _, ok := arg.(string); !ok {
							t.Errorf("Logger.Printf received non-string panic arg %T (%v); want pre-formatted type string only", arg, arg)
						}
					}
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
				const wantLine = `agentanalytics: ToolResultClassifier returned an unusable verdict 7; recording the call as if the classifier abstained`
				if !logger.contains(wantLine) {
					t.Errorf("logger lacks %q; got:\n%s", wantLine, logger)
				}
				if logger.contains(secret) || logger.contains("ignored") || logger.contains("submit_answer") {
					t.Errorf("the warning leaks the result, verdict, or tool name:\n%s", logger)
				}
			})
		}
	})

	t.Run("an unregistered model-hallucinated tool name is not echoed to Logger", func(t *testing.T) {
		const hallucinatedTool = "hallucinated_unregistered_secret_tool_xyz"
		logger := &printfArgsLogger{recordingLogger: &recordingLogger{}}
		var n atomic.Int32
		p, requests, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.Logger = logger
			c.ToolResultClassifier = func(_ agent.Context, _ tool.Tool, _, _ map[string]any) ToolResultVerdict {
				if n.Add(1) == 1 {
					panic("classifier boom")
				}
				return ToolResultVerdict{Decision: ToolResultDecision(9)}
			}
		})
		answerMissing, err := baseplugin.New(baseplugin.Config{
			Name: "answer_missing_tool",
			BeforeToolCallback: func(_ agent.Context, tl tool.Tool, _ map[string]any) (map[string]any, error) {
				if tl != nil && tl.Name() == hallucinatedTool {
					return map[string]any{"status": "intercepted"}, nil
				}
				return nil, nil
			},
		})
		if err != nil {
			t.Fatalf("baseplugin.New(answer_missing_tool): %v", err)
		}
		bot, err := llmagent.New(llmagent.Config{
			Name: "bot",
			Model: &scriptedLLM{responses: []*model.LLMResponse{
				{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: hallucinatedTool, Args: map[string]any{"q": "1"}}},
				}}},
				{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "call-2", Name: hallucinatedTool, Args: map[string]any{"q": "2"}}},
				}}},
				{Content: textContent("model", "done"), TurnComplete: true},
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot): %v", err)
		}
		for _, runErr := range newTestRunner(t, bot, answerMissing, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error: %v", runErr)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		if got := n.Load(); got != 2 {
			t.Fatalf("classifier called %d times, want 2", got)
		}
		if !logger.contains("ToolResultClassifier failed: recovered panic of type string") {
			t.Errorf("logger lacks panic warning; got:\n%s", logger.recordingLogger)
		}
		if !logger.contains("ToolResultClassifier returned an unusable verdict 9") {
			t.Errorf("logger lacks unusable-verdict warning; got:\n%s", logger.recordingLogger)
		}
		if logger.contains(hallucinatedTool) || logger.contains("classifier boom") {
			t.Errorf("logger echoed unregistered model-hallucinated tool name or panic payload:\n%s", logger.recordingLogger)
		}
		if gotArgs := logger.recordedArgs(); !cmp.Equal(gotArgs, [][]any{{"string"}, {9}}) {
			t.Errorf("Logger.Printf args = %#v, want [[\"string\"], [9]]", gotArgs)
		}
	})
}

type printfArgsLogger struct {
	*recordingLogger
	mu   sync.Mutex
	args [][]any
}

func (l *printfArgsLogger) Printf(format string, args ...any) {
	l.mu.Lock()
	l.args = append(l.args, slices.Clone(args))
	l.mu.Unlock()
	l.recordingLogger.Printf(format, args...)
}

func (l *printfArgsLogger) recordedArgs() [][]any {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.Clone(l.args)
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
									if out := classifyTaskResponses(ev.Content); out.ends && !out.failed {
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
									if out := classifyTaskResponses(ev.Content); out.ends && !out.failed {
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
							if out := classifyTaskResponses(ev.Content); out.ends && !out.failed {
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
						if out := classifyTaskResponses(ev.Content); out.ends && !out.failed {
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
	if ev == nil || ev.Content == nil {
		return false
	}
	for _, part := range ev.Content.Parts {
		if part != nil && part.FunctionResponse != nil &&
			part.FunctionResponse.Name == finishTaskToolName &&
			!isTaskEndingResult(part.FunctionResponse.Response) {
			return true
		}
	}
	return false
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
			out := classifyTaskResponses(ev.Content)
			if out.ends && !out.failed && out.endingCallID == "call_y" &&
				slices.Equal(out.retiredCallIDs, []string{"call_x"}) {
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

// TestPlugin_FinishTask_FailedFirstThenCompletedInSameEventStaysOpen verifies
// ADK's first-response finish_task rule end-to-end: when a single model turn
// emits a failed finish_task call followed by a sentinel-bearing finish_task
// call in the same event, ADK does not end the task on that merged event. The
// plugin must keep the task activation span open so subsequent LLM and tool
// calls stay parented to the task span, retire the overruled call ID so a
// second activation reusing it can complete cleanly even while the first
// activation is still open, and close the first activation only on its real
// completion.
func TestPlugin_FinishTask_FailedFirstThenCompletedInSameEventStaysOpen(t *testing.T) {
	type stepArgs struct {
		Note string `json:"note"`
	}
	stepTool, err := functiontool.New(
		functiontool.Config{Name: "record_step", Description: "records intermediate work"},
		func(_ agent.Context, args stepArgs) (map[string]any, error) {
			return map[string]any{"recorded": args.Note}, nil
		},
	)
	if err != nil {
		t.Fatalf("functiontool.New(record_step): %v", err)
	}

	p, requests, _ := setupTestPluginWithConfig(t, nil)
	var overruledEvents atomic.Int32
	inspect, err := baseplugin.New(baseplugin.Config{
		Name: "inspect_overruled_event",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			out := classifyTaskResponses(ev.Content)
			if !out.ends &&
				slices.Equal(out.retiredCallIDs, []string{"call_0"}) &&
				slices.Equal(out.overruledEndingCallIDs, []string{"call_1"}) {
				overruledEvents.Add(1)
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("baseplugin.New(inspect_overruled_event): %v", err)
	}

	var modelTurn atomic.Int32
	workerModel := &funcStreamLLM{
		fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
			switch modelTurn.Add(1) {
			case 1:
				// Run 1, turn 1: first finish_task fails validation, second carries valid args.
				yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "call_0", Name: finishTaskToolName, Args: map[string]any{}}},
					{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: finishTaskToolName, Args: map[string]any{"result": "premature"}}},
				}}}, nil)
			case 2:
				// Run 2, turn 1 (interleaved while Run 1 is still open): reuses "call_1" and succeeds!
				yield(finishTaskCall(map[string]any{"result": "run 2 done"}, "call_1"), nil)
			case 3:
				// Run 1, turn 2: calls regular tool while task span must still be open.
				yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "call_step", Name: "record_step", Args: map[string]any{"note": "still in run 1"}}},
				}}}, nil)
			case 4:
				// Run 1, turn 3: real completion.
				yield(finishTaskCall(map[string]any{"result": "run 1 done"}, "call_2"), nil)
			default:
				yield(&model.LLMResponse{Content: textContent("model", "unexpected turn"), TurnComplete: true}, nil)
			}
		},
	}
	worker, err := llmagent.New(llmagent.Config{
		Name:  "worker",
		Mode:  llmagent.ModeTask,
		Tools: []tool.Tool{stepTool},
		Model: workerModel,
	})
	if err != nil {
		t.Fatalf("llmagent.New(worker): %v", err)
	}

	root, err := agent.New(agent.Config{
		Name: "coordinator",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
				next1, stop1 := iter.Pull2(llmagent.RunLLMAgentAsNode(worker, nodeCtx, "task 1"))
				defer stop1()

				// Pull Run 1 until its merged [failed, completed] FunctionResponse event is yielded,
				// leaving Run 1 paused before turn 2 while its activation is still open!
				for {
					ev, runErr, ok := next1()
					if !ok {
						t.Errorf("Run 1 ended prematurely on turn 1")
						return
					}
					if !yield(ev, runErr) || runErr != nil {
						return
					}
					out := classifyTaskResponses(ev.Content)
					if !out.ends && len(out.overruledEndingCallIDs) > 0 {
						break
					}
				}

				// Run Run 2 to completion (reusing "call_1") while Run 1 is still open!
				for ev, runErr := range llmagent.RunLLMAgentAsNode(worker, nodeCtx, "task 2") {
					if !yield(ev, runErr) || runErr != nil {
						return
					}
				}

				// Resume Run 1 for turn 2 (record_step) and turn 3 (real finish_task completion).
				for {
					ev, runErr, ok := next1()
					if !ok {
						break
					}
					if !yield(ev, runErr) || runErr != nil {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(coordinator): %v", err)
	}
	for _, runErr := range newTestRunner(t, root, p, inspect).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("runner.Run error: %v", runErr)
		}
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if got := overruledEvents.Load(); got != 1 {
		t.Fatalf("got %d overruled mixed finish_task events, want 1", got)
	}
	requireInvocationCompletedOK(t, rows)

	starts := agentRowsOfType(rows, EventAgentStarting, "worker")
	ends := agentRowsOfType(rows, EventAgentCompleted, "worker")
	if len(starts) != 2 || len(ends) != 2 {
		t.Fatalf("got %d worker AGENT_STARTING and %d worker AGENT_COMPLETED rows, want 2 of each; events: %v", len(starts), len(ends), eventTypes(rows))
	}
	run1Span, run2Span := starts[0]["span_id"], starts[1]["span_id"]
	// Run 2 finished first while Run 1 stayed open, then Run 1 finished on call_2!
	if got := ends[0]["span_id"]; got != run2Span {
		t.Errorf("first worker AGENT_COMPLETED span_id = %v, want Run 2 span %v (Run 1 span = %v)", got, run2Span, run1Span)
	}
	if got := ends[1]["span_id"]; got != run1Span {
		t.Errorf("second worker AGENT_COMPLETED span_id = %v, want Run 1 span %v (Run 2 span = %v)", got, run1Span, run2Span)
	}

	// Verify Run 1's turn-2 record_step tool rows and turn-3 call_2 finish_task rows remained parented to Run 1's task span.
	var sawStepStart, sawStepEnd, sawFinalFinishStart, sawFinalFinishEnd bool
	for _, row := range rows {
		switch row["event_type"] {
		case EventToolStarting:
			switch functionCallIDOf(t, row) {
			case "call_step":
				sawStepStart = true
				if got := row["parent_span_id"]; got != run1Span {
					t.Errorf("record_step TOOL_STARTING parent_span_id = %v, want Run 1 span %v", got, run1Span)
				}
			case "call_2":
				sawFinalFinishStart = true
				if got := row["parent_span_id"]; got != run1Span {
					t.Errorf("call_2 TOOL_STARTING parent_span_id = %v, want Run 1 span %v", got, run1Span)
				}
			}
		case EventToolCompleted:
			switch functionCallIDOf(t, row) {
			case "call_step":
				sawStepEnd = true
				if got := row["parent_span_id"]; got != run1Span {
					t.Errorf("record_step TOOL_COMPLETED parent_span_id = %v, want Run 1 span %v", got, run1Span)
				}
			case "call_2":
				sawFinalFinishEnd = true
				if got := row["parent_span_id"]; got != run1Span {
					t.Errorf("call_2 TOOL_COMPLETED parent_span_id = %v, want Run 1 span %v", got, run1Span)
				}
			}
		}
	}
	if !sawStepStart || !sawStepEnd || !sawFinalFinishStart || !sawFinalFinishEnd {
		t.Errorf("missing expected Run 1 post-retry tool rows: step=(%v,%v) call_2=(%v,%v)", sawStepStart, sawStepEnd, sawFinalFinishStart, sawFinalFinishEnd)
	}
}

// TestPlugin_FinishTask_ReusedCallID_AttributionOrderings verifies that an
// observed finish_task completion on Run 1 is never stolen or erased by a
// reused call ID failing on Run 2 across event delivery orderings and plugin
// short-circuiting.
func TestPlugin_FinishTask_ReusedCallID_AttributionOrderings(t *testing.T) {
	const sharedCall = "call_0"

	t.Run("held Run 1 success delivered right before Run 2 validation failure", func(t *testing.T) {
		p, requests, _ := setupTestPluginWithConfig(t, nil)
		worker, err := llmagent.New(llmagent.Config{
			Name: "worker",
			Mode: llmagent.ModeTask,
			Model: &scriptedLLM{responses: []*model.LLMResponse{
				// Run 1: succeeds with call_0.
				finishTaskCall(map[string]any{"result": "run 1 ok"}, sharedCall),
				// Run 2, turn 1: fails validation with call_0 (missing result).
				finishTaskCall(map[string]any{}, sharedCall),
				// Run 2, turn 2: succeeds reusing call_0.
				finishTaskCall(map[string]any{"result": "run 2 ok"}, sharedCall),
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(worker): %v", err)
		}

		root, err := agent.New(agent.Config{
			Name: "coordinator",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
					var heldRun1 []*session.Event
					for ev, runErr := range llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work 1") {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if out := classifyTaskResponses(ev.Content); out.ends && !out.failed {
							heldRun1 = append(heldRun1, ev)
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}

					next2, stop2 := iter.Pull2(llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work 2"))
					defer stop2()
					flushed := false
					for {
						ev, runErr, ok := next2()
						if !ok {
							break
						}
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						// Right when Run 2's validation failure event is ready (meaning Run 2's
						// pending call_0 attempt is already recorded on agentStack!), deliver
						// Run 1's held call_0 completion BEFORE Run 2's validation failure!
						if !flushed && isFailedTaskResponse(ev) {
							flushed = true
							for _, h := range heldRun1 {
								if !yield(h, nil) {
									return
								}
							}
							heldRun1 = nil
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(coordinator): %v", err)
		}
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error: %v", runErr)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		requireTaskSpansInOrder(t, rows)
	})

	t.Run("held Run 1 success delivered after Run 2 batch validation failures", func(t *testing.T) {
		p, requests, _ := setupTestPluginWithConfig(t, nil)
		var run atomic.Int32
		worker, err := llmagent.New(llmagent.Config{
			Name: "worker",
			Mode: llmagent.ModeTask,
			Model: &scriptedLLM{responses: []*model.LLMResponse{
				// Run 1: succeeds with call_0.
				finishTaskCall(map[string]any{"result": "run 1 ok"}, sharedCall),
				// Run 2, turn 1: batch of three failed finish_task attempts.
				{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: sharedCall, Name: finishTaskToolName, Args: map[string]any{}}},
					{FunctionCall: &genai.FunctionCall{ID: "call_extra_1", Name: finishTaskToolName, Args: map[string]any{}}},
					{FunctionCall: &genai.FunctionCall{ID: "call_extra_2", Name: finishTaskToolName, Args: map[string]any{}}},
				}}},
				// Run 2, turn 2: succeeds reusing call_0.
				finishTaskCall(map[string]any{"result": "run 2 ok"}, sharedCall),
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(worker): %v", err)
		}
		runHeldTaskCompletions(t, worker, &run, isFailedTaskResponse, p)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		requireTaskSpansInOrder(t, rows)
	})

	for _, alsoShortCircuitAfter := range []bool{false, true} {
		subName := "earlier_BeforeToolCallback_short_circuits_Run_2_failed_call"
		if alsoShortCircuitAfter {
			subName = "earlier_BeforeToolCallback_and_AfterToolCallback_short_circuit_Run_2_failed_call"
		}
		t.Run(subName, func(t *testing.T) {
			p, requests, _ := setupTestPluginWithConfig(t, nil)
			var (
				run        atomic.Int32
				run2Failed atomic.Bool
			)
			cfg := baseplugin.Config{
				Name: "short_circuit_run2_failure",
				BeforeToolCallback: func(ctx agent.Context, tl tool.Tool, _ map[string]any) (map[string]any, error) {
					if run.Load() == 2 && tl != nil && tl.Name() == finishTaskToolName &&
						ctx.FunctionCallID() == sharedCall && run2Failed.CompareAndSwap(false, true) {
						return map[string]any{"error": "blocked in before_tool"}, nil
					}
					return nil, nil
				},
			}
			if alsoShortCircuitAfter {
				cfg.AfterToolCallback = func(ctx agent.Context, tl tool.Tool, _, res map[string]any, _ error) (map[string]any, error) {
					if run.Load() == 2 && tl != nil && tl.Name() == finishTaskToolName &&
						ctx.FunctionCallID() == sharedCall && res["error"] == "blocked in before_tool" {
						return res, nil
					}
					return nil, nil
				}
			}
			earlier, err := baseplugin.New(cfg)
			if err != nil {
				t.Fatalf("baseplugin.New(short_circuit_run2_failure): %v", err)
			}
			worker, err := llmagent.New(llmagent.Config{
				Name: "worker",
				Mode: llmagent.ModeTask,
				Model: &scriptedLLM{responses: []*model.LLMResponse{
					finishTaskCall(map[string]any{"result": "run 1 ok"}, sharedCall),
					finishTaskCall(map[string]any{"result": "run 2 blocked"}, sharedCall),
					finishTaskCall(map[string]any{"result": "run 2 ok"}, sharedCall),
				}},
			})
			if err != nil {
				t.Fatalf("llmagent.New(worker): %v", err)
			}
			runHeldTaskCompletions(t, worker, &run, isFailedTaskResponse, earlier, p)

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			requireTaskSpansInOrder(t, rows)
		})
	}
}

func TestClassifyTaskResponses_MultiPartOrderings(t *testing.T) {
	frPart := func(id, name string, resp map[string]any) *genai.Part {
		return &genai.Part{
			FunctionResponse: &genai.FunctionResponse{
				ID:       id,
				Name:     name,
				Response: resp,
			},
		}
	}
	endingResp := func() map[string]any {
		return map[string]any{"result": "Task completed."}
	}
	endingErrResp := func() map[string]any {
		return map[string]any{"result": "Task completed.", "isError": true}
	}
	failedResp := func() map[string]any {
		return map[string]any{"error": "validation failed"}
	}

	for _, tc := range []struct {
		name    string
		content *genai.Content
		want    taskResponseOutcome
	}{
		{
			name:    "nil_content",
			content: nil,
			want:    taskResponseOutcome{},
		},
		{
			name: "single_ending",
			content: &genai.Content{Parts: []*genai.Part{
				frPart("call_0", finishTaskToolName, endingResp()),
			}},
			want: taskResponseOutcome{
				ends:                true,
				leadingActionCallID: "call_0",
				endingCallID:        "call_0",
				endingCallIDs:       []string{"call_0"},
			},
		},
		{
			name: "single_failed",
			content: &genai.Content{Parts: []*genai.Part{
				frPart("call_0", finishTaskToolName, failedResp()),
			}},
			want: taskResponseOutcome{
				ends:                false,
				leadingActionCallID: "call_0",
				retiredCallIDs:      []string{"call_0"},
			},
		},
		{
			name: "ending_then_non_ending",
			content: &genai.Content{Parts: []*genai.Part{
				frPart("call_0", finishTaskToolName, endingResp()),
				frPart("call_1", finishTaskToolName, failedResp()),
			}},
			want: taskResponseOutcome{
				ends:                true,
				mergedResponse:      true,
				leadingActionCallID: "call_0",
				endingCallID:        "call_0",
				endingCallIDs:       []string{"call_0"},
				retiredCallIDs:      []string{"call_1"},
			},
		},
		{
			name: "ending_then_ending",
			content: &genai.Content{Parts: []*genai.Part{
				frPart("call_0", finishTaskToolName, endingErrResp()),
				frPart("call_1", finishTaskToolName, endingResp()),
			}},
			want: taskResponseOutcome{
				ends:                true,
				failed:              true,
				mergedResponse:      true,
				errMsg:              "Tool returned an MCP result with isError=true.",
				leadingActionCallID: "call_0",
				endingCallID:        "call_0",
				endingCallIDs:       []string{"call_0", "call_1"},
			},
		},
		{
			name: "ending_with_empty_ID_then_ending",
			content: &genai.Content{Parts: []*genai.Part{
				frPart("", finishTaskToolName, endingResp()),
				frPart("call_1", finishTaskToolName, endingResp()),
			}},
			want: taskResponseOutcome{
				ends:           true,
				mergedResponse: true,
				endingCallID:   "call_1",
				endingCallIDs:  []string{"call_1"},
			},
		},
		{
			name: "failed_with_empty_ID_then_ending",
			content: &genai.Content{Parts: []*genai.Part{
				frPart("", finishTaskToolName, failedResp()),
				frPart("call_1", finishTaskToolName, endingResp()),
			}},
			want: taskResponseOutcome{
				ends:                   false,
				mergedResponse:         true,
				overruledEndingCallIDs: []string{"call_1"},
			},
		},
		{
			name: "failed_then_failed_duplicate_ID",
			content: &genai.Content{Parts: []*genai.Part{
				frPart("call_0", finishTaskToolName, failedResp()),
				frPart("call_0", finishTaskToolName, failedResp()),
			}},
			want: taskResponseOutcome{
				ends:                false,
				mergedResponse:      true,
				leadingActionCallID: "call_0",
				retiredCallIDs:      []string{"call_0", "call_0"},
			},
		},
		{
			name: "interleaved_non_finish_task_parts",
			content: &genai.Content{Parts: []*genai.Part{
				nil,
				{Text: "preface"},
				frPart("other_0", "lookup_item", map[string]any{"result": "Task completed."}),
				frPart("call_0", finishTaskToolName, failedResp()),
				frPart("other_1", "lookup_item", failedResp()),
				frPart("call_1", finishTaskToolName, endingResp()),
				frPart("", finishTaskToolName, endingResp()),
				frPart("call_2", finishTaskToolName, failedResp()),
			}},
			want: taskResponseOutcome{
				ends:                   false,
				mergedResponse:         true,
				retiredCallIDs:         []string{"call_0", "call_2"},
				overruledEndingCallIDs: []string{"call_1"},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := classifyTaskResponses(tc.content)
			if diff := cmp.Diff(tc.want, got, cmp.AllowUnexported(taskResponseOutcome{})); diff != "" {
				t.Errorf("classifyTaskResponses() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestPlugin_FinishTask_FailedEmptyIDThenCompletedStaysOpenAndRetiresOverruledCall
// verifies that when a merged finish_task FunctionResponse event begins with a
// failed finish_task response whose ID is empty followed by a sentinel-bearing
// finish_task response ("call_1"), overruledEndingCallIDs is still retired on
// the originating activation even though retiredCallIDs is empty: a second
// activation reusing "call_1" closes its own span rather than stealing Run 1's
// open span, and Run 1 later completes cleanly reusing "call_1".
func TestPlugin_FinishTask_FailedEmptyIDThenCompletedStaysOpenAndRetiresOverruledCall(t *testing.T) {
	p, requests, _ := setupTestPluginWithConfig(t, nil)

	var sawEmptyIDOverruled atomic.Int32
	stripFailedFirstID, err := baseplugin.New(baseplugin.Config{
		Name: "strip_failed_first_id",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if ev == nil || ev.Content == nil {
				return nil, nil
			}
			for _, part := range ev.Content.Parts {
				if part != nil && part.FunctionResponse != nil &&
					part.FunctionResponse.Name == finishTaskToolName &&
					part.FunctionResponse.ID == "call_empty_first" {
					part.FunctionResponse.ID = ""
				}
			}
			out := classifyTaskResponses(ev.Content)
			if !out.ends && len(out.retiredCallIDs) == 0 &&
				slices.Equal(out.overruledEndingCallIDs, []string{"call_1"}) &&
				isFailedTaskResponse(ev) {
				sawEmptyIDOverruled.Add(1)
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("baseplugin.New(strip_failed_first_id): %v", err)
	}

	worker, err := llmagent.New(llmagent.Config{
		Name: "worker",
		Mode: llmagent.ModeTask,
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			// Run 1, turn 1: first finish_task fails validation (its response ID is
			// cleared to "" before analyticsPlugin.onEvent); second carries the
			// completion sentinel with "call_1". ADK keeps Run 1 open.
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call_empty_first", Name: finishTaskToolName, Args: map[string]any{}}},
				{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: finishTaskToolName, Args: map[string]any{"result": "overruled"}}},
			}}},
			// Run 2, turn 1 (while Run 1 is paused): completes with "call_1".
			finishTaskCall(map[string]any{"result": "run 2 done"}, "call_1"),
			// Run 1, turn 2: completes reusing "call_1".
			finishTaskCall(map[string]any{"result": "run 1 done"}, "call_1"),
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(worker): %v", err)
	}

	root, err := agent.New(agent.Config{
		Name: "coordinator",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
				next1, stop1 := iter.Pull2(llmagent.RunLLMAgentAsNode(worker, nodeCtx, "task 1"))
				defer stop1()
				for {
					ev, runErr, ok := next1()
					if !ok {
						t.Error("Run 1 ended prematurely on overruled finish_task event")
						return
					}
					if !yield(ev, runErr) || runErr != nil {
						return
					}
					if out := classifyTaskResponses(ev.Content); !out.ends && len(out.overruledEndingCallIDs) > 0 {
						break
					}
				}

				for ev, runErr := range llmagent.RunLLMAgentAsNode(worker, nodeCtx, "task 2") {
					if !yield(ev, runErr) || runErr != nil {
						return
					}
				}

				for {
					ev, runErr, ok := next1()
					if !ok {
						break
					}
					if !yield(ev, runErr) || runErr != nil {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(coordinator): %v", err)
	}
	for _, runErr := range newTestRunner(t, root, stripFailedFirstID, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("runner.Run error: %v", runErr)
		}
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	if got := sawEmptyIDOverruled.Load(); got != 1 {
		t.Fatalf("sawEmptyIDOverruled = %d, want 1", got)
	}
	requireInvocationCompletedOK(t, rows)

	starts := agentRowsOfType(rows, EventAgentStarting, "worker")
	ends := agentRowsOfType(rows, EventAgentCompleted, "worker")
	if len(starts) != 2 || len(ends) != 2 {
		t.Fatalf("got %d worker AGENT_STARTING and %d worker AGENT_COMPLETED rows, want 2 of each; events: %v", len(starts), len(ends), eventTypes(rows))
	}
	run1Span, run2Span := starts[0]["span_id"], starts[1]["span_id"]
	if got := ends[0]["span_id"]; got != run2Span {
		t.Errorf("first worker AGENT_COMPLETED span_id = %v, want Run 2 span %v (Run 1 span = %v)", got, run2Span, run1Span)
	}
	if got := ends[1]["span_id"]; got != run1Span {
		t.Errorf("second worker AGENT_COMPLETED span_id = %v, want Run 1 span %v (Run 2 span = %v)", got, run1Span, run2Span)
	}
}

// TestPlugin_FinishTask_PromotedRun2DeliveredBeforeHeldRun1Completion verifies
// that when Run 1 completes with "call_0" (event held by the coordinator) and
// Run 2 reuses "call_0" (either promoted from a non-ending attempt by an
// agent-level or later-plugin AfterToolCallback, or completed directly) and
// delivers Run 2's completion event BEFORE Run 1's held completion event,
// Run 2's event closes Run 2's span and Run 1's held event closes Run 1's span.
func TestPlugin_FinishTask_PromotedRun2DeliveredBeforeHeldRun1Completion(t *testing.T) {
	const sharedCall = "call_0"
	toolErr := errors.New("transient finish failure")

	for _, tc := range []struct {
		name      string
		configure func(run *atomic.Int32, cfg *llmagent.Config, p *baseplugin.Plugin) []*baseplugin.Plugin
	}{
		{
			name: "agent_AfterToolCallback_promotes_Run_2_error_delivered_before_held_Run_1",
			configure: func(run *atomic.Int32, cfg *llmagent.Config, p *baseplugin.Plugin) []*baseplugin.Plugin {
				cfg.BeforeToolCallbacks = []llmagent.BeforeToolCallback{
					func(_ agent.Context, tl tool.Tool, _ map[string]any) (map[string]any, error) {
						if run.Load() == 2 && tl != nil && tl.Name() == finishTaskToolName {
							return nil, toolErr
						}
						return nil, nil
					},
				}
				cfg.AfterToolCallbacks = []llmagent.AfterToolCallback{
					func(_ agent.Context, tl tool.Tool, _, _ map[string]any, err error) (map[string]any, error) {
						if run.Load() == 2 && tl != nil && tl.Name() == finishTaskToolName && errors.Is(err, toolErr) {
							return map[string]any{"result": "Task completed."}, nil
						}
						return nil, nil
					},
				}
				return []*baseplugin.Plugin{p}
			},
		},
		{
			name: "later_plugin_AfterToolCallback_promotes_Run_2_error_delivered_before_held_Run_1",
			configure: func(run *atomic.Int32, cfg *llmagent.Config, p *baseplugin.Plugin) []*baseplugin.Plugin {
				cfg.BeforeToolCallbacks = []llmagent.BeforeToolCallback{
					func(_ agent.Context, tl tool.Tool, _ map[string]any) (map[string]any, error) {
						if run.Load() == 2 && tl != nil && tl.Name() == finishTaskToolName {
							return nil, toolErr
						}
						return nil, nil
					},
				}
				promote, err := baseplugin.New(baseplugin.Config{
					Name: "promote_run2_after_analytics",
					AfterToolCallback: func(_ agent.Context, tl tool.Tool, _, _ map[string]any, err error) (map[string]any, error) {
						if run.Load() == 2 && tl != nil && tl.Name() == finishTaskToolName && errors.Is(err, toolErr) {
							return map[string]any{"result": "Task completed."}, nil
						}
						return nil, nil
					},
				})
				if err != nil {
					t.Fatalf("baseplugin.New(promote_run2_after_analytics): %v", err)
				}
				return []*baseplugin.Plugin{p, promote}
			},
		},
		{
			name: "direct_Run_2_completion_delivered_before_held_Run_1_completion",
			configure: func(_ *atomic.Int32, _ *llmagent.Config, p *baseplugin.Plugin) []*baseplugin.Plugin {
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
					finishTaskCall(map[string]any{"result": "run 1 ok"}, sharedCall),
					finishTaskCall(map[string]any{"result": "run 2 ok"}, sharedCall),
				}},
			}
			plugins := tc.configure(&run, &cfg, p)
			worker, err := llmagent.New(cfg)
			if err != nil {
				t.Fatalf("llmagent.New(worker): %v", err)
			}

			root, err := agent.New(agent.Config{
				Name: "coordinator",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
						var heldRun1 []*session.Event
						run.Store(1)
						for ev, runErr := range llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work") {
							if runErr != nil {
								yield(nil, runErr)
								return
							}
							if out := classifyTaskResponses(ev.Content); out.ends {
								heldRun1 = append(heldRun1, ev)
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}

						run.Store(2)
						for ev, runErr := range llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work") {
							if !yield(ev, runErr) || runErr != nil {
								return
							}
						}
						for _, ev := range heldRun1 {
							if !yield(ev, nil) {
								return
							}
						}
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

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			starts := agentRowsOfType(rows, EventAgentStarting, "worker")
			ends := agentRowsOfType(rows, EventAgentCompleted, "worker")
			if len(starts) != 2 || len(ends) != 2 {
				t.Fatalf("got %d worker AGENT_STARTING and %d worker AGENT_COMPLETED rows, want 2 of each; events: %v", len(starts), len(ends), eventTypes(rows))
			}
			run1Span, run2Span := starts[0]["span_id"], starts[1]["span_id"]
			// Run 2's completion event was yielded before Run 1's held completion event.
			if got := ends[0]["span_id"]; got != run2Span {
				t.Errorf("first worker AGENT_COMPLETED span_id = %v, want Run 2 span %v (Run 1 span = %v)", got, run2Span, run1Span)
			}
			if got := ends[1]["span_id"]; got != run1Span {
				t.Errorf("second worker AGENT_COMPLETED span_id = %v, want Run 1 span %v (Run 2 span = %v)", got, run1Span, run2Span)
			}
		})
	}
}

// TestPlugin_FinishTask_TrailingObservedEndingCallIDResolvesActivation verifies
// that when a merged ending finish_task event begins with an unobserved call ID
// (answered by an earlier plugin) or an empty call ID followed by a second
// observed ending call ID ("call_2"), popTaskAgentSpanForCalls resolves the
// activation via the trailing observed call ID instead of falling into
// unobservedTaskLocked.
func TestPlugin_FinishTask_TrailingObservedEndingCallIDResolvesActivation(t *testing.T) {
	p, requests, _ := setupTestPluginWithConfig(t, nil)

	earlier, err := baseplugin.New(baseplugin.Config{
		Name: "short_circuit_unobserved_first",
		BeforeToolCallback: func(ctx agent.Context, tl tool.Tool, _ map[string]any) (map[string]any, error) {
			if tl != nil && tl.Name() == finishTaskToolName && ctx.FunctionCallID() == "call_unobs" {
				return map[string]any{"result": "Task completed."}, nil
			}
			return nil, nil
		},
		AfterToolCallback: func(ctx agent.Context, tl tool.Tool, _, res map[string]any, _ error) (map[string]any, error) {
			if tl != nil && tl.Name() == finishTaskToolName && ctx.FunctionCallID() == "call_unobs" {
				return res, nil
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("baseplugin.New(short_circuit_unobserved_first): %v", err)
	}

	worker, err := llmagent.New(llmagent.Config{
		Name: "worker",
		Mode: llmagent.ModeTask,
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			// Run 1: observed completion "call_1", held until after Run 2 finishes.
			finishTaskCall(map[string]any{"result": "run 1 ok"}, "call_1"),
			// Run 2: first ending call ("call_unobs") is completely unobserved by
			// analyticsPlugin; trailing ending call ("call_2") is observed on Run 2.
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call_unobs", Name: finishTaskToolName, Args: map[string]any{"result": "first"}}},
				{FunctionCall: &genai.FunctionCall{ID: "call_2", Name: finishTaskToolName, Args: map[string]any{"result": "second"}}},
			}}},
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(worker): %v", err)
	}

	root, err := agent.New(agent.Config{
		Name: "coordinator",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
				var heldRun1 []*session.Event
				for ev, runErr := range llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work") {
					if runErr != nil {
						yield(nil, runErr)
						return
					}
					if out := classifyTaskResponses(ev.Content); out.ends {
						heldRun1 = append(heldRun1, ev)
						continue
					}
					if !yield(ev, nil) {
						return
					}
				}
				for ev, runErr := range llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work") {
					if !yield(ev, runErr) || runErr != nil {
						return
					}
				}
				for _, ev := range heldRun1 {
					if !yield(ev, nil) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(coordinator): %v", err)
	}
	for _, runErr := range newTestRunner(t, root, earlier, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("runner.Run error: %v", runErr)
		}
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	starts := agentRowsOfType(rows, EventAgentStarting, "worker")
	ends := agentRowsOfType(rows, EventAgentCompleted, "worker")
	if len(starts) != 2 || len(ends) != 2 {
		t.Fatalf("got %d worker AGENT_STARTING and %d worker AGENT_COMPLETED rows, want 2 of each; events: %v", len(starts), len(ends), eventTypes(rows))
	}
	run1Span, run2Span := starts[0]["span_id"], starts[1]["span_id"]
	if got := ends[0]["span_id"]; got != run2Span {
		t.Errorf("first worker AGENT_COMPLETED span_id = %v, want Run 2 span %v (Run 1 span = %v)", got, run2Span, run1Span)
	}
	if got := ends[1]["span_id"]; got != run1Span {
		t.Errorf("second worker AGENT_COMPLETED span_id = %v, want Run 1 span %v (Run 2 span = %v)", got, run1Span, run2Span)
	}
}

func TestPlugin_FinishTask_ConcurrentCall0ReuseFanOut(t *testing.T) {
	t.Run("deterministic_two_overlapping_runs_three_waves_on_frozen_clock", func(t *testing.T) {
		p, requests, _ := setupTestPluginWithConfig(t, nil)
		fixedNow := time.Date(2026, 11, 19, 12, 0, 0, 0, time.UTC)
		ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return fixedNow })

		const waves = 3
		var responses []*model.LLMResponse
		for range waves * 2 {
			responses = append(responses, finishTaskCall(map[string]any{"result": "done"}, "call_0"))
		}
		worker, err := llmagent.New(llmagent.Config{
			Name:  "worker",
			Mode:  llmagent.ModeTask,
			Model: &scriptedLLM{responses: responses},
		})
		if err != nil {
			t.Fatalf("llmagent.New(worker): %v", err)
		}

		root, err := agent.New(agent.Config{
			Name: "fanout_parent",
			Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					nodeCtx := agent.PromoteWithDelta(ic, &agent.CommonContextDelta{})
					for range waves {
						// Start both Run A and Run B concurrently so both record "call_0"
						// while open on the same frozen clock, then yield Run B's completion
						// before Run A's buffered completion.
						nextA, stopA := iter.Pull2(llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work"))
						nextB, stopB := iter.Pull2(llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work"))
						var heldA, eventsB []*session.Event
						for {
							ev, runErr, ok := nextA()
							if !ok {
								break
							}
							if runErr != nil {
								stopA()
								stopB()
								yield(nil, runErr)
								return
							}
							if classifyTaskResponses(ev.Content).ends {
								heldA = append(heldA, ev)
								continue
							}
							if !yield(ev, nil) {
								stopA()
								stopB()
								return
							}
						}
						for {
							ev, runErr, ok := nextB()
							if !ok {
								break
							}
							if runErr != nil {
								stopA()
								stopB()
								yield(nil, runErr)
								return
							}
							eventsB = append(eventsB, ev)
						}
						stopA()
						stopB()
						for _, ev := range eventsB {
							if !yield(ev, nil) {
								return
							}
						}
						for _, ev := range heldA {
							if !yield(ev, nil) {
								return
							}
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(fanout_parent): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(ctx, "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error: %v", runErr)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		starts := agentRowsOfType(rows, EventAgentStarting, "worker")
		ends := agentRowsOfType(rows, EventAgentCompleted, "worker")
		if len(starts) != waves*2 || len(ends) != waves*2 {
			t.Fatalf("worker rows: %d AGENT_STARTING and %d AGENT_COMPLETED, want %d of each; events: %v", len(starts), len(ends), waves*2, eventTypes(rows))
		}
		for w := range waves {
			spanA := starts[w*2]["span_id"]
			spanB := starts[w*2+1]["span_id"]
			if got := ends[w*2]["span_id"]; got != spanB {
				t.Errorf("wave %d first AGENT_COMPLETED span_id = %v, want Run B span %v (Run A span = %v)", w, got, spanB, spanA)
			}
			if got := ends[w*2+1]["span_id"]; got != spanA {
				t.Errorf("wave %d second AGENT_COMPLETED span_id = %v, want Run A span %v (Run B span = %v)", w, got, spanA, spanB)
			}
		}
	})

	t.Run("eight_concurrent_workers_reusing_call_0_close_every_run", func(t *testing.T) {
		p, requests, _ := setupTestPluginWithConfig(t, nil)
		const (
			fanout = 8
			rounds = 10
			total  = fanout * rounds
		)
		fixedModel := &callbackLLM{
			name: "fixed-worker",
			fn: func(_ context.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
				return finishTaskCall(map[string]any{"result": "ok"}, "call_0"), nil
			},
		}
		worker, err := llmagent.New(llmagent.Config{
			Name:  "worker",
			Mode:  llmagent.ModeTask,
			Model: fixedModel,
		})
		if err != nil {
			t.Fatalf("llmagent.New(worker): %v", err)
		}

		root, err := agent.New(agent.Config{
			Name: "eight_way_parent",
			Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					nodeCtx := agent.PromoteWithDelta(ic, &agent.CommonContextDelta{})
					for range rounds {
						type item struct {
							ev  *session.Event
							err error
						}
						ch := make(chan item, fanout*4)
						var wg sync.WaitGroup
						for range fanout {
							wg.Go(func() {
								for ev, runErr := range llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work") {
									ch <- item{ev: ev, err: runErr}
								}
							})
						}
						wg.Wait()
						close(ch)
						for it := range ch {
							if !yield(it.ev, it.err) || it.err != nil {
								return
							}
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(eight_way_parent): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error: %v", runErr)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		starts := agentRowsOfType(rows, EventAgentStarting, "worker")
		ends := agentRowsOfType(rows, EventAgentCompleted, "worker")
		errs := agentRowsOfType(rows, EventAgentError, "worker")
		if len(starts) != total || len(ends) != total || len(errs) != 0 {
			t.Fatalf("worker rows: starts=%d ends=%d errors=%d, want %d/%d/0", len(starts), len(ends), len(errs), total, total)
		}
	})
}

func TestPlugin_FinishTask_MergedReusedLeadingShortCircuitedAndUniqueTrailingCallID(t *testing.T) {
	for _, frozen := range []bool{false, true} {
		name := "wall_clock"
		if frozen {
			name = "frozen_clock"
		}
		t.Run(name, func(t *testing.T) {
			p, requests, _ := setupTestPluginWithConfig(t, nil)
			var runPhase atomic.Int32
			earlier, err := baseplugin.New(baseplugin.Config{
				Name: "short_circuit_run2_call0",
				BeforeToolCallback: func(ctx agent.Context, tl tool.Tool, _ map[string]any) (map[string]any, error) {
					if runPhase.Load() == 2 && tl != nil && tl.Name() == finishTaskToolName && ctx.FunctionCallID() == "call_0" {
						return map[string]any{"result": "Task completed."}, nil
					}
					return nil, nil
				},
				AfterToolCallback: func(ctx agent.Context, tl tool.Tool, _, res map[string]any, _ error) (map[string]any, error) {
					if runPhase.Load() == 2 && tl != nil && tl.Name() == finishTaskToolName && ctx.FunctionCallID() == "call_0" {
						return res, nil
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New: %v", err)
			}

			worker, err := llmagent.New(llmagent.Config{
				Name: "worker",
				Mode: llmagent.ModeTask,
				Model: &scriptedLLM{responses: []*model.LLMResponse{
					// Run 1: observed completion "call_0", buffered until after Run 2 completes.
					finishTaskCall(map[string]any{"result": "run 1"}, "call_0"),
					// Run 2: leading "call_0" is answered by earlier plugin (unobserved by analyticsPlugin),
					// while trailing "call_1" is observed on Run 2.
					{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{ID: "call_0", Name: finishTaskToolName, Args: map[string]any{"result": "r2-first"}}},
						{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: finishTaskToolName, Args: map[string]any{"result": "r2-second"}}},
					}}},
				}},
			})
			if err != nil {
				t.Fatalf("llmagent.New(worker): %v", err)
			}

			root, err := agent.New(agent.Config{
				Name: "coordinator",
				Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						nodeCtx := agent.PromoteWithDelta(ic, &agent.CommonContextDelta{})
						runPhase.Store(1)
						next1, stop1 := iter.Pull2(llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work"))
						defer stop1()
						var heldRun1 []*session.Event
						for {
							ev, runErr, ok := next1()
							if !ok {
								break
							}
							if runErr != nil {
								yield(nil, runErr)
								return
							}
							if classifyTaskResponses(ev.Content).ends {
								heldRun1 = append(heldRun1, ev)
								break
							}
							if !yield(ev, nil) {
								return
							}
						}

						runPhase.Store(2)
						for ev, runErr := range llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work") {
							if !yield(ev, runErr) || runErr != nil {
								return
							}
						}
						for {
							ev, runErr, ok := next1()
							if !ok {
								break
							}
							if !yield(ev, runErr) || runErr != nil {
								return
							}
						}
						for _, ev := range heldRun1 {
							if !yield(ev, nil) {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(coordinator): %v", err)
			}

			ctx := t.Context()
			if frozen {
				fixed := time.Date(2026, 11, 19, 13, 0, 0, 0, time.UTC)
				ctx = platform.WithTimeProvider(ctx, func() time.Time { return fixed })
			}
			for _, runErr := range newTestRunner(t, root, earlier, p).Run(ctx, "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("runner.Run error: %v", runErr)
				}
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			starts := agentRowsOfType(rows, EventAgentStarting, "worker")
			ends := agentRowsOfType(rows, EventAgentCompleted, "worker")
			if len(starts) != 2 || len(ends) != 2 {
				t.Fatalf("got %d worker AGENT_STARTING and %d worker AGENT_COMPLETED rows, want 2 of each; events: %v", len(starts), len(ends), eventTypes(rows))
			}
			run1Span, run2Span := starts[0]["span_id"], starts[1]["span_id"]
			if got := ends[0]["span_id"]; got != run2Span {
				t.Errorf("first worker AGENT_COMPLETED span_id = %v, want Run 2 span %v (swapped with Run 1 span %v)", got, run2Span, run1Span)
			}
			if got := ends[1]["span_id"]; got != run1Span {
				t.Errorf("second worker AGENT_COMPLETED span_id = %v, want Run 1 span %v (swapped with Run 2 span %v)", got, run1Span, run2Span)
			}
		})
	}
}

func TestPlugin_FinishTask_CoordinatorBufferedFailureDoesNotRetireNewerWorkerAttempt(t *testing.T) {
	for _, frozen := range []bool{false, true} {
		name := "advancing_clock"
		if frozen {
			name = "frozen_clock"
		}
		t.Run(name, func(t *testing.T) {
			p, requests, _ := setupTestPluginWithConfig(t, nil)
			var (
				workerTurn atomic.Int32
				promoteNow atomic.Bool
			)
			promoter, err := baseplugin.New(baseplugin.Config{
				Name: "promote_worker_b",
				AfterToolCallback: func(_ agent.Context, tl tool.Tool, _, _ map[string]any, toolErr error) (map[string]any, error) {
					if tl != nil && tl.Name() == finishTaskToolName && toolErr != nil && promoteNow.Load() {
						return map[string]any{"result": "Task completed."}, nil
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New(promoter): %v", err)
			}

			workerOrdered, err := llmagent.New(llmagent.Config{
				Name: "worker",
				Mode: llmagent.ModeTask,
				BeforeToolCallbacks: []llmagent.BeforeToolCallback{
					func(_ agent.Context, _ tool.Tool, _ map[string]any) (map[string]any, error) {
						switch workerTurn.Add(1) {
						case 1:
							return nil, errors.New("worker A first failure")
						case 2:
							return nil, errors.New("worker B promoted failure")
						default:
							return nil, nil
						}
					},
				},
				Model: &scriptedLLM{responses: []*model.LLMResponse{
					finishTaskCall(map[string]any{"result": "a-fail"}, "call_0"),
					finishTaskCall(map[string]any{"result": "b-promoted"}, "call_0"),
					finishTaskCall(map[string]any{"result": "a-retry-ok"}, "call_0"),
				}},
			})
			if err != nil {
				t.Fatalf("llmagent.New(workerOrdered): %v", err)
			}

			root, err := agent.New(agent.Config{
				Name: "coordinator",
				Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						nodeCtx := agent.PromoteWithDelta(ic, &agent.CommonContextDelta{})
						nextA, stopA := iter.Pull2(llmagent.RunLLMAgentAsNode(workerOrdered, nodeCtx, "work"))
						defer stopA()

						var heldAFailure *session.Event
						for {
							ev, runErr, ok := nextA()
							if !ok {
								break
							}
							if runErr != nil {
								yield(nil, runErr)
								return
							}
							out := classifyTaskResponses(ev.Content)
							if len(out.retiredCallIDs) > 0 {
								heldAFailure = ev
								break
							}
							if !yield(ev, nil) {
								return
							}
						}

						promoteNow.Store(true)
						var eventsB []*session.Event
						for ev, runErr := range llmagent.RunLLMAgentAsNode(workerOrdered, nodeCtx, "work") {
							if runErr != nil {
								yield(nil, runErr)
								return
							}
							eventsB = append(eventsB, ev)
						}
						promoteNow.Store(false)

						if heldAFailure != nil {
							if !yield(heldAFailure, nil) {
								return
							}
						}
						for _, ev := range eventsB {
							if !yield(ev, nil) {
								return
							}
						}

						for {
							ev, runErr, ok := nextA()
							if !ok {
								break
							}
							if !yield(ev, runErr) || runErr != nil {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(coordinator): %v", err)
			}

			ctx := t.Context()
			if frozen {
				fixed := time.Date(2026, 11, 19, 13, 15, 0, 0, time.UTC)
				ctx = platform.WithTimeProvider(ctx, func() time.Time { return fixed })
			} else {
				var tick atomic.Int64
				base := time.Date(2026, 11, 19, 13, 15, 0, 0, time.UTC)
				ctx = platform.WithTimeProvider(ctx, func() time.Time {
					return base.Add(time.Duration(tick.Add(1)) * time.Millisecond)
				})
			}

			for _, runErr := range newTestRunner(t, root, p, promoter).Run(ctx, "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("runner.Run error: %v", runErr)
				}
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			starts := agentRowsOfType(rows, EventAgentStarting, "worker")
			ends := agentRowsOfType(rows, EventAgentCompleted, "worker")
			if len(starts) != 2 || len(ends) != 2 {
				t.Fatalf("got %d worker AGENT_STARTING and %d worker AGENT_COMPLETED rows, want 2 of each; events: %v", len(starts), len(ends), eventTypes(rows))
			}
			spanA, spanB := starts[0]["span_id"], starts[1]["span_id"]
			if got := ends[0]["span_id"]; got != spanB {
				t.Errorf("first worker AGENT_COMPLETED span_id = %v, want Worker B span %v (Worker A span = %v)", got, spanB, spanA)
			}
			if got := ends[1]["span_id"]; got != spanA {
				t.Errorf("second worker AGENT_COMPLETED span_id = %v, want Worker A span %v (Worker B span = %v)", got, spanA, spanB)
			}
		})
	}
}

func TestPlugin_FinishTask_MergedParallelToolTimestampPredatingAfterToolRecordedAt(t *testing.T) {
	p, requests, _ := setupTestPluginWithConfig(t, nil)

	type emptyIn struct{}
	fastDone := make(chan struct{})
	var synchronizeRun2 atomic.Bool
	fastTool, err := functiontool.New(functiontool.Config{
		Name:        "fast_lookup",
		Description: "returns immediately before finish_task finishes",
	}, func(_ agent.Context, _ emptyIn) (map[string]any, error) {
		return map[string]any{"ok": true}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New(fast_lookup): %v", err)
	}

	orderGate, err := baseplugin.New(baseplugin.Config{
		Name: "order_parallel_tools",
		BeforeToolCallback: func(_ agent.Context, tl tool.Tool, _ map[string]any) (map[string]any, error) {
			if synchronizeRun2.Load() && tl != nil && tl.Name() == finishTaskToolName {
				<-fastDone
				time.Sleep(5 * time.Millisecond)
			}
			return nil, nil
		},
		AfterToolCallback: func(_ agent.Context, tl tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
			if synchronizeRun2.Load() && tl != nil && tl.Name() == "fast_lookup" {
				go func() {
					time.Sleep(2 * time.Millisecond)
					close(fastDone)
				}()
			}
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("baseplugin.New(orderGate): %v", err)
	}

	worker, err := llmagent.New(llmagent.Config{
		Name:  "worker",
		Mode:  llmagent.ModeTask,
		Tools: []tool.Tool{fastTool},
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			finishTaskCall(map[string]any{"result": "run 1"}, "call_0"),
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "call_fast", Name: "fast_lookup", Args: map[string]any{}}},
				{FunctionCall: &genai.FunctionCall{ID: "call_0", Name: finishTaskToolName, Args: map[string]any{"result": "run 2"}}},
			}}},
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(worker): %v", err)
	}

	root, err := agent.New(agent.Config{
		Name: "coordinator",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				nodeCtx := agent.PromoteWithDelta(ic, &agent.CommonContextDelta{})
				next1, stop1 := iter.Pull2(llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work"))
				defer stop1()
				var heldRun1 []*session.Event
				for {
					ev, runErr, ok := next1()
					if !ok {
						break
					}
					if runErr != nil {
						yield(nil, runErr)
						return
					}
					if classifyTaskResponses(ev.Content).ends {
						heldRun1 = append(heldRun1, ev)
						break
					}
					if !yield(ev, nil) {
						return
					}
				}

				synchronizeRun2.Store(true)
				for ev, runErr := range llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work") {
					if !yield(ev, runErr) || runErr != nil {
						return
					}
				}
				synchronizeRun2.Store(false)

				for {
					ev, runErr, ok := next1()
					if !ok {
						break
					}
					if !yield(ev, runErr) || runErr != nil {
						return
					}
				}
				for _, ev := range heldRun1 {
					if !yield(ev, nil) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(coordinator): %v", err)
	}

	var tick atomic.Int64
	base := time.Date(2026, 11, 19, 14, 0, 0, 0, time.UTC)
	ctx := platform.WithTimeProvider(t.Context(), func() time.Time {
		return base.Add(time.Duration(tick.Add(1)) * time.Millisecond)
	})
	for _, runErr := range newTestRunner(t, root, orderGate, p).Run(ctx, "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("runner.Run error: %v", runErr)
		}
	}

	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	starts := agentRowsOfType(rows, EventAgentStarting, "worker")
	ends := agentRowsOfType(rows, EventAgentCompleted, "worker")
	if len(starts) != 2 || len(ends) != 2 {
		t.Fatalf("got %d worker AGENT_STARTING and %d worker AGENT_COMPLETED rows, want 2 of each; events: %v", len(starts), len(ends), eventTypes(rows))
	}
	run1Span, run2Span := starts[0]["span_id"], starts[1]["span_id"]
	if got := ends[0]["span_id"]; got != run2Span {
		t.Errorf("first worker AGENT_COMPLETED span_id = %v, want Run 2 span %v (Run 1 span = %v)", got, run2Span, run1Span)
	}
	if got := ends[1]["span_id"]; got != run1Span {
		t.Errorf("second worker AGENT_COMPLETED span_id = %v, want Run 1 span %v (Run 2 span = %v)", got, run1Span, run2Span)
	}
}

func TestPlugin_FinishTask_NoteTaskStartAndOverruledEndingGuards(t *testing.T) {
	t.Run("noteTaskStart attributes finish_task when earlier plugin short-circuits AfterToolCallback", func(t *testing.T) {
		p, requests, _ := setupTestPluginWithConfig(t, nil)
		shortAfter, err := baseplugin.New(baseplugin.Config{
			Name: "short_circuit_after_tool",
			AfterToolCallback: func(_ agent.Context, tl tool.Tool, _, res map[string]any, _ error) (map[string]any, error) {
				if tl != nil && tl.Name() == finishTaskToolName {
					return res, nil
				}
				return nil, nil
			},
		})
		if err != nil {
			t.Fatalf("baseplugin.New(shortAfter): %v", err)
		}

		worker, err := llmagent.New(llmagent.Config{
			Name: "worker",
			Mode: llmagent.ModeTask,
			Model: &scriptedLLM{responses: []*model.LLMResponse{
				finishTaskCall(map[string]any{"result": "run 1"}, "call_0"),
				finishTaskCall(map[string]any{"result": "run 2"}, "call_0"),
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(worker): %v", err)
		}

		root, err := agent.New(agent.Config{
			Name: "coordinator",
			Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					nodeCtx := agent.PromoteWithDelta(ic, &agent.CommonContextDelta{})
					next1, stop1 := iter.Pull2(llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work"))
					next2, stop2 := iter.Pull2(llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work"))
					defer stop1()
					defer stop2()

					var held1, held2 []*session.Event
					for {
						ev, runErr, ok := next1()
						if !ok {
							break
						}
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if classifyTaskResponses(ev.Content).ends {
							held1 = append(held1, ev)
							break
						}
						if !yield(ev, nil) {
							return
						}
					}
					for {
						ev, runErr, ok := next2()
						if !ok {
							break
						}
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if classifyTaskResponses(ev.Content).ends {
							held2 = append(held2, ev)
							break
						}
						if !yield(ev, nil) {
							return
						}
					}
					for _, ev := range held1 {
						if !yield(ev, nil) {
							return
						}
					}
					for _, ev := range held2 {
						if !yield(ev, nil) {
							return
						}
					}
					for {
						ev, runErr, ok := next1()
						if !ok {
							break
						}
						if !yield(ev, runErr) || runErr != nil {
							return
						}
					}
					for {
						ev, runErr, ok := next2()
						if !ok {
							break
						}
						if !yield(ev, runErr) || runErr != nil {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(coordinator): %v", err)
		}

		fixed := time.Date(2026, 11, 19, 14, 30, 0, 0, time.UTC)
		ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return fixed })
		for _, runErr := range newTestRunner(t, root, shortAfter, p).Run(ctx, "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error: %v", runErr)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		starts := agentRowsOfType(rows, EventAgentStarting, "worker")
		ends := agentRowsOfType(rows, EventAgentCompleted, "worker")
		if len(starts) != 2 || len(ends) != 2 {
			t.Fatalf("got %d starts and %d ends, want 2 of each", len(starts), len(ends))
		}
		if ends[0]["span_id"] != starts[0]["span_id"] || ends[1]["span_id"] != starts[1]["span_id"] {
			t.Errorf("AGENT_COMPLETED spans = [%v, %v], want [%v, %v]", ends[0]["span_id"], ends[1]["span_id"], starts[0]["span_id"], starts[1]["span_id"])
		}
	})

	t.Run("overruledEndingCallIDs retires trailing completion behind empty-ID failure across concurrent runs on frozen clock", func(t *testing.T) {
		p, requests, _ := setupTestPluginWithConfig(t, nil)
		worker, err := llmagent.New(llmagent.Config{
			Name: "worker",
			Mode: llmagent.ModeTask,
			BeforeToolCallbacks: []llmagent.BeforeToolCallback{
				func(tc agent.Context, _ tool.Tool, _ map[string]any) (map[string]any, error) {
					if tc.FunctionCallID() == "call_empty_fail" {
						return nil, errors.New("first part failed")
					}
					return nil, nil
				},
			},
			Model: &scriptedLLM{responses: []*model.LLMResponse{
				{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "call_empty_fail", Name: finishTaskToolName, Args: map[string]any{"result": "bad"}}},
					{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: finishTaskToolName, Args: map[string]any{"result": "overruled"}}},
				}}},
				finishTaskCall(map[string]any{"result": "run 2 ok"}, "call_1"),
				finishTaskCall(map[string]any{"result": "run 1 retry ok"}, "call_2"),
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(worker): %v", err)
		}

		root, err := agent.New(agent.Config{
			Name: "coordinator",
			Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					nodeCtx := agent.PromoteWithDelta(ic, &agent.CommonContextDelta{})
					next1, stop1 := iter.Pull2(llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work"))
					next2, stop2 := iter.Pull2(llmagent.RunLLMAgentAsNode(worker, nodeCtx, "work"))
					defer stop1()
					defer stop2()

					for {
						ev, runErr, ok := next1()
						if !ok {
							break
						}
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev.Content != nil {
							sawEmptyFail := false
							for _, part := range ev.Content.Parts {
								if part != nil && part.FunctionResponse != nil && part.FunctionResponse.ID == "call_empty_fail" {
									part.FunctionResponse.ID = ""
									sawEmptyFail = true
								}
							}
							if sawEmptyFail {
								if !yield(ev, nil) {
									return
								}
								break
							}
						}
						if !yield(ev, nil) {
							return
						}
					}

					for {
						ev, runErr, ok := next2()
						if !ok {
							break
						}
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if classifyTaskResponses(ev.Content).ends {
							ev.Actions = session.EventActions{}
						}
						if !yield(ev, nil) {
							return
						}
					}

					for {
						ev, runErr, ok := next1()
						if !ok {
							break
						}
						if !yield(ev, runErr) || runErr != nil {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(coordinator): %v", err)
		}

		fixed := time.Date(2026, 11, 19, 14, 45, 0, 0, time.UTC)
		ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return fixed })
		for _, runErr := range newTestRunner(t, root, p).Run(ctx, "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error: %v", runErr)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		starts := agentRowsOfType(rows, EventAgentStarting, "worker")
		ends := agentRowsOfType(rows, EventAgentCompleted, "worker")
		if len(starts) != 2 || len(ends) != 2 {
			t.Fatalf("got %d starts and %d ends, want 2 of each", len(starts), len(ends))
		}
		run1Span, run2Span := starts[0]["span_id"], starts[1]["span_id"]
		if got := ends[0]["span_id"]; got != run2Span {
			t.Errorf("first AGENT_COMPLETED span_id = %v, want Run 2 span %v (unretired overruled call_1 on Run 1 stole completion: %v)", got, run2Span, run1Span)
		}
		if got := ends[1]["span_id"]; got != run1Span {
			t.Errorf("second AGENT_COMPLETED span_id = %v, want Run 1 span %v", got, run1Span)
		}
	})
}
