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
	"errors"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/retryandreflect"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// TestRunner_RetryAndReflectOrder runs a failing tool with retryandreflect
// registered before and after the analytics plugin. Either way the call has
// exactly one terminal row, TOOL_ERROR, placed on the TOOL_STARTING span.
func TestRunner_RetryAndReflectOrder(t *testing.T) {
	for _, tc := range []struct {
		name        string
		reflectLast bool
	}{
		{name: "reflect before analytics"},
		{name: "reflect after analytics", reflectLast: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			rr, err := retryandreflect.New()
			if err != nil {
				t.Fatalf("retryandreflect.New: %v", err)
			}
			errTool, err := functiontool.New(functiontool.Config{Name: "err_tool", Description: "fails"},
				func(agent.Context, struct{}) (struct{}, error) {
					return struct{}{}, errors.New("tool execution failed")
				})
			if err != nil {
				t.Fatalf("functiontool.New: %v", err)
			}
			bot, err := llmagent.New(llmagent.Config{
				Name: "bot",
				Model: &scriptedLLM{responses: []*model.LLMResponse{
					{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "err_tool", Args: map[string]any{}}}}}},
					{Content: textContent("model", "done")},
				}},
				Tools: []tool.Tool{errTool},
			})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			plugins := []*baseplugin.Plugin{rr, p}
			if tc.reflectLast {
				plugins = []*baseplugin.Plugin{p, rr}
			}
			r := newTestRunner(t, bot, plugins...)
			for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("r.Run: %v", err)
				}
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
				t.Errorf("got %d TOOL_COMPLETED rows, want 0", n)
			}
			start := onlyRowOfType(t, rows, EventToolStarting)
			row := onlyRowOfType(t, rows, EventToolError)
			if got, want := placeOf(row), placeOf(start); got != want {
				t.Errorf("TOOL_ERROR is at %+v, want %+v", got, want)
			}
			if row["status"] != "ERROR" {
				t.Errorf("TOOL_ERROR status = %v, want ERROR", row["status"])
			}
			if row["error_message"] != "tool execution failed" {
				t.Errorf("TOOL_ERROR error_message = %v, want %q", row["error_message"], "tool execution failed")
			}
			if got, want := functionCallIDOf(t, row), functionCallIDOf(t, start); got != want {
				t.Errorf("TOOL_ERROR function_call_id = %q, want %q", got, want)
			}
			mustLatency(t, row)
		})
	}
}

// TestReflectAndRetryAnswerErrorMessage checks the error_message of the
// TOOL_ERROR row written for a reflect answer, with and without error_details.
func TestReflectAndRetryAnswerErrorMessage(t *testing.T) {
	for _, tc := range []struct {
		name   string
		answer map[string]any
		want   string
	}{
		{"details", map[string]any{"error_details": "boom", "error_type": "*errors.errorString"}, "boom"},
		{"error type", map[string]any{"error_details": "", "error_type": "*errors.errorString"}, "*errors.errorString"},
		{"fixed message", map[string]any{}, "tool call failed and was handled by the reflect and retry plugin"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			toolCtx := withFunctionCall(mCtx, "call-reflect")
			lookup := &fakeTool{name: "lookup"}
			args := map[string]any{}
			tc.answer["response_type"] = reflectAndRetryResponseType
			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			errs.add(p.BeforeToolCallback()(toolCtx, lookup, args))
			errs.add(p.AfterToolCallback()(toolCtx, lookup, args, tc.answer, nil))
			errs.add(p.AfterAgentCallback()(mCtx))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
			row := onlyRowOfType(t, rows, EventToolError)
			if row["error_message"] != tc.want {
				t.Errorf("error_message = %v, want %q", row["error_message"], tc.want)
			}
			if got, want := placeOf(row), placeOf(onlyRowOfType(t, rows, EventToolStarting)); got != want {
				t.Errorf("TOOL_ERROR is at %+v, want %+v", got, want)
			}
			if functionCallIDOf(t, row) != "call-reflect" {
				t.Errorf("TOOL_ERROR function_call_id = %q, want call-reflect", functionCallIDOf(t, row))
			}
			mustLatency(t, row)
		})
	}
}

func fnCallResponse(id, name string) *model.LLMResponse {
	return &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: id, Name: name, Args: map[string]any{}}}}}}
}

func newLookupTool(t *testing.T, fail bool) tool.Tool {
	t.Helper()
	lookup, err := functiontool.New(functiontool.Config{Name: "lookup", Description: "looks up"},
		func(agent.Context, struct{}) (struct{}, error) {
			if fail {
				return struct{}{}, errors.New("lookup failed")
			}
			return struct{}{}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}
	return lookup
}

// TestRunner_ReusedCallIDAfterUnknownTool checks that a successful call still
// gets TOOL_COMPLETED when an earlier plugin answers its BeforeToolCallback
// and a call of a missing tool used the same function call ID before.
func TestRunner_ReusedCallIDAfterUnknownTool(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	early, err := baseplugin.New(baseplugin.Config{
		Name: "early",
		BeforeToolCallback: func(agent.Context, tool.Tool, map[string]any) (map[string]any, error) {
			return map[string]any{"ok": true}, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New: %v", err)
	}
	bot, err := llmagent.New(llmagent.Config{
		Name: "bot",
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			fnCallResponse("call_0", "ghost"),
			fnCallResponse("call_0", "lookup"),
			{Content: textContent("model", "done")},
		}},
		Tools: []tool.Tool{newLookupTool(t, false)},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r := newTestRunner(t, bot, early, p)
	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventToolCompleted)); n != 1 {
		t.Errorf("got %d TOOL_COMPLETED rows, want 1", n)
	}
	if n := len(rowsOfType(rows, EventToolError)); n != 1 {
		t.Errorf("got %d TOOL_ERROR rows, want 1 (the missing tool)", n)
	}
}

// TestRunner_AgentOnToolErrorCallback checks that a failure recovered by the
// agent's own OnToolErrorCallbacks has exactly one terminal row, TOOL_ERROR.
func TestRunner_AgentOnToolErrorCallback(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	bot, err := llmagent.New(llmagent.Config{
		Name: "bot",
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			fnCallResponse("call_0", "lookup"),
			{Content: textContent("model", "done")},
		}},
		Tools: []tool.Tool{newLookupTool(t, true)},
		OnToolErrorCallbacks: []llmagent.OnToolErrorCallback{
			func(agent.Context, tool.Tool, map[string]any, error) (map[string]any, error) {
				return map[string]any{"recovered": true}, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r := newTestRunner(t, bot, p)
	for _, err := range r.Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("r.Run: %v", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
		t.Errorf("got %d TOOL_COMPLETED rows, want 0", n)
	}
	row := onlyRowOfType(t, rows, EventToolError)
	if got, want := placeOf(row), placeOf(onlyRowOfType(t, rows, EventToolStarting)); got != want {
		t.Errorf("TOOL_ERROR is at %+v, want %+v", got, want)
	}
	if row["error_message"] != "lookup failed" {
		t.Errorf("error_message = %v, want %q", row["error_message"], "lookup failed")
	}
}
