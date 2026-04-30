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

package task_test

import (
	"context"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/llmagent/task"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// stubToolContext implements tool.Context just enough for the task tools
// to populate Actions and read FunctionCallID.
type stubToolContext struct {
	context.Context
	actions  *session.EventActions
	callID   string
	branch   string
	stateMap map[string]any
}

func (c *stubToolContext) AgentName() string                             { return "stub_agent" }
func (c *stubToolContext) AppName() string                               { return "stub_app" }
func (c *stubToolContext) Branch() string                                { return c.branch }
func (c *stubToolContext) InvocationID() string                          { return "inv-stub" }
func (c *stubToolContext) SessionID() string                             { return "sess-stub" }
func (c *stubToolContext) UserID() string                                { return "u" }
func (c *stubToolContext) UserContent() *genai.Content                   { return nil }
func (c *stubToolContext) Actions() *session.EventActions                { return c.actions }
func (c *stubToolContext) FunctionCallID() string                        { return c.callID }
func (c *stubToolContext) Artifacts() agent.Artifacts                    { return nil }
func (c *stubToolContext) ReadonlyState() session.ReadonlyState          { return nil }
func (c *stubToolContext) State() session.State                          { return nil }
func (c *stubToolContext) ToolConfirmation() any                         { return nil }
func (c *stubToolContext) RequestConfirmation(string, any) error         { return nil }
func (c *stubToolContext) SearchMemory(context.Context, string) (*memory.SearchResponse, error) {
	return nil, nil
}

func mustAgent(t *testing.T, name, desc string) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{Name: name, Description: desc})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	return a
}

func TestNewRequestTaskTool_NameAndDescription(t *testing.T) {
	taskAgent := mustAgent(t, "summarizer", "summarizes documents")
	tt, err := task.NewRequestTaskTool(taskAgent)
	if err != nil {
		t.Fatalf("NewRequestTaskTool: %v", err)
	}
	if tt.Name() != "summarizer" {
		t.Errorf("Name = %q, want summarizer", tt.Name())
	}
	if tt.Description() == "" {
		t.Error("Description should not be empty")
	}
}

func TestNewFinishTaskTool_Name(t *testing.T) {
	taskAgent := mustAgent(t, "summarizer", "")
	tt, err := task.NewFinishTaskTool(taskAgent)
	if err != nil {
		t.Fatalf("NewFinishTaskTool: %v", err)
	}
	if tt.Name() != "finish_task" {
		t.Errorf("Name = %q, want finish_task", tt.Name())
	}
}

func TestNewRequestTaskTool_NilAgentRejected(t *testing.T) {
	if _, err := task.NewRequestTaskTool(nil); err == nil {
		t.Error("expected error for nil agent")
	}
}

func TestNewFinishTaskTool_NilAgentRejected(t *testing.T) {
	if _, err := task.NewFinishTaskTool(nil); err == nil {
		t.Error("expected error for nil agent")
	}
}

func TestModeConstants(t *testing.T) {
	if task.MultiTurn == task.SingleTurn {
		t.Error("MultiTurn and SingleTurn should differ")
	}
}

// Confirm the data types referenced from session match the task package's
// expectations so wire-compatibility holds.
func TestSessionDataShapes(t *testing.T) {
	_ = session.TaskRequest{AgentName: "x", Input: map[string]any{"goal": "y"}}
	_ = session.TaskResult{AgentName: "x", Output: map[string]any{"result": "z"}}
}

var _ = tool.Tool(nil) // silence unused import
