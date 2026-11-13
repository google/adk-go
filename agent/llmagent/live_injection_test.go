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

package llmagent

import (
	"testing"

	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

func newTestTool(t *testing.T, name string) tool.Tool {
	t.Helper()
	created, err := functiontool.New(functiontool.Config{Name: name}, dummyHandler)
	if err != nil {
		t.Fatalf("functiontool.New(%q) error = %v", name, err)
	}
	return created
}

func newAgentWithTools(t *testing.T, tools ...tool.Tool) *llmAgent {
	t.Helper()
	created, err := New(Config{Name: "test_agent", Tools: tools})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	a, ok := created.(*llmAgent)
	if !ok {
		t.Fatalf("New() returned %T, want *llmAgent", created)
	}
	return a
}

func TestLiveInjection_SkipsWhenToolAlreadyPresent(t *testing.T) {
	a := newAgentWithTools(t, newTestTool(t, "task_completed"))

	got := a.liveInjection(llminternal.LiveInjection{
		Tool:              newTestTool(t, "task_completed"),
		InstructionSuffix: "suffix",
	})
	if got != nil {
		t.Errorf("liveInjection() = %+v, want nil because the agent already has the tool", got)
	}
}

func TestLiveInjection_ReturnsAdditionsForNewTool(t *testing.T) {
	a := newAgentWithTools(t, newTestTool(t, "other_tool"))
	injected := newTestTool(t, "task_completed")

	got := a.liveInjection(llminternal.LiveInjection{
		Tool:              injected,
		InstructionSuffix: "suffix",
	})
	if got == nil {
		t.Fatal("liveInjection() = nil, want the additions")
	}
	if got.Tool != injected {
		t.Errorf("liveInjection().Tool = %v, want %v", got.Tool, injected)
	}
	if got.InstructionSuffix != "suffix" {
		t.Errorf("liveInjection().InstructionSuffix = %q, want %q", got.InstructionSuffix, "suffix")
	}
}

func TestLiveInjection_IgnoresEmptyInjection(t *testing.T) {
	a := newAgentWithTools(t)

	if got := a.liveInjection(llminternal.LiveInjection{}); got != nil {
		t.Errorf("liveInjection(zero) = %+v, want nil", got)
	}
}

func TestLiveInjection_AcceptsSuffixWithoutTool(t *testing.T) {
	a := newAgentWithTools(t)

	got := a.liveInjection(llminternal.LiveInjection{InstructionSuffix: "suffix"})
	if got == nil || got.Tool != nil || got.InstructionSuffix != "suffix" {
		t.Errorf("liveInjection(suffix only) = %+v, want the suffix and no tool", got)
	}
}

func TestLiveInjection_HandlesNilToolInState(t *testing.T) {
	a := &llmAgent{}
	a.State.Tools = []tool.Tool{nil}

	got := a.liveInjection(llminternal.LiveInjection{Tool: newTestTool(t, "task_completed")})
	if got == nil {
		t.Fatal("liveInjection() = nil, want the additions")
	}
}

func TestLiveFlow_CarriesInjection(t *testing.T) {
	a := newAgentWithTools(t)
	injected := newTestTool(t, "task_completed")

	f := a.liveFlow(llminternal.LiveInjection{Tool: injected, InstructionSuffix: "suffix"})
	if f.LiveInjection == nil {
		t.Fatal("liveFlow().LiveInjection = nil, want the additions")
	}
	if f.LiveInjection.Tool != injected || f.LiveInjection.InstructionSuffix != "suffix" {
		t.Errorf("liveFlow().LiveInjection = %+v, want the passed additions", f.LiveInjection)
	}
}

func TestLiveFlow_WithoutAdditionsHasNoInjection(t *testing.T) {
	a := newAgentWithTools(t)

	if got := a.liveFlow(llminternal.LiveInjection{}).LiveInjection; got != nil {
		t.Errorf("liveFlow(zero).LiveInjection = %+v, want nil", got)
	}
}
