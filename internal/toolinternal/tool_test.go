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

package toolinternal_test

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/workflowinternal"
)

// TestResponseDeferrer_Contract verifies that TaskAgentTool
// implements ResponseDeferrer
func TestResponseDeferrer_Contract(t *testing.T) {
	t.Parallel()

	t.Run("TaskAgentTool implements ResponseDeferrer and returns true", func(t *testing.T) {
		t.Parallel()
		a, err := llmagent.New(llmagent.Config{
			Name: "doer",
			Mode: llmagent.ModeTask,
		})
		if err != nil {
			t.Fatal(err)
		}
		tool, err := workflowinternal.NewTaskAgentTool(a)
		if err != nil {
			t.Fatal(err)
		}
		d, ok := tool.(toolinternal.ResponseDeferrer)
		if !ok {
			t.Fatalf("TaskAgentTool does not implement toolinternal.ResponseDeferrer")
		}
		if !d.DefersResponse() {
			t.Errorf("TaskAgentTool.DefersResponse() = false, want true")
		}
	})

	t.Run("plain non-deferring tool does NOT implement ResponseDeferrer", func(t *testing.T) {
		t.Parallel()
		var tool toolinternal.FunctionTool = &nonDeferringTool{}
		if _, ok := tool.(toolinternal.ResponseDeferrer); ok {
			t.Errorf("nonDeferringTool unexpectedly implements ResponseDeferrer")
		}
	})
}

// nonDeferringTool is a minimal toolinternal.FunctionTool stub used
// only to assert that tools without an explicit DefersResponse method
// are NOT assignable to toolinternal.ResponseDeferrer.
type nonDeferringTool struct{}

func (*nonDeferringTool) Name() string        { return "non_deferring" }
func (*nonDeferringTool) Description() string { return "" }
func (*nonDeferringTool) IsLongRunning() bool { return false }
func (*nonDeferringTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: "non_deferring"}
}

func (*nonDeferringTool) Run(_ agent.Context, _ any) (map[string]any, error) {
	return nil, nil
}
