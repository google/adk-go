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

package tool_test

import (
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// TestSequentialRunner_SurvivesWrappers checks that WithConfirmation and
// WithToolNamePrefix, alone and stacked, answer RunsSequentially with the
// value of the function tool they wrap.
func TestSequentialRunner_SurvivesWrappers(t *testing.T) {
	for _, tc := range []struct {
		name       string
		sequential bool
	}{
		{name: "sequential", sequential: true},
		{name: "plain", sequential: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ft, err := functiontool.New(
				functiontool.Config{Name: "a", Description: "a", Sequential: tc.sequential},
				func(agent.Context, struct{}) (map[string]any, error) { return nil, nil })
			if err != nil {
				t.Fatalf("functiontool.New() error = %v", err)
			}
			base := tool.AsToolset(ft)

			if got := runsSequentially(ft); got != tc.sequential {
				t.Errorf("function tool RunsSequentially = %v, want %v", got, tc.sequential)
			}
			if got := runsSequentially(onlyTool(t, tool.WithConfirmation(base, false, nil))); got != tc.sequential {
				t.Errorf("WithConfirmation RunsSequentially = %v, want %v", got, tc.sequential)
			}
			if got := runsSequentially(onlyTool(t, tool.WithToolNamePrefix(base, "p"))); got != tc.sequential {
				t.Errorf("WithToolNamePrefix RunsSequentially = %v, want %v", got, tc.sequential)
			}
			prefixedConfirmed := tool.WithToolNamePrefix(tool.WithConfirmation(base, false, nil), "p")
			if got := runsSequentially(onlyTool(t, prefixedConfirmed)); got != tc.sequential {
				t.Errorf("prefix over confirmation RunsSequentially = %v, want %v", got, tc.sequential)
			}
		})
	}
}

// runsSequentially reports whether tl is a tool.SequentialRunner that
// answers true.
func runsSequentially(tl tool.Tool) bool {
	s, ok := tl.(tool.SequentialRunner)
	return ok && s.RunsSequentially()
}
