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

package generatortool_test

import (
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// A27. Fails if the run id ignores the earlier answered call of the same id:
// the sub-scheduler would then answer the second call from its record of the
// first, and the body would not run again.
func TestRun_RepeatedCallIDRunsAgain(t *testing.T) {
	var runs atomic.Int32
	analyze := mustNew(t, functiontool.Config{Name: "analyze"}, func(_ agent.Context, _ queryArgs, emit func(*session.Event) error) (int, error) {
		n := int(runs.Add(1))
		if err := emit(message(fmt.Sprintf("run %d", n))); err != nil {
			return 0, err
		}
		return n, nil
	})
	dup := calls(call("dup", "analyze", map[string]any{"query": "sales"}))
	h := newHarness(t, harnessConfig{tools: []tool.Tool{analyze}, responses: []*genai.Content{dup, dup, modelText("done")}})

	h.mustRun(userText("go"))

	if got := runs.Load(); got != 2 {
		t.Errorf("handler runs = %d, want 2", got)
	}
	events := h.events()
	want := []*genai.FunctionResponse{
		{ID: "dup", Name: "analyze", Response: map[string]any{"result": 1}},
		{ID: "dup", Name: "analyze", Response: map[string]any{"result": 2}},
	}
	if diff := cmp.Diff(want, responses(events, "analyze")); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
	var messages []evSummary
	for _, s := range summarize(authoredBy(events, "analyze")) {
		if s.Text != "" {
			messages = append(messages, s)
		}
	}
	wantMessages := []evSummary{
		{Author: "analyze", Branch: "analyze@dup", Path: "root_agent/analyze@dup", Text: "run 1"},
		{Author: "analyze", Branch: "analyze@dup", Path: "root_agent/analyze@~dup~1", Text: "run 2"},
	}
	if diff := cmp.Diff(wantMessages, messages); diff != "" {
		t.Errorf("message events mismatch (-want +got):\n%s", diff)
	}
}
