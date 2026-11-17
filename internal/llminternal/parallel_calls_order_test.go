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

package llminternal_test

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
)

// TestParallelCalls_MergedEventKeepsCallOrderWhenCallsFinishOutOfOrder pins
// that the event merging the responses of one turn lists them in call order,
// and merges their state writes, when call 0 finishes after call 1. It fails
// if the merge takes the responses in completion order.
func TestParallelCalls_MergedEventKeepsCallOrderWhenCallsFinishOutOfOrder(t *testing.T) {
	slow := psoTool(t, "slow", func(ctx agent.Context, _ struct{}) (map[string]any, error) {
		time.Sleep(100 * time.Millisecond)
		return map[string]any{"who": "slow"}, ctx.State().Set("k0", "v0")
	})
	fast := psoTool(t, "fast", func(ctx agent.Context, _ struct{}) (map[string]any, error) {
		return map[string]any{"who": "fast"}, ctx.State().Set("k1", "v1")
	})

	turn := psoRun(t, []tool.Tool{slow, fast}, nil, psoCall("c0", "slow", nil), psoCall("c1", "fast", nil))

	if turn.responseEvent == nil {
		t.Fatal("no function response event")
	}
	var ids []string
	for _, p := range turn.responseEvent.Content.Parts {
		ids = append(ids, p.FunctionResponse.ID)
	}
	if diff := cmp.Diff([]string{"c0", "c1"}, ids); diff != "" {
		t.Errorf("function response order mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"k0": "v0", "k1": "v1"}, turn.responseEvent.Actions.StateDelta); diff != "" {
		t.Errorf("merged state delta mismatch (-want +got):\n%s", diff)
	}
}
