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

package runner_test

import (
	"iter"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// tkDecorate puts a value in every field of an event that a yielded copy has
// to carry over, next to the "temp:" key the runner drops.
func tkDecorate(ev *session.Event) {
	ev.Content = tkText("answer")
	ev.TurnComplete = true
	ev.OutputTranscription = &genai.Transcription{Text: "spoken"}
	ev.Branch = "b1.b2"
	ev.IsolationScope = "iso"
	ev.LongRunningToolIDs = []string{"lr1"}
	ev.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{TotalTokenCount: 7}
	ev.CustomMetadata = map[string]any{"m": "x"}
	ev.Actions.StateDelta = tkDelta()
	ev.Actions.ArtifactDelta = map[string]int64{"a.txt": 3}
	ev.Actions.SkipSummarization = true
	ev.Actions.Escalate = true
}

func TestRun_YieldedEventKeepsEveryFieldButTempKeys(t *testing.T) {
	tests := []struct {
		name   string
		author string
		cfg    func(t *testing.T) runner.Config
	}{
		{
			name:   "classic_root",
			author: "root",
			cfg: func(t *testing.T) runner.Config {
				a, err := agent.New(agent.Config{Name: "root", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "root"
						tkDecorate(ev)
						yield(ev, nil)
					}
				}})
				if err != nil {
					t.Fatalf("agent.New() error = %v", err)
				}
				return runner.Config{Agent: a}
			},
		},
		{
			name:   "node_root",
			author: "producer",
			cfg: func(*testing.T) runner.Config {
				return runner.Config{Node: &tkDecoratingNode{tkNode: tkNewNode(nil, false)}}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r, svc := tkNew(t, tc.cfg(t))

			got := tkAuthored(tkRun(t, r, userText("hi")), tc.author)
			stored := tkAuthored(tkStored(t, svc), tc.author)

			if len(got) != 1 || len(stored) != 1 {
				t.Fatalf("yielded %d and stored %d events by %q, want 1 each", len(got), len(stored), tc.author)
			}
			y := got[0]
			if y.ID == "" {
				t.Error("yielded event has no ID, want the ID the service assigned")
			}
			if diff := cmp.Diff(stored[0], y, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("yielded event differs from the stored one (-stored +yielded):\n%s", diff)
			}
			if diff := cmp.Diff(tkText("answer"), y.Content); diff != "" {
				t.Errorf("yielded content (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(map[string]int64{"a.txt": 3}, y.Actions.ArtifactDelta); diff != "" {
				t.Errorf("yielded artifact delta (-want +got):\n%s", diff)
			}
			if y.Branch != "b1.b2" || y.IsolationScope != "iso" || y.OutputTranscription == nil || y.OutputTranscription.Text != "spoken" || !slices.Equal(y.LongRunningToolIDs, []string{"lr1"}) {
				t.Errorf("yielded event lost a field: Branch=%q IsolationScope=%q OutputTranscription=%v LongRunningToolIDs=%v, want \"b1.b2\", \"iso\", \"spoken\", [lr1]",
					y.Branch, y.IsolationScope, y.OutputTranscription, y.LongRunningToolIDs)
			}
			if diff := cmp.Diff(map[string]any{"m": "x"}, y.CustomMetadata); diff != "" {
				t.Errorf("yielded custom metadata (-want +got):\n%s", diff)
			}
			if !y.TurnComplete || !y.Actions.SkipSummarization || !y.Actions.Escalate || y.UsageMetadata == nil || y.UsageMetadata.TotalTokenCount != 7 {
				t.Errorf("yielded event lost a field: TurnComplete=%v SkipSummarization=%v Escalate=%v usage=%v, want true, true, true and 7 tokens",
					y.TurnComplete, y.Actions.SkipSummarization, y.Actions.Escalate, y.UsageMetadata)
			}
		})
	}
}

// tkDecoratingNode is tkNode with every field of its event set.
type tkDecoratingNode struct{ *tkNode }

func (n *tkDecoratingNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Author = n.Name()
		ev.NodeInfo = &session.NodeInfo{Path: n.Name()}
		tkDecorate(ev)
		yield(ev, nil)
	}
}
