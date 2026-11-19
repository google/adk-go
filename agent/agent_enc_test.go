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

package agent_test

import (
	"errors"
	"iter"
	"testing"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/plugininternal"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

// An error from a sub-agent passes through every agent layer, and each layer
// reports it once, innermost first, with the same value (adk-python
// agents/base_agent.py, run_async: every BaseAgent notifies on_agent_error for
// an exception that escapes it).
func TestAgentRunOnAgentErrorCallbackNotifiesAncestorsInnermostFirst(t *testing.T) {
	errBoom := errors.New("boom")
	type report struct {
		agent string
		err   error
	}
	var reports []report
	p, err := plugin.New(plugin.Config{
		Name: "recorder",
		OnAgentErrorCallback: func(_ agent.Context, a agent.Agent, err error) {
			reports = append(reports, report{agent: a.Name(), err: err})
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	pm, err := plugininternal.NewPluginManager(plugininternal.PluginConfig{Plugins: []*plugin.Plugin{p}})
	if err != nil {
		t.Fatalf("NewPluginManager() error = %v", err)
	}
	child, err := agent.New(agent.Config{
		Name: "child",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, errBoom)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(child) error = %v", err)
	}
	parent, err := agent.New(agent.Config{
		Name:      "parent",
		SubAgents: []agent.Agent{child},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for ev, err := range child.Run(ctx) {
					if !yield(ev, err) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(parent) error = %v", err)
	}

	ctx := plugininternal.ToContext(t.Context(), pm)
	ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{Agent: parent})
	var yielded []error
	for _, err := range parent.Run(ic) {
		if err != nil {
			yielded = append(yielded, err)
		}
	}

	if len(reports) != 2 || reports[0].agent != "child" || reports[1].agent != "parent" {
		t.Fatalf("reports = %v, want child then parent", reports)
	}
	if reports[0].err != errBoom || reports[1].err != errBoom {
		t.Errorf("reported errors = %v, %v, want errBoom for both", reports[0].err, reports[1].err)
	}
	if len(yielded) != 1 || yielded[0] != errBoom {
		t.Errorf("caller received %v, want [errBoom]", yielded)
	}
}
