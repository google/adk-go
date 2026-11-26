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
	"context"
	"iter"
	"slices"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// cacheProbeModel answers each call with the next of its responses, repeating
// the last, and records the context cache config each request carried and the
// ledger its context held.
type cacheProbeModel struct {
	responses []*genai.Content

	mu      sync.Mutex
	configs []*model.ContextCacheConfig
	ledgers []*llminternal.ContextCacheLedger
}

func (m *cacheProbeModel) Name() string { return "probe" }

func (*cacheProbeModel) Capabilities() model.Capabilities { return model.Capabilities{} }

func (m *cacheProbeModel) GenerateContent(ctx context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	reply := m.responses[min(len(m.configs), len(m.responses)-1)]
	m.configs = append(m.configs, req.CacheConfig)
	m.ledgers = append(m.ledgers, llminternal.ContextCacheLedgerFromContext(ctx))
	m.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: reply}, nil)
	}
}

// seen returns the config and ledger of every call so far.
func (m *cacheProbeModel) seen() ([]*model.ContextCacheConfig, []*llminternal.ContextCacheLedger) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.configs), slices.Clone(m.ledgers)
}

// TestRunner_ContextCacheOffInsideCachingRun checks that a runner with caching
// off gives its agent neither the context cache config nor the ledger of a
// caching run it is started inside. Here a tool of the caching run starts it,
// with the context the tool was given. A workflow root takes a different path
// through the runner than an LLM agent root, so both run.
func TestRunner_ContextCacheOffInsideCachingRun(t *testing.T) {
	tests := []struct {
		name string
		root func(leaf agent.Agent) (agent.Agent, error)
	}{
		{
			name: "LLM agent root",
			root: func(leaf agent.Agent) (agent.Agent, error) { return leaf, nil },
		},
		{
			name: "sequential agent root",
			root: func(leaf agent.Agent) (agent.Agent, error) {
				return sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "sequence", SubAgents: []agent.Agent{leaf}}})
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			innerModel := &cacheProbeModel{responses: []*genai.Content{genai.NewContentFromText("inner done", genai.RoleModel)}}
			leaf, err := llmagent.New(llmagent.Config{Name: "leaf", Model: innerModel})
			if err != nil {
				t.Fatalf("llmagent.New() error = %v", err)
			}
			root, err := tt.root(leaf)
			if err != nil {
				t.Fatalf("building the root agent: %v", err)
			}
			inner, err := runner.New(runner.Config{
				AppName:           "inner",
				Agent:             root,
				SessionService:    session.InMemoryService(),
				AutoCreateSession: true,
			})
			if err != nil {
				t.Fatalf("runner.New() error = %v", err)
			}
			delegate, err := functiontool.New(functiontool.Config{Name: "delegate", Description: "runs the inner agent"},
				func(ctx agent.Context, _ struct{}) (string, error) {
					for _, err := range inner.Run(ctx, "u", "s", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
						if err != nil {
							return "", err
						}
					}
					return "done", nil
				})
			if err != nil {
				t.Fatalf("functiontool.New() error = %v", err)
			}

			outerModel := &cacheProbeModel{responses: []*genai.Content{
				callContent("delegate", nil),
				genai.NewContentFromText("all done", genai.RoleModel),
			}}
			parent, err := llmagent.New(llmagent.Config{Name: "parent", Model: outerModel, Tools: []tool.Tool{delegate}})
			if err != nil {
				t.Fatalf("llmagent.New() error = %v", err)
			}
			outer, err := runner.New(runner.Config{
				AppName:           "outer",
				Agent:             parent,
				SessionService:    session.InMemoryService(),
				AutoCreateSession: true,
				ContextCache:      &model.ContextCacheConfig{},
			})
			if err != nil {
				t.Fatalf("runner.New() error = %v", err)
			}

			for _, err := range outer.Run(t.Context(), "u", "s", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("Run() error = %v", err)
				}
			}

			outerConfigs, outerLedgers := outerModel.seen()
			if len(outerConfigs) != 2 {
				t.Fatalf("outer model calls = %d, want 2", len(outerConfigs))
			}
			if outerConfigs[0] == nil || outerLedgers[0] == nil {
				t.Fatalf("the outer run has no cache config or no ledger (config %v, ledger %v), so this test checks nothing", outerConfigs[0], outerLedgers[0])
			}
			innerConfigs, innerLedgers := innerModel.seen()
			if len(innerConfigs) != 1 {
				t.Fatalf("inner model calls = %d, want 1", len(innerConfigs))
			}
			if innerConfigs[0] != nil {
				t.Errorf("the inner request's CacheConfig = %+v, want none", innerConfigs[0])
			}
			if innerLedgers[0] != nil {
				t.Error("the inner model call saw the outer run's ledger, want none")
			}
		})
	}
}
