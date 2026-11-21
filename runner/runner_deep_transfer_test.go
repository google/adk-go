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
	"fmt"
	"iter"
	"slices"
	"strings"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

type recordedLLMRequest struct {
	agentName         string
	systemInstruction string
	hasTransferTool   bool
}

type transferChainLLM struct {
	mu       *sync.Mutex
	records  *[]recordedLLMRequest
	next     string
	finalTxt string
	called   bool
}

func (m *transferChainLLM) Name() string { return "transfer-chain-mock" }

func (m *transferChainLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		var sysText strings.Builder
		if req != nil && req.Config != nil && req.Config.SystemInstruction != nil {
			for _, p := range req.Config.SystemInstruction.Parts {
				if p != nil && p.Text != "" {
					sysText.WriteString(p.Text)
				}
			}
		}
		hasTransfer := req != nil && req.Tools != nil && req.Tools["transfer_to_agent"] != nil
		agentName := ""
		if rc, ok := ctx.(agent.ReadonlyContext); ok {
			agentName = rc.AgentName()
		}
		m.mu.Lock()
		*m.records = append(*m.records, recordedLLMRequest{
			agentName:         agentName,
			systemInstruction: sysText.String(),
			hasTransferTool:   hasTransfer,
		})
		alreadyCalled := m.called
		m.called = true
		m.mu.Unlock()

		if m.next != "" && !alreadyCalled {
			yield(&model.LLMResponse{
				Content: &genai.Content{
					Role: genai.RoleModel,
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							ID:   "xfer-" + m.next,
							Name: "transfer_to_agent",
							Args: map[string]any{"agent_name": m.next},
						},
					}},
				},
			}, nil)
			return
		}
		txt := m.finalTxt
		if txt == "" {
			txt = "done"
		}
		yield(&model.LLMResponse{
			Content: genai.NewContentFromText(txt, genai.RoleModel),
		}, nil)
	}
}

func TestRunner_DeepTransferChain_KeepsGlobalInstructionAndTransferTool(t *testing.T) {
	const globalSentinel = "GLOBAL_INSTRUCTION_SENTINEL"

	for _, depth := range []int{2, 32, 65} {
		t.Run(fmt.Sprintf("depth_%d", depth), func(t *testing.T) {
			var mu sync.Mutex
			var records []recordedLLMRequest

			var child agent.Agent
			for i := depth - 1; i >= 0; i-- {
				name := fmt.Sprintf("agent_%02d", i)
				cfg := llmagent.Config{
					Name:        name,
					Description: name,
				}
				if i == 0 {
					cfg.GlobalInstruction = globalSentinel
				}
				m := &transferChainLLM{
					mu:      &mu,
					records: &records,
				}
				if child != nil {
					cfg.SubAgents = []agent.Agent{child}
					m.next = child.Name()
				} else {
					m.finalTxt = "leaf-complete"
				}
				cfg.Model = m
				a, err := llmagent.New(cfg)
				if err != nil {
					t.Fatalf("llmagent.New(%s): %v", name, err)
				}
				child = a
			}

			r, err := runner.New(runner.Config{
				AppName:           "deep-transfer-test",
				Agent:             child,
				SessionService:    session.InMemoryService(),
				AutoCreateSession: true,
			})
			if err != nil {
				t.Fatalf("runner.New: %v", err)
			}

			var lastText string
			for ev, runErr := range r.Run(t.Context(), "u1", fmt.Sprintf("s-depth-%d", depth), genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("r.Run error at depth %d: %v", depth, runErr)
				}
				if ev != nil && ev.Content != nil {
					for _, p := range ev.Content.Parts {
						if p != nil && p.Text != "" {
							lastText = p.Text
						}
					}
				}
			}
			if lastText != "leaf-complete" {
				t.Fatalf("final text = %q, want %q", lastText, "leaf-complete")
			}

			mu.Lock()
			got := slices.Clone(records)
			mu.Unlock()
			if len(got) != depth {
				t.Fatalf("recorded %d LLM requests, want %d", len(got), depth)
			}
			for _, rec := range got {
				if !strings.Contains(rec.systemInstruction, globalSentinel) {
					t.Fatalf("depth %d agent %s lost GlobalInstruction (%q)", depth, rec.agentName, rec.systemInstruction)
				}
				if !rec.hasTransferTool {
					t.Fatalf("depth %d agent %s lost transfer_to_agent tool in req.Tools", depth, rec.agentName)
				}
			}
		})
	}
}

func TestRunner_TransferAcross64WithContextDerivations(t *testing.T) {
	const globalSentinel = "WRAPPED_GLOBAL_SENTINEL"

	for _, hops := range []int{63, 64, 65} {
		t.Run(fmt.Sprintf("withcontext_hops_%d", hops), func(t *testing.T) {
			var mu sync.Mutex
			var records []recordedLLMRequest
			var leafToMidCalled bool

			leafModel := &transferChainLLM{
				mu:      &mu,
				records: &records,
				next:    "mid",
			}
			leaf, err := llmagent.New(llmagent.Config{
				Name:        "leaf",
				Description: "leaf",
				Model:       leafModel,
			})
			if err != nil {
				t.Fatalf("llmagent.New(leaf): %v", err)
			}

			mid, err := llmagent.New(llmagent.Config{
				Name:        "mid",
				Description: "mid",
				SubAgents:   []agent.Agent{leaf},
				Model: &transferChainLLM{
					mu:       &mu,
					records:  &records,
					next:     "leaf",
					finalTxt: "mid-after-leaf-transfer",
				},
			})
			if err != nil {
				t.Fatalf("llmagent.New(mid): %v", err)
			}

			wrap, err := agent.New(agent.Config{
				Name:        "wrap",
				Description: "derives >=64 WithContext layers",
				SubAgents:   []agent.Agent{mid},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						derived := ctx
						for range hops {
							derived = derived.WithContext(derived)
						}
						for ev, runErr := range mid.Run(derived) {
							if ev != nil && ev.Author == "leaf" && ev.Content != nil {
								for _, p := range ev.Content.Parts {
									if p != nil && p.FunctionCall != nil && p.FunctionCall.Name == "transfer_to_agent" {
										leafToMidCalled = true
									}
								}
							}
							if !yield(ev, runErr) || runErr != nil {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(wrap): %v", err)
			}

			root, err := llmagent.New(llmagent.Config{
				Name:              "root",
				Description:       "root",
				GlobalInstruction: globalSentinel,
				SubAgents:         []agent.Agent{wrap},
				Model: &transferChainLLM{
					mu:      &mu,
					records: &records,
					next:    "wrap",
				},
			})
			if err != nil {
				t.Fatalf("llmagent.New(root): %v", err)
			}

			r, err := runner.New(runner.Config{
				AppName:           "wrapped-transfer-test",
				Agent:             root,
				SessionService:    session.InMemoryService(),
				AutoCreateSession: true,
			})
			if err != nil {
				t.Fatalf("runner.New: %v", err)
			}

			var lastText string
			for ev, runErr := range r.Run(t.Context(), "u1", fmt.Sprintf("s-hops-%d", hops), genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("r.Run(hops=%d) error: %v", hops, runErr)
				}
				if ev != nil && ev.Content != nil {
					for _, p := range ev.Content.Parts {
						if p != nil && p.Text != "" {
							lastText = p.Text
						}
					}
				}
			}
			if !leafToMidCalled {
				t.Fatalf("hops=%d: leaf did not transfer back to mid", hops)
			}
			if lastText != "mid-after-leaf-transfer" {
				t.Fatalf("hops=%d: final text = %q, want %q", hops, lastText, "mid-after-leaf-transfer")
			}

			mu.Lock()
			got := slices.Clone(records)
			mu.Unlock()
			var sawLeaf bool
			for _, rec := range got {
				if !strings.Contains(rec.systemInstruction, globalSentinel) {
					t.Fatalf("hops=%d agent %s lost GlobalInstruction (%q)", hops, rec.agentName, rec.systemInstruction)
				}
				if !rec.hasTransferTool {
					t.Fatalf("hops=%d agent %s lost transfer_to_agent tool", hops, rec.agentName)
				}
				if rec.agentName == "leaf" {
					sawLeaf = true
				}
			}
			if !sawLeaf {
				t.Fatalf("hops=%d: leaf model request was not recorded", hops)
			}
		})
	}
}
