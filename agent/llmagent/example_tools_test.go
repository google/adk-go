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

package llmagent_test

import (
	"context"
	"fmt"
	"iter"
	"log"
	"maps"
	"slices"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
)

// auditTool is a tool without a function declaration. Its ProcessRequest
// adds an instruction and registers the tool, as a declared tool would.
type auditTool struct{ runs int }

func (*auditTool) Name() string                            { return "audit_log" }
func (*auditTool) Description() string                     { return "Writes an audit record." }
func (*auditTool) IsLongRunning() bool                     { return false }
func (*auditTool) Declaration() *genai.FunctionDeclaration { return nil }

func (t *auditTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	req.Config.SystemInstruction = genai.NewContentFromText("Every answer is audited.", genai.RoleUser)
	return toolutils.PackTool(req, t)
}

func (t *auditTool) Run(agent.Context, any) (map[string]any, error) {
	t.runs++
	return map[string]any{"ok": true}, nil
}

// callsAuditModel calls audit_log on its first request and answers "done"
// on every later one. It prints what each request registers.
type callsAuditModel struct{ calls int }

func (*callsAuditModel) Name() string { return "calls-audit" }

func (m *callsAuditModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls++
	fmt.Printf("request %d: tools %v, %d genai.Tool, instruction %q\n",
		m.calls, slices.Sorted(maps.Keys(req.Tools)), len(req.Config.Tools), req.Config.SystemInstruction.Parts[0].Text)
	content := genai.NewContentFromText("done", genai.RoleModel)
	if m.calls == 1 {
		content = genai.NewContentFromFunctionCall("audit_log", nil, genai.RoleModel)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop}, nil)
	}
}

// A tool whose Declaration returns nil still runs its ProcessRequest, but the
// model is not offered it and cannot call it: the request holds no entry for
// it, and a function call to its name is answered with the tool-not-found
// error instead of running the tool.
func ExampleConfig_tools() {
	audit := &auditTool{}
	a, err := llmagent.New(llmagent.Config{
		Name:  "assistant",
		Model: &callsAuditModel{},
		Tools: []tool.Tool{audit},
	})
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: sessions})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("hello", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if fr := p.FunctionResponse; fr != nil {
				first, _, _ := strings.Cut(fmt.Sprint(fr.Response["error"]), "\n")
				fmt.Printf("response to %s: %s\n", fr.Name, first)
			}
		}
	}
	fmt.Println("audit_log runs:", audit.runs)
	// Output:
	// request 1: tools [], 0 genai.Tool, instruction "Every answer is audited."
	// response to audit_log: Invoking `audit_log()` failed as no tool with that name is available. The tools you can call are: none. You could retry, but it is IMPORTANT that you only call a tool from that list.
	// request 2: tools [], 0 genai.Tool, instruction "Every answer is audited."
	// audit_log runs: 0
}
