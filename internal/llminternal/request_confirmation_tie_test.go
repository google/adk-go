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
	"context"
	"fmt"
	"iter"
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"google.golang.org/genai"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/database"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

type tieTransferArgs struct {
	Amount int `json:"amount"`
}

// tieTransferModel calls the transfer tool on its first call and answers with text after that.
type tieTransferModel struct {
	model.LLM
	calls int
}

func (m *tieTransferModel) Name() string { return "transfer-model" }

func (m *tieTransferModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.calls++
		part := genai.NewPartFromText("done")
		if m.calls == 1 {
			part = &genai.Part{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "transfer", Args: map[string]any{"amount": 100}}}
		}
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}, nil)
	}
}

// TestRepeatedApprovalTimestampTie checks that a repeated approval does not rerun
// the tool when the database session service stores the tool result and the first
// approval with the same timestamp, which it then orders by random event ID.
func TestRepeatedApprovalTimestampTie(t *testing.T) {
	const sessions = 40
	reran := 0
	for i := 0; i < sessions; i++ {
		executed := 0
		transfer, err := functiontool.New(functiontool.Config{Name: "transfer", Description: "sends money", RequireConfirmation: true},
			func(ctx agent.Context, a tieTransferArgs) (map[string]any, error) {
				executed++
				return map[string]any{"status": "sent"}, nil
			})
		if err != nil {
			t.Fatal(err)
		}
		a, err := llmagent.New(llmagent.Config{Name: "bank", Model: &tieTransferModel{}, Tools: []tool.Tool{transfer}})
		if err != nil {
			t.Fatal(err)
		}
		svc, err := database.NewSessionService(sqlite.Open(fmt.Sprintf("file:tie%d?mode=memory&cache=shared", i)),
			&gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
		if err != nil {
			t.Fatal(err)
		}
		if err := database.AutoMigrate(svc); err != nil {
			t.Fatal(err)
		}
		if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
			t.Fatal(err)
		}
		r, err := runner.New(runner.Config{Agent: a, SessionService: svc, AppName: "app"})
		if err != nil {
			t.Fatal(err)
		}

		// Every event written during one runner.Run call gets the same timestamp.
		turn := 0
		run := func(msg *genai.Content) (confirmationID string) {
			t.Helper()
			now := time.Date(2026, 1, 1, 0, 0, i, 0, time.UTC).Add(time.Duration(turn) * time.Second / 2)
			turn++
			ctx := platform.WithTimeProvider(t.Context(), func() time.Time { return now })
			for ev, err := range r.Run(ctx, "u", "s", msg, agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("Run: %v", err)
				}
				if ev.Content == nil {
					continue
				}
				for _, p := range ev.Content.Parts {
					if p.FunctionCall != nil && p.FunctionCall.Name == toolconfirmation.FunctionCallName {
						confirmationID = p.FunctionCall.ID
					}
				}
			}
			return confirmationID
		}

		confirmationID := run(genai.NewContentFromText("send 100", "user"))
		if confirmationID == "" {
			t.Fatal("no confirmation requested")
		}
		approval := &genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			Name: toolconfirmation.FunctionCallName, ID: confirmationID, Response: map[string]any{"confirmed": true},
		}}}}
		run(approval)
		if executed != 1 {
			t.Fatalf("session %d: tool ran %d times after the first approval, want 1", i, executed)
		}
		run(approval) // the same approval, delivered again
		if executed > 1 {
			reran++
		}
	}
	if reran > 0 {
		t.Errorf("a repeated approval reran the tool in %d of %d sessions", reran, sessions)
	}
}
