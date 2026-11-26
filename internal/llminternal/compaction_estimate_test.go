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
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/agent/compactionctx"
	"google.golang.org/adk/v2/internal/compactioninternal"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// estimateFixture stores a session of n exchanges whose text is textChars
// characters long. Each answer reports usage when it is non-nil.
func estimateFixture(t *testing.T, n, textChars int, usage *genai.GenerateContentResponseUsageMetadata) (session.Service, session.Session) {
	t.Helper()

	svc := session.InMemoryService()
	created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	sess := created.Session

	text := strings.Repeat("x", textChars)
	base := time.Unix(1, 0)
	for i := range n {
		q := session.NewEvent(t.Context(), "inv")
		q.Author = "user"
		q.Timestamp = base.Add(time.Duration(2*i) * time.Second)
		q.LLMResponse.Content = genai.NewContentFromText(text, "user")
		if err := svc.AppendEvent(t.Context(), sess, q); err != nil {
			t.Fatalf("AppendEvent() error = %v", err)
		}

		a := session.NewEvent(t.Context(), "inv")
		a.Author = "assistant"
		a.Timestamp = base.Add(time.Duration(2*i+1) * time.Second)
		a.LLMResponse.Content = genai.NewContentFromText(text, "model")
		a.LLMResponse.UsageMetadata = usage
		if err := svc.AppendEvent(t.Context(), sess, a); err != nil {
			t.Fatalf("AppendEvent() error = %v", err)
		}
	}
	return svc, sess
}

// appendUserTurn appends a user event with textChars characters and no usage.
func appendUserTurn(t *testing.T, svc session.Service, sess session.Session, textChars int) {
	t.Helper()

	ev := session.NewEvent(t.Context(), "inv")
	ev.Author = "user"
	ev.Timestamp = time.Unix(1000, 0)
	ev.LLMResponse.Content = genai.NewContentFromText(strings.Repeat("y", textChars), "user")
	if err := svc.AppendEvent(t.Context(), sess, ev); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}
}

// runCompactionProcessorWithRequest runs the processor with the system
// instruction on req and the tools on the flow, which is where the default
// chain leaves them: the instructions processor puts the resolved instruction
// on req, and the tools processor collects the agent's tools, toolsets
// included, into the flow. Tool declarations are copied onto req only after
// every request processor has run. The agent itself is bare, so an estimate
// must read req and the flow rather than the agent's configuration.
func runCompactionProcessorWithRequest(t *testing.T, svc session.Service, sess session.Session, req *model.LLMRequest, tools []tool.Tool, cfg *compaction.Config) {
	t.Helper()

	ctx := compactionctx.ToContext(t.Context(), compactionctx.New(cfg, svc))
	testAgent := utils.Must(llmagent.New(llmagent.Config{Name: "assistant", Model: &testModel{}}))
	ictx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Agent:   testAgent,
		Session: sess,
	})
	for _, err := range llminternal.CompactionRequestProcessor(ictx, req, &llminternal.Flow{Tools: tools}) {
		if err != nil {
			t.Fatalf("CompactionRequestProcessor() error = %v", err)
		}
	}
}

type lookupArgs struct {
	Query string `json:"query" jsonschema:"The search query. The search query. The search query. The search query."`
	Limit int    `json:"limit" jsonschema:"Maximum number of results. Maximum number of results. Maximum number of results."`
}

// lookupTools returns n function tools with a long description and a
// parameter schema, the way an agent with many tools fills the request.
func lookupTools(t *testing.T, n int) []tool.Tool {
	t.Helper()
	tools := make([]tool.Tool, 0, n)
	for i := range n {
		lookup, err := functiontool.New(functiontool.Config{
			Name:        fmt.Sprintf("tool_%02d", i),
			Description: strings.Repeat("Looks up records and returns them. ", 12),
		}, func(agent.Context, lookupArgs) (struct{}, error) { return struct{}{}, nil })
		if err != nil {
			t.Fatalf("functiontool.New() error = %v", err)
		}
		tools = append(tools, lookup)
	}
	return tools
}

// preambleChars returns the size in characters of the system instruction and
// the tool declarations, for a test log.
func preambleChars(t *testing.T, req *model.LLMRequest, tools []tool.Tool) int {
	t.Helper()
	parts := []any{req.Config.SystemInstruction}
	for _, tl := range tools {
		if d, ok := tl.(toolinternal.FunctionTool); ok {
			parts = append(parts, d.Declaration())
		}
	}
	b, err := json.Marshal(parts)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	return len(b)
}

// TestCompactionProcessorEstimateCountsSystemInstructionAndTools checks that a
// prompt dominated by the system instruction or tool declarations triggers
// compaction when the model has never reported usage.
//
// With no reported count, the decision rests on an estimate. When the
// compaction processor runs, the system instruction is already on the request
// and the agent's tools are already on the flow, so both belong in that
// estimate. Today it counts only the contents (text and tool traffic), and a
// short history under a huge preamble reads as far below the threshold.
func TestCompactionProcessorEstimateCountsSystemInstructionAndTools(t *testing.T) {
	t.Parallel()

	const (
		threshold = 1000
		// Short history: well under the threshold on its own.
		historyChars = 200
	)

	tests := []struct {
		name           string
		instruction    string
		tools          int
		wantCompaction bool
	}{
		{
			name:           "large system instruction",
			instruction:    strings.Repeat("Follow the house style guide. ", 400),
			wantCompaction: true,
		},
		{
			name:           "many tool declarations",
			tools:          40,
			wantCompaction: true,
		},
		{
			name:           "small system instruction and one tool stay under the threshold",
			instruction:    "Be brief.",
			tools:          1,
			wantCompaction: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			svc, sess := estimateFixture(t, 4, historyChars, nil)
			summarizer := &fixedSummarizer{}
			req := &model.LLMRequest{Config: &genai.GenerateContentConfig{}}
			if tc.instruction != "" {
				req.Config.SystemInstruction = genai.NewContentFromText(tc.instruction, "user")
			}
			tools := lookupTools(t, tc.tools)

			var contents []*genai.Content
			for ev := range sess.Events().All() {
				contents = append(contents, ev.Content)
			}
			chars := preambleChars(t, req, tools)
			t.Logf("threshold=%d tokens; contents-only estimate=%d tokens; system instruction and tool declarations=%d chars (about %d tokens)",
				threshold, compactioninternal.EstimateTokensFromContents(contents), chars, chars/4)

			runCompactionProcessorWithRequest(t, svc, sess, req, tools, &compaction.Config{
				TokenThreshold:     threshold,
				EventRetentionSize: 2,
				Summarizer:         summarizer,
			})

			if got := summarizer.calls > 0; got != tc.wantCompaction {
				t.Errorf("compaction ran = %v, want %v: the estimate must count the system instruction and the tool declarations, not only the contents", got, tc.wantCompaction)
			}
		})
	}
}

// TestCompactionProcessorDoesNotTrustAnImpossiblyLowCount checks that a
// reported prompt token count far below the size of the prompt does not
// suppress compaction.
//
// A server can report a constant such as 1. The prompt is plainly larger than
// that, so the decision should use the larger of the reported count plus the
// estimate of the events after it, and the estimate of the whole prompt. Today
// a positive reported count plus the later events is used as is, and the
// whole-prompt estimate is never compared with it.
func TestCompactionProcessorDoesNotTrustAnImpossiblyLowCount(t *testing.T) {
	t.Parallel()

	const threshold = 5000

	tests := []struct {
		name           string
		reported       int32
		textChars      int
		instruction    string
		tools          int
		trailingChars  int
		wantCompaction bool
	}{
		{
			name:           "reported 1 for a prompt far above the threshold",
			reported:       1,
			textChars:      8000,
			wantCompaction: true,
		},
		{
			name:           "reported 1 for a prompt under the threshold",
			reported:       1,
			textChars:      100,
			wantCompaction: false,
		},
		{
			// The contents alone stay under the threshold, but the whole
			// prompt, instruction included, is well above it.
			name:           "reported 1 with a large instruction above the threshold",
			reported:       1,
			textChars:      200,
			instruction:    strings.Repeat("Follow the house style guide. ", 800),
			wantCompaction: true,
		},
		{
			name:           "accurate report above the threshold",
			reported:       6000,
			textChars:      100,
			wantCompaction: true,
		},
		{
			// The reported count already includes the system instruction, so
			// adding the instruction again to the estimate of the later turn
			// would count it twice and cross the threshold.
			name:           "accurate report with a large instruction and a later turn stays under the threshold",
			reported:       3500,
			textChars:      100,
			instruction:    strings.Repeat("Follow the house style guide. ", 400),
			trailingChars:  100,
			wantCompaction: false,
		},
		{
			// Same for the tool declarations: the reported count already
			// includes them, so adding them again would cross the threshold.
			name:           "accurate report with tools on the flow and a later turn stays under the threshold",
			reported:       3500,
			textChars:      100,
			tools:          15,
			trailingChars:  100,
			wantCompaction: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			usage := &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: tc.reported}
			svc, sess := estimateFixture(t, 4, tc.textChars, usage)
			if tc.trailingChars > 0 {
				appendUserTurn(t, svc, sess, tc.trailingChars)
			}
			summarizer := &fixedSummarizer{}
			req := &model.LLMRequest{Config: &genai.GenerateContentConfig{}}
			if tc.instruction != "" {
				req.Config.SystemInstruction = genai.NewContentFromText(tc.instruction, "user")
			}

			var contents []*genai.Content
			for ev := range sess.Events().All() {
				contents = append(contents, ev.Content)
			}
			t.Logf("threshold=%d tokens; reported=%d; contents estimate=%d tokens",
				threshold, tc.reported, compactioninternal.EstimateTokensFromContents(contents))

			runCompactionProcessorWithRequest(t, svc, sess, req, lookupTools(t, tc.tools), &compaction.Config{
				TokenThreshold:     threshold,
				EventRetentionSize: 2,
				Summarizer:         summarizer,
			})

			if got := summarizer.calls > 0; got != tc.wantCompaction {
				t.Errorf("compaction ran = %v, want %v: the token count must be the larger of the reported count plus later events and the estimate of the whole prompt", got, tc.wantCompaction)
			}
		})
	}
}
