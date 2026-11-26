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

	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/agent/compactionctx"
	"google.golang.org/adk/v2/internal/compactioninternal"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
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

// runCompactionProcessorWithRequest runs the processor on a request that
// already carries what the tools and instructions processors add before it in
// the default chain, through cfg.
func runCompactionProcessorWithRequest(t *testing.T, svc session.Service, sess session.Session, req *model.LLMRequest, cfg *compaction.Config) {
	t.Helper()

	ctx := compactionctx.ToContext(t.Context(), compactionctx.New(cfg, svc))
	testAgent := utils.Must(llmagent.New(llmagent.Config{Name: "assistant", Model: &testModel{}}))
	ictx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Agent:   testAgent,
		Session: sess,
	})
	for _, err := range llminternal.CompactionRequestProcessor(ictx, req, &llminternal.Flow{}) {
		if err != nil {
			t.Fatalf("CompactionRequestProcessor() error = %v", err)
		}
	}
}

// toolDeclarations returns n function declarations with a description and a
// parameter schema, the way a toolset with many tools fills the request.
func toolDeclarations(n int) []*genai.Tool {
	decls := make([]*genai.FunctionDeclaration, 0, n)
	for i := range n {
		decls = append(decls, &genai.FunctionDeclaration{
			Name:        fmt.Sprintf("tool_%02d", i),
			Description: strings.Repeat("Looks up records and returns them. ", 12),
			Parameters: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"query": {Type: genai.TypeString, Description: strings.Repeat("The search query. ", 10)},
					"limit": {Type: genai.TypeInteger, Description: strings.Repeat("Maximum number of results. ", 10)},
				},
			},
		})
	}
	return []*genai.Tool{{FunctionDeclarations: decls}}
}

// requestChars returns the size in characters of the request parts outside
// contents, for a test log.
func requestChars(t *testing.T, cfg *genai.GenerateContentConfig) int {
	t.Helper()
	b, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	return len(b)
}

// TestCompactionProcessorEstimateCountsSystemInstructionAndTools checks that a
// prompt dominated by the system instruction or tool declarations triggers
// compaction when the model has never reported usage.
//
// With no reported count, the decision rests on an estimate. The system
// instruction and the tool declarations are already on the request when the
// compaction processor runs, so they belong in that estimate. Today it counts
// only the conversation text, and a short history under a huge preamble reads
// as far below the threshold.
func TestCompactionProcessorEstimateCountsSystemInstructionAndTools(t *testing.T) {
	t.Parallel()

	const (
		threshold = 1000
		// Short history: well under the threshold on its own.
		historyChars = 200
	)

	tests := []struct {
		name           string
		config         *genai.GenerateContentConfig
		wantCompaction bool
	}{
		{
			name:           "large system instruction",
			config:         &genai.GenerateContentConfig{SystemInstruction: genai.NewContentFromText(strings.Repeat("Follow the house style guide. ", 400), "user")},
			wantCompaction: true,
		},
		{
			name:           "many tool declarations",
			config:         &genai.GenerateContentConfig{Tools: toolDeclarations(40)},
			wantCompaction: true,
		},
		{
			name:           "small system instruction and few tools stay under the threshold",
			config:         &genai.GenerateContentConfig{SystemInstruction: genai.NewContentFromText("Be brief.", "user"), Tools: toolDeclarations(1)},
			wantCompaction: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			svc, sess := estimateFixture(t, 4, historyChars, nil)
			summarizer := &fixedSummarizer{}

			var contents []*genai.Content
			for ev := range sess.Events().All() {
				contents = append(contents, ev.Content)
			}
			t.Logf("threshold=%d tokens; contents-only estimate=%d tokens; system instruction and tools=%d chars (about %d tokens)",
				threshold, compactioninternal.EstimateTokensFromContents(contents),
				requestChars(t, tc.config), requestChars(t, tc.config)/4)

			runCompactionProcessorWithRequest(t, svc, sess, &model.LLMRequest{Config: tc.config}, &compaction.Config{
				TokenThreshold:     threshold,
				EventRetentionSize: 2,
				Summarizer:         summarizer,
			})

			if got := summarizer.calls > 0; got != tc.wantCompaction {
				t.Errorf("compaction ran = %v, want %v: the estimate must count the system instruction and tool declarations on the request, not only the contents", got, tc.wantCompaction)
			}
		})
	}
}

// TestCompactionProcessorDoesNotTrustAnImpossiblyLowCount checks that a
// reported prompt token count far below the size of the prompt does not
// suppress compaction.
//
// A server can report a constant such as 1. The prompt is plainly larger than
// that, so the decision should use the larger of the reported count plus
// everything appended since, and the estimate of the whole history. Today the
// reported count wins whenever it is positive.
func TestCompactionProcessorDoesNotTrustAnImpossiblyLowCount(t *testing.T) {
	t.Parallel()

	const threshold = 5000

	tests := []struct {
		name           string
		reported       int32
		textChars      int
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
			name:           "accurate report above the threshold",
			reported:       6000,
			textChars:      100,
			wantCompaction: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			usage := &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: tc.reported}
			svc, sess := estimateFixture(t, 4, tc.textChars, usage)
			summarizer := &fixedSummarizer{}

			var contents []*genai.Content
			for ev := range sess.Events().All() {
				contents = append(contents, ev.Content)
			}
			t.Logf("threshold=%d tokens; reported=%d; contents estimate=%d tokens",
				threshold, tc.reported, compactioninternal.EstimateTokensFromContents(contents))

			runCompactionProcessorWithRequest(t, svc, sess, &model.LLMRequest{Config: &genai.GenerateContentConfig{}}, &compaction.Config{
				TokenThreshold:     threshold,
				EventRetentionSize: 2,
				Summarizer:         summarizer,
			})

			if got := summarizer.calls > 0; got != tc.wantCompaction {
				t.Errorf("compaction ran = %v, want %v: the token count must be the larger of the reported count plus later events and the estimate", got, tc.wantCompaction)
			}
		})
	}
}
