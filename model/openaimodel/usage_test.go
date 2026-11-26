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

package openaimodel

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// TestInvalidUsageDoesNotDriveCompaction runs whole turns against a server
// whose input_tokens wraps to math.MaxInt32 when narrowed to int32. Narrowed,
// every few-token prompt would look over any threshold, and tail-retention
// compaction would summarize older turns as soon as there were enough of them
// (twice in these four turns). Reported as unknown, compaction falls back to
// its own estimate and leaves the conversation alone.
func TestInvalidUsageDoesNotDriveCompaction(t *testing.T) {
	for _, api := range []API{APIResponses, APIChatCompletions} {
		t.Run(string(api), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				if strings.HasSuffix(r.URL.Path, "/chat/completions") {
					_, _ = fmt.Fprint(w, `{"id":"c","model":"m","choices":[{"index":0,"finish_reason":"stop",`+
						`"message":{"role":"assistant","content":"ack"}}],`+
						`"usage":{"prompt_tokens":-2147483649,"completion_tokens":1,"total_tokens":1}}`)
					return
				}
				_, _ = fmt.Fprint(w, `{"id":"r","model":"m","status":"completed",`+
					`"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ack"}]}],`+
					`"usage":{"input_tokens":-2147483649,"output_tokens":1,"total_tokens":1}}`)
			}))
			t.Cleanup(server.Close)

			llm, err := NewModel(t.Context(), "test-model", &ClientConfig{
				APIKey: "test", BaseURL: server.URL + "/v1", HTTPClient: server.Client(), API: api,
			})
			if err != nil {
				t.Fatalf("NewModel() error = %v", err)
			}
			a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: llm})
			if err != nil {
				t.Fatalf("llmagent.New() error = %v", err)
			}
			svc := session.InMemoryService()
			r, err := runner.New(runner.Config{
				AppName:           "app",
				Agent:             a,
				SessionService:    svc,
				AutoCreateSession: true,
				Compaction:        &compaction.Config{TokenThreshold: 1000, EventRetentionSize: 2},
			})
			if err != nil {
				t.Fatalf("runner.New() error = %v", err)
			}
			for i := range 4 {
				for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("turn %d: %v", i+1, err)
					}
				}
			}

			got, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			for ev := range got.Session.Events().All() {
				if ev.Actions.Compaction != nil {
					t.Errorf("event %s compacts the conversation, want none: the prompts are a few tokens each", ev.ID)
				}
				if ev.Author == "assistant" && ev.UsageMetadata != nil {
					t.Errorf("event %s stores usage %+v, want none from a server whose counts are impossible", ev.ID, ev.UsageMetadata)
				}
			}
		})
	}
}
