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

package adkrest_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"google.golang.org/genai"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/database"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// This benchmark follows the shape adk-python used to measure its runner
// (commit 772f139): a real LlmAgent and a real Gemini client on a fake
// transport, driven through the HTTP server for a 50-turn conversation on one
// session, so the history grows as it does in production. It reports the
// median turn cost early (turns 2-11) and late (turns 41-50), plus how many
// session.Service.Get calls each turn made, which is the number this RFC
// changes and does not depend on the machine.
//
//	go test ./server/adkrest/ -run '^$' -bench BenchmarkConversation -benchtime=3x -count=5
//
// A fake model reply is around 300 characters, a short chat answer. The
// "tool" shape answers with a function call first and text second, so each
// turn appends four events instead of two. The "sequential3" shape runs
// three LlmAgents in a row, so each turn appends four events and makes three
// model calls.

const conversationTurns = 50

// fakeGemini answers every GenerateContent request. It replies with a
// function call when the agent has a tool and the request does not yet carry
// the tool's response, and with text otherwise.
func fakeGemini() http.RoundTripper {
	text := strings.Repeat("The answer depends on the context you gave me. ", 6)
	textReply := fmt.Sprintf(`{"candidates":[{"content":{"role":"model","parts":[{"text":%q}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":60,"totalTokenCount":160}}`, text)
	callReply := `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"lookup","args":{"q":"context"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":20,"totalTokenCount":120}}`
	return roundTripFunc(func(req *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		reply := textReply
		// The tool is declared but the latest content is not its response yet.
		if bytes.Contains(body, []byte(`"functionDeclarations"`)) && !bytes.Contains(lastContent(body), []byte(`"functionResponse"`)) {
			reply = callReply
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       io.NopCloser(strings.NewReader(reply)),
		}, nil
	})
}

// lastContent returns the JSON of the final element of the request's
// "contents" array, or the whole body when it cannot be found.
func lastContent(body []byte) []byte {
	var req struct {
		Contents []json.RawMessage `json:"contents"`
	}
	if json.Unmarshal(body, &req) != nil || len(req.Contents) == 0 {
		return body
	}
	return req.Contents[len(req.Contents)-1]
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

// countingSessionService counts Get calls so the benchmark can report reads
// per turn, which does not depend on the machine the way latency does.
type countingSessionService struct {
	session.Service
	gets atomic.Int64
}

func (c *countingSessionService) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	c.gets.Add(1)
	return c.Service.Get(ctx, req)
}

func convSessionService(b *testing.B, backend string) session.Service {
	b.Helper()
	switch backend {
	case "inmemory":
		return session.InMemoryService()
	case "sqlite":
		svc, err := database.NewSessionService(sqlite.Open(filepath.Join(b.TempDir(), "s.db")), &gorm.Config{Logger: logger.Discard})
		if err != nil {
			b.Fatalf("open sqlite: %v", err)
		}
		if err := database.AutoMigrate(svc); err != nil {
			b.Fatalf("migrate: %v", err)
		}
		return svc
	}
	b.Fatalf("unknown backend %q", backend)
	return nil
}

func convAgent(b *testing.B, shape string) agent.Agent {
	b.Helper()
	ctx := context.Background()
	m, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		Backend:    genai.BackendGeminiAPI,
		APIKey:     "fake",
		HTTPClient: &http.Client{Transport: fakeGemini()},
	})
	if err != nil {
		b.Fatalf("gemini.NewModel: %v", err)
	}
	llm := func(name string, tools []tool.Tool) agent.Agent {
		a, err := llmagent.New(llmagent.Config{Name: name, Model: m, Instruction: "Answer briefly.", Tools: tools})
		if err != nil {
			b.Fatalf("llmagent.New(%s): %v", name, err)
		}
		return a
	}
	switch shape {
	case "llmagent":
		return llm("root", nil)
	case "llmagent+tool":
		type args struct {
			Q string `json:"q"`
		}
		type result struct {
			Answer string `json:"answer"`
		}
		lookup, err := functiontool.New(functiontool.Config{Name: "lookup", Description: "Looks something up."},
			func(agent.Context, args) (result, error) { return result{Answer: "found"}, nil })
		if err != nil {
			b.Fatalf("functiontool.New: %v", err)
		}
		return llm("root", []tool.Tool{lookup})
	case "sequential3":
		seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{llm("a", nil), llm("b", nil), llm("c", nil)},
		}})
		if err != nil {
			b.Fatalf("sequentialagent.New: %v", err)
		}
		return seq
	}
	b.Fatalf("unknown shape %q", shape)
	return nil
}

// runConversation drives conversationTurns POSTs to /run_sse on one session
// and returns the wall time of each turn.
func runConversation(b *testing.B, srv *httptest.Server, svc session.Service, sessionID string) []time.Duration {
	b.Helper()
	ctx := context.Background()
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "root", UserID: "u", SessionID: sessionID}); err != nil {
		b.Fatalf("create session: %v", err)
	}
	durations := make([]time.Duration, 0, conversationTurns)
	for turn := range conversationTurns {
		body, _ := json.Marshal(map[string]any{
			"appName":    "root",
			"userId":     "u",
			"sessionId":  sessionID,
			"newMessage": map[string]any{"role": "user", "parts": []map[string]any{{"text": fmt.Sprintf("turn %d", turn)}}},
		})
		start := time.Now()
		resp, err := srv.Client().Post(srv.URL+"/run_sse", "application/json", bytes.NewReader(body))
		if err != nil {
			b.Fatalf("turn %d: %v", turn, err)
		}
		out, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		durations = append(durations, time.Since(start))
		if resp.StatusCode != http.StatusOK || bytes.Contains(out, []byte("event: error")) {
			b.Fatalf("turn %d: status %d: %s", turn, resp.StatusCode, out)
		}
	}
	return durations
}

func median(ds []time.Duration) time.Duration {
	s := slices.Clone(ds)
	slices.Sort(s)
	return s[len(s)/2]
}

func BenchmarkConversation(b *testing.B) {
	for _, backend := range []string{"inmemory", "sqlite"} {
		for _, shape := range []string{"llmagent", "llmagent+tool", "sequential3"} {
			b.Run(fmt.Sprintf("%s/%s", backend, shape), func(b *testing.B) {
				svc := &countingSessionService{Service: convSessionService(b, backend)}
				server, err := adkrest.NewServer(adkrest.ServerConfig{
					SessionService: svc,
					AgentLoader:    agent.NewSingleLoader(convAgent(b, shape)),
				})
				if err != nil {
					b.Fatalf("adkrest.NewServer: %v", err)
				}
				srv := httptest.NewServer(server)
				defer srv.Close()

				var early, late, total []time.Duration
				var gets int64
				for i := 0; b.Loop(); i++ {
					before := svc.gets.Load()
					ds := runConversation(b, srv, svc, fmt.Sprintf("s-%d", i))
					gets += svc.gets.Load() - before
					early = append(early, median(ds[1:11]))
					late = append(late, median(ds[40:50]))
					total = append(total, median(ds))
				}
				b.ReportMetric(float64(median(early).Nanoseconds()), "ns/turn-2-11")
				b.ReportMetric(float64(median(late).Nanoseconds()), "ns/turn-41-50")
				b.ReportMetric(float64(median(total).Nanoseconds()), "ns/turn-median")
				b.ReportMetric(float64(gets)/float64(b.N*conversationTurns), "gets/turn")
			})
		}
	}
}
