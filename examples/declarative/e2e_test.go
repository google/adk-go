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

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/internal/httprr"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/openaimodel"
)

// TestStudioEndToEnd runs the embedded document against traffic recorded from
// a real model behind an OpenAI-compatible endpoint. To re-record it, point the
// OPENAI_* variables at a model that supports tool calls, for example through a
// local Ollama server:
//
//	export OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama
//	export OPENAI_MODEL=glm-5.3-flash:cloud
//	go generate ./examples/declarative/...
//
//go:generate go test -httprecord=^testdata[/\\]TestStudioEndToEnd\.httprr$ -run=^TestStudioEndToEnd$
func TestStudioEndToEnd(t *testing.T) {
	llm := openRecordedModel(t, filepath.Join("testdata", "TestStudioEndToEnd.httprr"))
	agents, err := loadSystem(agentSystem, testRegistry(t, llm))
	if err != nil {
		t.Fatalf("loadSystem() error = %v", err)
	}

	t.Run("writer_critics", func(t *testing.T) {
		llm.reset()
		got := runAgent(t, agents, "studio", "Write a short paragraph about the history of the bicycle.")

		m := regexp.MustCompile(`^Final draft after ([1-5]) turn\(s\), [0-3]/3 approvals:\n\n\S`).FindStringSubmatch(got)
		if m == nil {
			// The reply is model output, so the message gives only its size.
			t.Fatalf("reply of %d bytes is not a published final draft", len(got))
		}
		turns, _ := strconv.Atoi(m[1])
		counts := map[string]int{}
		for _, c := range llm.callers() {
			counts[c]++
		}
		want := map[string]int{
			"studio": 1, "writer": turns,
			"critic_clarity": turns, "critic_accuracy": turns, "critic_style": turns,
		}
		if diff := cmp.Diff(want, counts); diff != "" {
			t.Errorf("model calls per caller mismatch (-want +got):\n%s", diff)
		}
	})

	routerTests := []struct {
		prompt string
		branch string
	}{
		{prompt: "Is the Eiffel Tower taller than the Statue of Liberty?", branch: "answer_question"},
		{prompt: "I just won the marathon!", branch: "react_exclamation"},
		{prompt: "Paris is the capital of France.", branch: "comment_statement"},
	}
	for _, tc := range routerTests {
		t.Run("llm_router/"+tc.branch, func(t *testing.T) {
			llm.reset()
			got := runAgent(t, agents, "studio", tc.prompt)

			if strings.TrimSpace(got) == "" {
				t.Error("reply is empty")
			}
			if diff := cmp.Diff([]string{"studio", "classify", tc.branch}, llm.callers()); diff != "" {
				t.Errorf("model callers mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// recordedModel is an openaimodel whose HTTP traffic goes through httprr, and
// which notes the agent or llm node behind each request.
type recordedModel struct {
	model.LLM

	mu    sync.Mutex
	calls []string
}

func (m *recordedModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	caller, err := callerOf(req)
	if err != nil {
		caller = "<unknown>"
	}
	m.mu.Lock()
	m.calls = append(m.calls, caller)
	m.mu.Unlock()
	return m.LLM.GenerateContent(ctx, req, stream)
}

func (m *recordedModel) callers() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.calls)
}

func (m *recordedModel) reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.calls = nil
}

// openRecordedModel replays file, or records it when -httprecord matches it.
// Requests are recorded against a fixed endpoint and model name, so the
// recording replays whichever endpoint produced it.
func openRecordedModel(t *testing.T, file string) *recordedModel {
	t.Helper()
	rr, err := httprr.Open(file, http.DefaultTransport)
	if err != nil {
		t.Fatalf("httprr.Open(%q) error = %v", file, err)
	}
	t.Cleanup(func() { _ = rr.Close() })
	rr.ScrubReq(scrubOpenAIRequest)
	rr.ScrubResp(scrubOpenAIResponse)

	cfg := &openaimodel.ClientConfig{
		APIKey:     "test-key",
		BaseURL:    "http://openai.test/v1",
		API:        openaimodel.APIChatCompletions,
		HTTPClient: rr.Client(),
	}
	name := "test-model"
	if rr.Recording() {
		cfg.APIKey, cfg.BaseURL, name = os.Getenv("OPENAI_API_KEY"), os.Getenv("OPENAI_BASE_URL"), os.Getenv("OPENAI_MODEL")
		if name == "" || (cfg.APIKey == "" && cfg.BaseURL == "") {
			t.Fatal("recording needs OPENAI_MODEL, and OPENAI_API_KEY or OPENAI_BASE_URL")
		}
	}
	llm, err := openaimodel.NewModel(t.Context(), name, cfg)
	if err != nil {
		t.Fatalf("openaimodel.NewModel() error = %v", err)
	}
	return &recordedModel{LLM: llm}
}

// scrubOpenAIRequest keeps only the request headers that describe the body,
// so neither credentials nor headers the client reads from the environment,
// such as OPENAI_ORG_ID, reach the recording or the replay key. It also
// rewrites the endpoint and model name to the ones replay uses.
func scrubOpenAIRequest(req *http.Request) error {
	for k := range req.Header {
		if k != "Accept" && k != "Content-Type" {
			delete(req.Header, k)
		}
	}
	i := strings.LastIndex(req.URL.Path, "/chat/completions")
	if i < 0 {
		return fmt.Errorf("unexpected request path %q", req.URL.Path)
	}
	req.URL.Scheme, req.URL.Host, req.Host = "http", "openai.test", "openai.test"
	req.URL.Path = "/v1" + req.URL.Path[i:]
	req.RequestURI = ""

	b, ok := req.Body.(*httprr.Body)
	if !ok || len(b.Data) == 0 {
		return nil
	}
	var body map[string]any
	if err := json.Unmarshal(b.Data, &body); err != nil {
		return err
	}
	body["model"] = "test-model"
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	b.Data = data
	return nil
}

// scrubOpenAIResponse keeps the status line and only the response headers
// needed to read the body back, dropping any that identify the account, the
// request or the serving infrastructure.
func scrubOpenAIResponse(buf *bytes.Buffer) error {
	head, body, ok := strings.Cut(buf.String(), "\r\n\r\n")
	if !ok {
		return nil
	}
	lines := strings.Split(head, "\r\n")
	kept := lines[:1]
	for _, line := range lines[1:] {
		name, _, _ := strings.Cut(line, ":")
		switch http.CanonicalHeaderKey(name) {
		case "Content-Type", "Content-Length", "Transfer-Encoding":
			kept = append(kept, line)
		}
	}
	// httprr may still be reading the caller's copy of the response from buf's
	// backing array, so the scrubbed copy goes in a new one.
	*buf = *bytes.NewBufferString(strings.Join(kept, "\r\n") + "\r\n\r\n" + body)
	return nil
}
