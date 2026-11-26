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

package anthropicmodel

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// benchSchemaExtras are top-level tool schema keywords other than type,
// properties and required, in the shape MCP servers built with the TypeScript
// SDK ($schema, additionalProperties) and documented schemas (title,
// description, $defs) send them.
var benchSchemaExtras = []struct {
	key   string
	value any
}{
	{"additionalProperties", false},
	{"$schema", "http://json-schema.org/draft-07/schema#"},
	{"title", "SearchArguments"},
	{"description", "Arguments of the search tool."},
	{"$defs", map[string]any{
		"filter": map[string]any{
			"type":       "object",
			"properties": map[string]any{"field": map[string]any{"type": "string"}, "value": map[string]any{"type": "string"}},
		},
	}},
}

const benchMessageResponse = `{"id":"msg_b","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}`

// benchToolsRequest returns a request declaring numTools tools whose JSON
// schemas carry the first numExtras keywords of benchSchemaExtras.
func benchToolsRequest(numTools, numExtras int) *model.LLMRequest {
	decls := make([]*genai.FunctionDeclaration, numTools)
	for i := range decls {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query":  map[string]any{"type": "string", "description": "Search query."},
				"limit":  map[string]any{"type": "integer", "minimum": 1, "maximum": 100},
				"tags":   map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
				"filter": map[string]any{"$ref": "#/$defs/filter"},
			},
			"required": []any{"query"},
		}
		for _, e := range benchSchemaExtras[:numExtras] {
			schema[e.key] = e.value
		}
		decls[i] = &genai.FunctionDeclaration{
			Name:                 fmt.Sprintf("search_%d", i),
			Description:          "Searches the document index.",
			ParametersJsonSchema: schema,
		}
	}
	return &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Find the latest report.", genai.RoleUser)},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("You are a search assistant.", genai.RoleUser),
			Tools:             []*genai.Tool{{FunctionDeclarations: decls}},
		},
	}
}

// requestBodyRecorder is an httptest server answering every Messages call with
// a minimal message and keeping the request bodies it received.
type requestBodyRecorder struct {
	srv    *httptest.Server
	mu     sync.Mutex
	bodies []string
}

func newRequestBodyRecorder(tb testing.TB) *requestBodyRecorder {
	tb.Helper()
	rec := &requestBodyRecorder{}
	rec.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		rec.mu.Lock()
		rec.bodies = append(rec.bodies, string(data))
		rec.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, benchMessageResponse)
	}))
	tb.Cleanup(rec.srv.Close)
	return rec
}

// counts returns how many distinct bodies were received and how many bodies
// differ from the body of the call before.
func (rec *requestBodyRecorder) counts() (distinct, changes int) {
	rec.mu.Lock()
	defer rec.mu.Unlock()
	seen := make(map[string]bool)
	for i, body := range rec.bodies {
		seen[body] = true
		if i > 0 && body != rec.bodies[i-1] {
			changes++
		}
	}
	return len(seen), changes
}

func newPromptCachingModel(tb testing.TB, baseURL string) *anthropicModel {
	tb.Helper()
	llm, err := NewModel(tb.Context(), "claude-sonnet-4-5-20250929", ClientConfig{
		APIKey:      "test-key",
		BaseURL:     baseURL,
		PromptCache: &PromptCacheConfig{TTL: PromptCacheTTL5m},
	})
	if err != nil {
		tb.Fatalf("NewModel() error = %v", err)
	}
	return llm.(*anthropicModel)
}

// BenchmarkToolSchemaRequest measures building and serializing a Messages
// request (the work GenerateContent does before sending it) for tool schemas
// with 0, 1, 2 and 5 extra top-level keywords, on 1 and 20 tools.
//
// distinct-bodies and body-changes count, over 200 identical GenerateContent
// calls with prompt caching on, the distinct request bodies the server
// received and the calls whose body differs from the previous call's. Tool
// definitions head Anthropic's cached prompt prefix, so every body change on
// identical input is a prompt cache miss.
func BenchmarkToolSchemaRequest(b *testing.B) {
	const calls = 200
	for _, numTools := range []int{1, 20} {
		for _, numExtras := range []int{0, 1, 2, 5} {
			b.Run(fmt.Sprintf("tools=%d/extras=%d", numTools, numExtras), func(b *testing.B) {
				req := benchToolsRequest(numTools, numExtras)
				rec := newRequestBodyRecorder(b)
				m := newPromptCachingModel(b, rec.srv.URL)
				for range calls {
					for _, err := range m.GenerateContent(b.Context(), req, false) {
						if err != nil {
							b.Fatalf("GenerateContent() error = %v", err)
						}
					}
				}
				distinct, changes := rec.counts()

				b.ReportAllocs()
				for b.Loop() {
					params, _, _, err := m.buildMessageParams(b.Context(), req)
					if err != nil {
						b.Fatalf("buildMessageParams() error = %v", err)
					}
					if _, err := params.MarshalJSON(); err != nil {
						b.Fatalf("MarshalJSON() error = %v", err)
					}
				}
				b.ReportMetric(float64(distinct), "distinct-bodies/200calls")
				b.ReportMetric(float64(changes), "body-changes/200calls")
			})
		}
	}
}
