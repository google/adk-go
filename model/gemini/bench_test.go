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

package gemini

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"cloud.google.com/go/auth"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolutils"
)

var (
	geminiText = []byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"It is 22.5°C and sunny in Paris."}]},"finishReason":"STOP"}]}`)
	geminiCall = []byte(`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris"}}}]},"finishReason":"STOP"}]}`)
)

// fakeTransport answers Gemini API requests with bodies in turn, without
// opening a connection, and records each request body. A streaming request
// gets its answer as one server-sent event.
type fakeTransport struct {
	bodies  [][]byte
	discard bool // do not record request bodies

	mu       sync.Mutex
	calls    int
	requests [][]byte
}

func (t *fakeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	t.mu.Lock()
	if !t.discard {
		t.requests = append(t.requests, body)
	}
	answer := t.bodies[t.calls%len(t.bodies)]
	t.calls++
	t.mu.Unlock()
	if strings.Contains(req.URL.Path, "streamGenerateContent") {
		answer = append(append([]byte("data: "), answer...), "\n\n"...)
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(bytes.NewReader(answer)),
		Request:    req,
	}, nil
}

type staticToken struct{}

func (staticToken) Token(context.Context) (*auth.Token, error) {
	return &auth.Token{Value: "test-token", Type: "Bearer"}, nil
}

func newFakeModel(t testing.TB, transport *fakeTransport) model.LLM {
	t.Helper()
	return newFakeModelOn(t, genai.BackendGeminiAPI, transport)
}

func newFakeModelOn(t testing.TB, backend genai.Backend, transport *fakeTransport) model.LLM {
	t.Helper()
	cfg := &genai.ClientConfig{Backend: backend, HTTPClient: &http.Client{Transport: transport}}
	if backend == genai.BackendVertexAI {
		cfg.Project, cfg.Location, cfg.Credentials = "test-project", "us-central1", auth.NewCredentials(&auth.CredentialsOptions{TokenProvider: staticToken{}})
	} else {
		cfg.APIKey = "test-key"
	}
	m, err := NewModel(t.Context(), "gemini-2.5-flash", cfg)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

type benchWeatherArgs struct {
	City string `json:"city" jsonschema:"the city to report on"`
	Unit string `json:"unit,omitempty" jsonschema:"celsius or fahrenheit"`
}

type benchWeatherResult struct {
	TempC      float64 `json:"temp_c"`
	Conditions string  `json:"conditions"`
	Humidity   int     `json:"humidity"`
}

type benchOrderArgs struct {
	OrderID      string `json:"order_id" jsonschema:"the order number"`
	IncludeItems bool   `json:"include_items,omitempty"`
}

type benchOrderItem struct {
	SKU   string  `json:"sku"`
	Qty   int     `json:"qty"`
	Price float64 `json:"price"`
}

type benchOrderResult struct {
	Status string           `json:"status"`
	Items  []benchOrderItem `json:"items,omitempty"`
}

type benchConvertArgs struct {
	Amount float64 `json:"amount"`
	From   string  `json:"from" jsonschema:"ISO 4217 code"`
	To     string  `json:"to" jsonschema:"ISO 4217 code"`
}

type benchConvertResult struct {
	Value float64 `json:"value"`
	Rate  float64 `json:"rate"`
}

// benchTools returns three small function tools with struct arguments and
// results, the shape of a typical tool-using agent.
func benchTools(b *testing.B) []tool.Tool {
	b.Helper()
	weather, err := functiontool.New(functiontool.Config{Name: "get_weather", Description: "Returns the current weather for a city."},
		func(_ agent.Context, in benchWeatherArgs) (benchWeatherResult, error) {
			return benchWeatherResult{TempC: 22.5, Conditions: "sunny", Humidity: 40}, nil
		})
	if err != nil {
		b.Fatal(err)
	}
	order, err := functiontool.New(functiontool.Config{Name: "lookup_order", Description: "Looks up an order by its number."},
		func(_ agent.Context, in benchOrderArgs) (benchOrderResult, error) {
			return benchOrderResult{Status: "shipped"}, nil
		})
	if err != nil {
		b.Fatal(err)
	}
	convert, err := functiontool.New(functiontool.Config{Name: "convert_currency", Description: "Converts an amount between currencies."},
		func(_ agent.Context, in benchConvertArgs) (benchConvertResult, error) {
			return benchConvertResult{Value: in.Amount * 1.1, Rate: 1.1}, nil
		})
	if err != nil {
		b.Fatal(err)
	}
	return []tool.Tool{weather, order, convert}
}

// BenchmarkGenerateContentTools is the client-side cost of one Gemini call
// carrying three function tools and n history contents: building and
// encoding the request, and decoding the response.
func BenchmarkGenerateContentTools(b *testing.B) {
	m := newFakeModel(b, &fakeTransport{bodies: [][]byte{geminiText}, discard: true})
	tools := benchTools(b)
	for _, n := range []int{0, 100} {
		b.Run(fmt.Sprintf("contents=%d", n), func(b *testing.B) {
			req := &model.LLMRequest{Config: &genai.GenerateContentConfig{}}
			for _, t := range tools {
				if err := toolutils.PackTool(req, t.(toolutils.Tool)); err != nil {
					b.Fatal(err)
				}
			}
			for i := range n {
				role := genai.RoleUser
				if i%2 == 1 {
					role = genai.RoleModel
				}
				req.Contents = append(req.Contents, genai.NewContentFromText(fmt.Sprintf("message %d about the weather and an order", i), genai.Role(role)))
			}
			// Ending on a user turn keeps GenerateContent from appending a
			// continuation prompt to req on every iteration.
			req.Contents = append(req.Contents, genai.NewContentFromText("What is the weather in Paris?", genai.RoleUser))
			b.ReportAllocs()
			for b.Loop() {
				for _, err := range m.GenerateContent(b.Context(), req, false) {
					if err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}

// BenchmarkToolTurn is one runner turn of an agent with three function tools
// whose model answers with a get_weather call and then a text, so the turn
// makes two Gemini calls through the adapter around one tool call.
func BenchmarkToolTurn(b *testing.B) {
	a, err := llmagent.New(llmagent.Config{
		Name:  "agent",
		Model: newFakeModel(b, &fakeTransport{bodies: [][]byte{geminiCall, geminiText}, discard: true}),
		Tools: benchTools(b),
	})
	if err != nil {
		b.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		b.Fatal(err)
	}
	msg := genai.NewContentFromText("What is the weather in Paris?", genai.RoleUser)
	b.ReportAllocs()
	i := 0
	for b.Loop() {
		events := 0
		for _, err := range r.Run(b.Context(), "user", fmt.Sprintf("s%d", i), msg, agent.RunConfig{}) {
			if err != nil {
				b.Fatal(err)
			}
			events++
		}
		if events != 3 {
			b.Fatalf("turn %d yielded %d events, want 3 (call, response, answer)", i, events)
		}
		i++
	}
}
