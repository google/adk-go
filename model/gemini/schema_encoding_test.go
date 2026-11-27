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
	"encoding/json"
	"fmt"
	"iter"
	"reflect"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/schemacache"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type Address struct {
	Street string  `json:"street" jsonschema:"street and number"`
	Zip    *string `json:"zip,omitempty"`
}

type Item struct {
	SKU   string  `json:"sku"`
	Qty   int     `json:"qty" jsonschema:"how many"`
	Price float64 `json:"price"`
}

type OrderArgs struct {
	OrderID  string            `json:"order_id" jsonschema:"the order number"`
	Items    []Item            `json:"items"`
	Ship     Address           `json:"ship"`
	Bill     *Address          `json:"bill,omitempty"`
	Tags     map[string]string `json:"tags,omitempty"`
	Priority *int              `json:"priority,omitempty"`
	Express  bool              `json:"express,omitempty"`
	Matrix   [][]float64       `json:"matrix,omitempty"`
}

type OrderResult struct {
	Status string `json:"status"`
	Items  []Item `json:"items,omitempty"`
}

type WeatherArgs struct {
	City string `json:"city" jsonschema:"the city to report on"`
}

// schemaCorpus returns function tools covering the schema shapes
// functiontool produces: inferred nested structs, slices, maps, optional
// pointers, descriptions and required lists, a map argument, an override with
// an enum, and a streaming tool.
func schemaCorpus(t testing.TB) []tool.Tool {
	t.Helper()
	unit := &jsonschema.Schema{
		Type:     "object",
		Required: []string{"city"},
		Properties: map[string]*jsonschema.Schema{
			"city": {Type: "string", Description: "the city"},
			"unit": {Type: "string", Enum: []any{"celsius", "fahrenheit"}},
			"days": {Type: "integer", Minimum: new(float64)},
		},
	}
	var tools []tool.Tool
	add := func(tl tool.Tool, err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		tools = append(tools, tl)
	}
	add(functiontool.New(functiontool.Config{Name: "lookup_order", Description: "Looks up an order."},
		func(agent.Context, OrderArgs) (OrderResult, error) { return OrderResult{}, nil }))
	add(functiontool.New(functiontool.Config{Name: "get_weather", Description: "Reports the weather."},
		func(agent.Context, WeatherArgs) (string, error) { return "sunny", nil }))
	add(functiontool.New(functiontool.Config{Name: "forecast", Description: "Forecasts.", InputSchema: unit},
		func(agent.Context, map[string]any) (map[string]any, error) { return nil, nil }))
	add(functiontool.NewStreaming(functiontool.Config{Name: "watch", Description: "Streams updates."},
		func(agent.Context, WeatherArgs) iter.Seq2[string, error] { return func(func(string, error) bool) {} }))
	// A pointer argument, and an any result whose schema is the literal true.
	add(functiontool.New(functiontool.Config{Name: "loose", Description: "Takes a pointer, returns anything."},
		func(agent.Context, *WeatherArgs) (any, error) { return nil, nil }))
	// A recursive override through $defs and $ref, which inference cannot build.
	tree := &jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"root": {Ref: "#/$defs/node"}},
		Defs: map[string]*jsonschema.Schema{"node": {
			Type: "object",
			Properties: map[string]*jsonschema.Schema{
				"name": {Type: "string"},
				"kids": {Type: "array", Items: &jsonschema.Schema{Ref: "#/$defs/node"}},
			},
		}},
	}
	add(functiontool.New(functiontool.Config{Name: "walk", Description: "Walks a tree.", InputSchema: tree},
		func(agent.Context, map[string]any) (map[string]any, error) { return nil, nil }))
	return tools
}

type declarer interface {
	Declaration() *genai.FunctionDeclaration
}

func declarations(t testing.TB, tools []tool.Tool) []*genai.FunctionDeclaration {
	t.Helper()
	var decls []*genai.FunctionDeclaration
	for _, tl := range tools {
		d, ok := tl.(declarer)
		if !ok {
			t.Fatalf("%s (%T) has no Declaration method", tl.Name(), tl)
		}
		decls = append(decls, d.Declaration())
	}
	return decls
}

// unremembered returns a copy of decl whose schemas are equal clones that
// schemacache has never seen, so genai encodes them itself.
func unremembered(t testing.TB, decl *genai.FunctionDeclaration) *genai.FunctionDeclaration {
	t.Helper()
	c := *decl
	for _, f := range []*any{&c.ParametersJsonSchema, &c.ResponseJsonSchema} {
		if s, ok := (*f).(*jsonschema.Schema); ok && s != nil {
			clone := s.CloneSchemas()
			if _, ok := schemacache.Encoded(clone); ok {
				t.Fatal("a fresh clone is remembered")
			}
			*f = clone
		}
	}
	return &c
}

// handBuilt are declarations nobody remembered: the map schema an MCP tool
// sets, a hand-built *jsonschema.Schema, and the typed genai.Schema.
func handBuilt() []*genai.FunctionDeclaration {
	return []*genai.FunctionDeclaration{
		{Name: "mcp_tool", ParametersJsonSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"q": map[string]any{"type": "string"}},
		}},
		{Name: "hand_built", ParametersJsonSchema: &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{"x": {Type: "number"}}}},
		{Name: "typed", Parameters: &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{"y": {Type: genai.TypeString}}}},
	}
}

// send makes one call with decls and returns the request body on the wire.
// It also fails the test if the call changed the request's own tools.
func send(t *testing.T, stream bool, decls []*genai.FunctionDeclaration) []byte {
	t.Helper()
	return sendOn(t, genai.BackendGeminiAPI, stream, decls)
}

func sendOn(t *testing.T, backend genai.Backend, stream bool, decls []*genai.FunctionDeclaration) []byte {
	t.Helper()
	transport := &fakeTransport{bodies: [][]byte{geminiText}}
	m := newFakeModelOn(t, backend, transport)
	tools := []*genai.Tool{{FunctionDeclarations: decls}}
	type snapshot struct {
		decl             *genai.FunctionDeclaration
		params, response any
	}
	var before []snapshot
	for _, d := range decls {
		before = append(before, snapshot{d, d.ParametersJsonSchema, d.ResponseJsonSchema})
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
		Config:   &genai.GenerateContentConfig{Tools: tools},
	}
	for _, err := range m.GenerateContent(t.Context(), req, stream) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(req.Config.Tools) != 1 || req.Config.Tools[0] != tools[0] || len(tools[0].FunctionDeclarations) != len(before) {
		t.Fatal("GenerateContent replaced the request's tools")
	}
	for i, b := range before {
		d := tools[0].FunctionDeclarations[i]
		if d != b.decl || !same(d.ParametersJsonSchema, b.params) || !same(d.ResponseJsonSchema, b.response) {
			t.Errorf("GenerateContent replaced declaration %q or one of its schemas", b.decl.Name)
		}
	}
	if len(transport.requests) != 1 {
		t.Fatalf("got %d requests, want 1", len(transport.requests))
	}
	return transport.requests[0]
}

// same reports whether a and b are the same schema value: the same pointer or
// the same map.
func same(a, b any) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	va, vb := reflect.ValueOf(a), reflect.ValueOf(b)
	return va.Type() == vb.Type() && va.UnsafePointer() == vb.UnsafePointer()
}

// TestGenerateContent_RememberedSchemasSendTheSameBody checks that sending
// function tools' remembered schemas as their recorded encoding puts the same
// bytes on the wire as letting genai encode equal schemas, for every shape in
// the corpus and alongside declarations nobody remembered.
func TestGenerateContent_RememberedSchemasSendTheSameBody(t *testing.T) {
	decls := declarations(t, schemaCorpus(t))
	for i, d := range decls {
		if _, ok := schemacache.Encoded(d.ParametersJsonSchema); !ok {
			t.Fatalf("declaration %d (%s) parameters schema is not remembered", i, d.Name)
		}
	}
	var plain []*genai.FunctionDeclaration
	for _, d := range decls {
		plain = append(plain, unremembered(t, d))
	}
	cached := append(decls, handBuilt()...)
	plain = append(plain, handBuilt()...)
	for _, backend := range []genai.Backend{genai.BackendGeminiAPI, genai.BackendVertexAI} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%v/stream=%v", backend, stream), func(t *testing.T) {
				got, want := sendOn(t, backend, stream, cached), sendOn(t, backend, stream, plain)
				if !bytes.Equal(got, want) {
					t.Errorf("body with remembered schemas differs from the body with encoded ones:\n got: %s\nwant: %s", got, want)
				}
				for _, d := range decls {
					if !bytes.Contains(got, []byte(fmt.Sprintf("%q", d.Name))) {
						t.Errorf("body does not declare %s: %s", d.Name, got)
					}
				}
			})
		}
	}
}

// TestEncodeToolSchemas_LeavesUnrememberedAlone checks that tools and
// declarations without a remembered schema go to genai as the same values,
// not copies with anything swapped.
func TestEncodeToolSchemas_LeavesUnrememberedAlone(t *testing.T) {
	decls := handBuilt()
	other := &genai.Tool{GoogleSearch: &genai.GoogleSearch{}}
	tools := []*genai.Tool{{FunctionDeclarations: decls}, other, nil}
	got := encodeToolSchemas(tools)
	if &got[0] != &tools[0] {
		t.Error("encodeToolSchemas copied a tool list with nothing remembered")
	}

	remembered := declarations(t, schemaCorpus(t))[1]
	tools = []*genai.Tool{{FunctionDeclarations: append([]*genai.FunctionDeclaration{remembered, nil}, decls...)}, other, nil}
	got = encodeToolSchemas(tools)
	if got[1] != other || got[2] != nil {
		t.Error("encodeToolSchemas replaced a tool without function declarations")
	}
	gotDecls := got[0].FunctionDeclarations
	if _, ok := gotDecls[0].ParametersJsonSchema.(json.RawMessage); !ok {
		t.Errorf("remembered parameters went to genai as %T, want json.RawMessage", gotDecls[0].ParametersJsonSchema)
	}
	if _, ok := gotDecls[0].ResponseJsonSchema.(json.RawMessage); !ok {
		t.Errorf("remembered response went to genai as %T, want json.RawMessage", gotDecls[0].ResponseJsonSchema)
	}
	if gotDecls[1] != nil {
		t.Error("a nil declaration became non-nil")
	}
	for i, d := range decls {
		if gotDecls[i+2] != d {
			t.Errorf("declaration %s was copied, want it shared", d.Name)
		}
	}

	// One remembered schema is enough to swap it, and the other is left as is.
	paramsOnly := declarations(t, schemaCorpus(t))[3] // a streaming tool declares no response
	responseOnly := *remembered
	responseOnly.ParametersJsonSchema = decls[0].ParametersJsonSchema
	got = encodeToolSchemas([]*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{paramsOnly, &responseOnly}}})
	gotDecls = got[0].FunctionDeclarations
	if _, ok := gotDecls[0].ParametersJsonSchema.(json.RawMessage); !ok || gotDecls[0].ResponseJsonSchema != nil {
		t.Errorf("params-only declaration went to genai with parameters %T and response %v, want json.RawMessage and nil", gotDecls[0].ParametersJsonSchema, gotDecls[0].ResponseJsonSchema)
	}
	if _, ok := gotDecls[1].ResponseJsonSchema.(json.RawMessage); !ok || !same(gotDecls[1].ParametersJsonSchema, decls[0].ParametersJsonSchema) {
		t.Errorf("response-only declaration went to genai with parameters %T and response %T, want the MCP map and json.RawMessage", gotDecls[1].ParametersJsonSchema, gotDecls[1].ResponseJsonSchema)
	}
}

// TestGenerateContent_SchemaMutatedAfterCreationSendsTheRecordedEncoding
// documents what a schema changed after its tool was created does, which
// functiontool.Config forbids: the call still sends the encoding recorded when
// the tool was created. Detecting the change would cost the encoding this
// cache exists to avoid.
func TestGenerateContent_SchemaMutatedAfterCreationSendsTheRecordedEncoding(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			decl := declarations(t, schemaCorpus(t))[1]
			s := decl.ParametersJsonSchema.(*jsonschema.Schema)
			s.Properties["city"].Description = "changed after creation"
			body := send(t, stream, []*genai.FunctionDeclaration{decl})
			if bytes.Contains(body, []byte("changed after creation")) || !bytes.Contains(body, []byte("the city to report on")) {
				t.Errorf("body = %s, want the description recorded at creation", body)
			}
		})
	}
}

// TestBeforeModelCallback_SeesTypedSchemas checks that across the two model
// calls of a tool turn, a BeforeModel callback sees a function tool's
// parameters schema as the *jsonschema.Schema value the tool declared, not its
// encoding.
func TestBeforeModelCallback_SeesTypedSchemas(t *testing.T) {
	weather, err := functiontool.New(functiontool.Config{Name: "get_weather", Description: "Reports the weather."},
		func(agent.Context, WeatherArgs) (string, error) { return "sunny", nil })
	if err != nil {
		t.Fatal(err)
	}
	want := weather.(declarer).Declaration().ParametersJsonSchema
	calls := 0
	a, err := llmagent.New(llmagent.Config{
		Name:  "agent",
		Model: newFakeModel(t, &fakeTransport{bodies: [][]byte{geminiCall, geminiText}}),
		Tools: []tool.Tool{weather},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
			calls++
			got := req.Config.Tools[0].FunctionDeclarations[0].ParametersJsonSchema
			if _, ok := got.(*jsonschema.Schema); !ok || got != want {
				t.Errorf("call %d: callback saw parameters %T %p, want the tool's *jsonschema.Schema %p", calls, got, got, want)
			}
			return nil, nil
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range r.Run(t.Context(), "user", "s", genai.NewContentFromText("Weather in Paris?", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if calls != 2 {
		t.Errorf("BeforeModel ran %d times, want 2 (the call, then the answer)", calls)
	}
}

// TestBeforeModelCallback_ReplacedSchemaIsSent checks that a callback that
// replaces a function tool's schema for one call has its schema sent on that
// call, and that the next call sends the tool's own schema again.
func TestBeforeModelCallback_ReplacedSchemaIsSent(t *testing.T) {
	weather, err := functiontool.New(functiontool.Config{Name: "get_weather", Description: "Reports the weather."},
		func(agent.Context, WeatherArgs) (string, error) { return "sunny", nil })
	if err != nil {
		t.Fatal(err)
	}
	transport := &fakeTransport{bodies: [][]byte{geminiCall, geminiText}}
	calls := 0
	a, err := llmagent.New(llmagent.Config{
		Name:  "agent",
		Model: newFakeModel(t, transport),
		Tools: []tool.Tool{weather},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
			calls++
			if calls > 1 {
				return nil, nil
			}
			decl := req.Config.Tools[0].FunctionDeclarations[0]
			replaced := decl.ParametersJsonSchema.(*jsonschema.Schema).CloneSchemas()
			replaced.Properties["city"].Description = "set by the callback"
			c := *decl
			c.ParametersJsonSchema = replaced
			req.Config.Tools[0].FunctionDeclarations[0] = &c
			return nil, nil
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range r.Run(t.Context(), "user", "s", genai.NewContentFromText("Weather in Paris?", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(transport.requests) != 2 {
		t.Fatalf("got %d requests, want 2", len(transport.requests))
	}
	for i, want := range []string{"set by the callback", "the city to report on"} {
		if !bytes.Contains(transport.requests[i], []byte(want)) {
			t.Errorf("request %d = %s, want the description %q", i, transport.requests[i], want)
		}
	}
}
