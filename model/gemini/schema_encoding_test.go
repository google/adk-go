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

package gemini_test

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/benchutil"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
)

// capturingTransport records each request body and answers with a final
// text.
type capturingTransport struct{ bodies [][]byte }

func (t *capturingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	t.bodies = append(t.bodies, body)
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(bytes.NewReader(benchutil.GeminiText)),
		Request:    req,
	}, nil
}

// TestGenerateContent_RememberedSchemasSendTheSameBody checks that sending a
// function tool's remembered schemas as their stored encoding puts the same
// bytes on the wire as encoding the schemas, and leaves the request's own
// declarations untouched. The map schema an MCP tool sets rides along
// unchanged.
func TestGenerateContent_RememberedSchemasSendTheSameBody(t *testing.T) {
	tools, err := benchutil.Tools()
	if err != nil {
		t.Fatal(err)
	}
	remembered := tools[1].(toolinternal.FunctionTool).Declaration()
	params, ok := remembered.ParametersJsonSchema.(*jsonschema.Schema)
	if !ok {
		t.Fatalf("ParametersJsonSchema is %T, want *jsonschema.Schema", remembered.ParametersJsonSchema)
	}
	mcp := &genai.FunctionDeclaration{Name: "mcp_tool", ParametersJsonSchema: map[string]any{
		"type":       "object",
		"properties": map[string]any{"q": map[string]any{"type": "string"}},
	}}

	send := func(decl *genai.FunctionDeclaration) []byte {
		t.Helper()
		origParams, origResponse := decl.ParametersJsonSchema, decl.ResponseJsonSchema
		transport := &capturingTransport{}
		m, err := gemini.NewModel(t.Context(), "gemini-2.5-flash", &genai.ClientConfig{
			APIKey: "test-key", Backend: genai.BackendGeminiAPI, HTTPClient: &http.Client{Transport: transport},
		})
		if err != nil {
			t.Fatal(err)
		}
		req := &model.LLMRequest{
			Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
			Config:   &genai.GenerateContentConfig{Tools: []*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{decl, mcp}}}},
		}
		for _, err := range m.GenerateContent(t.Context(), req, false) {
			if err != nil {
				t.Fatal(err)
			}
		}
		if req.Config.Tools[0].FunctionDeclarations[0] != decl || decl.ParametersJsonSchema != origParams || decl.ResponseJsonSchema != origResponse {
			t.Error("GenerateContent replaced the request's own declaration or one of its schemas")
		}
		if len(transport.bodies) != 1 {
			t.Fatalf("got %d requests, want 1", len(transport.bodies))
		}
		return transport.bodies[0]
	}

	// The same schema as a fresh value, which nothing has remembered.
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	var fresh jsonschema.Schema
	if err := json.Unmarshal(raw, &fresh); err != nil {
		t.Fatal(err)
	}
	notRemembered := *remembered
	notRemembered.ParametersJsonSchema = &fresh
	if got, want := send(remembered), send(&notRemembered); !bytes.Equal(got, want) {
		t.Errorf("body with the remembered schema differs from the body with an encoded one:\n got: %s\nwant: %s", got, want)
	}
}
