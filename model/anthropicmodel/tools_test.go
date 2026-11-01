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
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// buildToolInputSchemaReference is buildToolInputSchema with every extra
// keyword left in ExtraFields, which the SDK writes in map iteration order.
func buildToolInputSchemaReference(decl *genai.FunctionDeclaration) (anthropic.ToolInputSchemaParam, error) {
	var rawSchema any
	switch {
	case decl.ParametersJsonSchema != nil:
		rawSchema = decl.ParametersJsonSchema
	case decl.Parameters != nil:
		rawSchema = decl.Parameters
	default:
		return anthropic.ToolInputSchemaParam{
			Properties: map[string]any{},
		}, nil
	}

	data, err := json.Marshal(rawSchema)
	if err != nil {
		return anthropic.ToolInputSchemaParam{}, fmt.Errorf("%w: failed to marshal tool schema for %q", ErrInvalidConfig, decl.Name)
	}

	var schemaMap map[string]any
	if err := json.Unmarshal(data, &schemaMap); err != nil || schemaMap == nil {
		return anthropic.ToolInputSchemaParam{}, fmt.Errorf("%w: tool schema for %q must be a JSON object", ErrInvalidConfig, decl.Name)
	}

	lowercaseSchemaTypes(schemaMap)

	var out anthropic.ToolInputSchemaParam
	if props, ok := schemaMap["properties"]; ok && props != nil {
		out.Properties = props
	} else {
		out.Properties = map[string]any{}
	}

	if reqRaw, ok := schemaMap["required"]; ok {
		switch reqTyped := reqRaw.(type) {
		case []string:
			out.Required = slices.Clone(reqTyped)
		case []any:
			for _, item := range reqTyped {
				if s, ok := item.(string); ok {
					out.Required = append(out.Required, s)
				}
			}
		}
	}

	for k, v := range schemaMap {
		if k == "type" || k == "properties" || k == "required" {
			continue
		}
		if out.ExtraFields == nil {
			out.ExtraFields = make(map[string]any)
		}
		out.ExtraFields[k] = v
	}

	return out, nil
}

// toolSchemaCorpus returns declarations with the schema shapes ADK passes to
// Claude: none, genai schemas, MCP and FastMCP JSON schemas, jsonschema-go
// structs, and edge cases (unicode, HTML, keys special to the SDK's sjson
// paths, unusual values, large schemas, malformed properties and required).
func toolSchemaCorpus() []*genai.FunctionDeclaration {
	str := func(desc string) map[string]any { return map[string]any{"type": "string", "description": desc} }
	largeProps := make(map[string]any)
	var largeRequired []any
	for i := range 300 {
		name := fmt.Sprintf("field_%03d", i)
		largeProps[name] = map[string]any{"type": "STRING", "description": strings.Repeat("Long description. ", 10)}
		if i%3 == 0 {
			largeRequired = append(largeRequired, name)
		}
	}
	largeDefs := make(map[string]any)
	for i := range 50 {
		largeDefs[fmt.Sprintf("def_%02d", i)] = map[string]any{"type": "object", "properties": map[string]any{"v": map[string]any{"type": "number"}}}
	}
	manyKeys := map[string]any{"type": "object", "properties": map[string]any{"a": str("a")}}
	for i := range 12 {
		manyKeys[fmt.Sprintf("x-extra-%02d", i)] = i
	}

	return []*genai.FunctionDeclaration{
		{Name: "no_schema"},
		{Name: "genai_no_extras", Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"city": {Type: genai.TypeString, Description: "City name"},
				"days": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeInteger}},
			},
			Required: []string{"city"},
		}},
		{Name: "genai_description", Parameters: &genai.Schema{
			Type:        genai.TypeObject,
			Description: "Weather lookup arguments.",
			Properties:  map[string]*genai.Schema{"city": {Type: genai.TypeString}},
		}},
		{Name: "genai_description_title", Parameters: &genai.Schema{
			Type:        genai.TypeObject,
			Title:       "Booking",
			Description: "Booking arguments.",
			Properties: map[string]*genai.Schema{
				"mode": {Type: genai.TypeString, Enum: []string{"fast", "cheap"}},
				"note": {Type: genai.TypeString, Nullable: genai.Ptr(true)},
				"guest": {
					Type:       genai.TypeObject,
					Properties: map[string]*genai.Schema{"name": {Type: genai.TypeString}, "age": {Type: genai.TypeInteger, Minimum: genai.Ptr(0.0)}},
					Required:   []string{"name"},
				},
			},
			Required: []string{"mode", "guest"},
		}},
		{Name: "genai_many_extras", Parameters: &genai.Schema{
			Type:             genai.TypeObject,
			Title:            "Many",
			Description:      "Many top-level keywords.",
			Nullable:         genai.Ptr(false),
			MinProperties:    genai.Ptr[int64](1),
			MaxProperties:    genai.Ptr[int64](5),
			PropertyOrdering: []string{"b", "a"},
			Default:          map[string]any{"a": "x"},
			Example:          map[string]any{"b": 2},
			Properties:       map[string]*genai.Schema{"a": {Type: genai.TypeString}, "b": {Type: genai.TypeNumber, Format: "double"}},
		}},
		{Name: "mcp_typescript", ParametersJsonSchema: map[string]any{
			"type":                 "object",
			"properties":           map[string]any{"path": str("File path."), "recursive": map[string]any{"type": "boolean", "default": false}},
			"required":             []any{"path"},
			"additionalProperties": false,
			"$schema":              "http://json-schema.org/draft-07/schema#",
		}},
		{Name: "mcp_defs_ref", ParametersJsonSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"shipping": map[string]any{"$ref": "#/$defs/address"},
				"billing":  map[string]any{"anyOf": []any{map[string]any{"$ref": "#/$defs/address"}, map[string]any{"type": "null"}}},
			},
			"required": []any{"shipping"},
			"$defs": map[string]any{
				"address": map[string]any{
					"type":                 "object",
					"properties":           map[string]any{"street": str("Street."), "zip": map[string]any{"type": "string", "pattern": "^[0-9]{5}$"}},
					"required":             []any{"street"},
					"additionalProperties": false,
				},
			},
			"$schema":              "https://json-schema.org/draft/2020-12/schema",
			"additionalProperties": false,
			"description":          "Order arguments.",
		}},
		{Name: "fastmcp", ParametersJsonSchema: map[string]any{
			"type":       "object",
			"title":      "create_issueArguments",
			"properties": map[string]any{"title": map[string]any{"title": "Title", "type": "string"}, "labels": map[string]any{"title": "Labels", "type": "array", "items": map[string]any{"$ref": "#/$defs/Label"}}},
			"required":   []any{"title"},
			"$defs":      map[string]any{"Label": map[string]any{"title": "Label", "type": "string", "enum": []any{"bug", "feature"}}},
		}},
		{Name: "one_extra_additional_properties", ParametersJsonSchema: map[string]any{
			"type": "object", "properties": map[string]any{"q": str("Query.")}, "required": []any{"q"}, "additionalProperties": false,
		}},
		{Name: "one_extra_schema", ParametersJsonSchema: map[string]any{
			"type": "object", "properties": map[string]any{"q": str("Query.")}, "$schema": "http://json-schema.org/draft-07/schema#",
		}},
		{Name: "jsonschema_go_struct", ParametersJsonSchema: &jsonschema.Schema{
			Type:                 "object",
			Title:                "Args",
			Description:          "Arguments from a jsonschema-go schema.",
			Properties:           map[string]*jsonschema.Schema{"n": {Type: "integer", Description: "Count."}, "s": {Ref: "#/$defs/S"}},
			Required:             []string{"n"},
			AdditionalProperties: &jsonschema.Schema{Not: &jsonschema.Schema{}},
			Defs:                 map[string]*jsonschema.Schema{"S": {Type: "string", Enum: []any{"x", "y"}}},
		}},
		{Name: "unicode", ParametersJsonSchema: map[string]any{
			"type":        "object",
			"properties":  map[string]any{"zażółć": str("gęślą jaźń 📄 \u2028 \u2029")},
			"títle":       "Wyszukiwanie 🔎",
			"description": "naïve café — ünïcødé",
			"emoji🙂":      "\u2028",
		}},
		{Name: "html", ParametersJsonSchema: map[string]any{
			"type":        "object",
			"properties":  map[string]any{"q": str("<i>query</i> & more")},
			"description": "<b>bold</b> & 'quotes' \"dq\"",
			"x-<tag>&":    "<script>alert(1)</script>",
			"title":       "a & b",
		}},
		{Name: "special_keys", ParametersJsonSchema: map[string]any{
			"type": "object", "properties": map[string]any{},
			"a.b": 1, "a|b": 2, "#": 3, "@this": 4, "*": 5, "?": 6, ":colon": 7, `back\slash`: 8, `quote"key`: 9,
			"[x]": 10, "{y}": 11, "!true": 12, "0": 13, "123": 14, "-1": 15, "tab\tkey": 16, "  spaced  ": 17,
			"$": 18, "x-vendor": 19, "\x7f": 20, "nul\x00": 21, "a.b.c|d#e@f*g?h:i": 22,
		}},
		{Name: "values", ParametersJsonSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"v": map[string]any{"type": "NUMBER"}},
			"f_big":      1.5e300,
			"f_small":    1e-7,
			"f_frac":     0.1,
			"f_negzero":  math.Copysign(0, -1),
			"big_int":    int64(123456789012345678),
			"null":       nil,
			"empty_obj":  map[string]any{},
			"empty_arr":  []any{},
			"bool":       true,
			"nested":     []any{[]any{1, "two", map[string]any{"z": 1, "a": map[string]any{"type": "STRING"}}}},
		}},
		{Name: "large", ParametersJsonSchema: map[string]any{
			"type":                 "object",
			"properties":           largeProps,
			"required":             largeRequired,
			"$defs":                largeDefs,
			"$schema":              "http://json-schema.org/draft-07/schema#",
			"additionalProperties": false,
		}},
		{Name: "required_not_array", ParametersJsonSchema: map[string]any{
			"type": "object", "properties": map[string]any{"q": str("Query.")}, "required": "q", "description": "d", "title": "t",
		}},
		{Name: "required_mixed", ParametersJsonSchema: map[string]any{
			"properties": map[string]any{"a": str("a"), "b": str("b")}, "required": []any{"a", 1, "b", nil}, "description": "d", "title": "t",
		}},
		{Name: "required_empty", ParametersJsonSchema: map[string]any{
			"type": "object", "properties": map[string]any{"a": str("a")}, "required": []any{}, "description": "d", "title": "t",
		}},
		{Name: "properties_null", ParametersJsonSchema: map[string]any{"type": "object", "properties": nil, "description": "d", "title": "t"}},
		{Name: "properties_missing", ParametersJsonSchema: map[string]any{"type": "object", "description": "d", "title": "t"}},
		{Name: "properties_not_object", ParametersJsonSchema: map[string]any{"type": "object", "properties": []any{"x"}, "description": "d", "title": "t"}},
		{Name: "type_array", ParametersJsonSchema: map[string]any{
			"type": []any{"OBJECT", "NULL"}, "properties": map[string]any{"x": map[string]any{"type": "INTEGER"}}, "description": "d", "title": "t",
		}},
		{Name: "many_keys", ParametersJsonSchema: manyKeys},
		{Name: "raw_message", ParametersJsonSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"$comment":"raw","additionalProperties":true,"examples":[{"q":"a"}]}`)},
	}
}

// jsonMember is one member of a JSON object: its raw key and value bytes as
// they appear in the document, and the decoded key.
type jsonMember struct {
	key, rawKey, rawValue string
}

func objectMembers(t *testing.T, data []byte) []jsonMember {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(data))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		t.Fatalf("%s: not a JSON object (token %v, err %v)", data, tok, err)
	}
	var members []jsonMember
	for dec.More() {
		start := dec.InputOffset()
		tok, err := dec.Token()
		if err != nil {
			t.Fatalf("%s: reading key: %v", data, err)
		}
		rawKey := bytes.TrimLeft(data[start:dec.InputOffset()], ", \t\r\n")
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			t.Fatalf("%s: reading value of %q: %v", data, tok, err)
		}
		members = append(members, jsonMember{key: tok.(string), rawKey: string(rawKey), rawValue: string(value)})
	}
	return members
}

func decodeJSON(t *testing.T, data []byte) any {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		t.Fatalf("decoding %s: %v", data, err)
	}
	return v
}

func errString(err error) string {
	if err == nil {
		return "<nil>"
	}
	return err.Error()
}

func isSDKSchemaField(key string) bool {
	return key == "properties" || key == "required" || key == "type"
}

// TestBuildToolInputSchemaMatchesReference checks that every schema in the
// corpus serializes to the same JSON object as the reference, byte for byte
// with at most one extra keyword, and otherwise with the same members in
// which only the extra keywords are reordered, into sorted key order. Keys
// containing <, > or & are the one byte-level difference: the reference
// writes them raw, the deterministic path escapes them as encoding/json does.
func TestBuildToolInputSchemaMatchesReference(t *testing.T) {
	for _, decl := range toolSchemaCorpus() {
		t.Run(decl.Name, func(t *testing.T) {
			ref, refErr := buildToolInputSchemaReference(decl)
			got, gotErr := buildToolInputSchema(decl)
			if errString(gotErr) != errString(refErr) {
				t.Fatalf("buildToolInputSchema() error = %v, reference error = %v", gotErr, refErr)
			}
			refJSON, refErr := ref.MarshalJSON()
			gotJSON, gotErr := got.MarshalJSON()
			if refErr != nil || gotErr != nil {
				t.Fatalf("MarshalJSON() error = %v, reference error = %v", gotErr, refErr)
			}

			if len(ref.ExtraFields) <= 1 && !bytes.Equal(gotJSON, refJSON) {
				t.Fatalf("with %d extra keys got\n%s\nwant (byte-identical)\n%s", len(ref.ExtraFields), gotJSON, refJSON)
			}
			if diff := cmp.Diff(decodeJSON(t, refJSON), decodeJSON(t, gotJSON)); diff != "" {
				t.Fatalf("decoded schema mismatch (-reference +got):\n%s", diff)
			}

			refMembers, gotMembers := objectMembers(t, refJSON), objectMembers(t, gotJSON)
			if len(gotMembers) != len(refMembers) {
				t.Fatalf("got %d members, reference %d", len(gotMembers), len(refMembers))
			}
			nSDK := 0
			for nSDK < len(refMembers) && isSDKSchemaField(refMembers[nSDK].key) {
				nSDK++
			}
			if diff := cmp.Diff(refMembers[:nSDK], gotMembers[:nSDK], cmp.AllowUnexported(jsonMember{})); diff != "" {
				t.Errorf("SDK fields differ (-reference +got):\n%s", diff)
			}
			refExtras := slices.Clone(refMembers[nSDK:])
			slices.SortFunc(refExtras, func(a, b jsonMember) int { return strings.Compare(a.key, b.key) })
			gotExtras := gotMembers[nSDK:]
			if len(gotExtras) != len(ref.ExtraFields) {
				t.Errorf("got %d extra keys, want %d", len(gotExtras), len(ref.ExtraFields))
			}
			for i := range refExtras {
				r, g := refExtras[i], gotExtras[i]
				if g.key != r.key || g.rawValue != r.rawValue {
					t.Errorf("extra key %d = %q: %s, want %q: %s (sorted reference)", i, g.key, g.rawValue, r.key, r.rawValue)
				}
				if !strings.ContainsAny(r.key, "<>&") && g.rawKey != r.rawKey {
					t.Errorf("extra key %d written as %s, reference %s", i, g.rawKey, r.rawKey)
				}
			}

			for range 200 {
				again, err := buildToolInputSchema(decl)
				if err != nil {
					t.Fatalf("buildToolInputSchema() error = %v", err)
				}
				againJSON, err := again.MarshalJSON()
				if err != nil {
					t.Fatalf("MarshalJSON() error = %v", err)
				}
				if !bytes.Equal(againJSON, gotJSON) {
					t.Fatalf("serialization changed between identical calls:\n%s\n%s", gotJSON, againJSON)
				}
			}
		})
	}
}

// TestBuildToolInputSchemaErrorsMatchReference checks that schemas the
// reference rejects, when building or when the SDK serializes them, fail the
// same way.
func TestBuildToolInputSchemaErrorsMatchReference(t *testing.T) {
	for _, decl := range []*genai.FunctionDeclaration{
		{Name: "empty_key_and_title", ParametersJsonSchema: map[string]any{"type": "object", "properties": map[string]any{}, "": 1, "title": "t"}},
		{Name: "empty_key_only", ParametersJsonSchema: map[string]any{"type": "object", "": 1}},
		{Name: "empty_key_many", ParametersJsonSchema: map[string]any{"": 1, "a": 2, "b": 3, "c": 4}},
		{Name: "not_an_object", ParametersJsonSchema: []any{1, 2}},
		{Name: "string_schema", ParametersJsonSchema: "object"},
		{Name: "null_schema", ParametersJsonSchema: json.RawMessage(`null`)},
		{Name: "unmarshalable", ParametersJsonSchema: map[string]any{"type": "object", "f": func() {}}},
	} {
		t.Run(decl.Name, func(t *testing.T) {
			ref, refErr := buildToolInputSchemaReference(decl)
			got, gotErr := buildToolInputSchema(decl)
			if errString(gotErr) != errString(refErr) {
				t.Fatalf("buildToolInputSchema() error = %v, reference error = %v", gotErr, refErr)
			}
			if refErr != nil {
				return
			}
			refJSON, refErr := ref.MarshalJSON()
			gotJSON, gotErr := got.MarshalJSON()
			if refErr == nil {
				t.Fatalf("reference MarshalJSON() = %s, want an error", refJSON)
			}
			if errString(gotErr) != errString(refErr) {
				t.Fatalf("MarshalJSON() = %s, %v; reference error = %v", gotJSON, gotErr, refErr)
			}
		})
	}
}

// TestToolSchemaRequestBodyMatchesReference checks whole request bodies: the
// body GenerateContent sends equals, as a JSON value, the body built with the
// reference schemas, byte for byte when no tool has more than one extra
// keyword, and is identical across 200 calls.
func TestToolSchemaRequestBodyMatchesReference(t *testing.T) {
	corpus := toolSchemaCorpus()
	corpusReq := benchToolsRequest(0, 0)
	corpusReq.Config.Tools = []*genai.Tool{{FunctionDeclarations: corpus[:4]}, nil, {FunctionDeclarations: corpus[4:]}}

	for _, tc := range []struct {
		name       string
		req        *model.LLMRequest
		byteEquals bool
	}{
		{"extras=0", benchToolsRequest(20, 0), true},
		{"extras=1", benchToolsRequest(20, 1), true},
		{"extras=2", benchToolsRequest(20, 2), false},
		{"extras=5", benchToolsRequest(20, 5), false},
		{"corpus", corpusReq, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := newRequestBodyRecorder(t)
			m := newPromptCachingModel(t, rec.srv.URL)

			params, _, _, err := m.buildMessageParams(t.Context(), tc.req)
			if err != nil {
				t.Fatalf("buildMessageParams() error = %v", err)
			}
			gotBody, err := params.MarshalJSON()
			if err != nil {
				t.Fatalf("MarshalJSON() error = %v", err)
			}

			var decls []*genai.FunctionDeclaration
			for _, tool := range tc.req.Config.Tools {
				if tool != nil {
					decls = append(decls, tool.FunctionDeclarations...)
				}
			}
			if len(decls) != len(params.Tools) {
				t.Fatalf("got %d tools, want %d", len(params.Tools), len(decls))
			}
			refParams := params
			refParams.Tools = make([]anthropic.ToolUnionParam, len(params.Tools))
			for i, tool := range params.Tools {
				refTool := *tool.OfTool
				if refTool.InputSchema, err = buildToolInputSchemaReference(decls[i]); err != nil {
					t.Fatalf("buildToolInputSchemaReference(%q) error = %v", decls[i].Name, err)
				}
				refParams.Tools[i] = anthropic.ToolUnionParam{OfTool: &refTool}
			}
			refBody, err := refParams.MarshalJSON()
			if err != nil {
				t.Fatalf("reference MarshalJSON() error = %v", err)
			}

			if tc.byteEquals && !bytes.Equal(gotBody, refBody) {
				t.Fatalf("got body\n%s\nwant (byte-identical)\n%s", gotBody, refBody)
			}
			if diff := cmp.Diff(decodeJSON(t, refBody), decodeJSON(t, gotBody)); diff != "" {
				t.Fatalf("decoded body mismatch (-reference +got):\n%s", diff)
			}

			for range 200 {
				for _, err := range m.GenerateContent(t.Context(), tc.req, false) {
					if err != nil {
						t.Fatalf("GenerateContent() error = %v", err)
					}
				}
			}
			if distinct, changes := rec.counts(); distinct != 1 || changes != 0 {
				t.Errorf("200 identical calls sent %d distinct bodies with %d changes, want 1 and 0", distinct, changes)
			}
			if rec.bodies[0] != string(gotBody) {
				t.Errorf("sent body\n%s\nwant\n%s", rec.bodies[0], gotBody)
			}
		})
	}
}
