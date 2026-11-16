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

// Package functioncallmodifier provides a plugin to modify function calls.
package functioncallmodifier

import (
	"encoding/json"
	"fmt"
	"maps"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/typeutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
)

// FunctionCallModifierConfig configures a FunctionCallModifierPlugin.
type FunctionCallModifierConfig struct {
	Predicate           func(toolName string) bool
	Args                map[string]*genai.Schema
	OverrideDescription func(originalDescription string) string
}

// NewPlugin creates a FunctionCallModifierPlugin.
func NewPlugin(cfg FunctionCallModifierConfig) (*plugin.Plugin, error) {
	return plugin.New(plugin.Config{
		Name:                "FunctionCallModifierPlugin",
		BeforeModelCallback: beforeModelCallback(cfg),
		AfterModelCallback:  afterModelCallback(cfg),
	})
}

// MustNewPlugin is like NewPlugin but panics if there is an error.
func MustNewPlugin(cfg FunctionCallModifierConfig) *plugin.Plugin {
	p, err := NewPlugin(cfg)
	if err != nil {
		panic(err)
	}
	return p
}

func beforeModelCallback(cfg FunctionCallModifierConfig) func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
	return func(ctx agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
		if req.Config == nil || len(req.Config.Tools) == 0 {
			return nil, nil
		}

		for _, tool := range req.Config.Tools {
			if tool.FunctionDeclarations == nil {
				continue
			}
			for i, decl := range tool.FunctionDeclarations {
				_, exists := req.Tools[decl.Name]
				if !exists {
					continue
				}

				if cfg.Predicate(decl.Name) {
					modified, err := modifiedDeclaration(cfg, decl)
					if err != nil {
						return nil, err
					}
					tool.FunctionDeclarations[i] = modified
				}
			}
		}
		return nil, nil
	}
}

// modifiedDeclaration returns a copy of decl with the configured arguments
// and description applied, and leaves decl alone. The task and single_turn
// tools an LLM agent installs for its sub-agents hand out the same declaration
// on every call, so editing it in place would wrap the description again on
// every request and race between concurrent ones. An agent tool built on an
// agent with an InputSchema rebuilds its declaration on each call but uses that
// schema as its parameters, so an in-place edit would add the arguments to the
// agent's own InputSchema.
//
// The arguments go into the form of parameters decl has, so the result sets
// exactly one of Parameters and ParametersJsonSchema. A ParametersJsonSchema
// that is a *jsonschema.Schema (function tools, transfer_to_agent) or a
// map[string]any (MCP tools) gets the arguments as JSON Schema properties; a
// nil one of either type counts as unset, and then the arguments go into
// Parameters. An argument replaces a property of the same name. Any other
// ParametersJsonSchema type, or an argument that does not convert to JSON
// Schema, is an error.
func modifiedDeclaration(cfg FunctionCallModifierConfig, decl *genai.FunctionDeclaration) (*genai.FunctionDeclaration, error) {
	modified := *decl
	if cfg.OverrideDescription != nil {
		modified.Description = cfg.OverrideDescription(decl.Description)
	}

	if !isUnsetJSONSchema(decl.ParametersJsonSchema) {
		schema, err := jsonSchemaWithArgs(decl.ParametersJsonSchema, cfg.Args)
		if err != nil {
			return nil, fmt.Errorf("function call modifier: tool %q: %w", decl.Name, err)
		}
		modified.ParametersJsonSchema = schema
		return &modified, nil
	}

	params := genai.Schema{Type: "OBJECT"}
	if decl.Parameters != nil {
		params = *decl.Parameters
	}
	params.Properties = maps.Clone(params.Properties)
	if params.Properties == nil {
		params.Properties = make(map[string]*genai.Schema, len(cfg.Args))
	}
	maps.Copy(params.Properties, cfg.Args)
	modified.Parameters = &params
	modified.ParametersJsonSchema = nil
	return &modified, nil
}

// isUnsetJSONSchema reports whether schema, a ParametersJsonSchema, declares
// no parameters: a nil interface, or a nil value of a type the plugin extends.
func isUnsetJSONSchema(schema any) bool {
	switch s := schema.(type) {
	case nil:
		return true
	case *jsonschema.Schema:
		return s == nil
	case map[string]any:
		return s == nil
	}
	return false
}

// jsonSchemaWithArgs returns a copy of schema, a non-nil ParametersJsonSchema,
// with args added to its properties.
func jsonSchemaWithArgs(schema any, args map[string]*genai.Schema) (any, error) {
	switch s := schema.(type) {
	case *jsonschema.Schema:
		return schemaWithArgs(s, args)
	case map[string]any:
		return mapSchemaWithArgs(s, args)
	}
	return nil, fmt.Errorf("parameters of type %T cannot take arguments", schema)
}

// schemaWithArgs returns a copy of s with args added to its properties. s and
// its Properties map are not modified.
func schemaWithArgs(s *jsonschema.Schema, args map[string]*genai.Schema) (*jsonschema.Schema, error) {
	cp := *s
	cp.Properties = maps.Clone(s.Properties)
	if cp.Properties == nil {
		cp.Properties = make(map[string]*jsonschema.Schema, len(args))
	}
	for name, arg := range args {
		prop, err := argumentJSONSchema(name, arg)
		if err != nil {
			return nil, err
		}
		var converted jsonschema.Schema
		if err := remarshal(prop, &converted); err != nil {
			return nil, fmt.Errorf("argument %q: %w", name, err)
		}
		cp.Properties[name] = &converted
	}
	return &cp, nil
}

// mapSchemaWithArgs returns a copy of s, a JSON Schema decoded into a map,
// with args added to its "properties" map. s and its "properties" map are not
// modified. A "properties" value that is not a map[string]any is an error.
func mapSchemaWithArgs(s map[string]any, args map[string]*genai.Schema) (map[string]any, error) {
	props := make(map[string]any, len(args))
	if existing, ok := s["properties"]; ok {
		existingMap, isMap := existing.(map[string]any)
		if !isMap {
			return nil, fmt.Errorf(`"properties" of type %T is not a map`, existing)
		}
		maps.Copy(props, existingMap)
	}
	for name, arg := range args {
		prop, err := argumentJSONSchema(name, arg)
		if err != nil {
			return nil, err
		}
		props[name] = prop
	}
	cp := maps.Clone(s)
	cp["properties"] = props
	return cp, nil
}

// argumentJSONSchema converts the argument schema arg, named name, to a JSON
// Schema map.
func argumentJSONSchema(name string, arg *genai.Schema) (map[string]any, error) {
	prop, err := typeutil.GenaiSchemaToJSONMap(arg)
	if err != nil {
		return nil, fmt.Errorf("argument %q: %w", name, err)
	}
	return prop, nil
}

// remarshal decodes the JSON encoding of from into to.
func remarshal(from, to any) error {
	data, err := json.Marshal(from)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, to)
}

func afterModelCallback(cfg FunctionCallModifierConfig) func(agent.Context, *model.LLMResponse, error) (*model.LLMResponse, error) {
	return func(ctx agent.Context, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
		if llmResponseError != nil {
			return nil, llmResponseError // Pass through error
		}
		// Nothing to rewrite. A non-nil response here would replace the model's
		// own and end the after-model chain, so the agent's callbacks and any
		// later plugin's would never see it.
		if llmResponse == nil || llmResponse.Content == nil || len(llmResponse.Content.Parts) == 0 {
			return nil, nil
		}

		for _, part := range llmResponse.Content.Parts {
			if part == nil {
				continue
			}
			if fc := part.FunctionCall; fc != nil {
				if !cfg.Predicate(fc.Name) {
					continue
				}
				for name := range cfg.Args {
					arg, hasArg := fc.Args[name]
					if !hasArg {
						continue
					}
					delete(fc.Args, name)
					stateKey := fmt.Sprintf("%s/%s", fc.ID, name)
					if err := ctx.State().Set(stateKey, arg); err != nil {
						return nil, fmt.Errorf("failed to set state: %w", err)
					}
				}
			}
		}
		return nil, nil
	}
}
