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
	"fmt"
	"maps"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
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
					tool.FunctionDeclarations[i] = modifiedDeclaration(cfg, decl)
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
func modifiedDeclaration(cfg FunctionCallModifierConfig, decl *genai.FunctionDeclaration) *genai.FunctionDeclaration {
	modified := *decl
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

	if cfg.OverrideDescription != nil {
		modified.Description = cfg.OverrideDescription(decl.Description)
	}
	return &modified
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
