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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
)

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

func beforeModelCallback(cfg FunctionCallModifierConfig) func(agent.CallbackContext, *model.LLMRequest) (*model.LLMResponse, error) {
	return func(ctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
		if req.Config == nil || len(req.Config.Tools) == 0 {
			return nil, nil
		}

		for _, tool := range req.Config.Tools {
			if tool.FunctionDeclarations == nil {
				continue
			}
			for _, decl := range tool.FunctionDeclarations {
				_, exists := req.Tools[decl.Name]
				if !exists {
					continue
				}

				shouldAddArgs := cfg.Predicate(decl.Name)

				if shouldAddArgs {
					if decl.Parameters == nil {
						decl.Parameters = &genai.Schema{Type: "OBJECT", Properties: map[string]*genai.Schema{}}
					}
					if decl.Parameters.Properties == nil {
						decl.Parameters.Properties = map[string]*genai.Schema{}
					}

					maps.Copy(decl.Parameters.Properties, cfg.Args)

					if cfg.OverrideDescription != nil {
						decl.Description = cfg.OverrideDescription(decl.Description)
					}
				}
			}
		}
		return nil, nil
	}
}

func afterModelCallback(cfg FunctionCallModifierConfig) func(agent.CallbackContext, *model.LLMResponse, error) (*model.LLMResponse, error) {
	return func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
		if llmResponseError != nil {
			return nil, llmResponseError // Pass through error
		}
		if llmResponse == nil || llmResponse.Content == nil || len(llmResponse.Content.Parts) == 0 {
			return llmResponse, nil // No function calls to process
		}

		for _, part := range llmResponse.Content.Parts {
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
