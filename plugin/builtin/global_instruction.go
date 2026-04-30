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

// Package builtin provides plugin implementations that ship with ADK. They
// are thin factories that return a *plugin.Plugin configured with the
// appropriate callback hooks.
//
// Mirrors adk-python's google.adk.plugins package (DebugLoggingPlugin,
// LoggingPlugin, GlobalInstructionPlugin, SaveFilesAsArtifactsPlugin, etc).
package builtin

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
)

// GlobalInstructionConfig configures GlobalInstruction.
type GlobalInstructionConfig struct {
	// Name is the plugin name. Defaults to "global_instruction".
	Name string

	// Instruction is the static text to prepend as a system instruction.
	// Mutually exclusive with InstructionFunc; if both are set,
	// InstructionFunc wins.
	Instruction string

	// InstructionFunc is a dynamic provider invoked once per LLM request.
	// It receives a CallbackContext (so it can read app/user/session state)
	// and returns the instruction text. Return "" to skip injection for this
	// call. Mirrors adk-python's InstructionProvider.
	InstructionFunc func(agent.CallbackContext) (string, error)
}

// GlobalInstruction returns a plugin that prepends a global instruction to
// every LLM request's SystemInstruction. Useful for app-wide identity,
// safety guidance, or formatting rules.
//
// If the request already has a SystemInstruction with text, the global
// instruction is prepended with a blank-line separator. If there is no
// SystemInstruction, one is created.
//
// Mirrors adk-python's GlobalInstructionPlugin.
func GlobalInstruction(cfg GlobalInstructionConfig) (*plugin.Plugin, error) {
	name := cfg.Name
	if name == "" {
		name = "global_instruction"
	}
	resolve := func(cctx agent.CallbackContext) (string, error) {
		if cfg.InstructionFunc != nil {
			return cfg.InstructionFunc(cctx)
		}
		return cfg.Instruction, nil
	}
	before := func(cctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
		text, err := resolve(cctx)
		if err != nil {
			return nil, fmt.Errorf("global_instruction: resolve: %w", err)
		}
		if text == "" {
			return nil, nil
		}
		if req.Config == nil {
			req.Config = &genai.GenerateContentConfig{}
		}
		existing := req.Config.SystemInstruction
		switch {
		case existing == nil:
			req.Config.SystemInstruction = &genai.Content{
				Parts: []*genai.Part{{Text: text}},
			}
		default:
			// Prepend a Part with the global instruction so it leads.
			parts := append([]*genai.Part{{Text: text}}, existing.Parts...)
			existing.Parts = parts
		}
		return nil, nil
	}
	return plugin.New(plugin.Config{
		Name:                name,
		BeforeModelCallback: llmagent.BeforeModelCallback(before),
	})
}
