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

// Package globalinstruction provides a plugin that puts one instruction at the
// start of the system instruction of every model request a runner makes, for
// every agent of that runner: an application-wide identity, personality or set
// of rules, configured once instead of on each agent.
//
// Register the plugin with a runner:
//
//	p, err := globalinstruction.New(globalinstruction.Config{
//		Instruction: "You are a support assistant for {company}.",
//	})
//	if err != nil {
//		return err
//	}
//	r, err := runner.New(runner.Config{
//		AppName:        "app",
//		Agent:          rootAgent,
//		SessionService: session.InMemoryService(),
//		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
//	})
//
// The plugin runs before every model call, after the request is fully built.
// It adds the instruction as a new first part of the request's system
// instruction and keeps the existing parts after it, so the instruction comes
// before the agent's own instructions, including the text of
// [llmagent.Config.GlobalInstruction]. adk-python deprecates that field in
// favor of this plugin. Plugins registered earlier see the request without the
// instruction; plugins registered later and the agent's own before-model
// callbacks see it with the instruction.
package globalinstruction

import (
	"errors"
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/util/instructionutil"
)

const defaultName = "global_instruction"

// Config configures a global instruction plugin.
// The zero value is valid and represents an inert plugin that skips processing,
// using "global_instruction" as its default Name.
type Config struct {
	// Name is the plugin's name, unique among the plugins of one runner.
	// Empty selects "global_instruction".
	Name string

	// Instruction is the string template for the global instruction.
	// Placeholders like {key} are replaced with session state values.
	//
	// The template follows the rules of [llmagent.Config.Instruction]:
	// {key} reads the session state, {key?} reads as "" when the key is
	// absent, {artifact.name} inserts the text of an artifact, and a
	// placeholder whose name is not a valid state name is left as written.
	// A missing non-optional key or artifact makes the callback fail, which
	// fails the model call. The template is resolved again before every model
	// call.
	Instruction string

	// InstructionProvider is the provider function for the global instruction.
	// Exactly one of Instruction and InstructionProvider may be set; New returns
	// ErrMultipleProviders otherwise.
	//
	// It is called before every model call, with a read-only view of that
	// call's context, and its result is used as given, without placeholder
	// substitution; call [instructionutil.InjectSessionState] on the context
	// it receives to substitute them. Its error is returned unchanged from the
	// callback and fails the model call. It must be safe for concurrent use.
	InstructionProvider llmagent.InstructionProvider
}

// ErrMultipleProviders is returned by New when both Instruction and InstructionProvider are set.
var ErrMultipleProviders = errors.New("globalinstruction: cannot provide both Instruction and InstructionProvider")

// New returns a plugin that prepends an instruction to every model request of
// every agent in the runner. The plugin runs before every model call.
//
// The plugin's before-model callback resolves the instruction for the call.
// When the result is empty it leaves the request as it is. Otherwise it sets
// the request's system instruction to a copy of the existing one whose parts
// are a new text part holding the instruction followed by the existing parts;
// the copy keeps the other fields of the content, such as its role, and the
// existing content is not modified. With no system instruction (or no request
// config), the system instruction becomes a "user" content holding only the
// instruction. The callback never returns a response, so the model is still
// called. When resolving fails, the callback returns the error and leaves the
// request unchanged.
//
// New fails, returning a nil plugin and ErrMultipleProviders, when both
// Instruction and InstructionProvider are set. cfg is read once and not
// retained. The returned plugin keeps no per-call state and is safe for
// concurrent use when InstructionProvider is.
func New(cfg Config) (*plugin.Plugin, error) {
	if cfg.Instruction != "" && cfg.InstructionProvider != nil {
		return nil, ErrMultipleProviders
	}
	name := cfg.Name
	if name == "" {
		name = defaultName
	}
	g := &globalInstruction{instruction: cfg.Instruction, provider: cfg.InstructionProvider}
	return plugin.New(plugin.Config{
		Name:                name,
		BeforeModelCallback: g.beforeModel,
	})
}

// globalInstruction holds a plugin's configuration. It is never modified after
// New, so one value serves concurrent model calls.
type globalInstruction struct {
	instruction string
	provider    llmagent.InstructionProvider
}

func (g *globalInstruction) beforeModel(ctx agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	text, err := g.resolve(ctx)
	if err != nil {
		return nil, err
	}
	if text == "" {
		return nil, nil
	}
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	req.Config.SystemInstruction = prepend(req.Config.SystemInstruction, text)
	return nil, nil
}

// resolve returns the instruction for one model call. Both paths read through
// icontext.NewReadonlyContext, the context type instructionutil.InjectSessionState
// accepts, so a provider that calls it works as it does for an agent's own
// providers.
func (g *globalInstruction) resolve(ctx agent.Context) (string, error) {
	if g.provider != nil {
		return g.provider(icontext.NewReadonlyContext(ctx))
	}
	text, err := instructionutil.InjectSessionState(icontext.NewReadonlyContext(ctx), g.instruction)
	if err != nil {
		return "", fmt.Errorf("failed to inject session state into global instruction: %w", err)
	}
	return text, nil
}

// prepend returns a copy of si whose parts are a text part holding text
// followed by si's parts. It does not modify si, which an earlier plugin may
// share between requests.
func prepend(si *genai.Content, text string) *genai.Content {
	if si == nil {
		return genai.NewContentFromText(text, genai.RoleUser)
	}
	out := *si
	out.Parts = append([]*genai.Part{genai.NewPartFromText(text)}, si.Parts...)
	return &out
}
