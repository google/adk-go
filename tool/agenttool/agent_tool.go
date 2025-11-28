// Copyright 2025 Google LLC
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

// Package agenttool provides a tool that allows an agent to call another agent.
// This enables composition of agents, which can be useful for scenarios where
// different types of `genai` tools cannot be used together.
package agenttool

import (
	"encoding/json"
	"fmt"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// agentTool implements a tool that allows an agent to call another agent.
type agentTool struct {
	agent             agent.Agent
	skipSummarization bool
}

// Config holds the configuration for an agent tool.
type Config struct {
	// SkipSummarization, if true, will cause the agent to skip summarization
	// after the sub-agent finishes execution.
	SkipSummarization bool
}

// New creates a new agent tool.
// If cfg is nil, skipSummarization defaults to false.
func New(agent agent.Agent, cfg *Config) tool.Tool {
	if cfg == nil {
		return &agentTool{
			agent:             agent,
			skipSummarization: false,
		}
	}
	return &agentTool{
		agent:             agent,
		skipSummarization: cfg.SkipSummarization,
	}
}

// Name implements tool.Tool.
func (t *agentTool) Name() string {
	return t.agent.Name()
}

// Description implements tool.Tool.
func (t *agentTool) Description() string {
	return t.agent.Description()
}

// IsLongRunning implements tool.Tool.
func (t *agentTool) IsLongRunning() bool {
	return false
}

// Declaration returns the function declaration for the wrapped agent.
// It generates a function declaration based on the agent's input schema.
// If the agent does not have an input schema, a default schema with a
// "request" string parameter is used.
func (t *agentTool) Declaration() *genai.FunctionDeclaration {
	decl := &genai.FunctionDeclaration{
		Name:        t.Name(),
		Description: t.Description(),
	}

	var agentInputSchema *genai.Schema
	llmAgent, ok := t.agent.(llminternal.Agent)
	if ok && llmAgent != nil {
		// TODO - understand what build_function_declaration does in python and apply if needed.
		internalLlmAgent, ok := t.agent.(llminternal.Agent)
		if !ok {
			return nil
		}
		agentInputSchema = llminternal.Reveal(internalLlmAgent).InputSchema
	}

	if agentInputSchema != nil {
		decl.Parameters = agentInputSchema
	} else {
		decl.Parameters = &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"request": {Type: "STRING"},
			},
			Required: []string{"request"},
		}
	}
	// TODO - understand how _api_variant affects response type.
	return decl
}

// Run executes the wrapped agent with the provided arguments.
// It creates a new session for the sub-agent, runs the agent, and returns
// the final result.
func (t *agentTool) Run(toolCtx tool.Context, args any) (map[string]any, error) {
	margs, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("agentTool expects map[string]any arguments, got %T", args)
	}

	if t.skipSummarization {
		if actions := toolCtx.Actions(); actions != nil {
			actions.SkipSummarization = true
		}
	}

	var agentInputSchema *genai.Schema
	llmAgent, ok := t.agent.(llminternal.Agent)
	isLllmAgent := (ok && llmAgent != nil)
	if isLllmAgent {
		internalLlmAgent, ok := t.agent.(llminternal.Agent)
		if !ok {
			return nil, fmt.Errorf("internal error: failed to convert to llm agent")
		}
		agentInputSchema = llminternal.Reveal(internalLlmAgent).InputSchema
	}

	var content *genai.Content
	var err error
	if agentInputSchema != nil {
		if err = utils.ValidateMapOnSchema(margs, agentInputSchema, true); err != nil {
			return nil, fmt.Errorf("argument validation failed for agent %s: %w", t.agent.Name(), err)
		}
		jsonData, err := json.Marshal(margs)
		if err != nil {
			return nil, fmt.Errorf("error serializing tool arguments for agent %s: %w", t.agent.Name(), err)
		}
		content = genai.NewContentFromText(string(jsonData), genai.RoleUser)
	} else {
		input, ok := margs["request"]
		if !ok {
			return nil, fmt.Errorf("missing required argument 'request' for agent %s", t.agent.Name())
		}
		inputText, ok := input.(string)
		if !ok {
			// Try to convert to string if not already one
			inputText = fmt.Sprint(input)
		}
		content = genai.NewContentFromText(inputText, genai.RoleUser)
	}

	sessionService := session.InMemoryService()

	r, err := runner.New(runner.Config{
		AppName:        t.agent.Name(),
		Agent:          t.agent,
		SessionService: sessionService,
		// TODO - use forwarding_artifact_service as in python.
		ArtifactService: artifact.InMemoryService(),
		MemoryService:   memory.InMemoryService(),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create runner")
	}

	stateMap := make(map[string]any)

	for k, v := range toolCtx.State().All() {
		// Filter out adk internal states.
		if !strings.HasPrefix(k, "_adk") {
			stateMap[k] = v
		}
	}

	subSession, err := sessionService.Create(toolCtx, &session.CreateRequest{
		AppName: t.agent.Name(),
		UserID:  toolCtx.UserID(),
		State:   stateMap,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create session for sub-agent %s: %w", t.agent.Name(), err)
	}

	// TODO(dpasiukevich): verify agent loop termination.
	eventCh := r.Run(toolCtx, subSession.Session.UserID(), subSession.Session.ID(), content, agent.RunConfig{
		StreamingMode: agent.StreamingModeSSE,
	})

	var lastEvent *session.Event
	for event, err := range eventCh {
		if err != nil {
			return nil, fmt.Errorf("error during execution of sub-agent %s: %w", t.agent.Name(), err)
		}
		if event.LLMResponse.Content != nil {
			lastEvent = event
		}
	}

	if lastEvent == nil {
		return map[string]any{}, nil
	}

	lastContent := lastEvent.LLMResponse.Content
	var textParts []string
	for _, part := range lastContent.Parts {
		if part != nil && part.Text != "" {
			textParts = append(textParts, part.Text)
		}
	}
	outputText := strings.Join(textParts, "\n")

	if outputText == "" {
		return map[string]any{}, nil
	}
	if isLllmAgent {
		internalLlmAgent, ok := t.agent.(llminternal.Agent)
		if !ok {
			return nil, fmt.Errorf("internal error: failed to convert to llm agent")
		}
		if agentOutputSchema := llminternal.Reveal(internalLlmAgent).OutputSchema; agentOutputSchema != nil {
			// Assuming schemautils.ValidateOutputSchema parses the JSON string outputText
			// and validates it against the agentOutputSchema, returning a map[string]any.
			parsedOutput, err := utils.ValidateOutputSchema(outputText, agentOutputSchema)
			if err != nil {
				return nil, fmt.Errorf("output validation failed for sub-agent %s: %w", t.agent.Name(), err)
			}
			return parsedOutput, nil
		}
	}

	return map[string]any{"result": outputText}, nil
}

// ProcessRequest adds the agent tool's function declaration to the LLM request.
func (t *agentTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	// TODO extract this function somewhere else, simillar operations are done for
	// other tools with function declaration.
	if req.Tools == nil {
		req.Tools = make(map[string]any)
	}

	name := t.Name()
	if _, ok := req.Tools[name]; ok {
		return fmt.Errorf("duplicate tool: %q", name)
	}
	req.Tools[name] = t

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if decl := t.Declaration(); decl == nil {
		return nil
	}
	var funcTool *genai.Tool
	for _, tool := range req.Config.Tools {
		if tool != nil && tool.FunctionDeclarations != nil {
			funcTool = tool
			break
		}
	}
	if funcTool == nil {
		req.Config.Tools = append(req.Config.Tools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{t.Declaration()},
		})
	} else {
		funcTool.FunctionDeclarations = append(funcTool.FunctionDeclarations, t.Declaration())
	}
	return nil
}
