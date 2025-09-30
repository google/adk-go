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

package agenttool

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"google.golang.org/adk/agent" // Assuming internal package for schema validation
	"google.golang.org/adk/artifactservice"
	agentinternal "google.golang.org/adk/internal/agent"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/memoryservice"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"  // Assuming runner package
	"google.golang.org/adk/session" // Assuming session package for Event type
	"google.golang.org/adk/sessionservice"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// LlmAgent represents an agent powered by an LLM.
// This interface is assumed based on the Java code's structur

// AgentTool implements a tool that allows an agent to call another agent.
type AgentTool struct {
	agent             agent.Agent
	skipSummarization bool
	name              string
	description       string
}

// NewAgentTool creates a new AgentTool.
func NewAgentTool(agent agent.Agent, skipSummarization bool) tool.Tool {
	return &AgentTool{
		agent:             agent,
		skipSummarization: skipSummarization,
		name:              agent.Name(),
		description:       agent.Description(),
	}
}

// NewAgentToolDefault creates a new AgentTool with skipSummarization set to false.
func NewAgentToolDefault(agent agent.Agent) tool.Tool {
	return NewAgentTool(agent, false)
}

// Name implements tool.Tool.
func (t *AgentTool) Name() string {
	return t.name
}

// Description implements tool.Tool.
func (t *AgentTool) Description() string {
	return t.description
}

// IsLongRunning implements tool.Tool.
func (t *AgentTool) IsLongRunning() bool {
	return false
}

// Declaration implements tool.Tool.
func (t *AgentTool) Declaration() *genai.FunctionDeclaration {
	decl := &genai.FunctionDeclaration{
		Name:        t.Name(),
		Description: t.Description(),
	}

	var agentInputSchema *genai.Schema
	internalAgent, ok := t.agent.(agentinternal.Agent)
	if !ok {
		return nil
	}
	if agentinternal.Reveal(internalAgent).AgentType == agentinternal.TypeLLMAgent {
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

// Run implements tool.Tool.
// It executes the wrapped agent.
func (t *AgentTool) Run(toolCtx tool.Context, args any) (any, error) {
	margs, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("AgentTool expects map[string]any arguments, got %T", args)
	}

	if t.skipSummarization {
		if actions := toolCtx.EventActions(); actions != nil {
			actions.SkipSummarization = true
		}
	}

	var agentInputSchema *genai.Schema
	internalAgent, ok := t.agent.(agentinternal.Agent)
	if !ok {
		return nil, fmt.Errorf("internal error: failed to convert to internal agent")
	}
	agentState := agentinternal.Reveal(internalAgent)
	isLllmAgent := (agentState != nil && agentState.AgentType == agentinternal.TypeLLMAgent)
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

	sessionService := sessionservice.Mem()

	r, err := runner.New(&runner.Config{
		AppName:        toolCtx.Agent().Name(),
		Agent:          t.agent,
		SessionService: sessionService,
		// TODO - use forwarding_artifact_service as in python.
		ArtifactService: artifactservice.Mem(),
		MemoryService:   memoryservice.Mem(),
	})

	if err != nil {
		return nil, fmt.Errorf("failed to create runner")
	}

	stateMap := make(map[string]any)

	for k, v := range toolCtx.Session().State().All() {
		// Filter out adk internal states.
		if !strings.HasPrefix(k, "_adk") {
			stateMap[k] = v
		}
	}

	subSession, err := sessionService.Create(toolCtx, &sessionservice.CreateRequest{
		AppName: toolCtx.Agent().Name(),
		UserID:  toolCtx.Session().ID().UserID,
		State:   stateMap,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create session for sub-agent %s: %w", t.agent.Name(), err)
	}

	eventCh := r.Run(context.Background(), subSession.Session.ID().UserID, subSession.Session.ID().SessionID, content, &runner.RunConfig{
		StreamingMode: runner.StreamingModeSSE,
	})

	var lastEvent *session.Event
	for event, err := range eventCh {
		if err != nil {
			return nil, fmt.Errorf("error during execution of sub-agent %s: %w", t.agent.Name(), err)
		}
		if event.LLMResponse != nil && event.LLMResponse.Content != nil {
			lastEvent = event
		}
	}

	if lastEvent == nil {
		return map[string]any{}, nil
	}

	lastContent := lastEvent.LLMResponse.Content
	var outputText string
	for _, part := range lastContent.Parts {
		if part != nil && part.Text != "" {
			if outputText != "" {
				outputText += "\n"
			}
			outputText += part.Text
		}
	}

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

// ProcessRequest implements tool.Tool.
func (t *AgentTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
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
