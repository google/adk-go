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

package configurable

import (
	"context"
	"fmt"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/agent/workflowagents/parallelagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/internal/llminternal/googlellm"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
)

// codeConfig represents a reference to a function or callback.
// Equivalent to: common_configs.codeConfig
type codeConfig struct {
	// Name of the function/method (e.g., "my_pkg.security.Check")
	Name string `yaml:"name"`

	// Optional params if your system supports parameterized callbacks
	Params map[string]any `yaml:"params,omitempty"`
}

// agentRefConfig represents a reference to a sub-agent.
// Equivalent to: common_configs.agentRefConfig
type agentRefConfig struct {
	// Path to another agent's YAML file
	ConfigPath string `yaml:"config_path,omitempty"`

	// OR an inline code reference
	Code string `yaml:"code,omitempty"`
}

type ToolConfig struct {
	// Name of the tool/method (e.g., "my_pkg.security.Check")
	Name string `yaml:"name"`

	// Optional params if your system supports parameterized callbacks
	Args map[string]any `yaml:"args,omitempty"`
}

// baseAgentConfig matches the Python baseAgentConfig Pydantic model.
//
// Usage: Do not use this struct directly for unmarshalling specific agents.
// Embed it into concrete agent configs (see Example below).
type baseAgentConfig struct {
	// Required. The class of the agent.
	// Default is "BaseAgent" in Python, but usually overridden by concrete agents.
	AgentClass string `yaml:"agent_class"`

	// Required. The name of the agent.
	Name string `yaml:"name"`

	// Optional. Description of the agent.
	Description string `yaml:"description,omitempty"`

	// Optional. List of sub-agents.
	SubAgents []agentRefConfig `yaml:"sub_agents,omitempty"`

	// Optional. Callbacks to run before execution.
	BeforeAgentCallbacks []codeConfig `yaml:"before_agent_callbacks,omitempty"`

	// Optional. Callbacks to run after execution.
	AfterAgentCallbacks []codeConfig `yaml:"after_agent_callbacks,omitempty"`

	// Path to the config file.
	ConfigPath string `yaml:"-"`

	// Handle extra fields (extra='allow'):
	// If you use this struct standalone, this map catches unknown fields.
	// However, the preferred pattern is to embed this struct in a concrete config
	// so specific fields are strongly typed.
	AdditionalProperties map[string]any `yaml:",inline"`
}

// llmAgentYAMLConfig is the concrete config for a specific agent.
type llmAgentYAMLConfig struct {
	// 1. Embed baseAgentConfig with ",inline".
	// This pulls "name", "sub_agents", etc. to the top level of the YAML.
	baseAgentConfig `yaml:",inline"`

	// 2. Define the "extra" fields specific to this agent here.
	Model string `yaml:"model"`

	Instruction string `yaml:"instruction"`

	Tools []ToolConfig `yaml:"tools,omitempty"`

	DisallowTransferToPeers bool `yaml:"disallow_transfer_to_peers,omitempty"`

	DisallowTransferToParent bool `yaml:"disallow_transfer_to_parent,omitempty"`

	GenerateContentConfig *genai.GenerateContentConfig `yaml:"generate_content_config,omitempty"`
}

func (c *llmAgentYAMLConfig) toLLMAgentConfig(ctx context.Context) (*llmagent.Config, error) {
	if !googlellm.IsGeminiModel(c.Model) {
		return nil, fmt.Errorf("model %s is not supported", c.Model)
	}

	model, err := gemini.NewModel(ctx, c.Model, &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create model: %w", err)
	}

	subAgents, err := resolveSubAgents(ctx, c.ConfigPath, c.SubAgents)
	if err != nil {
		return nil, err
	}

	tools, toolsets, err := resolveTools(ctx, c.ConfigPath, c.Tools)
	if err != nil {
		return nil, err
	}

	beforeCallbacks, err := resolveCallbacks[agent.BeforeAgentCallback](ctx, c.BeforeAgentCallbacks)
	if err != nil {
		return nil, err
	}

	afterCallbacks, err := resolveCallbacks[agent.AfterAgentCallback](ctx, c.AfterAgentCallbacks)
	if err != nil {
		return nil, err
	}

	return &llmagent.Config{
		Name:                     c.Name,
		Description:              c.Description,
		SubAgents:                subAgents,
		Model:                    model,
		Instruction:              c.Instruction,
		DisallowTransferToPeers:  c.DisallowTransferToPeers,
		DisallowTransferToParent: c.DisallowTransferToParent,
		Tools:                    tools,
		Toolsets:                 toolsets,
		GenerateContentConfig:    c.GenerateContentConfig,
		BeforeAgentCallbacks:     beforeCallbacks,
		AfterAgentCallbacks:      afterCallbacks,
	}, nil
}

type loopAgentYAMLConfig struct {
	baseAgentConfig `yaml:",inline"`
	MaxIterations   uint `yaml:"max_iterations"`
}

func (c *loopAgentYAMLConfig) toLoopAgentConfig(ctx context.Context) (*loopagent.Config, error) {
	subAgents, err := resolveSubAgents(ctx, c.ConfigPath, c.SubAgents)
	if err != nil {
		return nil, err
	}

	return &loopagent.Config{
		AgentConfig: agent.Config{
			Name:        c.Name,
			Description: c.Description,
			SubAgents:   subAgents,
		},
		MaxIterations: c.MaxIterations,
	}, nil
}

// ParallelAgentYAMLConfig is the concrete config for a specific agent.
type parallelAgentYAMLConfig struct {
	baseAgentConfig `yaml:",inline"`
}

func (c *parallelAgentYAMLConfig) toParallelAgentConfig(ctx context.Context) (*parallelagent.Config, error) {
	subAgents, err := resolveSubAgents(ctx, c.ConfigPath, c.SubAgents)
	if err != nil {
		return nil, err
	}

	return &parallelagent.Config{
		AgentConfig: agent.Config{
			Name:        c.Name,
			Description: c.Description,
			SubAgents:   subAgents,
		},
	}, nil
}

// SequentialAgentYAMLConfig is the concrete config for a specific agent.
type sequentialAgentYAMLConfig struct {
	baseAgentConfig `yaml:",inline"`
}

func (c *sequentialAgentYAMLConfig) toSequentialAgentConfig(ctx context.Context) (*sequentialagent.Config, error) {
	subAgents, err := resolveSubAgents(ctx, c.ConfigPath, c.SubAgents)
	if err != nil {
		return nil, err
	}

	return &sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:        c.Name,
			Description: c.Description,
			SubAgents:   subAgents,
		},
	}, nil
}

func resolveSubAgents(ctx context.Context, parentPath string, refs []agentRefConfig) ([]agent.Agent, error) {
	var agents []agent.Agent
	for _, ref := range refs {
		if ref.ConfigPath != "" {
			a, err := ResolveAgentReference(ctx, parentPath, ref.ConfigPath)
			if err != nil {
				return nil, fmt.Errorf("failed to resolve agent reference %s: %w", ref.ConfigPath, err)
			}
			agents = append(agents, a)
		} else if ref.Code != "" {
			return nil, fmt.Errorf("inline code agent references are not yet supported for %s", ref.Code)
		}
	}
	return agents, nil
}

type contextKey string

const parentPathKey contextKey = "parentPath"

func resolveTools(ctx context.Context, parentPath string, toolConfigs []ToolConfig) ([]tool.Tool, []tool.Toolset, error) {
	var tools []tool.Tool
	var toolsets []tool.Toolset
	for _, tc := range toolConfigs {
		if tc.Name != "" {
			ctx = context.WithValue(ctx, parentPathKey, parentPath)
			a, ts, err := ResolveToolReference(ctx, tc.Name, tc.Args)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to resolve tool reference %s: %w", tc.Name, err)
			}
			if a != nil {
				tools = append(tools, a)
			}
			if ts != nil {
				toolsets = append(toolsets, ts)
			}
		}
	}
	return tools, toolsets, nil
}

func resolveCallbacks[T any](ctx context.Context, callbacks []codeConfig) ([]T, error) {
	var cbs []T
	for _, ref := range callbacks {
		if ref.Name != "" {
			c, err := ResolveCallbackReference(ctx, ref.Name)
			if err != nil {
				return nil, fmt.Errorf("failed to resolve callback reference %s: %w", ref.Name, err)
			}
			cb, ok := c.(T)
			if !ok {
				return nil, fmt.Errorf("callback %s is of type %T and not %T", ref.Name, c, *new(T))
			}
			cbs = append(cbs, cb)
		}
	}
	return cbs, nil
}
