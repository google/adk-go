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

// configutils.go provides utility functions for working with configurable agents.
package configurable

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"gopkg.in/yaml.v3"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/agent/workflowagents/parallelagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/agenttool"
	"google.golang.org/adk/tool/exitlooptool"
	"google.golang.org/adk/tool/geminitool"
	"google.golang.org/adk/tool/mcptoolset"
)

type AgentFactory func(ctx context.Context, configBytes []byte, configPath string) (agent.Agent, error)

type ToolFactory func(ctx context.Context, args map[string]any) (tool.Tool, error)

type ToolsetFactory func(ctx context.Context, args map[string]any) (tool.Toolset, error)

var (
	registryMu       sync.RWMutex
	registry         = make(map[string]AgentFactory)
	agentRegistry    = make(map[string]agent.Agent)
	toolRegistry     = make(map[string]any)
	callbackRegistry = make(map[string]any)
)

func init() {
	if err := Register("LlmAgent", newLLMAgent); err != nil {
		panic(err)
	}
	if err := Register("LoopAgent", newLoopAgent); err != nil {
		panic(err)
	}
	if err := Register("ParallelAgent", newParallelAgent); err != nil {
		panic(err)
	}
	if err := Register("SequentialAgent", newSequentialAgent); err != nil {
		panic(err)
	}
	err := RegisterToolFactory("exit_loop", func(_ context.Context, _ map[string]any) (tool.Tool, error) {
		return exitlooptool.New()
	})
	if err != nil {
		panic(err)
	}
	err = RegisterToolFactory("google_search", func(_ context.Context, _ map[string]any) (tool.Tool, error) {
		return geminitool.GoogleSearch{}, nil
	})
	if err != nil {
		panic(err)
	}
	err = RegisterToolFactory("url_context", func(_ context.Context, _ map[string]any) (tool.Tool, error) {
		// TODO: return geminitool.New("url_context", "url context", &genai.Tool{URLContext: &genai.URLContext{}}), nil
		return geminitool.New("url_context", &genai.Tool{URLContext: &genai.URLContext{}}), nil
	})
	if err != nil {
		panic(err)
	}
	err = RegisterToolFactory("google_maps_grounding", func(_ context.Context, _ map[string]any) (tool.Tool, error) {
		// TODO: return geminitool.New("google_maps_grounding", "google maps grounding", &genai.Tool{GoogleMaps: &genai.GoogleMaps{}}), nil
		return geminitool.New("google_maps_grounding", &genai.Tool{GoogleMaps: &genai.GoogleMaps{}}), nil
	})
	if err != nil {
		panic(err)
	}
	err = RegisterToolFactory("AgentTool", func(ctx context.Context, args map[string]any) (tool.Tool, error) {
		if args == nil {
			return nil, fmt.Errorf("args is nil")
		}
		a, ok := args["agent"].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("agent not found in args")
		}
		skipSummarization := false
		if ss, ok := a["skip_summarization"].(bool); ok {
			skipSummarization = ss
		}
		parentPath, ok := ctx.Value(parentPathKey).(string)
		if !ok {
			return nil, fmt.Errorf("parentPath not found in context")
		}
		if configPath, ok := a["config_path"].(string); ok {
			ag, err := ResolveAgentReference(ctx, parentPath, configPath)
			if err != nil {
				return nil, err
			}
			return agenttool.New(ag, &agenttool.Config{SkipSummarization: skipSummarization}), nil
		} else {
			return nil, fmt.Errorf("config_path not found in args")
		}
	})
	if err != nil {
		panic(err)
	}
	err = RegisterToolFactory("LongRunningFunctionTool", func(ctx context.Context, args map[string]any) (tool.Tool, error) {
		if args == nil {
			return nil, fmt.Errorf("args is nil")
		}
		funcName, ok := args["func"].(string)
		if !ok {
			return nil, fmt.Errorf("func not found in args")
		}
		tool, _, err := ResolveToolReference(ctx, funcName, args)
		if err != nil {
			return nil, err
		}
		if tool == nil {
			return nil, fmt.Errorf("tool '%s' not found", funcName)
		}
		return tool, nil
	})
	if err != nil {
		panic(err)
	}
	// TODO: ExampleTool
	err = RegisterToolsetFactory("McpToolset", func(ctx context.Context, args map[string]any) (tool.Toolset, error) {
		stdioConnectionParams, ok := args["stdio_connection_params"].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("stdio_connection_params not found in args")
		}
		serverParams, ok := stdioConnectionParams["server_params"].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("server_params not found in stdio_connection_params")
		}
		command, ok := serverParams["command"].(string)
		if !ok {
			return nil, fmt.Errorf("command not found in server_params")
		}
		serverArgs, ok := serverParams["args"].([]any)
		if !ok {
			return nil, fmt.Errorf("args not found in server_params")
		}
		toolFilter, ok := args["tool_filter"].([]any)
		if !ok {
			return nil, fmt.Errorf("tool_filter not found in args")
		}
		serverArgsStr := make([]string, len(serverArgs))
		for i, arg := range serverArgs {
			serverArgsStr[i] = arg.(string)
		}
		toolFilterStr := make([]string, len(toolFilter))
		for i, t := range toolFilter {
			toolFilterStr[i] = t.(string)
		}

		mcpSet, err := mcptoolset.New(mcptoolset.Config{
			Transport: &mcp.CommandTransport{
				Command: exec.Command(command, serverArgsStr...),
			},
			ToolFilter: tool.StringPredicate(toolFilterStr),
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create mcp toolset: %v", err)
		}
		return mcpSet, nil
	})
	if err != nil {
		panic(err)
	}
}

// Register allows concrete implementations to add themselves to the system.
// This replaces Python's dynamic importlib logic.
func Register(name string, factory AgentFactory) error {
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, dup := registry[name]; dup {
		return fmt.Errorf("Register called twice for agent %s", name)
	}
	registry[name] = factory
	return nil
}

// RegisterToolFactory allows concrete implementations to add themselves to the system.
func RegisterToolFactory(name string, factory ToolFactory) error {
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, dup := toolRegistry[name]; dup {
		return fmt.Errorf("RegisterToolFactory called twice for tool %s", name)
	}
	toolRegistry[name] = factory
	return nil
}

func RegisterToolsetFactory(name string, factory ToolsetFactory) error {
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, dup := toolRegistry[name]; dup {
		return fmt.Errorf("RegisterToolsetFactory called twice for toolset %s", name)
	}
	toolRegistry[name] = factory
	return nil
}

func RegisterCallback(name string, callback any) error {
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, dup := callbackRegistry[name]; dup {
		return fmt.Errorf("RegisterCallback called twice for callback %s", name)
	}
	callbackRegistry[name] = callback
	return nil
}

// FromConfig builds an agent from a config file path.
// Equivalent to: def from_config(config_path: str) -> BaseAgent
func FromConfig(ctx context.Context, configPath string) (agent.Agent, error) {
	absPath, err := filepath.Abs(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve absolute path: %w", err)
	}

	// 1. Read the file
	data, err := os.ReadFile(absPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("config file not found: %s", absPath)
		}
		return nil, err
	}

	// 2. Peek at the "agent_class" field to know which factory to use.
	var baseConfig baseAgentConfig
	if err := yaml.Unmarshal(data, &baseConfig); err != nil {
		return nil, fmt.Errorf("invalid YAML content: %w", err)
	}

	// Default fallback similar to Python's handling
	agentClass := baseConfig.AgentClass
	if agentClass == "" {
		agentClass = "LlmAgent"
	}

	// 3. Resolve the factory (The Go equivalent of _resolve_agent_class)
	registryMu.RLock()
	factory, exists := registry[agentClass]
	registryMu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("invalid agent class '%s': not registered. Ensure the package is imported", agentClass)
	}

	// 4. Delegate creation to the specific factory.
	// We pass the raw data so the factory can unmarshal into its specific Config struct.
	return factory(ctx, data, absPath)
}

func ResolveToolReference(ctx context.Context, toolName string, args map[string]any) (tool.Tool, tool.Toolset, error) {
	if toolName == "" {
		return nil, nil, fmt.Errorf("tool name cannot be empty")
	}

	registryMu.RLock()
	if t, ok := toolRegistry[toolName]; ok {
		registryMu.RUnlock()
		if factory, ok := t.(ToolFactory); ok {
			tool, err := factory(ctx, args)
			return tool, nil, err
		}
		if toolsetFactory, ok := t.(ToolsetFactory); ok {
			toolset, err := toolsetFactory(ctx, args)
			return nil, toolset, err
		}
		return nil, nil, fmt.Errorf("tool '%s' is not a tool or toolset factory", toolName)
	}
	registryMu.RUnlock()
	return nil, nil, fmt.Errorf("tool '%s' not found", toolName)
}

func ResolveCallbackReference(ctx context.Context, callbackName string) (any, error) {
	if callbackName == "" {
		return nil, fmt.Errorf("callback name cannot be empty")
	}

	registryMu.RLock()
	if c, ok := callbackRegistry[callbackName]; ok {
		registryMu.RUnlock()
		return c, nil
	}
	registryMu.RUnlock()
	return nil, fmt.Errorf("callback '%s' not found", callbackName)
}

// ResolveAgentReference builds an agent from a reference config.
func ResolveAgentReference(ctx context.Context, parentPath, refPath string) (agent.Agent, error) {
	if refPath == "" {
		return nil, fmt.Errorf("agent reference path cannot be empty")
	}

	targetPath := refPath
	// Handle relative paths
	if !filepath.IsAbs(refPath) {
		targetPath = filepath.Join(filepath.Dir(parentPath), refPath)
	}

	absPath, err := filepath.Abs(targetPath)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve absolute path: %w", err)
	}

	registryMu.RLock()
	if a, ok := agentRegistry[absPath]; ok {
		registryMu.RUnlock()
		return a, nil
	}
	registryMu.RUnlock()

	a, err := FromConfig(ctx, absPath)
	if err != nil {
		return nil, err
	}

	registryMu.Lock()
	defer registryMu.Unlock()
	if existing, ok := agentRegistry[absPath]; ok {
		return existing, nil
	}
	agentRegistry[absPath] = a
	return a, nil
}

// NewLLMAgent is the factory function registered in the system.
func newLLMAgent(ctx context.Context, data []byte, configPath string) (agent.Agent, error) {
	var cfg llmAgentYAMLConfig

	// Unmarshal parses the shared fields (Name) into BaseAgentConfig
	// AND the specific fields (ModelName) into LLMAgentConfig simultaneously.
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse LLM agent config: %w", err)
	}

	// Validation Logic (Pydantic equivalent)
	if cfg.Name == "" {
		return nil, fmt.Errorf("'name' is required")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("'model' is required for LlmAgent")
	}

	cfg.ConfigPath = configPath

	agentConfig, err := cfg.toLLMAgentConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create LLM agent config: %w", err)
	}

	return llmagent.New(*agentConfig)
}

func newLoopAgent(ctx context.Context, data []byte, configPath string) (agent.Agent, error) {
	var cfg loopAgentYAMLConfig

	// Unmarshal parses the shared fields (Name) into BaseAgentConfig
	// AND the specific fields (ModelName) into LLMAgentConfig simultaneously.
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse Loop agent config: %w", err)
	}

	// Validation Logic (Pydantic equivalent)
	if cfg.Name == "" {
		return nil, fmt.Errorf("'name' is required")
	}

	cfg.ConfigPath = configPath

	agentConfig, err := cfg.toLoopAgentConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create Loop agent config: %w", err)
	}

	return loopagent.New(*agentConfig)
}

func newParallelAgent(ctx context.Context, data []byte, configPath string) (agent.Agent, error) {
	var cfg parallelAgentYAMLConfig

	// Unmarshal parses the shared fields (Name) into BaseAgentConfig
	// AND the specific fields (ModelName) into LLMAgentConfig simultaneously.
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse Parallel agent config: %w", err)
	}

	// Validation Logic (Pydantic equivalent)
	if cfg.Name == "" {
		return nil, fmt.Errorf("'name' is required")
	}

	cfg.ConfigPath = configPath

	agentConfig, err := cfg.toParallelAgentConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create Parallel agent config: %w", err)
	}

	return parallelagent.New(*agentConfig)
}

func newSequentialAgent(ctx context.Context, data []byte, configPath string) (agent.Agent, error) {
	var cfg sequentialAgentYAMLConfig

	// Unmarshal parses the shared fields (Name) into BaseAgentConfig
	// AND the specific fields (ModelName) into LLMAgentConfig simultaneously.
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse Sequential agent config: %w", err)
	}

	// Validation Logic (Pydantic equivalent)
	if cfg.Name == "" {
		return nil, fmt.Errorf("'name' is required")
	}

	cfg.ConfigPath = configPath

	agentConfig, err := cfg.toSequentialAgentConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create Sequential agent config: %w", err)
	}

	return sequentialagent.New(*agentConfig)
}
