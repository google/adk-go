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
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/workflow"
)

// nodeFunctionRegistry stores pre-registered custom node functions.
// nodeRegistry caches resolved workflow nodes.
var (
	nodeFunctionRegistry = make(map[string]func(agent.Context, any) (any, error))
	nodeRegistry         = make(map[string]workflow.Node)
)

// RegisterNodeFunction registers a custom node function so it can be referenced inside Workflow YAML configurations.
func RegisterNodeFunction(name string, fn any) {
	registryMu.Lock()
	defer registryMu.Unlock()

	typedFn, err := castNodeFunction(name, fn)
	if err != nil {
		panic(fmt.Sprintf("RegisterNodeFunction failed for %q: %v", name, err))
	}

	nodeFunctionRegistry[name] = typedFn
}

// workflowYAMLConfig represents the YAML schema for a Workflow agent.
type workflowYAMLConfig struct {
	baseAgentConfig `yaml:",inline"`
	Edges           []yaml.Node `yaml:"edges"`
	MaxConcurrency  int         `yaml:"max_concurrency,omitempty"`
}

// functionNodeYAMLConfig represents the YAML schema for an inline FunctionNode.
type functionNodeYAMLConfig struct {
	Name           string `yaml:"name"`
	FuncCode       string `yaml:"func_code"`
	RerunOnResume  *bool  `yaml:"rerun_on_resume,omitempty"`
	ParallelWorker bool   `yaml:"parallel_worker,omitempty"`
}

// joinNodeYAMLConfig represents the YAML schema for a JoinNode.
type joinNodeYAMLConfig struct {
	Name string `yaml:"name"`
}

// toolNodeYAMLConfig represents the YAML schema for a ToolNode.
type toolNodeYAMLConfig struct {
	Name           string         `yaml:"name"`
	ToolCode       string         `yaml:"tool_code"`
	RerunOnResume  *bool          `yaml:"rerun_on_resume,omitempty"`
	ParallelWorker bool           `yaml:"parallel_worker,omitempty"`
	Args           map[string]any `yaml:"args,omitempty"`
}

// newWorkflowAgent builds a workflow agent from YAML data.
func newWorkflowAgent(ctx context.Context, data []byte, configPath string) (agent.Agent, error) {
	var cfg workflowYAMLConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal Workflow config: %w", err)
	}

	if cfg.Name == "" {
		return nil, fmt.Errorf("workflow 'name' is required")
	}

	// We track loaded sub-agents to populate the Agent hierarchy
	var subAgents []agent.Agent
	edgeParserCtx := context.WithValue(ctx, subAgentsCollectorKey, &subAgents)

	edges, err := parseEdges(edgeParserCtx, configPath, cfg.Edges)
	if err != nil {
		return nil, fmt.Errorf("failed to parse workflow edges: %w", err)
	}

	wf, err := workflow.New(cfg.Name, edges, workflow.WithMaxConcurrency(cfg.MaxConcurrency))
	if err != nil {
		return nil, fmt.Errorf("failed to build workflow: %w", err)
	}

	// Wrap workflow execution in standard custom agent
	return agent.New(agent.Config{
		Name:        cfg.Name,
		Description: cfg.Description,
		SubAgents:   subAgents,
		Run:         wf.Run,
	})
}

type contextKeyCollector string

const subAgentsCollectorKey contextKeyCollector = "subAgentsCollector"

// parseEdges converts dynamic sequence nodes from YAML into structural edges.
func parseEdges(ctx context.Context, parentPath string, nodes []yaml.Node) ([]workflow.Edge, error) {
	var edges []workflow.Edge

	for _, edgeNode := range nodes {
		if edgeNode.Kind != yaml.SequenceNode {
			return nil, fmt.Errorf("workflow edge chain must be a YAML list/sequence")
		}

		var chainNodes []workflow.Node
		var routeMaps []map[string]string

		for _, item := range edgeNode.Content {
			switch item.Kind {
			case yaml.ScalarNode:
				n, err := resolveNodeLike(ctx, parentPath, item.Value)
				if err != nil {
					return nil, err
				}
				chainNodes = append(chainNodes, n)
			case yaml.MappingNode:
				var m map[string]string
				if err := item.Decode(&m); err != nil {
					return nil, fmt.Errorf("invalid route map format: %w", err)
				}
				routeMaps = append(routeMaps, m)
			default:
				return nil, fmt.Errorf("unsupported YAML node kind in edge chain: %v", item.Kind)
			}
		}

		if len(routeMaps) == 0 {
			if len(chainNodes) < 2 {
				return nil, fmt.Errorf("workflow edge chain must have at least 2 nodes")
			}
			edges = append(edges, workflow.Chain(chainNodes...)...)
		} else {
			if len(chainNodes) < 1 {
				return nil, fmt.Errorf("routing edge chain must have a source node")
			}
			if len(chainNodes) >= 2 {
				edges = append(edges, workflow.Chain(chainNodes...)...)
			}

			routerNode := chainNodes[len(chainNodes)-1]

			for _, routeMap := range routeMaps {
				for routeVal, targetRef := range routeMap {
					targetNode, err := resolveNodeLike(ctx, parentPath, targetRef)
					if err != nil {
						return nil, err
					}

					var route workflow.Route
					if strings.EqualFold(routeVal, "default") {
						route = workflow.Default
					} else {
						route = workflow.StringRoute(routeVal)
					}

					edges = append(edges, workflow.Edge{
						From:  routerNode,
						To:    targetNode,
						Route: route,
					})
				}
			}
		}
	}

	return edges, nil
}

// resolveNodeLike maps a YAML identifier to a concrete workflow.Node.
func resolveNodeLike(ctx context.Context, parentPath, ref string) (workflow.Node, error) {
	if ref == "START" {
		return workflow.Start, nil
	}

	var cacheKey string
	var absPath string
	isYAML := strings.HasSuffix(ref, ".yaml") || strings.HasSuffix(ref, ".yml")

	if isYAML {
		targetPath := ref
		if !filepath.IsAbs(ref) {
			targetPath = filepath.Join(filepath.Dir(parentPath), ref)
		}
		var err error
		absPath, err = filepath.Abs(targetPath)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve absolute path: %w", err)
		}
		cacheKey = absPath
	} else {
		cacheKey = ref
	}

	registryMu.RLock()
	n, exists := nodeRegistry[cacheKey]
	registryMu.RUnlock()
	if exists {
		return n, nil
	}

	if !isYAML {
		registryMu.RLock()
		typedFn, exists := nodeFunctionRegistry[ref]
		registryMu.RUnlock()
		if !exists {
			return nil, fmt.Errorf("node or function reference %q not found in registries", ref)
		}
		return workflow.NewFunctionNode(ref, typedFn, workflow.NodeConfig{}), nil
	}

	n, err := resolveNodeFromYAML(ctx, parentPath, ref, absPath)
	if err != nil {
		return nil, err
	}

	registryMu.Lock()
	nodeRegistry[cacheKey] = n
	registryMu.Unlock()
	return n, nil
}

func resolveNodeFromYAML(ctx context.Context, parentPath, ref, absPath string) (workflow.Node, error) {
	data, err := os.ReadFile(absPath)
	if err != nil {
		return nil, err
	}

	var baseConfig baseAgentConfig
	if err := yaml.Unmarshal(data, &baseConfig); err != nil {
		return nil, fmt.Errorf("failed to parse config %q: %w", absPath, err)
	}

	if baseConfig.AgentClass == "FunctionNode" {
		var cfg functionNodeYAMLConfig
		if err := yaml.Unmarshal(data, &cfg); err != nil {
			return nil, fmt.Errorf("failed to parse FunctionNode config %q: %w", absPath, err)
		}

		if cfg.Name == "" {
			return nil, fmt.Errorf("function node 'name' is required")
		}
		if cfg.FuncCode == "" {
			return nil, fmt.Errorf("function node 'func_code' is required")
		}

		registryMu.RLock()
		typedFn, exists := nodeFunctionRegistry[cfg.FuncCode]
		registryMu.RUnlock()
		if !exists {
			return nil, fmt.Errorf("function %q not found in registry", cfg.FuncCode)
		}

		nodeConfig := workflow.NodeConfig{
			RerunOnResume:  cfg.RerunOnResume,
			ParallelWorker: cfg.ParallelWorker,
		}

		return workflow.NewFunctionNode(cfg.Name, typedFn, nodeConfig), nil
	}

	if baseConfig.AgentClass == "JoinNode" {
		var cfg joinNodeYAMLConfig
		if err := yaml.Unmarshal(data, &cfg); err != nil {
			return nil, fmt.Errorf("failed to parse JoinNode config %q: %w", absPath, err)
		}

		if cfg.Name == "" {
			return nil, fmt.Errorf("join node 'name' is required")
		}

		return workflow.NewJoinNode(cfg.Name), nil
	}

	if baseConfig.AgentClass == "ToolNode" {
		var cfg toolNodeYAMLConfig
		if err := yaml.Unmarshal(data, &cfg); err != nil {
			return nil, fmt.Errorf("failed to parse ToolNode config %q: %w", absPath, err)
		}

		if cfg.Name == "" {
			return nil, fmt.Errorf("tool node 'name' is required")
		}
		if cfg.ToolCode == "" {
			return nil, fmt.Errorf("tool node 'tool_code' is required")
		}

		t, _, err := ResolveToolReference(ctx, cfg.ToolCode, cfg.Args)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve tool %q for tool node %q: %w", cfg.ToolCode, cfg.Name, err)
		}

		nodeConfig := workflow.NodeConfig{
			RerunOnResume:  cfg.RerunOnResume,
			ParallelWorker: cfg.ParallelWorker,
		}

		return workflow.NewNamedToolNode(cfg.Name, t, nodeConfig)
	}

	// Otherwise, it is a standard agent class. Resolve it as an agent and wrap in NewAgentNode.
	ag, err := ResolveAgentReference(ctx, parentPath, ref)
	if err != nil {
		return nil, err
	}

	// Track subagent dependency
	if collector, ok := ctx.Value(subAgentsCollectorKey).(*[]agent.Agent); ok && collector != nil {
		*collector = append(*collector, ag)
	}

	// Wrap standard agent in workflow Node
	return workflow.NewAgentNode(ag, workflow.NodeConfig{})
}

// castNodeFunction normalizes function signatures to a standard non-generic workflow function.
func castNodeFunction(name string, fn any) (func(agent.Context, any) (any, error), error) {
	if typed, ok := fn.(func(agent.Context, any) (any, error)); ok {
		return typed, nil
	}

	if strFn, ok := fn.(func(agent.Context, string) (string, error)); ok {
		return func(ctx agent.Context, input any) (any, error) {
			var s string
			if input != nil {
				if val, ok := input.(string); ok {
					s = val
				} else {
					s = fmt.Sprint(input)
				}
			}
			return strFn(ctx, s)
		}, nil
	}

	return nil, fmt.Errorf("registered node function %q has unsupported signature. Must be func(agent.Context, any) (any, error) or func(agent.Context, string) (string, error)", name)
}
