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
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"gopkg.in/yaml.v3"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/exampletool"
	"google.golang.org/adk/v2/tool/exitlooptool"
	"google.golang.org/adk/v2/tool/geminitool"
	"google.golang.org/adk/v2/tool/mcptoolset"
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
	if err := Register("Workflow", newWorkflowAgent); err != nil {
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
		return geminitool.New("url_context", "url context", &genai.Tool{URLContext: &genai.URLContext{}}), nil
	})
	if err != nil {
		panic(err)
	}
	err = RegisterToolFactory("google_maps_grounding", func(_ context.Context, _ map[string]any) (tool.Tool, error) {
		return geminitool.New("google_maps_grounding", "google maps grounding", &genai.Tool{GoogleMaps: &genai.GoogleMaps{}}), nil
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
	err = RegisterToolFactory("ExampleTool", func(ctx context.Context, args map[string]any) (tool.Tool, error) {
		if args == nil {
			return nil, fmt.Errorf("args is nil")
		}

		raw, ok := args["examples"]
		if !ok {
			return nil, fmt.Errorf("examples not found in args")
		}

		// 1. Cast the top-level 'examples' to a generic slice
		examplesSlice, ok := raw.([]any)
		if !ok {
			return nil, fmt.Errorf("examples is not a list")
		}

		// 2. Iterate and normalize the 'output' field
		for i, item := range examplesSlice {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}

			output := m["output"]
			if output == nil {
				continue
			}

			// Check if 'output' is NOT a slice. If it's a single object,
			// wrap it in a new slice []any{output}
			if _, isSlice := output.([]any); !isSlice {
				m["output"] = []any{output}
				examplesSlice[i] = m
			}
		}

		// 3. Now marshal/unmarshal as usual into your clean struct
		bytes, _ := json.Marshal(examplesSlice)
		var examples []*exampletool.Example
		if err := json.Unmarshal(bytes, &examples); err != nil {
			return nil, fmt.Errorf("failed to decode normalized examples: %w", err)
		}

		return exampletool.New(exampletool.ExampleToolConfig{
			Examples: examples,
		})
	})
	if err != nil {
		panic(err)
	}
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
			s, ok := arg.(string)
			if !ok {
				return nil, fmt.Errorf("server_params.args[%d]: expected string, got %T (%v)", i, arg, arg)
			}
			serverArgsStr[i] = s
		}
		toolFilterStr := make([]string, len(toolFilter))
		for i, t := range toolFilter {
			s, ok := t.(string)
			if !ok {
				return nil, fmt.Errorf("tool_filter[%d]: expected string, got %T (%v)", i, t, t)
			}
			toolFilterStr[i] = s
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

// Reasons a config reference can be rejected. They are sentinels so that callers
// and tests can identify the rejection without matching on the message text.
var (
	// errConfigReferenceNotLocal reports a reference that names a file outside
	// the referencing config's directory by its spelling alone.
	errConfigReferenceNotLocal = errors.New("config reference must be a relative path inside the agent directory")
	// errConfigReferenceSymlink reports a reference that reaches its target
	// through a symbolic link, or through any other reparse point, below that
	// directory.
	errConfigReferenceSymlink = errors.New("config reference traverses a link")
)

// resolveConfigReference turns a config-supplied reference into an absolute path
// that is guaranteed to sit inside the referencing config's own directory.
//
// A reference must be local to the parent directory in filepath.IsLocal's sense,
// which settles the lexical half of the question on every platform. The other
// half is symlinks, and they are refused rather than resolved: a link is not
// followed to see where it points, it simply disqualifies the reference. That is
// the stricter of the two rules, and unlike resolving it does not depend on the
// link's target existing at the moment of the check.
//
// Refusing rather than resolving means a link that stays inside the directory
// is refused too. That is a deliberate trade with a real cost: a config
// directory materialised entirely out of links — a Kubernetes ConfigMap or
// projected volume, a Bazel runfiles forest, a Nix store path — cannot use
// nested references at all, and has to be staged into a directory of real files
// first. The alternative, resolving each link and re-testing containment, cannot
// be made safe: a link pointing outside at a target that does not exist yet
// resolves to nothing and passes the test, and it stops being missing the moment
// anything creates the target.
//
// Only components below the parent directory are refused. The parent itself may
// be reached through any number of links.
//
// Every place that loads a nested YAML config from a reference must route
// through here: the check is the trust boundary between the config being loaded
// and the rest of the filesystem, and a second copy of it is a second place to
// forget.
func resolveConfigReference(parentPath, refPath string) (string, error) {
	// IsLocal rejects the empty string along with the escaping spellings, but an
	// empty reference is almost always an unfilled template rather than an
	// attempt to escape, and the generic message renders it as a dangling ": ".
	if refPath == "" {
		return "", fmt.Errorf("%w: reference is empty", errConfigReferenceNotLocal)
	}
	// IsLocal is purely lexical and rejects, in one call, everything that could
	// name a file outside the directory the reference is evaluated in: absolute
	// paths, any ".." that escapes, and on Windows drive-relative refs such as
	// `C:node.yaml`, UNC paths and reserved names such as NUL.
	if !filepath.IsLocal(refPath) {
		return "", fmt.Errorf("%w: %s", errConfigReferenceNotLocal, refPath)
	}

	parentDir, err := filepath.Abs(filepath.Dir(parentPath))
	if err != nil {
		return "", fmt.Errorf("failed to resolve agent directory: %w", err)
	}
	// Canonicalise the parent directory before joining, so that the path returned
	// from here is the real one. Callers use it as the key of agentRegistry and
	// nodeRegistry, and without this two spellings of one directory — the real
	// path and a symlink to it — would each get their own cache entry and each
	// build their own copy of the same agent.
	//
	// It is not what makes the containment check correct. The component walk
	// below relies on os.Lstat, which follows every component except the last, so
	// the two sides cannot end up rooted differently whether or not this runs.
	if resolved, err := filepath.EvalSymlinks(parentDir); err == nil {
		parentDir = resolved
	}

	// parentDir is absolute and Join cleans the result, so this is the absolute,
	// lexically-normalized target path.
	absPath := filepath.Join(parentDir, refPath)

	// IsLocal already guarantees the join lands inside parentDir lexically, so the
	// only remaining way out is a symlink on the way down. Refusing links is
	// stronger than resolving them: a link whose target does not exist yet
	// resolves to nothing, and resolving would wave exactly that through.
	if err := refuseSymlinkComponents(parentDir, refPath); err != nil {
		return "", err
	}

	return absPath, nil
}

// isLinkLike reports whether a component may redirect the read somewhere other
// than where its own name sits in the tree.
//
// ModeSymlink alone is not enough on Windows. A directory junction is a reparse
// point with tag IO_REPARSE_TAG_MOUNT_POINT, and since Go 1.23 (godebug
// winsymlink=1) os.Lstat reports it as ModeIrregular, not ModeSymlink: see
// os/types_windows.go, where IO_REPARSE_TAG_SYMLINK sets ModeSymlink and mount
// points fall through to ModeIrregular. Before Go 1.23 mount points did carry
// ModeSymlink, which is why testing for it alone looks sufficient. This module
// declares go 1.26, so it gets the newer mapping and a junction would walk
// straight past a ModeSymlink-only test.
//
// ModeIrregular is a broad term: it covers reparse tags that do not redirect
// anywhere, such as cloud-provider placeholder files, and those are refused
// along with the rest. It is not universal either — IO_REPARSE_TAG_AF_UNIX maps
// to ModeSocket and IO_REPARSE_TAG_DEDUP is reported as an ordinary file — but
// neither of those redirects a read out of the directory, so neither matters
// here.
func isLinkLike(mode fs.FileMode) bool {
	return mode&(fs.ModeSymlink|fs.ModeIrregular) != 0
}

// refuseSymlinkComponents fails if any component of refPath below dir is a
// link: a symbolic link on any platform, or a junction or other reparse point
// on Windows. Where the link points is not consulted, so one that stays inside
// dir is refused too — see resolveConfigReference for why the check is drawn
// that way.
//
// The boundary is narrower than "cannot reach outside dir" in two directions. A
// hard link inside dir to a file outside it is indistinguishable from a regular
// file here, and defending against it belongs wherever the directory is
// populated, typically archive extraction. A FIFO or device node is likewise
// left alone, since it redirects nothing.
//
// A component that cannot exist cannot be a link, so a reference naming a
// missing file, or one below a component that is not a directory, is left for
// the caller's own read to report as not found.
func refuseSymlinkComponents(dir, refPath string) error {
	cur := dir
	// IsLocal guarantees Clean leaves no ".." components to walk through.
	for _, part := range strings.Split(filepath.Clean(refPath), string(os.PathSeparator)) {
		cur = filepath.Join(cur, part)
		fi, err := os.Lstat(cur)
		// ENOTDIR is the same situation as ErrNotExist for this check: nothing can
		// exist below a component that is not a directory, so there is no link to
		// find. Reporting it as an inspection failure would give callers a third
		// class of error to handle for a reference that is simply not there.
		if errors.Is(err, fs.ErrNotExist) || errors.Is(err, syscall.ENOTDIR) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("failed to inspect config reference %q: %w", refPath, err)
		}
		if isLinkLike(fi.Mode()) {
			return fmt.Errorf("%w: %s", errConfigReferenceSymlink, refPath)
		}
	}
	return nil
}

// ResolveAgentReference builds an agent from a reference config.
func ResolveAgentReference(ctx context.Context, parentPath, refPath string) (agent.Agent, error) {
	absPath, err := resolveConfigReference(parentPath, refPath)
	if err != nil {
		return nil, err
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
