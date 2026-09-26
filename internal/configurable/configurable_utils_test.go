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
	"sync"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
)

func resetRegistries(t *testing.T) {
	t.Helper()
	registryMu.Lock()
	oldCallbacks := make(map[string]any, len(callbackRegistry))
	for k, v := range callbackRegistry {
		oldCallbacks[k] = v
	}
	oldTools := make(map[string]any, len(toolRegistry))
	for k, v := range toolRegistry {
		oldTools[k] = v
	}
	oldAgents := make(map[string]agent.Agent, len(agentRegistry))
	for k, v := range agentRegistry {
		oldAgents[k] = v
	}
	registryMu.Unlock()

	t.Cleanup(func() {
		registryMu.Lock()
		defer registryMu.Unlock()
		callbackRegistry = oldCallbacks
		toolRegistry = oldTools
		agentRegistry = oldAgents
	})
}

func TestRegisterCallback(t *testing.T) {
	resetRegistries(t)

	t.Run("EmptyNameValidation", func(t *testing.T) {
		err := RegisterCallback("", func() {})
		if err == nil {
			t.Errorf("expected error when registering callback with empty name, got nil")
		}
	})

	t.Run("NilCallbackValidation", func(t *testing.T) {
		err := RegisterCallback("test_nil_cb", nil)
		if err == nil {
			t.Errorf("expected error when registering nil callback, got nil")
		}
		_, resolveErr := ResolveCallbackReference(context.Background(), "test_nil_cb")
		if resolveErr == nil {
			t.Errorf("expected nil callback not to be found in registry")
		}
	})

	t.Run("TypeCompatibility", func(t *testing.T) {
		type dummyStruct struct {
			Name string
		}

		dummyFn := func(ctx context.Context, input string) (string, error) {
			return "hello " + input, nil
		}
		dummyObj := dummyStruct{Name: "test"}
		dummyClosure := func() int { return 42 }

		tests := []struct {
			name string
			val  any
		}{
			{name: "test_func", val: dummyFn},
			{name: "test_struct", val: dummyObj},
			{name: "test_closure", val: dummyClosure},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				err := RegisterCallback(tc.name, tc.val)
				if err != nil {
					t.Fatalf("unexpected error registering callback %s: %v", tc.name, err)
				}
				resolved, err := ResolveCallbackReference(context.Background(), tc.name)
				if err != nil {
					t.Fatalf("unexpected error resolving callback %s: %v", tc.name, err)
				}
				if resolved == nil {
					t.Fatalf("expected resolved callback for %s to be non-nil", tc.name)
				}
			})
		}
	})

	t.Run("DuplicateRegistrationRejection", func(t *testing.T) {
		name := "test_dup_cb"
		firstCb := func() string { return "first" }
		secondCb := func() string { return "second" }

		err := RegisterCallback(name, firstCb)
		if err != nil {
			t.Fatalf("failed initial registration: %v", err)
		}

		err = RegisterCallback(name, secondCb)
		if err == nil {
			t.Fatalf("expected error on duplicate callback registration, got nil")
		}

		resolved, err := ResolveCallbackReference(context.Background(), name)
		if err != nil {
			t.Fatalf("failed to resolve callback after duplicate attempt: %v", err)
		}
		fn, ok := resolved.(func() string)
		if !ok || fn() != "first" {
			t.Fatalf("expected original callback to remain registered")
		}
	})
}

func TestRegisterToolFactory(t *testing.T) {
	resetRegistries(t)

	t.Run("EmptyNameValidation", func(t *testing.T) {
		factory := func(ctx context.Context, args map[string]any) (tool.Tool, error) {
			return nil, nil
		}
		err := RegisterToolFactory("", factory)
		if err == nil {
			t.Errorf("expected error when registering tool factory with empty name, got nil")
		}
	})

	t.Run("NilFactoryValidation", func(t *testing.T) {
		err := RegisterToolFactory("test_nil_tool", nil)
		if err == nil {
			t.Errorf("expected error when registering nil tool factory, got nil")
		}
	})

	t.Run("HappyPathAndDuplicateRejection", func(t *testing.T) {
		name := "test_tool_factory"
		factory1 := func(ctx context.Context, args map[string]any) (tool.Tool, error) {
			return nil, nil
		}
		factory2 := func(ctx context.Context, args map[string]any) (tool.Tool, error) {
			return nil, fmt.Errorf("factory2")
		}

		err := RegisterToolFactory(name, factory1)
		if err != nil {
			t.Fatalf("unexpected error registering tool factory: %v", err)
		}

		err = RegisterToolFactory(name, factory2)
		if err == nil {
			t.Fatalf("expected error on duplicate tool factory registration, got nil")
		}

		tl, toolset, err := ResolveToolReference(context.Background(), name, nil)
		if err != nil {
			t.Fatalf("unexpected error resolving registered tool factory: %v", err)
		}
		if tl != nil || toolset != nil {
			t.Fatalf("expected tool and toolset to be nil from dummy factory")
		}
	})
}

func TestRegisterToolsetFactory(t *testing.T) {
	resetRegistries(t)

	t.Run("EmptyNameValidation", func(t *testing.T) {
		factory := func(ctx context.Context, args map[string]any) (tool.Toolset, error) {
			return nil, nil
		}
		err := RegisterToolsetFactory("", factory)
		if err == nil {
			t.Errorf("expected error when registering toolset factory with empty name, got nil")
		}
	})

	t.Run("NilFactoryValidation", func(t *testing.T) {
		err := RegisterToolsetFactory("test_nil_toolset", nil)
		if err == nil {
			t.Errorf("expected error when registering nil toolset factory, got nil")
		}
	})

	t.Run("HappyPathAndDuplicateRejection", func(t *testing.T) {
		name := "test_toolset_factory"
		factory1 := func(ctx context.Context, args map[string]any) (tool.Toolset, error) {
			return nil, nil
		}
		factory2 := func(ctx context.Context, args map[string]any) (tool.Toolset, error) {
			return nil, fmt.Errorf("factory2")
		}

		err := RegisterToolsetFactory(name, factory1)
		if err != nil {
			t.Fatalf("unexpected error registering toolset factory: %v", err)
		}

		err = RegisterToolsetFactory(name, factory2)
		if err == nil {
			t.Fatalf("expected error on duplicate toolset factory registration, got nil")
		}

		_, toolset, err := ResolveToolReference(context.Background(), name, nil)
		if err != nil {
			t.Fatalf("unexpected error resolving registered toolset factory: %v", err)
		}
		if toolset != nil {
			t.Fatalf("expected toolset to be nil from dummy factory")
		}
	})

	t.Run("McpToolsetInvalidArgsHandling", func(t *testing.T) {
		resetRegistries(t)

		t.Run("NonStringServerParamsArgs", func(t *testing.T) {
			args := map[string]any{
				"stdio_connection_params": map[string]any{
					"server_params": map[string]any{
						"command": "node",
						"args":    []any{"valid", 123},
					},
				},
				"tool_filter": []any{"tool1"},
			}
			_, _, err := ResolveToolReference(context.Background(), "McpToolset", args)
			if err == nil {
				t.Fatalf("expected error for non-string server_params args element, got nil")
			}
		})

		t.Run("NonStringToolFilter", func(t *testing.T) {
			args := map[string]any{
				"stdio_connection_params": map[string]any{
					"server_params": map[string]any{
						"command": "node",
						"args":    []any{"valid"},
					},
				},
				"tool_filter": []any{"tool1", true},
			}
			_, _, err := ResolveToolReference(context.Background(), "McpToolset", args)
			if err == nil {
				t.Fatalf("expected error for non-string tool_filter element, got nil")
			}
		})
	})

	t.Run("CrossTypeCollisionWithToolFactory", func(t *testing.T) {
		toolName := "cross_collision_tool"
		toolFactory := func(ctx context.Context, args map[string]any) (tool.Tool, error) {
			return nil, nil
		}
		toolsetFactory := func(ctx context.Context, args map[string]any) (tool.Toolset, error) {
			return nil, nil
		}

		if err := RegisterToolFactory(toolName, toolFactory); err != nil {
			t.Fatalf("unexpected error registering tool factory: %v", err)
		}

		err := RegisterToolsetFactory(toolName, toolsetFactory)
		if err == nil {
			t.Fatalf("expected error registering toolset factory with duplicate tool name, got nil")
		}

		toolsetName := "cross_collision_toolset"
		if err := RegisterToolsetFactory(toolsetName, toolsetFactory); err != nil {
			t.Fatalf("unexpected error registering toolset factory: %v", err)
		}

		err = RegisterToolFactory(toolsetName, toolFactory)
		if err == nil {
			t.Fatalf("expected error registering tool factory with duplicate toolset name, got nil")
		}
	})
}

func TestConcurrentRegistration(t *testing.T) {
	resetRegistries(t)

	const numGoroutines = 100
	var wg sync.WaitGroup
	wg.Add(numGoroutines * 2)

	for i := 0; i < numGoroutines; i++ {
		keyName := fmt.Sprintf("concurrent_cb_%d", i)
		go func(k string) {
			defer wg.Done()
			_ = RegisterCallback(k, func() {})
		}(keyName)

		collidingKey := "concurrent_colliding_cb"
		go func() {
			defer wg.Done()
			_ = RegisterCallback(collidingKey, func() {})
		}()
	}

	wg.Wait()
}

func TestResolveAgentReference(t *testing.T) {
	resetRegistries(t)

	t.Run("EmptyRefPathValidation", func(t *testing.T) {
		resetRegistries(t)
		_, err := ResolveAgentReference(context.Background(), "/some/parent.yaml", "")
		if err == nil {
			t.Fatalf("expected error for empty refPath, got nil")
		}
		if want := "agent reference path cannot be empty"; err.Error() != want {
			t.Errorf("got error %q, want %q", err.Error(), want)
		}
	})

	t.Run("RelativePathResolution", func(t *testing.T) {
		resetRegistries(t)
		dir := t.TempDir()
		subDir := filepath.Join(dir, "sub")
		if err := os.MkdirAll(subDir, 0o755); err != nil {
			t.Fatalf("failed to create sub dir: %v", err)
		}
		parentPath := filepath.Join(subDir, "parent.yaml")
		refPath := "child.yaml"
		childAbsPath := filepath.Join(subDir, "child.yaml")

		yamlContent := []byte(`
agent_class: LoopAgent
name: child_agent
max_iterations: 5
`)
		if err := os.WriteFile(childAbsPath, yamlContent, 0o644); err != nil {
			t.Fatalf("failed to write child config: %v", err)
		}

		ag, err := ResolveAgentReference(context.Background(), parentPath, refPath)
		if err != nil {
			t.Fatalf("unexpected error resolving relative agent reference: %v", err)
		}
		if ag == nil {
			t.Fatalf("expected non-nil agent")
		}
		if ag.Name() != "child_agent" {
			t.Errorf("got agent name %q, want %q", ag.Name(), "child_agent")
		}

		registryMu.RLock()
		cached, exists := agentRegistry[childAbsPath]
		registryMu.RUnlock()
		if !exists || cached != ag {
			t.Errorf("agent was not properly cached in agentRegistry")
		}
	})

	t.Run("AbsolutePathResolution", func(t *testing.T) {
		resetRegistries(t)
		dir := t.TempDir()
		childAbsPath := filepath.Join(dir, "child_abs.yaml")

		yamlContent := []byte(`
agent_class: SequentialAgent
name: abs_agent
`)
		if err := os.WriteFile(childAbsPath, yamlContent, 0o644); err != nil {
			t.Fatalf("failed to write abs config: %v", err)
		}

		ag, err := ResolveAgentReference(context.Background(), "/any/parent/path.yaml", childAbsPath)
		if err != nil {
			t.Fatalf("unexpected error resolving absolute agent reference: %v", err)
		}
		if ag == nil || ag.Name() != "abs_agent" {
			t.Errorf("unexpected agent: %v", ag)
		}
	})

	t.Run("PathTraversalResolution", func(t *testing.T) {
		resetRegistries(t)
		dir := t.TempDir()
		subDir := filepath.Join(dir, "sub")
		if err := os.MkdirAll(subDir, 0o755); err != nil {
			t.Fatalf("failed to create sub directory: %v", err)
		}
		parentPath := filepath.Join(subDir, "parent.yaml")
		refPath := "../traversal_agent.yaml"
		expectedAbsPath := filepath.Join(dir, "traversal_agent.yaml")

		yamlContent := []byte(`
agent_class: ParallelAgent
name: traversal_agent
`)
		if err := os.WriteFile(expectedAbsPath, yamlContent, 0o644); err != nil {
			t.Fatalf("failed to write traversal agent config: %v", err)
		}

		ag, err := ResolveAgentReference(context.Background(), parentPath, refPath)
		if err != nil {
			t.Fatalf("unexpected error resolving path traversal reference: %v", err)
		}
		if ag == nil || ag.Name() != "traversal_agent" {
			t.Errorf("unexpected agent: %v", ag)
		}

		registryMu.RLock()
		_, exists := agentRegistry[expectedAbsPath]
		registryMu.RUnlock()
		if !exists {
			t.Errorf("expected canonical path %q to be present in agentRegistry", expectedAbsPath)
		}
	})

	t.Run("EmptyParentPathResolution", func(t *testing.T) {
		resetRegistries(t)
		relFile := "empty_parent_test_agent.yaml"
		absPath, err := filepath.Abs(relFile)
		if err != nil {
			t.Fatalf("failed to get absolute path: %v", err)
		}

		yamlContent := []byte(`
agent_class: LoopAgent
name: empty_parent_agent
max_iterations: 2
`)
		if err := os.WriteFile(absPath, yamlContent, 0o644); err != nil {
			t.Fatalf("failed to write config file: %v", err)
		}
		t.Cleanup(func() {
			_ = os.Remove(absPath)
		})

		ag, err := ResolveAgentReference(context.Background(), "", relFile)
		if err != nil {
			t.Fatalf("unexpected error resolving with empty parentPath: %v", err)
		}
		if ag == nil || ag.Name() != "empty_parent_agent" {
			t.Errorf("unexpected agent: %v", ag)
		}
	})

	t.Run("CacheHitAvoidsFileRead", func(t *testing.T) {
		resetRegistries(t)
		dir := t.TempDir()
		childAbsPath := filepath.Join(dir, "cached_agent.yaml")

		yamlContent := []byte(`
agent_class: LoopAgent
name: initial_cached_agent
max_iterations: 1
`)
		if err := os.WriteFile(childAbsPath, yamlContent, 0o644); err != nil {
			t.Fatalf("failed to write initial config: %v", err)
		}

		ag1, err := ResolveAgentReference(context.Background(), "", childAbsPath)
		if err != nil {
			t.Fatalf("failed first resolution: %v", err)
		}

		if err := os.Remove(childAbsPath); err != nil {
			t.Fatalf("failed to remove config file: %v", err)
		}

		ag2, err := ResolveAgentReference(context.Background(), "", childAbsPath)
		if err != nil {
			t.Fatalf("expected resolution from cache to succeed, got: %v", err)
		}
		if ag1 != ag2 {
			t.Errorf("expected identical agent instance from cache")
		}
	})

	t.Run("ErrorNonExistentFile", func(t *testing.T) {
		resetRegistries(t)
		dir := t.TempDir()
		nonExistentPath := filepath.Join(dir, "does_not_exist.yaml")
		_, err := ResolveAgentReference(context.Background(), "", nonExistentPath)
		if err == nil {
			t.Fatalf("expected error for non-existent file, got nil")
		}
	})

	t.Run("ErrorInvalidYAML", func(t *testing.T) {
		resetRegistries(t)
		dir := t.TempDir()
		invalidPath := filepath.Join(dir, "invalid.yaml")
		if err := os.WriteFile(invalidPath, []byte("invalid: yaml: : content"), 0o644); err != nil {
			t.Fatalf("failed to write invalid file: %v", err)
		}
		_, err := ResolveAgentReference(context.Background(), "", invalidPath)
		if err == nil {
			t.Fatalf("expected error for invalid YAML, got nil")
		}
	})

	t.Run("ErrorUnregisteredAgentClass", func(t *testing.T) {
		resetRegistries(t)
		dir := t.TempDir()
		unregisteredPath := filepath.Join(dir, "unregistered.yaml")
		yamlContent := []byte(`
agent_class: NonExistentClass
name: test
`)
		if err := os.WriteFile(unregisteredPath, yamlContent, 0o644); err != nil {
			t.Fatalf("failed to write config: %v", err)
		}
		_, err := ResolveAgentReference(context.Background(), "", unregisteredPath)
		if err == nil {
			t.Fatalf("expected error for unregistered agent class, got nil")
		}
	})

	t.Run("ConcurrentResolution", func(t *testing.T) {
		resetRegistries(t)
		dir := t.TempDir()
		const numAgents = 5
		paths := make([]string, numAgents)
		for i := 0; i < numAgents; i++ {
			p := filepath.Join(dir, fmt.Sprintf("agent_%d.yaml", i))
			content := fmt.Sprintf("agent_class: LoopAgent\nname: agent_%d\nmax_iterations: 1\n", i)
			if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
				t.Fatalf("failed to write config %d: %v", i, err)
			}
			paths[i] = p
		}

		const numGoroutines = 50
		var wg sync.WaitGroup
		wg.Add(numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			go func(idx int) {
				defer wg.Done()
				targetPath := paths[idx%numAgents]
				ag, err := ResolveAgentReference(context.Background(), "", targetPath)
				if err != nil {
					t.Errorf("concurrent resolution error: %v", err)
					return
				}
				if ag == nil {
					t.Errorf("got nil agent in goroutine %d", idx)
				}
			}(i)
		}

		wg.Wait()
	})
}
