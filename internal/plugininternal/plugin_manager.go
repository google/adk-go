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

package plugininternal

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/plugininternal/plugincontext"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

type PluginConfig struct {
	Plugins      []*plugin.Plugin
	CloseTimeout time.Duration
}

// PluginManager manages the registration and execution of plugins.
type PluginManager struct {
	plugins      []*plugin.Plugin
	closeTimeout time.Duration
}

// NewPluginManager creates a new PluginManager.
func NewPluginManager(cfg PluginConfig) (*PluginManager, error) {
	pm := &PluginManager{
		closeTimeout: cfg.CloseTimeout,
		plugins:      make([]*plugin.Plugin, 0, len(cfg.Plugins)),
	}

	// Register plugins defined in the config
	for _, p := range cfg.Plugins {
		err := pm.registerPlugin(p)
		if err != nil {
			return nil, err
		}
	}

	return pm, nil
}

// RegisterPlugin adds a new plugin to the manager.
func (pm *PluginManager) registerPlugin(plugin *plugin.Plugin) error {
	if plugin == nil {
		return fmt.Errorf("cannot register nil plugin")
	}
	for _, p := range pm.plugins {
		if p.Name() == plugin.Name() {
			return fmt.Errorf("plugin with name '%s' already registered", plugin.Name())
		}
	}
	pm.plugins = append(pm.plugins, plugin)
	return nil
}

// RunOnUserMessageCallback runs the OnUserMessageCallback for all plugins.
func (pm *PluginManager) RunOnUserMessageCallback(cctx agent.InvocationContext, userMessage *genai.Content) (*genai.Content, error) {
	for _, plugin := range pm.plugins {
		callback := plugin.OnUserMessageCallback()
		if callback != nil {
			newContent, err := callback(cctx, userMessage)
			if err != nil {
				return nil, err
			}
			if newContent != nil {
				return newContent, nil // Early exit
			}
		}
	}
	return nil, nil
}

// RunBeforeRunCallback runs the BeforeRunCallback for all plugins.
func (pm *PluginManager) RunBeforeRunCallback(cctx agent.InvocationContext) (*genai.Content, error) {
	for _, plugin := range pm.plugins {
		callback := plugin.BeforeRunCallback()
		if callback != nil {
			newContent, err := callback(cctx)
			if err != nil {
				return nil, err
			}
			if newContent != nil {
				return newContent, nil // Early exit
			}
		}
	}
	return nil, nil
}

// RunAfterRunCallback runs the AfterRunCallback for all plugins.
func (pm *PluginManager) RunAfterRunCallback(cctx agent.InvocationContext) {
	for i := len(pm.plugins) - 1; i >= 0; i-- {
		plugin := pm.plugins[i]
		callback := plugin.AfterRunCallback()
		if callback != nil {
			callback(cctx)
		}
	}
}

// RunOnEventCallback runs the OnEventCallback for all plugins.
func (pm *PluginManager) RunOnEventCallback(cctx agent.InvocationContext, event *session.Event) (*session.Event, error) {
	for _, plugin := range pm.plugins {
		callback := plugin.OnEventCallback()
		if callback != nil {
			newEvent, err := callback(cctx, event)
			if err != nil {
				return nil, err
			}
			if newEvent != nil {
				return newEvent, nil // Early exit
			}
		}
	}
	return nil, nil
}

// RunBeforeAgentCallback runs the BeforeAgentCallback for all plugins.
func (pm *PluginManager) RunBeforeAgentCallback(cctx agent.CallbackContext) (*genai.Content, error) {
	for _, plugin := range pm.plugins {
		callback := plugin.BeforeAgentCallback()
		if callback != nil {
			newContent, err := callback(cctx)
			if err != nil {
				return nil, err
			}
			if newContent != nil {
				return newContent, nil // Early exit
			}
		}
	}
	return nil, nil
}

// RunAfterAgentCallback runs the AfterAgentCallback for all plugins.
func (pm *PluginManager) RunAfterAgentCallback(cctx agent.CallbackContext) (*genai.Content, error) {
	for i := len(pm.plugins) - 1; i >= 0; i-- {
		plugin := pm.plugins[i]
		callback := plugin.AfterAgentCallback()
		if callback != nil {
			newContent, err := callback(cctx)
			if err != nil {
				return nil, err
			}
			if newContent != nil {
				return newContent, nil // Early exit
			}
		}
	}
	return nil, nil
}

// RunBeforeToolCallback runs the BeforeToolCallback for all plugins.
func (pm *PluginManager) RunBeforeToolCallback(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
	for _, plugin := range pm.plugins {
		callback := plugin.BeforeToolCallback()
		if callback != nil {
			newArgs, err := callback(ctx, tool, args)
			if err != nil {
				return nil, err
			}
			if newArgs != nil {
				return newArgs, nil // Early exit
			}
		}
	}
	return nil, nil
}

// RunAfterToolCallback runs the AfterToolCallback for all plugins.
func (pm *PluginManager) RunAfterToolCallback(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
	for i := len(pm.plugins) - 1; i >= 0; i-- {
		plugin := pm.plugins[i]
		callback := plugin.AfterToolCallback()
		if callback != nil {
			newResult, err := callback(ctx, tool, args, result, err)
			if err != nil {
				return nil, err
			}
			if newResult != nil {
				return newResult, nil // Early exit
			}
		}
	}
	return nil, nil
}

// RunOnToolErrorCallback runs the OnToolErrorCallback for all plugins.
func (pm *PluginManager) RunOnToolErrorCallback(ctx tool.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
	for _, plugin := range pm.plugins {
		callback := plugin.OnToolErrorCallback()
		if callback != nil {
			newResult, err := callback(ctx, tool, args, err)
			if err != nil {
				return nil, err
			}
			if newResult != nil {
				return newResult, nil // Early exit
			}
		}
	}
	return nil, nil
}

// RunBeforeModelCallback runs the BeforeModelCallback for all plugins.
func (pm *PluginManager) RunBeforeModelCallback(cctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
	for _, plugin := range pm.plugins {
		callback := plugin.BeforeModelCallback()
		if callback != nil {
			newResponse, err := callback(cctx, llmRequest)
			if err != nil {
				return nil, err
			}
			if newResponse != nil {
				return newResponse, nil // Early exit
			}
		}
	}
	return nil, nil
}

// RunAfterModelCallback runs the AfterModelCallback for all plugins.
func (pm *PluginManager) RunAfterModelCallback(cctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
	for i := len(pm.plugins) - 1; i >= 0; i-- {
		plugin := pm.plugins[i]
		callback := plugin.AfterModelCallback()
		if callback != nil {
			newResponse, err := callback(cctx, llmResponse, llmResponseError)
			if err != nil {
				return nil, err
			}
			if newResponse != nil {
				return newResponse, nil // Early exit
			}
		}
	}
	return nil, nil
}

// RunOnModelErrorCallback runs the OnModelErrorCallback for all plugins.
func (pm *PluginManager) RunOnModelErrorCallback(cctx agent.CallbackContext, llmRequest *model.LLMRequest, llmResponseError error) (*model.LLMResponse, error) {
	for _, plugin := range pm.plugins {
		callback := plugin.OnModelErrorCallback()
		if callback != nil {
			newResponse, err := callback(cctx, llmRequest, llmResponseError)
			if err != nil {
				return nil, err
			}
			if newResponse != nil {
				return newResponse, nil // Early exit
			}
		}
	}
	return nil, nil
}

// Close calls the CloseFunc on all registered plugins.
func (pm *PluginManager) Close() error {
	var errors []error
	for _, plugin := range pm.plugins {
		if err := plugin.Close(); err != nil {
			errors = append(errors, fmt.Errorf("error closing plugin '%s': %w", plugin.Name(), err))
		}
	}
	if len(errors) > 0 {
		return fmt.Errorf("failed to close plugins: %v", errors)
	}
	return nil
}

func ToContext(ctx context.Context, cfg *PluginManager) context.Context {
	return context.WithValue(ctx, plugincontext.PluginManagerCtxKey, cfg)
}
