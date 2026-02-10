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

package adka2a

import (
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

type executorPlugin struct {
	plugin *plugin.Plugin

	invocationSession session.Session
}

func withExecutorPlugin(cfg runner.Config) (runner.Config, *executorPlugin, error) {
	executorPlugin, err := newExecutorPlugin()
	if err != nil {
		return cfg, nil, err
	}
	cfg.PluginConfig.Plugins = append(slices.Clone(cfg.PluginConfig.Plugins), executorPlugin.plugin)
	return cfg, executorPlugin, nil
}

func newExecutorPlugin() (*executorPlugin, error) {
	execPlugin := &executorPlugin{}
	plugin, err := plugin.New(plugin.Config{
		Name: "a2a-executor",
		BeforeRunCallback: func(ic agent.InvocationContext) (*genai.Content, error) {
			execPlugin.invocationSession = ic.Session()
			return nil, nil
		},
	})
	if err != nil {
		return nil, err
	}
	execPlugin.plugin = plugin
	return execPlugin, nil
}
