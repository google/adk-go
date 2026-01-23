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

package controllers

import (
	"testing"
	"time"

	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
)

func TestNewRuntimeAPIController_PluginsAssignment(t *testing.T) {
	p1, err := plugin.New(plugin.Config{Name: "plugin1"})
	if err != nil {
		t.Fatalf("plugin.New() failed for plugin1: %v", err)
	}

	p2, err := plugin.New(plugin.Config{Name: "plugin2"})
	if err != nil {
		t.Fatalf("plugin.New() failed for plugin2: %v", err)
	}

	tc := []struct {
		name        string
		plugins     []*plugin.Plugin
		wantPlugins int
	}{
		{
			name:        "with no plugins",
			plugins:     nil,
			wantPlugins: 0,
		},
		{
			name:        "with empty plugin list",
			plugins:     []*plugin.Plugin{},
			wantPlugins: 0,
		},
		{
			name:        "with single plugin",
			plugins:     []*plugin.Plugin{p1},
			wantPlugins: 1,
		},
		{
			name:        "with multiple plugins",
			plugins:     []*plugin.Plugin{p1, p2},
			wantPlugins: 2,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			controller := NewRuntimeAPIController(nil, nil, nil, nil, 10*time.Second, runner.PluginConfig{
				Plugins: tt.plugins,
			})

			if controller == nil {
				t.Fatal("NewRuntimeAPIController returned nil")
			}

			if got := len(controller.pluginConfig.Plugins); got != tt.wantPlugins {
				t.Errorf("NewRuntimeAPIController() plugins count = %v, want %v", got, tt.wantPlugins)
			}
		})
	}
}
