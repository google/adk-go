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
	"testing"

	"google.golang.org/adk/plugin"
)

func TestPluginManager_HasPlugins(t *testing.T) {
	t.Run("empty manager has no plugins", func(t *testing.T) {
		pm, err := NewPluginManager(PluginConfig{})
		if err != nil {
			t.Fatalf("NewPluginManager() error = %v", err)
		}
		if pm.HasPlugins() {
			t.Error("HasPlugins() = true, want false for empty manager")
		}
	})

	t.Run("manager with nil plugins slice has no plugins", func(t *testing.T) {
		pm, err := NewPluginManager(PluginConfig{Plugins: nil})
		if err != nil {
			t.Fatalf("NewPluginManager() error = %v", err)
		}
		if pm.HasPlugins() {
			t.Error("HasPlugins() = true, want false for nil plugins")
		}
	})

	t.Run("manager with plugins has plugins", func(t *testing.T) {
		p, err := plugin.New(plugin.Config{Name: "test-plugin"})
		if err != nil {
			t.Fatalf("plugin.New() error = %v", err)
		}
		pm, err := NewPluginManager(PluginConfig{Plugins: []*plugin.Plugin{p}})
		if err != nil {
			t.Fatalf("NewPluginManager() error = %v", err)
		}
		if !pm.HasPlugins() {
			t.Error("HasPlugins() = false, want true for manager with plugins")
		}
	})
}
