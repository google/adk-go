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
	"testing"

	"google.golang.org/adk/v2/plugin"
)

func TestPluginManager_HasPlugins(t *testing.T) {
	p, err := plugin.New(plugin.Config{Name: "test-plugin"})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	withPlugin, err := NewPluginManager(PluginConfig{Plugins: []*plugin.Plugin{p}})
	if err != nil {
		t.Fatalf("NewPluginManager() error = %v", err)
	}
	empty, err := NewPluginManager(PluginConfig{})
	if err != nil {
		t.Fatalf("NewPluginManager() error = %v", err)
	}

	tests := []struct {
		name string
		pm   *PluginManager
		want bool
	}{
		{name: "nil manager", pm: nil, want: false},
		{name: "empty manager", pm: empty, want: false},
		{name: "manager with a plugin", pm: withPlugin, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.pm.HasPlugins(); got != tt.want {
				t.Errorf("HasPlugins() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFromContext(t *testing.T) {
	pm, err := NewPluginManager(PluginConfig{})
	if err != nil {
		t.Fatalf("NewPluginManager() error = %v", err)
	}
	if got := FromContext(ToContext(context.Background(), pm)); got != pm {
		t.Errorf("FromContext() did not return the manager stored by ToContext")
	}
	if got := FromContext(context.Background()); got != nil {
		t.Errorf("FromContext() on an empty context = non-nil, want nil")
	}
}
