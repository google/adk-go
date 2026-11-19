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

package toolsearch_test

import (
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolauth"
	"google.golang.org/adk/v2/tool/toolsearch"
)

// plainToolset is a toolset with no tools and no auth config.
type plainToolset struct{}

func (plainToolset) Name() string                                     { return "plain" }
func (plainToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return nil, nil }

// authenticatedToolset is a toolset that declares cfg as its auth config.
type authenticatedToolset struct {
	plainToolset
	cfg *toolauth.Config
}

func (a authenticatedToolset) AuthConfig() *toolauth.Config { return a.cfg }

func TestToolsetWrappers_ForwardAuthConfig(t *testing.T) {
	t.Run("authenticated base", func(t *testing.T) {
		cfg := &toolauth.Config{
			AuthScheme:    toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
			CredentialKey: "wrapped-key",
		}
		gated, err := toolsearch.New(authenticatedToolset{cfg: cfg}, toolsearch.Config{})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		wrapped, ok := gated.(tool.AuthenticatedToolset)
		if !ok {
			t.Fatal("toolsearch.New() toolset is not a tool.AuthenticatedToolset")
		}
		if got := wrapped.AuthConfig(); got != cfg {
			t.Errorf("AuthConfig() = %p, want the base toolset's config %p", got, cfg)
		}
	})
	t.Run("plain base", func(t *testing.T) {
		gated, err := toolsearch.New(plainToolset{}, toolsearch.Config{})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		wrapped, ok := gated.(tool.AuthenticatedToolset)
		if !ok {
			t.Fatal("toolsearch.New() toolset is not a tool.AuthenticatedToolset")
		}
		if got := wrapped.AuthConfig(); got != nil {
			t.Errorf("AuthConfig() = %p, want nil", got)
		}
	})
}
