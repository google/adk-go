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

package adkrest

import (
	"errors"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

// Config defines the services and loaders required by the adkrest package.
type Config struct {
	SessionService  session.Service
	ArtifactService artifact.Service
	AgentLoader     agent.Loader
	MemoryService   memory.Service
	SSEWriteTimeout time.Duration
	PluginConfig    runner.PluginConfig
}

// validate validates the config
func (c *Config) validate() error {
	if c.SessionService == nil {
		return errors.New("session service is required")
	}
	if c.ArtifactService == nil {
		return errors.New("artifact service is required")
	}
	if c.AgentLoader == nil {
		return errors.New("agent loader is required")
	}
	if c.MemoryService == nil {
		return errors.New("memory service is required")
	}
	return nil
}
