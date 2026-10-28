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

package slack

import (
	"context"
	"errors"
	"fmt"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// Backend runs one conversational turn. The Slack layer does not know where
// the agent is: in this process (LocalBackend) or behind Agent Runtime.
//
// Run should stop iterating when ctx is done. It may yield ctx.Err() but does
// not have to: the service reports a turn whose ctx ended as failed either way.
type Backend interface {
	Run(ctx context.Context, userID, sessionID string, msg *genai.Content) iter.Seq2[*session.Event, error]
}

// LocalBackendConfig configures a LocalBackend.
type LocalBackendConfig struct {
	AppName string
	Agent   agent.Agent
	// SessionService stores the thread sessions. Nil means
	// session.InMemoryService(), which loses them on restart.
	SessionService session.Service
	// MemoryService is optional.
	MemoryService memory.Service
	// ArtifactService is optional.
	ArtifactService artifact.Service
	PluginConfig    runner.PluginConfig
}

// LocalBackend runs the agent in this process.
type LocalBackend struct {
	runner *runner.Runner
}

// NewLocalBackend runs the agent in this process through runner.Runner with
// AutoCreateSession, so the first message of a thread creates its session.
func NewLocalBackend(cfg LocalBackendConfig) (*LocalBackend, error) {
	if cfg.AppName == "" {
		return nil, errors.New("slack: LocalBackendConfig.AppName is required")
	}
	if cfg.Agent == nil {
		return nil, errors.New("slack: LocalBackendConfig.Agent is required")
	}
	sessions := cfg.SessionService
	if sessions == nil {
		sessions = session.InMemoryService()
	}
	r, err := runner.New(runner.Config{
		AppName:           cfg.AppName,
		Agent:             cfg.Agent,
		SessionService:    sessions,
		MemoryService:     cfg.MemoryService,
		ArtifactService:   cfg.ArtifactService,
		PluginConfig:      cfg.PluginConfig,
		AutoCreateSession: true,
	})
	if err != nil {
		return nil, fmt.Errorf("slack: creating runner: %w", err)
	}
	return &LocalBackend{runner: r}, nil
}

// Run runs one turn without streaming: Slack shows whole messages, so
// partial events would only be discarded.
func (b *LocalBackend) Run(ctx context.Context, userID, sessionID string, msg *genai.Content) iter.Seq2[*session.Event, error] {
	return b.runner.Run(ctx, userID, sessionID, msg, agent.RunConfig{StreamingMode: agent.StreamingModeNone})
}
