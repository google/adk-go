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

// Package compactionvalidate answers one question for every serving surface:
// can this compaction config actually serve the applications behind it.
package compactionvalidate

import (
	"fmt"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session/compaction"
)

// AgainstAgents reports whether cfg can serve every application loader knows
// about, given the services on base.
//
// A dry run of runner.New per application rather than a reimplementation of its
// checks: resolving the default summarizer needs the root agent's model, and a
// copy of that reasoning would drift from the one the requests use.
// Constructing a runner does no I/O.
//
// Shape-only validation is not enough, which is the reason this exists. A
// config can be internally consistent and still be unservable, and the case
// operators actually hit is the plainest one: no Summarizer supplied, over a
// root agent with no model to default to. compaction.Config.Validate accepts
// that, because it cannot see the agent. Without this the process starts,
// reports healthy, and returns an error on every request instead.
func AgainstAgents(cfg *compaction.Config, loader agent.Loader, base runner.Config) error {
	if loader == nil {
		return againstAgents(cfg, nil, base)
	}
	agents := make(map[string]agent.Agent)
	for _, name := range loader.ListAgents() {
		a, err := loader.LoadAgent(name)
		if err != nil {
			// Not this function's business: an application that cannot be
			// loaded fails its own requests with an error that says so.
			continue
		}
		agents[name] = a
	}
	return againstAgents(cfg, agents, base)
}

// AgainstRootAgent is [AgainstAgents] for a surface that serves only the
// loader's root agent.
//
// Agent Engine and A2A are both single-agent surfaces: every request they
// handle runs loader.RootAgent(). Checking every application the loader can
// list would refuse a config over an application they will never serve, which
// is a startup failure with no request behind it.
func AgainstRootAgent(cfg *compaction.Config, root agent.Agent, base runner.Config) error {
	if root == nil {
		return againstAgents(cfg, nil, base)
	}
	return againstAgents(cfg, map[string]agent.Agent{root.Name(): root}, base)
}

func againstAgents(cfg *compaction.Config, agents map[string]agent.Agent, base runner.Config) error {
	if cfg == nil {
		return nil
	}
	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid EventsCompactionConfig: %w", err)
	}
	for name, a := range agents {
		// Two runs, so only a compaction problem is reported. Everything else a
		// runner needs may legitimately be missing at construction time, and
		// failing on that here would refuse configurations that work.
		withoutCompaction := base
		withoutCompaction.AppName = name
		withoutCompaction.Agent = a
		withoutCompaction.EventsCompactionConfig = nil
		if _, err := runner.New(withoutCompaction); err != nil {
			continue
		}
		withCompaction := withoutCompaction
		withCompaction.EventsCompactionConfig = cfg
		if _, err := runner.New(withCompaction); err != nil {
			return fmt.Errorf("EventsCompactionConfig cannot serve app %q: %w", name, err)
		}
	}
	return nil
}
