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

// Package agentengine provides easy way to deploy to AgentEngine.
package agentengine

import (
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/universal"
	"google.golang.org/adk/cmd/launcher/web"
	webagentengine "google.golang.org/adk/cmd/launcher/web/agentengine"
)

// NewLauncher returns a launcher capable of serving queries from AgentEngine.
func NewLauncher(agentEngineId string) launcher.Launcher {
	return universal.NewLauncher(web.NewLauncher(webagentengine.NewLauncher(agentEngineId)))
}
