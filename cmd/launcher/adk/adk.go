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

// package adk defines common config for all agents & ways of launching
package adk

import (
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/cmd/restapi/services"
	"google.golang.org/adk/session"
)

// Config contains parameters for web & console execution: sessions, artifacts, agents etc
type Config struct {
	SessionService  session.Service
	ArtifactService artifact.Service
	AgentLoader     services.AgentLoader
}
