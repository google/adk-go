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

// Package workflowwalk lets the experimental app-info endpoint (exp/appinfo)
// walk workflow graphs, and exists for that feature only.
//
// The init function in workflow/walk.go assigns workflow's unexported graph
// walk to [WalkAgents], and app-info calls the walk through that variable.
// Because this package is internal, only code in this module can reach it. The
// comment on that init function explains why the walk is not exported instead.
package workflowwalk

import "google.golang.org/adk/v2/agent"

// WalkAgents returns the agents run by the nodes of edges, including those in
// nested workflows and ParallelWorkers. It is set by the init function in
// workflow/walk.go, so it is nil in a program that does not import workflow.
// edges must be a []workflow.Edge; it is an any because this package cannot
// import workflow without an import cycle.
var WalkAgents func(edges any) []agent.Agent
