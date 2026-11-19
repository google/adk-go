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

// Package delegation is an llmagent delegating to another through agenttool,
// which runs the delegate in a runner of its own: adk-python's "agent_tool"
// scenario.
package delegation

import "google.golang.org/adk/v2/internal/telemetry/functionaltest/scenarios/testcasedata"

// Scenario is the scenario. It has no variants yet.
type Scenario struct{}

// Name implements [testcasedata.Scenario].
func (Scenario) Name() string { return "delegation" }

// Variant implements [testcasedata.Scenario].
func (Scenario) Variant() string { return "" }

// WantErr implements [testcasedata.Scenario].
func (Scenario) WantErr() bool { return false }

// Matrix returns the scenario under every capture mode and schema version.
func Matrix() []testcasedata.Case { return testcasedata.Matrix(Scenario{}) }
