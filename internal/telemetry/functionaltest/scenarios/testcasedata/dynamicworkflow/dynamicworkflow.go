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

// Package dynamicworkflow is a workflow agent running a static node, then a
// dynamic node that delegates to two collaborative agents and runs a function
// node twice, the second time from the RunNode cache.
package dynamicworkflow

import "google.golang.org/adk/v2/internal/telemetry/functionaltest/scenarios/testcasedata"

// Failure is the stage that fails. Any failure ends the run.
type Failure string

const (
	// NoFailure runs every stage to completion.
	NoFailure Failure = ""
	// StaticNodeError fails the static node, before the dynamic one starts.
	StaticNodeError Failure = "static-node-error"
	// DynamicNodeError fails the dynamic node before it delegates.
	DynamicNodeError Failure = "dynamic-node-error"
	// FirstAgentError fails the model call of the first delegated agent.
	FirstAgentError Failure = "first-agent-error"
	// SecondAgentError fails the model call of the second delegated agent.
	SecondAgentError Failure = "second-agent-error"
)

// Scenario is one variant of the scenario.
type Scenario struct {
	Failure Failure
}

// Name implements [testcasedata.Scenario].
func (Scenario) Name() string { return "dynamicworkflow" }

// Variant implements [testcasedata.Scenario].
func (s Scenario) Variant() string { return string(s.Failure) }

// WantErr implements [testcasedata.Scenario].
func (s Scenario) WantErr() bool { return s.Failure != NoFailure }

// Matrix returns every failure mode under every capture mode and schema
// version.
func Matrix() []testcasedata.Case {
	return testcasedata.Matrix(
		Scenario{NoFailure},
		Scenario{StaticNodeError},
		Scenario{DynamicNodeError},
		Scenario{FirstAgentError},
		Scenario{SecondAgentError},
	)
}
