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

// Package toolcall is an llmagent calling one function tool, then answering
// in text: adk-python's "agent" scenario, TOOL_CALLING_TURNS.
package toolcall

import "google.golang.org/adk/v2/internal/telemetry/functionaltest/scenarios/testcasedata"

// Failure is the component that fails.
type Failure string

const (
	// NoFailure runs both turns to completion.
	NoFailure Failure = ""
	// InferenceError fails the first model call, which ends the run.
	InferenceError Failure = "inference-error"
	// ToolError fails the tool. The error goes back to the model as the tool
	// result, so the run still completes.
	ToolError Failure = "tool-error"
)

// Scenario is one variant of the scenario.
type Scenario struct {
	Failure Failure
}

// Name implements [testcasedata.Scenario].
func (Scenario) Name() string { return "toolcall" }

// Variant implements [testcasedata.Scenario].
func (s Scenario) Variant() string { return string(s.Failure) }

// WantErr implements [testcasedata.Scenario].
func (s Scenario) WantErr() bool { return s.Failure == InferenceError }

// Matrix returns every failure mode under every capture mode and schema
// version.
func Matrix() []testcasedata.Case {
	return testcasedata.Matrix(Scenario{NoFailure}, Scenario{InferenceError}, Scenario{ToolError})
}
