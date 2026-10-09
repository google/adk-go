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

// Package testcaseimpl turns a scenario from the testcasedata packages into
// the agents, models and tools it drives, and runs it through a real Runner.
// Each scenario's builder is in the file named for it.
package testcaseimpl

import (
	"slices"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/telemetry/functionaltest/scenarios/testcasedata"
	"google.golang.org/adk/v2/internal/telemetry/functionaltest/scenarios/testcasedata/delegation"
	"google.golang.org/adk/v2/internal/telemetry/functionaltest/scenarios/testcasedata/dynamicworkflow"
	"google.golang.org/adk/v2/internal/telemetry/functionaltest/scenarios/testcasedata/streaming"
	"google.golang.org/adk/v2/internal/telemetry/functionaltest/scenarios/testcasedata/toolcall"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

const (
	appName = "test_app"
	userID  = "test_user"
	prompt  = "hello"
)

// Run drives s through one user turn of a real Runner, backed by an in-memory
// session service, and returns the first error the run yields. Failing to set
// the run up fails t instead.
//
// Mirrors functional_test_helpers.run_agent_scenario in adk-python.
func Run(t *testing.T, s testcasedata.Scenario) error {
	t.Helper()
	ctx := t.Context()
	root, cfg := build(t, s)

	sessSvc := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          root,
		SessionService: sessSvc,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	sess, err := sessSvc.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID})
	if err != nil {
		t.Fatalf("session create: %v", err)
	}

	msg := genai.NewContentFromText(prompt, genai.RoleUser)
	for _, runErr := range r.Run(ctx, userID, sess.Session.ID(), msg, cfg) {
		if runErr != nil {
			return runErr
		}
	}
	return nil
}

// Cases returns every case of every scenario build knows.
func Cases() []testcasedata.Case {
	return slices.Concat(toolcall.Matrix(), streaming.Matrix(), delegation.Matrix(), dynamicworkflow.Matrix())
}

// build returns the root agent s runs, and the config to run it with.
func build(t *testing.T, s testcasedata.Scenario) (agent.Agent, agent.RunConfig) {
	t.Helper()
	switch s := s.(type) {
	case toolcall.Scenario:
		return newToolCall(t, s), agent.RunConfig{}
	case streaming.Scenario:
		return newStreaming(t), agent.RunConfig{StreamingMode: agent.StreamingModeSSE}
	case delegation.Scenario:
		return newDelegation(t), agent.RunConfig{}
	case dynamicworkflow.Scenario:
		return newDynamicWorkflow(t, s), agent.RunConfig{}
	}
	t.Fatalf("no scaffolding for scenario %T", s)
	return nil, agent.RunConfig{}
}
