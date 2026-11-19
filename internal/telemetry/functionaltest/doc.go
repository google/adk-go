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

// Package functionaltest holds the record/replay telemetry tests, the Go
// counterpart of adk-python's tests/unittests/telemetry/test_functional.py.
//
// Each case drives a real agent, backed by a hermetic
// [google.golang.org/adk/v2/internal/testutil.MockModel], through a real
// Runner under one telemetry configuration, and compares the spans and log
// records it emits with testdata/<test_id>.json. After an intentional
// telemetry change, re-record every golden with go generate, and review the
// diff, which is the change users will see:
//
//	go generate ./internal/telemetry/functionaltest
//
// The cases live under scenarios:
//
//   - scenarios/testcasedata: the data. One package per scenario, each with its
//     own Scenario type and a Matrix of every variant under every telemetry
//     configuration.
//   - scenarios/testcaseimpl: turns a Scenario into the agents, models and
//     tools it drives, runs it, and lists the cases to record.
//
// The package lives outside internal/telemetry to avoid an import cycle: the
// scenarios import agents and the runner, which import internal/telemetry.
package functionaltest

//go:generate go test . -update
