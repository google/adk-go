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

// Package functionaltest contains hermetic functional tests for ADK
// telemetry: each Test* function builds a real ADK agent
// (llmagent, workflowagent, ...) backed by a hermetic
// [google.golang.org/adk/internal/testutil.MockModel], drives it
// through a real Runner, and asserts the emitted span tree + log
// records against the expected shape declared in the
// telemetrytestcase package.
//
// Layout:
//
//   - The expected shapes (SpanDigest + LogDigest literals) live in
//     internal/telemetry/telemetrytestcase, one file per scenario.
//   - The helpers used to build the digests, install the
//     in-memory tracer/logger, and stand in for the LLM live in
//     internal/telemetry/telemetrytest.
//   - This package only holds the runners: scenario setup +
//     comparison code, so failures in this package always indicate
//     either a regression in the production code or a stale
//     expectation in telemetrytestcase.
//
// The package lives outside internal/telemetry to avoid import
// cycles (the runners pull in agent/llmagent, agent/workflowagent,
// runner, etc., which transitively import internal/telemetry).
package functionaltest
