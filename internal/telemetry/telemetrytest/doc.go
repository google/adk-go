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

// Package telemetrytest provides reusable helpers for hermetic
// telemetry tests in ADK.
//
// Helpers in this package:
//
//   - [SpanDigest] / [BuildDigests] / [PRESENT] — capture and
//     normalise the emitted OTel span tree so tests can compare
//     against a literal expected shape. Log records nest inside
//     the SpanDigest of the span they were emitted under via
//     [SpanDigest.Logs], yielding a single causal tree.
//   - [LogDigest] / [InMemoryLogExporter] — log-record snapshot
//     type and the in-memory sink to install via
//     telemetry.OverrideLoggerForTesting.
//
// The expected shapes for each scenario live in the sibling
// telemetrytestcase package; the runners that drive the agent and
// compare actual vs expected live in the telemetry/functionaltest
// test package. Functional tests use the shared
// [google.golang.org/adk/v2/internal/testutil.MockModel] as the
// deterministic LLM stand-in.
package telemetrytest
