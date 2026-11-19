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

// Package telemetrytest provides the helpers behind ADK's record/replay
// telemetry tests:
//
//   - [Digest] / [BuildDigest]: the emitted span tree, with log
//     records nested under the span they were emitted in, normalized so it can
//     be stored as a golden and compared exactly.
//   - [InMemoryLogExporter]: the log sink to install via
//     telemetry.OverrideLoggerForTesting.
//
// The goldens and the tests replaying them live in the functionaltest package.
package telemetrytest
