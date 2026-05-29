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

// Package telemetrytestcase declares the expected telemetry shape
// (spans + log records nested inside their owning span) emitted by
// each canonical end-to-end ADK scenario.
//
// Each test case lives in its own file and exports exactly one
// symbol — a *[telemetrytest.SpanDigest] holding the expected root
// span and its full subtree — so the literal SpanDigest /
// LogDigest expectations are visible at a glance, undivided by
// helpers, and the package's exported surface is a 1:1 inventory
// of the scenarios under test.
//
// The actual runners that build the agent, drive a Runner, and
// compare against the expected digest live in the
// telemetry/functionaltest test package.
package telemetrytestcase
