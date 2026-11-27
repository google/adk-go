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

package agentanalytics_test

import . "google.golang.org/adk/plugin/agentanalytics"

// These external-test variables take the names that were removed from the
// package's exported API without re-exporting them on package agentanalytics
// during `go test`. If package agentanalytics (including via `go test -overlay`)
// exports any of them again as a function, type, variable, or constant, the
// dot-import collides with the package-block declaration below and fails to
// compile, while any reference such as `agentanalytics.SmartTruncate` in
// agentanalytics_test fails as undefined just as it does for downstream callers.
// Because of the dot-import, external test helpers in package agentanalytics_test
// also must not declare names that collide with any exported identifier of
// package agentanalytics.
// Note that this check only sees files included in the current build
// configuration (for example, a `//go:build !race` file re-adding a name is
// not compiled under `go test -race`).
var (
	ArrowSchema           struct{}
	SerializedArrowSchema struct{}
	BatchProcessor        struct{}
	NewBatchProcessor     struct{}
	FormatContentParts    struct{}
	SmartTruncate         struct{}
)

// Reference an exported identifier from the dot-import and each tripwire
// variable so neither the compiler nor the unused linter reports them unused.
var _ = [7]any{
	SchemaVersion,
	ArrowSchema,
	SerializedArrowSchema,
	BatchProcessor,
	NewBatchProcessor,
	FormatContentParts,
	SmartTruncate,
}
