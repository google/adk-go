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
package agent

//  ResumabilityConfig is the config of the resumability for an application.
// The "resumability" in ADK refers to the ability to:
// 1. pause an invocation upon a long-running function call.
// 2. resume an invocation from the last event, if it's paused or failed midway
// through.

// Note: ADK resumes the invocation in a best-effort manner:
// 1. Tool call to resume needs to be idempotent because we only guarantee
// an at-least-once behavior once resumed.
// 2. Any temporary / in-memory state will be lost upon resumption.
type ResumabilityConfig struct {
	// IsResumable indicates whether the app supports agent resumption.
	// If enabled, the feature will be enabled for all agents in the app.
	IsResumable bool
}
