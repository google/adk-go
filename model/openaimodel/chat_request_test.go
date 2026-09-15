// Copyright 2025 Google LLC
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

package openaimodel

import "testing"

func TestBuildChatParams_Roles(t *testing.T) {
	t.Skip("scaffold")
}

func TestBuildChatParams_SystemInstructionLeadsMessages(t *testing.T) {
	t.Skip("scaffold")
}

func TestBuildChatParams_MultiTurnHistory(t *testing.T) {
	t.Skip("scaffold")
}

func TestBuildChatParams_ToolCallAndResultPairing(t *testing.T) {
	t.Skip("scaffold")
}

func TestBuildChatParams_ToolResultWithoutCallID(t *testing.T) {
	t.Skip("scaffold")
}

// TestBuildChatParams_ThoughtsNotReplayed pins the deliberate divergence from
// the Responses path, which replays prior-turn reasoning as assistant text.
func TestBuildChatParams_ThoughtsNotReplayed(t *testing.T) {
	t.Skip("scaffold")
}

// TestBuildChatParams_WirePayload asserts the marshalled JSON rather than the
// Go structs, where omitzero and union arms hide this package's bugs.
func TestBuildChatParams_WirePayload(t *testing.T) {
	t.Skip("scaffold")
}

func TestApplyChatGenerationConfig_TranslatedFields(t *testing.T) {
	t.Skip("scaffold")
}

// TestApplyChatGenerationConfig_ResponsesOnlyRejections covers the fields Chat
// Completions honours that the Responses path rejects: stop sequences, the
// penalties and seed.
func TestApplyChatGenerationConfig_ResponsesOnlyRejections(t *testing.T) {
	t.Skip("scaffold")
}

func TestApplyChatGenerationConfig_RejectedFields(t *testing.T) {
	t.Skip("scaffold")
}

func TestApplyChatThinkingConfig(t *testing.T) {
	t.Skip("scaffold")
}
