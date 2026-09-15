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

func TestNewModel_SelectsAPI(t *testing.T) {
	t.Skip("scaffold")
}

func TestNewModel_RejectsUnknownAPI(t *testing.T) {
	t.Skip("scaffold")
}

func TestChatModel_GenerateContent_NilRequest(t *testing.T) {
	t.Skip("scaffold")
}

func TestChatModel_GenerateContent_PostsToChatCompletions(t *testing.T) {
	t.Skip("scaffold")
}

func TestChatModel_GenerateContent_HonoursTimeout(t *testing.T) {
	t.Skip("scaffold")
}

// TestChatModel_GenerateStream_MatchesBlocking is the Chat-path counterpart of
// the guard that keeps the two Responses paths from drifting apart.
func TestChatModel_GenerateStream_MatchesBlocking(t *testing.T) {
	t.Skip("scaffold")
}
