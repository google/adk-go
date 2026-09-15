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

// TestConvertChatTools_NestedFunctionShape pins the extra "function" level Chat
// Completions requires and the Responses shape does not have.
func TestConvertChatTools_NestedFunctionShape(t *testing.T) {
	t.Skip("scaffold")
}

func TestConvertChatTools_RejectsNonFunctionTools(t *testing.T) {
	t.Skip("scaffold")
}

func TestConvertChatToolChoice_Modes(t *testing.T) {
	t.Skip("scaffold")
}

func TestConvertChatToolChoice_AllowedTools(t *testing.T) {
	t.Skip("scaffold")
}
