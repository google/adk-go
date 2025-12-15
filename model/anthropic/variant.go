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

package anthropic

import (
	"os"
	"strconv"
	"strings"
)

// Backend variant constants.
const (
	// VariantAnthropicAPI uses the direct Anthropic API.
	VariantAnthropicAPI = "ANTHROPIC_API"

	// VariantVertexAI uses Anthropic models via Google Cloud Vertex AI.
	VariantVertexAI = "VERTEX_AI"
)

// GetVariant returns the configured variant for the Anthropic backend.
// It checks the ANTHROPIC_USE_VERTEX environment variable.
// If set to "1" or "true" (case-insensitive), it returns VariantVertexAI.
// Otherwise, it returns VariantAnthropicAPI.
func GetVariant() string {
	b, _ := strconv.ParseBool(strings.TrimSpace(os.Getenv("ANTHROPIC_USE_VERTEX")))
	if b {
		return VariantVertexAI
	}
	return VariantAnthropicAPI
}
