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

// Config holds configuration for creating an Anthropic Claude model.
type Config struct {
	// APIKey is the Anthropic API key for direct API access.
	// If not provided, it will be read from the ANTHROPIC_API_KEY environment variable.
	// This is only used when Variant is VariantAnthropicAPI.
	APIKey string

	// VertexProjectID is the Google Cloud project ID for Vertex AI access.
	// If not provided, it will be read from the GOOGLE_CLOUD_PROJECT environment variable.
	// This is only used when Variant is VariantVertexAI.
	VertexProjectID string

	// VertexRegion is the Google Cloud region for Vertex AI access.
	// If not provided, it will be read from the GOOGLE_CLOUD_REGION environment variable.
	// Common regions include "us-central1", "us-east5", and "europe-west1".
	// This is only used when Variant is VariantVertexAI.
	VertexRegion string

	// Variant determines which backend to use for API calls.
	// Valid values are VariantAnthropicAPI and VariantVertexAI.
	// If empty, the variant is determined from the ANTHROPIC_USE_VERTEX environment variable.
	Variant string

	// DefaultMaxTokens is the default maximum number of tokens to generate.
	// Anthropic requires max_tokens to be explicitly set for all requests.
	// If not provided, defaults to 4096.
	DefaultMaxTokens int
}
