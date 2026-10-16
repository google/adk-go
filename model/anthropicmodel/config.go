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

package anthropicmodel

import (
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"github.com/anthropics/anthropic-sdk-go/option"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/genai"
)

// DefaultMaxTokens is the default max_tokens value sent to Claude when neither
// [genai.GenerateContentConfig.MaxOutputTokens] nor [ClientConfig.MaxTokens] is set.
const DefaultMaxTokens = 8192

// Backend identifies the serving backend for Claude models.
type Backend string

const (
	// BackendAnthropic selects the first-party Anthropic Messages API.
	BackendAnthropic Backend = "anthropic"
	// BackendVertex selects Claude served on Google Cloud Vertex AI (Agent Platform / Model Garden).
	BackendVertex Backend = "vertex"
)

// Effort controls how much reasoning effort Claude spends on its response.
type Effort string

const (
	// EffortLow requests low reasoning effort.
	EffortLow Effort = "low"
	// EffortMedium requests medium reasoning effort.
	EffortMedium Effort = "medium"
	// EffortHigh requests high reasoning effort.
	EffortHigh Effort = "high"
	// EffortXHigh requests extra-high reasoning effort.
	EffortXHigh Effort = "xhigh"
	// EffortMax requests maximum reasoning effort.
	EffortMax Effort = "max"
)

// PromptCacheTTL specifies the time-to-live for an ephemeral prompt cache breakpoint.
type PromptCacheTTL string

const (
	// PromptCacheTTL5m configures a 5-minute ephemeral cache lifetime (the default).
	PromptCacheTTL5m PromptCacheTTL = "5m"
	// PromptCacheTTL1h configures a 1-hour ephemeral cache lifetime.
	PromptCacheTTL1h PromptCacheTTL = "1h"
)

// PromptCacheConfig configures Claude ephemeral prompt caching.
//
// When a non-nil [*PromptCacheConfig] is set on [ClientConfig.PromptCache],
// cache control breakpoints are placed on the last tool declaration, the
// system instruction block, and the last non-thinking content block in the
// conversation history.
type PromptCacheConfig struct {
	// TTL is the cache lifetime ("5m" or "1h"). When empty, PromptCacheTTL5m is used.
	TTL PromptCacheTTL
}

// ClientConfig holds configuration for constructing a Claude [model.LLM] via [NewModel].
type ClientConfig struct {
	// Backend selects BackendAnthropic (default when empty) or BackendVertex.
	Backend Backend

	// APIKey is the Anthropic API key for BackendAnthropic (sent in x-api-key).
	// When empty and BaseURL is not set, ANTHROPIC_API_KEY is consulted.
	APIKey string

	// AuthToken is the Anthropic bearer token for BackendAnthropic (sent in Authorization).
	// When empty and APIKey is not set and BaseURL is not set, ANTHROPIC_AUTH_TOKEN is consulted.
	AuthToken string

	// BaseURL overrides the default API endpoint URL.
	// When empty on BackendAnthropic and no custom BaseURL is set in RequestOptions,
	// ANTHROPIC_BASE_URL is consulted.
	// When set on ClientConfig or via option.WithBaseURL in RequestOptions, an
	// explicit credential must also be set on ClientConfig (APIKey or AuthToken
	// for BackendAnthropic; Credentials or TokenSource for BackendVertex);
	// ambient credentials from environment variables or ADC are never sent to a custom BaseURL.
	BaseURL string

	// Project is the Google Cloud project ID for BackendVertex.
	// When empty, it is resolved from a full Vertex model resource name,
	// GOOGLE_CLOUD_PROJECT, or CLOUD_ML_PROJECT_ID.
	Project string

	// Location is the Google Cloud region (such as "us-east5" or "global") for BackendVertex.
	// When empty, it is resolved from a full Vertex model resource name,
	// GOOGLE_CLOUD_LOCATION, or CLOUD_ML_REGION.
	Location string

	// Credentials supplies explicit Google Cloud credentials for BackendVertex.
	Credentials *google.Credentials

	// TokenSource supplies an explicit OAuth2 token source for BackendVertex when Credentials is nil.
	TokenSource oauth2.TokenSource

	// HTTPClient overrides the HTTP client used for requests.
	HTTPClient *http.Client

	// MaxTokens is the default max_tokens sent when GenerateContentConfig.MaxOutputTokens is 0.
	// Defaults to DefaultMaxTokens (8192) when 0.
	MaxTokens int

	// MaxRetries overrides the Anthropic SDK retry count (default 2 when nil).
	MaxRetries *int

	// Effort configures Claude's output_config.effort ("low", "medium", "high", "xhigh", "max").
	Effort Effort

	// PromptCache enables Claude ephemeral prompt caching when non-nil.
	PromptCache *PromptCacheConfig

	// Logger is the structured logger used for warnings (such as skipped unsupported content).
	// When nil, slog.Default() is used.
	Logger *slog.Logger

	// RequestOptions are additional Anthropic SDK request options applied to the client.
	RequestOptions []option.RequestOption
}

var (
	vertexResourceNamePattern  = regexp.MustCompile(`^projects/([^/]+)/locations/([^/]+)/publishers/anthropic/models/([^/:]+)$`)
	validVertexLocationPattern = regexp.MustCompile(`^[a-z]+(-[a-z0-9]+)*$`)
	validVertexProjectPattern  = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._:-]*$`)
)

func validateVertexProject(project string) error {
	if !validVertexProjectPattern.MatchString(project) {
		return fmt.Errorf("%w: invalid Vertex project %q", ErrInvalidConfig, project)
	}
	return nil
}

func validateVertexLocation(location string) error {
	if !validVertexLocationPattern.MatchString(location) {
		return fmt.Errorf("%w: invalid Vertex location %q; expected a lowercase region such as \"us-east5\" or \"global\"", ErrInvalidConfig, location)
	}
	return nil
}

// parseModelName parses a Claude model identifier (`claude-*`, `anthropic/claude-*`,
// or `projects/<project>/locations/<location>/publishers/anthropic/models/<model>`).
func parseModelName(raw string) (project, location, modelID string, isResource bool, err error) {
	name := strings.TrimSpace(raw)
	if name == "" {
		return "", "", "", false, fmt.Errorf("%w: model name must not be empty", ErrInvalidConfig)
	}
	if strings.HasPrefix(name, "projects/") {
		matches := vertexResourceNamePattern.FindStringSubmatch(name)
		if matches == nil {
			return "", "", "", false, fmt.Errorf("%w: invalid Vertex model resource name format; expected projects/<project>/locations/<location>/publishers/anthropic/models/<model>", ErrInvalidConfig)
		}
		if err := validateVertexProject(matches[1]); err != nil {
			return "", "", "", false, err
		}
		if err := validateVertexLocation(matches[2]); err != nil {
			return "", "", "", false, err
		}
		return matches[1], matches[2], matches[3], true, nil
	}
	modelID = strings.TrimPrefix(name, "anthropic/")
	if modelID == "" || strings.Contains(modelID, "/") {
		return "", "", "", false, fmt.Errorf("%w: invalid model name", ErrInvalidConfig)
	}
	return "", "", modelID, false, nil
}

func validateBaseURL(raw string) error {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return fmt.Errorf("%w: BaseURL must not be whitespace-only", ErrInvalidConfig)
	}
	parsed, err := url.Parse(trimmed)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return fmt.Errorf("%w: BaseURL must be a valid http or https URL with a host", ErrInvalidConfig)
	}
	return nil
}

func validateEffort(e Effort) error {
	switch e {
	case "", EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax:
		return nil
	default:
		return fmt.Errorf("%w: unsupported Effort %q; valid values are low, medium, high, xhigh, max", ErrInvalidConfig, e)
	}
}

func validatePromptCacheConfig(cfg *PromptCacheConfig) error {
	if cfg == nil {
		return nil
	}
	switch cfg.TTL {
	case "", PromptCacheTTL5m, PromptCacheTTL1h:
		return nil
	default:
		return fmt.Errorf("%w: unsupported PromptCache.TTL %q; valid values are 5m, 1h", ErrInvalidConfig, cfg.TTL)
	}
}

// validateGenerateContentConfig validates every top-level field of genai.GenerateContentConfig.
func validateGenerateContentConfig(cfg *genai.GenerateContentConfig, reqTools map[string]any) error {
	if cfg == nil {
		return nil
	}

	if cfg.SystemInstruction != nil {
		for _, part := range cfg.SystemInstruction.Parts {
			if part == nil {
				continue
			}
			if part.Thought || part.InlineData != nil || part.FileData != nil ||
				part.FunctionCall != nil || part.FunctionResponse != nil ||
				part.ExecutableCode != nil || part.CodeExecutionResult != nil ||
				part.VideoMetadata != nil {
				return fmt.Errorf("%w: SystemInstruction only supports text parts", ErrInvalidConfig)
			}
		}
	}

	if cfg.Temperature != nil && cfg.TopP != nil {
		return fmt.Errorf("%w: Temperature and TopP cannot both be set on Claude models", ErrInvalidConfig)
	}
	if cfg.TopK != nil && *cfg.TopK < 0 {
		return fmt.Errorf("%w: TopK must be non-negative", ErrInvalidConfig)
	}
	if cfg.CandidateCount < 0 || cfg.CandidateCount > 1 {
		return fmt.Errorf("%w: CandidateCount %d is not supported; Claude only supports 1 candidate", ErrInvalidConfig, cfg.CandidateCount)
	}
	if cfg.MaxOutputTokens < 0 {
		return fmt.Errorf("%w: MaxOutputTokens must be non-negative", ErrInvalidConfig)
	}

	for _, mod := range cfg.ResponseModalities {
		if !strings.EqualFold(strings.TrimSpace(mod), "TEXT") {
			return fmt.Errorf("%w: ResponseModalities only supports TEXT", ErrInvalidConfig)
		}
	}

	if err := validateThinkingConfig(cfg.ThinkingConfig); err != nil {
		return err
	}

	hasDecls, err := validateTools(cfg.Tools)
	if err != nil {
		return err
	}
	hasAnyTools := hasDecls || len(reqTools) > 0

	if err := validateToolConfig(cfg.ToolConfig, hasAnyTools, cfg.ThinkingConfig); err != nil {
		return err
	}

	if cfg.HTTPOptions != nil {
		if cfg.HTTPOptions.Timeout != nil && *cfg.HTTPOptions.Timeout < 0 {
			return fmt.Errorf("%w: HTTPOptions.Timeout must be non-negative", ErrInvalidConfig)
		}
		if cfg.HTTPOptions.BaseURL != "" {
			return fmt.Errorf("%w: HTTPOptions.BaseURL is not supported; set ClientConfig.BaseURL instead", ErrInvalidConfig)
		}
		if cfg.HTTPOptions.BaseURLResourceScope != "" {
			return fmt.Errorf("%w: HTTPOptions.BaseURLResourceScope is not supported", ErrInvalidConfig)
		}
		if cfg.HTTPOptions.APIVersion != "" {
			return fmt.Errorf("%w: HTTPOptions.APIVersion is not supported", ErrInvalidConfig)
		}
		if cfg.HTTPOptions.ExtraBody != nil {
			return fmt.Errorf("%w: HTTPOptions.ExtraBody is not supported", ErrInvalidConfig)
		}
		if cfg.HTTPOptions.ExtrasRequestProvider != nil {
			return fmt.Errorf("%w: HTTPOptions.ExtrasRequestProvider is not supported", ErrInvalidConfig)
		}
		if cfg.HTTPOptions.RetryOptions != nil {
			return fmt.Errorf("%w: HTTPOptions.RetryOptions is not supported; set ClientConfig.MaxRetries instead", ErrInvalidConfig)
		}
		// HTTPOptions.Headers is ignored by design so Gemini-specific headers never reach Anthropic or Vertex.
	}

	if err := validateOutputFormatConfig(cfg); err != nil {
		return err
	}

	// Reject unsupported GenerateContentConfig fields with errors naming the field.
	if cfg.ResponseLogprobs {
		return fmt.Errorf("%w: ResponseLogprobs is not supported", ErrInvalidConfig)
	}
	if cfg.Logprobs != nil {
		return fmt.Errorf("%w: Logprobs is not supported", ErrInvalidConfig)
	}
	if cfg.PresencePenalty != nil {
		return fmt.Errorf("%w: PresencePenalty is not supported", ErrInvalidConfig)
	}
	if cfg.FrequencyPenalty != nil {
		return fmt.Errorf("%w: FrequencyPenalty is not supported", ErrInvalidConfig)
	}
	if cfg.Seed != nil {
		return fmt.Errorf("%w: Seed is not supported", ErrInvalidConfig)
	}
	if len(cfg.SafetySettings) > 0 {
		return fmt.Errorf("%w: SafetySettings is not supported", ErrInvalidConfig)
	}
	if cfg.CachedContent != "" {
		return fmt.Errorf("%w: CachedContent is not supported; use ClientConfig.PromptCache instead", ErrInvalidConfig)
	}
	if len(cfg.Labels) > 0 {
		return fmt.Errorf("%w: Labels is not supported", ErrInvalidConfig)
	}
	if cfg.MediaResolution != "" {
		return fmt.Errorf("%w: MediaResolution is not supported", ErrInvalidConfig)
	}
	if cfg.SpeechConfig != nil {
		return fmt.Errorf("%w: SpeechConfig is not supported", ErrInvalidConfig)
	}
	if cfg.AudioTimestamp {
		return fmt.Errorf("%w: AudioTimestamp is not supported", ErrInvalidConfig)
	}
	if cfg.ImageConfig != nil {
		return fmt.Errorf("%w: ImageConfig is not supported", ErrInvalidConfig)
	}
	if cfg.RoutingConfig != nil {
		return fmt.Errorf("%w: RoutingConfig is not supported", ErrInvalidConfig)
	}
	if cfg.ModelSelectionConfig != nil {
		return fmt.Errorf("%w: ModelSelectionConfig is not supported", ErrInvalidConfig)
	}
	if cfg.ModelArmorConfig != nil {
		return fmt.Errorf("%w: ModelArmorConfig is not supported", ErrInvalidConfig)
	}
	if cfg.EnableEnhancedCivicAnswers != nil {
		return fmt.Errorf("%w: EnableEnhancedCivicAnswers is not supported", ErrInvalidConfig)
	}
	if cfg.ServiceTier != "" {
		return fmt.Errorf("%w: ServiceTier is not supported", ErrInvalidConfig)
	}
	if cfg.AudioTranscriptionConfig != nil {
		return fmt.Errorf("%w: AudioTranscriptionConfig is not supported", ErrInvalidConfig)
	}
	if len(cfg.ContinuationToken) > 0 {
		return fmt.Errorf("%w: ContinuationToken is not supported", ErrInvalidConfig)
	}
	// AutomaticContinuation is ignored by design.
	return nil
}
