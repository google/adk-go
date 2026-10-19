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

package shared

import (
	"fmt"
	"time"

	oaishared "github.com/openai/openai-go/v3/shared"
	"google.golang.org/genai"
)

// RequestTimeout reports the bound the caller asked for, or zero for none; the
// caller applies it to the context, which bounds retries as openai-go's own
// per-request option would not, and on a streamed turn spans the consumer's
// time in the range body.
//
// Non-positive is treated as unset here rather than trusted to the endpoint
// packages' applyGenerationConfig having rejected it, since openai-go reads zero
// as no deadline at all.
func RequestTimeout(cfg *genai.GenerateContentConfig) time.Duration {
	if cfg == nil || cfg.HTTPOptions == nil || cfg.HTTPOptions.Timeout == nil {
		return 0
	}
	if *cfg.HTTPOptions.Timeout <= 0 {
		return 0
	}
	return *cfg.HTTPOptions.Timeout
}

// IgnoredHTTPOptionFields names the HTTPOptions fields neither endpoint
// translates nor rejects: forwarding a header would let a caller's
// Authorization displace the configured API key and carry a Gemini credential
// to OpenAI, while refusing one would break the configs model/gemini fills in
// itself.
//
// Headers meant for OpenAI belong on ClientConfig.Options, which is scoped to
// the one backend that sees them.
var IgnoredHTTPOptionFields = []string{"Headers"}

// UnsupportedHTTPOptionFields lists the HTTPOptions fields that describe the
// Gemini wire format rather than transport, and so cannot cross to OpenAI.
// Unlike IgnoredHTTPOptionFields these have never been accepted here, so naming
// them costs no compatibility.
var UnsupportedHTTPOptionFields = []struct {
	Name  string
	IsSet func(*genai.HTTPOptions) bool
}{
	// The endpoint belongs to ClientConfig, which is also the only place it can
	// be set coherently alongside the API key that authenticates against it.
	{Name: "BaseURL", IsSet: func(o *genai.HTTPOptions) bool { return o.BaseURL != "" }},
	{Name: "BaseURLResourceScope", IsSet: func(o *genai.HTTPOptions) bool { return o.BaseURLResourceScope != "" }},
	{Name: "APIVersion", IsSet: func(o *genai.HTTPOptions) bool { return o.APIVersion != "" }},
	// Both shape a Gemini request body, which is not the body being sent.
	{Name: "ExtraBody", IsSet: func(o *genai.HTTPOptions) bool { return o.ExtraBody != nil }},
	{Name: "ExtrasRequestProvider", IsSet: func(o *genai.HTTPOptions) bool { return o.ExtrasRequestProvider != nil }},
	// openai-go retries too, but on its own schedule; honoring only the retry
	// count would quietly discard the backoff the caller asked for.
	{Name: "RetryOptions", IsSet: func(o *genai.HTTPOptions) bool { return o.RetryOptions != nil }},
}

// ReasoningEfforts maps every genai thinking level onto an OpenAI reasoning
// effort. An explicit THINKING_LEVEL_UNSPECIFIED is distinct from unset and
// still asks the model to think, so it resolves to medium as adk-python does —
// unlike a dynamic budget, which has no such precedent and defers to the model.
var ReasoningEfforts = map[genai.ThinkingLevel]oaishared.ReasoningEffort{
	genai.ThinkingLevelUnspecified: oaishared.ReasoningEffortMedium,
	genai.ThinkingLevelMinimal:     oaishared.ReasoningEffortMinimal,
	genai.ThinkingLevelLow:         oaishared.ReasoningEffortLow,
	genai.ThinkingLevelMedium:      oaishared.ReasoningEffortMedium,
	genai.ThinkingLevelHigh:        oaishared.ReasoningEffortHigh,
}

// DynamicThinkingBudget is genai's "let the model size its own thinking".
const DynamicThinkingBudget = -1

// ReasoningEffortFor resolves a thinking config to the reasoning effort both
// endpoints accept, a budget surviving only as the distinction between none,
// some, and the model's own choice, since neither endpoint has a token-budget
// knob. An empty effort means send none and let the model choose.
//
// Callers assign the result themselves because the endpoints carry it
// differently: Responses nests it in reasoning, Chat Completions takes a bare
// reasoning_effort. IncludeThoughts is not read here — it maps onto
// reasoning.summary, which only Responses has.
func ReasoningEffortFor(cfg *genai.ThinkingConfig) (oaishared.ReasoningEffort, error) {
	if cfg == nil {
		return "", nil
	}
	if cfg.ThinkingBudget != nil && *cfg.ThinkingBudget < DynamicThinkingBudget {
		// Rejected up here rather than in the branch that reads the budget,
		// because a level set alongside it wins and would otherwise carry the
		// request through with the nonsense value unmentioned.
		return "", fmt.Errorf("%w: ThinkingConfig.ThinkingBudget %d", ErrUnsupportedConfigField, *cfg.ThinkingBudget)
	}
	// A level outranks a budget, but only when it names one: UNSPECIFIED is the
	// caller declining to choose, so a budget they did set is the more specific
	// instruction and takes over.
	level := cfg.ThinkingLevel
	if level == genai.ThinkingLevelUnspecified && cfg.ThinkingBudget != nil {
		level = ""
	}
	switch {
	case level != "":
		effort, ok := ReasoningEfforts[level]
		if !ok {
			// A level genai grew after this map was written: better an error
			// naming it than an effort string the API will reject obscurely.
			return "", fmt.Errorf("%w: ThinkingConfig.ThinkingLevel %q", ErrUnsupportedConfigField, level)
		}
		return effort, nil
	case cfg.ThinkingBudget != nil:
		// Anything below DynamicThinkingBudget was rejected above, so what is
		// left is none of it, the model's choice, or some positive amount.
		switch *cfg.ThinkingBudget {
		case 0:
			// "Do not think" is what the none effort says. Not minimal: minimal
			// is the least thinking rather than none of it, and models are
			// dropping it — gpt-5.4-nano rejects minimal while accepting none.
			return oaishared.ReasoningEffortNone, nil
		case DynamicThinkingBudget:
			// The caller asked the model to decide, so no effort is sent and it
			// does. Pinning a number here would be us deciding instead.
			return "", nil
		default:
			return oaishared.ReasoningEffortMedium, nil
		}
	}
	return "", nil
}

// RejectUntranslatableValues catches the settings whose field is translated but
// whose particular value would vanish, which the presence check below cannot
// see; a value that instead reaches the wire and draws a named 400, as an
// out-of-range Logprobs does, is already diagnosable and is left to the API.
//
// It runs after every named error so that a caller who set one of those too
// gets the error they have always got, rather than this sentinel jumping the
// queue and breaking their errors.Is.
func RejectUntranslatableValues(cfg *genai.GenerateContentConfig) error {
	switch {
	case cfg.Logprobs != nil && !cfg.ResponseLogprobs:
		// Logprobs only sizes the list ResponseLogprobs asks for, so alone it
		// reaches neither params nor the wire.
		return fmt.Errorf("%w: Logprobs without ResponseLogprobs", ErrUnsupportedConfigField)
	case cfg.MaxOutputTokens < 0:
		// Only a positive cap is translated. A negative one is neither a cap
		// nor the absence of one, so it would otherwise vanish.
		return fmt.Errorf("%w: negative MaxOutputTokens", ErrUnsupportedConfigField)
	case cfg.CandidateCount < 0:
		// Above one is ErrMultipleCandidatesNotSupported; zero and one both mean
		// the single candidate either endpoint asks for. Below zero means nothing.
		return fmt.Errorf("%w: negative CandidateCount", ErrUnsupportedConfigField)
	}
	// HTTPOptions is taken field by field rather than whole. Timeout is
	// extracted by RequestTimeout, Headers is deliberately ignored for the
	// compatibility reason in IgnoredHTTPOptionFields, and what is left
	// describes the Gemini wire format and is named here.
	if cfg.HTTPOptions != nil {
		for _, field := range UnsupportedHTTPOptionFields {
			if field.IsSet(cfg.HTTPOptions) {
				return fmt.Errorf("%w: HTTPOptions.%s", ErrUnsupportedConfigField, field.Name)
			}
		}
		// openai-go treats a zero timeout as "no deadline", so forwarding a
		// non-positive one would lift the caller's bound rather than apply it —
		// the inverse of what they asked for, and worse than not asking.
		if cfg.HTTPOptions.Timeout != nil && *cfg.HTTPOptions.Timeout <= 0 {
			return fmt.Errorf("%w: non-positive HTTPOptions.Timeout %v", ErrUnsupportedConfigField, *cfg.HTTPOptions.Timeout)
		}
	}
	return nil
}

// UnsupportedConfigFields lists the GenerateContentConfig fields the Responses
// API cannot take, each with a predicate reporting whether the caller set it;
// the Chat Completions path removes the ones it can with ConfigFieldsWithout.
// Presence, not value: setting a knob at all means the caller expected an effect.
var UnsupportedConfigFields = []ConfigField{
	{Name: "Seed", IsSet: func(c *genai.GenerateContentConfig) bool { return c.Seed != nil }},
	{Name: "RoutingConfig", IsSet: func(c *genai.GenerateContentConfig) bool { return c.RoutingConfig != nil }},
	{Name: "ModelSelectionConfig", IsSet: func(c *genai.GenerateContentConfig) bool { return c.ModelSelectionConfig != nil }},
	{Name: "CachedContent", IsSet: func(c *genai.GenerateContentConfig) bool { return c.CachedContent != "" }},
	{Name: "ResponseModalities", IsSet: func(c *genai.GenerateContentConfig) bool { return c.ResponseModalities != nil }},
	{Name: "MediaResolution", IsSet: func(c *genai.GenerateContentConfig) bool { return c.MediaResolution != "" }},
	{Name: "SpeechConfig", IsSet: func(c *genai.GenerateContentConfig) bool { return c.SpeechConfig != nil }},
	{Name: "AudioTimestamp", IsSet: func(c *genai.GenerateContentConfig) bool { return c.AudioTimestamp }},
	{Name: "ImageConfig", IsSet: func(c *genai.GenerateContentConfig) bool { return c.ImageConfig != nil }},
	{Name: "EnableEnhancedCivicAnswers", IsSet: func(c *genai.GenerateContentConfig) bool {
		return c.EnableEnhancedCivicAnswers != nil
	}},
	{Name: "ModelArmorConfig", IsSet: func(c *genai.GenerateContentConfig) bool { return c.ModelArmorConfig != nil }},
	{Name: "AudioTranscriptionConfig", IsSet: func(c *genai.GenerateContentConfig) bool { return c.AudioTranscriptionConfig != nil }},
	{Name: "ContinuationToken", IsSet: func(c *genai.GenerateContentConfig) bool { return c.ContinuationToken != nil }},
}

// ConfigField names a GenerateContentConfig field alongside a predicate
// reporting whether the caller set it.
type ConfigField struct {
	Name  string
	IsSet func(*genai.GenerateContentConfig) bool
}

// RejectUnsupportedConfigFields reports the first unsupported field the caller set.
func RejectUnsupportedConfigFields(cfg *genai.GenerateContentConfig) error {
	for _, field := range UnsupportedConfigFields {
		if field.IsSet(cfg) {
			return fmt.Errorf("%w: %s", ErrUnsupportedConfigField, field.Name)
		}
	}
	return nil
}

// ConfigFieldsWithout returns UnsupportedConfigFields minus the named entries,
// so an endpoint that supports one of them does not refuse it.
func ConfigFieldsWithout(names ...string) []ConfigField {
	drop := make(map[string]bool, len(names))
	for _, name := range names {
		drop[name] = true
	}
	kept := make([]ConfigField, 0, len(UnsupportedConfigFields))
	for _, field := range UnsupportedConfigFields {
		if !drop[field.Name] {
			kept = append(kept, field)
		}
	}
	return kept
}

// RejectConfigFields reports the first field in the list the caller set.
func RejectConfigFields(fields []ConfigField, cfg *genai.GenerateContentConfig) error {
	for _, field := range fields {
		if field.IsSet(cfg) {
			return fmt.Errorf("%w: %s", ErrUnsupportedConfigField, field.Name)
		}
	}
	return nil
}
