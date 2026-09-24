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

// Package openaicommon holds the parts of the OpenAI conversion that neither
// endpoint owns: schema normalisation, server-text handling, call-id tracking,
// the generation-config fields both reject, and the error sentinels both
// return.
//
// Its identifiers are exported so the sibling endpoint packages can reach them.
// The package is internal, so none of them is public API.
package openaicommon

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"math"
	"reflect"
	"sort"
	"strings"
	"time"

	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// MaxServerTextRunes bounds any server-chosen string an error quotes.
const MaxServerTextRunes = 256

// ClipServerText trims a server-chosen string and caps its length, so that a
// pathological one — a megabyte of message — cannot become the error a caller
// logs. Truncation is marked, so a clipped value does not read as the whole of
// what the server said.
//
// Callers render the result with %q. That, rather than an escaper of our own,
// is what stops a control character in it from forging a line in an operator's
// log; see the same reasoning at [google.golang.org/adk/v2/auth/gcp].
func ClipServerText(s string) string {
	s = strings.TrimSpace(s)
	// Counted by ranging rather than by materialising []rune: an 8 MiB message
	// would otherwise cost 32 MiB to yield at most a kilobyte. Ranging a string
	// yields the byte index of each rune, so s[:i] never splits one.
	n := 0
	for i := range s {
		if n == MaxServerTextRunes {
			return strings.TrimSpace(s[:i]) + "…"
		}
		n++
	}
	return s
}

// SafeInt32 narrows a token count to the int32 genai carries, saturating at
// math.MaxInt32 rather than wrapping.
func SafeInt32(v int64) int32 {
	if v > math.MaxInt32 {
		return math.MaxInt32
	}
	return int32(v)
}

// CallTracker helps us manage function call IDs, ensuring that function responses
// can be correctly associated with their corresponding calls, especially when IDs are not
// explicitly provided in the input.
type CallTracker struct {
	NextID  int
	Pending []string
}

// ResolveResponseID reports the call ID a function response answers, consuming
// it from the pending list. An unset ID pairs with the oldest outstanding call,
// which is the only pairing available when the caller did not supply one.
func (t *CallTracker) ResolveResponseID(fr *genai.FunctionResponse) (string, error) {
	if fr.ID == "" {
		if len(t.Pending) == 0 {
			return "", fmt.Errorf("openai: response for %q missing call id", fr.Name)
		}
		callID := t.Pending[0]
		t.Pending = t.Pending[1:]
		return callID, nil
	}
	for i, pending := range t.Pending {
		if pending == fr.ID {
			t.Pending = append(t.Pending[:i], t.Pending[i+1:]...)
			return fr.ID, nil
		}
	}
	return "", fmt.Errorf("openai: received function response for unknown or already completed call id %q", fr.ID)
}

// TakeCallID reports the ID to send for a function call, minting one when the
// caller left it unset so the matching response can still be paired.
func (t *CallTracker) TakeCallID(fc *genai.FunctionCall) string {
	callID := fc.ID
	if callID == "" {
		callID = fmt.Sprintf("adk-openai-call-%d", t.NextID)
		t.NextID++
	}
	t.Pending = append(t.Pending, callID)
	return callID
}

// MarshalFunctionArgs encodes a call's arguments, reading a nil map as a call
// that takes none rather than as JSON null.
func MarshalFunctionArgs(args map[string]any) (string, error) {
	if args == nil {
		args = map[string]any{}
	}
	encoded, err := json.Marshal(args)
	if err != nil {
		return "", fmt.Errorf("openai: marshal function args: %w", err)
	}
	return string(encoded), nil
}

// RequestTimeout reports the bound the caller asked for, or zero for none; the
// caller applies it to the context, which bounds retries as openai-go's own
// per-request option would not, and on a streamed turn spans the consumer's
// time in the range body.
//
// Non-positive is treated as unset here rather than trusted to
// applyGenerationConfig having rejected it, since openai-go reads zero as no
// deadline at all.
func RequestTimeout(cfg *genai.GenerateContentConfig) time.Duration {
	if cfg == nil || cfg.HTTPOptions == nil || cfg.HTTPOptions.Timeout == nil {
		return 0
	}
	if *cfg.HTTPOptions.Timeout <= 0 {
		return 0
	}
	return *cfg.HTTPOptions.Timeout
}

// IgnoredHTTPOptionFields names the HTTPOptions fields this package neither
// translates nor rejects: forwarding a header would let a caller's
// Authorization displace the configured API key and carry a Gemini credential
// to OpenAI, while refusing one would break the configs model/gemini fills in
// itself.
//
// Headers meant for OpenAI belong on ClientConfig.Options, which is scoped to
// the one backend that sees them.
var IgnoredHTTPOptionFields = []string{"Headers"}

// UnsupportedHTTPOptionFields lists the HTTPOptions fields that describe the
// Gemini wire format rather than transport, and so cannot cross to Responses.
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

// ReasoningEfforts maps every genai thinking level onto a Responses reasoning
// effort. An explicit THINKING_LEVEL_UNSPECIFIED is distinct from unset and
// still asks the model to think, so it resolves to medium as adk-python does —
// unlike a dynamic budget, which has no such precedent and defers to the model.
var ReasoningEfforts = map[genai.ThinkingLevel]shared.ReasoningEffort{
	genai.ThinkingLevelUnspecified: shared.ReasoningEffortMedium,
	genai.ThinkingLevelMinimal:     shared.ReasoningEffortMinimal,
	genai.ThinkingLevelLow:         shared.ReasoningEffortLow,
	genai.ThinkingLevelMedium:      shared.ReasoningEffortMedium,
	genai.ThinkingLevelHigh:        shared.ReasoningEffortHigh,
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
func ReasoningEffortFor(cfg *genai.ThinkingConfig) (shared.ReasoningEffort, error) {
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
			return shared.ReasoningEffortNone, nil
		case DynamicThinkingBudget:
			// The caller asked the model to decide, so no effort is sent and it
			// does. Pinning a number here would be us deciding instead.
			return "", nil
		default:
			return shared.ReasoningEffortMedium, nil
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
		// the single candidate Responses returns. Below zero means nothing.
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

// UnsupportedConfigFields lists the GenerateContentConfig fields this package
// cannot translate, each with a predicate reporting whether the caller set it.
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

// FlattenContentText joins a content's text parts with newlines, and errors on
// a part without text.
func FlattenContentText(content *genai.Content) (string, error) {
	if content == nil {
		return "", nil
	}
	var b strings.Builder
	for _, part := range content.Parts {
		if part == nil {
			continue
		}
		if part.Text == "" {
			return "", fmt.Errorf("non-text system instruction part %T", part)
		}
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(part.Text)
	}
	return b.String(), nil
}

// NormalizeSchema round-trips a JSON schema of any Go shape into a map,
// keeping its numbers as written.
func NormalizeSchema(schema any) (map[string]any, error) {
	if schema == nil {
		return nil, ErrEmptyJSONSchema
	}
	data, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal json schema: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var result map[string]any
	if err := decoder.Decode(&result); err != nil {
		return nil, fmt.Errorf("openai: unmarshal json schema: %w", err)
	}
	PreserveSchemaNumbers(result)
	return result, nil
}

// PreserveSchemaNumbers keeps numeric constraints as raw JSON. The OpenAI SDK
// otherwise serializes json.Number values as strings.
func PreserveSchemaNumbers(val any) any {
	switch v := val.(type) {
	case json.Number:
		return json.RawMessage(v.String())
	case map[string]any:
		for key, child := range v {
			v[key] = PreserveSchemaNumbers(child)
		}
	case []any:
		for i, child := range v {
			v[i] = PreserveSchemaNumbers(child)
		}
	}
	return val
}

// EnforceStrictOpenAISchema recursively walks the schema and enforces the rules
// required by OpenAI's structured outputs with strict=true: every object type
// carries properties, additionalProperties=false and a required array naming
// every property, and a $ref keeps no siblings. An object that declares no
// properties is given an empty properties map and an empty required array
// alongside additionalProperties=false, because the API rejects the whole
// request when any object in the schema omits one of the three. Any
// additionalProperties the caller wrote is replaced: strict mode accepts only
// false, so a schema spelling a map as additionalProperties={"type":"string"}
// becomes an empty object rather than the 400 it would otherwise draw.
//
// Treating a property-less object that way diverges from adk-python
// deliberately. Its _enforce_strict_openai_schema rewrites an object only when
// the schema already carries a properties key, which leaves one without to fail
// the same request.
func EnforceStrictOpenAISchema(val any) {
	schema, ok := val.(map[string]any)
	if !ok {
		return
	}

	if _, hasRef := schema["$ref"]; hasRef {
		for key := range schema {
			if key != "$ref" {
				delete(schema, key)
			}
		}
		return
	}

	t, hasType := schema["type"]
	isObj := hasType && t == "object"
	propsMap, _ := schema["properties"].(map[string]any)

	if isObj {
		if propsMap == nil {
			propsMap = map[string]any{}
			schema["properties"] = propsMap
		}
		schema["additionalProperties"] = false
		req := make([]string, 0, len(propsMap))
		for k := range propsMap {
			req = append(req, k)
		}
		sort.Strings(req)
		schema["required"] = req
	}

	if defsVal, ok := schema["$defs"]; ok {
		if defsMap, ok := defsVal.(map[string]any); ok {
			for _, defn := range defsMap {
				EnforceStrictOpenAISchema(defn)
			}
		}
	}

	for _, prop := range propsMap {
		EnforceStrictOpenAISchema(prop)
	}

	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if arrVal, ok := schema[key]; ok {
			if arr, ok := arrVal.([]any); ok {
				for _, item := range arr {
					EnforceStrictOpenAISchema(item)
				}
			}
		}
	}

	if itemsVal, ok := schema["items"]; ok {
		if _, isMap := itemsVal.(map[string]any); isMap {
			EnforceStrictOpenAISchema(itemsVal)
		}
	}
}

// EnsureFunctionToolOnly rejects a nil tool, one declaring no functions, and
// one carrying a non-function tool neither API is sent; idx names it.
func EnsureFunctionToolOnly(idx int, tool *genai.Tool) error {
	if tool == nil {
		return fmt.Errorf("openai: tool %d is nil", idx)
	}
	if tool.Retrieval != nil || tool.GoogleSearch != nil || tool.GoogleSearchRetrieval != nil ||
		tool.GoogleMaps != nil || tool.EnterpriseWebSearch != nil ||
		tool.URLContext != nil || tool.ComputerUse != nil || tool.CodeExecution != nil {
		return fmt.Errorf("openai: non-function tools are not supported (tool %d)", idx)
	}
	if len(tool.FunctionDeclarations) == 0 {
		return fmt.Errorf("openai: tool %d does not declare any functions", idx)
	}
	return nil
}

// SchemaToMap converts a genai schema to the JSON-schema map both APIs take,
// with its type names lowercased.
func SchemaToMap(schema *genai.Schema) (map[string]any, error) {
	if schema == nil {
		return nil, nil
	}
	bytes, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal schema: %w", err)
	}
	var result map[string]any
	if err := json.Unmarshal(bytes, &result); err != nil {
		return nil, fmt.Errorf("openai: unmarshal schema: %w", err)
	}
	LowercaseSchemaTypes(result)
	return result, nil
}

// LowercaseSchemaTypes rewrites genai's upper-case type names, at every depth
// of a decoded schema, to the lower case JSON Schema uses.
func LowercaseSchemaTypes(val any) {
	switch v := val.(type) {
	case map[string]any:
		if t, ok := v["type"]; ok {
			switch tVal := t.(type) {
			case string:
				v["type"] = strings.ToLower(tVal)
			case []any:
				for i, item := range tVal {
					if str, ok := item.(string); ok {
						tVal[i] = strings.ToLower(str)
					}
				}
			}
		}
		for _, child := range v {
			LowercaseSchemaTypes(child)
		}
	case []any:
		for _, child := range v {
			LowercaseSchemaTypes(child)
		}
	}
}

// SingleErrorSequence is a response sequence that yields err and ends.
func SingleErrorSequence(err error) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(nil, err)
	}
}

// IsEmptyOutput reports whether a conversion failed for want of anything to
// convert, as against something unusable.
func IsEmptyOutput(err error) bool {
	return errors.Is(err, ErrNoOutputItems) || errors.Is(err, ErrNoTextOrToolContent) ||
		errors.Is(err, ErrNoChoices)
}

// CarriesContent reports whether a response holds anything a caller can read.
// An aggregated turn can arrive empty: a streamed function call with no name is
// dropped, and deltas may contribute no part at all.
func CarriesContent(resp *model.LLMResponse) bool {
	return resp != nil && resp.Content != nil && len(resp.Content.Parts) > 0
}

// CompletedContentSupersedes reports whether a terminal snapshot can safely
// replace content assembled from stream deltas. Reasoning alone is not a usable
// replacement, and the snapshot must retain all visible text and function calls
// that callers already received from the stream.
// When replacement is allowed, reasoning follows the completed response to match
// the blocking path. Streamed thoughts absent from that snapshot are not added
// back; they have already been delivered as partial responses.
func CompletedContentSupersedes(aggregate, completed *genai.Content) bool {
	if completed == nil {
		return false
	}

	var aggregateText, completedText strings.Builder
	var aggregateCalls, completedCalls []*genai.FunctionCall
	usable := false
	for _, part := range aggregate.Parts {
		if part == nil {
			continue
		}
		if part.Text != "" && !part.Thought {
			aggregateText.WriteString(part.Text)
		}
		if part.FunctionCall != nil {
			aggregateCalls = append(aggregateCalls, part.FunctionCall)
		}
	}
	for _, part := range completed.Parts {
		if part == nil {
			continue
		}
		if part.Text != "" && !part.Thought {
			completedText.WriteString(part.Text)
			usable = true
		}
		if part.FunctionCall != nil {
			completedCalls = append(completedCalls, part.FunctionCall)
			usable = true
		}
	}
	if !usable || !strings.Contains(completedText.String(), aggregateText.String()) {
		return false
	}

	matched := make([]bool, len(completedCalls))
	for _, aggregateCall := range aggregateCalls {
		found := false
		for i, completedCall := range completedCalls {
			if !matched[i] && reflect.DeepEqual(aggregateCall, completedCall) {
				matched[i] = true
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// AnswerText is the response's text as a caller reads it, thoughts excluded:
// what logprobs have to describe.
func AnswerText(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var text strings.Builder
	for _, part := range content.Parts {
		if part.Thought {
			continue
		}
		text.WriteString(part.Text)
	}
	return text.String()
}

// SinglePartResponse wraps one streamed part as a genai response.
//
// The candidate deliberately carries no finish reason: the aggregator treats any
// non-empty one as terminal, and genai.FinishReasonUnspecified is the non-empty
// string "FINISH_REASON_UNSPECIFIED", so setting it marks every delta the end of
// the turn. generateStream reports the real reason on the final response.
func SinglePartResponse(part *genai.Part) *genai.GenerateContentResponse {
	if part == nil {
		return nil
	}
	return &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role:  string(genai.RoleModel),
					Parts: []*genai.Part{part},
				},
			},
		},
	}
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
