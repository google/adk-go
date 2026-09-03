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

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// buildOpenAIParams converts a generic LLMRequest into the OpenAI-specific
// responses.ResponseNewParams format, preparing it for an API call.
func buildOpenAIParams(modelName string, req *model.LLMRequest) (responses.ResponseNewParams, error) {
	if req == nil {
		return responses.ResponseNewParams{}, ErrRequestNil
	}

	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(modelName),
	}
	if req.Model != "" {
		params.Model = shared.ResponsesModel(req.Model)
	}

	// We convert the generic content parts into OpenAI's input format.
	input, err := convertContents(req.Contents)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}
	if len(input) == 0 {
		return responses.ResponseNewParams{}, ErrNoContents
	}
	params.Input = responses.ResponseNewParamsInputUnion{
		OfInputItemList: input,
	}

	// Apply generation configuration settings like temperature and max output tokens.
	if err := applyGenerationConfig(&params, req.Config); err != nil {
		return responses.ResponseNewParams{}, err
	}

	// Convert any specified tools into the OpenAI tool format.
	tools, err := convertTools(req.Config)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	// Handle tool choice configuration, if provided.
	if cfg := req.Config; cfg != nil && cfg.ToolConfig != nil {
		choice, err := convertToolChoice(cfg.ToolConfig)
		if err != nil {
			return responses.ResponseNewParams{}, err
		}
		if choice != nil {
			params.ToolChoice = *choice
		}
	}

	return params, nil
}

func convertContents(contents []*genai.Content) (responses.ResponseInputParam, error) {
	var (
		items     responses.ResponseInputParam
		tracker   callTracker
		textParts []string
		curRole   genai.Role = genai.RoleUser
		// flushText is a helper function that takes any accumulated text parts
		// and converts them into a message, then appends it to our items.
		flushText = func() error {
			if len(textParts) == 0 {
				return nil
			}
			msgRole, err := normalizeRole(curRole)
			if err != nil {
				return err
			}
			// The Responses API rejects "input_text" for the assistant role, so
			// a replayed assistant turn goes out as an output message instead.
			if msgRole == responses.EasyInputMessageRoleAssistant {
				if msg := newOutputMessage(textParts); msg != nil {
					items = append(items, responses.ResponseInputItemUnionParam{OfOutputMessage: msg})
				}
			} else {
				if msg := newMessage(msgRole, textParts); msg != nil {
					items = append(items, responses.ResponseInputItemUnionParam{OfMessage: msg})
				}
			}
			textParts = textParts[:0]
			return nil
		}
	)

	for _, content := range contents {
		if content == nil || len(content.Parts) == 0 {
			continue
		}
		curRole = genai.Role(content.Role)
		for _, part := range content.Parts {
			switch {
			case part == nil:
				continue
			case part.Text != "":
				textParts = append(textParts, part.Text)
			case part.FunctionCall != nil:
				// If we encounter a function call, we first flush any accumulated text.
				if err := flushText(); err != nil {
					return nil, err
				}
				callParam, err := tracker.newFunctionCall(part.FunctionCall)
				if err != nil {
					return nil, err
				}
				items = append(items, responses.ResponseInputItemUnionParam{OfFunctionCall: callParam})
			case part.FunctionResponse != nil:
				// Similarly, for a function response, we flush text before adding the response.
				if err := flushText(); err != nil {
					return nil, err
				}
				respParam, err := tracker.newFunctionResponse(part.FunctionResponse)
				if err != nil {
					return nil, err
				}
				items = append(items, responses.ResponseInputItemUnionParam{OfFunctionCallOutput: respParam})
			default:
				return nil, fmt.Errorf("openai: unsupported content part %T", part)
			}
		}
		// After processing all parts in a content block, we flush any remaining text.
		if err := flushText(); err != nil {
			return nil, err
		}
	}

	return items, nil
}

// newMessage builds an easy input message for an already-normalized role.
func newMessage(msgRole responses.EasyInputMessageRole, texts []string) *responses.EasyInputMessageParam {
	if len(texts) == 0 {
		return nil
	}
	contentList := make(responses.ResponseInputMessageContentListParam, 0, len(texts))
	for _, txt := range texts {
		if strings.TrimSpace(txt) == "" {
			continue
		}
		textParam := responses.ResponseInputTextParam{
			Text: txt,
			Type: constant.InputText("input_text"),
		}
		contentList = append(contentList, responses.ResponseInputContentUnionParam{
			OfInputText: &textParam,
		})
	}
	if len(contentList) == 0 {
		return nil
	}
	return &responses.EasyInputMessageParam{
		Role: msgRole,
		Type: responses.EasyInputMessageTypeMessage,
		Content: responses.EasyInputMessageContentUnionParam{
			OfInputItemContentList: contentList,
		},
	}
}

// newOutputMessage builds an assistant output message whose content uses the
// "output_text" type, as required when replaying a prior assistant turn to the
// OpenAI Responses API.
func newOutputMessage(texts []string) *responses.ResponseOutputMessageParam {
	if len(texts) == 0 {
		return nil
	}
	contentList := make([]responses.ResponseOutputMessageContentUnionParam, 0, len(texts))
	for _, txt := range texts {
		if strings.TrimSpace(txt) == "" {
			continue
		}
		contentList = append(contentList, responses.ResponseOutputMessageContentUnionParam{
			OfOutputText: &responses.ResponseOutputTextParam{
				Text: txt,
				Type: constant.OutputText("output_text"),
			},
		})
	}
	if len(contentList) == 0 {
		return nil
	}
	return &responses.ResponseOutputMessageParam{
		Content: contentList,
		Status:  responses.ResponseOutputMessageStatusCompleted,
	}
}

func normalizeRole(role genai.Role) (responses.EasyInputMessageRole, error) {
	switch role {
	case "", genai.RoleUser:
		return responses.EasyInputMessageRoleUser, nil
	case genai.RoleModel:
		return responses.EasyInputMessageRoleAssistant, nil
	case "system":
		return responses.EasyInputMessageRoleSystem, nil
	case "developer":
		return responses.EasyInputMessageRoleDeveloper, nil
	default:
		return "", fmt.Errorf("openai: unsupported role %q", role)
	}
}

// callTracker helps us manage function call IDs, ensuring that function responses
// can be correctly associated with their corresponding calls, especially when IDs are not
// explicitly provided in the input.
type callTracker struct {
	nextID  int
	pending []string
}

// newFunctionCall converts a generic genai.FunctionCall into an OpenAI-specific
// ResponseFunctionToolCallParam. We generate a unique callID if one isn't
// provided, and then marshal the function arguments into a JSON string.
func (t *callTracker) newFunctionCall(fc *genai.FunctionCall) (*responses.ResponseFunctionToolCallParam, error) {
	if fc.Name == "" {
		return nil, ErrFunctionCallMissingName
	}
	callID := fc.ID
	if callID == "" {
		callID = fmt.Sprintf("adk-openai-call-%d", t.nextID)
		t.nextID++
	}
	t.pending = append(t.pending, callID)
	argsValue := fc.Args
	if argsValue == nil {
		argsValue = map[string]any{}
	}
	args, err := json.Marshal(argsValue)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal function args: %w", err)
	}
	return &responses.ResponseFunctionToolCallParam{
		Name:      fc.Name,
		CallID:    callID,
		Arguments: string(args),
		Type:      constant.FunctionCall("function_call"),
	}, nil
}

// newFunctionResponse converts a generic genai.FunctionResponse into an OpenAI-specific
// ResponseInputItemFunctionCallOutputParam. We try to match the response to a pending
// function call. If an explicit callID is provided, we find and remove it from our
// pending list. Otherwise, we assume it corresponds to the oldest pending call.
func (t *callTracker) newFunctionResponse(fr *genai.FunctionResponse) (*responses.ResponseInputItemFunctionCallOutputParam, error) {
	callID := fr.ID
	if callID == "" {
		if len(t.pending) == 0 {
			return nil, fmt.Errorf("openai: response for %q missing call id", fr.Name)
		}
		callID = t.pending[0]
		t.pending = t.pending[1:]
	} else {
		found := false
		for i, pending := range t.pending {
			if pending == callID {
				t.pending = append(t.pending[:i], t.pending[i+1:]...)
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("openai: received function response for unknown or already completed call id %q", callID)
		}
	}
	payload, err := json.Marshal(fr.Response)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal function response: %w", err)
	}
	return &responses.ResponseInputItemFunctionCallOutputParam{
		CallID: param.NewOpt(callID),
		Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
			OfString: param.NewOpt(string(payload)),
		},
		Type: constant.FunctionCallOutput("function_call_output"),
	}, nil
}

// applyGenerationConfig translates our generic generation configuration into
// OpenAI-specific parameters. We also validate and return errors for features
// that are not supported by the OpenAI Responses API.
func applyGenerationConfig(params *responses.ResponseNewParams, cfg *genai.GenerateContentConfig) error {
	if cfg == nil {
		return nil
	}
	if cfg.Temperature != nil {
		params.Temperature = param.NewOpt(float64(*cfg.Temperature))
	}
	if cfg.TopP != nil {
		params.TopP = param.NewOpt(float64(*cfg.TopP))
	}
	if cfg.TopK != nil {
		return ErrTopKNotSupported
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(cfg.MaxOutputTokens))
	}
	if len(cfg.StopSequences) > 0 {
		return ErrStopSequencesNotSupported
	}
	if cfg.CandidateCount > 1 {
		return ErrMultipleCandidatesNotSupported
	}
	if cfg.FrequencyPenalty != nil || cfg.PresencePenalty != nil {
		return ErrPenaltiesNotSupported
	}
	if cfg.ResponseLogprobs {
		if cfg.Logprobs != nil {
			params.TopLogprobs = param.NewOpt(int64(*cfg.Logprobs))
		} else {
			params.TopLogprobs = param.NewOpt(int64(1))
		}
		// Responses returns logprobs only when explicitly included.
		params.Include = append(params.Include, responses.ResponseIncludableMessageOutputTextLogprobs)
	}
	if cfg.SystemInstruction != nil {
		inst, err := flattenContentText(cfg.SystemInstruction)
		if err != nil {
			return fmt.Errorf("openai: system instruction: %w", err)
		}
		if inst != "" {
			params.Instructions = param.NewOpt(inst)
		}
	}
	if cfg.ResponseMIMEType != "" && cfg.ResponseMIMEType != "text/plain" && cfg.ResponseMIMEType != "application/json" {
		return fmt.Errorf("%w: %s", ErrUnsupportedMIMEType, cfg.ResponseMIMEType)
	}
	if cfg.ResponseMIMEType == "application/json" || cfg.ResponseSchema != nil || cfg.ResponseJsonSchema != nil {
		if cfg.ResponseSchema == nil && cfg.ResponseJsonSchema == nil {
			obj := shared.NewResponseFormatJSONObjectParam()
			params.Text = responses.ResponseTextConfigParam{
				Format: responses.ResponseFormatTextConfigUnionParam{
					OfJSONObject: &obj,
				},
			}
		} else {
			format, err := newJSONSchemaFormat(cfg)
			if err != nil {
				return err
			}
			params.Text = responses.ResponseTextConfigParam{
				Format: responses.ResponseFormatTextConfigUnionParam{
					OfJSONSchema: format,
				},
			}
		}
	}
	if cfg.Labels != nil {
		return ErrLabelsNotSupported
	}
	if cfg.SafetySettings != nil {
		return ErrSafetySettingsNotSupported
	}
	if err := applyThinkingConfig(params, cfg.ThinkingConfig); err != nil {
		return err
	}
	if err := rejectUntranslatableValues(cfg); err != nil {
		return err
	}
	// Last, so the named errors above win when a caller sets both.
	return rejectUnsupportedConfigFields(cfg)
}

// reasoningEfforts maps every genai thinking level onto a Responses reasoning
// effort. An explicit THINKING_LEVEL_UNSPECIFIED is distinct from unset and
// still asks the model to think, so it resolves to medium.
var reasoningEfforts = map[genai.ThinkingLevel]shared.ReasoningEffort{
	genai.ThinkingLevelUnspecified: shared.ReasoningEffortMedium,
	genai.ThinkingLevelMinimal:     shared.ReasoningEffortMinimal,
	genai.ThinkingLevelLow:         shared.ReasoningEffortLow,
	genai.ThinkingLevelMedium:      shared.ReasoningEffortMedium,
	genai.ThinkingLevelHigh:        shared.ReasoningEffortHigh,
}

// dynamicThinkingBudget is genai's "let the model size its own thinking".
const dynamicThinkingBudget = -1

// applyThinkingConfig maps genai's thinking config onto effort-based reasoning.
// Responses has no token-budget knob, so a budget survives only as the
// distinction between none, some, and the model's own choice.
//
// Summary is sent only when the caller sets IncludeThoughts. Reasoning
// summaries require a verified OpenAI organization, so requesting one
// unprompted would fail every reasoning call made by an unverified org — the
// reason adk-python's unconditional summary is not copied here.
func applyThinkingConfig(params *responses.ResponseNewParams, cfg *genai.ThinkingConfig) error {
	if cfg == nil {
		return nil
	}
	if cfg.ThinkingBudget != nil && *cfg.ThinkingBudget < dynamicThinkingBudget {
		// Rejected up here rather than in the branch that reads the budget,
		// because a level set alongside it wins and would otherwise carry the
		// request through with the nonsense value unmentioned.
		return fmt.Errorf("%w: ThinkingConfig.ThinkingBudget %d", ErrUnsupportedConfigField, *cfg.ThinkingBudget)
	}
	// A level outranks a budget, but only when it names one: UNSPECIFIED is the
	// caller declining to choose, so a budget they did set is the more specific
	// instruction and takes over.
	level := cfg.ThinkingLevel
	if level == genai.ThinkingLevelUnspecified && cfg.ThinkingBudget != nil {
		level = ""
	}

	var reasoning shared.ReasoningParam
	switch {
	case level != "":
		effort, ok := reasoningEfforts[level]
		if !ok {
			// A level genai grew after this map was written: better an error
			// naming it than an effort string the API will reject obscurely.
			return fmt.Errorf("%w: ThinkingConfig.ThinkingLevel %q", ErrUnsupportedConfigField, level)
		}
		reasoning.Effort = effort
	case cfg.ThinkingBudget != nil:
		// Anything below dynamicThinkingBudget was rejected above, so what is
		// left is none of it, the model's choice, or some positive amount.
		switch *cfg.ThinkingBudget {
		case 0:
			// "Do not think" is what the none effort says. Not minimal: minimal
			// is the least thinking rather than none of it, and models are
			// dropping it — gpt-5.4-nano rejects minimal while accepting none.
			reasoning.Effort = shared.ReasoningEffortNone
		case dynamicThinkingBudget:
			// The caller asked the model to decide, so no effort is sent and it
			// does. Pinning a number here would be us deciding instead.
		default:
			reasoning.Effort = shared.ReasoningEffortMedium
		}
	case !cfg.IncludeThoughts:
		return fmt.Errorf("%w: ThinkingConfig needs ThinkingLevel, ThinkingBudget or IncludeThoughts", ErrUnsupportedConfigField)
	}
	// IncludeThoughts alone leaves Effort unset, letting the model pick it, and
	// asks only for the summaries that response.go surfaces as thought parts.
	if cfg.IncludeThoughts {
		reasoning.Summary = shared.ReasoningSummaryAuto
	}
	params.Reasoning = reasoning
	return nil
}

// rejectUntranslatableValues catches the settings whose field is translated but
// whose particular value is not, which the presence check below cannot see.
//
// It runs after every named error so that a caller who set one of those too
// gets the error they have always got, rather than this sentinel jumping the
// queue and breaking their errors.Is.
func rejectUntranslatableValues(cfg *genai.GenerateContentConfig) error {
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
	return nil
}

// unsupportedConfigFields lists the GenerateContentConfig fields this package
// cannot translate, each with a predicate reporting whether the caller set it.
// Presence, not value: setting a knob at all means the caller expected an effect.
// ServiceTier could be honored (openai-go has service_tier) and is left for a
// separate change; HTTPOptions is transport, taken from ClientConfig instead.
var unsupportedConfigFields = []struct {
	name  string
	isSet func(*genai.GenerateContentConfig) bool
}{
	{"HTTPOptions", func(c *genai.GenerateContentConfig) bool { return c.HTTPOptions != nil }},
	{"Seed", func(c *genai.GenerateContentConfig) bool { return c.Seed != nil }},
	{"RoutingConfig", func(c *genai.GenerateContentConfig) bool { return c.RoutingConfig != nil }},
	{"ModelSelectionConfig", func(c *genai.GenerateContentConfig) bool { return c.ModelSelectionConfig != nil }},
	{"CachedContent", func(c *genai.GenerateContentConfig) bool { return c.CachedContent != "" }},
	{"ResponseModalities", func(c *genai.GenerateContentConfig) bool { return c.ResponseModalities != nil }},
	{"MediaResolution", func(c *genai.GenerateContentConfig) bool { return c.MediaResolution != "" }},
	{"SpeechConfig", func(c *genai.GenerateContentConfig) bool { return c.SpeechConfig != nil }},
	{"AudioTimestamp", func(c *genai.GenerateContentConfig) bool { return c.AudioTimestamp }},
	{"ImageConfig", func(c *genai.GenerateContentConfig) bool { return c.ImageConfig != nil }},
	{"EnableEnhancedCivicAnswers", func(c *genai.GenerateContentConfig) bool {
		return c.EnableEnhancedCivicAnswers != nil
	}},
	{"ModelArmorConfig", func(c *genai.GenerateContentConfig) bool { return c.ModelArmorConfig != nil }},
	{"ServiceTier", func(c *genai.GenerateContentConfig) bool { return c.ServiceTier != "" }},
	{"AudioTranscriptionConfig", func(c *genai.GenerateContentConfig) bool { return c.AudioTranscriptionConfig != nil }},
}

// rejectUnsupportedConfigFields reports the first unsupported field the caller set.
func rejectUnsupportedConfigFields(cfg *genai.GenerateContentConfig) error {
	for _, field := range unsupportedConfigFields {
		if field.isSet(cfg) {
			return fmt.Errorf("%w: %s", ErrUnsupportedConfigField, field.name)
		}
	}
	return nil
}

func flattenContentText(content *genai.Content) (string, error) {
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

// newJSONSchemaFormat constructs an OpenAI-specific JSON schema format from our
// generic GenerateContentConfig. We handle cases where the schema is provided
// directly or needs to be converted, and assign a name to it.
func newJSONSchemaFormat(cfg *genai.GenerateContentConfig) (*responses.ResponseFormatTextJSONSchemaConfigParam, error) {
	var (
		schema map[string]any
		err    error
	)
	switch {
	case cfg.ResponseJsonSchema != nil:
		schema, err = normalizeSchema(cfg.ResponseJsonSchema)
	case cfg.ResponseSchema != nil:
		schema, err = schemaToMap(cfg.ResponseSchema)
	default:
		return nil, fmt.Errorf("openai: json schema requested without schema")
	}
	if err != nil {
		return nil, err
	}
	enforceStrictOpenAISchema(schema)
	name := "adk_response"
	if cfg.ResponseSchema != nil && cfg.ResponseSchema.Title != "" {
		name = cfg.ResponseSchema.Title
	}
	return &responses.ResponseFormatTextJSONSchemaConfigParam{
		Name:   name,
		Schema: schema,
		Strict: param.NewOpt(true),
		Type:   constant.JSONSchema("json_schema"),
	}, nil
}

func normalizeSchema(schema any) (map[string]any, error) {
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
	preserveSchemaNumbers(result)
	return result, nil
}

// preserveSchemaNumbers keeps numeric constraints as raw JSON. The OpenAI SDK
// otherwise serializes json.Number values as strings.
func preserveSchemaNumbers(val any) any {
	switch v := val.(type) {
	case json.Number:
		return json.RawMessage(v.String())
	case map[string]any:
		for key, child := range v {
			v[key] = preserveSchemaNumbers(child)
		}
	case []any:
		for i, child := range v {
			v[i] = preserveSchemaNumbers(child)
		}
	}
	return val
}

// enforceStrictOpenAISchema recursively walks the schema and enforces the rules
// required by OpenAI's structured outputs with strict=true. Specifically, it
// sets additionalProperties=false on all object types, and ensures that all
// properties are listed in the required array.
func enforceStrictOpenAISchema(val any) {
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
	propsVal, hasProps := schema["properties"]

	if isObj && hasProps {
		schema["additionalProperties"] = false
		if propsMap, ok := propsVal.(map[string]any); ok {
			req := make([]string, 0, len(propsMap))
			for k := range propsMap {
				req = append(req, k)
			}
			sort.Strings(req)
			schema["required"] = req
		}
	}

	if defsVal, ok := schema["$defs"]; ok {
		if defsMap, ok := defsVal.(map[string]any); ok {
			for _, defn := range defsMap {
				enforceStrictOpenAISchema(defn)
			}
		}
	}

	if hasProps {
		if propsMap, ok := propsVal.(map[string]any); ok {
			for _, prop := range propsMap {
				enforceStrictOpenAISchema(prop)
			}
		}
	}

	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if arrVal, ok := schema[key]; ok {
			if arr, ok := arrVal.([]any); ok {
				for _, item := range arr {
					enforceStrictOpenAISchema(item)
				}
			}
		}
	}

	if itemsVal, ok := schema["items"]; ok {
		if _, isMap := itemsVal.(map[string]any); isMap {
			enforceStrictOpenAISchema(itemsVal)
		}
	}
}
