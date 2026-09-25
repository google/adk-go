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

package responses

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"

	"google.golang.org/adk/v2/model/openaimodel/internal/openaicommon"
)

// buildParams converts a generic LLMRequest into the OpenAI-specific
// responses.ResponseNewParams format, preparing it for an API call.
func buildParams(modelName string, req *model.LLMRequest) (responses.ResponseNewParams, error) {
	if req == nil {
		return responses.ResponseNewParams{}, openaicommon.ErrRequestNil
	}

	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(modelName),
	}
	if req.Model != "" {
		params.Model = shared.ResponsesModel(req.Model)
	}

	// We convert the generic content parts into OpenAI's input format.
	input, droppedReasoning, err := convertContents(req.Contents)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}
	if len(input) == 0 {
		if droppedReasoning {
			// The drop is what emptied the request; don't report it as a
			// caller who sent nothing. Gated on a part actually having been
			// dropped, so a request that was empty on arrival still returns
			// the bare sentinel a caller may compare against directly.
			return responses.ResponseNewParams{}, fmt.Errorf(
				"%w: every part was dropped as replayed reasoning", openaicommon.ErrNoContents)
		}
		return responses.ResponseNewParams{}, openaicommon.ErrNoContents
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

// convertContents converts contents into Responses API input items, reporting
// separately whether any part was dropped as replayed reasoning. The caller
// needs that to tell a request the drop emptied from one that arrived empty.
func convertContents(contents []*genai.Content) (responses.ResponseInputParam, bool, error) {
	var (
		items            responses.ResponseInputParam
		tracker          openaicommon.CallTracker
		textParts        []string
		droppedReasoning bool
		curRole          genai.Role = genai.RoleUser
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
			if part == nil {
				continue
			}
			// Reported before anything is emitted, so that a field this
			// package cannot send is named even when text or a call rides on
			// the same part and would otherwise have carried it out unnoticed.
			if field := unsupportedPayload(part); field != "" {
				return nil, false, fmt.Errorf("openai: unsupported content part: %s", field)
			}
			// Text is read independently of a call or a response because one
			// part can carry both. A call and a response on the same part are
			// still alternatives, and the response is dropped, as on main.
			sendText := part.Text != "" && !part.Thought
			switch {
			case sendText:
				textParts = append(textParts, part.Text)
			case part.Text != "" || replayedReasoning(part):
				// Dropping reasoning must not hide a bad role, so the check
				// still runs. The drop counts toward the emptied-request
				// report only when it suppressed text the model would
				// otherwise have seen, blank text being skipped either way.
				if strings.TrimSpace(part.Text) != "" {
					droppedReasoning = true
				}
				if _, err := normalizeRole(curRole); err != nil {
					return nil, false, err
				}
			}
			switch {
			case part.FunctionCall != nil:
				// Flush first so buffered text keeps its place ahead of the call.
				if err := flushText(); err != nil {
					return nil, false, err
				}
				callParam, err := newFunctionCall(&tracker, part.FunctionCall)
				if err != nil {
					return nil, false, err
				}
				items = append(items, responses.ResponseInputItemUnionParam{OfFunctionCall: callParam})
			case part.FunctionResponse != nil:
				// Similarly, for a function response, we flush text before adding the response.
				if err := flushText(); err != nil {
					return nil, false, err
				}
				respParam, err := newFunctionResponse(&tracker, part.FunctionResponse)
				if err != nil {
					return nil, false, err
				}
				items = append(items, responses.ResponseInputItemUnionParam{OfFunctionCallOutput: respParam})
			case !sendText && !replayedReasoning(part):
				// Nothing in the part reaches the request. It keeps the
				// unsupported-content-part prefix the single message used
				// before, so a caller matching on that still matches here.
				return nil, false, errors.New("openai: unsupported content part: carries nothing to send")
			}
		}
		// After processing all parts in a content block, we flush any remaining text.
		if err := flushText(); err != nil {
			return nil, false, err
		}
	}

	return items, droppedReasoning, nil
}

// replayedReasoning reports whether part is reasoning carried over from an
// earlier turn that carries nothing else, so dropping it loses nothing: the
// Responses API accepts reasoning back only as an input item referencing the
// id that produced it, an id ADK does not carry, so sent as assistant text it
// would read as words the model never said.
//
// Whether to send a part's text is decided by part.Thought alone, because a
// part can carry both reasoning text and a call, and the call must survive.
func replayedReasoning(part *genai.Part) bool {
	if part == nil {
		return false
	}
	// A signature can arrive on a part of its own with the marker unset; there
	// is nowhere to put it in a Responses request either way.
	if !part.Thought && len(part.ThoughtSignature) == 0 {
		return false
	}
	// Text on a part not marked as a thought is an answer, signature or not.
	if part.Text != "" && !part.Thought {
		return false
	}
	// Marking a call or anything else as a thought must not make it vanish.
	return part.FunctionCall == nil && part.FunctionResponse == nil &&
		unsupportedPayload(part) == ""
}

// unsupportedPayload names the first field on part that this package has no way
// to send, or "" when the part holds nothing beyond what convertContents
// accounts for.
//
// The test is stated as the absence of anything unaccounted for rather than as
// a list of the fields that disqualify a part, so that a field added to
// genai.Part by a later release is reported here by default instead of leaving
// the request unnoticed.
func unsupportedPayload(part *genai.Part) string {
	if part == nil {
		return ""
	}
	rest := *part
	rest.Text = ""              // sent, or dropped when it is reasoning
	rest.Thought = false        // the marker deciding which
	rest.ThoughtSignature = nil // no Responses input item can carry one
	rest.FunctionCall = nil     // sent as a function_call item
	rest.FunctionResponse = nil // sent as a function_call_output item
	rest.VideoMetadata = nil    // qualifies media carried in another field
	rest.MediaResolution = nil  // likewise
	rest.PartMetadata = nil     // caller bookkeeping, never content

	v := reflect.ValueOf(rest)
	for i := range v.NumField() {
		if !v.Field(i).IsZero() {
			return v.Type().Field(i).Name
		}
	}
	return ""
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

// newFunctionCall converts a generic genai.FunctionCall into an OpenAI-specific
// ResponseFunctionToolCallParam. We generate a unique callID if one isn't
// provided, and then marshal the function arguments into a JSON string.
func newFunctionCall(t *openaicommon.CallTracker, fc *genai.FunctionCall) (*responses.ResponseFunctionToolCallParam, error) {
	if fc.Name == "" {
		return nil, openaicommon.ErrFunctionCallMissingName
	}
	callID := t.TakeCallID(fc)
	args, err := openaicommon.MarshalFunctionArgs(fc.Args)
	if err != nil {
		return nil, err
	}
	return &responses.ResponseFunctionToolCallParam{
		Name:      fc.Name,
		CallID:    callID,
		Arguments: args,
		Type:      constant.FunctionCall("function_call"),
	}, nil
}

// newFunctionResponse converts a generic genai.FunctionResponse into an OpenAI-specific
// ResponseInputItemFunctionCallOutputParam. We try to match the response to a pending
// function call. If an explicit callID is provided, we find and remove it from our
// pending list. Otherwise, we assume it corresponds to the oldest pending call.
func newFunctionResponse(t *openaicommon.CallTracker, fr *genai.FunctionResponse) (*responses.ResponseInputItemFunctionCallOutputParam, error) {
	callID, err := t.ResolveResponseID(fr)
	if err != nil {
		return nil, err
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
		return openaicommon.ErrTopKNotSupported
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(cfg.MaxOutputTokens))
	}
	if len(cfg.StopSequences) > 0 {
		return openaicommon.ErrStopSequencesNotSupported
	}
	if cfg.CandidateCount > 1 {
		return openaicommon.ErrMultipleCandidatesNotSupported
	}
	if cfg.FrequencyPenalty != nil || cfg.PresencePenalty != nil {
		return openaicommon.ErrPenaltiesNotSupported
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
		inst, err := openaicommon.FlattenContentText(cfg.SystemInstruction)
		if err != nil {
			return fmt.Errorf("openai: system instruction: %w", err)
		}
		if inst != "" {
			params.Instructions = param.NewOpt(inst)
		}
	}
	if cfg.ResponseMIMEType != "" && cfg.ResponseMIMEType != "text/plain" && cfg.ResponseMIMEType != "application/json" {
		return fmt.Errorf("%w: %s", openaicommon.ErrUnsupportedMIMEType, cfg.ResponseMIMEType)
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
		return openaicommon.ErrLabelsNotSupported
	}
	if cfg.SafetySettings != nil {
		return openaicommon.ErrSafetySettingsNotSupported
	}
	if err := applyThinkingConfig(params, cfg.ThinkingConfig); err != nil {
		return err
	}
	if cfg.ServiceTier != "" {
		tier, ok := serviceTiers[cfg.ServiceTier]
		if !ok {
			return fmt.Errorf("%w: ServiceTier %q", openaicommon.ErrUnsupportedConfigField, cfg.ServiceTier)
		}
		params.ServiceTier = tier
	}
	if err := openaicommon.RejectUntranslatableValues(cfg); err != nil {
		return err
	}
	// Last, so the named errors above win when a caller sets both.
	return openaicommon.RejectUnsupportedConfigFields(cfg)
}

// serviceTiers maps genai's processing tiers onto the Responses equivalents.
// "Unspecified" joins "standard" on default rather than auto, because genai
// documents it as "Default service tier, which is standard".
var serviceTiers = map[genai.ServiceTier]responses.ResponseNewParamsServiceTier{
	genai.ServiceTierUnspecified: responses.ResponseNewParamsServiceTierDefault,
	genai.ServiceTierStandard:    responses.ResponseNewParamsServiceTierDefault,
	genai.ServiceTierFlex:        responses.ResponseNewParamsServiceTierFlex,
	genai.ServiceTierPriority:    responses.ResponseNewParamsServiceTierPriority,
}

// applyThinkingConfig maps genai's thinking config onto effort-based reasoning.
//
// Summary rides on IncludeThoughts because summaries need a verified OpenAI
// organization, so requesting one unprompted would fail an unverified org's
// every reasoning call.
func applyThinkingConfig(params *responses.ResponseNewParams, cfg *genai.ThinkingConfig) error {
	if cfg == nil {
		return nil
	}
	effort, err := openaicommon.ReasoningEffortFor(cfg)
	if err != nil {
		return err
	}
	// A reasoning block holding neither is the zero value, which omitzero drops
	// from the request, so an unasked-for one costs nothing on the wire.
	params.Reasoning = shared.ReasoningParam{Effort: effort}
	// IncludeThoughts alone leaves Effort unset, letting the model pick it, and
	// asks only for the summaries that response.go surfaces as thought parts.
	if cfg.IncludeThoughts {
		params.Reasoning.Summary = shared.ReasoningSummaryAuto
	}
	return nil
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
		schema, err = openaicommon.NormalizeSchema(cfg.ResponseJsonSchema)
	case cfg.ResponseSchema != nil:
		schema, err = openaicommon.SchemaToMap(cfg.ResponseSchema)
	default:
		return nil, fmt.Errorf("openai: json schema requested without schema")
	}
	if err != nil {
		return nil, err
	}
	openaicommon.EnforceStrictOpenAISchema(schema)
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
