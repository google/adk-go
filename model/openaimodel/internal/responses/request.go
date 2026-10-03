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
	"strings"

	"github.com/openai/openai-go/v3/packages/param"
	oairesponses "github.com/openai/openai-go/v3/responses"
	oaishared "github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"

	"google.golang.org/adk/v2/model/openaimodel/internal/shared"
)

// buildParams converts a generic LLMRequest into the OpenAI-specific
// oairesponses.ResponseNewParams format, preparing it for an API call.
func buildParams(modelName string, req *model.LLMRequest) (oairesponses.ResponseNewParams, error) {
	if req == nil {
		return oairesponses.ResponseNewParams{}, shared.ErrRequestNil
	}

	params := oairesponses.ResponseNewParams{
		Model: oaishared.ResponsesModel(modelName),
	}
	if req.Model != "" {
		params.Model = oaishared.ResponsesModel(req.Model)
	}

	// We convert the generic content parts into OpenAI's input format.
	input, droppedReasoning, err := convertContents(req.Contents)
	if err != nil {
		return oairesponses.ResponseNewParams{}, err
	}
	if len(input) == 0 {
		if droppedReasoning {
			// The drop is what emptied the request; don't report it as a
			// caller who sent nothing. Gated on a part actually having been
			// dropped, so a request that was empty on arrival still returns
			// the bare sentinel a caller may compare against directly.
			return oairesponses.ResponseNewParams{}, fmt.Errorf(
				"%w: every part was dropped as replayed reasoning", shared.ErrNoContents)
		}
		return oairesponses.ResponseNewParams{}, shared.ErrNoContents
	}
	params.Input = oairesponses.ResponseNewParamsInputUnion{
		OfInputItemList: input,
	}

	// Apply generation configuration settings like temperature and max output tokens.
	if err := applyGenerationConfig(&params, req.Config); err != nil {
		return oairesponses.ResponseNewParams{}, err
	}

	// Convert any specified tools into the OpenAI tool format.
	tools, err := convertTools(req.Config)
	if err != nil {
		return oairesponses.ResponseNewParams{}, err
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	// Handle tool choice configuration, if provided.
	if cfg := req.Config; cfg != nil && cfg.ToolConfig != nil {
		choice, err := convertToolChoice(cfg.ToolConfig)
		if err != nil {
			return oairesponses.ResponseNewParams{}, err
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
func convertContents(contents []*genai.Content) (oairesponses.ResponseInputParam, bool, error) {
	var (
		items            oairesponses.ResponseInputParam
		tracker          shared.CallTracker
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
			if msgRole == oairesponses.EasyInputMessageRoleAssistant {
				if msg := newOutputMessage(textParts); msg != nil {
					items = append(items, oairesponses.ResponseInputItemUnionParam{OfOutputMessage: msg})
				}
			} else {
				if msg := newMessage(msgRole, textParts); msg != nil {
					items = append(items, oairesponses.ResponseInputItemUnionParam{OfMessage: msg})
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
			if field := shared.UnsupportedPayload(part); field != "" {
				return nil, false, fmt.Errorf("openai: unsupported content part: %s", field)
			}
			// Text is read independently of a call or a response because one
			// part can carry both. A call and a response on the same part are
			// still alternatives, and the response is dropped, as on main.
			sendText := part.Text != "" && !part.Thought
			switch {
			case sendText:
				textParts = append(textParts, part.Text)
			case part.Text != "" || shared.ReplayedReasoning(part):
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
				items = append(items, oairesponses.ResponseInputItemUnionParam{OfFunctionCall: callParam})
			case part.FunctionResponse != nil:
				// Similarly, for a function response, we flush text before adding the response.
				if err := flushText(); err != nil {
					return nil, false, err
				}
				respParam, err := newFunctionResponse(&tracker, part.FunctionResponse)
				if err != nil {
					return nil, false, err
				}
				items = append(items, oairesponses.ResponseInputItemUnionParam{OfFunctionCallOutput: respParam})
			case !sendText && !shared.ReplayedReasoning(part):
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

// newMessage builds an easy input message for an already-normalized role.
func newMessage(msgRole oairesponses.EasyInputMessageRole, texts []string) *oairesponses.EasyInputMessageParam {
	if len(texts) == 0 {
		return nil
	}
	contentList := make(oairesponses.ResponseInputMessageContentListParam, 0, len(texts))
	for _, txt := range texts {
		if strings.TrimSpace(txt) == "" {
			continue
		}
		textParam := oairesponses.ResponseInputTextParam{
			Text: txt,
			Type: constant.InputText("input_text"),
		}
		contentList = append(contentList, oairesponses.ResponseInputContentUnionParam{
			OfInputText: &textParam,
		})
	}
	if len(contentList) == 0 {
		return nil
	}
	return &oairesponses.EasyInputMessageParam{
		Role: msgRole,
		Type: oairesponses.EasyInputMessageTypeMessage,
		Content: oairesponses.EasyInputMessageContentUnionParam{
			OfInputItemContentList: contentList,
		},
	}
}

// newOutputMessage builds an assistant output message whose content uses the
// "output_text" type, as required when replaying a prior assistant turn to the
// OpenAI Responses API.
func newOutputMessage(texts []string) *oairesponses.ResponseOutputMessageParam {
	if len(texts) == 0 {
		return nil
	}
	contentList := make([]oairesponses.ResponseOutputMessageContentUnionParam, 0, len(texts))
	for _, txt := range texts {
		if strings.TrimSpace(txt) == "" {
			continue
		}
		contentList = append(contentList, oairesponses.ResponseOutputMessageContentUnionParam{
			OfOutputText: &oairesponses.ResponseOutputTextParam{
				Text: txt,
				Type: constant.OutputText("output_text"),
			},
		})
	}
	if len(contentList) == 0 {
		return nil
	}
	return &oairesponses.ResponseOutputMessageParam{
		Content: contentList,
		Status:  oairesponses.ResponseOutputMessageStatusCompleted,
	}
}

func normalizeRole(role genai.Role) (oairesponses.EasyInputMessageRole, error) {
	switch role {
	case "", genai.RoleUser:
		return oairesponses.EasyInputMessageRoleUser, nil
	case genai.RoleModel:
		return oairesponses.EasyInputMessageRoleAssistant, nil
	case "system":
		return oairesponses.EasyInputMessageRoleSystem, nil
	case "developer":
		return oairesponses.EasyInputMessageRoleDeveloper, nil
	default:
		return "", fmt.Errorf("openai: unsupported role %q", role)
	}
}

// newFunctionCall converts a generic genai.FunctionCall into an OpenAI-specific
// ResponseFunctionToolCallParam. We generate a unique callID if one isn't
// provided, and then marshal the function arguments into a JSON string.
func newFunctionCall(t *shared.CallTracker, fc *genai.FunctionCall) (*oairesponses.ResponseFunctionToolCallParam, error) {
	if fc.Name == "" {
		return nil, shared.ErrFunctionCallMissingName
	}
	callID := t.TakeCallID(fc)
	args, err := shared.MarshalFunctionArgs(fc.Args)
	if err != nil {
		return nil, err
	}
	return &oairesponses.ResponseFunctionToolCallParam{
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
func newFunctionResponse(t *shared.CallTracker, fr *genai.FunctionResponse) (*oairesponses.ResponseInputItemFunctionCallOutputParam, error) {
	callID, err := t.ResolveResponseID(fr)
	if err != nil {
		return nil, err
	}
	payload, err := json.Marshal(fr.Response)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal function response: %w", err)
	}
	return &oairesponses.ResponseInputItemFunctionCallOutputParam{
		CallID: param.NewOpt(callID),
		Output: oairesponses.ResponseInputItemFunctionCallOutputOutputUnionParam{
			OfString: param.NewOpt(string(payload)),
		},
		Type: constant.FunctionCallOutput("function_call_output"),
	}, nil
}

// applyGenerationConfig translates our generic generation configuration into
// OpenAI-specific parameters. We also validate and return errors for features
// that are not supported by the OpenAI Responses API.
func applyGenerationConfig(params *oairesponses.ResponseNewParams, cfg *genai.GenerateContentConfig) error {
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
		return shared.ErrTopKNotSupported
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(cfg.MaxOutputTokens))
	}
	if len(cfg.StopSequences) > 0 {
		return shared.ErrStopSequencesNotSupported
	}
	if cfg.CandidateCount > 1 {
		return shared.ErrMultipleCandidatesNotSupported
	}
	if cfg.FrequencyPenalty != nil || cfg.PresencePenalty != nil {
		return shared.ErrPenaltiesNotSupported
	}
	if cfg.ResponseLogprobs {
		if cfg.Logprobs != nil {
			params.TopLogprobs = param.NewOpt(int64(*cfg.Logprobs))
		} else {
			params.TopLogprobs = param.NewOpt(int64(1))
		}
		// Responses returns logprobs only when explicitly included.
		params.Include = append(params.Include, oairesponses.ResponseIncludableMessageOutputTextLogprobs)
	}
	if cfg.SystemInstruction != nil {
		inst, err := shared.FlattenContentText(cfg.SystemInstruction)
		if err != nil {
			return fmt.Errorf("openai: system instruction: %w", err)
		}
		if inst != "" {
			params.Instructions = param.NewOpt(inst)
		}
	}
	if cfg.ResponseMIMEType != "" && cfg.ResponseMIMEType != "text/plain" && cfg.ResponseMIMEType != "application/json" {
		return fmt.Errorf("%w: %s", shared.ErrUnsupportedMIMEType, cfg.ResponseMIMEType)
	}
	if cfg.ResponseMIMEType == "application/json" || cfg.ResponseSchema != nil || cfg.ResponseJsonSchema != nil {
		if cfg.ResponseSchema == nil && cfg.ResponseJsonSchema == nil {
			obj := oaishared.NewResponseFormatJSONObjectParam()
			params.Text = oairesponses.ResponseTextConfigParam{
				Format: oairesponses.ResponseFormatTextConfigUnionParam{
					OfJSONObject: &obj,
				},
			}
		} else {
			format, err := newJSONSchemaFormat(cfg)
			if err != nil {
				return err
			}
			params.Text = oairesponses.ResponseTextConfigParam{
				Format: oairesponses.ResponseFormatTextConfigUnionParam{
					OfJSONSchema: format,
				},
			}
		}
	}
	if cfg.Labels != nil {
		return shared.ErrLabelsNotSupported
	}
	if cfg.SafetySettings != nil {
		return shared.ErrSafetySettingsNotSupported
	}
	if err := applyThinkingConfig(params, cfg.ThinkingConfig); err != nil {
		return err
	}
	if cfg.ServiceTier != "" {
		tier, ok := serviceTiers[cfg.ServiceTier]
		if !ok {
			return fmt.Errorf("%w: ServiceTier %q", shared.ErrUnsupportedConfigField, cfg.ServiceTier)
		}
		params.ServiceTier = tier
	}
	if err := shared.RejectUntranslatableValues(cfg); err != nil {
		return err
	}
	// Last, so the named errors above win when a caller sets both.
	return shared.RejectUnsupportedConfigFields(cfg)
}

// serviceTiers maps genai's processing tiers onto the Responses equivalents.
// "Unspecified" joins "standard" on default rather than auto, because genai
// documents it as "Default service tier, which is standard".
var serviceTiers = map[genai.ServiceTier]oairesponses.ResponseNewParamsServiceTier{
	genai.ServiceTierUnspecified: oairesponses.ResponseNewParamsServiceTierDefault,
	genai.ServiceTierStandard:    oairesponses.ResponseNewParamsServiceTierDefault,
	genai.ServiceTierFlex:        oairesponses.ResponseNewParamsServiceTierFlex,
	genai.ServiceTierPriority:    oairesponses.ResponseNewParamsServiceTierPriority,
}

// applyThinkingConfig maps genai's thinking config onto effort-based reasoning.
//
// Summary rides on IncludeThoughts because summaries need a verified OpenAI
// organization, so requesting one unprompted would fail an unverified org's
// every reasoning call.
func applyThinkingConfig(params *oairesponses.ResponseNewParams, cfg *genai.ThinkingConfig) error {
	if cfg == nil {
		return nil
	}
	effort, err := shared.ReasoningEffortFor(cfg)
	if err != nil {
		return err
	}
	// A reasoning block holding neither is the zero value, which omitzero drops
	// from the request, so an unasked-for one costs nothing on the wire.
	params.Reasoning = oaishared.ReasoningParam{Effort: effort}
	// IncludeThoughts alone leaves Effort unset, letting the model pick it, and
	// asks only for the summaries that response.go surfaces as thought parts.
	if cfg.IncludeThoughts {
		params.Reasoning.Summary = oaishared.ReasoningSummaryAuto
	}
	return nil
}

// newJSONSchemaFormat constructs an OpenAI-specific JSON schema format from our
// generic GenerateContentConfig. We handle cases where the schema is provided
// directly or needs to be converted, and assign a name to it.
func newJSONSchemaFormat(cfg *genai.GenerateContentConfig) (*oairesponses.ResponseFormatTextJSONSchemaConfigParam, error) {
	var (
		schema map[string]any
		err    error
	)
	switch {
	case cfg.ResponseJsonSchema != nil:
		schema, err = shared.NormalizeSchema(cfg.ResponseJsonSchema)
	case cfg.ResponseSchema != nil:
		schema, err = shared.SchemaToMap(cfg.ResponseSchema)
	default:
		return nil, fmt.Errorf("openai: json schema requested without schema")
	}
	if err != nil {
		return nil, err
	}
	shared.EnforceStrictOpenAISchema(schema)
	name := "adk_response"
	if cfg.ResponseSchema != nil && cfg.ResponseSchema.Title != "" {
		name = cfg.ResponseSchema.Title
	}
	return &oairesponses.ResponseFormatTextJSONSchemaConfigParam{
		Name:   name,
		Schema: schema,
		Strict: param.NewOpt(true),
		Type:   constant.JSONSchema("json_schema"),
	}, nil
}
