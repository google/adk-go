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
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

// convertResponse takes an OpenAI API response and transforms it into our
// generic genai.GenerateContentResponse format.
func convertResponse(resp *responses.Response) (*genai.GenerateContentResponse, error) {
	if resp == nil {
		return nil, ErrEmptyResponse
	}
	candidate, err := buildCandidate(resp)
	if err != nil {
		return nil, err
	}
	return &genai.GenerateContentResponse{
		Candidates:    []*genai.Candidate{candidate},
		ModelVersion:  string(resp.Model),
		ResponseID:    resp.ID,
		UsageMetadata: convertUsage(resp.Usage),
	}, nil
}

func buildCandidate(resp *responses.Response) (*genai.Candidate, error) {
	parts, err := convertOutputItems(resp.Output)
	if err != nil {
		return nil, err
	}
	return &genai.Candidate{
		Content: &genai.Content{
			Role:  string(genai.RoleModel),
			Parts: parts,
		},
		// No event announced this response, so its own payload is all there is
		// to read the finish reason from.
		FinishReason:   finishReason(resp, false),
		LogprobsResult: convertLogprobs(resp.Output),
	}, nil
}

// convertOutputItems processes a slice of OpenAI ResponseOutputItemUnion and
// converts them into a slice of our generic genai.Part. We handle different
// types of output items, such as messages (text, refusal), function calls,
// and reasoning (thoughts and summaries), extracting the relevant information
// for each.
func convertOutputItems(items []responses.ResponseOutputItemUnion) ([]*genai.Part, error) {
	if len(items) == 0 {
		return nil, ErrNoOutputItems
	}
	var parts []*genai.Part
	for _, item := range items {
		switch item.Type {
		case "message":
			for _, content := range item.Content {
				switch content.Type {
				case "output_text":
					if content.Text != "" {
						parts = append(parts, &genai.Part{Text: content.Text})
					}
				case "refusal":
					parts = append(parts, &genai.Part{Text: content.Refusal})
				default:
					return nil, fmt.Errorf("%w: %q", ErrUnsupportedMessageContentType, content.Type)
				}
			}
		case "function_call":
			part, err := convertFunctionCall(item)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case "reasoning":
			for _, chunk := range item.Content {
				if chunk.Text != "" {
					parts = append(parts, &genai.Part{Text: chunk.Text, Thought: true})
				}
				// We also check for summary content within reasoning items.
			}
			for _, summary := range item.Summary {
				if summary.Text != "" {
					parts = append(parts, &genai.Part{Text: summary.Text, Thought: true})
				}
			}
		default:
			return nil, fmt.Errorf("%w: %q", ErrUnsupportedOutputItemType, item.Type)
		}
	}
	if len(parts) == 0 {
		return nil, ErrNoTextOrToolContent
	}
	return parts, nil
}

func convertFunctionCall(item responses.ResponseOutputItemUnion) (*genai.Part, error) {
	args, err := functionCallArgs(item.Arguments)
	if err != nil {
		// Name the offending call: convertOutputItems aborts the entire
		// response on the first error, so nothing else identifies it.
		return nil, fmt.Errorf("%w (name %q, call_id %q)", err, item.Name, item.CallID)
	}
	return &genai.Part{
		FunctionCall: &genai.FunctionCall{
			Name: item.Name,
			ID:   item.CallID,
			Args: args,
		},
	}, nil
}

// functionCallArgs decodes the arguments of a function call output item.
func functionCallArgs(arguments responses.ResponseOutputItemUnionArguments) (map[string]any, error) {
	var raw string
	switch v := arguments.OfResponseToolSearchCallArguments.(type) {
	case nil:
		// Absent, null, or a value built by hand rather than decoded.
		raw = arguments.OfString
	case string:
		// A JSON string; OfString holds the same value already unquoted.
		raw = arguments.OfString
	default:
		raw = arguments.JSON.OfResponseToolSearchCallArguments.Raw()
		if raw == "" {
			// No wire bytes to recover, so this value was built by hand; there
			// is no original payload for a re-encode to disagree with.
			b, err := json.Marshal(v)
			if err != nil {
				return nil, fmt.Errorf("%w: %w", ErrFunctionCallArgs, err)
			}
			raw = string(b)
		}
	}
	if raw == "" {
		return map[string]any{}, nil
	}
	args := map[string]any{}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrFunctionCallArgs, err)
	}
	if args == nil {
		// The payload was JSON null: the call takes no arguments.
		return map[string]any{}, nil
	}
	return args, nil
}

// finishReason reports why the model stopped generating. incompleteEvent says
// the terminal streaming event was a "response.incomplete" (see
// generateStream); blocking, having no event to read, passes false. Both paths
// otherwise decide from the same payload.
func finishReason(resp *responses.Response, incompleteEvent bool) genai.FinishReason {
	if resp == nil {
		return genai.FinishReasonUnspecified
	}
	switch resp.IncompleteDetails.Reason {
	case "max_output_tokens":
		return genai.FinishReasonMaxTokens
	case "content_filter":
		return genai.FinishReasonSafety
	case "":
		if truncated(resp, incompleteEvent) {
			return genai.FinishReasonOther
		}
		return genai.FinishReasonStop
	default:
		return genai.FinishReasonOther
	}
}

// truncated reports whether a turn that named no incomplete reason nonetheless
// ended before it was done; calling one a clean stop would have a caller that
// retries on anything but STOP accept a partial answer as final.
//
// openai-go marks incomplete_details required but neither its reason nor the
// status, so a provider may declare a turn truncated and leave either empty.
// Every signal that survives that is read here.
func truncated(resp *responses.Response, incompleteEvent bool) bool {
	if incompleteEvent {
		// The event stands in for "response.completed", so its name is the
		// provider's verdict, and it outranks a payload that says otherwise.
		return true
	}
	switch resp.Status {
	case responses.ResponseStatusCompleted:
		return false
	case "":
		// A finished turn carries incomplete_details as null.
		return resp.JSON.IncompleteDetails.Valid()
	default:
		// failed, cancelled, incomplete, in_progress and queued all describe an
		// unfinished turn. Enumerating the finished states instead keeps a
		// status added later from defaulting to a clean stop.
		return true
	}
}

// finishMessage is the provider's own account of why a turn ended, which the
// finish reason flattens away: an unmapped incomplete reason and a failure both
// arrive as OTHER.
func finishMessage(resp *responses.Response, incompleteEvent bool) string {
	if resp == nil {
		return ""
	}
	if msg := resp.Error.Message; msg != "" {
		return msg
	}
	if reason := resp.IncompleteDetails.Reason; reason != "" {
		return reason
	}
	// The contradiction truncated() resolves, resolved the same way: the event's
	// name outranks a payload calling the turn completed. Every other status is
	// still the provider's own wording for why.
	if resp.Status != "" && (!incompleteEvent || resp.Status != responses.ResponseStatusCompleted) {
		return string(resp.Status)
	}
	if truncated(resp, incompleteEvent) {
		// No usable reason or status, yet the turn did not finish: the event's
		// name, or a bare incomplete_details, is all the provider said.
		return string(responses.ResponseStatusIncomplete)
	}
	return ""
}

func safeInt32(v int64) int32 {
	if v > math.MaxInt32 {
		return math.MaxInt32
	}
	return int32(v)
}

func convertUsage(usage responses.ResponseUsage) *genai.GenerateContentResponseUsageMetadata {
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        safeInt32(usage.InputTokens),
		CandidatesTokenCount:    safeInt32(usage.OutputTokens),
		TotalTokenCount:         safeInt32(usage.TotalTokens),
		CachedContentTokenCount: safeInt32(usage.InputTokensDetails.CachedTokens),
		PromptTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityText, TokenCount: safeInt32(usage.InputTokens)},
		},
		CandidatesTokensDetails: []*genai.ModalityTokenCount{
			{Modality: genai.MediaModalityText, TokenCount: safeInt32(usage.OutputTokens)},
		},
		ThoughtsTokenCount: safeInt32(usage.OutputTokensDetails.ReasoningTokens),
	}
}

// logprobsFor returns a response's logprobs only when they describe the text
// given, which is what the caller reports them alongside. A streamed turn takes
// its content from the deltas and its logprobs from the terminal event, and the
// two disagree on a provider that resends different output; probabilities read
// against text they do not belong to are worse than none.
func logprobsFor(resp *responses.Response, text string) *genai.LogprobsResult {
	if resp == nil || outputText(resp.Output) != text {
		return nil
	}
	return convertLogprobs(resp.Output)
}

// outputText concatenates the text of a response's message items, the only
// items logprobs are attached to. It must agree with convertOutputItems on what
// a message's text is, refusals included, or the two paths would disagree over
// whether the logprobs describe the same answer.
func outputText(items []responses.ResponseOutputItemUnion) string {
	var text strings.Builder
	for _, item := range items {
		if item.Type != "message" {
			continue
		}
		for _, content := range item.Content {
			switch content.Type {
			case "output_text":
				text.WriteString(content.Text)
			case "refusal":
				text.WriteString(content.Refusal)
			}
		}
	}
	return text.String()
}

func convertLogprobs(items []responses.ResponseOutputItemUnion) *genai.LogprobsResult {
	if len(items) == 0 {
		return nil
	}
	var res *genai.LogprobsResult
	for _, item := range items {
		if item.Type == "message" {
			for _, content := range item.Content {
				if content.Type == "output_text" && len(content.Logprobs) > 0 {
					if res == nil {
						res = &genai.LogprobsResult{}
					}
					for _, lp := range content.Logprobs {
						res.ChosenCandidates = append(res.ChosenCandidates, &genai.LogprobsResultCandidate{
							Token:          lp.Token,
							LogProbability: float32(lp.Logprob),
						})
						var topCands []*genai.LogprobsResultCandidate
						for _, tlp := range lp.TopLogprobs {
							topCands = append(topCands, &genai.LogprobsResultCandidate{
								Token:          tlp.Token,
								LogProbability: float32(tlp.Logprob),
							})
						}
						res.TopCandidates = append(res.TopCandidates, &genai.LogprobsResultTopCandidates{
							Candidates: topCands,
						})
					}
				}
			}
		}
	}
	return res
}
