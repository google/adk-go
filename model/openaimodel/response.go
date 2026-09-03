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
	if reportsFailure(resp) {
		return nil, failedResponseError(resp)
	}
	candidate, err := buildCandidate(resp)
	if err != nil {
		return nil, err
	}
	return &genai.GenerateContentResponse{
		Candidates:     []*genai.Candidate{candidate},
		ModelVersion:   string(resp.Model),
		ResponseID:     resp.ID,
		UsageMetadata:  convertUsage(resp.Usage),
		PromptFeedback: promptFeedback(resp),
	}, nil
}

// reportsFailure reports whether the server is describing a failure rather than
// a turn, which on an HTTP 200 is the whole of what separates the two. Only
// "failed" qualifies — "incomplete" is an ordinary truncation — except that a
// body stating no status at all, which the API permits, is judged by its error
// object instead.
func reportsFailure(resp *responses.Response) bool {
	switch resp.Status {
	case responses.ResponseStatusFailed:
		return true
	case "":
		return resp.Error.Message != "" || resp.Error.Code != ""
	default:
		return false
	}
}

// failedResponseError renders a failure as an error quoting the server: the
// message as the text, the response ID and error code as a labelled
// parenthetical. The ID is there because the response is discarded with the
// failure, so nothing else is left to quote back to the provider.
//
// The ID and code are labelled and quoted because they are server-chosen: bare,
// an ID containing ", code " renders exactly as an ID and a code would, and
// quoting is what makes the field boundary the server's to state rather than to
// forge. Neither is elided when the message appears to repeat it — saying a code
// twice is cosmetic, dropping the one the caller needs is not.
func failedResponseError(resp *responses.Response) error {
	// Trimmed, so a value of nothing but spaces contributes nothing rather than
	// a label or separator with nothing after it.
	msg := strings.TrimSpace(resp.Error.Message)
	var details []string
	if id := strings.TrimSpace(resp.ID); id != "" {
		details = append(details, fmt.Sprintf("id %q", id))
	}
	if code := strings.TrimSpace(string(resp.Error.Code)); code != "" {
		details = append(details, fmt.Sprintf("code %q", code))
	}
	switch joined := strings.Join(details, ", "); {
	case joined != "" && msg != "":
		return fmt.Errorf("%w (%s): %s", ErrResponseFailed, joined, msg)
	case joined != "":
		return fmt.Errorf("%w (%s)", ErrResponseFailed, joined)
	case msg != "":
		return fmt.Errorf("%w: %s", ErrResponseFailed, msg)
	default:
		// The server said "failed" and nothing more.
		return ErrResponseFailed
	}
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
		FinishReason:   finishReason(resp),
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

func finishReason(resp *responses.Response) genai.FinishReason {
	if resp == nil {
		return genai.FinishReasonUnspecified
	}
	switch resp.IncompleteDetails.Reason {
	case "max_output_tokens":
		return genai.FinishReasonMaxTokens
	case "content_filter":
		return genai.FinishReasonSafety
	case "":
		// No reason given, so the status is all that is left to go on.
	default:
		return genai.FinishReasonOther
	}
	switch resp.Status {
	case responses.ResponseStatusCompleted, "":
		// Absent status: a hand-built response, or a provider that omits the
		// field. Reading it as completed keeps the reason-only path intact.
		return genai.FinishReasonStop
	default:
		// "cancelled", "queued", "in_progress", a bare "incomplete", or a
		// status a later SDK adds. None is a clean stop. "failed" does not reach
		// here: both callers fail the turn before asking why it ended.
		return genai.FinishReasonOther
	}
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

func promptFeedback(resp *responses.Response) *genai.GenerateContentResponsePromptFeedback {
	if resp == nil || resp.IncompleteDetails.Reason != "content_filter" {
		return nil
	}
	return &genai.GenerateContentResponsePromptFeedback{
		BlockReason:        genai.BlockedReasonSafety,
		BlockReasonMessage: resp.IncompleteDetails.Reason,
	}
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
