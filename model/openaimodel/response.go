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
		Candidates:    []*genai.Candidate{candidate},
		ModelVersion:  string(resp.Model),
		ResponseID:    resp.ID,
		UsageMetadata: convertUsage(resp.Usage),
	}, nil
}

// maxServerTextRunes bounds any server-chosen string an error quotes.
const maxServerTextRunes = 256

// clipServerText trims a server-chosen string and caps its length, so that a
// pathological one — a megabyte of message — cannot become the error a caller
// logs. Truncation is marked, so a clipped value does not read as the whole of
// what the server said.
//
// Callers render the result with %q. That, rather than an escaper of our own,
// is what stops a control character in it from forging a line in an operator's
// log; see the same reasoning at [google.golang.org/adk/v2/auth/gcp].
func clipServerText(s string) string {
	s = strings.TrimSpace(s)
	// Counted by ranging rather than by materialising []rune: an 8 MiB message
	// would otherwise cost 32 MiB to yield at most a kilobyte. Ranging a string
	// yields the byte index of each rune, so s[:i] never splits one.
	n := 0
	for i := range s {
		if n == maxServerTextRunes {
			return strings.TrimSpace(s[:i]) + "…"
		}
		n++
	}
	return s
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
		// Clipped, not raw: the renderer reports what clipping leaves, so a
		// predicate reading the raw value would send a body whose error is
		// nothing but whitespace down the failure path and then name nothing.
		return clipServerText(resp.Error.Message) != "" || clipServerText(string(resp.Error.Code)) != ""
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
	// Clipped, so a value of nothing but spaces contributes nothing rather than
	// a label or separator with nothing after it.
	msg := clipServerText(resp.Error.Message)
	var details []string
	if id := clipServerText(resp.ID); id != "" {
		details = append(details, fmt.Sprintf("id %q", id))
	}
	if code := clipServerText(string(resp.Error.Code)); code != "" {
		details = append(details, fmt.Sprintf("code %q", code))
	}
	switch joined := strings.Join(details, ", "); {
	case joined != "" && msg != "":
		return fmt.Errorf("%w (%s): %q", ErrResponseFailed, joined, msg)
	case joined != "":
		return fmt.Errorf("%w (%s)", ErrResponseFailed, joined)
	case msg != "":
		return fmt.Errorf("%w: %q", ErrResponseFailed, msg)
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
//
// A failed response never reaches here: both callers fail the turn, by way of
// reportsFailure, before asking why it ended.
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
		// No reason given, so the status and what the event was called are all
		// that is left to go on.
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
