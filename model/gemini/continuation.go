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

package gemini

import (
	"bytes"
	"cmp"
	"encoding/json"
	"log"
	"reflect"
	"slices"

	"google.golang.org/genai"
)

// Decoding continuation. A model with decoding continuation pauses a
// generation that reaches its per-request output limit with finish reason
// CONTINUATION and a continuation token on the candidate. The model resends
// the request with the output so far and the token until the generation
// finishes, and returns one response with the joined content and the summed
// usage of every request, as ADK Java and ADK Kotlin do. A response stopped by
// maxOutputTokens is not resumed, since the API applies maxOutputTokens to the
// whole generation.

// maxResumes bounds the requests that resume one generation. A model that
// pauses every 300 decoding steps takes about 150 of them for a 45K-token
// answer.
const maxResumes = 256

// continuationRetry is how a request that resumes a generation retries a
// transient failure (408, 429, 5xx or a transport error): with backoff from
// one second to a minute, 8 attempts in all. Such a request carries the output
// so far, which failing it outright would throw away. It is not set when the
// request or the client already sets retry options, and the first request of a
// generation is not retried, as before. ADK Java and ADK Kotlin do not retry a
// resend.
var continuationRetry = &genai.HTTPRetryOptions{Attempts: genai.Ptr[int32](8)}

// continuationToken returns the token that resumes resp, or nil if resp did
// not pause or carries no token.
func continuationToken(resp *genai.GenerateContentResponse) []byte {
	if len(resp.Candidates) == 0 || resp.Candidates[0] == nil || resp.Candidates[0].FinishReason != genai.FinishReasonContinuation {
		return nil
	}
	token := resp.Candidates[0].ContinuationToken
	if len(token) == 0 {
		log.Printf("adk: the model paused a generation for continuation without a continuation token; returning the partial output") //nolint:forbidigo // pre-slog call site
		return nil
	}
	return token
}

// continuation carries one generation across the requests that resume it.
type continuation struct {
	contents []*genai.Content
	config   *genai.GenerateContentConfig
	// retry is whether a resend sets continuationRetry.
	retry bool

	parts   []*genai.Part
	usage   *genai.GenerateContentResponseUsageMetadata
	token   []byte
	resumes int
	// grounding and citation are the last a request reported.
	grounding *genai.GroundingMetadata
	citation  *genai.CitationMetadata
}

func newContinuation(contents []*genai.Content, config *genai.GenerateContentConfig, retry bool) *continuation {
	return &continuation{contents: contents, config: config, retry: retry}
}

// resumed reports whether the generation took more than one request.
func (c *continuation) resumed() bool {
	return c.token != nil
}

// willResume reports whether a request that ended with token is resumed: not
// when the model returned the same token twice, which means it made no
// progress, nor after maxResumes resends.
func (c *continuation) willResume(token []byte) bool {
	return !bytes.Equal(token, c.token) && c.resumes < maxResumes
}

// advance records one request's output, its token if it paused, the parts it
// generated and its usage, and returns the contents and config of the request
// that resumes it. ok is false once the generation is finished, or paused in a
// way that is not resumed.
func (c *continuation) advance(token []byte, parts []*genai.Part, usage *genai.GenerateContentResponseUsageMetadata) (contents []*genai.Content, config *genai.GenerateContentConfig, ok bool) {
	c.usage = addUsage(c.usage, usage)
	if token == nil && !c.resumed() {
		return nil, nil, false
	}
	c.parts = appendParts(c.parts, parts)
	if token == nil {
		return nil, nil, false
	}
	if !c.willResume(token) {
		if bytes.Equal(token, c.token) {
			log.Printf("adk: the model returned the same continuation token twice; returning the output generated so far") //nolint:forbidigo // pre-slog call site
		} else {
			log.Printf("adk: the model paused a generation %d times; returning the output generated so far", maxResumes) //nolint:forbidigo // pre-slog call site
		}
		return nil, nil, false
	}
	c.token = token
	c.resumes++
	contents, config = c.nextRequest(token)
	return contents, config, true
}

// content returns everything generated across the requests recorded so far.
func (c *continuation) content() *genai.Content {
	return &genai.Content{Role: genai.RoleModel, Parts: slices.Clone(c.parts)}
}

// keepMetadata records the grounding and citation metadata of a request's
// candidate, so that the final response keeps the last reported even when its
// own request reports none, as the streaming aggregator does.
func (c *continuation) keepMetadata(candidate *genai.Candidate) {
	c.grounding = cmp.Or(candidate.GroundingMetadata, c.grounding)
	c.citation = cmp.Or(candidate.CitationMetadata, c.citation)
}

// complete returns resp, the last request's response, with the content, usage,
// grounding and citation metadata of every request if the generation was
// resumed, and resp itself otherwise. resp keeps the last request's finish
// reason. Converting the result gives the error code a single response with the
// same content would get: none, once any request produced content. ADK Java
// keeps the last request's error code.
func (c *continuation) complete(resp *genai.GenerateContentResponse) *genai.GenerateContentResponse {
	if !c.resumed() || len(resp.Candidates) == 0 || resp.Candidates[0] == nil {
		return resp
	}
	out := *resp
	candidate := *resp.Candidates[0]
	if len(c.parts) > 0 {
		candidate.Content = c.content()
	}
	candidate.GroundingMetadata = c.grounding
	candidate.CitationMetadata = c.citation
	out.Candidates = append([]*genai.Candidate{&candidate}, resp.Candidates[1:]...)
	out.UsageMetadata = c.usage
	return &out
}

func (c *continuation) nextRequest(token []byte) ([]*genai.Content, *genai.GenerateContentConfig) {
	contents := slices.Clone(c.contents)
	if len(c.parts) > 0 {
		contents = append(contents, c.content())
	}
	config := &genai.GenerateContentConfig{}
	if c.config != nil {
		*config = *c.config
	}
	config.ContinuationToken = token
	if c.retry {
		opts := &genai.HTTPOptions{}
		if config.HTTPOptions != nil {
			*opts = *config.HTTPOptions
		}
		opts.RetryOptions = continuationRetry
		config.HTTPOptions = opts
	}
	return contents, config
}

// appendParts appends copies of next to parts, joining text: consecutive
// non-empty text of the same kind becomes one part, and an empty part with
// only a thought signature, which is how a paused stream closes its text,
// joins onto the text before it. The later signature wins. A resend and the
// final response so carry one part per run of text rather than gaining two
// parts per pause. ADK Java and ADK Kotlin keep the signature part separate and
// the first signature instead. The copies share no memory with next, so a
// resend carries the parts as the model sent them even after a consumer edits
// the ones ADK yielded, a function call's arguments included.
//
// A part with no text and nothing else but the thought flag, such as the one a
// stream can end with, is dropped. genai omits empty text, so the part would
// encode as {} or {"thought": true}, which the API rejects with a 400. A
// signature-only part that follows no text is kept: the API accepts it.
func appendParts(parts, next []*genai.Part) []*genai.Part {
	for _, p := range next {
		if p == nil || isEmpty(p) {
			continue
		}
		if n := len(parts); n > 0 && canJoin(parts[n-1], p) {
			joined := *parts[n-1]
			joined.Text += p.Text
			if len(p.ThoughtSignature) > 0 {
				joined.ThoughtSignature = slices.Clone(p.ThoughtSignature)
			}
			parts[n-1] = &joined
			continue
		}
		parts = append(parts, copyPart(p))
	}
	return parts
}

// canJoin reports whether second continues the text first started.
func canJoin(first, second *genai.Part) bool {
	if first.Text == "" || !onlyText(first) || !onlyText(second) {
		return false
	}
	if second.Text == "" {
		return len(second.ThoughtSignature) > 0
	}
	return first.Thought == second.Thought
}

// copyPart returns a copy of p that shares no memory with it. p was decoded
// from a response, so encoding it again keeps every value, except that an
// empty map or slice, such as the arguments of a call without any, comes back
// nil. The copy covers every field genai.Part has or gains. A part that fails
// to encode, which decoded JSON cannot produce, is copied one level deep.
func copyPart(p *genai.Part) *genai.Part {
	if data, err := json.Marshal(p); err == nil {
		var cp genai.Part
		if json.Unmarshal(data, &cp) == nil {
			return &cp
		}
	}
	cp := *p
	return &cp
}

// onlyText reports whether p has nothing set but text, the thought flag and a
// thought signature.
func onlyText(p *genai.Part) bool {
	rest := *p
	rest.Text, rest.Thought, rest.ThoughtSignature = "", false, nil
	return reflect.ValueOf(rest).IsZero()
}

// isEmpty reports whether p has nothing set but the thought flag, counting an
// empty signature as none: it encodes as none.
func isEmpty(p *genai.Part) bool {
	rest := *p
	rest.Thought = false
	if len(rest.ThoughtSignature) == 0 {
		rest.ThoughtSignature = nil
	}
	return reflect.ValueOf(rest).IsZero()
}

// addUsage returns the combined token usage of two requests: counts are
// summed, per-modality counts summed by modality, and anything else is next's.
// Every request is billed, so the result is what the generation used, as ADK
// Java and ADK Kotlin report it. PromptTokenCount so counts the prompt of every
// resend, each carrying the output so far, and exceeds the size of any one
// prompt; compaction, which reads it as the prompt size, can compact a session
// early after a resumed generation.
func addUsage(total, next *genai.GenerateContentResponseUsageMetadata) *genai.GenerateContentResponseUsageMetadata {
	if total == nil {
		return next
	}
	if next == nil {
		return total
	}
	out := *next
	out.CacheTokensDetails = addModalityCounts(total.CacheTokensDetails, next.CacheTokensDetails)
	out.CachedContentTokenCount = total.CachedContentTokenCount + next.CachedContentTokenCount
	out.CandidatesTokenCount = total.CandidatesTokenCount + next.CandidatesTokenCount
	out.CandidatesTokensDetails = addModalityCounts(total.CandidatesTokensDetails, next.CandidatesTokensDetails)
	out.PromptTokenCount = total.PromptTokenCount + next.PromptTokenCount
	out.PromptTokensDetails = addModalityCounts(total.PromptTokensDetails, next.PromptTokensDetails)
	out.ThoughtsTokenCount = total.ThoughtsTokenCount + next.ThoughtsTokenCount
	out.ToolUsePromptTokenCount = total.ToolUsePromptTokenCount + next.ToolUsePromptTokenCount
	out.ToolUsePromptTokensDetails = addModalityCounts(total.ToolUsePromptTokensDetails, next.ToolUsePromptTokensDetails)
	out.TotalTokenCount = total.TotalTokenCount + next.TotalTokenCount
	return &out
}

// addModalityCounts sums token counts that share a modality, keeping
// first-seen order.
func addModalityCounts(a, b []*genai.ModalityTokenCount) []*genai.ModalityTokenCount {
	if a == nil && b == nil {
		return nil
	}
	var order []genai.MediaModality
	totals := map[genai.MediaModality]int32{}
	for _, c := range slices.Concat(a, b) {
		if c == nil {
			continue
		}
		if _, ok := totals[c.Modality]; !ok {
			order = append(order, c.Modality)
		}
		totals[c.Modality] += c.TokenCount
	}
	out := make([]*genai.ModalityTokenCount, 0, len(order))
	for _, m := range order {
		out = append(out, &genai.ModalityTokenCount{Modality: m, TokenCount: totals[m]})
	}
	return out
}
