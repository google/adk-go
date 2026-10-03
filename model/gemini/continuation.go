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
	"log"
	"reflect"
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// Decoding continuation. A model with decoding continuation pauses a
// generation that reaches its per-request output limit with finish reason
// CONTINUATION and a continuation token on the candidate. The model resends
// the request with the output so far and the token until the generation
// finishes, and returns one response with the joined content and the summed
// usage, as adk-python, ADK Java and ADK Kotlin do. A response stopped by
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
		log.Printf("adk: the model paused a generation for continuation without a continuation token; returning the partial output")
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
			log.Printf("adk: the model returned the same continuation token twice; returning the output generated so far")
		} else {
			log.Printf("adk: the model paused a generation %d times; returning the output generated so far", maxResumes)
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

// complete sets the content and usage of every request on resp, the last
// request's response, if the generation was resumed. resp keeps the last
// request's finish reason and error code, so a resumed generation that ends
// blocked still reports why.
func (c *continuation) complete(resp *model.LLMResponse) *model.LLMResponse {
	if !c.resumed() {
		return resp
	}
	if len(c.parts) > 0 {
		resp.Content = c.content()
	}
	resp.UsageMetadata = c.usage
	return resp
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

// appendParts appends copies of next to parts, joining text the way the
// streaming aggregator does: consecutive non-empty text of the same kind
// becomes one part, keeping the first thought signature. Copies keep a resend
// from carrying anything set on the parts after they were received, such as
// client function call IDs. A part with no text and nothing else but the
// thought flag and a thought signature, such as the one a stream can end with,
// is dropped: genai omits empty text, so the part would encode with no data,
// which the API rejects with a 400. ADK Java drops such a part only when it
// carries no signature, since it sends the empty text.
func appendParts(parts, next []*genai.Part) []*genai.Part {
	for _, p := range next {
		if p == nil || isEmpty(p) {
			continue
		}
		if n := len(parts); n > 0 && isText(parts[n-1]) && isText(p) && parts[n-1].Thought == p.Thought {
			joined := *parts[n-1]
			joined.Text += p.Text
			if len(joined.ThoughtSignature) == 0 {
				joined.ThoughtSignature = p.ThoughtSignature
			}
			parts[n-1] = &joined
			continue
		}
		cp := *p
		if p.FunctionCall != nil {
			fc := *p.FunctionCall
			cp.FunctionCall = &fc
		}
		parts = append(parts, &cp)
	}
	return parts
}

// isText reports whether p is non-empty text with nothing else set but the
// thought flag and a thought signature.
func isText(p *genai.Part) bool {
	return p.Text != "" && onlyText(p)
}

// isEmpty reports whether p has no text and nothing else set but the thought
// flag and a thought signature.
func isEmpty(p *genai.Part) bool {
	return p.Text == "" && onlyText(p)
}

func onlyText(p *genai.Part) bool {
	rest := *p
	rest.Text, rest.Thought, rest.ThoughtSignature = "", false, nil
	return reflect.ValueOf(rest).IsZero()
}

// addUsage returns the combined token usage of two requests: counts are
// summed, per-modality counts summed by modality, and anything else is next's.
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
