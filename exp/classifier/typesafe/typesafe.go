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

package typesafe

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"math/rand/v2"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/exp/classifier"
)

const (
	// DefaultModel is the Jev model used when [Config.Model] is empty.
	DefaultModel = "jev-latest"
	// DefaultTimeout is the per-attempt timeout used when both [Config.Timeout]
	// and [Config.HTTPClient] are unset.
	DefaultTimeout = 30 * time.Second
	// DefaultMaxAttempts is the number of attempts per [Classifier.Classify] call
	// used when [Config.MaxAttempts] is 0.
	DefaultMaxAttempts = 3

	defaultBaseURL   = "https://api.typesafe.ai"
	apiKeyEnv        = "TYPESAFE_API_KEY"
	baseURLEnv       = "TYPESAFE_BASE_URL"
	systemOnePath    = "/v1/systemone"
	initialBackoff   = 500 * time.Millisecond
	maxBackoff       = 8 * time.Second
	backoffJitterMax = time.Second
	clampTolerance   = 1e-6
	maxResponseBytes = 16 << 20 // 16 MiB
)

// sharedDefaultClient pools connections across calls when Config.HTTPClient is
// nil. Redirect following is disabled to match Python httpx's default
// (follow_redirects=False) and to prevent forwarding Bearer credentials or POST
// payloads across 3xx redirects.
var sharedDefaultClient = &http.Client{
	CheckRedirect: func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	},
}

// Config configures a [Classifier] constructed by [New].
type Config struct {
	// Model is the Jev model name, such as "jev-latest" or a pinned version like
	// "jev-1.13". Empty defaults to [DefaultModel].
	Model string
	// APIKey is the TypeSafe bearer API key. When empty, [Classifier.Classify]
	// reads TYPESAFE_API_KEY on each call. Set APIKey explicitly when BaseURL
	// points at a non-TypeSafe endpoint so TYPESAFE_API_KEY is not sent there.
	APIKey string
	// BaseURL overrides the API origin. When empty, [Classifier.Classify] reads
	// TYPESAFE_BASE_URL and otherwise uses "https://api.typesafe.ai". Unlike
	// Python's httpx.AsyncClient, Go's *http.Client carries no base URL.
	BaseURL string
	// Timeout bounds each individual HTTP attempt (not the total call across
	// retries). When 0, HTTPClient.Timeout is used if HTTPClient is non-nil, and
	// [DefaultTimeout] (30s) is used otherwise.
	Timeout time.Duration
	// MaxAttempts is the total attempts per [Classifier.Classify] call, counting
	// the first. Zero defaults to [DefaultMaxAttempts] (3); set 1 to disable
	// retries. HTTP 429, HTTP 529, per-attempt timeouts, and transient network
	// errors are retried with exponential backoff and jitter.
	MaxAttempts int
	// HTTPClient sends outbound requests. When nil, a shared default *http.Client
	// (with redirect following disabled) is used so keep-alive connections are
	// pooled safely across concurrent calls. Classifier never closes or mutates a
	// caller-supplied client.
	HTTPClient *http.Client
}

// String formats Config with APIKey redacted.
func (c Config) String() string {
	key := ""
	if c.APIKey != "" {
		key = "[REDACTED]"
	}
	return fmt.Sprintf("typesafe.Config{Model:%q, APIKey:%q, BaseURL:%q, Timeout:%s, MaxAttempts:%d}",
		c.Model, key, c.BaseURL, c.Timeout, c.MaxAttempts)
}

// GoString formats Config for %#v with APIKey redacted.
func (c Config) GoString() string {
	return c.String()
}

var _ classifier.Classifier = (*Classifier)(nil)

// Classifier answers typed questions using TypeSafe's Jev decision models.
//
// The zero value is not usable; construct one with [New]. A Classifier
// constructed with [New] is safe for concurrent use.
type Classifier struct {
	model       string
	apiKey      string
	baseURL     string
	timeout     time.Duration
	maxAttempts int
	httpClient  *http.Client
	sleep       func(context.Context, time.Duration) error
}

// New validates cfg and constructs a TypeSafe Jev [Classifier].
//
// Missing API credentials are checked on [Classifier.Classify] rather than in
// New so agents and workflows can be constructed before TYPESAFE_API_KEY is
// set in the environment.
func New(cfg Config) (*Classifier, error) {
	model := DefaultModel
	if cfg.Model != "" {
		if strings.TrimSpace(cfg.Model) == "" {
			return nil, fmt.Errorf("%w: Model must not be whitespace-only", ErrInvalidConfig)
		}
		model = cfg.Model
	}
	if cfg.BaseURL != "" {
		if err := validateBaseURL("Config.BaseURL", cfg.BaseURL); err != nil {
			return nil, err
		}
	}
	if cfg.Timeout < 0 {
		return nil, fmt.Errorf("%w: Timeout must be non-negative", ErrInvalidConfig)
	}
	maxAttempts := DefaultMaxAttempts
	if cfg.MaxAttempts < 0 {
		return nil, fmt.Errorf("%w: MaxAttempts must be >= 1 (or 0 for default)", ErrInvalidConfig)
	}
	if cfg.MaxAttempts > 0 {
		maxAttempts = cfg.MaxAttempts
	}
	return &Classifier{
		model:       model,
		apiKey:      strings.TrimSpace(cfg.APIKey),
		baseURL:     strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/"),
		timeout:     cfg.Timeout,
		maxAttempts: maxAttempts,
		httpClient:  cfg.HTTPClient,
		sleep:       sleepWithContext,
	}, nil
}

// Name returns the configured Jev model name.
func (c *Classifier) Name() string {
	return c.model
}

// String formats Classifier without exposing unexported credentials. It has a
// value receiver so fmt also uses it for a Classifier value, as when a struct
// holding one is printed with %+v.
func (c Classifier) String() string {
	return fmt.Sprintf("typesafe.Classifier{Model:%q}", c.model)
}

// GoString formats Classifier for %#v without exposing unexported credentials.
func (c Classifier) GoString() string {
	return c.String()
}

// Classify sends input and questions to TypeSafe's /v1/systemone endpoint in a
// single request and returns one validated [classifier.Answer] per question.
func (c *Classifier) Classify(
	ctx context.Context,
	input any,
	questions map[string]classifier.Question,
) (*classifier.Response, error) {
	if c == nil || c.maxAttempts < 1 {
		return nil, fmt.Errorf("%w: Classifier must be constructed with New", ErrInvalidConfig)
	}
	// Reject an empty question map up front (consistent with adk-python's
	// LlmClassifier) rather than issuing a no-op network round-trip.
	if len(questions) == 0 {
		return nil, fmt.Errorf("%w: classify requires at least one question", classifier.ErrInvalidQuestion)
	}
	state, err := inputToState(input)
	if err != nil {
		return nil, err
	}

	names := slices.Sorted(maps.Keys(questions))
	wireQuestions := make(map[string]wireQuestion, len(questions))
	for _, name := range names {
		wq, qErr := questionToWire(name, questions[name])
		if qErr != nil {
			return nil, qErr
		}
		wireQuestions[name] = wq
	}

	apiKey := c.apiKey
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv(apiKeyEnv))
	}
	if apiKey == "" {
		return nil, fmt.Errorf("%w: set %s or pass Config.APIKey", ErrMissingAPIKey, apiKeyEnv)
	}

	endpoint, err := c.endpointURL()
	if err != nil {
		return nil, err
	}

	bodyBytes, err := json.Marshal(wireRequest{
		Model:     c.model,
		State:     state,
		Questions: wireQuestions,
	})
	if err != nil {
		return nil, fmt.Errorf("%w: encoding request payload", classifier.ErrInvalidInput)
	}

	respBytes, err := c.sendWithRetry(ctx, endpoint, apiKey, bodyBytes)
	if err != nil {
		return nil, err
	}

	return parseResponse(respBytes, names, questions)
}

type wireRequest struct {
	Model     string                  `json:"model"`
	State     any                     `json:"state"`
	Questions map[string]wireQuestion `json:"questions"`
}

type wireQuestion struct {
	Type         string `json:"type"`
	Instructions any    `json:"instructions"`
	Criteria     any    `json:"criteria,omitempty"`
}

func validateBaseURL(source, raw string) error {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return fmt.Errorf("%w: %s must not be whitespace-only", ErrInvalidConfig, source)
	}
	parsed, err := url.Parse(trimmed)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return fmt.Errorf("%w: %s must be a valid http or https URL with a host", ErrInvalidConfig, source)
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" || strings.ContainsAny(trimmed, "?#") {
		return fmt.Errorf("%w: %s must not contain userinfo, query parameters, or fragments", ErrInvalidConfig, source)
	}
	return nil
}

func (c *Classifier) endpointURL() (string, error) {
	if c.baseURL != "" {
		return c.baseURL + systemOnePath, nil
	}
	if envURL := os.Getenv(baseURLEnv); envURL != "" {
		if err := validateBaseURL(baseURLEnv, envURL); err != nil {
			return "", err
		}
		return strings.TrimRight(strings.TrimSpace(envURL), "/") + systemOnePath, nil
	}
	return defaultBaseURL + systemOnePath, nil
}

// inputToState converts input to Jev's "state" payload, mirroring adk-python's
// _input_to_json with two Go-specific adjustments:
//   - Go's genai.Part.Text is string rather than *string, so an empty text Part
//     is accepted while nil Parts and any Part carrying non-text fields are
//     rejected.
//   - Parts with Thought=true are rejected (unlike adk-python's _input_to_json,
//     which only checks whether part.text is None) so internal model reasoning
//     is never concatenated into classified text. Non-thought parts carrying a
//     ThoughtSignature (including trailing empty-text streamed signature parts)
//     are accepted.
//
// Only a single top-level Content is converted; classifier.ValidateInput
// rejects genai struct values nested in JSON input, such as a []*genai.Content
// history or a bare genai.Part, before they could be sent as "state".
func inputToState(input any) (any, error) {
	if err := classifier.ValidateInput(input); err != nil {
		return nil, err
	}
	switch v := input.(type) {
	case *genai.Content:
		return contentToText(v.Parts)
	case genai.Content:
		return contentToText(v.Parts)
	default:
		return input, nil
	}
}

func contentToText(parts []*genai.Part) (string, error) {
	var b strings.Builder
	for _, part := range parts {
		if part == nil || isNonTextPart(part) {
			return "", fmt.Errorf(
				"%w: typesafe classifier accepts text and JSON input only; content contains a non-text part",
				classifier.ErrInvalidInput,
			)
		}
		b.WriteString(part.Text)
	}
	return b.String(), nil
}

// isNonTextPart checks for any field on part unaccounted for by plain text
// extraction, using zero-comparison so fields added to genai.Part in future
// SDK releases are rejected by default rather than silently dropped.
func isNonTextPart(part *genai.Part) bool {
	rest := *part
	rest.Text = ""
	rest.PartMetadata = nil
	// Gemini 2.5+ attaches ThoughtSignature to prose parts and trailing
	// empty-text streaming parts. Leave rest.Thought untouched so thought parts
	// still fail IsZero.
	rest.ThoughtSignature = nil
	return !reflect.ValueOf(rest).IsZero()
}

func questionToWire(name string, q classifier.Question) (wireQuestion, error) {
	if err := classifier.ValidateQuestion(q); err != nil {
		return wireQuestion{}, fmt.Errorf("question %q: %w", name, err)
	}
	var wq wireQuestion
	switch v := derefQuestion(q).(type) {
	case classifier.Choice:
		wq = wireQuestion{
			Type:         "choice",
			Instructions: v.Instructions,
			Criteria:     v.Criteria,
		}
	case classifier.Binary:
		wq = wireQuestion{
			Type:         "noul",
			Instructions: v.Instructions,
		}
		if v.Criteria != nil {
			wq.Criteria = map[string]any{"true": v.Criteria}
		}
	case classifier.Score:
		wq = wireQuestion{
			Type:         "score",
			Instructions: v.Instructions,
			Criteria:     v.Criteria,
		}
	}
	if wq.Instructions == nil {
		return wireQuestion{}, fmt.Errorf("%w: typesafe classifier needs instructions for question %q", classifier.ErrInvalidQuestion, name)
	}
	return wq, nil
}

func derefQuestion(q classifier.Question) classifier.Question {
	switch v := q.(type) {
	case *classifier.Choice:
		return *v
	case *classifier.Binary:
		return *v
	case *classifier.Score:
		return *v
	default:
		return q
	}
}

func (c *Classifier) clientForAttempt() (*http.Client, time.Duration) {
	if c.httpClient == nil {
		timeout := c.timeout
		if timeout == 0 {
			timeout = DefaultTimeout
		}
		return sharedDefaultClient, timeout
	}
	if c.timeout > 0 {
		// Shallow-copy http.Client with Timeout cleared so Config.Timeout
		// overrides HTTPClient.Timeout whether shorter or longer, without
		// mutating the caller's *http.Client.
		copied := *c.httpClient
		copied.Timeout = 0
		return &copied, c.timeout
	}
	return c.httpClient, 0
}

func (c *Classifier) sendWithRetry(ctx context.Context, endpoint, apiKey string, bodyBytes []byte) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	client, perAttemptTimeout := c.clientForAttempt()

	for attempt := 0; ; attempt++ {
		respBytes, retry, err := c.doAttempt(ctx, client, perAttemptTimeout, endpoint, apiKey, bodyBytes)
		if err == nil {
			return respBytes, nil
		}
		if !retry || attempt >= c.maxAttempts-1 {
			return nil, err
		}
		// When the context ends during backoff, keep the error that triggered
		// the retry alongside the context error so callers can still see the
		// 429, 529, or transport failure. The second check covers the sleep's
		// select picking the elapsed timer over the done context.
		if sleepErr := c.sleep(ctx, retryBackoff(attempt)); sleepErr != nil {
			return nil, errors.Join(err, sleepErr)
		}
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, errors.Join(err, ctxErr)
		}
	}
}

func (c *Classifier) doAttempt(
	ctx context.Context,
	client *http.Client,
	timeout time.Duration,
	endpoint, apiKey string,
	bodyBytes []byte,
) ([]byte, bool, error) {
	attemptCtx := ctx
	var cancel context.CancelFunc
	if timeout > 0 {
		attemptCtx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	req, err := http.NewRequestWithContext(attemptCtx, http.MethodPost, endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, false, fmt.Errorf("%w: invalid request URL", ErrInvalidConfig)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, false, ctxErr
		}
		return nil, isRetryableTransportError(err), sanitizeTransportError(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// Drain a small prefix before closing so keep-alive connections can be
		// reused across retry attempts.
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		retry := resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode == statusOverloaded
		return nil, retry, newAPIError(resp)
	}

	respBytes, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes+1))
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, false, ctxErr
		}
		return nil, isRetryableTransportError(err), sanitizeTransportError(err)
	}
	if len(respBytes) > maxResponseBytes {
		return nil, false, fmt.Errorf("%w: response body exceeds %d bytes", ErrMalformedResponse, maxResponseBytes)
	}
	return respBytes, false, nil
}

// retryBackoff mirrors tenacity.wait_exponential_jitter(initial=0.5, max=8.0):
// min(0.5s * 2^attempt + uniform(0, 1s), 8s).
func retryBackoff(attempt int) time.Duration {
	exp := float64(initialBackoff) * math.Pow(2, float64(attempt))
	jitter := rand.Float64() * float64(backoffJitterMax)
	wait := min(exp+jitter, float64(maxBackoff))
	return time.Duration(wait)
}

func sleepWithContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func parseResponse(
	data []byte,
	names []string,
	questions map[string]classifier.Question,
) (*classifier.Response, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var doc any
	if err := dec.Decode(&doc); err != nil {
		return nil, fmt.Errorf("%w: invalid JSON response", ErrMalformedResponse)
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("%w: trailing JSON tokens in response", ErrMalformedResponse)
	}

	top, ok := doc.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%w: response is not a JSON object", ErrMalformedResponse)
	}

	var modelVersion string
	if rawModel, exists := top["model"]; exists && rawModel != nil {
		mv, isStr := rawModel.(string)
		if !isStr {
			return nil, fmt.Errorf("%w: model field is not a string", ErrMalformedResponse)
		}
		modelVersion = mv
	}

	rawAnswers, ok := top["answers"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%w: missing or non-object answers field", ErrMalformedResponse)
	}

	answers := make(map[string]classifier.Answer, len(questions))
	for _, name := range names {
		ans, err := parseAnswer(name, derefQuestion(questions[name]), rawAnswers[name])
		if err != nil {
			return nil, err
		}
		answers[name] = ans
	}
	return &classifier.Response{
		Answers:      answers,
		ModelVersion: modelVersion,
	}, nil
}

func parseAnswer(name string, q classifier.Question, rawAny any) (classifier.Answer, error) {
	raw, ok := rawAny.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%w: missing answer object for question %q", ErrMalformedResponse, name)
	}

	expectedType := expectedWireType(q)
	gotType, ok := raw["type"].(string)
	if !ok || gotType != expectedType {
		return nil, fmt.Errorf("%w: unexpected answer type for question %q (expected %q)", ErrMalformedResponse, name, expectedType)
	}

	var confidence *float64
	if rawConf, exists := raw["confidence"]; exists && rawConf != nil {
		c, err := readNumber(name, "confidence", rawConf, 1.0)
		if err != nil {
			return nil, err
		}
		confidence = &c
	}

	switch v := q.(type) {
	case classifier.Binary:
		prob, err := readNumber(name, "noul", raw["noul"], 1.0)
		if err != nil {
			return nil, err
		}
		return classifier.BinaryAnswer{
			Probability: prob,
			Confidence:  confidence,
		}, nil

	case classifier.Choice:
		choice, ok := raw["choice"].(string)
		if !ok {
			return nil, fmt.Errorf("%w: non-string choice for question %q", ErrMalformedResponse, name)
		}
		if _, validLabel := v.Criteria[choice]; !validLabel {
			return nil, fmt.Errorf("%w: choice for question %q is not one of its declared labels", ErrMalformedResponse, name)
		}
		keys := slices.Sorted(maps.Keys(v.Criteria))
		probs, err := readProbabilities(name, expectedType, raw["probabilities"], keys)
		if err != nil {
			return nil, err
		}
		return classifier.ChoiceAnswer{
			Choice:        choice,
			Probabilities: probs,
			Confidence:    confidence,
		}, nil

	default:
		sc := q.(classifier.Score)
		top := len(sc.Criteria) - 1
		keys := make([]string, top+1)
		for level := range top + 1 {
			keys[level] = strconv.Itoa(level)
		}
		strProbs, err := readProbabilities(name, expectedType, raw["probabilities"], keys)
		if err != nil {
			return nil, err
		}
		scoreVal, err := readNumber(name, "score", raw["score"], float64(top))
		if err != nil {
			return nil, err
		}
		levelProbs := make(map[int]float64, top+1)
		for level := range top + 1 {
			levelProbs[level] = strProbs[keys[level]]
		}
		return classifier.ScoreAnswer{
			Score:         scoreVal,
			Probabilities: levelProbs,
			Confidence:    confidence,
		}, nil
	}
}

func expectedWireType(q classifier.Question) string {
	switch q.(type) {
	case classifier.Binary:
		return "noul"
	case classifier.Choice:
		return "choice"
	default:
		return "score"
	}
}

func readProbabilities(name, answerType string, rawProbs any, expectedKeys []string) (map[string]float64, error) {
	probMap, ok := rawProbs.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%w: malformed %q answer for question %q: probabilities is not an object", ErrMalformedResponse, answerType, name)
	}
	if len(probMap) != len(expectedKeys) {
		return nil, fmt.Errorf(
			"%w: probabilities count (%d) for question %q does not match expected options (%d)",
			ErrMalformedResponse, len(probMap), name, len(expectedKeys),
		)
	}
	out := make(map[string]float64, len(expectedKeys))
	for _, key := range expectedKeys {
		num, err := readNumber(name, key, probMap[key], 1.0)
		if err != nil {
			return nil, err
		}
		out[key] = num
	}
	return out, nil
}

// readNumber parses value as a finite float64 in [0, high], clamping rounding
// drift within clampTolerance (1e-6) at either boundary. Mirrors adk-python's
// _read_number without echoing raw response values in error messages.
func readNumber(name, key string, value any, high float64) (float64, error) {
	num, ok := value.(json.Number)
	if !ok {
		return 0, fmt.Errorf("%w: field %q in question %q is not a number", ErrMalformedResponse, key, name)
	}
	f, err := num.Float64()
	if err != nil {
		return 0, fmt.Errorf("%w: field %q in question %q is not a finite number", ErrMalformedResponse, key, name)
	}
	if f < -clampTolerance || f > high+clampTolerance {
		return 0, fmt.Errorf("%w: field %q in question %q is outside [0, %g]", ErrMalformedResponse, key, name, high)
	}
	return min(max(f, 0.0), high), nil
}
