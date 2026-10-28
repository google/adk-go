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

// Package slackapi is a minimal client for the Slack Web API methods the service uses.
package slackapi

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"
)

// Client calls the Slack Web API with one bot token.
type Client struct {
	token   string
	baseURL string
	http    *http.Client
	sleep   func(context.Context, time.Duration) error
}

// Option configures a Client.
type Option func(*Client)

// WithBaseURL replaces the default "https://slack.com/api/". It must end in "/".
func WithBaseURL(u string) Option { return func(c *Client) { c.baseURL = u } }

// WithHTTPClient replaces the default client, which times out after 10s.
func WithHTTPClient(hc *http.Client) Option { return func(c *Client) { c.http = hc } }

// New returns a Client that authenticates with botToken.
func New(botToken string, opts ...Option) *Client {
	c := &Client{
		token:   botToken,
		baseURL: "https://slack.com/api/",
		http:    &http.Client{Timeout: 10 * time.Second},
		sleep:   sleepCtx,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	select {
	case <-time.After(d):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// APIError is a Slack-level failure: HTTP 200 with {"ok": false, "error": "..."}.
type APIError struct{ Code string }

func (e *APIError) Error() string { return "slack api: " + e.Code }

type response struct {
	OK     bool   `json:"ok"`
	Error  string `json:"error"`
	TS     string `json:"ts"`
	UserID string `json:"user_id"`
}

// PostMessage posts text into channel, inside threadTS when non-empty, and
// returns the new message's ts.
func (c *Client) PostMessage(ctx context.Context, channel, threadTS, text string) (ts string, err error) {
	resp, err := c.call(ctx, "chat.postMessage", struct {
		Channel     string `json:"channel"`
		Text        string `json:"text"`
		ThreadTS    string `json:"thread_ts,omitempty"`
		UnfurlLinks bool   `json:"unfurl_links"`
	}{channel, text, threadTS, false})
	if err != nil {
		return "", err
	}
	return resp.TS, nil
}

// UpdateMessage replaces the text of the message ts in channel.
func (c *Client) UpdateMessage(ctx context.Context, channel, ts, text string) error {
	_, err := c.call(ctx, "chat.update", map[string]string{"channel": channel, "ts": ts, "text": text})
	return err
}

// DeleteMessage deletes the message ts in channel.
func (c *Client) DeleteMessage(ctx context.Context, channel, ts string) error {
	_, err := c.call(ctx, "chat.delete", map[string]string{"channel": channel, "ts": ts})
	return err
}

// AuthTest returns the bot's own user ID (auth.test "user_id").
func (c *Client) AuthTest(ctx context.Context) (userID string, err error) {
	resp, err := c.call(ctx, "auth.test", struct{}{})
	if err != nil {
		return "", err
	}
	return resp.UserID, nil
}

// call sends one request and retries it once on 429, the one failure that
// tells the client when to retry (Retry-After). Other transient errors, such
// as internal_error and service_unavailable, are left to the caller.
func (c *Client) call(ctx context.Context, method string, body any) (*response, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("slack: %s: %w", method, err)
	}
	for attempt := 0; ; attempt++ {
		resp, retryAfter, err := c.do(ctx, method, payload)
		if retryAfter < 0 || attempt > 0 {
			return resp, err
		}
		if err := c.sleep(ctx, retryAfter); err != nil {
			return nil, fmt.Errorf("slack: %s: %w", method, err)
		}
	}
}

// do sends one request. A non-negative retryAfter means Slack rate-limited it.
func (c *Client) do(ctx context.Context, method string, payload []byte) (_ *response, retryAfter time.Duration, _ error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+method, bytes.NewReader(payload))
	if err != nil {
		return nil, -1, fmt.Errorf("slack: %s: %w", method, err)
	}
	req.Header.Set("Content-Type", "application/json; charset=utf-8")
	req.Header.Set("Authorization", "Bearer "+c.token)
	httpResp, err := c.http.Do(req)
	if err != nil {
		return nil, -1, fmt.Errorf("slack: %s: %w", method, err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	if httpResp.StatusCode == http.StatusTooManyRequests {
		_, _ = io.Copy(io.Discard, httpResp.Body)
		secs, err := strconv.Atoi(httpResp.Header.Get("Retry-After"))
		if err != nil || secs < 0 {
			// Slack always sends Retry-After on 429. One second is a cautious guess without it.
			secs = 1
		}
		return nil, time.Duration(secs) * time.Second, fmt.Errorf("slack: %s: HTTP %d", method, httpResp.StatusCode)
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode > 299 {
		_, _ = io.Copy(io.Discard, httpResp.Body)
		return nil, -1, fmt.Errorf("slack: %s: HTTP %d", method, httpResp.StatusCode)
	}
	var resp response
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, -1, fmt.Errorf("slack: %s: decoding response: %w", method, err)
	}
	if !resp.OK {
		return nil, -1, &APIError{Code: resp.Error}
	}
	return &resp, -1, nil
}
