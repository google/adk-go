// Copyright 2025 The A2A Authors
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

package push

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/log"
)

var tokenHeader = http.CanonicalHeaderKey("X-A2A-Notification-Token")

// HTTPPushSender sends A2A events to a push notification endpoint over HTTP.
type HTTPPushSender struct {
	client      *http.Client
	failOnError bool
}

// HTTPSenderConfig allows to configure [HTTPPushSender].
type HTTPSenderConfig struct {
	// Timeout is used to configure internal [http.Client].
	Timeout time.Duration
	// FailOnError can be set to true to make push sending errors trigger execution cancelation.
	FailOnError bool
}

// NewHTTPPushSender creates a new HTTPPushSender. It uses a default client
// with a 30-second timeout. An optional config can be provided to customize it.
func NewHTTPPushSender(config *HTTPSenderConfig) *HTTPPushSender {
	t := 30 * time.Second
	if config != nil {
		if config.Timeout > 0 {
			t = config.Timeout
		}
	}
	return &HTTPPushSender{
		client:      &http.Client{Timeout: t},
		failOnError: config != nil && config.FailOnError,
	}
}

// SendPush serializes the task to JSON and sends it as an HTTP POST request
// to the URL specified in the push configuration.
func (s *HTTPPushSender) SendPush(ctx context.Context, config *a2a.PushConfig, task *a2a.Task) error {
	jsonData, err := json.Marshal(task)
	if err != nil {
		return s.handleError(ctx, fmt.Errorf("failed to serialize event to JSON: %w", err))
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, config.URL, bytes.NewBuffer(jsonData))
	if err != nil {
		return s.handleError(ctx, fmt.Errorf("failed to create HTTP request: %w", err))
	}

	req.Header.Set("Content-Type", "application/json")
	if config.Token != "" {
		req.Header.Set(tokenHeader, config.Token)
	}
	if config.Auth != nil && config.Auth.Credentials != "" {
		// Find the first supported scheme and apply it.
		for _, scheme := range config.Auth.Schemes {
			switch strings.ToLower(scheme) {
			case "bearer":
				req.Header.Set("Authorization", "Bearer "+config.Auth.Credentials)
			case "basic":
				req.Header.Set("Authorization", "Basic "+config.Auth.Credentials)
			}
		}
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return s.handleError(ctx, fmt.Errorf("failed to send push notification: %w", err))
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			log.Error(ctx, "push response body close failed", err)
		}
	}()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return s.handleError(ctx, fmt.Errorf("push notification endpoint returned non-success status: %s", resp.Status))
	}

	return nil
}

func (s *HTTPPushSender) handleError(ctx context.Context, err error) error {
	if s.failOnError {
		return err
	}
	log.Error(ctx, "push sending failed", err)
	return nil
}
