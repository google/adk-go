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

package retry

import (
	"errors"
	"io"
	"net"
	"strings"

	"google.golang.org/genai"
)

// IsTransient reports whether err looks like a transient model-call error
// worth retrying. It returns true for:
//
//   - genai.APIError with Code == 429 or Code in [500, 599]
//   - net.Error reporting Timeout()
//   - io.ErrUnexpectedEOF / io.EOF mid-call
//   - error strings containing common transient signals ("rate limit",
//     "deadline exceeded", "connection reset", "service unavailable")
//
// It returns false for context cancellation; callers should check
// context.Err() separately so that parent-context cancellation is honored.
func IsTransient(err error) bool {
	if err == nil {
		return false
	}

	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		return isRetriableHTTPCode(apiErr.Code)
	}

	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}

	if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
		return true
	}

	// Last-resort string sniffing for providers that don't expose typed errors.
	msg := strings.ToLower(err.Error())
	for _, marker := range transientMarkers {
		if strings.Contains(msg, marker) {
			return true
		}
	}
	return false
}

func isRetriableHTTPCode(code int) bool {
	if code == 429 {
		return true
	}
	if code >= 500 && code <= 599 {
		return true
	}
	return false
}

var transientMarkers = []string{
	"rate limit",
	"resource_exhausted",
	"resource exhausted",
	"deadline exceeded",
	"connection reset",
	"connection refused",
	"service unavailable",
	"temporarily unavailable",
	"try again",
	"too many requests",
}
