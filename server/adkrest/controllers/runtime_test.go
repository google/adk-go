// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package controllers

import (
	"errors"
	"io"
	"net/http/httptest"
	"strings"
	"testing"
)

// errorCloser is a helper that wraps an io.Reader and returns an error on Close.
type errorCloser struct {
	io.Reader
}

func (errorCloser) Close() error {
	return errors.New("persistent close error")
}

func TestDecodeRequestBody_DoesNotOverwriteDecodeErrorOnClose(t *testing.T) {
	// Invalid JSON should produce a 400 error from decodeRequestBody.
	// This test reproduces a bug where a deferred Body.Close() overwrites
	// the decode error (named return value) and causes err to become nil.
	req := httptest.NewRequest("POST", "http://example/runtime", io.NopCloser(strings.NewReader("{")))

	_, err := decodeRequestBody(req)
	if err == nil {
		t.Fatalf("expected decodeRequestBody to return an error for invalid JSON, got nil")
	}
}

func TestDecodeRequestBody_ReturnsErrorOnClose(t *testing.T) {
	// Use valid JSON to ensure decoding succeeds.
	const validJSON = `{"appName":"test-app","userId":"test-user","sessionId":"test-session","newMessage":{"parts":[{"text":"hello"}]}}`
	req := httptest.NewRequest("POST", "http://example/runtime", errorCloser{strings.NewReader(validJSON)})

	_, err := decodeRequestBody(req)
	if err == nil {
		t.Fatal("expected decodeRequestBody to return an error from Body.Close(), but got nil")
	}
	if !strings.Contains(err.Error(), "persistent close error") {
		t.Fatalf("expected error to contain 'persistent close error', got: %v", err)
	}
}
