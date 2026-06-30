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

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const httpTimeout = 60 * time.Second

var httpClient = &http.Client{
	Timeout: httpTimeout,
}

// getHeaders get the headers with the required key-values
func getHeaders() map[string]string {
	return map[string]string{
		"Authorization": "token " + GitHubToken,
		"Accept":        "application/vnd.github.v3+json",
	}
}

// getRequest performs a GET request to the GitHub API.
func getRequest(requestURL string, params map[string]string) (map[string]any, error) {
	req, err := http.NewRequest("GET", requestURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Add headers
	headers := getHeaders()
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	// Add query parameters
	if params != nil {
		q := req.URL.Query()
		for k, v := range params {
			q.Add(k, v)
		}
		req.URL.RawQuery = q.Encode()
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyContent, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read response body after the request failed with status %d: %w", resp.StatusCode, err)
		}
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(bodyContent))
	}

	var result map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result, nil
}

// postRequest performs a POST request to the GitHub API.
func postRequest(requestURL string, payload any) (map[string]any, error) {
	return makeRequestWithBody("POST", requestURL, payload)
}

// patchRequest performs a PATCH request to the GitHub API.
func patchRequest(requestURL string, payload any) (map[string]any, error) {
	return makeRequestWithBody("PATCH", requestURL, payload)
}

func makeRequestWithBody(method, requestURL string, payload any) (map[string]any, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	req, err := http.NewRequest(method, requestURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Add headers
	headers := getHeaders()
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyContent, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read response body after the request failed with status %d: %w", resp.StatusCode, err)
		}
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(bodyContent))
	}

	var result map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result, nil
}
