//    Copyright 2026 Google LLC
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

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"sync"
	"time"
)

// ---------------- Retry Configuration ----------------

var retryStatusCodes = map[int]bool{
	429: true,
	500: true,
	502: true,
	503: true,
	504: true,
}

const (
	maxRetries    = 6
	backoffFactor = 2
)

// ---------------- API Call Counter ----------------

var (
	apiCallCount int
	counterLock  sync.Mutex
)

func GetAPICallCount() int {
	counterLock.Lock()
	defer counterLock.Unlock()
	return apiCallCount
}

func ResetAPICallCount() {
	counterLock.Lock()
	defer counterLock.Unlock()
	apiCallCount = 0
}

func incrementAPICallCount() {
	counterLock.Lock()
	apiCallCount++
	counterLock.Unlock()
}

// ---------------- HTTP Client ----------------

var httpClient = &http.Client{
	Timeout: 60 * time.Second,
}

// ---------------- Core HTTP Logic ----------------

func doRequest(req *http.Request) (*http.Response, error) {
	req.Header.Set("Authorization", "token "+GitHubToken)
	req.Header.Set("Accept", "application/vnd.github.v3+json")

	var resp *http.Response
	var err error
	backoff := time.Second

	for attempt := 0; attempt <= maxRetries; attempt++ {
		resp, err = httpClient.Do(req)

		if err == nil && !retryStatusCodes[resp.StatusCode] {
			return resp, nil
		}

		if resp != nil {
			if cerr := resp.Body.Close(); cerr != nil {
				log.Printf("error closing response body: %v", cerr)
			}
		}

		if attempt == maxRetries {
			break
		}

		time.Sleep(backoff)
		backoff *= backoffFactor
	}

	if err != nil {
		return nil, err
	}

	return nil, fmt.Errorf("request failed after retries")
}

// ---------------- Public Request Helpers ----------------

func GetRequest(rawURL string, params map[string]any) (any, error) {
	incrementAPICallCount()

	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, err
	}

	if params != nil {
		q := u.Query()
		for k, v := range params {
			q.Set(k, fmt.Sprintf("%v", v))
		}
		u.RawQuery = q.Encode()
	}

	req, err := http.NewRequest("GET", u.String(), nil)
	if err != nil {
		return nil, err
	}

	resp, err := doRequest(req)
	if err != nil {
		log.Printf("GET request failed for %s: %v", rawURL, err)
		return nil, err
	}
	defer func() {
		if cerr := resp.Body.Close(); cerr != nil {
			log.Printf("error closing response body: %v", cerr)
		}
	}()

	return decodeJSON(resp)
}

func PostRequest(url string, payload any) (any, error) {
	incrementAPICallCount()

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal POST request payload: %w", err)
	}
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}

	resp, err := doRequest(req)
	if err != nil {
		log.Printf("POST request failed for %s: %v", url, err)
		return nil, err
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			log.Printf("error closing response body: %v", err)
		}
	}()

	return decodeJSON(resp)
}

func PatchRequest(url string, payload any) (any, error) {
	incrementAPICallCount()

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal PATCH request payload: %w", err)
	}
	req, err := http.NewRequest("PATCH", url, bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}

	resp, err := doRequest(req)
	if err != nil {
		log.Printf("PATCH request failed for %s: %v", url, err)
		return nil, err
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			log.Printf("error closing response body: %v", err)
		}
	}()

	return decodeJSON(resp)
}

func DeleteRequest(url string) (any, error) {
	incrementAPICallCount()

	req, err := http.NewRequest("DELETE", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := doRequest(req)
	if err != nil {
		log.Printf("DELETE request failed for %s: %v", url, err)
		return nil, err
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			log.Printf("error closing response body: %v", err)
		}
	}()

	if resp.StatusCode == 204 {
		return map[string]any{
			"status":  "success",
			"message": "Deletion successful.",
		}, nil
	}

	return decodeJSON(resp)
}

// ---------------- JSON Helper ----------------

func decodeJSON(resp *http.Response) (any, error) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var data any
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, err
	}

	return data, nil
}

// ---------------- Issue Search ----------------

func GetOldOpenIssueNumbers(owner, repo string, daysOld *float64) ([]int, error) {
	days := STALE_HOURS_THRESHOLD / 24
	if daysOld != nil {
		days = *daysOld
	}

	cutoff := time.Now().UTC().
		Add(-time.Duration(days*24) * time.Hour).
		Format("2006-01-02T15:04:05Z")

	query := fmt.Sprintf(
		"repo:%s/%s is:issue state:open created:<%s",
		owner, repo, cutoff,
	)

	log.Printf("SEARCH QUERY: %s", query)
	log.Printf("Searching for issues created before %s...", cutoff)

	var issueNumbers []int
	page := 1

	for {
		params := map[string]any{
			"q":        query,
			"per_page": 100,
			"page":     page,
		}

		dataAny, err := GetRequest(
			"https://api.github.com/search/issues",
			params,
		)
		if err != nil {
			log.Printf("GitHub search failed on page %d: %v", page, err)
			break
		}

		data, ok := dataAny.(map[string]any)
		if !ok {
			log.Printf("Invalid API response format")
			break
		}

		items, ok := data["items"].([]any)
		if !ok || len(items) == 0 {
			break
		}

		for _, item := range items {
			m := item.(map[string]any)
			if _, isPR := m["pull_request"]; !isPR {
				if n, ok := m["number"].(float64); ok {
					issueNumbers = append(issueNumbers, int(n))
				}
			}
		}

		if len(items) < 100 {
			break
		}
		page++
	}

	log.Printf("Found %d stale issues.", len(issueNumbers))
	return issueNumbers, nil
}
