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

package agentcard

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/log"
)

// ErrStatusNotOK is an error returned by Resolver when HTTP request returned a non-OK status.
type ErrStatusNotOK struct {
	StatusCode int
	Status     string
}

func (e *ErrStatusNotOK) Error() string {
	return fmt.Sprintf("card request failed, status: %s", e.Status)
}

const defaultAgentCardPath = "/.well-known/agent-card.json"

var defaultClient = &http.Client{Timeout: 30 * time.Second}

// DefaultResolver is configured with an [http.Client] with a 30-second timeout.
var DefaultResolver = &Resolver{Client: defaultClient}

// Resolver is used to fetch an [a2a.AgentCard].
type Resolver struct {
	// Client can be used to configure appropriate timeout, retry policy, and connection pooling
	Client *http.Client
}

// NewResolver is a [Resolver] constructor function.
func NewResolver(client *http.Client) *Resolver {
	return &Resolver{Client: client}
}

// ResolveOption is used to customize Resolve behavior.
type ResolveOption func(r *resolveRequest)

type resolveRequest struct {
	path    string
	headers map[string]string
}

// Resolve fetches an [a2a.AgentCard] from the provided base URL.
// By default the request is sent for the  /.well-known/agent-card.json path.
func (r *Resolver) Resolve(ctx context.Context, baseURL string, opts ...ResolveOption) (*a2a.AgentCard, error) {
	reqSpec := &resolveRequest{path: defaultAgentCardPath, headers: make(map[string]string)}
	for _, o := range opts {
		o(reqSpec)
	}

	reqUrl, err := url.JoinPath(baseURL, reqSpec.path)
	if err != nil {
		return nil, fmt.Errorf("url construction failed: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "GET", reqUrl, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to construct a request: %w", err)
	}
	for h, val := range reqSpec.headers {
		req.Header.Add(h, val)
	}

	client := r.Client
	if client == nil {
		client = defaultClient
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("card request failed: %w", err)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			log.Error(ctx, "failed to close response body", err, "from", reqUrl)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		return nil, &ErrStatusNotOK{StatusCode: resp.StatusCode, Status: resp.Status}
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read card response: %w", err)
	}

	var card a2a.AgentCard
	if err := json.Unmarshal(body, &card); err != nil {
		return nil, fmt.Errorf("card parsing failed: %w", err)
	}

	return &card, nil
}

// WithPath makes Resolve fetch from the provided path relative to base URL.
func WithPath(path string) ResolveOption {
	return func(r *resolveRequest) {
		r.path = path
	}
}

// WithRequestHeader makes Resolve perform fetch attaching the provided HTTP headers.
func WithRequestHeader(name string, val string) ResolveOption {
	return func(r *resolveRequest) {
		r.headers[name] = val
	}
}
