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

package agentregistry

import (
	"context"
	"fmt"
	"iter"
	"regexp"
	"strings"
)

var (
	// reModelSuffix strips a trailing ":method" suffix (e.g. ":predict").
	reModelSuffix = regexp.MustCompile(`:\w+$`)
	// reProjectsSubstring extracts a "projects/..." resource name embedded in a URL.
	reProjectsSubstring = regexp.MustCompile(`projects/.+`)
)

// ListEndpoints returns one page of registered model endpoints. Use opts to set
// a filter and pagination; for automatic paging use [Client.AllEndpoints].
func (c *Client) ListEndpoints(ctx context.Context, opts ...ListOption) (*EndpointsPage, error) {
	var page EndpointsPage
	if err := c.get(ctx, "endpoints", listValues(opts...), &page); err != nil {
		return nil, err
	}
	return &page, nil
}

// GetEndpoint returns the metadata of a single endpoint. name is the full
// resource name (e.g. "projects/<p>/locations/<l>/endpoints/<id>").
func (c *Client) GetEndpoint(ctx context.Context, name string) (*Endpoint, error) {
	var endpoint Endpoint
	if err := c.get(ctx, name, nil, &endpoint); err != nil {
		return nil, err
	}
	return &endpoint, nil
}

// AllEndpoints iterates over every endpoint matching opts, fetching pages on
// demand. If a page fetch fails the iterator yields a single (nil, error) and
// stops.
func (c *Client) AllEndpoints(ctx context.Context, opts ...ListOption) iter.Seq2[*Endpoint, error] {
	return pages(ctx, opts, func(ctx context.Context, o ...ListOption) ([]Endpoint, string, error) {
		page, err := c.ListEndpoints(ctx, o...)
		if err != nil {
			return nil, "", err
		}
		return page.Endpoints, page.NextPageToken, nil
	})
}

// ModelName resolves an endpoint into a model resource name (e.g.
// "projects/<p>/locations/<l>/publishers/google/models/<model>"). endpointName
// is the full endpoint resource name.
func (c *Client) ModelName(ctx context.Context, endpointName string) (string, error) {
	endpoint, err := c.GetEndpoint(ctx, endpointName)
	if err != nil {
		return "", err
	}
	uri, _, _, ok := connectionURI(nil, endpoint.Interfaces, "", "")
	if !ok {
		return "", fmt.Errorf("agentregistry: connection URI not found for endpoint %q", endpointName)
	}
	return parseModelName(uri), nil
}

// parseModelName extracts a model resource name from an endpoint URI: it strips
// a trailing ":method" suffix and, if the URI is not already a "projects/..."
// resource name, extracts the embedded one. It mirrors adk-python's
// get_model_name parsing.
func parseModelName(uri string) string {
	uri = reModelSuffix.ReplaceAllString(uri, "")
	if strings.HasPrefix(uri, "projects/") {
		return uri
	}
	if m := reProjectsSubstring.FindString(uri); m != "" {
		return m
	}
	return uri
}
