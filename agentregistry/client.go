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
	"net/url"
	"regexp"
	"slices"
	"strings"
)

// This file holds the [Client] discovery methods for the registry's resources:
// agents, MCP servers, and model endpoints. Each resource exposes List (one
// page), Get (single resource), and All (auto-paging iterator).

// ListAgents returns one page of registered A2A agents. For automatic paging
// use [Client.AllAgents].
func (c *Client) ListAgents(ctx context.Context, opts ...ListOption) (*ListAgentsResponse, error) {
	return getResource[ListAgentsResponse](ctx, c, "agents", listValues(opts...))
}

// GetAgent returns the metadata of a single agent. name is the full resource
// name (e.g. "projects/<p>/locations/<l>/agents/<id>").
func (c *Client) GetAgent(ctx context.Context, name string) (*Agent, error) {
	return getResource[Agent](ctx, c, name, nil)
}

// AllAgents iterates over every agent matching opts, fetching pages on demand.
// If a page fetch fails the iterator yields a single (nil, error) and stops.
func (c *Client) AllAgents(ctx context.Context, opts ...ListOption) iter.Seq2[*Agent, error] {
	return pages(ctx, opts, func(ctx context.Context, o ...ListOption) ([]Agent, string, error) {
		page, err := c.ListAgents(ctx, o...)
		if err != nil {
			return nil, "", err
		}
		return page.Agents, page.NextPageToken, nil
	})
}

// ListMCPServers returns one page of registered MCP servers. For automatic
// paging use [Client.AllMCPServers].
func (c *Client) ListMCPServers(ctx context.Context, opts ...ListOption) (*ListMCPServersResponse, error) {
	return getResource[ListMCPServersResponse](ctx, c, "mcpServers", listValues(opts...))
}

// GetMCPServer returns the metadata of a single MCP server. name is the full
// resource name (e.g. "projects/<p>/locations/<l>/mcpServers/<id>").
func (c *Client) GetMCPServer(ctx context.Context, name string) (*MCPServer, error) {
	return getResource[MCPServer](ctx, c, name, nil)
}

// AllMCPServers iterates over every MCP server matching opts, fetching pages on
// demand. If a page fetch fails the iterator yields a single (nil, error) and
// stops.
func (c *Client) AllMCPServers(ctx context.Context, opts ...ListOption) iter.Seq2[*MCPServer, error] {
	return pages(ctx, opts, func(ctx context.Context, o ...ListOption) ([]MCPServer, string, error) {
		page, err := c.ListMCPServers(ctx, o...)
		if err != nil {
			return nil, "", err
		}
		return page.MCPServers, page.NextPageToken, nil
	})
}

// ListEndpoints returns one page of registered model endpoints. For automatic
// paging use [Client.AllEndpoints].
func (c *Client) ListEndpoints(ctx context.Context, opts ...ListOption) (*ListEndpointsResponse, error) {
	return getResource[ListEndpointsResponse](ctx, c, "endpoints", listValues(opts...))
}

// GetEndpoint returns the metadata of a single endpoint. name is the full
// resource name (e.g. "projects/<p>/locations/<l>/endpoints/<id>").
func (c *Client) GetEndpoint(ctx context.Context, name string) (*Endpoint, error) {
	return getResource[Endpoint](ctx, c, name, nil)
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

var (
	// reModelSuffix strips a trailing ":method" suffix (e.g. ":predict").
	reModelSuffix = regexp.MustCompile(`:\w+$`)
	// reProjectsSubstring extracts a "projects/..." resource name embedded in a URL.
	reProjectsSubstring = regexp.MustCompile(`projects/.+`)
)

// parseModelName extracts a model resource name from an endpoint URI, mirroring
// the parsing in adk-python's get_model_name.
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

// getResource performs an authenticated GET against the registry and decodes
// the JSON response into a new T. It is the primitive the List/Get methods build
// on.
func getResource[T any](ctx context.Context, c *Client, resourcePath string, params url.Values) (*T, error) {
	var v T
	if err := c.requester.Get(ctx, resourcePath, params, &v); err != nil {
		return nil, err
	}
	return &v, nil
}

// pages returns an iterator that yields every item across all pages, fetching
// subsequent pages on demand via fetch. fetch returns one page of items and the
// next page token (empty when there are no more pages). On error the iterator
// yields a single (nil, err) and stops. It backs the All* discovery methods.
func pages[T any](
	ctx context.Context,
	opts []ListOption,
	fetch func(context.Context, ...ListOption) (items []T, nextPageToken string, err error),
) iter.Seq2[*T, error] {
	return func(yield func(*T, error) bool) {
		token := ""
		for {
			pageOpts := opts
			if token != "" {
				// Clone so we never append into opts' backing array, which the
				// caller owns and we reuse on every iteration.
				pageOpts = append(slices.Clone(opts), WithPageToken(token))
			}

			items, next, err := fetch(ctx, pageOpts...)
			if err != nil {
				yield(nil, err)
				return
			}
			for i := range items {
				if !yield(&items[i], nil) {
					return
				}
			}
			if next == "" {
				return
			}
			token = next
		}
	}
}
