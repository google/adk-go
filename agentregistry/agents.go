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
	"iter"
)

// ListAgents returns one page of registered A2A agents. Use opts to set a
// filter and pagination; for automatic paging use [Client.AllAgents].
func (c *Client) ListAgents(ctx context.Context, opts ...ListOption) (*AgentsPage, error) {
	var page AgentsPage
	if err := c.get(ctx, "agents", listValues(opts...), &page); err != nil {
		return nil, err
	}
	return &page, nil
}

// GetAgent returns the metadata of a single agent. name is the full resource
// name (e.g. "projects/<p>/locations/<l>/agents/<id>").
func (c *Client) GetAgent(ctx context.Context, name string) (*Agent, error) {
	var agent Agent
	if err := c.get(ctx, name, nil, &agent); err != nil {
		return nil, err
	}
	return &agent, nil
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
