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

// Package agentengine calls an agent deployed on Vertex AI Agent Runtime
// (a reasoning engine) through its stream-query method and exposes the
// resulting events as an iterator. It knows nothing about any chat surface,
// so a chat ingress and a plain CLI can share it.
package agentengine

import (
	"context"
	"fmt"
	"regexp"

	aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
	"google.golang.org/api/option"

	aiplatformutil "google.golang.org/adk/v2/util/aiplatform"
	vertexaiutil "google.golang.org/adk/v2/util/vertexai"
)

// Config identifies the reasoning engine to call. Either set Name, or set
// Project, Location and ReasoningEngine. Name wins when both are set. Each
// segment is validated: the location picks the host the client dials, so it
// may contain only lowercase letters, digits and '-'.
type Config struct {
	Project         string
	Location        string
	ReasoningEngine string
	// Name is the full resource name,
	// projects/{project}/locations/{location}/reasoningEngines/{id}.
	Name string
}

// Client talks to one reasoning engine. It is safe for concurrent use by
// multiple goroutines: its fields are immutable after New and the underlying
// GAPIC client is concurrency-safe.
type Client struct {
	name string
	exec *aiplatform.ReasoningEngineExecutionClient
}

// The location is concatenated into the endpoint host, so its character set
// is narrow. Project and engine IDs only have to be one path segment: a
// project may be domain-scoped, like google.com:my-proj.
var fullNameRE = regexp.MustCompile(`^projects/([^/]+)/locations/([a-z0-9-]+)/reasoningEngines/([^/]+)$`)

// resourceName resolves cfg to a full resource name and the location that
// picks the regional aiplatform endpoint.
func resourceName(cfg Config) (name, location string, err error) {
	name = cfg.Name
	if name == "" {
		if cfg.Project == "" || cfg.Location == "" || cfg.ReasoningEngine == "" {
			return "", "", fmt.Errorf("agentengine: Project, Location and ReasoningEngine are required when Name is empty (got %q, %q, %q)", cfg.Project, cfg.Location, cfg.ReasoningEngine)
		}
		data := &vertexaiutil.AgentEngineData{ProjectID: cfg.Project, Location: cfg.Location, ReasoningEngine: cfg.ReasoningEngine}
		name = vertexaiutil.AgentEngineResource(data)
	}
	m := fullNameRE.FindStringSubmatch(name)
	if m == nil {
		return "", "", fmt.Errorf("agentengine: resource name %q is not projects/{p}/locations/{l}/reasoningEngines/{id} with valid segments", name)
	}
	return name, m[2], nil
}

// New creates a client for the reasoning engine described by cfg. Credentials
// come from Application Default Credentials unless opts say otherwise. The
// regional endpoint is set first so a caller's option.WithEndpoint or
// option.WithGRPCConn overrides it.
func New(ctx context.Context, cfg Config, opts ...option.ClientOption) (*Client, error) {
	name, location, err := resourceName(cfg)
	if err != nil {
		return nil, err
	}
	allOpts := append([]option.ClientOption{option.WithEndpoint(aiplatformutil.HostPortURL(location))}, opts...)
	exec, err := aiplatform.NewReasoningEngineExecutionClient(ctx, allOpts...)
	if err != nil {
		return nil, fmt.Errorf("agentengine: creating execution client: %w", err)
	}
	return &Client{name: name, exec: exec}, nil
}

// Name returns the full resource name of the reasoning engine.
func (c *Client) Name() string { return c.name }

// Close releases the underlying connection.
func (c *Client) Close() error { return c.exec.Close() }
